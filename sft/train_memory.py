"""Chunked, state-threaded SFT for mamba2_2_7b_memory.

Not reusable from train.py: that script calls `model(input_ids).logits` in one
shot, assuming a stateless forward. This model's `Model.forward(input_ids,
state)` returns `(logits, state)` and is designed to be called repeatedly with
`state` carried across calls (see models/mamba2_2_7b_memory/README.md) --
that's what makes training on long sessions tractable without holding every
token's activations in memory at once (see sft/README.md's "Long-context
data" section for why --max-len at data-prep time is the wrong lever for
this). Genuinely reused from train.py: apply_lora/save_lora/load_lora (lora.py)
and models.common's helpers -- both already model-agnostic.

Unlike train.py (and unlike standard SFT generally), loss is computed over
EVERY token, not just assistant turns: prepare_data.py's mask marks user turns
as non-trainable, which is correct for short Q&A-style chat, but for this
model most of the content that's supposed to exercise long-range recall is
*in* the long user turns (a document, a long context) -- masking that out
would throw away most of the signal this model exists to learn from. The
stored mask is therefore ignored entirely here.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from lora import apply_lora

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME = os.getenv("MODEL_NAME", "mamba2_2_7b_memory")
_model_mod = importlib.import_module(f"models.{MODEL_NAME}")
MODEL_ID = _model_mod.MODEL_ID
TARGET_MODULES = _model_mod.TARGET_LORA_MODULES

CKPT_DIR = Path(__file__).parent.parent / "models" / MODEL_NAME / "checkpoints"
EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b


def iter_checkpoints():
    if not CKPT_DIR.exists():
        return
    for epoch_dir in CKPT_DIR.glob("epoch-*"):
        if not epoch_dir.is_dir():
            continue
        for p in epoch_dir.iterdir():
            if p.is_dir() and p.name.startswith("step-"):
                yield int(p.name.split("-")[1]), p


def latest_checkpoint() -> Path | None:
    ckpts = sorted(iter_checkpoints())
    return ckpts[-1][1] if ckpts else None


def rotate_checkpoints(keep: int, epoch: int) -> None:
    epoch_dir = CKPT_DIR / f"epoch-{epoch + 1}"
    if not epoch_dir.exists():
        return
    steps = sorted(
        int(p.name.split("-")[1]) for p in epoch_dir.iterdir() if p.is_dir() and p.name.startswith("step-")
    )
    for step in steps[:-keep]:
        shutil.rmtree(epoch_dir / f"step-{step}")


def save_checkpoint(model, optimizer, step, epoch, example_idx, lora_rank, lora_alpha) -> Path:
    """Saves every trainable param -- not just LoRA adapters (save_lora's job
    for the other models), since this model's memory subsystem (front_end,
    injections) trains with full gradients and would otherwise be silently
    left out of every checkpoint."""
    path = CKPT_DIR / f"epoch-{epoch + 1}" / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path / "trainable.pt")
    (path / "lora_config.json").write_text(json.dumps({"rank": lora_rank, "alpha": lora_alpha}))
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save({"epoch": epoch, "example_idx": example_idx}, path / "state.pt")
    return path


def load_checkpoint(model, path: Path) -> None:
    state = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} trainable tensors from {path}")
    if loaded == 0:
        raise RuntimeError("load_checkpoint loaded 0 tensors -- checkpoint keys don't match model structure")


def chunk_loss(model, input_ids: torch.Tensor, target_ids: torch.Tensor, state, eos_weight: float):
    """One chunk through the model. Returns (weighted_loss_sum, weight_sum, new_state).
    Every position contributes (see module docstring) except eos_weight scaling."""
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    weight = torch.ones_like(loss)
    if eos_weight != 1.0:
        weight[target_ids.reshape(-1) == EOS_ID] = eos_weight
    return (loss * weight).sum(), weight.sum(), state


def process_example(model, ids: torch.Tensor, chunk_len: int, eos_weight: float, backward_scale: float) -> tuple[float, float]:
    """Runs one example chunk-by-chunk, calling .backward() per chunk (state is
    carried across chunks of the SAME example, detached at each boundary --
    truncated BPTT -- and never carried across different examples). Returns
    (total_loss, total_weight) as plain floats for logging.
    """
    state = None
    total_loss = total_weight = 0.0
    seqlen = ids.numel()
    for start in range(0, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1 : end + 1].unsqueeze(0)
        weighted_loss, weight, state = chunk_loss(model, input_ids, target_ids, state, eos_weight)
        if not torch.isfinite(weighted_loss):
            raise FloatingPointError(f"non-finite loss at chunk [{start}:{end}]")
        (weighted_loss / weight * backward_scale).backward()
        total_loss += weighted_loss.item()
        total_weight += weight.item()
        state = state.detach()
    return total_loss, total_weight


def eval_loss(model, eval_ids: list[torch.Tensor], device, chunk_len: int, max_len: int) -> float:
    model.eval()
    total_loss = total_tokens = 0.0
    with torch.no_grad():
        for ids in eval_ids:
            if ids.numel() > max_len or ids.numel() < 2:
                continue
            ids = ids.to(device)
            state = None
            seqlen = ids.numel()
            for start in range(0, seqlen - 1, chunk_len):
                end = min(start + chunk_len, seqlen - 1)
                input_ids = ids[start:end].unsqueeze(0)
                target_ids = ids[start + 1 : end + 1].unsqueeze(0)
                weighted_loss, weight, state = chunk_loss(model, input_ids, target_ids, state, eos_weight=1.0)
                total_loss += weighted_loss.item()
                total_tokens += weight.item()
                state = state.detach()
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def preflight(model, all_ids: list[torch.Tensor], device, chunk_len: int, max_len: int) -> None:
    sample = None
    for ids in all_ids:
        if 2 <= ids.numel() <= max_len:
            sample = ids
            break
    assert sample is not None, "no valid examples found in dataset"

    model.train()
    process_example(model, sample.to(device), chunk_len, eos_weight=1.0, backward_scale=1.0)

    lora_grads = memory_grads = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None or p.grad.abs().max() == 0:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_grads += 1
        else:
            memory_grads += 1
    model.zero_grad()

    assert lora_grads > 0, "preflight: 0 LoRA params received gradients"
    assert memory_grads > 0, "preflight: 0 memory-subsystem params received gradients"
    print(f"preflight OK -- {lora_grads} LoRA params and {memory_grads} memory-subsystem params have gradients")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked SFT for mamba2_2_7b_memory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--data", default="data/train_memory.pt", help="Tokenized dataset from prepare_data.py + merge_data.py")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--chunk-len", type=int, default=512, help="Tokens per forward/backward chunk (bounds training RAM, not example length)")
    parser.add_argument("--max-len", type=int, default=100000, help="Skip examples longer than this (entire example, not per-chunk)")
    parser.add_argument("--eval-examples", type=int, default=50, help="Examples held out for eval")
    parser.add_argument("--accum-chunks", type=int, default=8, help="Gradient accumulation steps, counted in CHUNKS not examples")
    parser.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoint every N optimizer steps")
    parser.add_argument("--keep-ckpts", type=int, default=20, help="Number of checkpoints to retain")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eos-weight", type=float, default=5.0, help="Loss weight for EOS tokens (>1 to emphasise stopping)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {device} -- {torch.cuda.get_device_name(device)} (index {torch.cuda.current_device()})")
        print(f"  VRAM total:  {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"  VRAM free:   {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
    else:
        print(f"device: {device} (no CUDA/ROCm device found)")

    print(f"loading {MODEL_ID} ...")
    base = _model_mod.load_base(str(device))  # quantizes TARGET_LORA_MODULES if QUANTIZE_LORA_BASE
    base = apply_lora(base, TARGET_MODULES, args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = _model_mod.Model(base).to(device)
    # Model.__init__ already freezes everything except lora_A/lora_B and the
    # memory subsystem (front_end, injections) -- nothing extra to freeze here.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    start_step = 0
    start_epoch = 0
    start_example = 0
    if args.resume:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            print(f"resuming from {ckpt}")
            load_checkpoint(model, ckpt)
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
            start_step = int(ckpt.name.split("-")[1])
            state_path = ckpt / "state.pt"
            if state_path.exists():
                state = torch.load(state_path, weights_only=True)
                start_epoch = state["epoch"]
                start_example = state["example_idx"] + 1
            print(f"resumed at step {start_step}, epoch {start_epoch + 1}, example {start_example}")
        else:
            print("no checkpoint found, starting fresh")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    all_ids: list[torch.Tensor] = data["ids"]  # masks ignored -- see module docstring

    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(all_ids), generator=rng).tolist()
    all_ids = [all_ids[i] for i in perm]

    n_eval = min(args.eval_examples, len(all_ids) // 10)
    eval_ids = all_ids[:n_eval]
    train_ids = all_ids[n_eval:]
    n = len(train_ids)
    print(f"train: {n}  eval: {n_eval}  epochs: {args.epochs}  chunk_len: {args.chunk_len}")

    preflight(model, train_ids, device, args.chunk_len, args.max_len)

    global_step = start_step
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(epoch)).tolist()
        skip = start_example if epoch == start_epoch else 0
        accum_count = 0
        window_loss_sum = 0.0
        window_tokens = 0.0

        for i, idx in enumerate(order[skip:], start=skip):
            ids = train_ids[idx]
            if ids.numel() > args.max_len or ids.numel() < 2:
                continue
            ids = ids.to(device)

            try:
                loss_sum, weight_sum = process_example(model, ids, args.chunk_len, args.eos_weight, 1.0 / args.accum_chunks)
            except FloatingPointError as e:
                print(f"  warning: {e}, skipping example")
                optimizer.zero_grad()
                accum_count = 0
                window_loss_sum = window_tokens = 0.0
                continue

            window_loss_sum += loss_sum
            window_tokens += weight_sum
            n_chunks = max(1, (ids.numel() - 1 + args.chunk_len - 1) // args.chunk_len)
            accum_count += n_chunks

            is_last = i == len(order) - 1
            if accum_count >= args.accum_chunks or is_last:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss = window_loss_sum / window_tokens if window_tokens > 0 else float("nan")
                accum_count = 0
                window_loss_sum = window_tokens = 0.0

                print(f"epoch {epoch + 1}  step {global_step:>6}  example {i:>6}/{n}  loss {avg_loss:.4f}  gnorm {grad_norm:.3f}")

                bad = [name for name, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
                if bad:
                    print(f"FATAL: non-finite weights after step {global_step}: {bad[:5]}")
                    print("Checkpoints NOT saved. Exiting.")
                    raise SystemExit(1)

                if global_step % args.ckpt_every == 0:
                    el = eval_loss(model, eval_ids, device, args.chunk_len, args.max_len)
                    path = save_checkpoint(model, optimizer, global_step, epoch, i, args.lora_rank, args.lora_alpha)
                    rotate_checkpoints(args.keep_ckpts, epoch)
                    print(f"  eval_loss {el:.4f}  saved {path}")

    if global_step == start_step:
        print("nothing to train -- already at or past the requested epochs. Pass a larger --epochs to continue.")
        return

    el = eval_loss(model, eval_ids, device, args.chunk_len, args.max_len)
    path = save_checkpoint(model, optimizer, global_step, epoch, len(order) - 1, args.lora_rank, args.lora_alpha)
    rotate_checkpoints(args.keep_ckpts, epoch)
    print(f"done. eval_loss {el:.4f}  final checkpoint: {path}")


if __name__ == "__main__":
    main()
