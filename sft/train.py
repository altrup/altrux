import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from lora import apply_lora, load_lora, save_lora

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME = os.getenv("MODEL_NAME", "mamba2_780m")
_model_mod = importlib.import_module(f"models.{MODEL_NAME}")
MODEL_ID = _model_mod.MODEL_ID
TARGET_MODULES = _model_mod.TARGET_LORA_MODULES

CKPT_DIR = Path(__file__).parent.parent / "models" / MODEL_NAME / "checkpoints"


def iter_checkpoints():
    """Yield (step, path) for every models/{MODEL_NAME}/checkpoints/epoch-*/step-* directory.
    Step numbers are globally monotonic, so they order checkpoints across epochs."""
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
    """Keep only the newest `keep` checkpoints within this epoch's folder, so a
    later epoch never prunes an earlier epoch's history."""
    epoch_dir = CKPT_DIR / f"epoch-{epoch + 1}"
    if not epoch_dir.exists():
        return
    steps = sorted(
        int(p.name.split("-")[1])
        for p in epoch_dir.iterdir()
        if p.is_dir() and p.name.startswith("step-")
    )
    for step in steps[:-keep]:
        shutil.rmtree(epoch_dir / f"step-{step}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    example_idx: int,
    lora_rank: int,
    lora_alpha: float,
) -> Path:
    path = CKPT_DIR / f"epoch-{epoch + 1}" / f"step-{step}"
    save_lora(model, path, lora_rank, lora_alpha)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save({"epoch": epoch, "example_idx": example_idx}, path / "state.pt")
    return path


EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b


def compute_loss(
    model: torch.nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    eos_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Returns (loss, weight). loss is the (optionally EOS-weighted) mean over
    assistant tokens; weight is the sum of loss weights — equal to the assistant
    token count when eos_weight == 1.0 — and is the correct factor for combining
    per-example losses into a corpus mean."""
    input_ids = ids[:-1].unsqueeze(0)
    target_ids = ids[1:].unsqueeze(0)
    loss_mask = mask[1:].float().clone()  # clone: never mutate the cached mask in place
    if eos_weight != 1.0:
        eos_positions = (target_ids.view(-1) == EOS_ID) & (loss_mask > 0)
        loss_mask[eos_positions] = eos_weight
    weight = loss_mask.sum()
    logits = model(input_ids).logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
    return (loss * loss_mask).sum() / weight, weight.item()


def eval_loss(
    model: torch.nn.Module,
    eval_ids: list[torch.Tensor],
    eval_masks: list[torch.Tensor],
    device: torch.device,
    max_len: int,
) -> float:
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for ids, mask in zip(eval_ids, eval_masks):
            if ids.numel() > max_len or mask.sum() == 0:
                continue
            loss, n = compute_loss(model, ids.to(device), mask.to(device))
            total_loss += loss.item() * n
            total_tokens += n
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def preflight(
    model: torch.nn.Module,
    lora_params: list[torch.nn.Parameter],
    all_ids: list[torch.Tensor],
    all_masks: list[torch.Tensor],
    device: torch.device,
    max_len: int,
    eos_weight: float = 1.0,
) -> None:
    sample_ids = sample_mask = None
    for ids, mask in zip(all_ids, all_masks):
        if mask.any() and ids.numel() <= max_len:
            sample_ids, sample_mask = ids, mask
            break
    assert sample_ids is not None, "no valid examples found in dataset"

    model.train()
    loss, _ = compute_loss(model, sample_ids.to(device), sample_mask.to(device), eos_weight)
    loss.backward()

    grads_nonzero = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().max() > 0)
    model.zero_grad()

    assert grads_nonzero > 0, f"preflight: 0/{len(lora_params)} LoRA params received gradients"
    print(f"preflight OK — loss {loss.item():.4f}, {grads_nonzero}/{len(lora_params)} params have gradients")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for mamba2-780m")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID (informational; actual ID comes from models/ file)")
    parser.add_argument("--data", default="data/train.pt", help="Tokenized dataset from prepare_data.py")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=2048, help="Skip examples longer than this")
    parser.add_argument("--eval-examples", type=int, default=200, help="Examples held out for eval")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoint every N optimizer steps")
    parser.add_argument("--keep-ckpts", type=int, default=20, help="Number of checkpoints to retain")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eos-weight", type=float, default=5.0, help="Loss weight for EOS tokens (>1 to emphasise stopping)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {device} — {torch.cuda.get_device_name(device)} (index {torch.cuda.current_device()})")
        print(f"  VRAM total:  {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"  VRAM free:   {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
    else:
        print(f"device: {device} (no CUDA/ROCm device found)")

    print(f"loading {args.model} ...")
    model = _model_mod.load_base(str(device))
    model = model.to(torch.float32)
    model = apply_lora(model, TARGET_MODULES, args.lora_rank, args.lora_alpha, args.lora_dropout)

    for param in model.parameters():
        param.requires_grad_(False)
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            lora_params.append(param)

    print(f"lora trainable params: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    start_step = 0
    start_epoch = 0
    start_example = 0
    if args.resume:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            print(f"resuming from {ckpt}")
            load_lora(model, ckpt)
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=True)
                )
            start_step = int(ckpt.name.split("-")[1])
            state_path = ckpt / "state.pt"
            if state_path.exists():
                state = torch.load(state_path, weights_only=True)
                start_epoch = state["epoch"]
                start_example = state["example_idx"] + 1  # resume after last-seen example
            print(f"resumed at step {start_step}, epoch {start_epoch + 1}, example {start_example}")
        else:
            print("no checkpoint found, starting fresh")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    all_ids: list[torch.Tensor] = data["ids"]
    all_masks: list[torch.Tensor] = data["masks"]

    # shuffle with fixed seed before splitting so eval isn't biased by dataset ordering
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(all_ids), generator=rng).tolist()
    all_ids = [all_ids[i] for i in perm]
    all_masks = [all_masks[i] for i in perm]

    n_eval = min(args.eval_examples, len(all_ids) // 10)
    eval_ids, eval_masks = all_ids[:n_eval], all_masks[:n_eval]
    train_ids, train_masks = all_ids[n_eval:], all_masks[n_eval:]
    n = len(train_ids)
    print(f"train: {n}  eval: {n_eval}  epochs: {args.epochs}")

    preflight(model, lora_params, train_ids, train_masks, device, args.max_len, args.eos_weight)

    global_step = start_step
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        # deterministic shuffle per epoch so resume can reproduce the same order
        order = torch.randperm(n, generator=torch.Generator().manual_seed(epoch)).tolist()
        skip = start_example if epoch == start_epoch else 0
        accum_count = 0
        window_loss_sum = 0.0
        window_tokens = 0

        for i, idx in enumerate(order[skip:], start=skip):
            ids = train_ids[idx]
            mask = train_masks[idx]

            if ids.numel() > args.max_len or mask.sum() == 0:
                continue

            ids, mask = ids.to(device), mask.to(device)
            loss, weight = compute_loss(model, ids, mask, args.eos_weight)

            if not torch.isfinite(loss):
                print(f"  warning: non-finite loss {loss.item()}, skipping example")
                optimizer.zero_grad()
                accum_count = 0
                window_loss_sum = 0.0
                window_tokens = 0
                continue

            (loss / args.accum_steps).backward()
            window_loss_sum += loss.item() * weight
            window_tokens += weight
            accum_count += 1

            is_last = i == len(order) - 1
            if accum_count == args.accum_steps or is_last:
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss = window_loss_sum / window_tokens
                accum_count = 0
                window_loss_sum = 0.0
                window_tokens = 0

                print(f"epoch {epoch + 1}  step {global_step:>6}  loss {avg_loss:.4f}  gnorm {grad_norm:.3f}")

                bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
                if bad:
                    print(f"FATAL: non-finite weights after step {global_step}: {bad[:5]}")
                    print("Checkpoints NOT saved. Exiting.")
                    raise SystemExit(1)

                if global_step % args.ckpt_every == 0:
                    el = eval_loss(model, eval_ids, eval_masks, device, args.max_len)
                    path = save_checkpoint(model, optimizer, global_step, epoch, i, args.lora_rank, args.lora_alpha)
                    rotate_checkpoints(args.keep_ckpts, epoch)
                    print(f"  eval_loss {el:.4f}  saved {path}")

    if global_step == start_step:
        print("nothing to train — already at or past the requested epochs. Pass a larger --epochs to continue.")
        return

    el = eval_loss(model, eval_ids, eval_masks, device, args.max_len)
    path = save_checkpoint(model, optimizer, global_step, epoch, len(order) - 1, args.lora_rank, args.lora_alpha)
    rotate_checkpoints(args.keep_ckpts, epoch)
    print(f"done. eval_loss {el:.4f}  final checkpoint: {path}")


if __name__ == "__main__":
    main()
