"""Generic SFT loop for any model in models/ that exports a train_hooks
module (models/{name}/train_hooks.py): setup_training, process_example,
eval_loss, preflight. This script owns everything that's the same across
models -- shuffling, gradient-accumulation counting, checkpoint
cadence/rotation, resume, non-finite checks -- and delegates the
irreducibly model-specific part (how to load the model for training, and how
to run forward+backward for one example) to those hooks. See
models/mamba2_780m/train_hooks.py for the simple (stateless) case and
models/mamba2_2_7b_memory/train_hooks.py for the chunked, state-threaded one.

Checkpointing is also generic: every parameter with requires_grad=True is
saved, which covers both a LoRA-only model (mamba2_780m) and a model with an
additional full-gradient subsystem (mamba2_2_7b_memory's front_end/
injections) with the same code, since "trainable" is exactly the right
criterion either way.
"""

import argparse
import importlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_NAME = os.getenv("MODEL_NAME", "mamba2_780m")
_model_mod = importlib.import_module(f"models.{MODEL_NAME}")
hooks = importlib.import_module(f"models.{MODEL_NAME}.train_hooks")
MODEL_ID = _model_mod.MODEL_ID

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
    """Saves every trainable parameter -- not just LoRA adapters, since a
    model like mamba2_2_7b_memory has an additional full-gradient subsystem
    that a LoRA-only save would silently drop."""
    path = CKPT_DIR / f"epoch-{epoch + 1}" / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path / "trainable.pt")
    (path / "lora_config.json").write_text(json.dumps({"rank": lora_rank, "alpha": lora_alpha}))
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save({"epoch": epoch, "example_idx": example_idx}, path / "state.pt")
    return path


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} trainable tensors from {path}")
    if loaded == 0:
        raise RuntimeError("load_checkpoint loaded 0 tensors -- checkpoint keys don't match model structure")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"SFT for {MODEL_NAME}")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID (informational; actual ID comes from models/ file)")
    parser.add_argument("--data", default="data/train.pt", help="Tokenized dataset from prepare_data.py")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=None, help="Skip examples longer than this (default: no limit)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward/backward chunk (only used by models whose train_hooks chunk -- defaults to that model's own DEFAULT_CHUNK_LEN, e.g. 512 for mamba2_2_7b_memory; ignored otherwise)")
    parser.add_argument("--eval-examples", type=int, default=200, help="Examples held out for eval")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps (counted per backward() call -- one per example for stateless models, one per chunk for chunked ones)")
    parser.add_argument("--ckpt-every", type=int, default=50, help="Save checkpoint every N optimizer steps")
    parser.add_argument("--keep-ckpts", type=int, default=20, help="Number of checkpoints to retain")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eos-weight", type=float, default=5.0, help="Loss weight for EOS tokens (>1 to emphasise stopping)")
    parser.add_argument("--preflight-only", action="store_true", help="Load the real model and data, run the preflight gradient check, then exit -- skips the full training loop. For sanity-checking a setup before committing to a real run.")
    args = parser.parse_args()
    max_len = args.max_len if args.max_len is not None else math.inf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {device} -- {torch.cuda.get_device_name(device)} (index {torch.cuda.current_device()})")
        print(f"  VRAM total:  {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"  VRAM free:   {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
    else:
        print(f"device: {device} (no CUDA/ROCm device found)")

    print(f"loading {args.model} ...")
    model, trainable_params = hooks.setup_training(device, args.lora_rank, args.lora_alpha, args.lora_dropout)

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
    all_masks: list[torch.Tensor] = data.get("masks") or [None] * len(all_ids)

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

    hooks.preflight(model, trainable_params, train_ids, train_masks, device, max_len, args.eos_weight, args.chunk_len)

    if args.preflight_only:
        print("preflight passed (--preflight-only set) -- exiting before the training loop")
        return

    global_step = start_step
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        # deterministic shuffle per epoch so resume can reproduce the same order
        order = torch.randperm(n, generator=torch.Generator().manual_seed(epoch)).tolist()
        skip = start_example if epoch == start_epoch else 0
        accum_count = 0
        window_loss_sum = 0.0
        window_tokens = 0.0

        for i, idx in enumerate(order[skip:], start=skip):
            ids = train_ids[idx]
            mask = train_masks[idx]

            if ids.numel() > max_len or ids.numel() < 2:
                continue

            ids = ids.to(device)
            mask = mask.to(device) if mask is not None else None

            try:
                loss_sum, weight_sum, n_backwards = hooks.process_example(
                    model, ids, mask, device, args.eos_weight, 1.0 / args.accum_steps, args.chunk_len
                )
            except FloatingPointError as e:
                print(f"  warning: {e}, skipping example")
                optimizer.zero_grad()
                accum_count = 0
                window_loss_sum = window_tokens = 0.0
                continue

            window_loss_sum += loss_sum
            window_tokens += weight_sum
            accum_count += n_backwards

            is_last = i == len(order) - 1
            if accum_count >= args.accum_steps or is_last:
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
                    el = hooks.eval_loss(model, eval_ids, eval_masks, device, max_len, args.chunk_len)
                    path = save_checkpoint(model, optimizer, global_step, epoch, i, args.lora_rank, args.lora_alpha)
                    rotate_checkpoints(args.keep_ckpts, epoch)
                    print(f"  eval_loss {el:.4f}  saved {path}")

    if global_step == start_step:
        print("nothing to train -- already at or past the requested epochs. Pass a larger --epochs to continue.")
        return

    el = hooks.eval_loss(model, eval_ids, eval_masks, device, max_len, args.chunk_len)
    path = save_checkpoint(model, optimizer, global_step, epoch, len(order) - 1, args.lora_rank, args.lora_alpha)
    rotate_checkpoints(args.keep_ckpts, epoch)
    print(f"done. eval_loss {el:.4f}  final checkpoint: {path}")


if __name__ == "__main__":
    main()
