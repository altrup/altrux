"""Generic SFT loop for any model in models/ that exports a train_hooks
module (models/{name}/train_hooks.py): setup_training, chunk_loss, and
optionally extra_log/chunk_extra_log. This script owns everything that's the
same across models -- shuffling, chunk iteration, gradient-accumulation
counting, checkpoint cadence/rotation (including mid-example resume),
evaluation, preflight, and non-finite checks -- and delegates the
irreducibly model-specific part (how to load the model for training, and how
to compute loss for one chunk) to those hooks. See
models/mamba2_780m/train_hooks.py for the simple case and
models/mamba2_780m_memory/train_hooks.py for the one that also defines
chunk_extra_log for its live per-chunk progress display.

Checkpointing is also generic: every parameter with requires_grad=True is
saved, which covers both a LoRA-only model (mamba2_780m) and a model with an
additional full-gradient subsystem (mamba2_780m_memory's front_end/
injections) with the same code, since "trainable" is exactly the right
criterion either way.

Checkpoint cadence is counted in cumulative tokens trained on (not optimizer
steps), since examples vary enormously in length (a few thousand to ~100k
tokens in this project's datasets) -- a step-based cadence means wildly
different amounts of actual training between checkpoints depending on what
examples happened to be in the window. Because the trigger is checked after
every accumulation boundary rather than only between examples, a checkpoint
can land mid-example for a long one; resuming such a checkpoint regenerates
the carried model state by replaying (forward-only, no backward) the
already-seen prefix of that example rather than caching the state itself --
see replay_state's docstring for why.
"""

import argparse
import importlib
import json
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# bitsandbytes (as of 0.49.2, the latest release) calls the deprecated
# torch._check_is_size internally (bitsandbytes/backends/cuda/ops.py) -- a
# bug in their code, not ours, and not yet fixed upstream. Suppress just
# this one warning rather than patching the installed package (make sync
# would overwrite that) or silencing FutureWarning project-wide.
warnings.filterwarnings("ignore", message=r".*_check_is_size.*", category=FutureWarning)

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
    chunk_pos: int,
    total_tokens: float,
    last_ckpt_tokens: float,
    lora_rank: int,
    lora_alpha: float,
) -> Path:
    """Saves every trainable parameter -- not just LoRA adapters, since a
    model like mamba2_780m_memory has an additional full-gradient subsystem
    that a LoRA-only save would silently drop. `chunk_pos` is the token
    offset of the next chunk to process within `example_idx` (0 if this
    checkpoint landed exactly on an example boundary) -- no model `state`
    tensor is saved alongside it; see replay_state."""
    path = CKPT_DIR / f"epoch-{epoch + 1}" / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path / "trainable.pt")
    (path / "lora_config.json").write_text(json.dumps({"rank": lora_rank, "alpha": lora_alpha}))
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(
        {
            "epoch": epoch,
            "example_idx": example_idx,
            "chunk_pos": chunk_pos,
            "total_tokens": total_tokens,
            "last_ckpt_tokens": last_ckpt_tokens,
        },
        path / "state.pt",
    )
    return path


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} trainable tensors from {path}")
    if loaded == 0:
        raise RuntimeError("load_checkpoint loaded 0 tensors -- checkpoint keys don't match model structure")


def _chunks(ids: torch.Tensor, mask: torch.Tensor | None, chunk_len: int, start_pos: int = 0):
    """Slices one example into (start, end, input_ids, target_ids, mask_slice)
    chunks of at most `chunk_len` tokens, shifted by one for next-token
    prediction. The only thing that's ever model-specific about chunking is
    how to compute loss for one chunk (hooks.chunk_loss) -- the slicing
    itself is identical for every model, which is why it lives here instead
    of in a model's train_hooks.py."""
    seqlen = ids.numel()
    for start in range(start_pos, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1:end + 1].unsqueeze(0)
        mask_slice = mask[start + 1:end + 1] if mask is not None else None
        yield start, end, input_ids, target_ids, mask_slice


def replay_state(hooks, model, ids: torch.Tensor, mask: torch.Tensor | None, chunk_len: int, chunk_pos: int, device):
    """Forward-only (no grad, no backward) regeneration of the carried model
    `state` up to `chunk_pos`, for resuming a checkpoint that landed
    mid-example. Saving the state tensor itself to disk on every checkpoint
    was considered and rejected: for a state-heavy model it can be 100+MB,
    and that cost would be paid on every routine checkpoint (every
    --ckpt-every-tokens) to cover a resume that's rare and one-time --
    replay is bounded by chunk_pos (itself bounded by roughly
    --ckpt-every-tokens), so it's cheap exactly when it's needed and free
    otherwise."""
    if chunk_pos == 0:
        return None
    ids = ids.to(device)
    mask = mask.to(device) if mask is not None else None
    state = None
    with torch.no_grad():
        for start, end, input_ids, target_ids, mask_slice in _chunks(ids, mask, chunk_len, start_pos=0):
            if start >= chunk_pos:
                break
            _, _, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight=1.0)
            state = state.detach()
    return state


def evaluate(hooks, model, eval_ids: list[torch.Tensor], eval_masks: list, device, max_len, chunk_len: int | None = None) -> float:
    """Held-out loss, no backward. Generic across models since chunk_loss is
    the only model-specific piece."""
    chunk_len = chunk_len or hooks.DEFAULT_CHUNK_LEN
    model.eval()
    total_loss = total_weight = 0.0
    with torch.no_grad():
        for ids, mask in zip(eval_ids, eval_masks):
            if ids.numel() > max_len or ids.numel() < 2:
                continue
            if mask is not None and not mask.any():
                continue
            ids = ids.to(device)
            mask = mask.to(device) if mask is not None else None
            state = None
            for start, end, input_ids, target_ids, mask_slice in _chunks(ids, mask, chunk_len):
                loss_sum, weight_sum, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight=1.0)
                total_loss += loss_sum.item()
                total_weight += weight_sum.item()
                state = state.detach()
    model.train()
    return total_loss / total_weight if total_weight > 0 else float("nan")


def preflight(
    hooks,
    model,
    trainable_params: list[torch.nn.Parameter],
    all_ids: list[torch.Tensor],
    all_masks: list,
    device,
    max_len,
    eos_weight: float = 1.0,
    chunk_len: int | None = None,
) -> None:
    """Runs one example through the same chunked forward+backward path the
    main loop uses, then asserts gradients actually reached the trainable
    parameters -- this is what catches a model whose forward silently fails
    to connect some part of itself to the loss (see
    models/mamba2_780m_memory/train_hooks.py's history for why this check
    matters). LoRA params and any other trainable params (e.g. a model's own
    full-gradient subsystem) are checked separately so one silently
    disconnected branch can't hide behind the other's gradient."""
    chunk_len = chunk_len or hooks.DEFAULT_CHUNK_LEN
    sample_ids = sample_mask = None
    for ids, mask in zip(all_ids, all_masks):
        if ids.numel() < 2 or ids.numel() > max_len:
            continue
        if mask is not None and not mask.any():
            continue
        sample_ids, sample_mask = ids, mask
        break
    assert sample_ids is not None, "no valid examples found in dataset"

    model.train()
    sample_ids = sample_ids.to(device)
    sample_mask = sample_mask.to(device) if sample_mask is not None else None
    seqlen = sample_ids.numel()
    chunk_extra_log = getattr(hooks, "chunk_extra_log", None)
    state = None
    prev_n_lines = 0
    for start, end, input_ids, target_ids, mask_slice in _chunks(sample_ids, sample_mask, chunk_len):
        loss_sum, weight_sum, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)
        if weight_sum > 0:
            (loss_sum / weight_sum).backward()
            prev_n_lines = _show_chunk_progress(model, chunk_extra_log, end, seqlen, loss_sum, weight_sum, prev_n_lines)
        state = state.detach()
    _clear_live(prev_n_lines)

    lora_total = other_total = 0
    lora_grads = other_grads = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        has_grad = int(p.grad is not None and p.grad.abs().max() > 0)
        if "lora_A" in name or "lora_B" in name:
            lora_total += 1
            lora_grads += has_grad
        else:
            other_total += 1
            other_grads += has_grad
    model.zero_grad()

    assert lora_grads + other_grads > 0, f"preflight: 0/{len(trainable_params)} params received gradients"
    assert lora_total == 0 or lora_grads > 0, "preflight: 0 LoRA params received gradients"
    assert other_total == 0 or other_grads > 0, "preflight: 0 non-LoRA trainable params received gradients"
    print(f"preflight OK -- {lora_grads} LoRA params and {other_grads} other trainable params have gradients")


def _print_live(lines: list[str], prev_n_lines: int) -> int:
    """Overwrites whatever this function last printed (prev_n_lines lines)
    with `lines`, in place -- moves the cursor up to the start of that
    block first (ESC[<n>A), then clears to end of each line as it's
    rewritten (ESC[K) so a shorter new line doesn't leave old characters
    trailing past its end. Only does anything useful on a real terminal;
    when piped (e.g. `make train`'s `tee`), the escape codes land in the
    log file as literal bytes, same as a plain \r."""
    out = (f"\033[{prev_n_lines - 1}A" if prev_n_lines > 1 else "") + "\r"
    out += "\n".join(f"{line}\033[K" for line in lines)
    sys.stdout.write(out)
    sys.stdout.flush()
    return len(lines)


def _clear_live(prev_n_lines: int) -> None:
    """Clears whatever _print_live last left on screen (ESC[J clears from
    the cursor to end of screen, removing every line of the block at once)."""
    if prev_n_lines == 0:
        return
    out = (f"\033[{prev_n_lines - 1}A" if prev_n_lines > 1 else "") + "\r\033[J"
    sys.stdout.write(out)
    sys.stdout.flush()


def _show_chunk_progress(model, chunk_extra_log, end: int, seqlen: int, loss_sum, weight_sum, prev_n_lines: int) -> int:
    """Updates the in-place live progress display for one chunk, if the
    model's hooks define chunk_extra_log -- used by both the main training
    loop and preflight, since a slow model's chunks (e.g.
    mamba2_780m_memory's manual per-token mixer step) need this visibility
    in either place, not just during real training."""
    if chunk_extra_log is None or weight_sum <= 0:
        return prev_n_lines
    chunk_loss_val = loss_sum.item() / weight_sum.item()
    lines = [f"  token {end:>6}/{seqlen:<6}  loss {chunk_loss_val:.4f}"]
    extra_line = chunk_extra_log(model)
    if extra_line is not None:
        lines.append(f"  {extra_line}")
    return _print_live(lines, prev_n_lines)


def run_training(
    hooks,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainable_params: list[torch.nn.Parameter],
    train_ids: list[torch.Tensor],
    train_masks: list,
    eval_ids: list[torch.Tensor],
    eval_masks: list,
    device,
    args,
    start_epoch: int,
    start_example: int,
    start_step: int,
    start_chunk_pos: int,
    start_total_tokens: float,
    start_last_ckpt_tokens: float,
) -> None:
    """Owns the entire training loop: shuffling, chunk iteration,
    gradient-accumulation counting, checkpoint cadence/rotation (by
    cumulative tokens, possibly landing mid-example), and the final
    unconditional save. `args` needs: epochs, eos_weight, accum_steps,
    chunk_len, ckpt_every_tokens, keep_ckpts, lora_rank, lora_alpha, max_len
    (already resolved to a number, not the raw CLI None default)."""
    chunk_len = args.chunk_len or hooks.DEFAULT_CHUNK_LEN
    extra_log = getattr(hooks, "extra_log", None)
    chunk_extra_log = getattr(hooks, "chunk_extra_log", None)
    on_step = getattr(hooks, "on_step", None)

    n = len(train_ids)
    global_step = start_step
    total_tokens = start_total_tokens
    last_ckpt_tokens = start_last_ckpt_tokens
    model.train()
    optimizer.zero_grad()
    if on_step is not None:
        on_step(model, total_tokens)

    trained_any = False
    last_epoch = last_example_idx = None
    last_chunk_pos = 0

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

            if ids.numel() > args.max_len or ids.numel() < 2:
                continue

            ids = ids.to(device)
            mask = mask.to(device) if mask is not None else None
            seqlen = ids.numel()

            resuming_here = epoch == start_epoch and i == start_example and start_chunk_pos > 0
            if resuming_here:
                state = replay_state(hooks, model, ids, mask, chunk_len, start_chunk_pos, device)
                chunk_start_pos = start_chunk_pos
            else:
                state = None
                chunk_start_pos = 0

            prev_n_lines = 0
            example_failed = False

            for start, end, input_ids, target_ids, mask_slice in _chunks(ids, mask, chunk_len, chunk_start_pos):
                loss_sum, weight_sum, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, args.eos_weight)

                if not torch.isfinite(loss_sum):
                    print(f"  warning: non-finite loss at chunk [{start}:{end}], skipping example")
                    optimizer.zero_grad()
                    accum_count = 0
                    window_loss_sum = window_tokens = 0.0
                    example_failed = True
                    break

                if weight_sum > 0:
                    (loss_sum / weight_sum).backward()
                    window_loss_sum += loss_sum.item()
                    window_tokens += weight_sum.item()
                    total_tokens += weight_sum.item()
                    accum_count += 1
                    trained_any = True
                    prev_n_lines = _show_chunk_progress(model, chunk_extra_log, end, seqlen, loss_sum, weight_sum, prev_n_lines)

                state = state.detach()

                is_example_done = end >= seqlen - 1
                if accum_count >= args.accum_steps or is_example_done:
                    if chunk_extra_log is not None:
                        _clear_live(prev_n_lines)
                        prev_n_lines = 0

                    # Each chunk's backward() above adds its (already
                    # per-token-averaged) gradient into .grad unscaled, since
                    # an example boundary can now force a step before
                    # accum_steps chunks have accumulated (see is_example_done
                    # above) -- dividing by the fixed args.accum_steps
                    # regardless of how many chunks actually contributed
                    # would underweight every such step. Dividing by the
                    # true accum_count here instead always yields the
                    # average gradient over however many chunks actually ran.
                    if accum_count > 0:
                        for p in trainable_params:
                            if p.grad is not None:
                                p.grad /= accum_count

                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    avg_loss = window_loss_sum / window_tokens if window_tokens > 0 else float("nan")
                    accum_count = 0
                    window_loss_sum = window_tokens = 0.0

                    if on_step is not None:
                        on_step(model, total_tokens)

                    print(f"epoch {epoch + 1}  step {global_step:>6}  example {i:>6}/{n}  loss {avg_loss:.4f}  gnorm {grad_norm:.3f}")
                    if extra_log is not None:
                        line = extra_log(model)
                        if line is not None:
                            print(f"    {line}")

                    bad = [name for name, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
                    if bad:
                        print(f"FATAL: non-finite weights after step {global_step}: {bad[:5]}")
                        print("Checkpoints NOT saved. Exiting.")
                        raise SystemExit(1)

                    next_chunk_pos = 0 if is_example_done else end
                    last_epoch, last_example_idx, last_chunk_pos = epoch, i, next_chunk_pos

                    if total_tokens - last_ckpt_tokens >= args.ckpt_every_tokens:
                        el = evaluate(hooks, model, eval_ids, eval_masks, device, args.max_len, chunk_len)
                        path = save_checkpoint(
                            model, optimizer, global_step, epoch, i, next_chunk_pos,
                            total_tokens, last_ckpt_tokens, args.lora_rank, args.lora_alpha,
                        )
                        last_ckpt_tokens = total_tokens
                        rotate_checkpoints(args.keep_ckpts, epoch)
                        print(f"  eval_loss {el:.4f}  saved {path}")

            if chunk_extra_log is not None:
                _clear_live(prev_n_lines)

            if example_failed:
                continue

    if not trained_any:
        print("nothing to train -- already at or past the requested epochs. Pass a larger --epochs to continue.")
        return

    el = evaluate(hooks, model, eval_ids, eval_masks, device, args.max_len, chunk_len)
    path = save_checkpoint(
        model, optimizer, global_step, last_epoch, last_example_idx, last_chunk_pos,
        total_tokens, last_ckpt_tokens, args.lora_rank, args.lora_alpha,
    )
    rotate_checkpoints(args.keep_ckpts, last_epoch)
    print(f"done. eval_loss {el:.4f}  final checkpoint: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"SFT for {MODEL_NAME}")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID (informational; actual ID comes from models/ file)")
    parser.add_argument("--data", default="data/train.pt", help="Tokenized dataset from prepare_data.py")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=None, help="Skip examples longer than this (default: no limit)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward/backward chunk -- defaults to the model's own DEFAULT_CHUNK_LEN (e.g. 48 for mamba2_780m, 8 for mamba2_780m_memory)")
    parser.add_argument("--eval-examples", type=int, default=200, help="Examples held out for eval")
    parser.add_argument("--accum-steps", type=int, default=25, help="Gradient accumulation steps (counted per backward() call -- one per chunk; capped by example boundaries, since accumulation never spans two examples -- see the training loop)")
    parser.add_argument("--ckpt-every-tokens", type=int, default=5000, help="Save checkpoint every N tokens of training, checked after every gradient-accumulation boundary -- can land mid-example for a long one (resume replays the seen prefix to regenerate model state, see replay_state)")
    parser.add_argument("--keep-ckpts", type=int, default=20, help="Number of checkpoints to retain")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eos-weight", type=float, default=5.0, help="Loss weight for EOS tokens (>1 to emphasise stopping)")
    parser.add_argument("--preflight-only", action="store_true", help="Load the real model and data, run the preflight gradient check, then exit -- skips the full training loop. For sanity-checking a setup before committing to a real run.")
    args = parser.parse_args()
    args.max_len = args.max_len if args.max_len is not None else math.inf

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
    start_chunk_pos = 0
    start_total_tokens = 0.0
    start_last_ckpt_tokens = 0.0
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
                start_chunk_pos = state.get("chunk_pos", 0)
                start_total_tokens = state.get("total_tokens", 0.0)
                start_last_ckpt_tokens = state.get("last_ckpt_tokens", 0.0)
                # A checkpoint that landed mid-example resumes the SAME
                # example from chunk_pos; one that landed on an example
                # boundary resumes the next example fresh, as before.
                start_example = state["example_idx"] if start_chunk_pos > 0 else state["example_idx"] + 1
            print(f"resumed at step {start_step}, epoch {start_epoch + 1}, example {start_example}, chunk_pos {start_chunk_pos}")
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

    preflight(hooks, model, trainable_params, train_ids, train_masks, device, args.max_len, args.eos_weight, args.chunk_len)

    if args.preflight_only:
        print("preflight passed (--preflight-only set) -- exiting before the training loop")
        return

    run_training(
        hooks, model, optimizer, trainable_params, train_ids, train_masks, eval_ids, eval_masks, device, args,
        start_epoch, start_example, start_step, start_chunk_pos, start_total_tokens, start_last_ckpt_tokens,
    )


if __name__ == "__main__":
    main()
