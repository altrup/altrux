"""Generic SFT loop for any model in models/ that exports a train_hooks
module (models/{name}/train_hooks.py): setup_training, chunk_loss, and
optionally extra_log/chunk_extra_log/on_step/reset_slot/set_slot_state/
init_state/replay_context. This script owns everything that's the same across
models -- shuffling, chunk iteration, gradient-accumulation counting,
checkpoint cadence/rotation (including mid-example resume), evaluation,
preflight, and non-finite checks -- and delegates the irreducibly
model-specific part (how to load the model for training, and how to compute
loss for one chunk) to those hooks. See models/mamba2_780m/train_hooks.py
for the simple case and models/mamba2_2_7b_memory/train_hooks.py for the
one that also defines chunk_extra_log for its live per-slot progress display.

Checkpointing is also generic: every parameter with requires_grad=True is
saved, which covers both a LoRA-only model (mamba2_780m) and a model with an
additional full-gradient subsystem (mamba2_2_7b_memory's front_end/
injections) with the same code, since "trainable" is exactly the right
criterion either way.

Checkpoint cadence is counted in cumulative tokens trained on (not optimizer
steps), since examples vary enormously in length (a few thousand to ~100k
tokens in this project's datasets) -- a step-based cadence means wildly
different amounts of actual training between checkpoints depending on what
examples happened to be in the window. Because the trigger is checked after
every accumulation boundary rather than only between examples, a checkpoint
can land mid-example for a long one; resuming such a checkpoint regenerates
the carried model state by replaying (forward-only) the already-seen prefix
of each slot rather than caching the state itself.

Training runs B examples in parallel (--batch-size, default 4) using a
slot-based loop: each slot independently progresses through its example, and
when a slot finishes its example the next example is assigned to that slot
(with a per-slot state reset). All B slots are processed in one batched
forward+backward call per chunk, so the GPU sees a (B, chunk_len) tensor at
every step rather than (1, chunk_len). This is the primary mechanism for
saturating GPU utilisation on large memory models like mamba2_2_7b_memory
whose per-token step prevents parallel-scan kernel exploitation.
"""

import argparse
import contextlib
import importlib
import json
import math
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
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


class _Slot:
    """One active training example in the batch.

    Tracks the example index, the tokenized ids/mask, and the current
    position within the example. Positions advance by chunk_len on each
    step; when pos >= seqlen-1 the example is done and the slot is either
    assigned the next example (with state reset) or set to None (epoch done).
    """

    __slots__ = ("slot_idx", "example_idx", "ids", "mask", "pos", "seqlen")

    def __init__(self, slot_idx: int, example_idx: int, ids: torch.Tensor, mask: torch.Tensor | None):
        self.slot_idx = slot_idx
        self.example_idx = example_idx
        self.ids = ids
        self.mask = mask
        self.pos = 0
        self.seqlen = ids.numel()

    def is_done(self) -> bool:
        return self.pos >= self.seqlen - 1


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
    slots: list,
    next_ptr: int,
    total_tokens: float,
    last_ckpt_tokens: float,
    lora_rank: int,
    lora_alpha: float,
) -> Path:
    """Saves every trainable parameter -- not just LoRA adapters, since a
    model like mamba2_2_7b_memory has an additional full-gradient subsystem
    that a LoRA-only save would silently drop. slot_states records each
    slot's (example_idx, pos) for resume; next_ptr is the next example to
    assign from the epoch's ordered list."""
    path = CKPT_DIR / f"epoch-{epoch + 1}" / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, path / "trainable.pt")
    (path / "lora_config.json").write_text(json.dumps({"rank": lora_rank, "alpha": lora_alpha}))
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    slot_states = [
        (s.example_idx, s.pos) if s is not None else None
        for s in slots
    ]
    torch.save(
        {
            "epoch": epoch,
            "slot_states": slot_states,
            "next_ptr": next_ptr,
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
    prediction. Used by evaluate, preflight, and replay (single-example
    paths); the slot-based training loop manages chunking directly."""
    seqlen = ids.numel()
    for start in range(start_pos, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1:end + 1].unsqueeze(0)
        mask_slice = mask[start + 1:end + 1] if mask is not None else None
        yield start, end, input_ids, target_ids, mask_slice


def replay_state(hooks, model, ids: torch.Tensor, mask: torch.Tensor | None, chunk_len: int, chunk_pos: int, device):
    """Forward-only (no backward) regeneration of the carried model state up
    to chunk_pos, for resuming a checkpoint that landed mid-example.

    If the hook defines replay_context (e.g. mamba2_2_7b_memory), it is
    entered here to disable create_graph in the neural memory write --
    the forward arithmetic is identical but without the backward graph,
    making replay materially faster."""
    if chunk_pos == 0:
        return None
    ids = ids.to(device)
    mask = mask.to(device) if mask is not None else None
    state = None
    replay_ctx_fn = getattr(hooks, "replay_context", None)
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        if replay_ctx_fn:
            stack.enter_context(replay_ctx_fn(model))
        for start, end, input_ids, target_ids, mask_slice in _chunks(ids, mask, chunk_len, start_pos=0):
            if start >= chunk_pos:
                break
            _, _, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight=1.0)
            state = state.detach() if state is not None else None
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
                state = state.detach() if state is not None else None
    model.train()
    return total_loss / total_weight if total_weight > 0 else float("nan")



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



def _show_batch_progress(
    model,
    chunk_extra_log_fn,
    slots: list,
    chunk_lens: list[int],
    loss_sum,
    weight_sum,
    prev_n_lines: int,
) -> int:
    """Updates the in-place live progress display for the slot-based training
    loop. Shows one header line (avg loss) followed by one line per slot
    (token position + memory stats if the model provides chunk_extra_log).
    Returns the new prev_n_lines for the next call."""
    if chunk_extra_log_fn is None and all(s is None for s in slots):
        return prev_n_lines
    if weight_sum <= 0:
        return prev_n_lines

    avg_loss = loss_sum.item() / weight_sum.item()
    ts = datetime.now().strftime("%H:%M:%S")

    per_slot_strs: list[str | None] | None = None
    if chunk_extra_log_fn is not None:
        extra = chunk_extra_log_fn(model)
        if extra is not None:
            if isinstance(extra, list):
                per_slot_strs = extra
            else:
                per_slot_strs = [str(extra)]

    lines = [f"[{ts}]  avg_loss {avg_loss:.4f}"]
    for b, slot in enumerate(slots):
        if slot is None:
            lines.append(f"  slot {b}  (idle)")
        else:
            pos_str = f"{slot.pos:>6}/{slot.seqlen:<6}"
            slot_line = f"  slot {b}  token {pos_str}"
            if per_slot_strs and b < len(per_slot_strs) and per_slot_strs[b]:
                slot_line += f"  {per_slot_strs[b]}"
            lines.append(slot_line)

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
    start_slot_states: list | None,
    start_next_ptr: int,
    start_step: int,
    start_total_tokens: float,
    start_last_ckpt_tokens: float,
) -> None:
    """Owns the entire training loop: slot-based batching, shuffling, chunk
    iteration, gradient-accumulation counting, checkpoint cadence/rotation
    (by cumulative tokens), and the final unconditional save.

    B examples (--batch-size) run in parallel; each slot independently
    advances through its example, resetting state when it finishes and
    loading the next. All B slots are processed in one batched
    forward+backward per chunk step.

    `args` needs: epochs, eos_weight, accum_steps, chunk_len,
    ckpt_every_tokens, keep_ckpts, lora_rank, lora_alpha, max_len, batch_size
    (already resolved to numbers, not raw CLI None defaults)."""
    chunk_len = args.chunk_len or hooks.DEFAULT_CHUNK_LEN
    batch_size = args.batch_size
    extra_log_fn = getattr(hooks, "extra_log", None)
    chunk_extra_log_fn = getattr(hooks, "chunk_extra_log", None)
    on_step_fn = getattr(hooks, "on_step", None)
    reset_slot_fn = getattr(hooks, "reset_slot", None)
    set_slot_fn = getattr(hooks, "set_slot_state", None)
    init_state_fn = getattr(hooks, "init_state", None)
    replay_ctx_fn = getattr(hooks, "replay_context", None)

    n = len(train_ids)
    global_step = start_step
    total_tokens = start_total_tokens
    last_ckpt_tokens = start_last_ckpt_tokens
    model.train()
    optimizer.zero_grad()
    if on_step_fn is not None:
        on_step_fn(model, total_tokens)

    trained_any = False

    for epoch in range(start_epoch, args.epochs):
        # deterministic shuffle per epoch so resume can reproduce the same order
        order = torch.randperm(n, generator=torch.Generator().manual_seed(epoch)).tolist()

        # Pre-filter: skip examples that are too short, too long, or fully masked.
        valid_order = [
            idx for idx in order
            if train_ids[idx].numel() >= 2
            and train_ids[idx].numel() <= args.max_len
            and (train_masks[idx] is None or train_masks[idx].any())
        ]
        n_valid = len(valid_order)
        if n_valid == 0:
            continue

        skip = start_next_ptr if epoch == start_epoch else 0
        next_ptr = skip

        # Assign initial examples to slots.
        slots: list[_Slot | None] = []
        for b in range(batch_size):
            if next_ptr < n_valid:
                idx = valid_order[next_ptr]
                next_ptr += 1
                ids = train_ids[idx].to(device)
                mask = train_masks[idx].to(device) if train_masks[idx] is not None else None
                slots.append(_Slot(b, idx, ids, mask))
            else:
                slots.append(None)

        if all(s is None for s in slots):
            continue

        # Initialize batched model state (batch_size slots).
        if init_state_fn is not None:
            batched_state = init_state_fn(model, batch_size, device)
        else:
            batched_state = None  # model initializes it on first chunk_loss call

        # On resume: restore each slot's position and replay its state.
        if epoch == start_epoch and start_slot_states is not None:
            for b, saved in enumerate(start_slot_states):
                if saved is None or b >= len(slots) or slots[b] is None:
                    continue
                example_idx, pos = saved
                # Override the slot's example and position from the checkpoint.
                idx_in_order = next((i for i, v in enumerate(valid_order) if v == example_idx), None)
                if idx_in_order is None:
                    continue  # example was filtered out -- start slot fresh
                ids = train_ids[example_idx].to(device)
                mask = train_masks[example_idx].to(device) if train_masks[example_idx] is not None else None
                slots[b] = _Slot(b, example_idx, ids, mask)
                slots[b].pos = pos

                if pos > 0 and set_slot_fn is not None and batched_state is not None:
                    single_state = replay_state(hooks, model, train_ids[example_idx], train_masks[example_idx], chunk_len, pos, device)
                    if single_state is not None:
                        set_slot_fn(model, batched_state, b, single_state)

        accum_count = 0
        window_loss_sum = 0.0
        window_tokens = 0.0
        prev_n_lines = 0

        while any(s is not None for s in slots):
            # Build the batched chunk: gather next chunk_len tokens from each slot.
            batch_inputs: list[torch.Tensor] = []
            batch_targets: list[torch.Tensor] = []
            batch_weights: list[torch.Tensor] = []
            chunk_actual_lens: list[int] = []

            for slot in slots:
                if slot is None:
                    # Idle slot: pad with zeros, zero weight.
                    batch_inputs.append(torch.zeros(chunk_len, dtype=torch.long, device=device))
                    batch_targets.append(torch.zeros(chunk_len, dtype=torch.long, device=device))
                    batch_weights.append(torch.zeros(chunk_len, dtype=torch.bool, device=device))
                    chunk_actual_lens.append(0)
                else:
                    end = min(slot.pos + chunk_len, slot.seqlen - 1)
                    actual = end - slot.pos
                    inp = slot.ids[slot.pos:end]
                    tgt = slot.ids[slot.pos + 1:end + 1]
                    if actual < chunk_len:
                        pad = chunk_len - actual
                        inp = F.pad(inp, (0, pad))
                        tgt = F.pad(tgt, (0, pad))
                        wt = torch.ones(chunk_len, dtype=torch.bool, device=device)
                        wt[actual:] = False
                    else:
                        wt = torch.ones(chunk_len, dtype=torch.bool, device=device)
                    batch_inputs.append(inp)
                    batch_targets.append(tgt)
                    batch_weights.append(wt)
                    chunk_actual_lens.append(actual)

            input_ids = torch.stack(batch_inputs)   # (B, chunk_len)
            target_ids = torch.stack(batch_targets)  # (B, chunk_len)
            weight_mask = torch.stack(batch_weights)  # (B, chunk_len) bool

            loss_sum, weight_sum, batched_state = hooks.chunk_loss(
                model, input_ids, target_ids, weight_mask, batched_state, args.eos_weight
            )

            if not torch.isfinite(loss_sum):
                print(f"  warning: non-finite loss, skipping chunk")
                optimizer.zero_grad()
                accum_count = 0
                window_loss_sum = window_tokens = 0.0
            elif weight_sum > 0:
                (loss_sum / weight_sum).backward()
                window_loss_sum += loss_sum.item()
                window_tokens += weight_sum.item()
                total_tokens += weight_sum.item()
                accum_count += 1
                trained_any = True
                prev_n_lines = _show_batch_progress(
                    model, chunk_extra_log_fn, slots, chunk_actual_lens, loss_sum, weight_sum, prev_n_lines
                )

            batched_state = batched_state.detach() if batched_state is not None else None

            # Advance slot positions; assign next example to any that finished.
            for b, slot in enumerate(slots):
                if slot is None:
                    continue
                slot.pos += chunk_actual_lens[b]
                if slot.is_done():
                    if next_ptr < n_valid:
                        idx = valid_order[next_ptr]
                        next_ptr += 1
                        ids = train_ids[idx].to(device)
                        mask = train_masks[idx].to(device) if train_masks[idx] is not None else None
                        slots[b] = _Slot(b, idx, ids, mask)
                        if reset_slot_fn is not None and batched_state is not None:
                            reset_slot_fn(model, batched_state, b)
                    else:
                        slots[b] = None
                        if reset_slot_fn is not None and batched_state is not None:
                            reset_slot_fn(model, batched_state, b)

            # Gradient accumulation step.
            if accum_count >= args.accum_steps:
                if chunk_extra_log_fn is not None:
                    _clear_live(prev_n_lines)
                    prev_n_lines = 0

                if accum_count == 0:
                    window_loss_sum = window_tokens = 0.0
                    continue

                for p in trainable_params:
                    if p.grad is not None:
                        p.grad /= accum_count

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_loss = window_loss_sum / window_tokens
                accum_count = 0
                window_loss_sum = window_tokens = 0.0
                ts = datetime.now().strftime("%H:%M:%S")

                if on_step_fn is not None:
                    on_step_fn(model, total_tokens)

                print(f"[{ts}]  epoch {epoch + 1}  step {global_step:>6}  examples {next_ptr}/{n_valid}  loss {avg_loss:.4f}  gnorm {grad_norm:.3f}")
                if extra_log_fn is not None:
                    line = extra_log_fn(model)
                    if line is not None:
                        print(f"    {line}")

                bad = [name for name, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
                if bad:
                    print(f"FATAL: non-finite weights after step {global_step}: {bad[:5]}")
                    print("Checkpoints NOT saved. Exiting.")
                    raise SystemExit(1)

                if total_tokens - last_ckpt_tokens >= args.ckpt_every_tokens:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}]  evaluating ...")
                    sys.stdout.flush()
                    el = evaluate(hooks, model, eval_ids, eval_masks, device, args.max_len, chunk_len)
                    path = save_checkpoint(
                        model, optimizer, global_step, epoch, slots, next_ptr,
                        total_tokens, last_ckpt_tokens, args.lora_rank, args.lora_alpha,
                    )
                    last_ckpt_tokens = total_tokens
                    rotate_checkpoints(args.keep_ckpts, epoch)
                    ts = datetime.now().strftime("%H:%M:%S")
                    if math.isnan(el):
                        print(f"[{ts}]  WARNING: eval_loss is nan -- checkpoint saved but eval metric is unreliable  {path}")
                    else:
                        print(f"[{ts}]  eval_loss {el:.4f}  saved {path}")

        if chunk_extra_log_fn is not None:
            _clear_live(prev_n_lines)

        # Step on any remaining accumulated gradient at epoch end.
        if accum_count > 0:
            for p in trainable_params:
                if p.grad is not None:
                    p.grad /= accum_count
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count = 0

    if not trained_any:
        print("nothing to train -- already at or past the requested epochs. Pass a larger --epochs to continue.")
        return

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  evaluating (final) ...")
    sys.stdout.flush()
    el = evaluate(hooks, model, eval_ids, eval_masks, device, args.max_len, chunk_len)
    path = save_checkpoint(
        model, optimizer, global_step, epoch, slots, next_ptr,
        total_tokens, last_ckpt_tokens, args.lora_rank, args.lora_alpha,
    )
    rotate_checkpoints(args.keep_ckpts, epoch)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  done. eval_loss {el:.4f}  final checkpoint: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"SFT for {MODEL_NAME}")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID (informational; actual ID comes from models/ file)")
    parser.add_argument("--data", default="data/train.pt", help="Tokenized dataset from prepare_data.py")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-len", type=int, default=None, help="Skip examples longer than this (default: no limit)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward/backward chunk -- defaults to the model's own DEFAULT_CHUNK_LEN")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of examples to train in parallel (slot-based batching)")
    parser.add_argument("--eval-examples", type=int, default=200, help="Examples held out for eval")
    parser.add_argument("--accum-steps", type=int, default=12, help="Gradient accumulation steps before each optimizer step (each step covers batch_size * chunk_len tokens)")
    parser.add_argument("--ckpt-every-tokens", type=int, default=2000, help="Save checkpoint every N tokens of training")
    parser.add_argument("--keep-ckpts", type=int, default=50, help="Number of checkpoints to retain")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eos-weight", type=float, default=5.0, help="Loss weight for EOS tokens (>1 to emphasise stopping)")
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
    start_slot_states = None
    start_next_ptr = 0
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
                start_slot_states = state.get("slot_states")
                start_next_ptr = state.get("next_ptr", 0)
                start_total_tokens = state.get("total_tokens", 0.0)
                start_last_ckpt_tokens = state.get("last_ckpt_tokens", 0.0)
                # Legacy checkpoint format (single-example, no slot_states):
                # map old example_idx/chunk_pos to a single-slot slot_states.
                if start_slot_states is None:
                    example_idx = state.get("example_idx", 0)
                    chunk_pos = state.get("chunk_pos", 0)
                    start_slot_states = [(example_idx, chunk_pos)]
                    start_next_ptr = example_idx + (0 if chunk_pos > 0 else 1)
            print(f"resumed at step {start_step}, epoch {start_epoch + 1}, next_ptr {start_next_ptr}")
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
    print(f"train: {n}  eval: {n_eval}  epochs: {args.epochs}  batch_size: {args.batch_size}")

    run_training(
        hooks, model, optimizer, trainable_params, train_ids, train_masks, eval_ids, eval_masks, device, args,
        start_epoch, start_slot_states, start_next_ptr, start_step, start_total_tokens, start_last_ckpt_tokens,
    )


if __name__ == "__main__":
    main()
