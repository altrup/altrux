"""Generic SFT loop for any model in models/ that exports a train_hooks
module (models/{name}/train_hooks.py): setup_training, chunk_loss, and
optionally extra_log/chunk_extra_log/on_step/reset_slot/init_state. This
script owns everything that's the same across models -- shuffling, chunk
iteration, gradient-accumulation counting,
checkpoint cadence/rotation (including mid-example resume),
and non-finite checks -- and delegates the irreducibly
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
can land mid-example for a long one. For stateful models, resuming such a
checkpoint uses the exact saved internal state when available (see
--keep-full-state); otherwise there's no cheap way to reconstruct it exactly
(replaying the prefix would use the model's current, already further-trained
weights rather than the weights that were actually live at each point in the
prefix, and is a stability risk besides -- see run_training's resume
comment), so the affected slot's example simply restarts from the
beginning.

Training runs B examples in parallel (--batch-size, default 4) using a
slot-based loop: each slot independently progresses through its example, and
when a slot finishes its example the next example is assigned to that slot
(with a per-slot state reset). All B slots are processed in one batched
forward+backward call per chunk, so the GPU sees a (B, chunk_len) tensor at
every step rather than (1, chunk_len). This is the primary mechanism for
saturating GPU utilisation on large memory models like mamba2_2_7b_memory
whose per-token step prevents parallel-scan kernel exploitation.

Several dataset slices can train at once (--data, repeated), each with its
own token share and its own chunk length / batch size / gradient-checkpoint
setting -- long-gap recall supervision needs a BPTT window wide enough to
reach the writes it should credit, which the cheaper conversational slices
don't. Slices whose config matches share a batch and interleave
example-by-example; slices whose config differs take turns in segments. See
DataSpec and run_training.
"""

import argparse
import gc
import importlib
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from training import checkpoints
from training.checkpoints import load_checkpoint
from training.datasets import (
    DataSpec,
    build_order,
    check_shares,
    dataset_fingerprint,
    datasets_fingerprint,
    group_specs,
    load_datasets,
    parse_data_spec,
    pick_deficit,
    recall_weight_at,
    resolve_share,
)

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

    `sleeps` (optional, from the dataset's sleep_positions) are token
    offsets where the model's backbone state gets wiped mid-example while
    its persistent memory carries on -- see the episodic-chains design spec.
    A sleep fires at the first chunk boundary at or past its offset (slots
    advance in exact chunk_len strides, so firing between chunks avoids
    feeding zero-pad garbage through the memory; the <chunk_len drift is
    negligible against multi-thousand-token episodes). `sleep_i` counts
    fired sleeps; `last_reset` is where the most recent backbone reset
    happened (0 = example start), the origin for --head-weight's ramp.
    """

    __slots__ = ("slot_idx", "example_idx", "ids", "mask", "recall", "sleeps", "sleep_i", "last_reset", "pos", "seqlen")

    def __init__(
        self,
        slot_idx: int,
        example_idx: int,
        ids: torch.Tensor,
        mask: torch.Tensor | None,
        recall: torch.Tensor | None = None,
        sleeps: torch.Tensor | None = None,
    ):
        self.slot_idx = slot_idx
        self.example_idx = example_idx
        self.ids = ids
        self.mask = mask
        self.recall = recall
        self.sleeps = sorted(int(s) for s in sleeps) if sleeps is not None else []
        self.sleep_i = 0
        self.last_reset = 0
        self.pos = 0
        self.seqlen = ids.numel()

    def seek(self, pos: int) -> None:
        """Set the position (resume), marking sleeps at or before it as
        already fired -- the saved internal state already reflects them, so
        the training loop must not fire them again. last_reset uses the
        sleep's own offset rather than the chunk boundary it originally
        fired at (off by < chunk_len, and only if --chunk-len changed
        between runs would even that differ) -- close enough for the
        --head-weight ramp it feeds."""
        self.pos = pos
        fired = [s for s in self.sleeps if s <= pos]
        self.sleep_i = len(fired)
        self.last_reset = fired[-1] if fired else 0

    def is_done(self) -> bool:
        return self.pos >= self.seqlen - 1


def _collect_tensors(obj):
    """Recursively yield every tensor reachable from obj (attributes, dict
    values, list/tuple items). Used to inspect a model's carried state
    (MixerState/MemoryState) generically, without each model needing to
    hand-write its own finiteness check -- every such state stores the batch
    dimension as dim 0 of each leaf tensor, so a generic walk is enough."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _collect_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _collect_tensors(item)
    elif hasattr(obj, "__dict__"):
        for item in vars(obj).values():
            yield from _collect_tensors(item)


def _slot_state_finite(state, slot_idx: int) -> bool:
    """True if every tensor in state is finite at batch index slot_idx."""
    return all(torch.isfinite(t[slot_idx]).all() for t in _collect_tensors(state))


def iter_checkpoints():
    return checkpoints.iter_checkpoints(CKPT_DIR)


def latest_checkpoint() -> Path | None:
    return checkpoints.latest_checkpoint(CKPT_DIR)


def rotate_checkpoints(keep: int, epoch: int) -> None:
    checkpoints.rotate_checkpoints(CKPT_DIR, keep, epoch)


def rotate_full_state(keep: int) -> None:
    checkpoints.rotate_full_state(CKPT_DIR, keep)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    slots: list,
    next_ptr: int,
    total_tokens: float,
    lora_rank: int,
    lora_alpha: float,
    batched_state=None,
    dataset_fingerprint: dict | None = None,
    memory_window: int | None = None,
    group_idx: int | None = None,
    group_ptrs: list[int] | None = None,
    group_tokens: list[float] | None = None,
) -> Path:
    return checkpoints.save_checkpoint(
        CKPT_DIR, model, optimizer, step, epoch, slots, next_ptr,
        total_tokens, lora_rank, lora_alpha, batched_state,
        dataset_fingerprint, memory_window, group_idx, group_ptrs, group_tokens,
    )




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
    train_recall: list,
    train_sleeps: list,
    device,
    args,
    start_epoch: int,
    start_slot_states: list | None,
    start_next_ptr: int,
    start_step: int,
    start_total_tokens: float,
    start_last_ckpt_tokens: float,
    start_full_state=None,
    specs: list[DataSpec] | None = None,
    start_group_idx: int = 0,
    start_group_ptrs: list[int] | None = None,
    start_group_tokens: list[float] | None = None,
) -> None:
    """Owns the entire training loop: slot-based batching, shuffling, chunk
    iteration, gradient-accumulation counting, checkpoint cadence/rotation
    (by cumulative tokens), and the final unconditional save.

    B examples (--batch-size) run in parallel; each slot independently
    advances through its example, resetting state when it finishes and
    loading the next. All B slots are processed in one batched
    forward+backward per chunk step.

    `specs` names the dataset slices and their per-slice training config
    (see DataSpec); omitted, the whole flat example list is treated as one
    slice under the run-wide --chunk-len/--batch-size. Slices whose config
    matches share a group and interleave example-by-example inside one
    batch; slices whose config differs cannot share a batch at all, so
    groups take turns in segments of --mix-segment-tokens, each segment
    going to whichever group is furthest behind its token share. A segment
    stops assigning new examples once it's over budget and then drains, so
    switching costs a shrinking tail rather than a cut example. With one
    group the budget is infinite and the loop is exactly the single-dataset
    one.

    `args` needs: epochs, eos_weight, recall_weight, head_weight,
    head_tokens, accum_tokens, chunk_len,
    ckpt_every_tokens, keep_ckpts, keep_full_state, lora_rank, lora_alpha,
    lr, warmup_steps, max_len, batch_size (already resolved to numbers, not raw CLI None
    defaults).

    start_full_state, if given, is a Path to a checkpoint's mem_state.pt
    (already confirmed by main() to exist and match the resumed group's
    batch size) -- loaded here, not by main(), and only for as long as it
    takes to seed the resumed segment's batched_state, so every slot
    continues exactly instead of restarting its example from the beginning.
    Deliberately not loaded eagerly in main() and handed over as an
    already-materialized tensor: main()'s own stack frame stays alive for
    this entire (very long) call, so any local variable it held bound to
    the loaded state would pin that whole extra copy in VRAM for the whole
    run, alongside the live copy batched_state diverges into after the
    first chunk's detach() -- see rotate_full_state for how checkpoints keep
    this file only for the most recent few."""
    if specs is None:
        specs = [DataSpec(path=str(args.data), chunk_len=args.chunk_len,
                          batch_size=args.batch_size, lo=0, hi=len(train_ids))]
    for spec in specs:
        if spec.chunk_len is None:
            spec.chunk_len = hooks.DEFAULT_CHUNK_LEN
    groups = group_specs(specs)
    data_fp = datasets_fingerprint(specs)
    # memory_window (models that define set_memory_window only -- currently
    # just mamba2_2_7b_memory) decouples how often that model's memory
    # subsystem consolidates a write from chunk_len's own VRAM/BPTT-window
    # role; see Model.forward's docstring in models/mamba2_2_7b_memory/
    # model.py and the design spec at docs/superpowers/specs/2026-07-02-
    # chunked-memory-injection-design.md. A window can't span across
    # forward() calls (each call is exactly one chunk_len-token chunk), so
    # every slice's chunk_len must be an exact multiple of it -- enforced
    # here rather than left to fail deep inside forward() with a less
    # obvious error.
    set_memory_window_fn = getattr(model, "set_memory_window", None)
    memory_window = None
    if set_memory_window_fn is not None:
        memory_window = getattr(args, "memory_window", None) or getattr(hooks, "DEFAULT_MEMORY_WINDOW", 1)
        for spec in specs:
            if spec.chunk_len % memory_window != 0:
                raise ValueError(
                    f"chunk-len ({spec.chunk_len}, for {spec.path}) must be an exact multiple "
                    f"of --memory-window ({memory_window})"
                )
        set_memory_window_fn(memory_window)
    extra_log_fn = getattr(hooks, "extra_log", None)
    chunk_extra_log_fn = getattr(hooks, "chunk_extra_log", None)
    on_step_fn = getattr(hooks, "on_step", None)
    reset_slot_fn = getattr(hooks, "reset_slot", None)
    sleep_slot_fn = getattr(hooks, "sleep_slot", None)
    init_state_fn = getattr(hooks, "init_state", None)
    set_grad_ckpt_fn = getattr(hooks, "set_grad_checkpoint", None)
    if set_grad_ckpt_fn is None and any(s.grad_checkpoint for s in specs):
        print(
            f"warning: grad-checkpoint requested for {[s.path for s in specs if s.grad_checkpoint]} "
            f"but {MODEL_NAME}'s train_hooks defines no set_grad_checkpoint -- ignored"
        )

    recall_end = args.recall_weight
    recall_start = getattr(args, "recall_ramp_start", 1.0)
    recall_ramp_steps = getattr(args, "recall_ramp_steps", 0)
    recall_ramp_shape = getattr(args, "recall_ramp_shape", "linear")
    segment_budget = math.inf if len(groups) == 1 else getattr(args, "mix_segment_tokens", 1_000_000)
    max_steps = getattr(args, "max_steps", None) or math.inf

    global_step = start_step
    total_tokens = start_total_tokens
    last_ckpt_tokens = start_last_ckpt_tokens
    model.train()
    optimizer.zero_grad()
    if on_step_fn is not None:
        on_step_fn(model, global_step)

    def set_lr(step: int) -> None:
        # Linear warmup from args.lr/warmup_steps up to args.lr over
        # --warmup-steps, then held at args.lr. `step` here means "the step
        # about to be taken" (1-indexed), NOT "steps completed so far" --
        # step/warmup_steps (0-indexed) would make the very first optimizer
        # step use lr=0, a fully wasted no-op step (confirmed: that's what
        # this looked like before the +1). Still a pure function of
        # global_step (same pattern as BETA_BIAS_ANNEAL_STEPS/
        # set_beta_anneal), so it resumes correctly with no extra checkpoint
        # state. Applying full --lr from step 1 against a freshly-attached,
        # near-randomly-initialized subsystem (gate projections, LoRA)
        # waking up at the same time beta's own suppression is fading is a
        # plausible source of oversized early gradients -- see the huge
        # pre-clip gnorm values a real run hit in its first ~20 steps.
        if args.warmup_steps > 0:
            frac = min((step + 1) / args.warmup_steps, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * frac

    set_lr(global_step)

    def budget_spent() -> bool:
        return global_step >= max_steps

    trained_any = False
    orders: list[list[int]] = []
    ptrs: list[int] = []
    group_tokens: list[float] = []
    final_save: dict = {}

    def keep(idx: int) -> bool:
        """Examples too short, too long, or fully masked are dropped."""
        return (
            train_ids[idx].numel() >= 2
            and train_ids[idx].numel() <= args.max_len
            and (train_masks[idx] is None or train_masks[idx].any())
        )

    def run_segment(gi: int, epoch: int, budget: float, resume_slot_states, resume_full_state) -> None:
        """Trains one config group for up to `budget` tokens, continuing
        that group's example order from ptrs[gi]."""
        nonlocal global_step, total_tokens, last_ckpt_tokens, trained_any

        group = groups[gi]
        cfg = group[0]
        chunk_len = cfg.chunk_len
        batch_size = cfg.batch_size
        # accum_steps (how many chunks to accumulate before an optimizer
        # step) is derived from accum_tokens (how many real tokens per slot
        # that should represent), not taken directly from the CLI -- a raw
        # step count would silently mean a different amount of real training
        # every time chunk_len changes (same reasoning as
        # --ckpt-every-tokens being token-based rather than step-based: see
        # this module's docstring), which now includes changing between
        # slices. Total tokens per optimizer step end up ~accum_tokens *
        # batch_size (each slot contributes accum_tokens, not accum_tokens /
        # batch_size).
        accum_steps = max(1, round(args.accum_tokens / chunk_len))
        if set_grad_ckpt_fn is not None:
            set_grad_ckpt_fn(model, cfg.grad_checkpoint, args.grad_ckpt_block)

        order = orders[gi]
        n_valid = len(order)
        next_ptr = ptrs[gi]
        segment_tokens = 0.0
        base_tokens = group_tokens[gi]

        def group_state() -> dict:
            if len(groups) == 1:
                return {}
            return {
                "group_idx": gi,
                "group_ptrs": [next_ptr if i == gi else p for i, p in enumerate(ptrs)],
                "group_tokens": [base_tokens + segment_tokens if i == gi else t
                                 for i, t in enumerate(group_tokens)],
            }

        if len(groups) > 1:
            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}]  segment: {', '.join(s.path for s in group)}  "
                f"chunk_len {chunk_len}  batch {batch_size}  "
                f"grad_ckpt {'on' if cfg.grad_checkpoint else 'off'}  "
                f"examples {next_ptr}/{n_valid}  budget {budget:,.0f} tok"
            )

        # Assign initial examples to slots.
        slots: list[_Slot | None] = []
        for b in range(batch_size):
            if next_ptr < n_valid:
                idx = order[next_ptr]
                next_ptr += 1
                ids = train_ids[idx].to(device)
                mask = train_masks[idx].to(device) if train_masks[idx] is not None else None
                recall = train_recall[idx].to(device) if train_recall[idx] is not None else None
                slots.append(_Slot(b, idx, ids, mask, recall, train_sleeps[idx]))
            else:
                slots.append(None)

        if all(s is None for s in slots):
            ptrs[gi] = next_ptr
            return

        # Whether this segment will resume from an exactly-saved internal
        # state (mem_state.pt) -- if so, initializing a fresh batched state
        # below would just be immediately discarded in favor of it, and for
        # a model with a sizeable per-layer/per-slot state (e.g.
        # mamba2_2_7b_memory's neural memory weights) that fresh allocation
        # briefly coexists with the just-loaded saved state right when VRAM
        # is already tightest (model + optimizer + mem_state.pt have all
        # just landed on the GPU) -- exactly the moment this project's dev
        # GPU has been observed to OOM. Skip it entirely on this path.
        use_full_state = resume_slot_states is not None and resume_full_state is not None

        # Initialize batched model state (batch_size slots).
        if use_full_state:
            # Loaded here (not by main(), see run_training's docstring) so
            # the only reference to it is this local variable, which we
            # drop immediately below -- from then on the only thing
            # holding it alive is batched_state itself, exactly like a
            # freshly-initialized state, so it's collected the same way
            # once the first chunk's detach() replaces it.
            batched_state = torch.load(resume_full_state, map_location=device, weights_only=False)
            del resume_full_state
        elif init_state_fn is not None:
            batched_state = init_state_fn(model, batch_size, device)
        else:
            batched_state = None  # model initializes it on first chunk_loss call

        # On resume: restore each slot's example/position from the checkpoint
        # first -- needed regardless of how (or whether) internal state is
        # recovered below.
        if resume_slot_states is not None:
            for b, saved in enumerate(resume_slot_states):
                if saved is None or b >= len(slots) or slots[b] is None:
                    continue
                example_idx, pos = saved
                if example_idx not in order:
                    continue  # example was filtered out -- start slot fresh
                ids = train_ids[example_idx].to(device)
                mask = train_masks[example_idx].to(device) if train_masks[example_idx] is not None else None
                recall = train_recall[example_idx].to(device) if train_recall[example_idx] is not None else None
                slots[b] = _Slot(b, example_idx, ids, mask, recall, train_sleeps[example_idx])
                slots[b].seek(pos)

            if use_full_state:
                # Exact resume: the checkpoint saved the full batched
                # internal state (see rotate_full_state) -- use it as-is,
                # continuing each slot from its saved position.
                print("loaded saved internal state for resume")
            else:
                # No exact state available for this checkpoint (older
                # checkpoint, pruned past --keep-full-state, or a
                # batch-size mismatch -- see main()). Reconstructing it by
                # replaying the prefix would use the model's *current*
                # (already further-trained) weights, not the weights that
                # were actually live token-by-token when that prefix was
                # first trained -- an approximation, not the real state --
                # and doing so batched alongside slots that need no replay
                # means feeding some rows dummy zero-token padding, which
                # is degenerate input the model has never been asked to
                # process and has been observed to produce non-finite
                # state. Simpler and more robust to just restart each such
                # slot's example from the beginning: a bounded amount of
                # duplicated training, not an approximation or a stability
                # risk.
                n_restarted = sum(1 for slot in slots if slot is not None and slot.pos > 0)
                for slot in slots:
                    if slot is not None:
                        slot.seek(0)
                if n_restarted:
                    print(f"no saved internal state for resume -- restarting {n_restarted} slot(s) from the beginning of their example")

        accum_count = 0
        window_loss_sum = 0.0
        window_tokens = 0.0
        prev_n_lines = 0

        while any(s is not None for s in slots) and not budget_spent():
            recall_w = recall_weight_at(global_step, recall_start, recall_end, recall_ramp_steps, recall_ramp_shape)

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
                    batch_weights.append(torch.zeros(chunk_len, dtype=torch.float32, device=device))
                    chunk_actual_lens.append(0)
                else:
                    # Fire any sleep whose offset this slot has reached: wipe
                    # its backbone state (persistent memory carries on) before
                    # the next chunk. See _Slot's docstring for the
                    # chunk-boundary snapping.
                    if sleep_slot_fn is not None and batched_state is not None:
                        while slot.sleep_i < len(slot.sleeps) and slot.pos >= slot.sleeps[slot.sleep_i]:
                            sleep_slot_fn(model, batched_state, slot.slot_idx)
                            slot.last_reset = slot.pos
                            slot.sleep_i += 1
                            print(
                                f"  [sleep] slot {slot.slot_idx}: backbone wiped at token "
                                f"{slot.pos}/{slot.seqlen} (example {slot.example_idx}, "
                                f"sleep {slot.sleep_i}/{len(slot.sleeps)}; memory persists)"
                            )
                    end = min(slot.pos + chunk_len, slot.seqlen - 1)
                    actual = end - slot.pos
                    inp = slot.ids[slot.pos:end]
                    tgt = slot.ids[slot.pos + 1:end + 1]
                    wt = torch.ones(chunk_len, dtype=torch.float32, device=device)
                    if actual < chunk_len:
                        pad = chunk_len - actual
                        inp = F.pad(inp, (0, pad))
                        tgt = F.pad(tgt, (0, pad))
                        wt[actual:] = 0.0
                    if args.head_weight != 1.0:
                        # Weights index target tokens: position i predicts the
                        # token at absolute position slot.pos + 1 + i. The ramp
                        # is measured from the last backbone reset (example
                        # start or a fired sleep), so every empty-state regime
                        # gets the boost, not just the example's first tokens.
                        tpos = torch.arange(
                            slot.pos + 1 - slot.last_reset,
                            slot.pos + 1 - slot.last_reset + chunk_len,
                            device=device, dtype=torch.float32,
                        )
                        wt *= 1.0 + (args.head_weight - 1.0) * (1.0 - tpos / args.head_tokens).clamp_(min=0.0)
                    if slot.recall is not None and recall_w != 1.0:
                        rm = slot.recall[slot.pos + 1:end + 1]
                        if actual < chunk_len:
                            rm = F.pad(rm, (0, chunk_len - actual))
                        wt = torch.where(rm, wt * recall_w, wt)
                    batch_inputs.append(inp)
                    batch_targets.append(tgt)
                    batch_weights.append(wt)
                    chunk_actual_lens.append(actual)

            input_ids = torch.stack(batch_inputs)   # (B, chunk_len)
            target_ids = torch.stack(batch_targets)  # (B, chunk_len)
            weight_mask = torch.stack(batch_weights)  # (B, chunk_len) float: 0 = padding, may carry >1 boosts

            loss_sum, weight_sum, batched_state = hooks.chunk_loss(
                model, input_ids, target_ids, weight_mask, batched_state, args.eos_weight
            )

            chunk_was_non_finite = not torch.isfinite(loss_sum)
            if chunk_was_non_finite:
                # Skip only this chunk's contribution -- no backward() was
                # called, so there's nothing of this chunk's to undo. Leave
                # any gradients already accumulated from other chunks earlier
                # in this window alone rather than discarding them too.
                print(f"  warning: non-finite loss, skipping chunk")
                del loss_sum
            elif weight_sum > 0:
                # torch.autograd.grad (not loss.backward()) so this chunk's
                # gradient comes back as its own tensor instead of being
                # summed straight into .grad -- lets us check finiteness
                # BEFORE merging it into the window's running accumulation,
                # so a single bad chunk only costs that chunk, not the
                # whole window's worth of already-accumulated good chunks
                # (loss_sum being finite, checked above, does NOT guarantee
                # a finite gradient -- an op can have a perfectly finite
                # forward value but a non-finite local derivative, e.g. near
                # a sqrt/div singularity; confirmed in practice, see git
                # history for the real run this was caught from).
                chunk_grads = torch.autograd.grad(
                    loss_sum / weight_sum, trainable_params, allow_unused=True
                )
                grad_is_finite = all(g is None or torch.isfinite(g).all() for g in chunk_grads)
                if not grad_is_finite:
                    if chunk_extra_log_fn is not None:
                        _clear_live(prev_n_lines)
                        prev_n_lines = 0
                    print(f"  warning: non-finite gradient, discarding chunk")
                    del chunk_grads
                else:
                    for p, g in zip(trainable_params, chunk_grads):
                        if g is None:
                            continue
                        p.grad = g if p.grad is None else p.grad + g
                    del chunk_grads
                    window_loss_sum += loss_sum.item()
                    window_tokens += weight_sum.item()
                    # Real token count, not weight_sum: eos/recall/head boosts
                    # inflate weight_sum, and checkpoint cadence should track
                    # actual tokens trained.
                    total_tokens += sum(chunk_actual_lens)
                    accum_count += 1
                    trained_any = True
                    prev_n_lines = _show_batch_progress(
                        model, chunk_extra_log_fn, slots, chunk_actual_lens, loss_sum, weight_sum, prev_n_lines
                    )

            # The segment/mix budget counts tokens fed, finite or not -- it
            # schedules work rather than accounting for training, and a
            # segment whose chunks all came back non-finite still has to end.
            segment_tokens += sum(chunk_actual_lens)

            batched_state = batched_state.detach() if batched_state is not None else None

            if chunk_was_non_finite:
                # The forward graph for a skipped chunk is never walked by
                # backward(), so it's never freed the way a normal chunk's
                # graph is. Some models' hooks (e.g. mamba2_2_7b_memory's
                # _NeuralMemory.write, which calls torch.autograd.grad(...,
                # create_graph=True) every token) build graphs that contain
                # genuine Python-level reference cycles for that reason --
                # ordinary CPython refcounting can't reclaim a cycle at all,
                # only the generational cyclic collector can, and that runs
                # on its own schedule rather than immediately when the last
                # external reference (loss_sum, the pre-detach state above)
                # is dropped. Left alone, a run of consecutive non-finite
                # chunks can pile up several uncollected graphs before the
                # collector catches up. Both locals' external references are
                # already dropped by this point, so an explicit collection
                # here reclaims the cycle right away instead of leaving it
                # for whenever gc's thresholds next trigger.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # A chunk with non-finite loss can leave the carried recurrent
            # state (batched_state) itself non-finite too -- if so, don't let
            # it keep propagating into that slot's future chunks (or a later
            # example that inherits the slot). Force the slot to look
            # "finished" so the assign-next-example logic below runs, which
            # already resets that slot's state via reset_slot_fn. Only
            # possible for models that expose per-slot state repair.
            if reset_slot_fn is not None and batched_state is not None:
                bad_slots = [
                    b for b, slot in enumerate(slots)
                    if slot is not None and not _slot_state_finite(batched_state, b)
                ]
                if bad_slots:
                    # Unlike the per-chunk "non-finite loss, skipping chunk"
                    # warning above (fine to be transient -- it's routine
                    # and would otherwise spam the scrollback every chunk),
                    # an abandoned example is rarer and worth keeping
                    # visible: clear the live block first so this doesn't
                    # just get silently overwritten by the next chunk's
                    # live update (prev_n_lines' cursor math has no idea an
                    # extra line was printed in between, so without this it
                    # clobbers the warning instead of the intended live
                    # line), then reset prev_n_lines so the live block
                    # resumes fresh below it instead of trying to overwrite
                    # up into it.
                    if chunk_extra_log_fn is not None:
                        _clear_live(prev_n_lines)
                        prev_n_lines = 0
                    for b in bad_slots:
                        print(f"  warning: non-finite internal state in slot {b}, abandoning example and resetting state")
                        slots[b].pos = slots[b].seqlen

            # Advance slot positions; assign next example to any that finished.
            # A slot goes idle instead once the segment is over budget, so
            # the group hands over after a drain rather than mid-example.
            for b, slot in enumerate(slots):
                if slot is None:
                    continue
                slot.pos += chunk_actual_lens[b]
                if slot.is_done():
                    if next_ptr < n_valid and segment_tokens < budget:
                        idx = order[next_ptr]
                        next_ptr += 1
                        ids = train_ids[idx].to(device)
                        mask = train_masks[idx].to(device) if train_masks[idx] is not None else None
                        recall = train_recall[idx].to(device) if train_recall[idx] is not None else None
                        slots[b] = _Slot(b, idx, ids, mask, recall, train_sleeps[idx])
                        if reset_slot_fn is not None and batched_state is not None:
                            reset_slot_fn(model, batched_state, b)
                    else:
                        slots[b] = None
                        if reset_slot_fn is not None and batched_state is not None:
                            reset_slot_fn(model, batched_state, b)

            # Gradient accumulation step.
            if accum_count >= accum_steps:
                if chunk_extra_log_fn is not None:
                    _clear_live(prev_n_lines)
                    prev_n_lines = 0

                for p in trainable_params:
                    if p.grad is not None:
                        p.grad /= accum_count

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                set_lr(global_step)
                avg_loss = window_loss_sum / window_tokens
                accum_count = 0
                window_loss_sum = window_tokens = 0.0
                ts = datetime.now().strftime("%H:%M:%S")

                if on_step_fn is not None:
                    on_step_fn(model, global_step)

                ramping = f"  recall_w {recall_w:.2f}" if recall_w != recall_end else ""
                print(f"[{ts}]  epoch {epoch + 1}  step {global_step:>6}  examples {next_ptr}/{n_valid}  loss {avg_loss:.4f}  gnorm {grad_norm:.3f}  lr {used_lr:.2e}{ramping}")
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
                    path = save_checkpoint(
                        model, optimizer, global_step, epoch, slots, next_ptr,
                        total_tokens, args.lora_rank, args.lora_alpha,
                        batched_state=batched_state if args.keep_full_state > 0 else None,
                        dataset_fingerprint=data_fp,
                        memory_window=memory_window,
                        **group_state(),
                    )
                    last_ckpt_tokens = total_tokens
                    rotate_checkpoints(args.keep_ckpts, epoch)
                    rotate_full_state(args.keep_full_state)
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}]  saved {path}")

        if chunk_extra_log_fn is not None:
            _clear_live(prev_n_lines)

        # Step on any remaining accumulated gradient before handing the
        # batch over to the next segment, whose slots and chunk length are
        # different ones.
        if accum_count > 0:
            for p in trainable_params:
                if p.grad is not None:
                    p.grad /= accum_count
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            set_lr(global_step)
            if on_step_fn is not None:
                on_step_fn(model, global_step)

        ptrs[gi] = next_ptr
        group_tokens[gi] = base_tokens + segment_tokens
        final_save.update(epoch=epoch, slots=slots, next_ptr=next_ptr,
                          batched_state=batched_state, group_state=group_state())

    for epoch in range(start_epoch, args.epochs):
        if budget_spent():
            break
        orders = [build_order(g, train_ids, epoch, keep) for g in groups]
        shares = [sum(resolve_share(s, train_ids) for s in g) for g in groups]
        ptrs = [0] * len(groups)
        group_tokens = [0.0] * len(groups)

        resume_slot_states = None
        resume_full_state = None
        resume_group = None
        if epoch == start_epoch and start_slot_states is not None:
            resume_slot_states = start_slot_states
            resume_full_state = start_full_state
            resume_group = min(start_group_idx, len(groups) - 1)
            if start_group_ptrs is not None and len(start_group_ptrs) == len(groups):
                ptrs = list(start_group_ptrs)
                group_tokens = list(start_group_tokens or [0.0] * len(groups))
            else:
                ptrs[resume_group] = start_next_ptr
        elif epoch == start_epoch:
            ptrs[min(start_group_idx, len(groups) - 1)] = start_next_ptr

        while any(ptrs[i] < len(orders[i]) for i in range(len(groups))) and not budget_spent():
            available = [i for i in range(len(groups)) if ptrs[i] < len(orders[i])]
            if resume_group is not None and resume_group in available:
                gi = resume_group
            else:
                # The resumed group is already finished, so its saved slot
                # positions have nothing to restore into.
                gi, resume_slot_states, resume_full_state = pick_deficit(group_tokens, shares, available), None, None
            run_segment(gi, epoch, segment_budget, resume_slot_states, resume_full_state)
            resume_slot_states = resume_full_state = resume_group = None

    if not trained_any:
        print("nothing to train -- already at or past the requested epochs. Pass a larger --epochs to continue.")
        return

    path = save_checkpoint(
        model, optimizer, global_step, final_save["epoch"], final_save["slots"], final_save["next_ptr"],
        total_tokens, args.lora_rank, args.lora_alpha,
        batched_state=final_save["batched_state"] if args.keep_full_state > 0 else None,
        dataset_fingerprint=data_fp,
        memory_window=memory_window,
        **final_save["group_state"],
    )
    rotate_checkpoints(args.keep_ckpts, final_save["epoch"])
    rotate_full_state(args.keep_full_state)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  done. final checkpoint: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"SFT for {MODEL_NAME}")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID (informational; actual ID comes from models/ file)")
    parser.add_argument("--data", action="append", default=None, help="Tokenized dataset from prepare_data.py (default: data/train.pt). Repeat to train on several slices at once; each may carry comma-separated per-slice overrides, e.g. --data 'data/train_cram.pt,share=35,chunk-len=512,batch-size=6,grad-checkpoint=1,shuffle=0'. share= is that slice's requested fraction of trained tokens (any units -- shares are normalised; give it for every slice or none, in which case each slice's own token count is used); chunk-len/batch-size/grad-checkpoint override --chunk-len/--batch-size/off for this slice only; shuffle=0 consumes the artifact in the order it was written, which is how a generator-side curriculum survives training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=32, help="Linearly ramp the learning rate from 0 to --lr over this many optimizer steps, then hold at --lr -- 0 to disable. A pure function of global_step, so it resumes correctly with no extra checkpoint state.")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps and save the final checkpoint (default: run to --epochs). Counts global_step, so a resumed run finishes the same budget rather than taking this many more.")
    parser.add_argument("--max-len", type=int, default=None, help="Skip examples longer than this (default: no limit)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward/backward chunk -- defaults to the model's own DEFAULT_CHUNK_LEN")
    parser.add_argument("--memory-window", type=int, default=None, help="Tokens per memory-subsystem write, for models that define set_memory_window (currently mamba2_2_7b_memory only; no-op otherwise) -- defaults to the model's own DEFAULT_MEMORY_WINDOW (1, i.e. a write every token, unless overridden). Must evenly divide --chunk-len. See docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md.")
    parser.add_argument("--batch-size", type=int, default=6, help="Number of examples to train in parallel (slot-based batching)")
    parser.add_argument("--accum-tokens", type=int, default=256, help="Target real tokens per slot to accumulate before each optimizer step -- converted internally to a chunk count (accum_tokens / chunk_len), so it means the same amount of real training regardless of --chunk-len. Total tokens per optimizer step end up ~accum_tokens * batch_size.")
    parser.add_argument("--ckpt-every-tokens", type=int, default=5000, help="Save checkpoint every N tokens of training")
    parser.add_argument("--keep-ckpts", type=int, default=50, help="Number of checkpoints to retain")
    parser.add_argument("--keep-full-state", type=int, default=2, help="Number of most-recent checkpoints to also save full internal model state for (mem_state.pt) -- lets resume continue mid-example slots exactly instead of restarting them from the beginning. 0 to disable. Only applies to models whose train_hooks define init_state (e.g. mamba2_2_7b_memory); no-op otherwise (falls back to always restarting mid-example slots).")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eos-weight", type=float, default=5.0, help="Loss weight for EOS tokens (>1 to emphasise stopping)")
    parser.add_argument("--recall-weight", type=float, default=8.0, help="Loss weight multiplier for tokens marked True in the dataset's optional recall_masks tensors (prepare_chains.py/prepare_interference.py mark their spliced query-answer tokens) -- amplifies the recall training signal, which is otherwise a tiny fraction (~0.1%%) of all tokens. 1.0 reproduces the unweighted objective. No-op on datasets without recall_masks.")
    parser.add_argument("--recall-ramp-start", type=float, default=1.0, help="Recall-mask multiplier at step 0, ramping to --recall-weight over --recall-ramp-steps. Equal to --recall-weight to disable the ramp.")
    parser.add_argument("--recall-ramp-steps", type=int, default=32, help="Optimizer steps over which the recall multiplier ramps from --recall-ramp-start to --recall-weight, then holds. Keep this aligned with the model's beta-anneal window (mamba2_2_7b_memory's BETA_BIAS_ANNEAL_STEPS, 32): the anneal window is where the gradient decides whether the memory path is useful or gets suppressed, and a full-strength recall multiplier landing in it amplifies the loss spike rather than the signal. 0 disables the ramp (constant --recall-weight from step 0).")
    parser.add_argument("--recall-ramp-shape", choices=("linear", "geometric"), default="linear", help="Interpolation between --recall-ramp-start and --recall-weight: linear in the multiplier, or linear in its log (geometric), which spends more of the window near the low end.")
    parser.add_argument("--grad-ckpt-block", type=int, default=None, help="Checkpoint block size in tokens for slices with grad-checkpoint=1 (default: the model's own, currently 64). The backward pass recomputes one block's live graph at a time, so block must stay under what the card can hold as a live graph -- this box's ~52-token ceiling means local grad-checkpointed runs need 48 or less, while the default is sized for a rented card. Snapped down to a multiple of --memory-window by the model.")
    parser.add_argument("--mix-segment-tokens", type=int, default=1_000_000, help="Tokens one config group trains for before the mix hands over to whichever group is furthest behind its share. Only applies when --data slices disagree on chunk-len/batch-size/grad-checkpoint (slices that agree share a batch and interleave example-by-example instead); with a single config the budget is unbounded and the loop is the single-dataset one. Smaller mixes more finely but pays a slot drain per handover.")
    parser.add_argument("--head-weight", type=float, default=4.0, help="Loss weight multiplier at the first token after each backbone reset (example start, and each sleep for datasets with sleep_positions), decaying linearly to 1.0 over --head-tokens -- emphasises the empty-state regime, which is otherwise underweighted because most tokens sit deep inside long examples. 1.0 reproduces the unweighted objective.")
    parser.add_argument("--head-tokens", type=int, default=1024, help="Length of the --head-weight linear decay ramp, in tokens from each backbone reset")
    parser.add_argument("--freeze-lora", action="store_true", help="Freeze the parametric LoRA weights and optimize only the memory subsystem (front_end + injection modules). Isolates whether the memory can carry recall on its own when the parametric path can no longer re-absorb the niche. Checkpoints stay complete (LoRA held at its resumed values).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for everything not covered by the per-epoch data-shuffle seed (see run_training) -- LoRA init/dropout, and for models with per-sequence random state (e.g. mamba2_2_7b_memory's neural-memory init/reset) -- fixed by default so a run (or a crash) is reproducible; pass a different value to sample a different random init.")
    parser.add_argument("--detect-anomaly", action="store_true", help="Enable torch.autograd.set_detect_anomaly -- when a chunk's gradient comes back non-finite (the run's existing per-chunk check, see run_training), instead of just discarding it and continuing, autograd raises immediately with a traceback pointing at the exact forward op responsible, and the run stops there. Diagnostic only: real, not-small overhead (extra bookkeeping on every op during forward), and turns the normally-recoverable non-finite-gradient path into a hard stop -- use a dedicated short run to localize a real crash, not the long unattended one. See `make detect-anomaly`.")
    args = parser.parse_args()
    args.max_len = args.max_len if args.max_len is not None else math.inf
    args.data = args.data or ["data/train.pt"]
    if args.recall_ramp_shape == "geometric" and args.recall_ramp_start <= 0:
        parser.error("--recall-ramp-start must be > 0 for a geometric ramp")
    try:
        specs = [parse_data_spec(text, args.chunk_len, args.batch_size) for text in args.data]
        check_shares(specs)
    except ValueError as e:
        parser.error(str(e))
    for spec in specs:
        if spec.chunk_len is None:
            spec.chunk_len = hooks.DEFAULT_CHUNK_LEN
    torch.manual_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("--detect-anomaly enabled: forward pass will be slower, and the run will stop with a full traceback the first time a chunk's gradient is non-finite (instead of discarding that chunk and continuing).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"device: {device} -- {torch.cuda.get_device_name(device)} (index {torch.cuda.current_device()})")
        print(f"  VRAM total:  {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"  VRAM free:   {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
    else:
        print(f"device: {device} (no CUDA/ROCm device found)")

    print(f"loading {args.model} ...")
    model, trainable_params = hooks.setup_training(device, args.lora_rank, args.lora_alpha, args.lora_dropout)

    if args.freeze_lora:
        # Freeze the parametric (LoRA) path and train only the memory subsystem
        # (front_end projections + injection modules): removes the parametric
        # re-absorption route so the memory alone must carry cross-sleep recall.
        # requires_grad stays True on everything so checkpoints remain complete
        # (save_checkpoint keys off requires_grad); only the optimizer's param
        # set shrinks, so LoRA values are held fixed at the resumed checkpoint.
        name_by_id = {id(p): n for n, p in model.named_parameters()}
        trainable_params = [p for p in trainable_params
                            if "lora" not in name_by_id.get(id(p), "").lower()]
        print(f"--freeze-lora: optimizing {len(trainable_params)} memory params "
              f"(front_end + injections); LoRA held fixed")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    start_step = 0
    start_epoch = 0
    start_slot_states = None
    start_next_ptr = 0
    start_total_tokens = 0.0
    start_last_ckpt_tokens = 0.0
    start_full_state = None
    start_dataset_fingerprint = None
    start_group_idx = 0
    start_group_ptrs = None
    start_group_tokens = None
    groups = group_specs(specs)
    if args.resume:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            print(f"resuming from {ckpt}")
            load_checkpoint(model, ckpt)
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                # A checkpoint saved under --freeze-lora carries AdamW state
                # over just the memory params; one saved under full training
                # covers the whole trainable set. Load whenever the saved
                # state matches the current optimizer's param count -- a
                # mismatch (e.g. full-training state resumed with
                # --freeze-lora) falls back to a fresh optimizer.
                saved = torch.load(opt_path, map_location=device, weights_only=True)
                n_current = sum(len(g["params"]) for g in optimizer.param_groups)
                n_saved = sum(len(g["params"]) for g in saved["param_groups"])
                if n_current == n_saved:
                    optimizer.load_state_dict(saved)
                else:
                    print(f"starting a fresh optimizer (saved state covers {n_saved} params, current set has {n_current})")
            start_step = int(ckpt.name.split("-")[1])
            state_path = ckpt / "state.pt"
            state_batch_size = None
            if state_path.exists():
                state = torch.load(state_path, weights_only=True)
                start_epoch = state["epoch"]
                start_slot_states = state.get("slot_states")
                start_next_ptr = state.get("next_ptr", 0)
                start_dataset_fingerprint = state.get("dataset_fingerprint")
                start_total_tokens = state.get("total_tokens", 0.0)
                start_last_ckpt_tokens = state.get("last_ckpt_tokens", 0.0)
                state_batch_size = state.get("state_batch_size")
                start_group_idx = state.get("group_idx", 0) or 0
                start_group_ptrs = state.get("group_ptrs")
                start_group_tokens = state.get("group_tokens")
                # Legacy checkpoint format (single-example, no slot_states):
                # map old example_idx/chunk_pos to a single-slot slot_states.
                if start_slot_states is None:
                    example_idx = state.get("example_idx", 0)
                    chunk_pos = state.get("chunk_pos", 0)
                    start_slot_states = [(example_idx, chunk_pos)]
                    start_next_ptr = example_idx + (0 if chunk_pos > 0 else 1)
            # mem_state.pt (the saved full internal state -- see
            # save_checkpoint/rotate_full_state) only exists on recent
            # checkpoints and only when its batch size still matches this
            # run's --batch-size; missing or mismatched just means any
            # mid-example slot restarts from the beginning instead of
            # continuing exactly, not a broken resume. Not loaded here --
            # just the path is handed to run_training, which loads it
            # itself right before use (see run_training's docstring for
            # why: main()'s own frame outlives the entire training run, so
            # a local variable here bound to the loaded tensors would keep
            # them VRAM-resident for the whole run).
            mem_state_path = ckpt / "mem_state.pt"
            if mem_state_path.exists():
                resume_batch_size = groups[min(start_group_idx, len(groups) - 1)][0].batch_size
                if state_batch_size == resume_batch_size:
                    start_full_state = mem_state_path
                else:
                    print(
                        f"mem_state.pt batch size ({state_batch_size}) doesn't match "
                        f"the resumed slice's batch size ({resume_batch_size}) -- ignoring it, restarting mid-example slots from the beginning"
                    )
            print(f"resumed at step {start_step}, epoch {start_epoch + 1}, next_ptr {start_next_ptr}")
        else:
            print("no checkpoint found, starting fresh")

    print(f"loading {len(specs)} dataset slice(s) ...")
    train_ids, train_masks, train_recall, train_sleeps = load_datasets(specs)
    if args.recall_weight != 1.0 and all(r is None for r in train_recall):
        print(f"warning: --recall-weight {args.recall_weight} given but no slice has recall_masks -- it will have no effect")
    if any(s is not None and len(s) for s in train_sleeps) and getattr(hooks, "sleep_slot", None) is None:
        print(f"warning: a slice has sleep_positions but {MODEL_NAME}'s train_hooks defines no sleep_slot -- sleeps will be ignored")

    n = len(train_ids)
    print(f"train: {n}  epochs: {args.epochs}")
    for group in groups:
        cfg = group[0]
        print(
            f"  config group: {', '.join(s.path for s in group)}  chunk_len {cfg.chunk_len}  "
            f"batch {cfg.batch_size}  grad_ckpt {'on' if cfg.grad_checkpoint else 'off'}"
        )

    if args.resume and start_slot_states is not None:
        current_fp = datasets_fingerprint(specs)
        discard = False
        if start_dataset_fingerprint is None:
            # Older checkpoint format, saved before dataset_fingerprint
            # existed -- unlike an outright mismatch below, this is
            # ambiguous (could be the same dataset that was always in use,
            # or a swap that just happens to predate fingerprinting), so
            # ask rather than silently guessing either way.
            print(f"resume: checkpoint has no dataset fingerprint (older format) -- current --data is {current_fp}")
            try:
                answer = input("Is this the same dataset the checkpoint was trained on? [Y/n] ").strip().lower()
            except EOFError:
                answer = ""
                print("(no input available -- defaulting to 'n': safer to restart slots than risk reindexing into the wrong dataset)")
            discard = answer.startswith("n")
        elif start_dataset_fingerprint != current_fp:
            print(f"resume: dataset changed ({start_dataset_fingerprint} vs {current_fp})")
            discard = True

        if discard:
            print(
                "resume: discarding slot_states/next_ptr from the checkpoint so slots start "
                "fresh against this dataset instead of reindexing into it with stale positions"
            )
            start_slot_states = None
            start_next_ptr = 0
            start_full_state = None
            start_group_idx = 0
            start_group_ptrs = None
            start_group_tokens = None

    run_training(
        hooks, model, optimizer, trainable_params, train_ids, train_masks, train_recall, train_sleeps, device, args,
        start_epoch, start_slot_states, start_next_ptr, start_step, start_total_tokens, start_last_ckpt_tokens,
        start_full_state, specs=specs, start_group_idx=start_group_idx,
        start_group_ptrs=start_group_ptrs, start_group_tokens=start_group_tokens,
    )


if __name__ == "__main__":
    main()
