"""Training slots, execution loop, and live progress."""

import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch
import torch.nn.functional as F

from progress import heartbeat
from progress import ts as timestamp
from training import checkpoints
from training.datasets import (
    DataSpec,
    build_order,
    datasets_fingerprint,
    group_specs,
    pick_deficit,
    recall_weight_at,
    resolve_share,
)

DatasetFingerprint = dict[str, str | int] | list[dict[str, str | int]]


class SegmentArgs(Protocol):
    accum_tokens: float
    ckpt_every_tokens: float
    eos_weight: float
    grad_ckpt_block: int | None
    head_tokens: int
    head_weight: float
    keep_ckpts: int
    keep_full_state: int
    lora_alpha: float
    lora_rank: int


class SegmentHooks(Protocol):
    def chunk_loss(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        mask_slice: torch.Tensor,
        state: object,
        eos_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, object]: ...


@dataclass
class SegmentExecution:
    """Model and optimizer inputs for one segment."""

    hooks: SegmentHooks
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    trainable_params: list[torch.nn.Parameter]
    device: torch.device | str
    args: SegmentArgs


@dataclass
class SegmentData:
    """Immutable examples and their checkpoint identity."""

    train_ids: list[torch.Tensor]
    train_masks: list[torch.Tensor | None]
    train_recall: list[torch.Tensor | None]
    train_sleeps: list[torch.Tensor | list[int] | None]
    data_fp: DatasetFingerprint


@dataclass
class SegmentCallbacks:
    """Model-specific operations used by the generic segment loop."""

    extra_log_fn: Callable[[torch.nn.Module], str | list[str] | None] | None
    chunk_extra_log_fn: Callable[[torch.nn.Module], str | list[str] | None] | None
    on_step_fn: Callable[[torch.nn.Module, int], None] | None
    reset_slot_fn: Callable[[torch.nn.Module, object, int], None] | None
    sleep_slot_fn: Callable[[torch.nn.Module, object, int], None] | None
    init_state_fn: Callable[[torch.nn.Module, int, torch.device | str], object] | None
    set_grad_ckpt_fn: Callable[[torch.nn.Module, bool, int | None], None] | None
    set_lr: Callable[[int], None]


@dataclass
class SegmentCheckpointPolicy:
    """Checkpoint and stopping controls shared by all segments."""

    ckpt_dir: Path
    memory_window: int | None
    max_steps: float


@dataclass
class SegmentLossPolicy:
    """Recall-loss ramp values fixed for the training run."""

    recall_start: float
    recall_end: float
    recall_ramp_steps: int
    recall_ramp_shape: str


@dataclass
class SegmentSchedule:
    """The group order and resume position for one segment."""

    groups: list[list[DataSpec]]
    orders: list[list[int]]
    ptrs: list[int]
    group_tokens: list[float]
    final_save: dict
    group_idx: int
    epoch: int
    budget: float
    resume_slot_states: list | None
    resume_full_state: Path | None


@dataclass
class SegmentProgress:
    """The counters that survive from one segment to the next."""

    global_step: int
    total_tokens: float
    last_ckpt_tokens: float
    trained_any: bool


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

    __slots__ = (
        "slot_idx",
        "example_idx",
        "ids",
        "mask",
        "recall",
        "sleeps",
        "sleep_i",
        "last_reset",
        "pos",
        "seqlen",
    )

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

    def seek(self, pos: int, sleep_i: int | None = None) -> None:
        """Set the position (resume). `sleep_i` is the checkpoint's count of
        fired sleeps; a sleep reached but not yet fired at save time (the
        save runs after pos advances, the sleep check before the next chunk)
        fires on the next chunk. Legacy checkpoints without it treat every
        sleep at or before pos as fired. last_reset uses the sleep's own
        offset rather than the chunk boundary it originally fired at (off by
        < chunk_len) -- close enough for the --head-weight ramp it feeds."""
        self.pos = pos
        if sleep_i is None:
            sleep_i = sum(1 for s in self.sleeps if s <= pos)
        self.sleep_i = sleep_i
        self.last_reset = self.sleeps[sleep_i - 1] if sleep_i else 0

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
    heartbeat()
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
    ts = timestamp()

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


def run_segment(
    execution: SegmentExecution,
    data: SegmentData,
    callbacks: SegmentCallbacks,
    checkpoint_policy: SegmentCheckpointPolicy,
    loss_policy: SegmentLossPolicy,
    schedule: SegmentSchedule,
    progress: SegmentProgress,
) -> SegmentProgress:
    """Trains one config group for up to `budget` tokens, continuing
    that group's example order from ptrs[gi]."""
    ckpt_dir = checkpoint_policy.ckpt_dir
    hooks = execution.hooks
    model = execution.model
    optimizer = execution.optimizer
    trainable_params = execution.trainable_params
    train_ids = data.train_ids
    train_masks = data.train_masks
    train_recall = data.train_recall
    train_sleeps = data.train_sleeps
    device = execution.device
    args = execution.args
    data_fp = data.data_fp
    memory_window = checkpoint_policy.memory_window
    recall_start = loss_policy.recall_start
    recall_end = loss_policy.recall_end
    recall_ramp_steps = loss_policy.recall_ramp_steps
    recall_ramp_shape = loss_policy.recall_ramp_shape
    extra_log_fn = callbacks.extra_log_fn
    chunk_extra_log_fn = callbacks.chunk_extra_log_fn
    on_step_fn = callbacks.on_step_fn
    reset_slot_fn = callbacks.reset_slot_fn
    sleep_slot_fn = callbacks.sleep_slot_fn
    init_state_fn = callbacks.init_state_fn
    set_grad_ckpt_fn = callbacks.set_grad_ckpt_fn
    set_lr = callbacks.set_lr
    max_steps = checkpoint_policy.max_steps
    groups = schedule.groups
    orders = schedule.orders
    ptrs = schedule.ptrs
    group_tokens = schedule.group_tokens
    final_save = schedule.final_save
    gi = schedule.group_idx
    epoch = schedule.epoch
    budget = schedule.budget
    resume_slot_states = schedule.resume_slot_states
    resume_full_state = schedule.resume_full_state
    global_step = progress.global_step
    total_tokens = progress.total_tokens
    last_ckpt_tokens = progress.last_ckpt_tokens
    trained_any = progress.trained_any
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
            "group_tokens": [
                base_tokens + segment_tokens if i == gi else t for i, t in enumerate(group_tokens)
            ],
        }

    if len(groups) > 1:
        ts = timestamp()
        print(
            f"[{ts}]  segment: {', '.join(s.path for s in group)}  "
            f"chunk_len {chunk_len}  batch {batch_size}  "
            f"grad_ckpt {'on' if cfg.grad_checkpoint else 'off'}  "
            f"examples {next_ptr}/{n_valid}  budget {budget:,.0f} tok"
        )

    # Assign initial examples to slots: a saved in-flight example first
    # (the checkpoint's next_ptr already points past it), else the next
    # fresh one.
    slots: list[_Slot | None] = []
    for b in range(batch_size):
        saved = (
            resume_slot_states[b]
            if resume_slot_states is not None and b < len(resume_slot_states)
            else None
        )
        if saved is not None and saved[0] in order:
            idx, pos, *rest = saved
        elif next_ptr < n_valid:
            idx, pos, rest = order[next_ptr], 0, []
            next_ptr += 1
        else:
            slots.append(None)
            continue
        ids = train_ids[idx].to(device)
        mask = train_masks[idx].to(device) if train_masks[idx] is not None else None
        recall = train_recall[idx].to(device) if train_recall[idx] is not None else None
        slots.append(_Slot(b, idx, ids, mask, recall, train_sleeps[idx]))
        slots[b].seek(pos, rest[0] if rest else None)

    if all(s is None for s in slots):
        ptrs[gi] = next_ptr
        return progress

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

    if resume_slot_states is not None:
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
                print(
                    f"no saved internal state for resume -- restarting {n_restarted} slot(s) from the beginning of their example"
                )

    accum_count = 0
    window_loss_sum = 0.0
    window_tokens = 0.0
    prev_n_lines = 0

    while any(s is not None for s in slots) and global_step < max_steps:
        recall_w = recall_weight_at(
            global_step, recall_start, recall_end, recall_ramp_steps, recall_ramp_shape
        )

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
                inp = slot.ids[slot.pos : end]
                tgt = slot.ids[slot.pos + 1 : end + 1]
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
                        device=device,
                        dtype=torch.float32,
                    )
                    wt *= 1.0 + (args.head_weight - 1.0) * (1.0 - tpos / args.head_tokens).clamp_(
                        min=0.0
                    )
                if slot.recall is not None and recall_w != 1.0:
                    rm = slot.recall[slot.pos + 1 : end + 1]
                    if actual < chunk_len:
                        rm = F.pad(rm, (0, chunk_len - actual))
                    wt = torch.where(rm, wt * recall_w, wt)
                if slot.mask is not None:
                    tm = slot.mask[slot.pos + 1 : end + 1]
                    if actual < chunk_len:
                        tm = F.pad(tm, (0, chunk_len - actual))
                    wt = torch.where(tm, wt, torch.zeros_like(wt))
                batch_inputs.append(inp)
                batch_targets.append(tgt)
                batch_weights.append(wt)
                chunk_actual_lens.append(actual)

        input_ids = torch.stack(batch_inputs)  # (B, chunk_len)
        target_ids = torch.stack(batch_targets)  # (B, chunk_len)
        weight_mask = torch.stack(
            batch_weights
        )  # (B, chunk_len) float: 0 = padding, may carry >1 boosts

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
                    model,
                    chunk_extra_log_fn,
                    slots,
                    chunk_actual_lens,
                    loss_sum,
                    weight_sum,
                    prev_n_lines,
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
                b
                for b, slot in enumerate(slots)
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
                    print(
                        f"  warning: non-finite internal state in slot {b}, abandoning example and resetting state"
                    )
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
            ts = timestamp()

            if on_step_fn is not None:
                on_step_fn(model, global_step)

            ramping = f"  recall_w {recall_w:.2f}" if recall_w != recall_end else ""
            print(
                f"[{ts}]  epoch {epoch + 1}  step {global_step:>6}  examples {next_ptr}/{n_valid}  loss {avg_loss:.4f}  gnorm {grad_norm:.3f}  lr {used_lr:.2e}{ramping}"
            )
            if extra_log_fn is not None:
                line = extra_log_fn(model)
                if line is not None:
                    print(f"    {line}")

            bad = [
                name
                for name, p in model.named_parameters()
                if p.requires_grad and not torch.isfinite(p).all()
            ]
            if bad:
                print(f"FATAL: non-finite weights after step {global_step}: {bad[:5]}")
                print("Checkpoints NOT saved. Exiting.")
                raise SystemExit(1)

            if total_tokens - last_ckpt_tokens >= args.ckpt_every_tokens:
                path = checkpoints.save_checkpoint(
                    ckpt_dir,
                    model,
                    optimizer,
                    global_step,
                    epoch,
                    slots,
                    next_ptr,
                    total_tokens,
                    args.lora_rank,
                    args.lora_alpha,
                    batched_state=batched_state if args.keep_full_state > 0 else None,
                    dataset_fingerprint=data_fp,
                    memory_window=memory_window,
                    **group_state(),
                )
                last_ckpt_tokens = total_tokens
                checkpoints.rotate_checkpoints(ckpt_dir, args.keep_ckpts, epoch)
                checkpoints.rotate_full_state(ckpt_dir, args.keep_full_state)
                ts = timestamp()
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
    final_save.update(
        epoch=epoch,
        slots=slots,
        next_ptr=next_ptr,
        batched_state=batched_state,
        group_state=group_state(),
    )
    return SegmentProgress(global_step, total_tokens, last_ckpt_tokens, trained_any)


def run_training(
    ckpt_dir: Path,
    model_name: str,
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
        specs = [
            DataSpec(
                path=str(args.data),
                chunk_len=args.chunk_len,
                batch_size=args.batch_size,
                lo=0,
                hi=len(train_ids),
            )
        ]
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
        memory_window = getattr(args, "memory_window", None) or getattr(
            hooks, "DEFAULT_MEMORY_WINDOW", 1
        )
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
            f"but {model_name}'s train_hooks defines no set_grad_checkpoint -- ignored"
        )

    recall_end = args.recall_weight
    recall_start = getattr(args, "recall_ramp_start", 1.0)
    recall_ramp_steps = getattr(args, "recall_ramp_steps", 0)
    recall_ramp_shape = getattr(args, "recall_ramp_shape", "linear")
    segment_budget = (
        math.inf if len(groups) == 1 else getattr(args, "mix_segment_tokens", 1_000_000)
    )
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
    execution = SegmentExecution(hooks, model, optimizer, trainable_params, device, args)
    data = SegmentData(train_ids, train_masks, train_recall, train_sleeps, data_fp)
    callbacks = SegmentCallbacks(
        extra_log_fn,
        chunk_extra_log_fn,
        on_step_fn,
        reset_slot_fn,
        sleep_slot_fn,
        init_state_fn,
        set_grad_ckpt_fn,
        set_lr,
    )
    checkpoint_policy = SegmentCheckpointPolicy(ckpt_dir, memory_window, max_steps)
    loss_policy = SegmentLossPolicy(recall_start, recall_end, recall_ramp_steps, recall_ramp_shape)
    progress = SegmentProgress(global_step, total_tokens, last_ckpt_tokens, trained_any)

    def keep(idx: int) -> bool:
        """Examples too short, too long, or fully masked are dropped."""
        return (
            train_ids[idx].numel() >= 2
            and train_ids[idx].numel() <= args.max_len
            and (train_masks[idx] is None or train_masks[idx].any())
        )

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

        # A resumed group still has its in-flight slots to finish even when
        # its pointer is already past the last example.
        def pending(i: int) -> bool:
            return ptrs[i] < len(orders[i]) or (
                i == resume_group and resume_slot_states is not None
            )

        while any(pending(i) for i in range(len(groups))) and not budget_spent():
            if resume_group is not None:
                gi = resume_group
            else:
                available = [i for i in range(len(groups)) if pending(i)]
                gi = pick_deficit(group_tokens, shares, available)
            progress = run_segment(
                execution,
                data,
                callbacks,
                checkpoint_policy,
                loss_policy,
                SegmentSchedule(
                    groups,
                    orders,
                    ptrs,
                    group_tokens,
                    final_save,
                    gi,
                    epoch,
                    segment_budget,
                    resume_slot_states,
                    resume_full_state,
                ),
                progress,
            )
            global_step = progress.global_step
            total_tokens = progress.total_tokens
            last_ckpt_tokens = progress.last_ckpt_tokens
            trained_any = progress.trained_any
            resume_slot_states = resume_full_state = resume_group = None

    if not trained_any:
        print(
            "nothing to train -- already at or past the requested epochs. Pass a larger --epochs to continue."
        )
        return

    path = checkpoints.save_checkpoint(
        ckpt_dir,
        model,
        optimizer,
        global_step,
        final_save["epoch"],
        final_save["slots"],
        final_save["next_ptr"],
        total_tokens,
        args.lora_rank,
        args.lora_alpha,
        batched_state=final_save["batched_state"] if args.keep_full_state > 0 else None,
        dataset_fingerprint=data_fp,
        memory_window=memory_window,
        **final_save["group_state"],
    )
    checkpoints.rotate_checkpoints(ckpt_dir, args.keep_ckpts, final_save["epoch"])
    checkpoints.rotate_full_state(ckpt_dir, args.keep_full_state)
    ts = timestamp()
    print(f"[{ts}]  done. final checkpoint: {path}")
