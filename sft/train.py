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

from pathlib import Path

import torch
from training import checkpoints, cli, loop
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

MODEL_NAME = cli.default_model_name()
_runtime = cli.load_model_runtime(MODEL_NAME)
MODEL_ID = _runtime.model_id
hooks = _runtime.hooks

CKPT_DIR = Path(__file__).parent.parent / "models" / MODEL_NAME / "checkpoints"


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


from training.loop import (
    _Slot,
    _clear_live,
    _collect_tensors,
    _print_live,
    _show_batch_progress,
    _slot_state_finite,
)


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
    return loop.run_training(
        CKPT_DIR, MODEL_NAME, hooks, model, optimizer, trainable_params,
        train_ids, train_masks, train_recall, train_sleeps, device, args,
        start_epoch, start_slot_states, start_next_ptr, start_step,
        start_total_tokens, start_last_ckpt_tokens, start_full_state, specs,
        start_group_idx, start_group_ptrs, start_group_tokens,
    )


def main() -> None:
    cli.main(CKPT_DIR, MODEL_NAME)


__all__ = [
    "CKPT_DIR",
    "DataSpec",
    "MODEL_ID",
    "MODEL_NAME",
    "_Slot",
    "_clear_live",
    "_collect_tensors",
    "_print_live",
    "_show_batch_progress",
    "_slot_state_finite",
    "build_order",
    "check_shares",
    "dataset_fingerprint",
    "datasets_fingerprint",
    "hooks",
    "group_specs",
    "iter_checkpoints",
    "latest_checkpoint",
    "load_checkpoint",
    "load_datasets",
    "main",
    "parse_data_spec",
    "pick_deficit",
    "recall_weight_at",
    "rotate_checkpoints",
    "rotate_full_state",
    "run_training",
    "resolve_share",
    "save_checkpoint",
]


if __name__ == "__main__":
    main()
