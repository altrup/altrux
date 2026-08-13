"""Training checkpoint discovery, persistence, and loading."""

import json
import os
import shutil
from pathlib import Path

import torch


def iter_checkpoints(ckpt_dir: Path):
    """Yield (step, path) for every models/{MODEL_NAME}/checkpoints/epoch-*/step-* directory.
    Step numbers are globally monotonic, so they order checkpoints across epochs."""
    if not ckpt_dir.exists():
        return
    for epoch_dir in ckpt_dir.glob("epoch-*"):
        if not epoch_dir.is_dir():
            continue
        for p in epoch_dir.iterdir():
            if p.is_dir() and p.name.startswith("step-"):
                yield int(p.name.split("-")[1]), p


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(iter_checkpoints(ckpt_dir))
    return ckpts[-1][1] if ckpts else None


def rotate_checkpoints(ckpt_dir: Path, keep: int, epoch: int) -> None:
    """Keep only the newest `keep` checkpoints within this epoch's folder, so a
    later epoch never prunes an earlier epoch's history."""
    epoch_dir = ckpt_dir / f"epoch-{epoch + 1}"
    if not epoch_dir.exists():
        return
    steps = sorted(
        int(p.name.split("-")[1])
        for p in epoch_dir.iterdir()
        if p.is_dir() and p.name.startswith("step-")
    )
    for step in steps[:-keep]:
        shutil.rmtree(epoch_dir / f"step-{step}")


def rotate_full_state(ckpt_dir: Path, keep: int) -> None:
    """Delete mem_state.pt (the saved full internal state -- see
    save_checkpoint) from every checkpoint except the newest `keep`,
    globally across epochs, since resume only ever reads it from
    latest_checkpoint(). The rest of the checkpoint (trainable.pt,
    optimizer.pt, state.pt) is untouched -- those checkpoints stay fully
    resumable, just by restarting any mid-example slot from the beginning
    instead of a direct state load. keep <= 0 means: don't keep
    mem_state.pt anywhere."""
    ckpts = sorted(iter_checkpoints(ckpt_dir))
    stale = ckpts if keep <= 0 else ckpts[:-keep]
    for _, path in stale:
        p = path / "mem_state.pt"
        if p.exists():
            p.unlink()


def save_checkpoint(
    ckpt_dir: Path,
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
    """Saves every trainable parameter -- not just LoRA adapters, since a
    model like mamba2_2_7b_memory has an additional full-gradient subsystem
    that a LoRA-only save would silently drop. slot_states records each
    slot's (example_idx, pos) for resume; next_ptr is the next example to
    assign from the epoch's ordered list.

    The saved "last_ckpt_tokens" is this checkpoint's own total_tokens, not
    whatever the caller's cadence baseline was before this save -- it's the
    anchor a future resume needs for --ckpt-every-tokens to count from the
    point actually captured on disk. Saving the caller's pre-save baseline
    here instead was a real bug: resuming from a checkpoint would reset the
    cadence countdown back to the *previous* checkpoint's token count,
    making saves after a resume arrive earlier than --ckpt-every-tokens
    specifies (confirmed from two real checkpoints' recorded fields both
    showing the same stale baseline instead of advancing).

    If batched_state is given (only for models that carry state across
    chunks, and only when the caller wants this checkpoint to keep it --
    see rotate_full_state), it's saved as mem_state.pt so resume can load
    it directly for an exact continuation. Cheap to omit: a checkpoint
    without mem_state.pt is still fully resumable, just by restarting any
    mid-example slot from the beginning -- see main()'s resume path.

    dataset_fingerprint (see the module-level dataset_fingerprint() function)
    is saved alongside slot_states/next_ptr so a future resume can tell
    whether --data still points at the same dataset those indices were
    recorded against.

    memory_window (models that define set_memory_window only -- see
    run_training) is recorded here too, in the same small
    training-hyperparameters JSON as rank/alpha, purely for provenance --
    unlike rank/alpha it's not needed to reconstruct the model's shape, and
    the backend never reads it back (inference always runs with
    memory_window=1 regardless of what a checkpoint was trained with -- see
    the design spec). Omitted from the JSON entirely when None, so
    checkpoints for models without this concept are unaffected.

    group_idx/group_ptrs/group_tokens describe where a multi-dataset run is
    in its mix: which config group was training, how far each group's
    example order has been consumed, and how many tokens each has received
    (the deficit the segment scheduler resumes against -- without it a
    resumed run re-derives the mix from zero and over-serves whichever group
    happened to be behind at the start). Omitted entirely for single-dataset
    runs, whose next_ptr alone already says everything."""
    path = ckpt_dir / f"epoch-{epoch + 1}" / f"step-{step}"
    # Written to a sibling temp dir and published with one atomic rename, so a
    # checkpoint is never observed half-written -- a crash mid-save would
    # otherwise leave a step-N/ that latest_checkpoint() picks as newest and
    # --resume then fails on. The leading dot keeps the temp dir out of
    # iter_checkpoints()/rotate_checkpoints(), which match a "step-" prefix.
    tmp = path.with_name(f".{path.name}.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, tmp / "trainable.pt")
    lora_config = {"rank": lora_rank, "alpha": lora_alpha}
    if memory_window is not None:
        lora_config["memory_window"] = memory_window
    (tmp / "lora_config.json").write_text(json.dumps(lora_config))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    slot_states = [
        (s.example_idx, s.pos) if s is not None else None
        for s in slots
    ]
    state_dict = {
        "epoch": epoch,
        "slot_states": slot_states,
        "next_ptr": next_ptr,
        "dataset_fingerprint": dataset_fingerprint,
        "total_tokens": total_tokens,
        "last_ckpt_tokens": total_tokens,
    }
    if group_ptrs is not None:
        state_dict["group_idx"] = group_idx
        state_dict["group_ptrs"] = list(group_ptrs)
        state_dict["group_tokens"] = list(group_tokens or [])
    if batched_state is not None:
        state_dict["state_batch_size"] = len(slots)
        torch.save(batched_state, tmp / "mem_state.pt")
    torch.save(state_dict, tmp / "state.pt")
    # os.replace onto a non-empty dir fails, so clear any same-step save first
    # (only reachable when a resume re-saves a step that already exists).
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return path


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
    # A checkpoint from before a special token was registered carries fewer
    # marker_delta rows than the model. Missing rows stay at MarkerDelta's
    # zero init -- exactly "this token's delta is untrained". Shrinking is
    # not migrated: fewer model rows than checkpoint rows stays an error.
    key = "marker_delta.delta"
    current = dict(model.named_parameters()).get(key)
    if key in state and current is not None and state[key].shape[0] < current.shape[0]:
        rows = state[key].shape[0]
        print(f"padding {key} {rows} -> {current.shape[0]} rows; "
              f"rows past {rows} keep their zero (untrained) init")
        padded = torch.zeros_like(current.detach().cpu())
        padded[:rows] = state[key]
        state[key] = padded
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} trainable tensors from {path}")
    if result.unexpected_keys:
        # Partial loads are legitimate (e.g. a --freeze-lora checkpoint holds
        # only memory params), but unexpected keys usually mean the model was
        # constructed as a different variant than the checkpoint was trained
        # as (e.g. a checkpoint cross-loaded from the other integration-arm
        # model folder) -- make that loud.
        print(
            f"WARNING: {len(result.unexpected_keys)} checkpoint tensors have no home in this model "
            f"(first: {result.unexpected_keys[0]}) -- wrong model variant/integration arm?"
        )
    if loaded == 0:
        raise RuntimeError("load_checkpoint loaded 0 tensors -- checkpoint keys don't match model structure")
