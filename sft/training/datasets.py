"""Experiment: shared

Training dataset specifications, loading, and deterministic ordering."""

from dataclasses import dataclass
from pathlib import Path

import torch


def dataset_fingerprint(data_path: str, n_examples: int) -> dict:
    """Identifies which dataset a checkpoint's slot_states/next_ptr indices
    are relative to, so resume can detect a --data swap (see save_checkpoint
    and main()'s resume path) instead of silently reindexing into the wrong
    dataset's examples."""
    return {"path": str(Path(data_path).resolve()), "n_examples": n_examples}


@dataclass
class DataSpec:
    """One --data slice: where it lives, what share of the training token
    budget it should receive, and the training config it runs under.

    chunk_len/batch_size/grad_checkpoint are per-slice because the plan's
    cram/needle slices need a chunk long enough for recall loss to reach the
    writes that produced it (notes/discussion/DISCUSSION-20260724-next-run-plan.md 2.1)
    while chains/ballast keep the cheaper locked constants. shuffle=False
    consumes the artifact in the order it was written, which is how a
    generator-side curriculum (the cram slices' growing gap ceiling) is
    realized. lo/hi are this slice's half-open range in the concatenated
    example lists (see load_datasets)."""

    path: str
    share: float | None = None
    chunk_len: int | None = None
    batch_size: int = 1
    grad_checkpoint: bool = False
    shuffle: bool = True
    lo: int = 0
    hi: int = 0

    @property
    def group_key(self) -> tuple:
        return (self.chunk_len, self.batch_size, self.grad_checkpoint)


_SPEC_BOOLS = {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False, "on": True, "off": False}


def _spec_value(conv, key: str, value: str):
    try:
        return conv(value)
    except (KeyError, ValueError):
        raise ValueError(f"bad value for --data option {key}: {value!r}") from None


def parse_data_spec(text: str, default_chunk_len: int | None, default_batch_size: int) -> DataSpec:
    """Parses one --data argument: a path, optionally followed by
    comma-separated key=value overrides, e.g.

        data/train_cram.pt,share=35,chunk-len=512,batch-size=6,grad-checkpoint=1,shuffle=0

    A bare path keeps the run-wide --chunk-len/--batch-size, so the
    single-dataset form is unchanged."""
    path, *fields = text.split(",")
    spec = DataSpec(path=path.strip(), chunk_len=default_chunk_len, batch_size=default_batch_size)
    for field in fields:
        if not field.strip():
            continue
        key, sep, value = field.partition("=")
        key, value = key.strip(), value.strip()
        if not sep:
            raise ValueError(f"--data option {key!r} in {text!r} needs a value (key=value)")
        if key == "share":
            spec.share = _spec_value(float, key, value)
        elif key == "chunk-len":
            spec.chunk_len = _spec_value(int, key, value)
        elif key == "batch-size":
            spec.batch_size = _spec_value(int, key, value)
        elif key == "grad-checkpoint":
            spec.grad_checkpoint = _spec_value(lambda v: _SPEC_BOOLS[v.lower()], key, value)
        elif key == "shuffle":
            spec.shuffle = _spec_value(lambda v: _SPEC_BOOLS[v.lower()], key, value)
        else:
            raise ValueError(
                f"unknown --data option {key!r} in {text!r} "
                "(known: share, chunk-len, batch-size, grad-checkpoint, shuffle)"
            )
    return spec


def check_shares(specs: list[DataSpec]) -> None:
    """A share is only meaningful relative to the other slices' shares, so a
    partially-specified set has no defensible reading -- reject it rather
    than mixing explicit shares with token-count-derived ones."""
    given = [s.share is not None for s in specs]
    if any(given) and not all(given):
        missing = [s.path for s in specs if s.share is None]
        raise ValueError(f"--data share= must be given for every dataset or none; missing for {missing}")
    if any(s.share is not None and s.share <= 0 for s in specs):
        raise ValueError("--data share= must be positive")


def load_datasets(specs: list[DataSpec]):
    """Loads every slice into one flat set of example lists, recording each
    slice's index range on its spec. Slices that omit an optional field
    (masks/recall_masks/sleep_positions) contribute Nones, so the flat lists
    stay index-aligned with ids."""
    ids: list[torch.Tensor] = []
    masks: list = []
    recall: list = []
    sleeps: list = []
    for spec in specs:
        data = torch.load(spec.path, map_location="cpu", weights_only=False)
        d_ids = data["ids"]
        n = len(d_ids)
        spec.lo, spec.hi = len(ids), len(ids) + n
        ids.extend(d_ids)
        masks.extend(data.get("masks") or [None] * n)
        recall.extend(data.get("recall_masks") or [None] * n)
        sleeps.extend(data.get("sleep_positions") or [None] * n)
        print(f"  {spec.path}: {n} examples, {sum(int(t.numel()) for t in d_ids):,} tokens")
    return ids, masks, recall, sleeps


def group_specs(specs: list[DataSpec]) -> list[list[DataSpec]]:
    """Slices that share a training config can share a batch, so they're
    trained together as one group; slices whose config differs cannot and
    are interleaved at segment granularity instead (see run_training)."""
    groups: list[list[DataSpec]] = []
    keys: list = []
    for spec in specs:
        if spec.group_key in keys:
            groups[keys.index(spec.group_key)].append(spec)
        else:
            keys.append(spec.group_key)
            groups.append([spec])
    return groups


def resolve_share(spec: DataSpec, train_ids: list[torch.Tensor]) -> float:
    """A slice's share, defaulting to its own token count -- which makes an
    unshared multi-dataset run one evenly-interleaved pass over everything."""
    if spec.share is not None:
        return spec.share
    return max(1.0, float(sum(int(train_ids[i].numel()) for i in range(spec.lo, spec.hi))))


def pick_deficit(consumed: list[float], shares: list[float], available: list[int]) -> int:
    """The slice furthest behind its share, by tokens. Used at both mixing
    levels (examples within a group, segments across groups) so the realized
    token mix tracks the requested shares from the first tokens onward
    rather than only in aggregate."""
    return min(available, key=lambda i: consumed[i] / shares[i])


def build_order(specs: list[DataSpec], train_ids: list[torch.Tensor], epoch: int, keep) -> list[int]:
    """The example order one config group consumes this epoch: each member
    contributes its own examples in its own order (per-epoch shuffle unless
    shuffle=0), and members are interleaved by token share."""
    lists: list[list[int]] = []
    for i, spec in enumerate(specs):
        n = spec.hi - spec.lo
        if spec.shuffle:
            local = torch.randperm(n, generator=torch.Generator().manual_seed(epoch + 977 * i)).tolist()
        else:
            local = range(n)
        lists.append([spec.lo + j for j in local if keep(spec.lo + j)])
    if len(lists) == 1:
        return lists[0]

    shares = [resolve_share(s, train_ids) for s in specs]
    ptrs = [0] * len(lists)
    consumed = [0.0] * len(lists)
    order: list[int] = []
    while True:
        available = [i for i in range(len(lists)) if ptrs[i] < len(lists[i])]
        if not available:
            return order
        i = pick_deficit(consumed, shares, available)
        idx = lists[i][ptrs[i]]
        ptrs[i] += 1
        order.append(idx)
        consumed[i] += int(train_ids[idx].numel())


def recall_weight_at(step: int, start: float, end: float, ramp_steps: int, shape: str = "linear") -> float:
    """The recall-mask loss multiplier at a given optimizer step. Ramped
    rather than constant so the retention pressure doesn't peak while the
    beta anneal is opening -- that window is where the gradient decides
    whether the memory path is useful or gets suppressed
    (notes/discussion/DISCUSSION-20260724-next-run-plan.md 2.2)."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return end
    frac = step / ramp_steps
    if shape == "geometric":
        return start * (end / start) ** frac
    return start + (end - start) * frac


def datasets_fingerprint(specs: list[DataSpec]):
    """A single slice keeps the plain dict fingerprint older checkpoints
    carry; several produce one per slice."""
    if len(specs) == 1:
        return dataset_fingerprint(specs[0].path, specs[0].hi - specs[0].lo)
    return [dataset_fingerprint(s.path, s.hi - s.lo) for s in specs]
