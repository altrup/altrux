"""Frozen-teacher and replay dream generation."""

from __future__ import annotations

import contextlib
import copy
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

from experiments.dream_generation import DREAM_MAX_TURNS, sample_next
from experiments.dream_types import Dream
from experiments.erasure.probe import group_by_layer
from progress import fmt_duration, heartbeat, ts

if TYPE_CHECKING:
    import torch

PRINT_EVERY = 16
CUE_DEFER_MAX = 20
CUE_STOPS = (".", "\n")
ERASE_OP = "deflated"


@contextlib.contextmanager
def frozen_teacher(model) -> Iterator[object]:
    """Run the enclosed forwards with LoRA and marker deltas bypassed."""
    from adapters.lora import LoRALinear

    adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
    scales = [m.scale for m in adapters]
    delta = getattr(model, "marker_delta", None)
    saved = delta.delta.detach().clone() if delta is not None else None
    for m in adapters:
        m.scale = 0.0
    if delta is not None:
        delta.delta.data.zero_()
    try:
        yield model
    finally:
        for m, scale in zip(adapters, scales, strict=True):
            m.scale = scale
        if delta is not None:
            delta.delta.data.copy_(saved)


def dream_seed_text(asst_open: str, prompt: str) -> str:
    """Return the assistant marker and the trained literal-space separator."""
    return f"{asst_open} {prompt}" if prompt else f"{asst_open} "


def rehearsal_fraction(
    token_texts: Sequence[str], needles: Sequence[str]
) -> tuple[float, dict[str, int]]:
    """Measure token spans that overlap an entity or code occurrence."""
    text = "".join(token_texts).lower()
    starts, pos = [], 0
    for piece in token_texts:
        starts.append((pos, pos + len(piece)))
        pos += len(piece)
    covered = [False] * len(token_texts)
    counts: dict[str, int] = {}
    for needle in needles:
        low = needle.lower()
        counts[needle] = 0
        at = text.find(low)
        while at != -1:
            counts[needle] += 1
            for i, (lo, hi) in enumerate(starts):
                if lo < at + len(low) and hi > at:
                    covered[i] = True
            at = text.find(low, at + 1)
    return (sum(covered) / len(token_texts) if token_texts else 0.0), counts


def teacher_dream(
    model,
    wake_state,
    seed_ids: torch.Tensor,
    n_tokens: int,
    temperature: float,
    drain: bool,
    decode_token,
    needles: Sequence[str],
    cues: Sequence[Sequence[int]] = (),
    cue_every: int = 0,
    cue_greedy: int = 0,
    frozen: bool = True,
    erase_op: str = ERASE_OP,
    stop_id: int | None = None,
    turn_id: int | None = None,
    max_turns: int = DREAM_MAX_TURNS,
    erase_state_fn: Callable[..., int] | None = None,
) -> Dream:
    """Generate one frozen-teacher dream, optionally draining each read."""
    import torch

    state = copy.deepcopy(wake_state)
    ids = [int(i) for i in seed_ids[0].tolist()]
    cue_flags = [False] * len(ids)
    next_cue, cue_at, greedy_left = 0, len(ids) + cue_every, 0
    deferred = -1
    logits_cache: list[torch.Tensor] = []
    queries: list[list[torch.Tensor]] = []
    texts: list[str] = []
    skipped = 0
    turns = 0
    stop_reason = "max-tokens"
    started = time.time()
    teacher_ctx = frozen_teacher(model) if frozen else contextlib.nullcontext()
    with teacher_ctx, torch.no_grad():
        for t in range(n_tokens):
            token = torch.tensor([[ids[t]]], dtype=torch.long, device=seed_ids.device)
            model.c_capture = []
            logits, state = model(token, state=state)
            per_layer = group_by_layer(model.c_capture, len(model.layers))[0]
            model.c_capture = None

            logits_cache.append(logits[0, -1].float().cpu())
            queries.append([c.cpu() for c in per_layer])
            texts.append(decode_token(ids[t]))
            if drain:
                if erase_state_fn is None:
                    raise RuntimeError("teacher_dream drain requires erase_state_fn")
                skipped += erase_state_fn(state, per_layer, op=erase_op)
            if t + 1 >= len(ids):
                if cues and cue_every and deferred < 0 and len(ids) >= cue_at:
                    deferred = 0
                at_boundary = any(stop in texts[-1] for stop in CUE_STOPS)
                if deferred >= 0 and (at_boundary or deferred >= CUE_DEFER_MAX):
                    cue = [int(i) for i in cues[next_cue % len(cues)]]
                    ids.extend(cue)
                    cue_flags.extend([True] * len(cue))
                    next_cue += 1
                    cue_at = len(ids) + cue_every
                    greedy_left = cue_greedy
                    deferred = -1
                else:
                    deferred += deferred >= 0
                    temp = 0.0 if greedy_left > 0 else temperature
                    greedy_left = max(0, greedy_left - 1)
                    sampled = int(sample_next(logits[:, -1], temp).item())
                    ids.append(sampled)
                    cue_flags.append(False)
                    turns += sampled == turn_id
                    if sampled == stop_id:
                        stop_reason = "eoc"
                    elif turn_id is not None and turns >= max_turns:
                        stop_reason = "turn-backstop"

            if (t + 1) % PRINT_EVERY == 0 or t + 1 == n_tokens or stop_reason != "max-tokens":
                frac, _ = rehearsal_fraction(texts, needles)
                rate = (t + 1) / (time.time() - started)
                heartbeat()
                print(
                    f"[{ts()}]  dream {t + 1}/{n_tokens} rehearsal {frac:.2f} {rate:.1f} tok/s "
                    f"ETA {fmt_duration((n_tokens - t - 1) / rate)} | "
                    f"{''.join(texts[-PRINT_EVERY:])!r}",
                    flush=True,
                )
            if stop_reason != "max-tokens":
                print(f"[{ts()}]  dream ended after {len(texts)} tokens: {stop_reason}", flush=True)
                break
    kept = len(logits_cache)
    return Dream(
        tokens=torch.tensor([ids[:kept]], dtype=torch.long, device=seed_ids.device),
        logits=torch.stack(logits_cache),
        queries=queries,
        final_state=state,
        token_texts=texts,
        skipped_cone=skipped,
        cue_flags=cue_flags[:kept],
        stop_reason=stop_reason,
        prefix_len=len(seed_ids[0]),
    )


__all__ = [
    "dream_seed_text",
    "frozen_teacher",
    "rehearsal_fraction",
    "teacher_dream",
]
