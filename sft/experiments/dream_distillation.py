"""Replay distillation of a cached dream set into the model."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from experiments.dream_generation import copy_state
from experiments.dream_types import CachedDream, Dream, DreamCache
from experiments.inference import kl_loss, replay_step


def dream_from_cached(cached: CachedDream | DreamCache, device) -> Dream:
    import torch

    return Dream(
        tokens=torch.tensor([cached.dream_ids], dtype=torch.long, device=device),
        logits=cached.teacher_logits,
        queries=cached.queries,
        final_state=None,
        token_texts=cached.token_texts,
        skipped_cone=0,
        cue_flags=cached.cue_flags,
        stop_reason=cached.stop_reason,
        prefix_len=cached.prefix_len,
    )


def target_keep_mask(cue_flags: Sequence[bool]) -> list[bool]:
    return [
        index + 1 >= len(cue_flags) or not cue_flags[index + 1] for index in range(len(cue_flags))
    ]


def scored_keep(cue_flags: Sequence[bool], prefix_len: int) -> list[bool]:
    masked = [cue or index < prefix_len for index, cue in enumerate(cue_flags)]
    return target_keep_mask(masked)


def distill_replay(
    model,
    opt,
    dream: Dream,
    steps: int,
    chunk_len: int,
    kl_temp: float,
    on_step,
    fresh_state: bool = False,
    keep: Sequence[bool] | None = None,
    ce: bool = False,
    init_state=None,
) -> int:
    import torch.nn.functional as F

    length = dream.tokens.shape[1]
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
    tokens = 0
    for step in range(steps):
        chunk, reset = replay_step(step, n_chunks, fresh_state)
        if reset:
            state = None if init_state is None else copy_state(init_state)
        lo, hi = chunk * chunk_len, min((chunk + 1) * chunk_len, length)
        scored = [
            t for t in range(lo, hi) if (keep is None or keep[t]) and not (ce and t + 1 >= length)
        ]
        if not scored:
            on_step(step, 0.0)
            continue
        logits, state = model(dream.tokens[:, lo:hi], state=state)
        state = state.detach()
        sel = [t - lo for t in scored]
        if ce:
            loss = F.cross_entropy(logits[0, sel].float(), dream.tokens[0, [t + 1 for t in scored]])
        else:
            loss = kl_loss(
                dream.logits[scored].unsqueeze(0).to(logits.device), logits[:, sel], kl_temp
            )
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += len(scored)
        on_step(step, loss.item())
    return tokens


def distill_dream_set(
    model,
    opt,
    dreams: Sequence[CachedDream],
    wake_state,
    epochs: int,
    kl_temp: float,
    on_step,
    on_boundary,
    start: Callable[[CachedDream], object] | None = None,
) -> int:
    """Distill every dream once per epoch; ``start`` picks a per-dream initial state."""
    tokens = step = 0
    for epoch in range(epochs):
        for i, cached in enumerate(dreams):
            dream = dream_from_cached(cached, wake_state.ssm_states[0].device)
            tokens += distill_replay(
                model,
                opt,
                dream,
                1,
                dream.tokens.shape[1],
                kl_temp,
                lambda s, loss, base=step: on_step(base + s, loss),
                keep=scored_keep(cached.cue_flags, cached.prefix_len),
                init_state=None if start is None else start(cached),
            )
            step += 1
            on_boundary(i, epoch, step)
    return tokens


__all__ = [
    "distill_dream_set",
    "distill_replay",
    "dream_from_cached",
    "scored_keep",
    "target_keep_mask",
]
