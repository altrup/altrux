"""Replay distillation of a cached dream set into the model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

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


LossWeights = Literal["turn", "uniform"]


def turn_weights(
    cached: CachedDream,
    *,
    user_id: int,
    asst_id: int,
    eos_id: int,
    eoc_id: int,
    loss_weights: LossWeights = "turn",
) -> dict[int, float]:
    """Map each scored position to its loss weight; the weights of a scored dream sum to 1.

    Position ``p`` predicts token ``p + 1`` and belongs to that token's turn. In the
    generated part a role marker opens a turn and belongs to it, EOS belongs to the
    turn it closes, and a token after EOS that is not a marker opens an unmarked
    turn. Tokens before the first marker form turn 0. A turn with no token other
    than its marker and EOS gets no weight; each of the T other turns gets 1/T,
    split evenly over its scored positions. The position predicting the terminal
    EOC is never scored. ``loss_weights="uniform"`` keeps the same positions and
    gives each of the dream's N scored positions 1/N.
    """
    if loss_weights not in ("turn", "uniform"):
        raise ValueError(f"loss_weights must be 'turn' or 'uniform', got {loss_weights!r}")
    ids, start = cached.dream_ids, cached.prefix_len
    end = len(ids)
    appended = end > start and ids[-1] == eoc_id
    end -= appended
    # The generator stops on EOC without appending it, so its last row predicts the EOC.
    end -= appended or cached.stop_reason == "eoc"
    markers = (user_id, asst_id)
    turn, closed = 0, False
    turn_of: dict[int, int] = {}
    content: set[int] = set()
    for index in range(start, len(ids)):
        token = ids[index]
        if token in markers or closed:
            turn, closed = turn + 1, False
        turn_of[index] = turn
        if token == eos_id:
            closed = True
        elif token not in markers:
            content.add(turn)
    keep = scored_keep(cached.cue_flags, start)
    groups: dict[int, list[int]] = {}
    for position in range(end):
        owner = turn_of.get(position + 1, turn)
        if keep[position] and owner in content:
            groups.setdefault(owner, []).append(position)
    if loss_weights == "uniform":
        n = sum(map(len, groups.values()))
        return {position: 1 / n for group in groups.values() for position in group}
    return {
        position: 1 / (len(groups) * len(group)) for group in groups.values() for position in group
    }


def dream_batch_gradient(
    model,
    dreams: Sequence[CachedDream],
    kl_temp: float,
    *,
    user_id: int,
    asst_id: int,
    eos_id: int,
    eoc_id: int,
    loss_weights: LossWeights = "turn",
) -> dict[str, object]:
    """Backpropagate the batch loss of the dreams into ``.grad`` in one batched forward; no step.

    The student replays each dream from an empty state. A dream's loss is its
    weighted KL (its weights sum to 1), None when it has no scored position;
    the batch loss is their mean over scored dreams. Called on one dream, the
    gradient is that dream's own. Gradients accumulate into ``.grad``.
    """
    import torch
    import torch.nn.functional as F

    weights = [
        turn_weights(
            d,
            user_id=user_id,
            asst_id=asst_id,
            eos_id=eos_id,
            eoc_id=eoc_id,
            loss_weights=loss_weights,
        )
        for d in dreams
    ]
    scored = sum(bool(w) for w in weights)
    out: dict[str, object] = {
        "loss": 0.0,
        "token_gradients": sum(map(len, weights)),
        "scored_dreams": scored,
        "scored_tokens": [len(w) for w in weights],
        "dream_losses": [None] * len(dreams),
        "max_dream_loss": None,
    }
    if not scored:
        return out
    device = next(model.parameters()).device
    length = max(len(d.dream_ids) for d in dreams)
    tokens = torch.tensor(
        [d.dream_ids + [d.dream_ids[-1]] * (length - len(d.dream_ids)) for d in dreams],
        dtype=torch.long,
        device=device,
    )
    logits, _ = model(tokens, state=None)
    rows = [row for row, w in enumerate(weights) for _ in w]
    cols = [position for w in weights for position in w]
    teacher = torch.cat(
        [d.teacher_logits[list(w)] for d, w in zip(dreams, weights, strict=True) if w]
    )
    weight = torch.tensor([v for w in weights for v in w.values()], device=device)
    per_position = F.kl_div(
        F.log_softmax(logits[rows, cols].float() / kl_temp, dim=-1),
        F.log_softmax(teacher.to(device).float() / kl_temp, dim=-1),
        log_target=True,
        reduction="none",
    ).sum(-1)
    per_dream = torch.zeros(len(dreams), device=device).index_add(
        0, torch.tensor(rows, device=device), weight * per_position * kl_temp**2
    )
    loss = per_dream.sum() / scored
    loss.backward()
    dream_losses = [
        value if w else None for value, w in zip(per_dream.tolist(), weights, strict=True)
    ]
    out["loss"] = loss.item()
    out["dream_losses"] = dream_losses
    out["max_dream_loss"] = max(value for value in dream_losses if value is not None)
    return out


def flat_grad(model):
    """The trainable parameters' ``.grad`` as one vector, zeros where a gradient is None."""
    import torch

    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for p in model.parameters()
            if p.requires_grad
        ]
    )


def distill_dream_batch(
    model,
    opt,
    dreams: Sequence[CachedDream],
    kl_temp: float,
    *,
    user_id: int,
    asst_id: int,
    eos_id: int,
    eoc_id: int,
    loss_weights: LossWeights = "turn",
    clip_norm: float | None = None,
) -> dict[str, object]:
    """Distill every dream in one batched forward and one optimizer step.

    ``dream_batch_gradient`` gives the gradient; ``clip_norm`` (None for no clip)
    caps its norm before ``opt.step()``. ``out`` adds the pre-clip ``grad_norm``
    and whether the clip fired. A batch with no scored dream takes no step.
    """
    import torch

    out = dream_batch_gradient(
        model,
        dreams,
        kl_temp,
        user_id=user_id,
        asst_id=asst_id,
        eos_id=eos_id,
        eoc_id=eoc_id,
        loss_weights=loss_weights,
    )
    out["grad_norm"], out["clipped"] = None, False
    if not out["scored_dreams"]:
        return out
    grads = [p for p in model.parameters() if p.grad is not None]
    if clip_norm is None:
        norm = torch.nn.utils.get_total_norm([p.grad for p in grads])
    else:
        norm = torch.nn.utils.clip_grad_norm_(grads, clip_norm)
    out["grad_norm"] = norm.item()
    out["clipped"] = clip_norm is not None and out["grad_norm"] > clip_norm
    opt.step()
    opt.zero_grad(set_to_none=True)
    return out


__all__ = [
    "LossWeights",
    "distill_dream_batch",
    "distill_dream_set",
    "distill_replay",
    "dream_batch_gradient",
    "dream_from_cached",
    "flat_grad",
    "scored_keep",
    "target_keep_mask",
    "turn_weights",
]
