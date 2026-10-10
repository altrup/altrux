"""Mixer-state helpers and batched replay-dream generation."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from experiments.dream_types import CachedDream, Dream
from progress import fmt_duration, heartbeat, ts

if TYPE_CHECKING:
    import torch


DREAM_MAX_TURNS = 32


def sample_next(logits: torch.Tensor, temperature: float, generator=None) -> torch.Tensor:
    """Sample one token from a batch of next-token logits."""
    import torch

    last = logits.float().clone()
    if temperature <= 0:
        return last.argmax(dim=-1, keepdim=True)
    return torch.multinomial(
        torch.softmax(last / temperature, dim=-1), num_samples=1, generator=generator
    )


def copy_state(state):
    """Copy mixer state tensors while preserving their autograd history."""
    new = copy.copy(state)
    for attr in ("conv_states", "ssm_states"):
        if hasattr(state, attr):
            setattr(new, attr, [tensor.clone() for tensor in getattr(state, attr)])
    return new


def state_to(state, device):
    """Move mixer state tensors to the requested device."""
    for attr in ("conv_states", "ssm_states"):
        if hasattr(state, attr):
            setattr(state, attr, [tensor.to(device) for tensor in getattr(state, attr)])
    return state


def _repeat_state(state, batch_size: int):
    repeated = copy.copy(state)
    for attr in ("conv_states", "ssm_states"):
        if hasattr(state, attr):
            tensors = getattr(state, attr)
            if any(tensor.shape[0] != 1 for tensor in tensors):
                raise ValueError("dream batching requires a single-row wake state")
            setattr(
                repeated,
                attr,
                [tensor.repeat_interleave(batch_size, dim=0) for tensor in tensors],
            )
    return repeated


def _teacher_dream_batch(
    model,
    wake_state,
    seed_ids,
    seeds: Sequence[int],
    n_tokens: int,
    temperature: float,
    decode_token,
    stop_id: int | None,
    turn_id: int | None,
) -> list[Dream]:
    """Generate independent uncued dreams in one model batch."""
    import torch

    batch_size = len(seeds)
    state = _repeat_state(wake_state, batch_size)
    prefix = [int(token) for token in seed_ids[0].tolist()]
    ids = [list(prefix) for _ in seeds]
    texts: list[list[str]] = [[] for _ in seeds]
    logits: list[list[torch.Tensor]] = [[] for _ in seeds]
    done = [False] * batch_size
    turns = [0] * batch_size
    reasons = ["max-tokens"] * batch_size
    generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
    filler = prefix[-1]
    model.c_capture = None
    with torch.no_grad():
        for position in range(n_tokens):
            tokens = [
                row[position] if not finished and position < len(row) else filler
                for row, finished in zip(ids, done, strict=True)
            ]
            token = torch.tensor(tokens, dtype=torch.long, device=seed_ids.device).unsqueeze(1)
            batch_logits, state = model(token, state=state)
            for row in range(batch_size):
                if done[row]:
                    continue
                last = batch_logits[row, -1].float().cpu()
                logits[row].append(last)
                texts[row].append(decode_token(tokens[row]))
                if position + 1 < len(ids[row]):
                    continue
                if temperature <= 0:
                    sampled = int(last.argmax())
                else:
                    sampled = int(
                        torch.multinomial(
                            torch.softmax(last / temperature, dim=-1), 1, generator=generators[row]
                        )
                    )
                if sampled == stop_id:
                    done[row], reasons[row] = True, "eoc"
                else:
                    turns[row] += sampled == turn_id
                    if turn_id is not None and turns[row] >= DREAM_MAX_TURNS:
                        done[row], reasons[row] = True, "turn-backstop"
                    else:
                        ids[row].append(sampled)
            if all(done):
                break
    return [
        Dream(
            tokens=torch.tensor([row[: len(row_logits)]], dtype=torch.long, device=seed_ids.device),
            logits=torch.stack(row_logits),
            queries=[],
            final_state=None,
            token_texts=row_texts,
            skipped_cone=0,
            cue_flags=[False] * len(row_logits),
            stop_reason=reason,
            prefix_len=len(prefix),
        )
        for row, row_logits, row_texts, reason in zip(ids, logits, texts, reasons, strict=True)
    ]


def dream_generation_seed(seed: int, index: int, attempt: int, offset: int = 0) -> int:
    """Derive deterministic, non-overlapping dream-generation seeds."""
    return seed * 1000 + offset + index + 1_000_000 * attempt


def generate_replay_dreams(
    model,
    wake_state,
    seed_ids,
    *,
    count: int,
    batch_size: int,
    seed: int,
    n_tokens: int,
    temperature: float,
    decode_token,
    stop_id: int | None,
    turn_id: int | None,
) -> tuple[list[CachedDream], list[int]]:
    """Generate the fixed replay set before training, batched across dreams."""
    if count < 1 or batch_size < 1:
        raise ValueError("dream count and batch size must be positive")
    cached: list[CachedDream] = []
    topology: list[int] = []
    started = time.time()
    for start in range(0, count, batch_size):
        width = min(batch_size, count - start)
        topology.append(width)
        dreams = _teacher_dream_batch(
            model,
            wake_state,
            seed_ids,
            [dream_generation_seed(seed, index, 0) for index in range(start, start + width)],
            n_tokens,
            temperature,
            decode_token,
            stop_id,
            turn_id,
        )
        for dream in dreams:
            cached.append(
                CachedDream(
                    dream_ids=[int(token) for token in dream.tokens[0].tolist()],
                    token_texts=dream.token_texts,
                    teacher_logits=dream.logits,
                    cue_flags=dream.cue_flags,
                    prefix_len=dream.prefix_len,
                    stop_reason=dream.stop_reason,
                    divergence=[],
                    gate_positions=[],
                    queries=[],
                    spectra=[],
                    ranks={},
                    bases={},
                )
            )
        heartbeat()
        elapsed = time.time() - started
        print(
            f"[{ts()}] replay dreams {len(cached)}/{count}, batch {width}, "
            f"{len(cached) / elapsed:.2f} dream/s, ETA "
            f"{fmt_duration(elapsed / len(cached) * (count - len(cached)))}",
            flush=True,
        )
    return cached, topology


__all__ = [
    "DREAM_MAX_TURNS",
    "copy_state",
    "dream_generation_seed",
    "generate_replay_dreams",
    "sample_next",
    "state_to",
]
