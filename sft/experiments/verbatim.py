"""Verbatim-copy metrics between a dream and the wake transcript."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def copy_fraction[Token: Hashable](
    dream_tokens: Sequence[Token],
    transcript_tokens: Sequence[Token],
    n: int = 12,
    cue_flags: Sequence[bool] | None = None,
) -> float:
    """Return the fraction in transcript runs of at least ``n`` tokens."""
    if cue_flags is not None:
        keep = [
            i for i, token in enumerate(dream_tokens) if not (i < len(cue_flags) and cue_flags[i])
        ]
        dream_tokens = [dream_tokens[i] for i in keep]
    if not dream_tokens or not transcript_tokens or n <= 0:
        return 0.0
    grams: set[tuple[str, ...]] = {
        tuple(transcript_tokens[i : i + n]) for i in range(len(transcript_tokens) - n + 1)
    }
    copied = [False] * len(dream_tokens)
    for i in range(len(dream_tokens) - n + 1):
        if tuple(dream_tokens[i : i + n]) in grams:
            for j in range(i, i + n):
                copied[j] = True
    return sum(copied) / len(copied)


def longest_verbatim_run[Token: Hashable](
    dream_tokens: Sequence[Token], transcript_tokens: Sequence[Token], n: int = 12
) -> int:
    """Return the longest contiguous transcript run in a dream."""
    if not dream_tokens or not transcript_tokens or n <= 0:
        return 0
    grams = {tuple(transcript_tokens[i : i + n]) for i in range(len(transcript_tokens) - n + 1)}
    best = current = 0
    covered = [False] * len(dream_tokens)
    for i in range(len(dream_tokens) - n + 1):
        if tuple(dream_tokens[i : i + n]) in grams:
            for j in range(i, i + n):
                covered[j] = True
    for flag in covered:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


__all__ = ["copy_fraction", "longest_verbatim_run"]
