"""Dream and dream-cache data types."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from experiments.facts import Fact

if TYPE_CHECKING:
    import torch


def token_sha(ids: Sequence[int]) -> str:
    """Return the SHA-256 identity of a token sequence."""
    return hashlib.sha256(",".join(str(int(index)) for index in ids).encode()).hexdigest()


@dataclass
class Dream:
    """A cached teacher pass and its state read queries."""

    tokens: torch.Tensor
    logits: torch.Tensor
    queries: list[list[torch.Tensor]]
    final_state: object
    token_texts: list[str]
    skipped_cone: int
    cue_flags: list[bool] = field(default_factory=list)
    stop_reason: str = "max-tokens"
    prefix_len: int = 0


@dataclass
class DreamCache:
    """The single teacher dream shared by every arm."""

    seed: int
    transcript_ids: list[int]
    dream_ids: list[int]
    wake_state: object
    teacher_logits: torch.Tensor
    queries: list[list[torch.Tensor]]
    token_texts: list[str]
    cue_flags: list[bool]
    distractors: dict[str, str]
    facts: list[tuple[str, str, str]]
    stop_reason: str = "max-tokens"
    dream_prompt: str = ""
    prefix_len: int = 0
    transcript_sha: str = ""
    dream_sha: str = ""
    generator: str = "base"

    def __post_init__(self) -> None:
        self.transcript_sha = self.transcript_sha or token_sha(self.transcript_ids)
        self.dream_sha = self.dream_sha or token_sha(self.dream_ids)

    @property
    def fact_list(self) -> list[Fact]:
        return [Fact(*fact) for fact in self.facts]

    @property
    def free_tokens(self) -> int:
        return sum(not flag for flag in self.cue_flags)


@dataclass
class CachedDream:
    """One dream in a multi-dream cache."""

    dream_ids: list[int]
    token_texts: list[str]
    teacher_logits: torch.Tensor
    cue_flags: list[bool]
    prefix_len: int
    stop_reason: str
    divergence: list[float]
    gate_positions: list[int]
    queries: list[list[torch.Tensor]]
    spectra: list[list[float]]
    ranks: dict[str, list[int]]
    bases: dict[str, list[torch.Tensor]]
    dream_sha: str = ""

    def __post_init__(self) -> None:
        self.dream_sha = self.dream_sha or token_sha(self.dream_ids)

    @property
    def free_tokens(self) -> int:
        return sum(not flag for flag in self.cue_flags)


def dream_set_sha(dreams: Sequence[CachedDream]) -> str:
    return hashlib.sha256("\n".join(dream.dream_sha for dream in dreams).encode()).hexdigest()


@dataclass
class DreamSetCache:
    """The shared N-dream cache used by the B4 family."""

    seed: int
    transcript_ids: list[int]
    wake_state: object
    dreams: list[CachedDream]
    distractors: dict[str, str]
    facts: list[tuple[str, str, str]]
    dream_seed_offset: int = 0
    gate_family: str = "hard"
    dream_prompt: str = ""
    gate_threshold: float = 1.0
    rank_rule: str = "ratio-gap"
    generator: str = "base"
    transcript_sha: str = ""
    set_sha: str = ""

    def __post_init__(self) -> None:
        self.transcript_sha = self.transcript_sha or token_sha(self.transcript_ids)
        self.set_sha = self.set_sha or dream_set_sha(self.dreams)

    @property
    def fact_list(self) -> list[Fact]:
        return [Fact(*fact) for fact in self.facts]


__all__ = [
    "CachedDream",
    "Dream",
    "DreamCache",
    "DreamSetCache",
    "dream_set_sha",
    "token_sha",
]
