"""The gate pilot (DISCUSSION-20260808 sec 2.10.6, 2.10.7): capture everything
once, then score every gating scheme offline.

`dream_sleep.py --build-dream-cache --dreams N --pilot-capture` writes the
capture this module reads. What is stored, and why:

  * **every position's per-layer read query**, fp16 -- not just the gated ones,
    since the point of the pilot is that a threshold sweep needs the positions
    a given threshold would have dropped.
  * **D_t at every position**, float -- the state-dependency divergence
    (`b4.state_divergence`), computed on the box from the with-state and
    blank-state logits.
  * the wake state, the facts, the dream texts, and the battery items' read
    queries (a forward pass, so it happens on the box).

Deliberately NOT stored: the (T, V) with-state and blank-state logit pairs
themselves. At the run's shape they are ~100 MB per dream against ~6 MB of
queries, and every scheme sec 2.10.7 names -- hard@any-tau, divergence-weighted
with a floor, weighted-capped -- is a function of D_t, which is stored in full.
The accepted limit: trying a DIFFERENT divergence measure needs a fresh capture
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class PilotDream:
    """One dream, unfiltered: the gate has not run on any of this."""

    dream_sha: str
    token_texts: list[str]
    divergence: list[float]
    queries: list[list[torch.Tensor]]  # T x n_layers, fp16
    cue_flags: list[bool]
    prefix_len: int
    stop_reason: str


@dataclass
class PilotCapture:
    seed: int
    facts: list[tuple[str, str, str]]  # entity, category, code
    wake_state: object
    dreams: list[PilotDream]
    # Battery prompt -> that prompt's per-position, per-layer read queries, run
    # from a copy of the wake state: sec 2.10.6's collateral pool.
    battery_queries: dict[str, list[list[torch.Tensor]]] = field(default_factory=dict)
    gate_threshold: float = 0.0
    rank_rule: str = ""
    set_sha: str = ""
