"""B4's mechanism: per-dream erase-then-replay (DISCUSSION-20260808 sec 2.7,
2.9.2, 2.10.3).

Pure tensor math over a state and a set of captured read queries. Sec 2.9.1's
prod-validity constraint is a property of this module rather than a promise
about it: nothing here is handed the injected facts, so no gate, weighting or
rank choice can consult them. The harness (dream_sleep.py) owns the capture,
the cache format and the binding-scan validation overlay.

The chain, per dream:

  1. the **state-dependency gate** -- re-score the dream's tokens under a blank
     state with the same weights and keep the positions where the with-state
     and blank-state next-token distributions diverge past a threshold. A
     memory read is a position where the state changed the prediction.
  2. one **SVD per layer** over those positions' raw queries -> an orthonormal
     basis, truncated by a rank rule inside the prod-side address budget.
  3. **one projection** of the dream's start state, S <- S(I - V^T V): never a
     sum of per-query cuts (which over-subtracts along the shared cone into a
     sign-flipped anti-memory) and never sigma-scaled (partial cuts compound as
     (1-g)^N across N per-dream re-applications; a projection is idempotent, so
     the damage is independent of N).

Bases are stored as ROWS: a basis is (r, n) with orthonormal rows, matching
`experiments.erasure.operators.state_top_dirs` and
`experiments.erasure.operators.deflate`. Sec 2.7's "V^T V = I" assertion is
`basis @ basis.T == I` here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from experiments.erasure.operators import (
    deflate,
    erase_state_subspace,
    erase_subspace,
    erase_subspace_scaled,
    sigma_gammas,
    state_top_dirs,
)
from experiments.erasure.gating import (
    ADDRESS_BUDGET_FRACTION,
    DEFLATE_K,
    MEDIAN_C,
    RANK_RULES,
    VARIANTS,
    address_budget,
    aggregate_basis,
    gated_positions,
    orthonormalize,
    rank_median,
    rank_ratio_gap,
    state_divergence,
    variant_basis,
)


__all__ = [
    "ADDRESS_BUDGET_FRACTION", "DEFLATE_K", "MEDIAN_C", "RANK_RULES", "VARIANTS",
    "address_budget", "aggregate_basis", "deflate", "erase_state_subspace",
    "erase_subspace", "erase_subspace_scaled", "gated_positions", "orthonormalize",
    "rank_median", "rank_ratio_gap", "sigma_gammas", "state_divergence",
    "state_top_dirs", "variant_basis",
]
