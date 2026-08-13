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

import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from experiments.erasure.operators import (
    deflate,
    erase_subspace,
    erase_subspace_scaled,
    sigma_gammas,
    state_top_dirs,
)

# The prod-side rank cap (sec 2.7): an eraser may spend at most 1/N of the
# state's address dimensions per sleep. A resource constraint statable without
# any fact knowledge -- NEVER the injected fact count.
ADDRESS_BUDGET_FRACTION = 16
# sigma > c * median(sigma); the median sits in the address space's noise tail.
MEDIAN_C = 4.0
RANK_RULES = ("ratio-gap", "median")
VARIANTS = ("raw", "deflated", "qcm")
# Deflation depth for the `deflated` variant -- the operator that won the
# 08-08 picker, ported to the aggregate.
DEFLATE_K = 1


def state_divergence(with_state: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
    """Per-position D_t between the with-state and blank-state next-token
    distributions: KL(with-state || blank-state), in nats, over (T, V) logits.

    The gate's only input (sec 2.9.2). The measure is swappable in place --
    everything downstream reads a per-position score and a threshold.
    """
    import torch

    p = torch.log_softmax(with_state.float(), dim=-1)
    q = torch.log_softmax(blank.float(), dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


def gated_positions(divergence: torch.Tensor, threshold: float, prefix_len: int = 0,
                    exclude: Sequence[bool] | None = None) -> list[int]:
    """The positions whose queries a dream's eraser is built from.

    The steer prefix influences the dream through state only (sec 2.10.8) and
    spliced cue text is not a read the dream performed, so neither can enter a
    basis.
    """
    skip = exclude or ()
    return [t for t, d in enumerate(divergence.tolist())
            if t >= prefix_len and not (t < len(skip) and skip[t]) and d >= threshold]


def address_budget(d_state: int, fraction: int = ADDRESS_BUDGET_FRACTION) -> int:
    return max(1, d_state // fraction)


def rank_ratio_gap(sigma: torch.Tensor, budget: int) -> int:
    """Rank rule 1 (sec 2.7): cut at the largest ratio gap sigma_i/sigma_i+1."""
    s = [float(x) for x in sigma]
    n = min(budget, len(s) - 1)
    if n < 1:
        return min(1, len(s))
    ratios = [s[i] / max(s[i + 1], 1e-12) for i in range(n)]
    return int(max(range(n), key=ratios.__getitem__)) + 1


def rank_median(sigma: torch.Tensor, budget: int, c: float = MEDIAN_C) -> int:
    """Rank rule 2 (sec 2.7): keep sigma > c * median(sigma). Scale-invariant
    and sigma_1-robust, unlike the hard-coded thresholds sec 5 rejects."""
    s = [float(x) for x in sigma]
    if not s:
        return 0
    threshold = c * statistics.median(s)
    return max(1, min(sum(1 for x in s if x > threshold), budget, len(s)))


def aggregate_basis(queries: Sequence[torch.Tensor],
                    weights: Sequence[float] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One SVD over one dream's gated queries for one layer. Returns
    (V, sigma): V's rows are the right-singular directions, most energetic
    first, and sigma is that layer's spectrum (printed into the sidecar).

    `weights` scales each query's row before the SVD, which is how the pilot's
    divergence-weighted schemes (sec 2.10.7) enter: a position's influence on
    the basis is its weight. Prod's frozen scheme passes none (all ones).
    """
    import torch

    m = torch.stack([q.reshape(-1).float() for q in queries])
    if weights is not None:
        m = m * torch.tensor([float(w) for w in weights]).unsqueeze(1)
    _, sigma, vh = torch.linalg.svd(m, full_matrices=False)
    return vh, sigma


def orthonormalize(rows: torch.Tensor) -> torch.Tensor:
    """Re-orthonormalize row vectors, dropping directions the deflation
    collapsed -- a rank-deficient "basis" would make the projection inexact."""
    import torch

    if rows.shape[0] == 0:
        return rows.reshape(0, rows.shape[-1])
    q, r = torch.linalg.qr(rows.float().T)
    diag = r.diagonal().abs()
    return q.T[diag > 1e-6 * diag.max().clamp_min(1e-12)]


def variant_basis(v_full: torch.Tensor, rank: int, variant: str,
                  ssm_state: torch.Tensor | None = None) -> torch.Tensor:
    """One of sec 2.7's three post-processings of the SHARED per-layer SVD:

      raw       -- V as-is.
      deflated  -- each direction deflated against the state's own top singular
                   direction, then re-orthonormalized.
      qcm       -- v_1 dropped, the query-consensus direction (08-07 sec 6).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown B4 variant {variant!r}; expected one of {VARIANTS}")
    if variant == "qcm":
        return orthonormalize(v_full[1 : rank + 1])
    rows = v_full[:rank]
    if variant == "deflated":
        if ssm_state is None:
            raise ValueError("the deflated variant needs the state it will be applied to")
        # Cached queries live on CPU while the wake state lives on the GPU.
        rows = deflate(rows, state_top_dirs(ssm_state, DEFLATE_K)[0].to(rows.device))
    return orthonormalize(rows)


def erase_state_subspace(state, bases: Sequence[torch.Tensor]) -> None:
    """Apply each layer's own basis to `state` in place."""
    for i, basis in enumerate(bases):
        state.ssm_states[i] = erase_subspace(state.ssm_states[i], basis)
