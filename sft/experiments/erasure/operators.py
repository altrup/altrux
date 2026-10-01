"""Experiment: shared

Shared tensor and state-direction operations for erasure experiments."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def rank1_erase(ssm_state: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """S(I - g chat chat^T): attenuate the state's read along `c` by `gamma`,
    leave orthogonal reads untouched. A (near-)zero query is a no-op rather
    than a divide-by-zero. Computed in fp32, returned in the state's dtype.

    The direction is differentiable: gradient runs through the query path that
    aimed the cut, so the landscape encodes "the cut follows the query" rather
    than a fixed cut the reads can dodge around (DISCUSSION-20260807 sec 3.4).
    Only the protected subspace stays stop-gradiented, at its source in
    `state_top_dirs`."""
    import torch

    s32, c32 = ssm_state.float(), c.float()
    norm = c32.norm(dim=-1, keepdim=True)
    if norm.max().item() < 1e-8:
        return ssm_state
    chat = c32 / norm.clamp_min(1e-8)
    read = torch.einsum("bhpn,bn->bhp", s32, chat)
    return (s32 - gamma * torch.einsum("bhp,bn->bhpn", read, chat)).to(ssm_state.dtype)


def deflate(c: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove `c`'s components along the rows of `basis` (assumed orthonormal):
    (I - V^T V) c. Shapes: c (b, n), basis either (k, n) shared across the
    batch or (b, k, n), one basis per element."""
    import torch

    basis = basis.detach().float()
    if basis.dim() == 2:
        basis = basis.unsqueeze(0)
    basis = basis.expand(c.shape[0], -1, -1)
    coeffs = torch.einsum("bn,bkn->bk", c.float(), basis)
    return c - torch.einsum("bk,bkn->bn", coeffs, basis).to(c.dtype)


_TOP_DIRS_CACHE: "OrderedDict[tuple, tuple[weakref.ref, torch.Tensor]]" = OrderedDict()
# One entry per (state, k) in flight; a gate sweep touches n_layers of them.
_TOP_DIRS_CACHE_MAX = 512


def clear_state_top_dirs_cache() -> None:
    _TOP_DIRS_CACHE.clear()


def state_top_dirs(ssm_state: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right-singular directions of the state's address space -- the
    directions the stored keys share (the interference cone), computed from
    the state alone. Heads stacked so the result is per layer, one basis per
    batch element (the fused arms ablate a whole batch of token-positions at
    once). Returns (b, k, n).

    Memoised on the state tensor: a gate sweep re-derives these for the SAME
    wake state once per dream per scheme row, and at (h*p, n) = (5120, 128) per
    layer that dominated its runtime. The key carries `_version`, so an
    in-place edit (a sleep boundary moving the state) recomputes rather than
    serving the previous sleep's directions.
    """
    import torch

    key = (id(ssm_state), ssm_state._version, k)
    entry = _TOP_DIRS_CACHE.get(key)
    if entry is not None:
        ref, hit = entry
        # A dead referent means this id was recycled by a NEW tensor: the key
        # is not evidence of identity on its own.
        if ref() is ssm_state:
            _TOP_DIRS_CACHE.move_to_end(key)
            return hit
        del _TOP_DIRS_CACHE[key]

    m = (
        ssm_state.detach().float().reshape(ssm_state.shape[0], -1, ssm_state.shape[-1])
    )  # (b, h*p, n)
    _, _, vh = torch.linalg.svd(m, full_matrices=False)
    out = vh[:, :k]

    try:
        _TOP_DIRS_CACHE[key] = (weakref.ref(ssm_state), out)
    except TypeError:  # a tensor that cannot be weak-referenced is simply not cached
        return out
    if len(_TOP_DIRS_CACHE) > _TOP_DIRS_CACHE_MAX:
        _TOP_DIRS_CACHE.popitem(last=False)
    return out


def erase_subspace(ssm_state: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """S(I - V^T V) for orthonormal rows V -- the whole eraser, applied ONCE
    per dream to a fresh copy of the wake state."""
    import torch

    if basis.numel() == 0:
        return ssm_state
    s, v = ssm_state.float(), basis.float().to(ssm_state.device)
    coeffs = torch.einsum("bhpn,rn->bhpr", s, v)
    return (s - torch.einsum("bhpr,rn->bhpn", coeffs, v)).to(ssm_state.dtype)


def sigma_gammas(spectrum: Sequence[float], rank: int) -> list[float]:
    """Per-direction erase strengths from the layer's own spectrum, normalised
    to its top singular value: the strongest direction is removed fully, the
    rest in proportion. Scale-free, and no constant beyond the rank already
    chosen."""
    head = list(spectrum)[:rank]
    top = max(head) if head else 0.0
    if top <= 0:
        return [0.0] * len(head)
    return [float(s) / float(top) for s in head]


def erase_subspace_scaled(
    ssm_state: torch.Tensor, basis: torch.Tensor, gammas: Sequence[float]
) -> torch.Tensor:
    """S(I - V^T diag(gamma) V): each direction removed in proportion to its
    own gamma rather than all-or-nothing.

    DISCUSSION sec 5 rejected this on two grounds, both about REPEATED
    application -- partial cuts leave re-amplifiable residue, and (1-gamma)^N
    compounds across re-applications. Single-sleep applies the eraser once per
    dream, so the objections do not bite and the question is empirical.
    """
    import torch

    if basis.numel() == 0 or not len(gammas):
        return ssm_state
    s, v = ssm_state.float(), basis.float().to(ssm_state.device)
    g = torch.tensor(list(gammas)[: v.shape[0]], dtype=s.dtype, device=s.device)
    coeffs = torch.einsum("bhpn,rn->bhpr", s, v) * g
    return (s - torch.einsum("bhpr,rn->bhpn", coeffs, v)).to(ssm_state.dtype)


def erase_state_subspace(state, bases: Sequence[torch.Tensor]) -> None:
    """Apply each layer's basis to the corresponding state tensor."""
    for index, basis in enumerate(bases):
        state.ssm_states[index] = erase_subspace(state.ssm_states[index], basis)
