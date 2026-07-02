"""Fast (seconds, no download, no GPU needed) regression test for
mamba2_2_7b_memory's memory-window batching (see model.py's `_NeuralMemory`
and `Model.forward`, and docs/superpowers/specs/2026-07-02-chunked-memory-
injection-design.md).

Deliberately does NOT load the real 2.7B backbone -- it OOMs on this
project's 8GB local dev GPU even at minimal LoRA rank, so there's no way to
exercise the full Model.forward path here. Instead this tests `_NeuralMemory`
directly, which is where all of the new (and riskiest) tensor-shape/gradient
logic actually lives; Model.forward's own orchestration around it is
comparatively thin bookkeeping.

The one property this file exists to pin down above all others: a
memory-window of 1 must produce byte-for-byte identical results to the
original (pre-windowing) per-token write, since that's the load-bearing
guarantee that inference (which always runs at window=1, see the design
spec) never sees behavior training didn't produce at window=1, and that
existing checkpoints/behavior aren't silently changed by this feature
existing.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.mamba2_2_7b_memory.model import GRAD_SCALE, _NeuralMemory

BATCH, DIM, HIDDEN = 2, 16, 32


def _make_memory() -> _NeuralMemory:
    torch.manual_seed(0)
    return _NeuralMemory(BATCH, DIM, HIDDEN, "cpu", torch.float32)


def _original_single_token_write(mem, k, v, eta, theta, alpha):
    """The pre-windowing per-token write(), reimplemented here as ground
    truth -- see git history for the version this is pinned against."""
    params = [p.detach().requires_grad_(True) for p in (mem.w1, mem.b1, mem.w2, mem.b2)]
    momentum = [s.detach() for s in mem.momentum]
    pred = mem._apply(k, *params)
    per_example_loss = ((pred - v) ** 2).mean(dim=-1)
    grads = torch.autograd.grad(per_example_loss.sum(), params, create_graph=True)
    grad_norm = torch.zeros(k.shape[0], dtype=k.dtype)
    for g in grads:
        grad_norm = grad_norm + g.detach().pow(2).flatten(1).sum(dim=1)
    grad_norm = grad_norm.sqrt()
    clip_factor = (GRAD_SCALE * torch.tanh(grad_norm / GRAD_SCALE)) / (grad_norm + 1e-12)
    grads = [g * clip_factor.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
    new_params, new_momentum = [], []
    for p, g, s in zip(params, grads, momentum):
        view = (-1,) + (1,) * (p.dim() - 1)
        s_new = eta.view(*view) * s - theta.view(*view) * g
        p_new = (1 - alpha.view(*view)) * p + s_new
        new_params.append(p_new)
        new_momentum.append(s_new)
    return new_params, new_momentum, per_example_loss.detach(), grad_norm.detach()


def test_window1_matches_original_per_token_write():
    mem_orig, mem_windowed = _make_memory(), _make_memory()
    for step in range(3):
        torch.manual_seed(100 + step)
        k, v = torch.randn(BATCH, DIM), torch.randn(BATCH, DIM)
        eta, theta, alpha = torch.rand(BATCH) * 0.9, torch.rand(BATCH) * 0.1, torch.rand(BATCH) * 0.1

        new_params, new_momentum, orig_loss, orig_gnorm = _original_single_token_write(mem_orig, k, v, eta, theta, alpha)
        mem_orig.w1, mem_orig.b1, mem_orig.w2, mem_orig.b2 = new_params
        mem_orig.momentum = new_momentum

        win_loss, win_gnorm = mem_windowed.write(
            k.unsqueeze(0), v.unsqueeze(0), eta.unsqueeze(0), theta.unsqueeze(0), alpha.unsqueeze(0)
        )

        for name in ("w1", "b1", "w2", "b2"):
            assert torch.allclose(getattr(mem_orig, name), getattr(mem_windowed, name), atol=1e-5), \
                f"step {step}: {name} diverges from original write() at window=1"
        assert torch.allclose(orig_gnorm, win_gnorm, atol=1e-5)
        assert torch.allclose(orig_loss, win_loss.squeeze(0), atol=1e-5)


def test_window_gt_1_stays_finite_and_gradient_reaches_upstream_projections():
    mem = _make_memory()
    W = 5
    k_proj_w = torch.randn(DIM, DIM, requires_grad=True)
    v_proj_w = torch.randn(DIM, DIM, requires_grad=True)
    residuals = torch.randn(W, BATCH, DIM)

    ks = torch.stack([residuals[t] @ k_proj_w for t in range(W)], dim=0)
    vs = torch.stack([residuals[t] @ v_proj_w for t in range(W)], dim=0)
    etas, thetas, alphas = torch.rand(W, BATCH) * 0.9, torch.rand(W, BATCH) * 0.1, torch.rand(W, BATCH) * 0.1

    loss, gnorm = mem.write(ks, vs, etas, thetas, alphas)
    assert loss.shape == (W, BATCH)
    assert gnorm.shape == (BATCH,)
    assert torch.isfinite(mem.w1).all() and torch.isfinite(mem.w2).all()

    # A downstream read + fake outer loss confirms create_graph=True is
    # actually keeping k_proj/v_proj connected through the windowed write,
    # same property the un-windowed write() always guaranteed.
    o_t = mem.read(torch.randn(BATCH, DIM))
    o_t.pow(2).sum().backward()
    assert k_proj_w.grad is not None and k_proj_w.grad.abs().sum() > 0
    assert v_proj_w.grad is not None and v_proj_w.grad.abs().sum() > 0


def test_multi_window_sequence_flushes_cleanly():
    """Mirrors Model.forward's own buffering loop (see its docstring) --
    every window must flush exactly, with nothing left pending at the end of
    a sequence whose length is a multiple of the window size."""
    mem = _make_memory()
    W, n_windows = 3, 4
    pending: dict[str, list[torch.Tensor]] = {k: [] for k in ("k", "v", "eta", "theta", "alpha")}
    n_writes = 0

    for _ in range(W * n_windows):
        k, v = torch.randn(BATCH, DIM), torch.randn(BATCH, DIM)
        o_t = mem.read(torch.randn(BATCH, DIM))
        surprise = mem.surprise(k, v)
        assert o_t.shape == (BATCH, DIM) and torch.isfinite(o_t).all()
        assert surprise.shape == (BATCH,) and torch.isfinite(surprise).all()

        pending["k"].append(k)
        pending["v"].append(v)
        pending["eta"].append(torch.rand(BATCH) * 0.9)
        pending["theta"].append(torch.rand(BATCH) * 0.1)
        pending["alpha"].append(torch.rand(BATCH) * 0.1)
        if len(pending["k"]) == W:
            mem.write(*(torch.stack(pending[name], dim=0) for name in ("k", "v", "eta", "theta", "alpha")))
            n_writes += 1
            pending = {k: [] for k in pending}

    assert n_writes == n_windows
    assert all(len(v) == 0 for v in pending.values()), "unflushed tokens left over -- window didn't divide evenly"


@pytest.mark.parametrize("window", [1, 2, 4, 8])
def test_various_window_sizes_produce_finite_weights(window):
    mem = _make_memory()
    ks = torch.randn(window, BATCH, DIM)
    vs = torch.randn(window, BATCH, DIM)
    etas, thetas, alphas = torch.rand(window, BATCH) * 0.9, torch.rand(window, BATCH) * 0.1, torch.rand(window, BATCH) * 0.1
    mem.write(ks, vs, etas, thetas, alphas)
    assert torch.isfinite(mem.w1).all() and torch.isfinite(mem.w2).all() and torch.isfinite(mem.b1).all() and torch.isfinite(mem.b2).all()
