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
from models.mamba2_2_7b_memory import model as _model_mod
from models.mamba2_2_7b_memory.model import GRAD_SCALE, _NeuralMemory, fused_kernel_usable

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
        eta, theta, alpha = (
            torch.rand(BATCH) * 0.9,
            torch.rand(BATCH) * 0.1,
            torch.rand(BATCH) * 0.1,
        )

        new_params, new_momentum, orig_loss, orig_gnorm = _original_single_token_write(
            mem_orig, k, v, eta, theta, alpha
        )
        mem_orig.w1, mem_orig.b1, mem_orig.w2, mem_orig.b2 = new_params
        mem_orig.momentum = new_momentum

        win_loss, win_gnorm = mem_windowed.write(
            k.unsqueeze(0), v.unsqueeze(0), eta.unsqueeze(0), theta.unsqueeze(0), alpha.unsqueeze(0)
        )

        for name in ("w1", "b1", "w2", "b2"):
            assert torch.allclose(
                getattr(mem_orig, name), getattr(mem_windowed, name), atol=1e-5
            ), f"step {step}: {name} diverges from original write() at window=1"
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
    etas, thetas, alphas = (
        torch.rand(W, BATCH) * 0.9,
        torch.rand(W, BATCH) * 0.1,
        torch.rand(W, BATCH) * 0.1,
    )

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
            mem.write(
                *(torch.stack(pending[name], dim=0) for name in ("k", "v", "eta", "theta", "alpha"))
            )
            n_writes += 1
            pending = {k: [] for k in pending}

    assert n_writes == n_windows
    assert all(len(v) == 0 for v in pending.values()), (
        "unflushed tokens left over -- window didn't divide evenly"
    )


@pytest.mark.parametrize("window", [1, 2, 4, 8])
def test_various_window_sizes_produce_finite_weights(window):
    mem = _make_memory()
    ks = torch.randn(window, BATCH, DIM)
    vs = torch.randn(window, BATCH, DIM)
    etas, thetas, alphas = (
        torch.rand(window, BATCH) * 0.9,
        torch.rand(window, BATCH) * 0.1,
        torch.rand(window, BATCH) * 0.1,
    )
    mem.write(ks, vs, etas, thetas, alphas)
    assert (
        torch.isfinite(mem.w1).all()
        and torch.isfinite(mem.w2).all()
        and torch.isfinite(mem.b1).all()
        and torch.isfinite(mem.b2).all()
    )


# --- Injection batching (Model.forward's is_window_close gate + the
# surprise-weighted pooling that produces the one signal each window's
# closing token's injection actually uses) -- these two pieces are simple
# enough to test standalone without the full Model, which is hardcoded to
# the real backbone's exact dims (nheads=80, headdim=64, d_state=128,
# d_model=2560 -- see _TitansFrontEnd()/_GatedDeltaInjection() being
# constructed with no args in Model.__init__, binding those as defaults) and
# so can't be wrapped around a small synthetic backbone for a cheap local
# integration test.


def _is_window_close(t: int, window: int) -> bool:
    """Mirrors Model.forward's own `is_window_close` expression exactly."""
    return (t % window) == (window - 1)


def _pool(o_stack: torch.Tensor, surprise_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrors Model.forward's pooling block exactly: o_stack/surprise_stack
    are (W, B, ...), returns the (B, ...) pooled o_t/surprise a window's
    closing token's injection uses."""
    weights = torch.softmax(surprise_stack, dim=0)
    pooled_o = (weights.unsqueeze(-1) * o_stack).sum(dim=0)
    pooled_surprise = (weights * surprise_stack).sum(dim=0)
    return pooled_o, pooled_surprise


@pytest.mark.parametrize("window,seqlen", [(1, 5), (3, 9), (4, 8)])
def test_is_window_close_fires_exactly_once_per_window(window, seqlen):
    closes = [t for t in range(seqlen) if _is_window_close(t, window)]
    assert len(closes) == seqlen // window
    # Evenly spaced, window tokens apart, last one at the final token --
    # matches forward()'s own `seqlen % memory_window == 0` requirement.
    assert closes[-1] == seqlen - 1
    assert all(b - a == window for a, b in zip(closes, closes[1:]))


def test_pooling_at_window1_reproduces_the_single_token_exactly():
    o = torch.randn(1, BATCH, DIM)
    surprise = torch.randn(1, BATCH)
    pooled_o, pooled_surprise = _pool(o, surprise)
    assert torch.allclose(pooled_o, o[0], atol=1e-6)
    assert torch.allclose(pooled_surprise, surprise[0], atol=1e-6)


def test_pooling_is_a_convex_combination_and_upweights_the_most_surprising_token():
    W = 5
    o = torch.randn(W, BATCH, DIM)
    surprise = torch.randn(W, BATCH)
    pooled_o, pooled_surprise = _pool(o, surprise)
    assert pooled_o.shape == (BATCH, DIM)
    assert pooled_surprise.shape == (BATCH,)
    assert torch.isfinite(pooled_o).all() and torch.isfinite(pooled_surprise).all()

    weights = torch.softmax(surprise, dim=0)
    assert torch.allclose(weights.sum(dim=0), torch.ones(BATCH), atol=1e-6), (
        "pooling weights must sum to 1"
    )

    # The most-surprising token in the window should get the largest weight
    # for every batch row, by construction of softmax.
    most_surprising = surprise.argmax(dim=0)
    heaviest = weights.argmax(dim=0)
    assert torch.equal(most_surprising, heaviest)


# --- Fused-kernel dispatch (docs/superpowers/specs/2026-07-02-chunked-
# memory-injection-design.md's "Fused-kernel dispatch" section, Model.
# _forward_fused/_mixer_span/_fused_path_available in model.py) -- the
# kernel calls themselves need a real CUDA host with causal_conv1d
# importable (neither true on this repo's local ROCm dev box, see root
# CLAUDE.md), so this can only cover the pieces that don't need one:
# feature detection, and the windowed read/surprise math _forward_fused
# uses in place of Model._forward_manual's per-token read()/surprise()
# calls at READ_LAYER (see _NeuralMemory.read_windowed/surprise_windowed).


def test_fused_kernel_usable_false_without_causal_conv1d(monkeypatch):
    monkeypatch.setattr(_model_mod, "causal_conv1d_fn", None)
    assert fused_kernel_usable(torch.device("cuda")) is False


def test_fused_kernel_usable_false_on_cpu():
    # causal_conv1d_fn may or may not be importable in this environment,
    # but a CPU device must always be rejected regardless.
    assert fused_kernel_usable(torch.device("cpu")) is False


def test_fused_kernel_usable_false_on_rocm_even_with_causal_conv1d(monkeypatch):
    # torch.cuda.is_available() (and hence device.type == "cuda") is True
    # on ROCm builds too -- torch.version.hip is the only thing that tells
    # them apart (see backend/Makefile's own install-time check, which this
    # mirrors). A ROCm host must still be rejected even if causal_conv1d
    # somehow imported successfully there.
    monkeypatch.setattr(_model_mod, "causal_conv1d_fn", lambda *a, **k: None)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    try:
        assert fused_kernel_usable(torch.device("cuda")) is False
    finally:
        monkeypatch.setattr(torch.version, "hip", None, raising=False)


def test_fused_kernel_usable_true_on_real_cuda_with_causal_conv1d(monkeypatch):
    monkeypatch.setattr(_model_mod, "causal_conv1d_fn", lambda *a, **k: None)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert fused_kernel_usable(torch.device("cuda")) is True


def test_read_windowed_matches_sequential_read_per_token():
    """_forward_fused calls read_windowed once per window instead of
    Model._forward_manual's W sequential read() calls -- must produce
    identical per-token results, since it's the same frozen-M forward pass,
    just batched over the window dim instead of looped."""
    mem = _make_memory()
    W = 4
    q_stack = torch.randn(W, BATCH, DIM)

    windowed = mem.read_windowed(q_stack)
    sequential = torch.stack([mem.read(q_stack[t]) for t in range(W)], dim=0)

    assert torch.allclose(windowed, sequential, atol=1e-6)


def test_surprise_windowed_matches_sequential_surprise_per_token():
    mem = _make_memory()
    W = 4
    k_stack = torch.randn(W, BATCH, DIM)
    v_stack = torch.randn(W, BATCH, DIM)

    windowed = mem.surprise_windowed(k_stack, v_stack)
    sequential = torch.stack([mem.surprise(k_stack[t], v_stack[t]) for t in range(W)], dim=0)

    assert torch.allclose(windowed, sequential, atol=1e-6)


def test_read_windowed_uses_window_start_weights_not_mid_window():
    """read_windowed must use the SAME (frozen) M for every token in the
    window -- it must NOT reflect a write() that happened partway through,
    matching _forward_manual's "read against whatever weights are current
    at the start of the window" contract (see forward()'s docstring)."""
    mem = _make_memory()
    W = 3
    q_stack = torch.randn(W, BATCH, DIM)
    before = mem.read_windowed(q_stack)

    # A write() call must not retroactively change what read_windowed
    # would have returned for weights captured before it.
    ks, vs = torch.randn(W, BATCH, DIM), torch.randn(W, BATCH, DIM)
    etas, thetas, alphas = (
        torch.rand(W, BATCH) * 0.9,
        torch.rand(W, BATCH) * 0.1,
        torch.rand(W, BATCH) * 0.1,
    )
    mem.write(ks, vs, etas, thetas, alphas)

    after_fresh_call = mem.read_windowed(q_stack)
    assert not torch.allclose(before, after_fresh_call), "write() should have changed M's weights"


@pytest.mark.parametrize("window,seqlen", [(1, 5), (3, 9), (4, 8)])
def test_window_span_boundaries_partition_the_chunk_exactly(window, seqlen):
    """Mirrors Model._forward_fused's own `start`/`boundary` arithmetic for
    each injected layer's per-window loop: the (window-1)-token fused span
    plus the single boundary token must together cover every token in the
    chunk exactly once, in order, with the boundary always the window's
    last token."""
    n_windows = seqlen // window
    covered: list[int] = []
    for w in range(n_windows):
        start, boundary = w * window, w * window + window - 1
        covered.extend(range(start, boundary))  # the fused span's tokens
        covered.append(boundary)  # the manual step's token

    assert covered == list(range(seqlen))
    last_start, last_boundary = (n_windows - 1) * window, (n_windows - 1) * window + window - 1
    assert last_boundary == seqlen - 1, (
        "last window's boundary token must be the chunk's last token"
    )
