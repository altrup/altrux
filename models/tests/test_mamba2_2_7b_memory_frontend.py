"""Fast (seconds, no download, no GPU) regression tests for
_TitansFrontEnd's write-knob parameterization (see model.py's knob_proj
init and observe()).

The property this file pins down: the knobs (eta/theta/alpha) must start at
their chosen operating points and keep usable sigmoid gradients when fed a
residual stream at the raw scale the real backbone produces (rms in the
tens at READ_LAYER) -- both with the zero-init weights and once the weights
have grown to ordinary magnitudes. A parameterization that feeds the raw
residual straight into the sigmoid saturates every knob at a rail (theta
~0: memory never written; alpha at a random rail per token) with ~no
gradient to recover, which sft/diagnostics/knobs.py confirmed on a real
checkpoint.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.mamba2_2_7b_memory.model import ALPHA_CAP, _TitansFrontEnd

D_MODEL, MEM_DIM, MEM_HIDDEN = 256, 16, 32
RAW_RESIDUAL_RMS = 30.0


def _make_front_end() -> _TitansFrontEnd:
    torch.manual_seed(0)
    return _TitansFrontEnd(d_model=D_MODEL, mem_dim=MEM_DIM, mem_hidden=MEM_HIDDEN)


def _raw_scale_residual(batch: int = 8) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, D_MODEL) * RAW_RESIDUAL_RMS


def test_knobs_start_at_operating_points():
    fe = _make_front_end()
    _, _, _, eta, theta, alpha = fe.observe(_raw_scale_residual())
    assert torch.allclose(eta, torch.full_like(eta, 0.45), atol=1e-3)
    assert torch.allclose(theta, torch.full_like(theta, 0.05), atol=1e-3)
    # Retention regime: well under the sigmoid midpoint's 0.05/window.
    assert (alpha < 0.005).all()
    # Zero-init weight makes the starting knobs data-independent: every
    # token gets the same value regardless of residual content.
    assert torch.allclose(eta, eta[0].expand_as(eta))
    assert torch.allclose(alpha, alpha[0].expand_as(alpha))


def test_knobs_stay_off_rails_with_ordinary_weights():
    fe = _make_front_end()
    with torch.no_grad():
        fe.knob_proj.weight.normal_(0, 0.02)
    _, _, _, eta, theta, alpha = fe.observe(_raw_scale_residual())
    # With the raw residual fed straight to knob_proj, weight std 0.02 puts
    # pre-activations ~10 deep into sigmoid's rails at this scale; the
    # unit-rms input keeps them O(0.3), so every knob stays strictly
    # interior with real gradient.
    assert (eta > 0.9 * 0.02).all() and (eta < 0.9 * 0.98).all()
    assert (theta > 0.1 * 0.02).all() and (theta < 0.1 * 0.98).all()


def test_alpha_never_exceeds_cap():
    fe = _make_front_end()
    # Slam the alpha pre-activation as deep into sigmoid's upper rail as a
    # trained projection ever could -- the ceiling must hold regardless.
    with torch.no_grad():
        fe.knob_proj.weight.normal_(0, 5.0)
        fe.knob_proj.bias.fill_(50.0)
    _, _, _, _, _, alpha = fe.observe(_raw_scale_residual())
    assert (alpha <= ALPHA_CAP).all()
    assert ALPHA_CAP <= 1e-4  # episodic retention prior, see model.py


def test_gradient_reaches_every_knob():
    fe = _make_front_end()
    _, _, _, eta, theta, alpha = fe.observe(_raw_scale_residual())
    (eta.sum() + theta.sum() + alpha.sum()).backward()
    per_knob_bias_grad = fe.knob_proj.bias.grad.abs()
    assert (per_knob_bias_grad > 0).all()
    assert fe.knob_proj.weight.grad.abs().sum() > 0
