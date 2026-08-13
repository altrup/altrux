"""Model.erase_hook: the per-layer interleaved erase the corrected B arms run
inside the forward (DISCUSSION-20260806 sec 3's micro-order).

What must hold: the hook sees each layer's own post-conv read query C before
that layer's decay+write, its return value is what the write lands on, and an
identity hook leaves the forward's result unchanged.

Tiny synthetic backbone (seconds, no download), same skip condition as the
other model tests -- mamba_ssm's norm path needs a working GPU.
"""

import sys
from pathlib import Path

import pytest
import torch


import models  # noqa: F401  (installs the selective_scan_cuda stub)
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import models.mamba2_780m.model as M780

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mixer path needs Triton norm kernels (any working GPU)"
)

DEVICE = "cuda"
VOCAB, SEQ = 96, 5


def _tiny_model() -> M780.Model:
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=64,
        n_layer=3,
        vocab_size=VOCAB,
        ssm_cfg={"layer": "Mamba2", "headdim": 16, "d_state": 16, "expand": 2, "ngroups": 1, "chunk_size": 8},
        rms_norm=True,
        fused_add_norm=False,
        tie_embeddings=True,
    )
    return M780.Model(MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.float32)).to(DEVICE)


def _ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (1, SEQ), device=DEVICE)


def _run(model: M780.Model, ids: torch.Tensor):
    with torch.no_grad():
        return model(ids, state=model._init_state(1, torch.float32))


def test_an_identity_hook_is_a_no_op():
    """Setting any hook forces the per-token path, so on a fused-kernel host
    this compares the loop against the chunk-scan -- same tolerance as
    test_mixer_fused.py's oracle, not bitwise equality."""
    model, ids = _tiny_model(), _ids()
    logits, state = _run(model, ids)

    model.erase_hook = lambda i, ssm, c: ssm
    hooked_logits, hooked_state = _run(model, ids)

    assert torch.allclose(logits, hooked_logits, atol=1e-4, rtol=1e-4)
    for a, b in zip(state.ssm_states, hooked_state.ssm_states, strict=True):
        assert torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4)


def test_hook_fires_once_per_layer_per_token_with_that_layer_s_read_query():
    model, ids = _tiny_model(), _ids()
    hooked: list[torch.Tensor] = []
    model.erase_hook = lambda i, ssm, c: (hooked.append(c), ssm)[1]
    model.c_capture = []
    _run(model, ids)
    captured = model.c_capture
    model.c_capture = None

    assert len(hooked) == SEQ * len(model.layers) == len(captured)
    for got, want in zip(hooked, captured, strict=True):
        assert torch.equal(got.detach(), want)


def test_hook_edits_the_state_the_write_lands_on():
    model, ids = _tiny_model(), _ids()
    plain_logits, _ = _run(model, ids)

    model.erase_hook = lambda i, ssm, c: torch.zeros_like(ssm)
    zeroed_logits, zeroed_state = _run(model, ids)

    assert not torch.equal(plain_logits, zeroed_logits)
    # Only the past is ablated: this token's own write still lands.
    assert zeroed_state.ssm_states[0].abs().max().item() > 0


def test_a_set_hook_keeps_a_multi_token_forward_on_the_per_token_path(monkeypatch):
    """The hook is per-token by construction (like c_capture); the fused
    chunk-scan has no place to interleave it."""
    model, ids = _tiny_model(), _ids()
    monkeypatch.setattr(model, "_forward_chunk", lambda *a, **k: pytest.fail("erase hook must use the per-token path"))
    model.erase_hook = lambda i, ssm, c: ssm
    _run(model, ids)
