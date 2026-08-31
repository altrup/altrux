"""Integration tests for mamba2_2_7b_memory's Model.forward around a TINY
synthetic Mamba2 backbone -- possible since Model derives its memory-subsystem
geometry (nheads/headdim/d_state/mem dims) from the backbone instead of
binding the real 2.7B dims. The mixer path still calls Triton-backed norm
kernels, so these need a working GPU (any -- the local ROCm box qualifies);
they skip on CPU-only hosts.

Pins the injection_enabled kill switch (sft/diagnostics/recall.py --ablation none):
with it off, forward must be invariant to the neural memory's content and
produce no memory activity at all; with it on (the default), memory content
must reach the logits.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import models  # noqa: F401  (installs the selective_scan_cuda stub -- see models/__init__.py)
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from models.mamba2_2_7b_memory.model import Model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mixer path needs Triton norm kernels (any working GPU)"
)

VOCAB, BATCH, SEQ = 96, 2, 4


def _tiny_model(seed: int = 0) -> Model:
    torch.manual_seed(seed)
    cfg = MambaConfig(
        d_model=64,
        n_layer=8,
        vocab_size=VOCAB,
        ssm_cfg={"layer": "Mamba2", "headdim": 16, "d_state": 16, "expand": 2, "ngroups": 1},
        rms_norm=True,
        fused_add_norm=False,
        tie_embeddings=True,
    )
    base = MambaLMHeadModel(cfg, device="cuda", dtype=torch.float32)
    return Model(base, read_layer=5, injected_layers=(3, 5, 7)).to("cuda")


def _ids(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")


def test_geometry_derived_from_backbone():
    m = _tiny_model()
    assert m.mem_dim == 64 and m.mem_hidden == 256
    inj = m.injections["3"]
    assert inj.nheads == m.layers[0].mixer.nheads
    assert inj.d_state == m.layers[0].mixer.d_state


def test_injection_fires_by_default():
    m = _tiny_model()
    assert m.injection_enabled is True
    logits, state = m(_ids())
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert m.pop_memory_stats() is not None
    assert m.last_token_log() is not None


def test_injection_disabled_is_invariant_to_memory_content():
    m = _tiny_model()
    m.injection_enabled = False
    ids = _ids()
    # Each call from state=None draws a DIFFERENT random neural-memory init;
    # with the pathway off the logits must not see that difference.
    logits_a, _ = m(ids)
    logits_b, _ = m(ids)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)
    # And no memory activity of any kind happened.
    assert m.pop_memory_stats() is None
    assert m.last_token_log() is None


def test_injection_enabled_is_sensitive_to_memory_content():
    m = _tiny_model()
    ids = _ids()
    # beta sits near sigmoid(0)+surprise term at init (no anneal set), so two
    # different random memory inits inject differently.
    logits_a, _ = m(ids)
    logits_b, _ = m(ids)
    assert not torch.allclose(logits_a, logits_b, atol=1e-6)


def test_windowed_forward_runs_with_flag_off_and_on():
    m = _tiny_model()
    m.set_memory_window(2)
    logits, _ = m(_ids())
    assert torch.isfinite(logits).all()
    m.injection_enabled = False
    logits, _ = m(_ids())
    assert torch.isfinite(logits).all()
