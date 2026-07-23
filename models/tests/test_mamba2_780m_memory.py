"""Tests for the mamba2_780m_memory package and the shared "mix" integration
arm (models/mamba2_2_7b_memory/model.py's _TokenMixInjection + the mix hooks
in Model.forward). Mix-arm behavior is exercised on a TINY synthetic backbone
(same approach/skip condition as test_mamba2_2_7b_memory_integration.py);
package-level checks (env knob, scaled layer indices) need no model at all.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import models  # noqa: F401  (installs the selective_scan_cuda stub)
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import models.mamba2_780m_memory.model as m780
from models.mamba2_2_7b_memory.model import Model

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mixer path needs Triton norm kernels (any working GPU)"
)

VOCAB, BATCH, SEQ = 96, 2, 4


# --- package-level (no model construction) ---


def test_scaled_layer_constants():
    # Same fractional depths as the 2.7B constants (42/64, 22..62 step 2).
    assert m780.N_LAYER == 48
    assert m780.READ_LAYER == 32
    assert m780.INJECTED_LAYERS == tuple(range(16, 48, 2))
    assert m780.MIX_READ_LAYER == 16


def test_integration_mode_env_knob(monkeypatch):
    monkeypatch.delenv("MEMORY_INTEGRATION", raising=False)
    assert m780.integration_mode() == "state"
    monkeypatch.setenv("MEMORY_INTEGRATION", "mix")
    assert m780.integration_mode() == "mix"
    monkeypatch.setenv("MEMORY_INTEGRATION", "bogus")
    with pytest.raises(ValueError):
        m780.integration_mode()


def test_package_interface():
    import models.mamba2_780m_memory as pkg

    assert pkg.MODEL_ID == "state-spaces/mamba2-780m"
    assert pkg.SPECIAL_TOKENS == [pkg.USER_OPEN, pkg.ASST_OPEN]
    assert callable(pkg.load_base) and callable(pkg.load_inference) and callable(pkg.post_load)


# --- mix arm on a tiny synthetic backbone ---


def _tiny_mix_model(seed: int = 0) -> Model:
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
    return Model(base, read_layer=3, injected_layers=(3, 5, 7), integration="mix").to("cuda")


def _ids(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")


@needs_gpu
def test_mix_mode_has_no_gated_delta_injections():
    m = _tiny_mix_model()
    assert m.injected_layers == ()
    assert len(m.injections) == 0
    assert m.mix is not None


@needs_gpu
def test_mix_is_exact_noop_at_init():
    # o_proj is zero-init, so at init the mix term is exactly 0 and the
    # logits must equal the pathway-disabled forward bit-for-bit -- the
    # resume-compatibility property the zero-init exists for.
    m = _tiny_mix_model()
    ids = _ids()
    logits_on, _ = m(ids)
    m.injection_enabled = False
    logits_off, _ = m(ids)
    assert torch.equal(logits_on, logits_off)


@needs_gpu
def test_mix_reaches_logits_once_o_proj_is_nonzero():
    m = _tiny_mix_model()
    with torch.no_grad():
        m.mix.o_proj.weight.normal_(std=0.1)
    ids = _ids()
    # Different random M inits (fresh state each call) must now produce
    # different logits: memory content reaches the output through the mix.
    logits_a, _ = m(ids)
    logits_b, _ = m(ids)
    assert not torch.allclose(logits_a, logits_b, atol=1e-6)
    # And the kill switch still removes the pathway entirely.
    m.injection_enabled = False
    logits_c, _ = m(ids)
    logits_d, _ = m(ids)
    assert torch.allclose(logits_c, logits_d, atol=1e-6)


@needs_gpu
def test_mix_is_causal():
    m = _tiny_mix_model()
    with torch.no_grad():
        m.mix.o_proj.weight.normal_(std=0.1)
    ids_a = _ids()
    ids_b = ids_a.clone()
    ids_b[:, -1] = (ids_b[:, -1] + 1) % VOCAB
    torch.manual_seed(7)
    logits_a, _ = m(ids_a)
    torch.manual_seed(7)
    logits_b, _ = m(ids_b)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)


@needs_gpu
def test_mix_windowed_forward_and_stats():
    m = _tiny_mix_model()
    m.set_memory_window(2)
    logits, _ = m(_ids())
    assert torch.isfinite(logits).all()
    stats = m.pop_memory_stats()
    assert stats is not None and stats["beta"] > 0
    logs = m.last_token_log()
    assert logs is not None and "beta" in logs[0] and "ssm_norm" not in logs[0]


@needs_gpu
def test_set_beta_anneal_covers_the_mix_gate():
    m = _tiny_mix_model()
    m.set_beta_anneal(0)
    assert m.mix.beta_anneal_offset == pytest.approx(-3.0)
    m.set_beta_anneal(10_000)
    assert m.mix.beta_anneal_offset == 0.0
