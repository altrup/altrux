"""Equivalence and dispatch tests for mamba2_780m's fused single-token decode
step (Model._mixer_step_fused: causal_conv1d_update + selective_state_update)
against the manual per-token step (Model._mixer_step), which is ground truth.

The equivalence tests need both kernels, so they skip wherever
`_decode_step_kernels()` is None (ROCm, or causal-conv1d absent). The dispatch
tests swap in sentinel kernels, so they run on any working GPU.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import models  # noqa: F401  (installs the selective_scan_cuda stub)
import models.mamba2_780m.model as M780

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mixer path needs Triton norm kernels (any working GPU)"
)

DEVICE = "cuda"
VOCAB, BATCH, PREFIX, SEQ = 96, 2, 4, 12

needs_kernels = pytest.mark.skipif(
    torch.cuda.is_available() and M780._decode_step_kernels() is None,
    reason=f"fused decode kernels unavailable on this platform "
    f"(hip={torch.version.hip}, cuda={torch.version.cuda}; needs CUDA + causal-conv1d)",
)


def _tiny_model(rmsnorm: bool = True) -> M780.Model:
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=64,
        n_layer=4,
        vocab_size=VOCAB,
        ssm_cfg={
            "layer": "Mamba2",
            "headdim": 16,
            "d_state": 16,
            "expand": 2,
            "ngroups": 1,
            "rmsnorm": rmsnorm,
        },
        rms_norm=True,
        fused_add_norm=False,
        tie_embeddings=True,
    )
    return M780.Model(MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.float32)).to(DEVICE)


def _ids(n: int, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, n), device=DEVICE)


def _branch(state: M780.MixerState) -> M780.MixerState:
    """Same tensors, fresh lists -- so a run from it shows any in-place
    mutation of the shared starting state."""
    return M780.MixerState(list(state.conv_states), list(state.ssm_states))


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a.float(), b.float(), atol=1e-4, rtol=1e-4)


@needs_kernels
@pytest.mark.parametrize("rmsnorm", [True, False])
def test_fused_decode_matches_manual_and_leaves_input_state_alone(rmsnorm):
    model = _tiny_model(rmsnorm)
    ids = _ids(PREFIX + SEQ)
    with torch.no_grad():
        _, start = model._forward_tokens(ids[:, :PREFIX], model._init_state(BATCH, torch.float32))
        snapshot = [t.clone() for t in start.flatten()]

        fused_logits, fused = model._forward_tokens(
            ids[:, PREFIX:], _branch(start), step=model._mixer_step_fused
        )
        for before, after in zip(snapshot, start.flatten()):
            assert torch.equal(before, after)
        manual_logits, manual = model._forward_tokens(ids[:, PREFIX:], _branch(start))

    assert _close(fused_logits, manual_logits)
    for f, m in zip(fused.flatten(), manual.flatten()):
        assert f.shape == m.shape
        assert _close(f, m)


@needs_kernels
def test_switching_between_fused_and_manual_mid_sequence_matches_manual():
    model = _tiny_model()
    ids = _ids(SEQ, seed=2)
    half = SEQ // 2
    with torch.no_grad():
        manual_logits, manual = model._forward_tokens(ids, model._init_state(BATCH, torch.float32))
        first, mixed = model._forward_tokens(
            ids[:, :half], model._init_state(BATCH, torch.float32), step=model._mixer_step_fused
        )
        second, mixed = model._forward_tokens(ids[:, half:], mixed)

    assert _close(torch.cat([first, second], dim=1), manual_logits)
    for f, m in zip(mixed.flatten(), manual.flatten()):
        assert _close(f, m)


class _KernelCalled(Exception):
    pass


def _boom(*args, **kwargs):
    raise _KernelCalled


@pytest.fixture
def sentinel_kernels(monkeypatch):
    monkeypatch.setattr(M780, "_decode_step_kernels", lambda: (_boom, _boom))


def test_inference_decode_dispatches_to_fused_step(sentinel_kernels):
    model = _tiny_model()
    with torch.no_grad(), pytest.raises(_KernelCalled):
        model(_ids(1))


def test_grad_enabled_decode_stays_manual(sentinel_kernels):
    model = _tiny_model()
    model(_ids(1))


def test_erase_hook_keeps_decode_manual(sentinel_kernels):
    model = _tiny_model()
    model.erase_hook = lambda layer_idx, ssm_state, C: ssm_state
    with torch.no_grad():
        model(_ids(1))


def test_c_capture_keeps_decode_manual(sentinel_kernels):
    model = _tiny_model()
    model.c_capture = []
    with torch.no_grad():
        model(_ids(1))
    assert len(model.c_capture) == len(model.layers)
