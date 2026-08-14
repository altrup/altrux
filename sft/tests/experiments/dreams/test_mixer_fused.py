"""Oracle tests for mamba2_780m's fused SSD chunk-scan path
(Model._mixer_chunk / _forward_chunk) against the per-token loop
(Model._mixer_step / _forward_tokens), which is ground truth.

What must hold: identical logits from a fresh state, identical logits *and*
final MixerState (conv_state and ssm_state) when a sequence is split into two
state-threaded chunks, and identical gradients on the LoRA parameters.

Runs only on a real CUDA host -- the fused kernel hangs on this project's
ROCm dev box, which is why the per-token loop exists at all. Tiny synthetic
backbone (seconds, no download), same fixture style as
models/tests/test_grad_checkpoint.py, with a small mixer chunk_size so the
kernel's own internal chunk-to-chunk state passing is exercised too.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


import models  # noqa: F401  (installs the selective_scan_cuda stub)
from adapters.lora import apply_lora
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import models.mamba2_780m.model as M780

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip is None),
    reason="fused SSD chunk-scan needs a CUDA (non-ROCm) GPU",
)

DEVICE = "cuda"
VOCAB, BATCH = 96, 2
SPLIT, SEQ = 9, 21  # split mid-kernel-chunk, and neither part a chunk_size multiple


def _tiny_model(dtype: torch.dtype) -> M780.Model:
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=64,
        n_layer=4,
        vocab_size=VOCAB,
        ssm_cfg={"layer": "Mamba2", "headdim": 16, "d_state": 16, "expand": 2, "ngroups": 1, "chunk_size": 8},
        rms_norm=True,
        fused_add_norm=False,
        tie_embeddings=True,
    )
    base = MambaLMHeadModel(cfg, device=DEVICE, dtype=dtype)
    base.marker_token_ids = [VOCAB - 2, VOCAB - 1]
    base = apply_lora(base, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = M780.Model(base).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "marker_delta" in name:
            p.requires_grad_(True)
    # lora_B is zero-init, which would leave lora_A with exactly zero gradient
    # and make the gradient comparison vacuous.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.05)
    return model


def _ids(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)


def _state(model: M780.Model, dtype: torch.dtype) -> M780.MixerState:
    return model._init_state(BATCH, dtype)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_fused_logits_match_loop_from_fresh_state():
    model = _tiny_model(torch.float32)
    ids = _ids()
    with torch.no_grad():
        loop, _ = model._forward_tokens(ids, _state(model, torch.float32))
        fused, _ = model._forward_chunk(ids, _state(model, torch.float32))
    assert torch.allclose(loop, fused, atol=1e-4, rtol=1e-4)


def test_fused_threaded_chunks_match_loop_logits_and_state():
    model = _tiny_model(torch.float32)
    ids = _ids(seed=2)
    with torch.no_grad():
        loop_logits, loop_state = model._forward_tokens(ids, _state(model, torch.float32))
        first, fused_state = model._forward_chunk(ids[:, :SPLIT], _state(model, torch.float32))
        second, fused_state = model._forward_chunk(ids[:, SPLIT:], fused_state)
    fused_logits = torch.cat([first, second], dim=1)

    assert torch.allclose(loop_logits, fused_logits, atol=1e-4, rtol=1e-4)
    for loop_conv, fused_conv in zip(loop_state.conv_states, fused_state.conv_states):
        assert torch.allclose(loop_conv, fused_conv, atol=1e-4, rtol=1e-4)
    for loop_ssm, fused_ssm in zip(loop_state.ssm_states, fused_state.ssm_states):
        assert torch.allclose(loop_ssm.float(), fused_ssm.float(), atol=1e-4, rtol=1e-4)


def test_fused_lora_gradients_match_loop():
    model = _tiny_model(torch.float32)
    ids = _ids(seed=3)
    target = (ids + 1) % VOCAB
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert named

    def grads(forward_fn):
        logits, _ = forward_fn(ids, _state(model, torch.float32))
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        return torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)

    for (name, _), loop_grad, fused_grad in zip(named, grads(model._forward_tokens), grads(model._forward_chunk)):
        assert (loop_grad is None) == (fused_grad is None), name
        if loop_grad is not None:
            assert torch.allclose(loop_grad, fused_grad, atol=1e-4, rtol=1e-3), name


def test_fused_logits_track_loop_in_bf16():
    model = _tiny_model(torch.bfloat16)
    ids = _ids(seed=4)
    with torch.no_grad():
        loop, loop_state = model._forward_tokens(ids, _state(model, torch.bfloat16))
        fused, fused_state = model._forward_chunk(ids, _state(model, torch.bfloat16))
    assert _cosine(loop, fused) > 0.99
    for loop_ssm, fused_ssm in zip(loop_state.ssm_states, fused_state.ssm_states):
        assert _cosine(loop_ssm, fused_ssm) > 0.99


def test_single_token_forward_stays_on_the_loop(monkeypatch):
    model = _tiny_model(torch.float32)
    monkeypatch.setattr(model, "_forward_chunk", lambda *a, **k: pytest.fail("decode must use the per-token path"))
    with torch.no_grad():
        model(_ids()[:, :1], state=_state(model, torch.float32))
