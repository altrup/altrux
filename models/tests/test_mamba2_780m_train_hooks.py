"""Fast (seconds, no download) regression test for mamba2_780m's training
path.

Covers the manual mixer step introduced after mamba_ssm's own fused kernels
(causal_conv1d's compiled extension, the Triton SSD chunk-scan kernel) were
found broken on this project's dev hardware -- see model.py's Model
docstring. The two things worth pinning down here: gradients actually reach
the LoRA adapters (a silently-disconnected forward wouldn't necessarily
crash), and chunked processing produces
identical results to a single non-chunked forward call (since that's the
whole point of threading MixerState across calls).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))

from lora import apply_lora
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import models.mamba2_780m.model as M780
import models.mamba2_780m.train_hooks as hooks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU (Mamba2's fused norm path uses a Triton kernel, no CPU fallback)")

DEVICE = "cuda"


def _build_tiny_model(marker_ids=None):
    torch.manual_seed(0)
    cfg = MambaConfig(d_model=64, n_layer=4, vocab_size=50, ssm_cfg=dict(layer="Mamba2", headdim=16, ngroups=1, d_state=16))
    mamba = MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.float32)
    if marker_ids is not None:
        mamba.marker_token_ids = marker_ids  # what load_base/extend_embeddings stamps
    mamba = apply_lora(mamba, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = M780.Model(mamba).to(DEVICE)
    # Same freeze/unfreeze selection as train_hooks.setup_training (which we
    # can't call here -- it downloads the real 780m).
    for p in model.parameters():
        p.requires_grad_(False)
    trainable_params = []
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "marker_delta" in name:
            p.requires_grad_(True)
            trainable_params.append(p)
    return model, trainable_params


def test_chunk_loss_weight_sum_counts_assistant_tokens_only():
    model, _ = _build_tiny_model()
    ids = torch.randint(0, 50, (1, 10), device=DEVICE)
    target_ids = torch.randint(0, 50, (1, 10), device=DEVICE)
    mask_slice = torch.zeros(10, dtype=torch.bool, device=DEVICE)
    mask_slice[3:] = True  # 7 assistant-turn positions

    loss_sum, weight_sum, state = hooks.chunk_loss(model, ids, target_ids, mask_slice, None, eos_weight=1.0)

    assert weight_sum == 7
    assert torch.isfinite(loss_sum)
    assert state is not None


def test_chunk_loss_upweights_eos_positions():
    model, _ = _build_tiny_model()
    ids = torch.randint(1, 50, (1, 10), device=DEVICE)
    target_ids = torch.randint(1, 50, (1, 10), device=DEVICE)
    target_ids[0, 5] = hooks.EOS_ID
    mask_slice = torch.ones(10, dtype=torch.bool, device=DEVICE)

    _, weight_sum, _ = hooks.chunk_loss(model, ids, target_ids, mask_slice, None, eos_weight=3.0)

    assert weight_sum == 9 + 3.0  # 9 ordinary positions weight 1, the EOS position weight 3


def test_chunk_loss_state_threads_across_calls_and_gradients_flow_to_lora():
    model, _ = _build_tiny_model()
    ids = torch.randint(0, 50, (37,), device=DEVICE)
    mask = torch.zeros(37, dtype=torch.bool, device=DEVICE)
    mask[20:] = True
    chunk_len = 10

    state = None
    total_weight = 0.0
    n_chunks = 0
    seqlen = ids.numel()
    for start in range(0, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1:end + 1].unsqueeze(0)
        mask_slice = mask[start + 1:end + 1]
        loss_sum, weight_sum, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight=2.0)
        if weight_sum > 0:
            (loss_sum / weight_sum).backward()
        state = state.detach()
        total_weight += weight_sum
        n_chunks += 1

    assert n_chunks == 4
    assert total_weight == 17  # the 17 assistant-turn tokens in `mask`

    # lora_B gets gradient on the very first step; lora_A's gradient is
    # exactly zero until lora_B (zero-initialized) moves off zero -- expected
    # LoRA-init behavior, not a wiring bug, so only check lora_B here.
    lora_b_grads = sum(1 for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None and p.grad.abs().max() > 0)
    total_lora_b = sum(1 for n, p in model.named_parameters() if "lora_B" in n)
    assert lora_b_grads == total_lora_b and lora_b_grads > 0


def test_chunking_matches_single_call():
    model, _ = _build_tiny_model()
    ids = torch.randint(0, 50, (1, 20), device=DEVICE)

    with torch.no_grad():
        logits_one, _ = model(ids)
        logits_chunk1, state = model(ids[:, :10])
        state = state.detach()
        logits_chunk2, _ = model(ids[:, 10:], state=state)
        logits_chunked = torch.cat([logits_chunk1, logits_chunk2], dim=1)

    diff = (logits_one - logits_chunked).abs().max().item()
    assert diff < 1e-3, f"chunking diverges from single-call forward: {diff}"


def test_marker_delta_wired_into_forward_and_receives_gradient():
    model, trainable_params = _build_tiny_model(marker_ids=[48, 49])
    assert model.marker_delta is not None
    # setup_training's name-based selection must catch the delta parameter.
    assert any(p is model.marker_delta.delta for p in trainable_params)

    ids = torch.tensor([[3, 48, 7, 49, 11]], device=DEVICE)
    with torch.no_grad():
        logits_zero, _ = model(ids)
        model.marker_delta.delta += 0.1
        logits_moved, _ = model(ids)
    assert not torch.equal(logits_zero, logits_moved), "delta does not reach the forward pass"

    with torch.no_grad():
        model.marker_delta.delta.zero_()
    target_ids = torch.tensor([[48, 7, 49, 11, 2]], device=DEVICE)
    mask_slice = torch.ones(5, dtype=torch.bool, device=DEVICE)
    loss_sum, weight_sum, _ = hooks.chunk_loss(model, ids, target_ids, mask_slice, None, eos_weight=1.0)
    (loss_sum / weight_sum).backward()
    grad = model.marker_delta.delta.grad
    assert grad is not None and grad.abs().sum() > 0
    assert model.embedding.weight.grad is None  # the frozen table stays frozen
