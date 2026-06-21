"""Fast (seconds, no download) regression test for mamba2_780m's training
path.

Covers the manual mixer step introduced after mamba_ssm's own fused kernels
(causal_conv1d's compiled extension, the Triton SSD chunk-scan kernel) were
found broken on this project's dev hardware -- see model.py's Model
docstring. The two things worth pinning down here: gradients actually reach
the LoRA adapters (a silently-disconnected forward wouldn't necessarily
crash, per mamba2_2_7b_memory's history), and chunked processing produces
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


def _build_tiny_model():
    torch.manual_seed(0)
    cfg = MambaConfig(d_model=64, n_layer=4, vocab_size=50, ssm_cfg=dict(layer="Mamba2", headdim=16, ngroups=1, d_state=16))
    mamba = MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.float32)
    mamba = apply_lora(mamba, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = M780.Model(mamba).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    trainable_params = []
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            trainable_params.append(p)
    return model, trainable_params


def test_process_example_chunks_and_gradients_flow():
    model, trainable_params = _build_tiny_model()
    ids = torch.randint(0, 50, (37,), device=DEVICE)
    mask = torch.zeros(37, dtype=torch.bool, device=DEVICE)
    mask[20:] = True

    loss_sum, weight_sum, n_chunks = hooks.process_example(
        model, ids, mask, DEVICE, eos_weight=2.0, backward_scale=1.0, chunk_len=10
    )

    assert n_chunks == 4
    assert weight_sum == 17  # the 17 assistant-turn tokens in `mask`
    assert torch.isfinite(torch.tensor(loss_sum))

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


def test_preflight_passes():
    model, trainable_params = _build_tiny_model()
    ids = [torch.randint(0, 50, (37,), device=DEVICE)]
    mask = torch.zeros(37, dtype=torch.bool, device=DEVICE)
    mask[20:] = True

    hooks.preflight(model, trainable_params, ids, [mask], DEVICE, max_len=1000, eos_weight=1.0, chunk_len=10)
