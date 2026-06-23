"""Fast (seconds, no download) regression test for mamba2_2_7b_memory's
training path.

Built at a tiny synthetic size instead of the real 2.7B checkpoint -- this is
what catches a model whose forward silently fails to connect some part of
itself to the loss (exactly the class of bug found in this model's history:
an int/string key mismatch that made the memory merge never run at all, and
a wrong autograd flag that disconnected k_proj/v_proj from the outer loss --
neither crashed, both produced grad=None for whole branches that should have
been nonzero). Run this before trusting a real training run, not after.

Requires a real GPU (skipped otherwise): Mamba2's mixer uses a Triton kernel
internally (the fused RMSNorm path) that doesn't run on CPU tensors.
"""

import functools
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))

from lora import apply_lora
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import models.mamba2_2_7b_memory.model as M
import models.mamba2_2_7b_memory.train_hooks as hooks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU (Mamba2's fused norm path uses a Triton kernel, no CPU fallback)")

DEVICE = "cuda"
D_MODEL, N_LAYER, NHEADS, HEADDIM, D_STATE = 32, 6, 8, 8, 8  # nheads = (expand=2 * d_model) / headdim
READ_LAYER, INJECTED_LAYERS = 3, (2, 4)
CHUNK_LEN = 10


def _build_tiny_model(monkeypatch):
    monkeypatch.setattr(M, "causal_conv1d_update", None)  # force the manual (non-fused) conv path
    monkeypatch.setattr(M, "D_MODEL", D_MODEL)
    monkeypatch.setattr(M, "N_LAYER", N_LAYER)
    monkeypatch.setattr(M, "NHEADS", NHEADS)
    monkeypatch.setattr(M, "HEADDIM", HEADDIM)
    monkeypatch.setattr(M, "D_STATE", D_STATE)
    monkeypatch.setattr(M, "READ_LAYER", READ_LAYER)
    monkeypatch.setattr(M, "INJECTED_LAYERS", INJECTED_LAYERS)
    monkeypatch.setattr(M, "MEM_DIM", D_MODEL)
    monkeypatch.setattr(M, "MEM_HIDDEN", 4 * D_MODEL)
    # _TitansFrontEnd/_GatedDeltaInjection's constructor defaults bind to the
    # module globals above at *class-definition* time, not call time -- the
    # monkeypatches above don't retroactively change them, so the classes
    # Model.__init__ instantiates need to be patched directly too.
    monkeypatch.setattr(M, "_TitansFrontEnd", functools.partial(M._TitansFrontEnd, d_model=D_MODEL, mem_dim=D_MODEL, mem_hidden=4 * D_MODEL))
    monkeypatch.setattr(
        M, "_GatedDeltaInjection",
        functools.partial(M._GatedDeltaInjection, mem_dim=D_MODEL, r=8, nheads=NHEADS, headdim=HEADDIM, d_state=D_STATE),
    )

    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=D_MODEL, n_layer=N_LAYER, vocab_size=50,
        ssm_cfg=dict(layer="Mamba2", headdim=HEADDIM, ngroups=1, d_state=D_STATE),
    )
    mamba = MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.float32)
    mamba = apply_lora(mamba, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = M.Model(mamba).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return model, trainable_params


def _run_chunked(model, ids, eos_weight, chunk_len=CHUNK_LEN):
    """Drives hooks.chunk_loss across an example's chunks -- this loop lives
    in sft/train.py's generic training loop now; tests reproduce just enough
    of it to exercise chunk_loss's state-threading and gradient wiring."""
    state = None
    total_weight = 0.0
    n_chunks = 0
    seqlen = ids.numel()
    for start in range(0, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1:end + 1].unsqueeze(0)
        loss_sum, weight_sum, state = hooks.chunk_loss(model, input_ids, target_ids, None, state, eos_weight)
        (loss_sum / weight_sum).backward()
        state = state.detach()
        total_weight += weight_sum.item()
        n_chunks += 1
    return total_weight, n_chunks


def test_chunk_loss_chunks_correctly(monkeypatch):
    model, _ = _build_tiny_model(monkeypatch)
    ids = torch.randint(0, 50, (37,), device=DEVICE)  # seqlen-1=36, chunk_len=10 -> 4 chunks

    total_weight, n_chunks = _run_chunked(model, ids, eos_weight=2.0)

    assert n_chunks == 4
    assert total_weight > 0


def test_chunk_loss_ignores_mask_and_trains_on_every_token(monkeypatch):
    """Unlike mamba2_780m, this model trains on every token (mask is
    accepted for interface parity but ignored) -- weight_sum should count
    every position in the chunk, not just a masked subset."""
    model, _ = _build_tiny_model(monkeypatch)
    input_ids = torch.randint(0, 50, (1, 9), device=DEVICE)
    target_ids = torch.randint(0, 50, (1, 9), device=DEVICE)

    _, weight_sum, _ = hooks.chunk_loss(model, input_ids, target_ids, None, None, eos_weight=1.0)

    assert weight_sum == 9


def test_gradients_reach_both_lora_and_memory_subsystem(monkeypatch):
    """The actual regression test: this model's history includes two bugs
    (an int/string key mismatch, and a wrong autograd flag) that each
    silently zeroed out gradients to a whole branch of the model without
    ever raising an error. Both would be caught here."""
    model, _ = _build_tiny_model(monkeypatch)
    ids = torch.randint(0, 50, (37,), device=DEVICE)

    _run_chunked(model, ids, eos_weight=1.0)

    lora_grads = memory_grads = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None or p.grad.abs().max() == 0:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_grads += 1
        else:
            memory_grads += 1

    assert lora_grads > 0, "no LoRA params received gradients"
    assert memory_grads > 0, "no memory-subsystem (front_end/injections) params received gradients"


def test_checkpoint_round_trip(monkeypatch):
    model, _ = _build_tiny_model(monkeypatch)
    ids = torch.randint(0, 50, (37,), device=DEVICE)
    _run_chunked(model, ids, eos_weight=1.0)

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(state, path / "trainable.pt")

        model2, _ = _build_tiny_model(monkeypatch)  # fresh, differently-initialized
        loaded = torch.load(path / "trainable.pt", map_location="cpu", weights_only=True)
        result = model2.load_state_dict(loaded, strict=False)
        assert len(result.unexpected_keys) == 0

    after = dict(model2.named_parameters())
    for name, value in before.items():
        assert torch.allclose(after[name].cpu(), value.cpu()), f"{name} did not round-trip correctly"
