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


def test_process_example_chunks_correctly(monkeypatch):
    model, _ = _build_tiny_model(monkeypatch)
    ids = torch.randint(0, 50, (37,), device=DEVICE)  # seqlen-1=36, chunk_len=10 -> 4 chunks

    loss_sum, weight_sum, n_chunks = hooks.process_example(
        model, ids, None, DEVICE, eos_weight=2.0, backward_scale=1.0, chunk_len=CHUNK_LEN
    )

    assert n_chunks == 4
    assert weight_sum > 0
    assert torch.isfinite(torch.tensor(loss_sum))


def test_gradients_reach_both_lora_and_memory_subsystem(monkeypatch):
    """The actual regression test: this model's history includes two bugs
    (an int/string key mismatch, and a wrong autograd flag) that each
    silently zeroed out gradients to a whole branch of the model without
    ever raising an error. Both would be caught here."""
    model, _ = _build_tiny_model(monkeypatch)
    ids = torch.randint(0, 50, (37,), device=DEVICE)

    hooks.process_example(model, ids, None, DEVICE, eos_weight=1.0, backward_scale=1.0, chunk_len=CHUNK_LEN)

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


def test_preflight_passes(monkeypatch):
    model, trainable_params = _build_tiny_model(monkeypatch)
    ids = [torch.randint(0, 50, (37,), device=DEVICE)]

    hooks.preflight(model, trainable_params, ids, [None], DEVICE, max_len=1000, eos_weight=1.0, chunk_len=CHUNK_LEN)


def test_checkpoint_round_trip(monkeypatch):
    model, _ = _build_tiny_model(monkeypatch)
    ids = torch.randint(0, 50, (37,), device=DEVICE)
    hooks.process_example(model, ids, None, DEVICE, eos_weight=1.0, backward_scale=1.0, chunk_len=CHUNK_LEN)

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
