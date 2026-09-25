"""Equivalence tests for block-wise gradient checkpointing (Model.forward's
`grad_checkpoint_block` path, driven by train_hooks.set_grad_checkpoint).

The whole point of checkpointing here is that it must be invisible: same
loss, same gradient on every trainable parameter, including the ones whose
gradient only exists because the outer loss differentiates through
`_NeuralMemory.write`'s own gradient step. That path has to be rebuilt
correctly when the checkpointed block is recomputed during backward, which
is the one thing worth actually proving rather than assuming (see
test_write_gradient_path_survives_recompute).

Tiny synthetic backbones only -- seconds, no download; same skip condition
as the other model tests (mamba_ssm's norm path needs a working GPU).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))

import models  # noqa: F401  (installs the selective_scan_cuda stub)
from adapters.lora import apply_lora
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import models.mamba2_780m.model as M780
import models.mamba2_780m.train_hooks as hooks780
import models.mamba2_780m_memory_mix.train_hooks as hooks_mix
from models.mamba2_2_7b_memory.model import Model as MemoryModel
from models.mamba2_2_7b_memory.model import _NeuralMemory

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mixer path needs Triton norm kernels (any working GPU)"
)

DEVICE = "cuda"
VOCAB, BATCH, SEQ = 96, 2, 8
BLOCK = 4  # two blocks over SEQ, two memory-windows per block


def _tiny_cfg() -> MambaConfig:
    return MambaConfig(
        d_model=64,
        n_layer=8,
        vocab_size=VOCAB,
        ssm_cfg={"layer": "Mamba2", "headdim": 16, "d_state": 16, "expand": 2, "ngroups": 1},
        rms_norm=True,
        fused_add_norm=False,
        tie_embeddings=True,
    )


def _tiny_mix_model() -> MemoryModel:
    torch.manual_seed(0)
    base = MambaLMHeadModel(_tiny_cfg(), device=DEVICE, dtype=torch.float32)
    base.marker_token_ids = [VOCAB - 2, VOCAB - 1]
    base = apply_lora(base, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = MemoryModel(base, read_layer=3, integration="mix").to(DEVICE)
    model.set_memory_window(2)
    with torch.no_grad():
        # o_proj is zero-init, which would leave the whole memory subsystem
        # with exactly zero gradient and make this test vacuous.
        model.mix.o_proj.weight.normal_(std=0.1)
        model.mix.o_proj.bias.normal_(std=0.1)
    return model


def _tiny_plain_model() -> M780.Model:
    torch.manual_seed(0)
    base = MambaLMHeadModel(_tiny_cfg(), device=DEVICE, dtype=torch.float32)
    base.marker_token_ids = [VOCAB - 2, VOCAB - 1]
    base = apply_lora(base, ["in_proj", "out_proj"], rank=4, alpha=8.0, dropout=0.0)
    model = M780.Model(base).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "marker_delta" in name:
            p.requires_grad_(True)
    # lora_B is zero-init; move it off zero so lora_A gets gradient too.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.05)
    return model


def _ids(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)


def _named_trainables(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _loss_and_grads(model, ids, state0, names_params):
    logits, _ = model(ids, state=state0)
    target = (ids + 1) % VOCAB
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    grads = torch.autograd.grad(loss, [p for _, p in names_params], allow_unused=True)
    return loss.detach(), {
        n: (g.detach() if g is not None else None) for (n, _), g in zip(names_params, grads)
    }


def _fresh_state(model, proto):
    """A structurally fresh MemoryState/MixerState off the same initial
    tensors, so two runs start byte-identical (forward rebinds rather than
    mutates, so sharing storage is safe)."""
    return proto.detach()


# --- memory model (the run's model: mamba2_780m_memory_mix) ---


def test_grad_checkpoint_matches_uncheckpointed_memory_model():
    model = _tiny_mix_model()
    ids = _ids()
    names_params = _named_trainables(model)
    torch.manual_seed(3)
    proto = model._init_state(BATCH, DEVICE, torch.float32)

    loss_ref, grads_ref = _loss_and_grads(model, ids, _fresh_state(model, proto), names_params)
    hooks_mix.set_grad_checkpoint(model, True, block=BLOCK)
    loss_ckpt, grads_ckpt = _loss_and_grads(model, ids, _fresh_state(model, proto), names_params)

    torch.testing.assert_close(loss_ckpt, loss_ref, rtol=1e-4, atol=1e-6)
    nonzero = 0
    for name, _ in names_params:
        gr, gc = grads_ref[name], grads_ckpt[name]
        assert (gr is None) == (gc is None), name
        if gr is None:
            continue
        torch.testing.assert_close(gc, gr, rtol=2e-4, atol=1e-5, msg=lambda m, n=name: f"{n}: {m}")
        nonzero += int(gr.abs().max() > 0)
    assert nonzero > 0


def test_memory_subsystem_actually_receives_gradient():
    """Guards the equivalence test above from being vacuous: the front-end
    projections that only exist to feed _NeuralMemory must have real,
    non-zero gradients in the setup those tests use."""
    model = _tiny_mix_model()
    names_params = _named_trainables(model)
    torch.manual_seed(3)
    proto = model._init_state(BATCH, DEVICE, torch.float32)
    _, grads = _loss_and_grads(model, _ids(), _fresh_state(model, proto), names_params)
    for name in (
        "front_end.k_proj.weight",
        "front_end.v_proj.weight",
        "front_end.knob_proj.weight",
    ):
        g = grads[name]
        assert g is not None and g.abs().max() > 0, name


def test_analytic_write_gradient_matches_autograd():
    """_write_grads replaced a nested autograd.grad (which forced a full
    checkpoint-frame recompute per window). It has to be the same function,
    including when something differentiates through the gradient itself --
    that second differentiation is exactly how k_proj/v_proj get their
    signal from the write."""
    torch.manual_seed(0)
    W, B, D, H = 4, 2, 32, 128
    ks = torch.randn(W, B, D, device=DEVICE, requires_grad=True)
    vs = torch.randn(W, B, D, device=DEVICE)
    params = [
        torch.randn(B, H, D, device=DEVICE),
        torch.randn(B, H, device=DEVICE),
        torch.randn(B, D, H, device=DEVICE),
        torch.randn(B, D, device=DEVICE),
    ]
    params = [p.requires_grad_(True) for p in params]

    _, loss_analytic, grads_analytic = _NeuralMemory._write_grads(ks, vs, *params)
    pred = _NeuralMemory._apply_windowed(ks, *params)
    loss_autograd = ((pred - vs) ** 2).mean(dim=-1)
    grads_autograd = torch.autograd.grad(loss_autograd.sum(), params, create_graph=True)

    torch.testing.assert_close(loss_analytic, loss_autograd)
    for i, (ga, gr) in enumerate(zip(grads_analytic, grads_autograd)):
        torch.testing.assert_close(
            ga, gr, rtol=1e-5, atol=1e-5, msg=lambda m, i=i: f"grad {i}: {m}"
        )

    probe_a = torch.autograd.grad(sum((g**2).sum() for g in grads_analytic), ks)[0]
    probe_r = torch.autograd.grad(sum((g**2).sum() for g in grads_autograd), ks)[0]
    torch.testing.assert_close(probe_a, probe_r, rtol=1e-4, atol=1e-4)


def test_write_gradient_path_survives_recompute(monkeypatch):
    """k_proj/v_proj's gradient flows through the `grads` that
    `_NeuralMemory.write` computes, i.e. through a differentiation of the
    write's own gradient step. Dropping create_graph must change those
    gradients -- if it doesn't, the equivalence test above proves nothing
    about that path being replayed correctly inside the recomputed
    checkpoint block."""
    model = _tiny_mix_model()
    ids = _ids()
    names_params = _named_trainables(model)
    torch.manual_seed(3)
    proto = model._init_state(BATCH, DEVICE, torch.float32)

    hooks_mix.set_grad_checkpoint(model, True, block=BLOCK)
    _, grads_second_order = _loss_and_grads(model, ids, _fresh_state(model, proto), names_params)

    original_write = _NeuralMemory.write
    monkeypatch.setattr(
        _NeuralMemory,
        "write",
        lambda self, ks, vs, etas, thetas, alphas, create_graph=True: original_write(
            self, ks, vs, etas, thetas, alphas, create_graph=False
        ),
    )
    _, grads_first_order = _loss_and_grads(model, ids, _fresh_state(model, proto), names_params)

    for name in ("front_end.k_proj.weight", "front_end.v_proj.weight"):
        assert not torch.allclose(
            grads_second_order[name], grads_first_order[name], rtol=1e-3, atol=1e-6
        ), name


def test_backward_recomputes_every_block():
    """Without this, the equivalence tests would still pass if the block loop
    silently never checkpointed anything."""
    model = _tiny_mix_model()
    hooks_mix.set_grad_checkpoint(model, True, block=BLOCK)
    calls = []
    real_path = model._forward_path
    model._forward_path = lambda ids, state: (calls.append(ids.shape[1]), real_path(ids, state))[1]

    torch.manual_seed(3)
    logits, _ = model(_ids(), state=model._init_state(BATCH, DEVICE, torch.float32).detach())
    assert calls == [BLOCK] * (SEQ // BLOCK)
    logits.sum().backward()
    assert calls == [BLOCK] * (2 * SEQ // BLOCK)


def test_grad_checkpoint_disabled_or_no_grad_is_bit_identical():
    model = _tiny_mix_model()
    ids = _ids()
    torch.manual_seed(3)
    proto = model._init_state(BATCH, DEVICE, torch.float32)
    with torch.no_grad():
        logits_never, _ = model(ids, state=_fresh_state(model, proto))
        hooks_mix.set_grad_checkpoint(model, False)
        logits_off, _ = model(ids, state=_fresh_state(model, proto))
        # Enabled but under no_grad (inference/eval): nothing to save for a
        # backward that will never happen, so the single-call path stands.
        hooks_mix.set_grad_checkpoint(model, True, block=BLOCK)
        logits_no_grad, _ = model(ids, state=_fresh_state(model, proto))
    assert torch.equal(logits_off, logits_never)
    assert torch.equal(logits_no_grad, logits_never)


def test_grad_checkpoint_block_rounds_to_memory_window():
    model = _tiny_mix_model()
    model.set_memory_window(3)
    hooks_mix.set_grad_checkpoint(model, True, block=BLOCK)
    ids = torch.randint(0, VOCAB, (BATCH, 6), device=DEVICE)
    # BLOCK=4 isn't a multiple of window 3; a window may never straddle a
    # forward() call, so the effective block has to snap down to 3.
    logits, _ = model(ids)
    assert logits.shape == (BATCH, 6, VOCAB)


# --- plain backbone (mamba2_780m) ---


def test_grad_checkpoint_matches_uncheckpointed_plain_model():
    model = _tiny_plain_model()
    ids = _ids()
    names_params = _named_trainables(model)
    proto = model._init_state(BATCH, torch.float32)

    loss_ref, grads_ref = _loss_and_grads(model, ids, _fresh_state(model, proto), names_params)
    hooks780.set_grad_checkpoint(model, True, block=BLOCK)
    loss_ckpt, grads_ckpt = _loss_and_grads(model, ids, _fresh_state(model, proto), names_params)

    torch.testing.assert_close(loss_ckpt, loss_ref, rtol=1e-4, atol=1e-6)
    for name, _ in names_params:
        gr, gc = grads_ref[name], grads_ckpt[name]
        assert (gr is None) == (gc is None), name
        if gr is not None:
            torch.testing.assert_close(
                gc, gr, rtol=2e-4, atol=1e-5, msg=lambda m, n=name: f"{n}: {m}"
            )
