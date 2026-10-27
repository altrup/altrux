"""Model.torch_norms: the plain-torch norms match the Triton norms, and with
them the manual forward is twice differentiable."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import models  # noqa: F401  (installs the selective_scan_cuda stub)
import models.mamba2_780m.model as M780

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mamba_ssm norms need a working GPU"
)


def _tiny_model(fused_add_norm: bool, dtype: torch.dtype = torch.float32) -> M780.Model:
    torch.manual_seed(0)
    cfg = MambaConfig(
        d_model=64,
        n_layer=4,
        vocab_size=96,
        ssm_cfg={"layer": "Mamba2", "headdim": 16, "d_state": 16, "expand": 2, "ngroups": 1},
        rms_norm=True,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=True,
        tie_embeddings=True,
    )
    model = M780.Model(MambaLMHeadModel(cfg, device="cuda", dtype=dtype)).to("cuda")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:  # off the init's all-ones so the norm weights matter
                p.uniform_(0.5, 1.5)
    return model


def _logits(model: M780.Model, ids: torch.Tensor) -> torch.Tensor:
    return model._forward_tokens(
        ids, model._init_state(ids.shape[0], model.embedding.weight.dtype)
    )[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused_add_norm", [True, False])
def test_torch_norms_match_triton(fused_add_norm, dtype):
    model = _tiny_model(fused_add_norm, dtype)
    ids = torch.randint(0, 96, (2, 12), device="cuda")
    with torch.no_grad():
        triton = _logits(model, ids).float()
        model.torch_norms = True
        plain = _logits(model, ids).float()
    error = float((plain - triton).abs().max() / triton.abs().max())
    assert error < (1e-5 if dtype == torch.float32 else 2e-2), error


def _double_backward_error(model: M780.Model) -> float:
    """Relative error of d<grad_theta L(pi), g>/d pi, by double backward, against
    a central finite difference along g, where L(pi) = CE(pi, student)."""
    params = list(model.parameters())
    torch.manual_seed(1)
    ids = torch.randint(0, 96, (1, 8), device="cuda")
    pi = torch.softmax(torch.randn(7, 96, device="cuda"), -1).requires_grad_()
    g = torch.randn(sum(p.numel() for p in params), device="cuda")

    def log_q() -> torch.Tensor:
        return torch.log_softmax(_logits(model, ids)[0, :-1], -1)

    grads = torch.autograd.grad(-(pi * log_q()).sum(-1).mean(), params, create_graph=True)
    dot = torch.cat([gr.flatten() for gr in grads]) @ g
    analytic = torch.autograd.grad(dot, pi)[0]

    eps = 1e-3 / float(g.norm())
    steps = [
        d.view_as(p) for d, p in zip(g.split([p.numel() for p in params]), params, strict=True)
    ]
    log_qs = []
    with torch.no_grad():
        for sign in (1, -1):
            torch._foreach_add_(params, steps, alpha=sign * eps)
            log_qs.append(log_q())
            torch._foreach_add_(params, steps, alpha=-sign * eps)
    expected = -(log_qs[0] - log_qs[1]) / (2 * eps) / pi.shape[0]
    return float((analytic - expected).norm() / expected.norm())


@pytest.mark.parametrize("fused_add_norm", [True, False])
def test_torch_norms_make_the_forward_twice_differentiable(fused_add_norm):
    model = _tiny_model(fused_add_norm)
    model.torch_norms = True
    assert _double_backward_error(model) < 1e-2


@pytest.mark.parametrize("fused_add_norm", [True, False])
def test_triton_norms_cut_the_second_order_path(fused_add_norm):
    assert _double_backward_error(_tiny_model(fused_add_norm)) > 0.5
