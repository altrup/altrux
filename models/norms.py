"""The norms the manual mixer path calls, each with a plain-torch twin
(`plain=True`) that autograd differentiates twice. Separate from common.py,
which CPU-only data preparation imports: `import mamba_ssm` needs a GPU."""

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, layer_norm_ref, rms_norm_ref
from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref as gated_rms_norm_ref


def add_norm(
    norm: nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    plain: bool = False,
):
    """mamba_ssm's `layer_norm_fn` with `norm`'s weight, bias and eps.
    `plain` runs the same arithmetic (fp32 math, output in x's dtype, residual
    in the kernel's dtype) in plain torch ops, which autograd differentiates
    twice; the Triton kernel's backward is not differentiable."""
    if not plain:
        return layer_norm_fn(
            x,
            norm.weight,
            norm.bias,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            eps=norm.eps,
            is_rms_norm=isinstance(norm, RMSNorm),
        )
    ref = rms_norm_ref if isinstance(norm, RMSNorm) else layer_norm_ref
    out = ref(
        x, norm.weight, norm.bias, residual=residual, eps=norm.eps, prenorm=prenorm, upcast=True
    )
    if not prenorm:
        return out
    if residual is not None:
        residual_dtype = residual.dtype
    else:
        residual_dtype = torch.float32 if residual_in_fp32 else x.dtype
    return out[0], out[1].to(residual_dtype)


def gated_norm(
    norm: nn.Module, x: torch.Tensor, z: torch.Tensor, plain: bool = False
) -> torch.Tensor:
    """The Mamba2 mixer's gated `RMSNorm`, `norm(x, z)`; `plain` as in `add_norm`."""
    if not plain:
        return norm(x, z)
    return gated_rms_norm_ref(
        x,
        norm.weight,
        norm.bias,
        z=z,
        eps=norm.eps,
        group_size=norm.group_size,
        norm_before_gate=norm.norm_before_gate,
    )
