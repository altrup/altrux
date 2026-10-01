"""Experiment: shared"""

import functools
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

from ..common import MarkerDelta, blockwise_checkpoint

MODEL_ID = "state-spaces/mamba2-780m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TARGET_LORA_MODULES = ["in_proj", "out_proj"]

# Chat format role markers — registered as tokenizer special tokens (see
# SPECIAL_TOKENS), so each is a single atomic token id. Callers append the
# separator between marker and content explicitly (e.g. USER_OPEN + " " + content)
# — keep that separator a literal space to match the SFT training format exactly.
USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
# Conversation boundary, of the same family as <|endoftext|> (which ends an
# assistant turn in this corpus). Named for what the data teaches it -- the end
# of a conversation -- not for the dream-end reading sampling puts on it.
EOC = "<|endofconversation|>"
SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN, EOC]

# Default tokens per gradient-checkpointing block (see
# Model.set_grad_checkpoint). This backbone carries no fast-weight memory, so
# the boundary state is just the per-layer conv+ssm state -- ~19.5M values,
# ~39 MB per batch slot in bf16 at this model's shape (48 layers, d_state
# 128) -- against a per-token activation graph the manual mixer step holds
# for every one of those layers. 64 matches the memory arms' default (see
# models/mamba2_2_7b_memory/model.py's GRAD_CHECKPOINT_BLOCK for the
# arithmetic), keeping one number to reason about across the models.
GRAD_CHECKPOINT_BLOCK = 64


@functools.cache
def _chunk_scan_kernel():
    """The fused SSD chunk-scan, or None where it can't be used: it hangs on
    this project's ROCm dev box (see Model), and a broken or absent kernel
    build must not break importing the model."""
    if torch.version.hip is not None:
        return None
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        return None
    return mamba_chunk_scan_combined


class MixerState:
    """Per-layer SSM/conv state threaded across calls to Model.forward, so a
    sequence can be processed incrementally (decode) or in chunks (long-
    sequence training) without holding every token's activations at once.

    Analogous to mamba2_2_7b_memory's MemoryState, minus anything
    memory-specific -- this model has no memory subsystem, just the
    backbone's own state.
    """

    def __init__(self, conv_states, ssm_states):
        self.conv_states = conv_states
        self.ssm_states = ssm_states

    def detach(self) -> "MixerState":
        return MixerState(
            [c.detach() for c in self.conv_states], [s.detach() for s in self.ssm_states]
        )

    def flatten(self) -> tuple[torch.Tensor, ...]:
        """Every carried tensor, in a fixed order, for crossing a
        torch.utils.checkpoint boundary (see models.common.
        blockwise_checkpoint)."""
        return (*self.conv_states, *self.ssm_states)

    def unflatten(self, tensors: tuple[torch.Tensor, ...]) -> "MixerState":
        n = len(self.conv_states)
        return MixerState(list(tensors[:n]), list(tensors[n:]))


class Model(nn.Module):
    """Mamba LM wrapper for inference and training.

    Drives the mixer directly rather than calling the library's own
    Mamba2.forward/Block.forward, over two interchangeable paths:
    `_mixer_chunk` runs a whole chunk through the fused Triton SSD chunk-scan
    (with the conv as a plain `F.conv1d`, so no causal_conv1d build is
    needed), and `_mixer_step` replicates Mamba2.step()'s arithmetic in plain
    PyTorch one token at a time. `_mixer_step` is what a single decode step
    uses, and the only path that runs at all on this project's dev box: both
    of mamba_ssm's fused kernel families are broken on that hardware (an
    unsupported ROCm gfx arch) -- causal_conv1d's compiled kernel segfaults
    (confirmed on both the multi-token "channellast" path and the
    single-token decode path) and the Triton SSD scan kernel hangs, both
    confirmed independent of model size, package version, and a from-source
    rebuild. See this README's `Model` wrapper quirks section for the full
    investigation.

    `forward` threads/returns a MixerState, so calling it repeatedly with
    state carried (and detached) across calls
    supports incremental decode and chunked training on long sequences
    without holding the whole sequence's activations at once -- the same
    pattern as mamba2_2_7b_memory's MemoryState. This replaces the previous
    `inference_params`-based forward; see git history if a comparison is
    ever needed.

    Generates tokens until EOS. The model is fine-tuned to emit a <revise>
    tag after its response; for now that tag is treated like any other token.
    TODO: detect the <revise> tag and act on it.
    """

    def __init__(self, mamba_model: MambaLMHeadModel):
        super().__init__()
        backbone = mamba_model.backbone

        self.d_model: int = backbone.layers[0].mixer.d_model
        self.fused_add_norm: bool = backbone.fused_add_norm
        self.residual_in_fp32: bool = backbone.residual_in_fp32

        self.embedding = backbone.embedding
        self.layers = nn.ModuleList(list(backbone.layers))
        self.norm_f = backbone.norm_f
        self.lm_head = mamba_model.lm_head

        # Trainable delta for the frozen role-marker embedding/lm_head rows
        # (see MarkerDelta); marker_token_ids is stamped by extend_embeddings,
        # absent on raw models (e.g. the tiny test fixtures).
        marker_ids = getattr(mamba_model, "marker_token_ids", None)
        self.marker_delta = (
            MarkerDelta(
                marker_ids,
                self.d_model,
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
            if marker_ids
            else None
        )

        for layer in self.layers:
            assert layer.mixer.ngroups == 1, "manual mixer step assumes ngroups=1"

        # Tokens per gradient-checkpointing block; 0 disables it entirely
        # (see set_grad_checkpoint). Plain attribute, never checkpointed.
        self.grad_checkpoint_block = 0

        # When set to a list, _mixer_step appends each layer's post-conv read
        # query C (detached) -- one entry per layer per token, token-major.
        # Used by sft/experiments/erasure/probe.py (and the dream-sleep loop) to address the
        # rank-1 state erase; per-token path only, so drive the model one
        # token at a time while capturing.
        self.c_capture: list[torch.Tensor] | None = None

        # When set, _mixer_step calls it as hook(layer_idx, ssm_state, C) just
        # before that layer's decay+write and writes onto whatever it returns
        # -- the ablate-then-write-then-read micro-order the dream-sleep
        # counterfactual arms need (sft/experiments/dreams/cli.py). Per-token path only.
        self.erase_hook: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    def _init_state(self, batch_size: int, dtype) -> MixerState:
        conv_states, ssm_states = [], []
        for layer in self.layers:
            conv_state, ssm_state = layer.mixer.allocate_inference_cache(batch_size, 1, dtype=dtype)
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        return MixerState(conv_states, ssm_states)

    def _prenorm(self, layer, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        if not layer.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            normed = layer.norm(residual.to(dtype=layer.norm.weight.dtype))
            if layer.residual_in_fp32:
                residual = residual.to(torch.float32)
            return normed, residual
        return layer_norm_fn(
            hidden_states,
            layer.norm.weight,
            layer.norm.bias,
            residual=residual,
            prenorm=True,
            residual_in_fp32=layer.residual_in_fp32,
            eps=layer.norm.eps,
            is_rms_norm=isinstance(layer.norm, RMSNorm),
        )

    def _mixer_step(
        self, mixer, hidden_states: torch.Tensor, conv_state, ssm_state, layer_idx: int = 0
    ):
        """One token through `mixer`, replicating Mamba2.step()'s arithmetic
        manually (see class docstring for why). `ssm_state`/`conv_state` are
        *not* mutated in place (unlike the library's own decode cache): each
        is rebound to a fresh tensor every call so gradients flow correctly
        across tokens within a forward pass.
        """
        dtype = hidden_states.dtype
        zxbcdt = mixer.in_proj(hidden_states)
        d_mlp = (
            zxbcdt.shape[-1] - 2 * mixer.d_ssm - 2 * mixer.ngroups * mixer.d_state - mixer.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                mixer.d_ssm,
                mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state,
                mixer.nheads,
            ],
            dim=-1,
        )

        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(conv_state * rearrange(mixer.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if mixer.conv1d.bias is not None:
            xBC = xBC + mixer.conv1d.bias
        xBC = mixer.act(xBC).to(dtype=dtype)

        x, B, C = torch.split(
            xBC, [mixer.d_ssm, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state], dim=-1
        )
        if self.c_capture is not None:
            self.c_capture.append(C.detach())
        if self.erase_hook is not None:
            ssm_state = self.erase_hook(layer_idx, ssm_state, C)
        A = -torch.exp(mixer.A_log.float())

        dt = F.softplus(dt + mixer.dt_bias.to(dtype=dt.dtype))
        dA = torch.exp(dt * A)
        x_h = rearrange(x, "b (h p) -> b h p", p=mixer.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x_h)
        ssm_state = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx

        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(mixer.D.to(dtype), "h -> h 1") * x_h
        y = rearrange(y, "b h p -> b (h p)")
        if not mixer.rmsnorm:
            y = y * mixer.act(z)
        else:
            y = mixer.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = mixer.out_proj(y)
        return out, conv_state, ssm_state

    def _mixer_chunk(self, mixer, hidden_states: torch.Tensor, conv_state, ssm_state):
        """A whole (B, T, d_model) chunk through `mixer` in one shot, via the
        fused SSD chunk-scan -- the parallel equivalent of looping
        `_mixer_step` over T, same states in and out. The conv runs as a plain
        grouped `F.conv1d` over the incoming `conv_state` prepended as left
        context, so no causal_conv1d build is required.
        """
        dtype = hidden_states.dtype
        zxbcdt = mixer.in_proj(hidden_states)
        d_mlp = (
            zxbcdt.shape[-1] - 2 * mixer.d_ssm - 2 * mixer.ngroups * mixer.d_state - mixer.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                mixer.d_ssm,
                mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state,
                mixer.nheads,
            ],
            dim=-1,
        )

        d_conv = conv_state.shape[-1]
        padded = torch.cat([conv_state[:, :, 1:], rearrange(xBC, "b l d -> b d l")], dim=-1)
        # .contiguous(): a view here would keep the whole chunk's conv input alive via the outgoing state.
        conv_state = padded[:, :, -d_conv:].contiguous()
        xBC = F.conv1d(padded, mixer.conv1d.weight, mixer.conv1d.bias, groups=mixer.conv1d.groups)
        xBC = mixer.act(rearrange(xBC, "b d l -> b l d")).to(dtype=dtype)

        x, B, C = torch.split(
            xBC, [mixer.d_ssm, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state], dim=-1
        )
        A = -torch.exp(mixer.A_log.float())

        y, ssm_state = _chunk_scan_kernel()(
            rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=mixer.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=mixer.ngroups),
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            z=None,
            dt_bias=mixer.dt_bias,
            initial_states=ssm_state,
            dt_softplus=True,
            return_final_states=True,
        )

        y = rearrange(y, "b l h p -> b l (h p)")
        if not mixer.rmsnorm:
            y = y * mixer.act(z)
        else:
            y = mixer.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = mixer.out_proj(y)
        return out, conv_state, ssm_state

    def _apply_norm_f(self, h: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        if not self.fused_add_norm:
            combined = (h + residual) if residual is not None else h
            return self.norm_f(combined.to(self.norm_f.weight.dtype))
        return layer_norm_fn(
            h,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=isinstance(self.norm_f, RMSNorm),
        )

    def forward(
        self, input_ids: torch.Tensor, state: MixerState | None = None
    ) -> tuple[torch.Tensor, MixerState]:
        """Returns (logits (B, T, vocab_size), updated MixerState).

        Pass state=None to start a fresh sequence; pass the returned state
        back in to continue it (incremental decode, or the next chunk of a
        long sequence). A multi-token chunk runs through the fused SSD
        chunk-scan where that kernel is usable, and otherwise (and always for
        a single decode step) one token at a time -- same states, same
        logits, see the class docstring.
        """
        batch_size, seqlen = input_ids.shape
        dtype = self.embedding.weight.dtype
        if state is None:
            state = self._init_state(batch_size, dtype)

        fused = seqlen > 1 and _chunk_scan_kernel() is not None and self.erase_hook is None
        forward_fn = self._forward_chunk if fused else self._forward_tokens
        block = self.grad_checkpoint_block if torch.is_grad_enabled() else 0
        if 0 < block < seqlen:
            return blockwise_checkpoint(forward_fn, input_ids, state, block)
        return forward_fn(input_ids, state)

    def set_grad_checkpoint(self, enabled: bool, block: int = GRAD_CHECKPOINT_BLOCK) -> None:
        """Turn block-wise gradient checkpointing on or off for subsequent
        forward() calls (see GRAD_CHECKPOINT_BLOCK for the block size).
        Disabled is an exact no-op: forward() takes the same single-call path
        it always did, so inference, evaluation and any no_grad() call are
        untouched, and nothing about this reaches a checkpoint file."""
        self.grad_checkpoint_block = block if enabled else 0

    def _forward_tokens(
        self, input_ids: torch.Tensor, state: MixerState
    ) -> tuple[torch.Tensor, MixerState]:
        seqlen = input_ids.shape[1]
        all_logits = []
        for t in range(seqlen):
            h = self.embedding(input_ids[:, t])
            if self.marker_delta is not None:
                h = self.marker_delta.embed(h, input_ids[:, t])
            residual = None
            for i, layer in enumerate(self.layers):
                h, residual = self._prenorm(layer, h, residual)
                h, conv_state, ssm_state = self._mixer_step(
                    layer.mixer, h, state.conv_states[i], state.ssm_states[i], i
                )
                state.conv_states[i] = conv_state
                state.ssm_states[i] = ssm_state

            h = self._apply_norm_f(h, residual)
            logits = self.lm_head(h)
            if self.marker_delta is not None:
                logits = self.marker_delta.head(logits, h)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1), state

    def _forward_chunk(
        self, input_ids: torch.Tensor, state: MixerState
    ) -> tuple[torch.Tensor, MixerState]:
        h = self.embedding(input_ids)
        if self.marker_delta is not None:
            h = self.marker_delta.embed(h, input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            h, residual = self._prenorm(layer, h, residual)
            h, conv_state, ssm_state = self._mixer_chunk(
                layer.mixer, h, state.conv_states[i], state.ssm_states[i]
            )
            state.conv_states[i] = conv_state
            state.ssm_states[i] = ssm_state

        h = self._apply_norm_f(h, residual)
        logits = self.lm_head(h)
        if self.marker_delta is not None:
            logits = self.marker_delta.head(logits, h)
        return logits, state


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/training/loop.py.

    Loads in bf16, not fp32 -- this model's manual per-token mixer step
    (see Model docstring) holds a live backward graph whose activation
    memory scales with chunk_len; bf16 halves both that and the frozen
    backbone's own weight memory.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer), tokenizer)
    return model


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device)).to(device)
