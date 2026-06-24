"""Mamba2-780M backbone (frozen) + a trainable long-term memory subsystem.

The memory is a single Titans-style fast-weight MLP that is test-time-trained
token by token, feeding a per-layer gated-delta rule that's merged directly
into a sparse subset of the backbone's own SSM states (not a separate
accumulator -- see `Model._mixer_step`). See README.md for the full design
write-up; this module is the literal implementation of it.

Implementation note: because the gated-delta merge at layer i is a
function of the memory read at READ_LAYER, and that read for token t must be
available to *later* layers of the *same* token while only being available to
*earlier* layers on the *next* token, the backbone can't run through Mamba2's
fused/chunked parallel-scan kernels here -- those process a whole sequence in
one kernel call and don't expose a per-token, pre-readout hook. Model.forward
therefore loops over time explicitly, one token at a time, replicating
Mamba2's own incremental-decode arithmetic (see `_mixer_step`) for every
layer. This is the same asymptotic cost as autoregressive decoding, just paid
during training too -- materially slower than the library's native chunked
training path. Revisit if this becomes a bottleneck (e.g. a custom chunked
kernel that exposes the pre-readout state).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_update
except ImportError:
    causal_conv1d_update = None

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

MODEL_ID = "state-spaces/mamba2-780m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
# The backbone is LoRA-adapted (frozen, full-precision in_proj/out_proj +
# trainable low-rank adapters) rather than left fully frozen, on the theory
# that a fully frozen backbone is unlikely to integrate a memory signal
# injected straight into its SSM state well. This 780M backbone is small
# enough to fit this project's dev GPU (8GB) without 4-bit quantization,
# unlike the 2.7B backbone this model used previously (see git history),
# which needed QLoRA -- so no QUANTIZE_LORA_BASE flag here, same as
# mamba2_780m. The memory subsystem (front-end, gate projections) is
# separate from LoRA: it has no pretrained weights to adapt, so it trains
# with ordinary full-parameter gradients from a random init.
TARGET_LORA_MODULES: list[str] = ["in_proj", "out_proj"]

# Chat format role markers -- see models/mamba2_780m/model.py for the
# rationale; identical convention here.
USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN]

# Mamba2-780M backbone shape (state-spaces/mamba2-780m).
D_MODEL = 1536
N_LAYER = 48
NHEADS = 48
HEADDIM = 64
D_STATE = 128

# Memory subsystem hyperparameters (see README.md for the rationale).
# READ_LAYER/INJECTED_LAYERS are rescaled proportionally from this model's
# previous 2.7B backbone (READ_LAYER=42/64, INJECTED_LAYERS=range(20,64,2))
# to preserve the same relative depth (~2/3) and coverage (~1/3) on this
# backbone's 48 layers, not re-derived from scratch -- see git history for
# the original rationale (weak keys too early, collapsed-to-next-token too
# late).
READ_LAYER = 32
INJECTED_LAYERS: tuple[int, ...] = tuple(range(16, N_LAYER, 2))
BOTTLENECK_R = 128
MEM_DIM = D_MODEL
MEM_HIDDEN = 4 * D_MODEL
# Caps the per-token test-time gradient step in _NeuralMemory.write() (see
# its comment there) -- bounds the update size regardless of how large the
# raw prediction error/gradient is, which an untrained memory MLP can
# produce on its very first few tokens. Tried raising this to 3000 on the
# theory that w1/w2's ~9.4M elements each (on this backbone's MEM_DIM/
# MEM_HIDDEN) make a *healthy* gradient norm naturally land in the
# thousands -- wrong in practice: it only delayed the explosion by a few
# tokens rather than preventing it, since momentum `s` in write() has no
# decay of its own (only `p` decays, via alpha) -- eta near 1 lets
# bounded-but-nonzero per-step contributions accumulate across tokens
# regardless of the single-step clip. 1.0 is the only value confirmed (with
# q/k/v normalized -- see _rms_normalize) to run a full example through
# preflight without going non-finite. Revisit alongside decaying/clamping
# momentum itself, not just the per-step gradient, before raising this again.
MAX_WRITE_GRAD_NORM = 1.0
# Diagnostic-only threshold for counting an injected layer as "actively"
# writing into ssm_state this token (see Model.last_token_log) -- beta > 0.5
# means that layer's gated-delta merge is overwriting more than half its
# forgotten state with the memory's content, vs. its near-0 no-op init.
# Used only to report active_layers for live logging; the actual gated-delta
# merge in _mixer_step always uses the raw, continuous beta value -- this
# threshold has no effect on the forward pass or training.
ACTIVE_BETA_THRESHOLD = 0.5


class _NeuralMemory:
    """Titans-style fast-weight MLP: a 2-layer MLP whose *weights* are the
    memory content, mutated by a test-time gradient step every token.

    Plain tensors with a batch dimension, not an nn.Module -- M's weights
    differ per sequence and change every step, so they can't live in
    nn.Parameter. M's initial weights are freshly random per sequence (only
    the surrounding projections in _TitansFrontEnd are actually learned);
    this matches the Titans formulation where M only ever memorizes the
    current context, never a global prior.
    """

    def __init__(self, batch_size: int, dim: int, hidden_dim: int, device, dtype):
        bound1 = 1.0 / math.sqrt(dim)
        bound2 = 1.0 / math.sqrt(hidden_dim)
        self.w1 = (torch.rand(batch_size, hidden_dim, dim, device=device, dtype=dtype) * 2 - 1) * bound1
        self.b1 = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
        self.w2 = (torch.rand(batch_size, dim, hidden_dim, device=device, dtype=dtype) * 2 - 1) * bound2
        self.b2 = torch.zeros(batch_size, dim, device=device, dtype=dtype)
        self.momentum = [torch.zeros_like(p) for p in (self.w1, self.b1, self.w2, self.b2)]

    @staticmethod
    def _apply(x: torch.Tensor, w1, b1, w2, b2) -> torch.Tensor:
        h = torch.tanh(torch.einsum("bhd,bd->bh", w1, x) + b1)
        return torch.einsum("bdh,bh->bd", w2, h) + b2

    def write(self, k: torch.Tensor, v: torch.Tensor, eta: torch.Tensor, theta: torch.Tensor, alpha: torch.Tensor):
        """One test-time gradient step on L = ||M(k) - v||^2 -- first-order
        (truncated) approximation: `params`/`momentum` carried in from the
        previous token are detached and re-leafed here, so M_{t-1} is treated
        as a constant w.r.t. autograd. THIS is what bounds the graph to O(1)
        per token instead of O(T) -- each token starts from a fresh leaf, so
        nothing chains back to token t-1, regardless of the create_graph value
        below. The outer (SFT) loss still backprops through *this* token's own
        write (via eta/theta/alpha/k/v, all functions of the front-end's
        learnable projections), but not through the chain of all earlier
        tokens' writes -- see git history / README for the exact math this
        trades away. `.detach().requires_grad_(True)` also happens to be what
        makes the very first call work at all: fresh `w1/b1/w2/b2` are plain
        leaf tensors with requires_grad=False (see __init__), and
        torch.autograd.grad requires `inputs` to require grad.

        Returns the loss magnitude per batch element ("surprise"), exported
        for Stage 2's write-strength gate.
        """
        params = [p.detach().requires_grad_(True) for p in (self.w1, self.b1, self.w2, self.b2)]
        momentum = [s.detach() for s in self.momentum]

        pred = self._apply(k, *params)
        per_example_loss = ((pred - v) ** 2).sum(dim=-1)
        # create_graph=True: g must stay differentiable w.r.t. k/v (and the
        # freshly-detached `params` above) so k_proj/v_proj actually receive
        # gradient -- with create_graph=False, g is a plain non-differentiable
        # number, severing k_proj/v_proj from the outer loss entirely
        # (confirmed: this was happening). This does NOT reintroduce the O(T)
        # blowup -- that came from *not* detaching params/momentum per token,
        # not from this flag; `params` being a fresh leaf each call means
        # differentiating g w.r.t. it can't chain past this single token.
        grads = torch.autograd.grad(per_example_loss.sum(), params, create_graph=True)
        # Clip the combined per-example gradient norm: M starts randomly
        # initialized, so its first prediction error (and hence g) can
        # already be huge -- uncapped, eta near 1 (near-zero momentum decay)
        # lets that single huge step make the next token's error (and g)
        # even bigger, a runaway that overflows fp32 within ~4 tokens
        # (confirmed: this was happening, independent of chunk_len/dtype).
        # Clipping g caps the per-token update size regardless of how bad
        # M's current prediction is, breaking that feedback loop.
        sq_norm = sum((g.float() ** 2).reshape(g.shape[0], -1).sum(dim=1) for g in grads)
        clip_scale = (MAX_WRITE_GRAD_NORM / sq_norm.clamp_min(1e-12).sqrt()).clamp(max=1.0)
        grads = [g * clip_scale.view(-1, *([1] * (g.dim() - 1))) for g in grads]

        new_params = []
        new_momentum = []
        for p, g, s in zip(params, grads, momentum):
            view = (-1,) + (1,) * (p.dim() - 1)
            s_new = eta.view(*view) * s - theta.view(*view) * g
            p_new = (1 - alpha.view(*view)) * p + s_new
            new_params.append(p_new)
            new_momentum.append(s_new)
        self.momentum = new_momentum
        self.w1, self.b1, self.w2, self.b2 = new_params
        return per_example_loss.detach()

    def read(self, q: torch.Tensor) -> torch.Tensor:
        """o_t = M_t(q_t), called after write() so the read uses the
        just-updated weights."""
        return self._apply(q, self.w1, self.b1, self.w2, self.b2)


def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scales x to unit RMS along its last dim -- no learnable weight, since
    q_proj/k_proj/v_proj already have one; this only strips the raw,
    unbounded magnitude those projections would otherwise pass through."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class _TitansFrontEnd(nn.Module):
    """Shared, single front-end: projects the layer-READ_LAYER residual to
    q/k/v and to the data-dependent write knobs (eta, theta, alpha), then
    drives one `_NeuralMemory` write+read per token."""

    def __init__(self, d_model: int = D_MODEL, mem_dim: int = MEM_DIM, mem_hidden: int = MEM_HIDDEN):
        super().__init__()
        self.mem_dim = mem_dim
        self.mem_hidden = mem_hidden
        self.q_proj = nn.Linear(d_model, mem_dim)
        self.k_proj = nn.Linear(d_model, mem_dim)
        self.v_proj = nn.Linear(d_model, mem_dim)
        # eta (momentum), theta (step size), alpha (decay) -- all small and
        # data-dependent, per Titans.
        self.knob_proj = nn.Linear(d_model, 3)

    def init_memory(self, batch_size: int, device, dtype) -> _NeuralMemory:
        return _NeuralMemory(batch_size, self.mem_dim, self.mem_hidden, device, dtype)

    def step(self, residual: torch.Tensor, memory: _NeuralMemory) -> tuple[torch.Tensor, torch.Tensor]:
        """residual: (batch, d_model) residual stream entering READ_LAYER.

        Returns (o_t, surprise) -- o_t is (batch, mem_dim), surprise is (batch,).

        The write is a genuine gradient step (autograd.grad inside
        _NeuralMemory.write), which needs grad tracking enabled regardless of
        whether the *caller* is in a torch.no_grad() block -- this isn't
        optional training-time machinery, it's the actual write operation, so
        it has to run even during eval/inference. torch.enable_grad() punches
        through an enclosing no_grad() for exactly this; it's scoped to just
        this method, so the rest of the model (the 64-layer backbone, the
        gated-delta merge, the C readout) stays grad-free and cheap under an
        outer no_grad() as normal -- only this small MLP's self-contained
        write pays for gradient tracking.
        """
        with torch.enable_grad():
            # RMS-normalize q/k/v to unit scale before they reach the memory:
            # raw projections of the (unnormalized) residual stream can have
            # individual components in the tens (confirmed: k/v components up
            # to ~50-60), which made M's write loss ||M(k) - v||^2 -- and its
            # gradient -- scale with that raw, unbounded magnitude rather than
            # with anything learned. Normalizing bounds the *input* magnitude
            # at the source so MAX_WRITE_GRAD_NORM clipping in
            # _NeuralMemory.write isn't permanently saturated; theta/eta
            # regain control over how strongly a given token's write lands.
            q = _rms_normalize(self.q_proj(residual))
            k = _rms_normalize(self.k_proj(residual))
            v = _rms_normalize(self.v_proj(residual))
            knobs = torch.sigmoid(self.knob_proj(residual))
            # eta capped at 0.9 (not left at sigmoid's full (0, 1) range like
            # theta/alpha's caps, which bound them *small* on purpose --
            # small theta/alpha is what makes the write step gentle and the
            # memory's content persist across a long document): S_t = eta *
            # S_{t-1} - theta * g_t accumulates with essentially no decay
            # when eta is allowed to approach 1 (confirmed: 0.9999974 in a
            # real run), turning the momentum into an undamped running sum of
            # every token's write -- the actual mechanism behind chunk 8:10
            # still going non-finite even with grads clipped to a generous
            # 3000. Capping below 1 guarantees at least 10% decay per token.
            eta, theta, alpha = knobs[..., 0] * 0.9, knobs[..., 1] * 0.1, knobs[..., 2] * 0.1
            surprise = memory.write(k, v, eta, theta, alpha)
            o_t = memory.read(q)
        return o_t, surprise


class _GatedDeltaInjection(nn.Module):
    """One per injected layer. Derives this layer's own gated-delta write
    signals (p, key, beta, clear) from the shared o_t through a private
    bottleneck. Unlike an earlier version of this module, it does not own
    any persistent state itself -- the gated-delta rule is applied directly
    to Mamba2's own `ssm_state` in `Model._mixer_step`, so memory and
    backbone state are the same tensor, not two tensors added together."""

    def __init__(
        self,
        mem_dim: int = MEM_DIM,
        r: int = BOTTLENECK_R,
        nheads: int = NHEADS,
        headdim: int = HEADDIM,
        d_state: int = D_STATE,
    ):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        self.d_state = d_state
        self.down = nn.Linear(mem_dim, r)
        self.value_proj = nn.Linear(r, nheads * headdim)
        self.key_proj = nn.Linear(r, d_state)
        self.beta_proj = nn.Linear(r, 1)
        self.clear_proj = nn.Linear(r, 1)
        # beta starts near 0 and clear starts near 1 (sigmoid(-4) ~ 0.02,
        # sigmoid(4) ~ 0.98) so the gated-delta merge is a no-op at init --
        # the wrapped backbone behaves exactly like the unmodified pretrained
        # model until these gates learn otherwise. There is no separate
        # gamma scalar (as an earlier version had): beta itself is now the
        # only knob controlling how strongly memory overwrites backbone
        # state, since the merge is applied directly to ssm_state.
        nn.init.constant_(self.beta_proj.bias, -4.0)
        nn.init.constant_(self.clear_proj.bias, 4.0)

    def signals(self, o_t: torch.Tensor, surprise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, key, beta, clear) for this layer's gated-delta merge.

        p: (batch, nheads, headdim) per-head value to write.
        key: (batch, d_state) shared write address.
        beta, clear: (batch, 1, 1, 1) gates, broadcastable against ssm_state.
        """
        z = self.down(o_t)
        p = rearrange(self.value_proj(z), "b (h p) -> b h p", p=self.headdim)
        key = self.key_proj(z)
        # surprise is a sum of squared error over MEM_DIM dims, so raw values
        # land in the hundreds-to-thousands -- added directly into beta's
        # logit it completely swamps the -4.0 no-op bias (confirmed: a real
        # run measured beta=0.9968 right at the first step, not the intended
        # ~0.02). Normalize to mean-squared-error-per-dim (an O(1) quantity,
        # ~0.45 in that same run) and squash with its own sigmoid -- bounded
        # to [0.5, 1) since surprise >= 0 -- before adding, so its
        # contribution to the logit is bounded and beta_proj's -4.0 bias can
        # still dominate by default, preserving the no-op-at-init behavior.
        surprise_signal = torch.sigmoid(surprise / MEM_DIM)
        beta = torch.sigmoid(self.beta_proj(z).squeeze(-1) + surprise_signal).view(-1, 1, 1, 1)
        clear = torch.sigmoid(self.clear_proj(z)).view(-1, 1, 1, 1)
        return p, key, beta, clear


class MemoryState:
    """Everything that needs to persist across calls to `Model.forward` for
    a given conversation: the backbone's own per-layer SSM/conv state (which
    now also carries the memory content -- see `Model._mixer_step`) and the
    Titans fast-weight memory's last read-out, used to derive next step's
    gated-delta write signals for layers that run before READ_LAYER."""

    def __init__(
        self,
        conv_states,
        ssm_states,
        neural_memory: _NeuralMemory,
        last_o_t: torch.Tensor,
        last_surprise: torch.Tensor,
    ):
        self.conv_states = conv_states
        self.ssm_states = ssm_states
        self.neural_memory = neural_memory
        self.last_o_t = last_o_t
        self.last_surprise = last_surprise

    def detach(self) -> "MemoryState":
        """Returns a copy with every tensor detached from the autograd graph.

        `_NeuralMemory.write` already detaches/re-leafs its params at the start
        of *each token's* write, but the state returned after a whole chunk of
        tokens still carries a live graph back through every token in that
        chunk (that's what makes within-chunk backprop work). For chunked
        training, the state handed to the *next* chunk must be cut here --
        otherwise a long sequence processed as many chunks would still build
        one ever-growing graph across chunk boundaries, defeating the point of
        chunking. Safe to call between tokens too (it's a no-op there, since
        write() already detached).
        """
        neural_memory = _NeuralMemory.__new__(_NeuralMemory)
        neural_memory.w1 = self.neural_memory.w1.detach()
        neural_memory.b1 = self.neural_memory.b1.detach()
        neural_memory.w2 = self.neural_memory.w2.detach()
        neural_memory.b2 = self.neural_memory.b2.detach()
        neural_memory.momentum = [s.detach() for s in self.neural_memory.momentum]
        return MemoryState(
            conv_states=[c.detach() for c in self.conv_states],
            ssm_states=[s.detach() for s in self.ssm_states],
            neural_memory=neural_memory,
            last_o_t=self.last_o_t.detach(),
            last_surprise=self.last_surprise.detach(),
        )


class Model(nn.Module):
    """LoRA-adapted Mamba2-780M backbone with the memory subsystem spliced in.

    in_proj/out_proj and the rest of the backbone are frozen at bf16 (see
    load_base for why not fp32); LoRA adapters on in_proj/out_proj and the
    memory subsystem itself are the only trainable parameters.

    `forward` does not take an `inference_params` object from the mamba_ssm
    library (none of this repo's models do -- see models/mamba2_780m/model.py
    for why) -- it manages its own `MemoryState` (see above), which on top of
    the per-layer SSM/conv state the other models also carry, additionally
    threads the Titans fast-weight memory across calls, since the memory
    subsystem's recurrence needs token-level access a generic cache object
    wouldn't expose. Pass `state=None` to start a fresh conversation; pass
    back the returned state to continue one.
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

        for p in self.embedding.parameters():
            p.requires_grad_(False)
        for name, p in self.layers.named_parameters():
            # LoRA adapters (lora_A/lora_B) may already be attached to
            # in_proj/out_proj before this model is constructed -- see
            # apply_lora in sft/lora.py / backend/app/model/lora.py. Leave
            # those trainable; freeze the rest of the (QLoRA-quantized)
            # backbone, same as before.
            p.requires_grad_("lora_A" in name or "lora_B" in name)
        for p in self.norm_f.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        for layer in self.layers:
            assert layer.mixer.ngroups == 1, "memory injection assumes ngroups=1"

        self.front_end = _TitansFrontEnd()
        self.injections = nn.ModuleDict({str(i): _GatedDeltaInjection() for i in INJECTED_LAYERS})
        # Running sums for pop_memory_stats() -- plain floats, not tensors, so
        # they never hold a reference into any autograd graph. beta/clear are
        # accumulated once per injected layer per token; surprise/o_t_norm
        # once per token (at READ_LAYER only). See pop_memory_stats for why
        # these matter: beta/clear start near 0/1 (no-op init, see
        # _GatedDeltaInjection.__init__) and are the only direct signal of
        # whether the memory subsystem is actually being used or still
        # sitting at its identity init.
        self._mem_stat_sums = {"beta": 0.0, "clear": 0.0, "surprise": 0.0, "o_t_norm": 0.0}
        self._mem_stat_inj_count = 0
        self._mem_stat_tok_count = 0
        # Snapshot of the single most-recently-processed token, for live
        # per-token logging (overwritten every token, never accumulated --
        # see last_token_log). "active" = beta above ACTIVE_BETA_THRESHOLD,
        # i.e. that layer's gated-delta merge is actually overwriting
        # ssm_state this token rather than sitting near its no-op init.
        self._last_token_log: dict = {}

    def _init_state(self, batch_size: int, device, dtype) -> MemoryState:
        conv_states, ssm_states = [], []
        for layer in self.layers:
            conv_state, ssm_state = layer.mixer.allocate_inference_cache(batch_size, 1, dtype=dtype)
            conv_states.append(conv_state)
            ssm_states.append(ssm_state)
        # The memory subsystem (front_end/injections) is full-precision and
        # trained from scratch, independent of whatever dtype the backbone
        # itself loads in (`dtype` here, bf16 -- see load_base) -- its own
        # state must use its own params' dtype, not the backbone's, or
        # Injection.signals/_TitansFrontEnd.step mismatch dtypes against it.
        mem_dtype = self.front_end.q_proj.weight.dtype
        neural_memory = self.front_end.init_memory(batch_size, device, mem_dtype)
        last_o_t = torch.zeros(batch_size, MEM_DIM, device=device, dtype=mem_dtype)
        last_surprise = torch.zeros(batch_size, device=device, dtype=mem_dtype)
        return MemoryState(conv_states, ssm_states, neural_memory, last_o_t, last_surprise)

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
        self,
        mixer,
        hidden_states: torch.Tensor,
        conv_state,
        ssm_state,
        gated_delta: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ):
        """One token through `mixer`, replicating Mamba2.step()'s arithmetic
        but always via the manual (non-Triton-fused) SSM readout, so the
        gated-delta memory merge (`gated_delta = (p, key, beta, clear)`) can
        be applied directly to `ssm_state` immediately after Mamba2's own
        decay+write and before the C readout -- the merged state is what
        gets both read out *and* persisted, so memory content now decays
        under Mamba2's own `A` like everything else in the state, rather
        than living in a separate accumulator. `ssm_state` is *not* mutated
        in place (unlike the library's decode cache): it's rebound to a
        fresh tensor each call so gradients flow through the whole sequence
        during training.
        """
        dtype = hidden_states.dtype
        zxbcdt = mixer.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * mixer.d_ssm - 2 * mixer.ngroups * mixer.d_state - mixer.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt, [d_mlp, d_mlp, mixer.d_ssm, mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state, mixer.nheads], dim=-1
        )

        if causal_conv1d_update is None:
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(mixer.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if mixer.conv1d.bias is not None:
                xBC = xBC + mixer.conv1d.bias
            xBC = mixer.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC, conv_state, rearrange(mixer.conv1d.weight, "d 1 w -> d w"), mixer.conv1d.bias, mixer.activation
            )

        x, B, C = torch.split(xBC, [mixer.d_ssm, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state], dim=-1)
        A = -torch.exp(mixer.A_log.float())

        dt = F.softplus(dt + mixer.dt_bias.to(dtype=dt.dtype))
        dA = torch.exp(dt * A)
        x_h = rearrange(x, "b (h p) -> b h p", p=mixer.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x_h)
        ssm_state = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx

        if gated_delta is not None:
            # gated_delta's tensors come from the memory subsystem, which runs
            # at its own (fp32) dtype regardless of the backbone's (see
            # _init_state) -- cast to ssm_state's dtype (bf16 since the
            # backbone now loads in bf16, see model.py's load_base) so the
            # merge below doesn't hit a dtype mismatch.
            p, key, beta, clear = (t.to(ssm_state.dtype) for t in gated_delta)
            readback = torch.einsum("bhpn,bn->bhp", ssm_state, key)
            forgotten = ssm_state - beta * torch.einsum("bhp,bn->bhpn", readback, key)
            written = beta * torch.einsum("bhp,bn->bhpn", p, key)
            ssm_state = clear * forgotten + written

        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(mixer.D.to(dtype), "h -> h 1") * x_h
        y = rearrange(y, "b h p -> b (h p)")
        if not mixer.rmsnorm:
            y = y * mixer.act(z)
        if mixer.rmsnorm:
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

    def forward(self, input_ids: torch.Tensor, state: MemoryState | None = None) -> tuple[torch.Tensor, MemoryState]:
        """Returns (logits (B, T, vocab_size), updated MemoryState).

        Processes `input_ids` one token at a time regardless of T (see the
        module docstring for why) -- correct for both a many-token prefill
        and a single incremental decode step, just not parallelized across T.
        """
        batch_size, seqlen = input_ids.shape
        device = input_ids.device
        dtype = self.embedding.weight.dtype
        if state is None:
            state = self._init_state(batch_size, device, dtype)

        all_logits = []
        for t in range(seqlen):
            h = self.embedding(input_ids[:, t])
            residual = None
            token_betas, token_clears = [], []
            for i, layer in enumerate(self.layers):
                h, residual = self._prenorm(layer, h, residual)
                gated_delta = None
                if i in INJECTED_LAYERS:
                    gated_delta = self.injections[str(i)].signals(state.last_o_t, state.last_surprise)
                    beta, clear = gated_delta[2], gated_delta[3]
                    beta_val = beta.detach().mean().item()
                    clear_val = clear.detach().mean().item()
                    self._mem_stat_sums["beta"] += beta_val
                    self._mem_stat_sums["clear"] += clear_val
                    self._mem_stat_inj_count += 1
                    token_betas.append(beta_val)
                    token_clears.append(clear_val)
                h, conv_state, ssm_state = self._mixer_step(
                    layer.mixer, h, state.conv_states[i], state.ssm_states[i], gated_delta
                )
                state.conv_states[i] = conv_state
                state.ssm_states[i] = ssm_state

                if i == READ_LAYER:
                    o_t, surprise = self.front_end.step(residual, state.neural_memory)
                    state.last_o_t = o_t
                    state.last_surprise = surprise
                    surprise_val = surprise.detach().mean().item()
                    o_t_norm_val = o_t.detach().norm(dim=-1).mean().item()
                    self._mem_stat_sums["surprise"] += surprise_val
                    self._mem_stat_sums["o_t_norm"] += o_t_norm_val
                    self._mem_stat_tok_count += 1
                    self._last_token_log["surprise"] = surprise_val
                    self._last_token_log["o_t_norm"] = o_t_norm_val

            if token_betas:
                self._last_token_log["beta"] = sum(token_betas) / len(token_betas)
                self._last_token_log["clear"] = sum(token_clears) / len(token_clears)
                self._last_token_log["active_layers"] = sum(b > ACTIVE_BETA_THRESHOLD for b in token_betas)
                self._last_token_log["n_layers"] = len(token_betas)

            h = self._apply_norm_f(h, residual)
            all_logits.append(self.lm_head(h))

        return torch.stack(all_logits, dim=1), state

    def last_token_log(self) -> dict | None:
        """Snapshot of the most recently processed token's memory-usage
        signals (beta/clear averaged across injected layers, active_layers =
        how many of them are above ACTIVE_BETA_THRESHOLD -- diagnostic only,
        doesn't affect the merge itself; surprise/o_t_norm from the
        front-end's last read). None until forward() has run at least once.
        Unlike pop_memory_stats, this never resets -- it's meant for live
        per-token logging, not a per-step average."""
        return dict(self._last_token_log) if self._last_token_log else None

    def pop_memory_stats(self) -> dict[str, float] | None:
        """Returns averages since the last call (None if forward hasn't run
        since then) and resets the running sums. `beta`/`clear` are the most
        direct usage signal: both start pinned near their no-op init
        (beta~0.02, clear~0.98, see _GatedDeltaInjection.__init__) so the
        backbone behaves like the unmodified pretrained model until training
        moves them -- beta staying near 0.02 over many steps means the memory
        is still effectively disconnected, not actually writing into
        ssm_state, regardless of how much gradient the memory params receive
        in preflight."""
        if self._mem_stat_tok_count == 0:
            return None
        stats = {
            "beta": self._mem_stat_sums["beta"] / max(self._mem_stat_inj_count, 1),
            "clear": self._mem_stat_sums["clear"] / max(self._mem_stat_inj_count, 1),
            "surprise": self._mem_stat_sums["surprise"] / self._mem_stat_tok_count,
            "o_t_norm": self._mem_stat_sums["o_t_norm"] / self._mem_stat_tok_count,
        }
        self._mem_stat_sums = {k: 0.0 for k in self._mem_stat_sums}
        self._mem_stat_inj_count = 0
        self._mem_stat_tok_count = 0
        return stats


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/train.py and the backend
    registry; LoRA adapters themselves are attached separately by the
    caller (see sft/lora.py / backend/app/model/lora.py), after this.

    Loads in bf16, not fp32, matching the other Mamba2 models in this repo
    (see models/mamba2_780m/model.py's load_base) -- this 780M-parameter
    backbone is small enough that plain LoRA (no 4-bit quantization) fits
    this project's dev GPU (8GB) comfortably, unlike the 2.7B backbone this
    model used previously (see git history), which needed QLoRA to fit.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer))
    return model


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device)).to(device)
