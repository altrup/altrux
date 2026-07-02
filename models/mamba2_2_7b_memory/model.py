"""Mamba2-2.7B backbone (frozen) + a trainable long-term memory subsystem.

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

MODEL_ID = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
# The backbone is LoRA-adapted (frozen, full-precision in_proj/out_proj +
# trainable low-rank adapters) rather than left fully frozen, on the theory
# that a fully frozen backbone is unlikely to integrate a memory signal
# injected straight into its SSM state well. The 2.7B backbone in bf16 is
# ~5.4 GB -- comfortable on a cloud A100/H100 without 4-bit quantization, so
# no QUANTIZE_LORA_BASE flag here. The memory subsystem (front-end, gate
# projections) is separate from LoRA: it has no pretrained weights to adapt,
# so it trains with ordinary full-parameter gradients from a random init.
TARGET_LORA_MODULES: list[str] = ["in_proj", "out_proj"]

# Chat format role markers -- see models/mamba2_780m/model.py for the
# rationale; identical convention here.
USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN]

# Mamba2-2.7B backbone shape (state-spaces/mamba2-2.7b).
D_MODEL = 2560
N_LAYER = 64
NHEADS = 80
HEADDIM = 64
D_STATE = 128

# Memory subsystem hyperparameters (see README.md for the rationale).
# READ_LAYER sits at roughly 2/3 depth and INJECTED_LAYERS covers roughly
# the last two thirds of the stack on this backbone's 64 layers -- early
# layers produce weak keys (too little semantic content yet) while late layers
# are already collapsed toward next-token prediction, so the read/injection
# window sits in between.
READ_LAYER = 42
INJECTED_LAYERS: tuple[int, ...] = tuple(range(22, N_LAYER, 2))
BOTTLENECK_R = 128
MEM_DIM = D_MODEL
MEM_HIDDEN = 4 * D_MODEL
# Diagnostic-only threshold for counting an injected layer as "actively"
# writing into ssm_state this token (see Model.last_token_log). Measured as
# cosine similarity between ssm_state right before vs. right after the
# gated-delta merge (_mixer_step) -- a high beta gate alone doesn't mean the
# merge actually moved the state (see mamba2_780m_memory/model.py's
# ACTIVE_COS_SIM_THRESHOLD for the real-run observation that motivated
# this: beta staying near a fixed non-trivial value while the memory's own
# output o_t was still tiny, i.e. the gate was open but there was nothing
# substantial to inject), so this measures the actual effect on ssm_state
# directly rather than trusting the gate value as a proxy for it. Below
# this threshold counts as "active"; arccos(0.995) ~= 5.7 degrees of
# rotation in ssm_state's (nheads*headdim*d_state)-dim space, picked as a
# starting guess for "a real, not negligible, shift" -- unverified against
# real training telemetry, watch active_layers/n_layers in the live logs
# and retune if it's saturating at 0 or n_layers either way. Diagnostic
# only -- the gated-delta merge in _mixer_step always uses the raw,
# continuous beta/retain values regardless of this threshold.
ACTIVE_COS_SIM_THRESHOLD = 0.995

# beta's startup suppression (see _GatedDeltaInjection) is a fixed additive
# offset on the gate logit, linearly annealed from BETA_BIAS_ANNEAL_START to
# 0 over the first BETA_BIAS_ANNEAL_TOKENS tokens of training, then held at
# 0 forever after -- a pure function of cumulative training tokens (see
# Model.set_beta_anneal), not a counter of its own, so it survives resume
# without extra checkpoint state. This anneals away from -3 rather than
# disabling the suppression outright: beta_proj's own *bias* is left at its
# untouched nn.Linear default (near 0) so gradient can shape it from the
# start, while this offset -- not the learnable bias -- is what keeps the
# gated-delta merge close to a no-op at init.
BETA_BIAS_ANNEAL_START = -3.0
BETA_BIAS_ANNEAL_TOKENS = 2000

# Soft ceiling (see _NeuralMemory.write) for the raw per-token gradient norm
# that drives the memory's test-time write -- capping eta below 1 (see
# _TitansFrontEnd.step) bounds the momentum recurrence *given* a bounded
# per-token gradient, but doesn't bound the gradient itself, which can spike
# hard enough (e.g. early in training, before eta/theta are learned) to push
# the memory non-finite. A naive dimensionality estimate (treating each of
# w1/w2's ~52M elements as an independent O(1) contribution, giving a
# combined norm around sqrt(52M)~=7211) is wrong: L is a *mean* over
# MEM_DIM, not a sum, so the gradient is attenuated by that factor too, not
# just the loss -- a numeric check (dL/dw1, dL/dw2 computed directly at
# w1/w2's init scale, MEM_DIM=2560, MEM_HIDDEN=10240) gives a combined norm
# of ~2, not ~7000, and comes out nearly identical to the 780m model's
# check despite the larger dims (see mamba2_780m_memory/model.py's
# GRAD_SCALE for that derivation). The blowup itself is quadratic in how
# far w2 has drifted from that scale (dL/dw1 chains through r @ w2, so
# residual and weight scale both grow together) -- by 100x drift the same
# check gives a combined norm of ~2754, by 300x it's ~24765, both close to
# the 780m case too. This is set to intervene well before that drift
# compounds too far, while leaving ~150x headroom over the healthy baseline
# for a genuinely large, real surprise. Still unverified against real
# training telemetry -- watch GRAD_NORM in the live logs (see
# Model.last_token_log) and retune from there.
GRAD_SCALE = 300.0


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

    def write(self, k: torch.Tensor, v: torch.Tensor, eta: torch.Tensor, theta: torch.Tensor, alpha: torch.Tensor, create_graph: bool = True):
        """One test-time gradient step on L = mean((M(k) - v)^2) -- mean, not
        sum, over MEM_DIM dims: since k/v are RMS-normalized to unit scale
        (see _rms_normalize), a sum over MEM_DIM=2560 dims of O(1) per-dim
        terms would land in the hundreds-to-thousands by dimensionality
        alone, before anything is actually wrong with the prediction --
        inflating both the loss and, via the ~52M-element w1/w2 it
        backprops through, the resulting gradient norm. Mean reduction keeps
        L (and `surprise`, the per-injected-layer write gate in
        _GatedDeltaInjection.signals) at the O(1) per-dim scale its
        consumers already assume.

        First-order (truncated) approximation: `params`/`momentum` carried in from the
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

        Returns (surprise, grad_norm): surprise is the loss magnitude per
        batch element, exported for Stage 2's write-strength gate; grad_norm
        is the raw (pre-soft-clip) combined gradient norm per batch element,
        exported purely for live diagnostics (see Model.last_token_log) --
        it plays no role in the write itself.
        """
        params = [p.detach().requires_grad_(True) for p in (self.w1, self.b1, self.w2, self.b2)]
        momentum = [s.detach() for s in self.momentum]

        pred = self._apply(k, *params)
        per_example_loss = ((pred - v) ** 2).mean(dim=-1)
        # create_graph=True: g must stay differentiable w.r.t. k/v (and the
        # freshly-detached `params` above) so k_proj/v_proj actually receive
        # gradient -- with create_graph=False, g is a plain non-differentiable
        # number, severing k_proj/v_proj from the outer loss entirely
        # (confirmed: this was happening). This does NOT reintroduce the O(T)
        # blowup -- that came from *not* detaching params/momentum per token,
        # not from this flag; `params` being a fresh leaf each call means
        # differentiating g w.r.t. it can't chain past this single token.
        grads = torch.autograd.grad(per_example_loss.sum(), params, create_graph=create_graph)

        # Soft-clip: treat (w1,b1,w2,b2)'s gradients as one combined vector
        # per batch row (same convention as torch.nn.utils.clip_grad_norm_)
        # and rescale that vector so its norm smoothly saturates toward
        # GRAD_SCALE instead of being left unbounded -- tanh(x)~=x near 0,
        # so a normal, healthy gradient passes through essentially
        # unchanged; only a gradient large enough to threaten non-finite
        # state gets pulled down, and direction is always preserved exactly
        # (the whole vector is scaled by one factor, not clipped
        # element-by-element). The factor is computed from a *detached* copy
        # of the gradient -- deliberately not differentiated through, same
        # as ordinary gradient clipping -- so it doesn't add a second-order
        # term to the k_proj/v_proj gradient create_graph exists for.
        grad_norm = torch.zeros(k.shape[0], device=k.device, dtype=k.dtype)
        for g in grads:
            grad_norm = grad_norm + g.detach().pow(2).flatten(1).sum(dim=1)
        grad_norm = grad_norm.sqrt()
        clip_factor = (GRAD_SCALE * torch.tanh(grad_norm / GRAD_SCALE)) / (grad_norm + 1e-12)
        grads = [g * clip_factor.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]

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
        return per_example_loss.detach(), grad_norm.detach()

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

    def step(self, residual: torch.Tensor, memory: _NeuralMemory, create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """residual: (batch, d_model) residual stream entering READ_LAYER.

        Returns (o_t, surprise, grad_norm) -- o_t is (batch, mem_dim),
        surprise and grad_norm are both (batch,).

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
            # to ~50-60 in the 780m variant), which made M's write loss
            # ||M(k) - v||^2 -- and its gradient -- scale with that raw,
            # unbounded magnitude rather than with anything learned.
            # Normalizing bounds the input magnitude at the source, so
            # theta/eta (rather than k/v's raw scale) control how strongly a
            # given token's write lands.
            q = _rms_normalize(self.q_proj(residual))
            k = _rms_normalize(self.k_proj(residual))
            v = _rms_normalize(self.v_proj(residual))
            knobs = torch.sigmoid(self.knob_proj(residual))
            # eta capped at 0.9 (not left at sigmoid's full (0, 1) range like
            # theta/alpha's caps, which bound them *small* on purpose --
            # small theta/alpha is what makes the write step gentle and the
            # memory's content persist across a long document): S_t = eta *
            # S_{t-1} - theta * g_t accumulates with essentially no decay
            # when eta is allowed to approach 1, turning the momentum into an
            # undamped running sum of every token's write. Capping below 1
            # guarantees at least 10% decay per token.
            eta, theta, alpha = knobs[..., 0] * 0.9, knobs[..., 1] * 0.1, knobs[..., 2] * 0.1
            surprise, grad_norm = memory.write(k, v, eta, theta, alpha, create_graph=create_graph)
            o_t = memory.read(q)
        return o_t, surprise, grad_norm


class _GatedDeltaInjection(nn.Module):
    """One per injected layer. Derives this layer's own gated-delta write
    signals (p, key, beta, retain) from the shared o_t through a private
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
        self.retain_proj = nn.Linear(r, 1)
        # retain starts near 1 (sigmoid(4) ~ 0.98) so the gated-delta merge's
        # global wipe never forces decay on its own -- this one is a true
        # fixed-forever init, not annealed.
        #
        # beta starts near 0 too, but via beta_anneal_offset (see
        # BETA_BIAS_ANNEAL_START/_TOKENS above and Model.set_beta_anneal),
        # not via beta_proj.bias itself: an earlier version pinned
        # beta_proj.bias to -4 permanently, which kept the merge a no-op at
        # init (matching the unmodified pretrained backbone until the gate
        # learned otherwise) but also saturated the gate's own gradient
        # (sigmoid'(-4) ~ 0.02x the gradient at 0), and in practice beta
        # never woke up over a full run. beta_proj.bias is left at its
        # ordinary nn.Linear default (small, ~0) so it gets full gradient
        # signal from step 0; beta_anneal_offset supplies the near-no-op
        # suppression instead, as a non-learnable term that decays away
        # rather than one the optimizer has to fight through saturation to
        # move.
        nn.init.constant_(self.retain_proj.bias, 4.0)
        # Defaults to 0 (i.e. *no* suppression), not BETA_BIAS_ANNEAL_START --
        # this is a plain Python float, not a buffer/parameter, so it is
        # never part of state_dict and is NOT restored by checkpoint
        # loading (see load_checkpoint in backend/app/model/lora.py, which
        # loads only named_parameters with requires_grad). Inference
        # (load_inference) never calls Model.set_beta_anneal, so it must
        # default to the *post-anneal* state (0) -- the state every
        # checkpoint actually represents once training has run past
        # BETA_BIAS_ANNEAL_TOKENS. Training itself overrides this to the
        # correct in-progress value immediately via the one-time
        # pre-training-loop on_step call in sft/train.py, before this would
        # otherwise matter.
        self.beta_anneal_offset = 0.0

    def signals(self, o_t: torch.Tensor, surprise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, key, beta, retain) for this layer's gated-delta merge.

        p: (batch, nheads, headdim) per-head value to write.
        key: (batch, d_state) shared write address.
        beta, retain: (batch, 1, 1, 1) gates, broadcastable against ssm_state.
        """
        z = self.down(o_t)
        p = rearrange(self.value_proj(z), "b (h p) -> b h p", p=self.headdim)
        key = self.key_proj(z)
        # surprise is already a mean-squared-error-per-dim (see
        # _NeuralMemory.write) -- an O(1) quantity -- so it can be squashed
        # with its own sigmoid directly, bounded to [0.5, 1) since surprise
        # >= 0, before adding into beta's logit; that bound keeps its
        # contribution small enough that beta_proj's near-0 bias can still
        # let the anneal offset dominate by default, preserving the
        # no-op-at-init behavior.
        surprise_signal = torch.sigmoid(surprise)
        beta = torch.sigmoid(
            self.beta_proj(z).squeeze(-1) + surprise_signal + self.beta_anneal_offset
        ).view(-1, 1, 1, 1)
        retain = torch.sigmoid(self.retain_proj(z)).view(-1, 1, 1, 1)
        return p, key, beta, retain


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
    """LoRA-adapted Mamba2-2.7B backbone with the memory subsystem spliced in.

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
            # those trainable; freeze the rest of the backbone.
            p.requires_grad_("lora_A" in name or "lora_B" in name)
        for p in self.norm_f.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        for layer in self.layers:
            assert layer.mixer.ngroups == 1, "memory injection assumes ngroups=1"

        self.front_end = _TitansFrontEnd()
        self.injections = nn.ModuleDict({str(i): _GatedDeltaInjection() for i in INJECTED_LAYERS})
        # Running sums for pop_memory_stats() -- accumulated as detached
        # tensors (all inputs are already .detach()'d at the accumulation
        # sites, so this never holds a reference into any autograd graph)
        # and only converted to Python floats once, in pop_memory_stats
        # itself. Calling .item() at every accumulation instead (the
        # original approach) forces a blocking GPU sync per injected layer
        # per token, which serializes the whole per-token training loop --
        # far more costly than the sync-free tensor accumulation here.
        # beta/retain are accumulated once per injected layer per token;
        # surprise/o_t_norm once per token (at READ_LAYER only). See
        # pop_memory_stats for why these matter: beta/retain start near 0/1
        # (no-op init, see _GatedDeltaInjection.__init__) and are the only
        # direct signal of whether the memory subsystem is actually being
        # used or still sitting at its identity init.
        self._mem_stat_sums = {"beta": 0.0, "retain": 0.0, "surprise": 0.0, "o_t_norm": 0.0, "grad_norm": 0.0}
        self._mem_stat_inj_count = 0
        self._mem_stat_tok_count = 0
        # Per-slot snapshot of the most-recently-processed token's memory
        # signals, for live per-slot logging (overwritten every token).
        # List of dicts, one per batch element -- see last_token_log().
        self._last_token_logs: list[dict] = []

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
        gated-delta memory merge (`gated_delta = (p, key, beta, retain)`) can
        be applied directly to `ssm_state` immediately after Mamba2's own
        decay+write and before the C readout -- the merged state is what
        gets both read out *and* persisted, so memory content now decays
        under Mamba2's own `A` like everything else in the state, rather
        than living in a separate accumulator. `ssm_state` is *not* mutated
        in place (unlike the library's decode cache): it's rebound to a
        fresh tensor each call so gradients flow through the whole sequence
        during training.

        Returns (out, conv_state, ssm_state, injection_cos_sim) --
        injection_cos_sim is None when gated_delta is None, else a (batch,)
        tensor: cosine similarity between ssm_state right before and right
        after the gated-delta merge, for live diagnostics only (see
        ACTIVE_COS_SIM_THRESHOLD).
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

        injection_cos_sim = None
        if gated_delta is not None:
            # gated_delta's tensors come from the memory subsystem, which runs
            # at its own (fp32) dtype regardless of the backbone's (see
            # _init_state) -- cast to ssm_state's dtype (bf16 since the
            # backbone now loads in bf16, see model.py's load_base) so the
            # merge below doesn't hit a dtype mismatch.
            p, key, beta, retain = (t.to(ssm_state.dtype) for t in gated_delta)
            pre_merge_state = ssm_state
            readback = torch.einsum("bhpn,bn->bhp", ssm_state, key)
            forgotten = ssm_state - beta * torch.einsum("bhp,bn->bhpn", readback, key)
            written = beta * torch.einsum("bhp,bn->bhpn", p, key)
            ssm_state = retain * forgotten + written
            # Diagnostic only (see ACTIVE_COS_SIM_THRESHOLD/last_token_log) --
            # detached, no autograd graph, negligible cost next to the merge
            # itself. (batch,), one value per batch row.
            injection_cos_sim = F.cosine_similarity(
                pre_merge_state.detach().flatten(1), ssm_state.detach().flatten(1), dim=1
            )

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
        return out, conv_state, ssm_state, injection_cos_sim

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
            token_betas: list[torch.Tensor] = []   # (B,) per injected layer
            token_retains: list[torch.Tensor] = []
            token_cos_sims: list[torch.Tensor] = []
            last_surprise_per_slot: torch.Tensor | None = None
            last_o_t_norm_per_slot: torch.Tensor | None = None
            last_grad_norm_per_slot: torch.Tensor | None = None
            for i, layer in enumerate(self.layers):
                h, residual = self._prenorm(layer, h, residual)
                gated_delta = None
                if i in INJECTED_LAYERS:
                    gated_delta = self.injections[str(i)].signals(state.last_o_t, state.last_surprise)
                    beta, retain = gated_delta[2], gated_delta[3]
                    beta_per_slot = beta[:, 0, 0, 0].detach()    # (B,)
                    retain_per_slot = retain[:, 0, 0, 0].detach()  # (B,)
                    self._mem_stat_sums["beta"] += beta_per_slot.mean()
                    self._mem_stat_sums["retain"] += retain_per_slot.mean()
                    self._mem_stat_inj_count += 1
                    token_betas.append(beta_per_slot)
                    token_retains.append(retain_per_slot)
                h, conv_state, ssm_state, injection_cos_sim = self._mixer_step(
                    layer.mixer, h, state.conv_states[i], state.ssm_states[i], gated_delta
                )
                state.conv_states[i] = conv_state
                state.ssm_states[i] = ssm_state
                if injection_cos_sim is not None:
                    token_cos_sims.append(injection_cos_sim)

                if i == READ_LAYER:
                    o_t, surprise, grad_norm = self.front_end.step(residual, state.neural_memory)
                    state.last_o_t = o_t
                    state.last_surprise = surprise
                    last_surprise_per_slot = surprise.detach()          # (B,)
                    last_o_t_norm_per_slot = o_t.detach().norm(dim=-1)  # (B,)
                    last_grad_norm_per_slot = grad_norm.detach()        # (B,)
                    self._mem_stat_sums["surprise"] += last_surprise_per_slot.mean()
                    self._mem_stat_sums["o_t_norm"] += last_o_t_norm_per_slot.mean()
                    self._mem_stat_sums["grad_norm"] += last_grad_norm_per_slot.mean()
                    self._mem_stat_tok_count += 1

            # Only materialize the per-slot log dict (hundreds of blocking
            # .item() syncs, one per injected layer per batch slot) on the
            # last token of this chunk -- _last_token_logs is overwritten
            # every token and only the final write is ever read (see
            # last_token_log()), so doing this on every token was paying
            # a full host sync ~seqlen times over for one that's read once.
            if token_betas and t == seqlen - 1:
                logs = []
                for b in range(batch_size):
                    betas_b = [tb[b].item() for tb in token_betas]
                    retains_b = [tr[b].item() for tr in token_retains]
                    cos_sims_b = [tc[b].item() for tc in token_cos_sims]
                    entry: dict = {
                        "beta": sum(betas_b) / len(betas_b),
                        "retain": sum(retains_b) / len(retains_b),
                        "active_layers": sum(cs < ACTIVE_COS_SIM_THRESHOLD for cs in cos_sims_b),
                        "n_layers": len(betas_b),
                        # Lowest (i.e. most-changed) per-layer cosine
                        # similarity this token for this slot -- the single
                        # most active layer's actual shift, a finer-grained
                        # companion to the active_layers/n_layers count.
                        "min_cos_sim": min(cos_sims_b),
                    }
                    if last_surprise_per_slot is not None:
                        entry["surprise"] = last_surprise_per_slot[b].item()
                        entry["o_t_norm"] = last_o_t_norm_per_slot[b].item()
                        entry["grad_norm"] = last_grad_norm_per_slot[b].item()
                    logs.append(entry)
                self._last_token_logs = logs

            h = self._apply_norm_f(h, residual)
            all_logits.append(self.lm_head(h))

        return torch.stack(all_logits, dim=1), state

    def set_beta_anneal(self, total_tokens: float) -> None:
        """Updates every injected layer's beta_anneal_offset for the given
        cumulative training-token count (see BETA_BIAS_ANNEAL_START/_TOKENS).
        A pure function of total_tokens -- not a counter incremented here --
        so calling this with a resumed run's total_tokens lands the offset
        exactly where it would have been had training never stopped, with no
        extra checkpoint state needed."""
        frac = min(total_tokens / BETA_BIAS_ANNEAL_TOKENS, 1.0)
        offset = BETA_BIAS_ANNEAL_START * (1.0 - frac)
        for injection in self.injections.values():
            injection.beta_anneal_offset = offset

    def last_token_log(self) -> list[dict] | None:
        """Per-slot snapshot of the most recently processed token's memory
        signals. Returns a list with one dict per batch element (beta/retain
        averaged across injected layers, active_layers = count of layers
        whose ssm_state moved past ACTIVE_COS_SIM_THRESHOLD during the
        gated-delta merge, min_cos_sim = that slot's single most-changed
        layer's actual similarity, surprise/o_t_norm from the front-end
        read). None until forward() has run at least once. Never resets --
        meant for live per-token logging, not a per-step average."""
        return list(self._last_token_logs) if self._last_token_logs else None

    def reset_slot(self, state: "MemoryState", slot_idx: int) -> None:
        """Reset slot slot_idx to a fresh random init in-place, leaving all
        other slots unchanged. Call only on a detached state."""
        device = state.last_o_t.device
        conv_dtype = state.conv_states[0].dtype
        mem_dtype = self.front_end.q_proj.weight.dtype

        for i, layer in enumerate(self.layers):
            fresh_conv, fresh_ssm = layer.mixer.allocate_inference_cache(1, 1, dtype=conv_dtype)
            state.conv_states[i][slot_idx].copy_(fresh_conv[0])
            state.ssm_states[i][slot_idx].copy_(fresh_ssm[0])

        nm = state.neural_memory
        bound1 = 1.0 / math.sqrt(MEM_DIM)
        bound2 = 1.0 / math.sqrt(MEM_HIDDEN)
        nm.w1[slot_idx].copy_((torch.rand(MEM_HIDDEN, MEM_DIM, device=device, dtype=mem_dtype) * 2 - 1) * bound1)
        nm.b1[slot_idx].zero_()
        nm.w2[slot_idx].copy_((torch.rand(MEM_DIM, MEM_HIDDEN, device=device, dtype=mem_dtype) * 2 - 1) * bound2)
        nm.b2[slot_idx].zero_()
        for s in nm.momentum:
            s[slot_idx].zero_()
        state.last_o_t[slot_idx].zero_()
        state.last_surprise[slot_idx].zero_()

    def pop_memory_stats(self) -> dict[str, float] | None:
        """Returns averages since the last call (None if forward hasn't run
        since then) and resets the running sums. `beta`/`retain` are the most
        direct usage signal: both start near their no-op init (beta~0.05 via
        beta_anneal_offset, retain~0.98, see _GatedDeltaInjection.__init__) so
        the backbone behaves like the unmodified pretrained model until
        training moves them -- beta staying near its init over many steps
        (well past BETA_BIAS_ANNEAL_TOKENS, once the anneal offset is gone)
        means the memory is still effectively disconnected, not actually
        writing into ssm_state, regardless of how much gradient the memory
        params receive in preflight."""
        if self._mem_stat_tok_count == 0:
            return None
        # Single sync here for the whole accumulation window, instead of one
        # per injected layer per token (see the accumulation sites above).
        stats = {
            "beta": (self._mem_stat_sums["beta"] / max(self._mem_stat_inj_count, 1)).item(),
            "retain": (self._mem_stat_sums["retain"] / max(self._mem_stat_inj_count, 1)).item(),
            "surprise": (self._mem_stat_sums["surprise"] / self._mem_stat_tok_count).item(),
            "o_t_norm": (self._mem_stat_sums["o_t_norm"] / self._mem_stat_tok_count).item(),
            "grad_norm": (self._mem_stat_sums["grad_norm"] / self._mem_stat_tok_count).item(),
        }
        self._mem_stat_sums = {k: 0.0 for k in self._mem_stat_sums}
        self._mem_stat_inj_count = 0
        self._mem_stat_tok_count = 0
        return stats


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/train.py and the backend
    registry; LoRA adapters themselves are attached separately by the
    caller (see sft/lora.py / backend/app/model/lora.py), after this.

    Loads in bf16. The 2.7B backbone in bf16 is ~5.4 GB -- comfortable on a
    cloud A100/H100 without 4-bit quantization.
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
