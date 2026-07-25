"""Mamba2-2.7B backbone (frozen) + a trainable long-term memory subsystem.

The memory is a single Titans-style fast-weight MLP that is test-time-trained,
feeding a per-layer gated-delta rule that's merged directly into a sparse
subset of the backbone's own SSM states (not a separate accumulator -- see
`Model._mixer_step`). See README.md for the full design write-up; this module
is the literal implementation of it.

Implementation note: because the gated-delta merge at layer i is a
function of the memory read at READ_LAYER, and that read for token t must be
available to *later* layers of the *same* token while only being available to
*earlier* layers on the *next* token, per-token injection can't run through
Mamba2's fused/chunked parallel-scan kernels -- those process a whole
sequence in one kernel call and don't expose a per-token, pre-readout hook.
But since injection only fires on a memory-window's closing token (see
`set_memory_window`), only that one token per window still needs the manual,
sequential treatment (`_mixer_step`, replicating Mamba2's own incremental-
decode arithmetic); the `window - 1` other tokens per window never touch the
memory subsystem at all, so they're free to run through the native fused/
chunked kernel (`_mixer_span`, wrapping `causal_conv1d_fn` +
`mamba_chunk_scan_combined`) instead. `Model.forward` dispatches between the
two on hardware where the fused kernel is available (CUDA, not ROCm, with
`causal_conv1d` importable -- see `_fused_path_available`); on hardware where
it isn't (this repo's local ROCm dev box, gfx1102 -- see root CLAUDE.md), it
falls back to `_forward_manual`, the original all-manual per-token loop,
unconditionally and unchanged. See
docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md for the
full design, including what's still deferred and why.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

from ..common import MarkerDelta
# Importing this is always safe even where causal_conv1d/Triton are broken
# (this repo's ROCm dev box, see root CLAUDE.md) -- it only hangs/segfaults
# if actually *called*, and `_fused_path_available` gates every call site.
# ssd_combined.py itself guards its own `from causal_conv1d import
# causal_conv1d_fn` the same try/except way this module does.
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

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
# merge actually moved the state (motivated by a real-run observation:
# beta staying near a fixed non-trivial value while the memory's own
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
# 0 over the first BETA_BIAS_ANNEAL_STEPS optimizer steps, then held at
# 0 forever after -- a pure function of the global step count (see
# Model.set_beta_anneal), not a counter of its own, so it survives resume
# without extra checkpoint state. Step-keyed rather than token-keyed so it
# scales with how many actual gradient updates the gate has received,
# independent of --chunk-len/--accum-tokens/--memory-window (all of which
# change how many tokens one step represents). This anneals away from -3
# rather than disabling the suppression outright: beta_proj's own *bias* is
# left at its untouched nn.Linear default (near 0) so gradient can shape it
# from the start, while this offset -- not the learnable bias -- is what
# keeps the gated-delta merge close to a no-op at init.
BETA_BIAS_ANNEAL_START = -3.0
BETA_BIAS_ANNEAL_STEPS = 32

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
# of ~2, not ~7000. The blowup itself is quadratic in how
# far w2 has drifted from that scale (dL/dw1 chains through r @ w2, so
# residual and weight scale both grow together) -- by 100x drift the same
# check gives a combined norm of ~2754, by 300x it's ~24765. This is set to
# intervene well before that drift compounds too far, while leaving ~150x
# headroom over the healthy baseline
# for a genuinely large, real surprise. Still unverified against real
# training telemetry -- watch GRAD_NORM in the live logs (see
# Model.last_token_log) and retune from there.
GRAD_SCALE = 300.0

# Hard ceiling on alpha, the write's global weight decay on M (the sigmoid
# multiplier in _TitansFrontEnd.observe). This is a prior, not something the
# model can learn: BPTT truncates at --chunk-len, so alpha's projection only
# ever receives short-horizon gradient ("decay now -> cleaner reads within
# this chunk") while the compounded long-horizon cost of erosion -- content
# decayed to nothing thousands of windows before the read that needed it --
# is invisible to it. Left at sigmoid's natural 0.1-scale cap, a real run
# walked alpha up ~20x in an hour and eroded ~80% of M's content within one
# long example (notes/EXPERIMENT_NOTES.md, 2026-07-17). At 1e-4 the
# worst-case half-life is ~7k writes (~55k tokens at memory-window 8) --
# sustained choice still compounds into era-level forgetting, but a fast
# wipe is out of reach. Targeted forgetting is the delta-overwrite path
# (write a new value at a key), not alpha. See
# docs/superpowers/specs/2026-07-17-episodic-chains-design.md.
ALPHA_CAP = 1e-4


_NONFINITE_DUMP_LIMIT = 5


def _dump_nonfinite_write(nm: "_NeuralMemory", ks, vs, etas, thetas, alphas, pred, per_token_loss) -> None:
    """Prints per-slot norms of every write() input and of M's weights the
    first few times the write loss goes non-finite, to localize whether the
    explosion arrives via the write targets (vs, i.e. the residual stream),
    the keys, or M's own weights. Capped so a persistent failure can't spam
    the log."""
    global _NONFINITE_DUMP_LIMIT
    if _NONFINITE_DUMP_LIMIT <= 0:
        return
    _NONFINITE_DUMP_LIMIT -= 1
    bad = (~per_token_loss.detach().isfinite()).any(dim=0).nonzero().flatten().tolist()
    print(f"  [nonfinite-write] loss non-finite for slots {bad}")
    for b in bad:
        print(
            f"  [nonfinite-write] slot {b}:"
            f" k_norm {ks.detach()[:, b].norm().item():.4g}"
            f" v_norm {vs.detach()[:, b].norm().item():.4g}"
            f" pred_norm {pred.detach()[:, b].norm().item():.4g}"
            f" eta {etas.detach()[:, b].mean().item():.4g}"
            f" theta {thetas.detach()[:, b].mean().item():.4g}"
            f" alpha {alphas.detach()[:, b].mean().item():.4g}"
            f" w1_absmax {nm.w1[b].detach().abs().max().item():.4g}"
            f" w2_absmax {nm.w2[b].detach().abs().max().item():.4g}"
            f" mom_absmax {max(s[b].detach().abs().max().item() for s in nm.momentum):.4g}"
        )


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
        # Snapshot of the random init, so the w*_drift stats (see
        # Model._accum_write_stats) can measure how far writes have
        # cumulatively moved M within the current example.
        self.w1_init = self.w1.clone()
        self.w2_init = self.w2.clone()

    @staticmethod
    def _apply(x: torch.Tensor, w1, b1, w2, b2) -> torch.Tensor:
        h = torch.tanh(torch.einsum("bhd,bd->bh", w1, x) + b1)
        return torch.einsum("bdh,bh->bd", w2, h) + b2

    @staticmethod
    def _apply_windowed(x_win: torch.Tensor, w1, b1, w2, b2) -> torch.Tensor:
        """Same computation as _apply, batched over an extra leading window
        dim: x_win is (W, batch, dim). w1/w2 are still (batch, hidden, dim)/
        (batch, dim, hidden) -- one M per batch row, shared/broadcast across
        the window dim (not one M per window position)."""
        h = torch.tanh(torch.einsum("bhd,wbd->wbh", w1, x_win) + b1)
        return torch.einsum("bdh,wbh->wbd", w2, h) + b2

    def surprise(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Per-token loss ||M(k) - v||^2 (mean over MEM_DIM, see write()'s
        docstring for why mean not sum) against the CURRENT weights, with no
        gradient step taken -- this is what lets every token get a fresh
        surprise/read signal even though the weight update itself only
        happens once per memory-window (see write() and Model.forward).
        Deliberately not detached: this feeds Stage 2's beta gate
        (_GatedDeltaInjection.signals), so gradient needs to flow from the
        outer loss back through here into self.w1/w2 -- i.e. into whichever
        window's write() call produced the weights currently in use -- and
        from there into k_proj/v_proj, same as read()."""
        pred = self._apply(k, self.w1, self.b1, self.w2, self.b2)
        return ((pred - v) ** 2).mean(dim=-1)

    def write(self, ks: torch.Tensor, vs: torch.Tensor, etas: torch.Tensor, thetas: torch.Tensor, alphas: torch.Tensor, create_graph: bool = True):
        """One test-time gradient step, consolidated over a window of W>=1
        tokens: ks/vs/etas/thetas/alphas are (W, ...) stacks -- one entry per
        token in the window, all computed against this SAME (frozen-for-the-
        window) M. This is the Titans paper's own chunk-size-b formulation
        (b>=1, Section 3.2 of arXiv:2501.00663): W=1 reduces exactly to a
        plain per-token update, not a separate code path, which is what
        keeps training (large W, for throughput) and inference (W=1, called
        every generated token -- see Model.forward) consistent with each
        other.

        Loss is summed over both the window and the batch before the single
        autograd.grad call, so W tokens' worth of gradient signal lands in
        one update instead of W sequential ones -- the approximation this
        trades away is that every token in the window computes its own loss
        against the window-START M rather than a continuously-updated one
        (see the design spec, docs/superpowers/specs/2026-07-02-chunked-
        memory-injection-design.md, for the full reasoning).
        eta/theta/alpha are averaged across the window for this one
        momentum/decay step -- at W=1 this is just that token's own value,
        identical to the un-windowed case.

        `params`/`momentum` are still fully detached and re-leafed once per
        *window* (not per token) before the gradient step, bounding the
        backward graph to O(1) windows of chain depth rather than O(T)
        tokens -- without this, chain depth would grow with total tokens
        processed and the backward graph would never be freed. `create_graph`
        and the GRAD_SCALE soft-clip below still operate per window: `eta`
        caps the momentum recurrence's decay but doesn't bound the gradient
        that feeds it, which can spike large enough (especially early in
        training, before eta/theta are learned) to push the memory
        non-finite in a single step regardless of eta.

        Returns (per_token_losses, grad_norm): per_token_losses is (W,
        batch) -- NOT reduced across the window, kept for diagnostics parity
        only (surprise() is what actually supplies Stage 2's per-token gate
        signal, independent of this method). grad_norm is (batch,), the one
        gradient norm for this window's single step.
        """
        params = [p.detach().requires_grad_(True) for p in (self.w1, self.b1, self.w2, self.b2)]
        momentum = [s.detach() for s in self.momentum]

        pred = self._apply_windowed(ks, *params)
        per_token_loss = ((pred - vs) ** 2).mean(dim=-1)  # (W, batch)
        if not per_token_loss.detach().isfinite().all():
            _dump_nonfinite_write(self, ks, vs, etas, thetas, alphas, pred, per_token_loss)
        # create_graph=True: g must stay differentiable w.r.t. params so
        # k_proj/v_proj/knob_proj receive gradient from the outer loss.
        grads = torch.autograd.grad(per_token_loss.sum(), params, create_graph=create_graph)

        # Soft-clip g's combined norm across (w1,b1,w2,b2) with tanh, from a
        # detached copy so it doesn't add a second-order term to the
        # create_graph=True path k_proj/v_proj's gradient depends on. Chosen
        # over a hard clip to preserve direction exactly, pass magnitude
        # through unchanged for typical gradients (tanh(x)~=x near 0), and
        # avoid the zero-gradient plateau a hard clip leaves above threshold.
        # GRAD_SCALE is a dimensional-analysis estimate, not a measured one:
        # w1/w2 alone are large enough that a healthy O(1)-per-element
        # gradient already has a combined norm around sqrt(element count)
        # from dimensionality alone -- clipping near that baseline would
        # constantly saturate on ordinary gradients, so GRAD_SCALE is set
        # with headroom above it. grad_norm here is the whole window's
        # combined gradient (one step per window), not a single token's.
        grad_norm = torch.zeros(ks.shape[1], device=ks.device, dtype=ks.dtype)
        for g in grads:
            grad_norm = grad_norm + g.detach().pow(2).flatten(1).sum(dim=1)
        grad_norm = grad_norm.sqrt()
        clip_factor = (GRAD_SCALE * torch.tanh(grad_norm / GRAD_SCALE)) / (grad_norm + 1e-12)
        grads = [g * clip_factor.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]

        eta = etas.mean(dim=0)
        theta = thetas.mean(dim=0)
        alpha = alphas.mean(dim=0)

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
        return per_token_loss.detach(), grad_norm.detach()

    def read(self, q: torch.Tensor) -> torch.Tensor:
        """o_t = M(q_t) against the CURRENT weights -- may be the
        window-frozen snapshot from before this token (most tokens in a
        window) or the just-updated one (the token whose write() call closed
        the window), depending on call order in Model.forward."""
        return self._apply(q, self.w1, self.b1, self.w2, self.b2)

    def read_windowed(self, q_win: torch.Tensor) -> torch.Tensor:
        """Same as read(), batched over an extra leading window dim: q_win
        is (W, batch, dim); returns (W, batch, dim). Always against the
        SAME window-start weights for every entry (`_apply_windowed`
        broadcasts w1/w2 across the W dim, same as write()), matching the
        per-token read()'s "frozen for the window" semantics exactly --
        used by `Model._forward_fused`'s per-window dispatch in place of W
        sequential read() calls."""
        return self._apply_windowed(q_win, self.w1, self.b1, self.w2, self.b2)

    def surprise_windowed(self, k_win: torch.Tensor, v_win: torch.Tensor) -> torch.Tensor:
        """Same as surprise(), batched over an extra leading window dim:
        k_win/v_win are (W, batch, dim); returns (W, batch). See
        read_windowed's docstring."""
        pred = self._apply_windowed(k_win, self.w1, self.b1, self.w2, self.b2)
        return ((pred - v_win) ** 2).mean(dim=-1)


def fused_kernel_usable(device: torch.device) -> bool:
    """True when `device` can safely run `Model._mixer_span` (mamba_ssm's
    native fused/chunked kernel path) instead of the manual per-token loop:
    `causal_conv1d` imported successfully AND `device` is a real CUDA device
    that isn't actually a ROCm/HIP build. `torch.cuda.is_available()` also
    reports True on ROCm, so `torch.version.hip is None` is the piece that
    actually distinguishes them -- same check `backend/Makefile`/
    `sft/Makefile` use to decide whether to install `causal-conv1d` at all.
    Module-level (not a `Model` method) so it's testable without
    constructing the real 2.7B backbone, which OOMs on this repo's local
    8GB dev GPU regardless (see models/tests/test_mamba2_2_7b_memory_
    windowing.py)."""
    if causal_conv1d_fn is None:
        return False
    return device.type == "cuda" and torch.version.hip is None


def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Scales x to unit RMS along its last dim -- no learnable weight, since
    q_proj/k_proj/v_proj already have one; this only strips the raw,
    unbounded magnitude those projections would otherwise pass through."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class _TitansFrontEnd(nn.Module):
    """Shared, single front-end: projects the layer-READ_LAYER residual to
    q/k/v and to the data-dependent write knobs (eta, theta, alpha), which
    Model.forward buffers across a memory-window and feeds to one
    `_NeuralMemory` write per window, plus a `_NeuralMemory` read/surprise
    every token (see observe()'s docstring)."""

    def __init__(self, d_model: int = D_MODEL, mem_dim: int = MEM_DIM, mem_hidden: int = MEM_HIDDEN):
        super().__init__()
        self.mem_dim = mem_dim
        self.mem_hidden = mem_hidden
        self.q_proj = nn.Linear(d_model, mem_dim)
        self.k_proj = nn.Linear(d_model, mem_dim)
        self.v_proj = nn.Linear(d_model, mem_dim)
        # eta (momentum), theta (step size), alpha (decay) -- all small and
        # data-dependent, per Titans. Zero-init weight + fixed bias so each
        # knob starts at a chosen operating point and learns data-dependence
        # from there: a default random init, fed by the residual stream
        # (rms in the tens at READ_LAYER), puts every pre-activation tens
        # deep into sigmoid's rails -- theta pinned at ~0 (the memory is
        # never written), eta pinned at its cap, alpha slammed to a random
        # rail per token -- with ~no gradient through the saturated sigmoids
        # to recover. sft/measure_knobs.py measures the actual per-token
        # knob distributions on a checkpoint or fresh init.
        self.knob_proj = nn.Linear(d_model, 3)
        nn.init.zeros_(self.knob_proj.weight)
        # eta/theta bias 0: sigmoid midpoint, full gradient (momentum decay
        # 0.45, write step 0.05). alpha bias -4: decay starts near the
        # bottom of its already-small range (sigmoid(-4) * ALPHA_CAP ~=
        # 2e-6/window) so the store retains by default and forgetting is
        # learned. -4 sits in the same low-gradient tail that beta's
        # comment below rules out for beta_proj.bias, but the asymmetry
        # makes it safe here: a slow-to-wake beta disables the whole
        # subsystem, while for alpha the saturated default IS the desired
        # behavior and forgetting is a refinement the optimizer can afford
        # to learn slowly.
        with torch.no_grad():
            self.knob_proj.bias.copy_(torch.tensor([0.0, 0.0, -4.0]))

    def init_memory(self, batch_size: int, device, dtype) -> _NeuralMemory:
        return _NeuralMemory(batch_size, self.mem_dim, self.mem_hidden, device, dtype)

    def observe(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """residual: (batch, d_model) residual stream entering READ_LAYER.

        Returns (q, k, v, eta, theta, alpha) -- the part of what used to be
        a single combined step() (see git history) that doesn't touch the
        memory itself, split out so Model.forward can buffer these across a
        memory-window before the one consolidated `_NeuralMemory.write` call
        that closes it (see `_NeuralMemory.write`'s docstring for why),
        while still reading/scoring every token individually via
        `_NeuralMemory.read`/`.surprise` against whatever weights are
        current at that point.

        Grad tracking must stay enabled here regardless of whether the
        *caller* is in a torch.no_grad() block: k/v/knobs need to remain
        part of the same differentiable chain that `_NeuralMemory.write`
        eventually backprops through (via create_graph=True) so
        q_proj/k_proj/v_proj/knob_proj actually receive gradient from the
        outer loss -- this isn't optional training-time machinery, it has to
        run even during eval/inference, same reasoning as write() itself.
        torch.enable_grad() punches through an enclosing no_grad() for
        exactly this; scoped to just this method (and the read/surprise/
        write calls around it in Model.forward), so the rest of the model
        (the 64-layer backbone, the gated-delta merge, the C readout) stays
        grad-free and cheap under an outer no_grad() as normal.
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
            # knob_proj gets the same unit-rms treatment, applied to its
            # *input* (q/k/v normalize the output instead so their learned
            # weight sets the output scale; the knobs' output scale is
            # already pinned by sigmoid + the caps below): its
            # pre-activations then track the learned weights rather than
            # the residual's raw magnitude, keeping the sigmoids off their
            # zero-gradient rails as the weights grow from zero-init.
            knobs = torch.sigmoid(self.knob_proj(_rms_normalize(residual)))
            # eta capped at 0.9 (not left at sigmoid's full (0, 1) range like
            # theta/alpha's caps, which bound them *small* on purpose --
            # small theta is what makes the write step gentle, and alpha's
            # far tighter ALPHA_CAP is what makes the memory's content
            # persist across a whole chain of episodes, see its comment):
            # S_t = eta * S_{t-1} - theta * g_t accumulates with essentially
            # no decay when eta is allowed to approach 1, turning the
            # momentum into an undamped running sum of every token's write.
            # Capping below 1 guarantees at least 10% decay per token.
            eta, theta, alpha = knobs[..., 0] * 0.9, knobs[..., 1] * 0.1, knobs[..., 2] * ALPHA_CAP
        return q, k, v, eta, theta, alpha


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
        # BETA_BIAS_ANNEAL_START/_STEPS above and Model.set_beta_anneal),
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
        # loads only named_parameters with requires_grad). `load_inference`
        # itself never calls Model.set_beta_anneal, so a bare load defaults
        # to the *post-anneal* state (0) -- correct for any real checkpoint,
        # since training runs almost always pass BETA_BIAS_ANNEAL_STEPS long
        # before being deployed. The backend actually derives the right
        # in-progress value from the checkpoint's own step number via
        # `post_load` (models/mamba2_2_7b_memory/__init__.py, called by
        # backend/app/model/registry.py right after load_checkpoint) rather
        # than relying on this default, so this only matters for the rare
        # case of loading a checkpoint saved before step
        # BETA_BIAS_ANNEAL_STEPS. Training itself overrides this to the
        # correct in-progress value immediately via the one-time
        # pre-training-loop on_step call in sft/train.py, before this would
        # otherwise matter there either.
        self.beta_anneal_offset = 0.0

    def signals(self, o_t: torch.Tensor, surprise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (p, key, beta, retain) for this layer's gated-delta merge.

        p: (batch, nheads, headdim) per-head value to write.
        key: (batch, d_state) shared write address.
        beta, retain: (batch, 1, 1, 1) gates, broadcastable against ssm_state.
        """
        z = self.down(o_t)
        p = rearrange(self.value_proj(z), "b (h p) -> b h p", p=self.headdim)
        # The gated-delta merge's forget term (see _mixer_step) only cancels
        # the state's key-component exactly for a unit-norm key: the per-write
        # gain along the key is retain * (1 - beta * |key|^2), which for
        # |key|^2 > 2/beta is below -1 -- every write then flips the sign of
        # that component AND grows it, an exponential ssm_state blow-up
        # (observed live as min_cos_sim pinned at -1 with ssm_norm -> inf on
        # slots deep into long examples once beta trained past ~0.3).
        # Normalizing bounds the gain to retain * (1 - beta), stable for any
        # learned beta/retain in (0, 1).
        key = F.normalize(self.key_proj(z), dim=-1)
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


class _TokenMixInjection(nn.Module):
    """The "mix" integration arm (see models/mamba2_780m_memory_mix/README.md):
    the gated read is ADDED to the residual stream at the read layer,
    same-token, instead of gated-delta-merged into ssm_state. Deliberately
    shares _GatedDeltaInjection's bottleneck/gate structure (down 128-dim,
    beta from the same o_t + sigmoid(surprise) + anneal-offset signals) so
    the two A/B arms differ ONLY in where the gated read lands.

    o_proj is zero-initialized (weight and bias): the mix is an exact no-op
    at init, and o_proj still receives gradient from step 0 because beta's
    sigmoid is nonzero. No fixed mixing ratio anywhere -- the magnitude
    equilibrium between the memory term and the residual stream is learned
    via o_proj/beta under the LM loss. (Known subtlety, deliberately not
    built: residual RMS grows with depth, so the effective ratio of the
    memory term shrinks as activations grow; if beta saturates compensating,
    the fix is an RMS-reference normalization in this branch.)"""

    def __init__(self, d_model: int, mem_dim: int = MEM_DIM, r: int = BOTTLENECK_R):
        super().__init__()
        self.down = nn.Linear(mem_dim, r)
        self.beta_proj = nn.Linear(r, 1)
        self.o_proj = nn.Linear(r, d_model)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        # Same startup-suppression mechanism as _GatedDeltaInjection's --
        # see its beta_anneal_offset comment; updated by Model.set_beta_anneal.
        self.beta_anneal_offset = 0.0

    def mix_term(self, o_t: torch.Tensor, surprise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mix, beta): mix is the gated residual addition, same
        leading shape as o_t with d_model channels; beta the per-token gate.
        Works on (B, mem_dim) and (B, L, mem_dim) alike."""
        z = self.down(o_t)
        beta = torch.sigmoid(
            self.beta_proj(z).squeeze(-1) + torch.sigmoid(surprise) + self.beta_anneal_offset
        )
        return beta.unsqueeze(-1) * self.o_proj(z), beta


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
        # getattr fallback: a state unpickled from a mem_state.pt saved before
        # w1_init/w2_init existed lacks them -- fall back to the current
        # weights as the drift baseline (heals fully at each slot reset).
        neural_memory.w1_init = getattr(self.neural_memory, "w1_init", self.neural_memory.w1).detach()
        neural_memory.w2_init = getattr(self.neural_memory, "w2_init", self.neural_memory.w2).detach()
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

    def __init__(
        self,
        mamba_model: MambaLMHeadModel,
        read_layer: int = READ_LAYER,
        injected_layers: tuple[int, ...] = INJECTED_LAYERS,
        integration: str = "state",
    ):
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

        # Trainable delta for the frozen role-marker embedding/lm_head rows
        # (see MarkerDelta); marker_token_ids is stamped by extend_embeddings,
        # absent on raw models (e.g. the tiny test fixtures). Created after
        # the freeze block above, so it stays trainable.
        marker_ids = getattr(mamba_model, "marker_token_ids", None)
        self.marker_delta = (
            MarkerDelta(marker_ids, self.d_model, device=self.embedding.weight.device, dtype=self.embedding.weight.dtype)
            if marker_ids
            else None
        )

        for layer in self.layers:
            assert layer.mixer.ngroups == 1, "memory injection assumes ngroups=1"

        # Memory-subsystem geometry is derived from the backbone (nheads/
        # headdim/d_state from its own mixers, mem dims from d_model), so the
        # same class wraps any Mamba2 stack -- read_layer/injected_layers
        # default to this module's 2.7B constants and are overridden by
        # other model packages (e.g. models/mamba2_780m_memory_state). The
        # BOTTLENECK_R=128 read bottleneck is deliberately NOT scaled with
        # d_model (see models/mamba2_780m_memory_state/README.md).
        assert integration in ("state", "mix"), integration
        mixer0 = self.layers[0].mixer
        self.integration = integration
        self.read_layer = read_layer
        # "mix" has no gated-delta injections by construction -- the read
        # lands on the residual stream at read_layer instead.
        self.injected_layers = () if integration == "mix" else tuple(injected_layers)
        self.mem_dim = self.d_model
        self.mem_hidden = 4 * self.d_model
        self.front_end = _TitansFrontEnd(self.d_model, self.mem_dim, self.mem_hidden)
        self.injections = nn.ModuleDict(
            {
                str(i): _GatedDeltaInjection(
                    self.mem_dim, BOTTLENECK_R, mixer0.nheads, mixer0.headdim, mixer0.d_state
                )
                for i in self.injected_layers
            }
        )
        self.mix = _TokenMixInjection(self.d_model, self.mem_dim) if integration == "mix" else None
        # Kill switch for the memory->backbone pathway (used by
        # sft/probe_recall.py's --ablation none): when False, no injection
        # events fire AND the front-end read/write machinery is skipped
        # entirely, so forward() is exactly the plain (LoRA'd) backbone and
        # is invariant to the neural memory's content. Plain attribute, not
        # checkpointed state, same reasoning as memory_window.
        self.injection_enabled = True
        # Tokens per memory-subsystem write (see set_memory_window and
        # _NeuralMemory.write) -- 1 reproduces the original exact per-token
        # behavior and is the default so nothing changes unless a caller
        # opts in. Deliberately a plain attribute, not a checkpointed
        # buffer/parameter (same reasoning as beta_anneal_offset): it's a
        # training-run configuration choice, not model state, and inference
        # (load_inference) always wants 1 regardless of what a checkpoint
        # was trained with (see the design spec for why train/inference
        # windows are intentionally decoupled).
        self.memory_window = 1
        # Running sums for pop_memory_stats() -- accumulated as detached
        # tensors (all inputs are already .detach()'d at the accumulation
        # sites, so this never holds a reference into any autograd graph)
        # and only converted to Python floats once, in pop_memory_stats
        # itself. Calling .item() at every accumulation instead (the
        # original approach) forces a blocking GPU sync per injected layer
        # per token, which serializes the whole per-token training loop --
        # far more costly than the sync-free tensor accumulation here.
        # beta/retain are accumulated once per injected layer per token;
        # surprise/o_t_norm once per token (at READ_LAYER only); grad_norm
        # only on tokens that close a memory-window (see set_memory_window),
        # since that's the only time _NeuralMemory.write actually produces a
        # fresh one -- hence its own _mem_stat_write_count divisor, separate
        # from _mem_stat_tok_count. See pop_memory_stats for why these
        # matter: beta/retain start near 0/1 (no-op init, see
        # _GatedDeltaInjection.__init__) and are the only direct signal of
        # whether the memory subsystem is actually being used or still
        # sitting at its identity init.
        self._mem_stat_sums = self._fresh_mem_stat_sums()
        self._mem_stat_inj_count = 0
        self._mem_stat_tok_count = 0
        self._mem_stat_write_count = 0
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
        # Injection.signals/_TitansFrontEnd.observe mismatch dtypes against it.
        mem_dtype = self.front_end.q_proj.weight.dtype
        neural_memory = self.front_end.init_memory(batch_size, device, mem_dtype)
        last_o_t = torch.zeros(batch_size, self.mem_dim, device=device, dtype=mem_dtype)
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

    def _mixer_span(
        self,
        mixer,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused-kernel counterpart to `_mixer_step`, for a whole span of L
        tokens in one call via `mamba_ssm`'s native `causal_conv1d_fn` +
        `mamba_chunk_scan_combined` instead of a sequential Python loop over
        L manual steps -- see the module docstring and the design spec's
        "Fused-kernel dispatch" section. Only ever called on a span where
        every token gets `gated_delta=None`: it has no parameter for
        injecting one, by construction. Caller must have already confirmed
        `_fused_path_available()`.

        `hidden_states`: (batch, L, d_model). `conv_state`/`ssm_state`: the
        exact same shapes `_mixer_step` uses, carrying history from BEFORE
        this span (zero-init, mid-chunk, or continued from a previous
        window/chunk/call) -- same contract as `_mixer_step`'s.

        Returns (out (batch, L, d_model), new_conv_state, new_ssm_state), in
        the same shapes/dtypes as the corresponding inputs, so a caller can
        freely interleave this with `_mixer_step` calls (see
        `_forward_fused`'s injected-layer branch) or persist the result onto
        `MemoryState` exactly like `_mixer_step`'s output.
        """
        dtype = hidden_states.dtype
        zxbcdt = mixer.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * mixer.d_ssm - 2 * mixer.ngroups * mixer.d_state - mixer.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt, [d_mlp, d_mlp, mixer.d_ssm, mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state, mixer.nheads], dim=-1
        )

        xBC_t = rearrange(xBC, "b l d -> b d l")
        # conv_state holds the last d_conv RAW (pre-conv, pre-activation)
        # xBC values ending at the token right before this span (see
        # _mixer_step's own roll-buffer convention) -- causal_conv1d_fn's
        # `initial_states` wants only the last d_conv-1 of those (the
        # left-context strictly before this span's first output).
        #
        # causal_conv1d_fn's CUDA kernel hard-requires initial_states in
        # channel-last layout (stride(1) == 1, i.e. the dim axis
        # contiguous) -- the same layout mamba_ssm's allocate_inference_cache
        # produces (torch.zeros(b, d_conv, dim).transpose(1, 2)), which is
        # why the very first span of a run works. But this function's own
        # new_conv_state below is built via torch.cat, which always returns
        # an ordinary row-major tensor (stride(2) == 1) -- so on every
        # subsequent span, conv_state has lost the channel-last layout and
        # the kernel asserts. causal_conv1d_interface's own defensive
        # `.contiguous()` fallback doesn't catch this either, since it only
        # triggers when BOTH stride(1) and stride(2) are non-1, and
        # stride(2) == 1 is already true here. Force channel-last
        # explicitly rather than relying on conv_state's incoming layout.
        conv_history = conv_state[:, :, 1:].transpose(1, 2).contiguous().transpose(1, 2)
        xBC_conv = causal_conv1d_fn(
            x=xBC_t,
            weight=rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
            bias=mixer.conv1d.bias,
            initial_states=conv_history,
            activation=mixer.activation,
        )
        # Reconstructed directly from this span's own raw (pre-conv) values
        # plus the pre-span history -- deliberately NOT causal_conv1d_fn's
        # own `return_final_states` option, whose exact output shape/
        # convention isn't independently verified in this environment
        # (causal-conv1d can't be installed/exercised on this repo's local
        # ROCm dev box, see root CLAUDE.md); concatenating and slicing the
        # last d_conv columns reproduces _mixer_step's roll-buffer exactly,
        # using only documented, already-verified shapes.
        new_conv_state = torch.cat([conv_history, xBC_t], dim=-1)[..., -conv_state.shape[-1]:]
        xBC = rearrange(xBC_conv, "b d l -> b l d").to(dtype=dtype)

        x, B, C = torch.split(xBC, [mixer.d_ssm, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state], dim=-1)
        A = -torch.exp(mixer.A_log.float())
        x_h = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=mixer.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=mixer.ngroups)

        y, final_state = mamba_chunk_scan_combined(
            x_h,
            dt,
            A,
            B,
            C,
            chunk_size=mixer.chunk_size,
            D=mixer.D,
            z=None,
            dt_bias=mixer.dt_bias,
            initial_states=ssm_state,
            dt_softplus=True,
            return_final_states=True,
        )
        new_ssm_state = final_state.to(dtype=ssm_state.dtype)

        y = rearrange(y, "b l h p -> b l (h p)")
        if not mixer.rmsnorm:
            y = y * mixer.act(z)
        else:
            y = mixer.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = mixer.out_proj(y)
        return out, new_conv_state, new_ssm_state

    def _fused_path_available(self) -> bool:
        """True when this process can safely dispatch through `_mixer_span`
        instead of the all-manual `_forward_manual` loop -- see
        `fused_kernel_usable`'s docstring for the actual check. Purely
        feature-detected, not a manual flag -- this repo's local ROCm dev
        box (gfx1102, confirmed broken -- see root CLAUDE.md) transparently
        keeps using `_forward_manual` with nothing for a caller to
        remember."""
        return fused_kernel_usable(self.embedding.weight.device)

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

        Both the memory subsystem's WRITE (`_NeuralMemory.write`) and its
        INJECTION into `ssm_state` (the gated-delta merge, `_mixer_step`) are
        consolidated every `self.memory_window` tokens instead of every
        token -- see set_memory_window and the design spec at
        docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md.
        The READ (`_NeuralMemory.read`/`.surprise`) still happens every
        token regardless (cheap forward passes against whatever weights are
        currently active, not a sequential recurrence) -- what changes is
        that only the window-closing token's injection actually uses a
        signal: a surprise-weighted pooling of every token's read in the
        window, not just that one token's own. For the other
        `memory_window - 1` tokens, every injected layer gets `gated_delta =
        None`, i.e. `ssm_state` evolves purely under Mamba2's own
        unmodified dynamics.

        This is the dispatcher: it validates `input_ids`/`state` once, then
        hands off to `_forward_fused` (the `window - 1` non-injection tokens
        per window run through `mamba_ssm`'s native fused/chunked kernel,
        `_mixer_span`; only the window-closing token still runs the manual
        per-token `_mixer_step`) when `_fused_path_available()`, else
        `_forward_manual` (the original all-manual per-token loop, byte-for-
        byte unchanged -- this repo's local ROCm dev box, gfx1102, always
        takes this path, see root CLAUDE.md). Both produce identical results
        by construction (same math, same call sequence into `_mixer_step`/
        `_NeuralMemory`, just batched differently across time where nothing
        depends on per-token ordering) -- this is feature-detected per call,
        not a flag a caller needs to set or remember.

        `seqlen` must be an exact multiple of `self.memory_window`: a window
        is buffered purely locally within one call and must fully flush
        before that call returns, so it can never span across two forward()
        calls (which may be separated by a detach() at a chunk boundary
        during chunked training -- see MemoryState.detach). sft/train.py
        enforces `chunk_len % memory_window == 0` for training; inference
        always keeps memory_window=1, which divides any seqlen and
        reproduces today's exact per-token injection as a special case, not
        a separate code path.
        """
        batch_size, seqlen = input_ids.shape
        if seqlen % self.memory_window != 0:
            raise ValueError(
                f"seqlen ({seqlen}) must be a multiple of memory_window ({self.memory_window}) "
                "-- a memory-window can't span across forward() calls"
            )
        device = input_ids.device
        dtype = self.embedding.weight.dtype
        if state is None:
            state = self._init_state(batch_size, device, dtype)

        if self._fused_path_available():
            return self._forward_fused(input_ids, state)
        return self._forward_manual(input_ids, state)

    def _forward_manual(self, input_ids: torch.Tensor, state: MemoryState) -> tuple[torch.Tensor, MemoryState]:
        """All-manual per-token fallback: processes `input_ids` one token at
        a time regardless of T (see the module docstring for why) --
        correct for both a many-token prefill and a single incremental
        decode step, just not parallelized across T. The only dispatch path
        on hardware without a working fused kernel (this repo's local ROCm
        dev box, see `_fused_path_available`); also always correct
        everywhere else, just slower than `_forward_fused` there. See
        `forward()`'s docstring for the windowed memory-write/injection
        semantics, unchanged here from before the fused dispatch existed.
        """
        batch_size, seqlen = input_ids.shape
        all_logits = []
        # Buffered per-token write inputs for the memory subsystem's current
        # (not-yet-closed) window -- purely local to this call, flushed via
        # _NeuralMemory.write whenever it reaches self.memory_window entries.
        # See the forward()/write() docstrings for why this can't persist on
        # `state` across calls.
        pending_k: list[torch.Tensor] = []
        pending_v: list[torch.Tensor] = []
        pending_eta: list[torch.Tensor] = []
        pending_theta: list[torch.Tensor] = []
        pending_alpha: list[torch.Tensor] = []
        # Every token's read, buffered the same way as the write inputs
        # above -- pooled (surprise-weighted) into the one signal the
        # window-closing token's injection actually uses. See the pooling
        # block below and the forward() docstring.
        pending_o: list[torch.Tensor] = []
        pending_surprise: list[torch.Tensor] = []
        # Carries the most recent window's grad_norm forward across tokens
        # within this call (unlike surprise/o_t_norm, grad_norm only has a
        # fresh value on the token that closes a window) so the last-token
        # log snapshot below always has a value once at least one write has
        # happened in this call.
        last_grad_norm_per_slot: torch.Tensor | None = None
        for t in range(seqlen):
            h = self.embedding(input_ids[:, t])
            if self.marker_delta is not None:
                h = self.marker_delta.embed(h, input_ids[:, t])
            residual = None
            # True on exactly one token per window -- the one that closes
            # it, per set_memory_window's contract. Both the write and the
            # injection are gated on this, not on "every token" (see
            # forward()'s docstring).
            is_window_close = (t % self.memory_window) == (self.memory_window - 1)
            token_betas: list[torch.Tensor] = []   # (B,) per injected layer
            token_retains: list[torch.Tensor] = []
            token_cos_sims: list[torch.Tensor] = []
            last_surprise_per_slot: torch.Tensor | None = None
            last_o_t_norm_per_slot: torch.Tensor | None = None
            mix_beta_per_slot: torch.Tensor | None = None
            for i, layer in enumerate(self.layers):
                # "mix" integration: the read happens on the stream ENTERING
                # read_layer and lands right back on it, same token, before
                # this layer's own computation -- so read_layer and everything
                # above computes over the memory term (see _TokenMixInjection).
                if self.integration == "mix" and i == self.read_layer and self.injection_enabled:
                    stream = (h + residual) if residual is not None else h
                    with torch.enable_grad():
                        q, k, v, eta, theta, alpha = self.front_end.observe(stream)
                        o_t = state.neural_memory.read(q)
                        surprise = state.neural_memory.surprise(k, v)
                        last_surprise_per_slot = surprise.detach()
                        last_o_t_norm_per_slot = o_t.detach().norm(dim=-1)
                        self._mem_stat_sums["surprise"] += last_surprise_per_slot.mean()
                        self._mem_stat_sums["o_t_norm"] += last_o_t_norm_per_slot.mean()
                        self._mem_stat_tok_count += 1

                        pending_k.append(k)
                        pending_v.append(v)
                        pending_eta.append(eta)
                        pending_theta.append(theta)
                        pending_alpha.append(alpha)
                        if is_window_close:
                            ks = torch.stack(pending_k, dim=0)
                            vs = torch.stack(pending_v, dim=0)
                            etas = torch.stack(pending_eta, dim=0)
                            thetas = torch.stack(pending_theta, dim=0)
                            alphas = torch.stack(pending_alpha, dim=0)
                            _, grad_norm = state.neural_memory.write(ks, vs, etas, thetas, alphas)
                            last_grad_norm_per_slot = grad_norm.detach()
                            self._mem_stat_sums["grad_norm"] += last_grad_norm_per_slot.mean()
                            self._accum_write_stats(state.neural_memory, alphas)
                            self._mem_stat_write_count += 1
                            pending_k, pending_v, pending_eta, pending_theta, pending_alpha = [], [], [], [], []
                        mix, mix_beta = self.mix.mix_term(o_t, surprise)
                    h = h + mix.to(h.dtype)
                    mix_beta_per_slot = mix_beta.detach()
                    self._mem_stat_sums["beta"] += mix_beta_per_slot.mean()
                    self._mem_stat_inj_count += 1

                h, residual = self._prenorm(layer, h, residual)
                gated_delta = None
                if i in self.injected_layers and is_window_close and self.injection_enabled:
                    gated_delta = self.injections[str(i)].signals(state.last_o_t, state.last_surprise)
                    beta, retain = gated_delta[2], gated_delta[3]
                    beta_per_slot = beta[:, 0, 0, 0].detach()    # (B,)
                    retain_per_slot = retain[:, 0, 0, 0].detach()  # (B,)
                    self._mem_stat_sums["beta"] += beta_per_slot.mean()
                    self._mem_stat_sums["retain"] += retain_per_slot.mean()
                    prev_min = self._mem_stat_sums["retain_min"]
                    slot_min = retain_per_slot.min()
                    self._mem_stat_sums["retain_min"] = (
                        slot_min if prev_min is None else torch.minimum(prev_min, slot_min)
                    )
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

                if i == self.read_layer and self.injection_enabled and self.integration == "state":
                    with torch.enable_grad():
                        q, k, v, eta, theta, alpha = self.front_end.observe(residual)
                        o_t = state.neural_memory.read(q)
                        surprise = state.neural_memory.surprise(k, v)
                        # Per-token diagnostics (surprise/o_t_norm stats,
                        # below) always reflect this token's own raw read,
                        # regardless of window size -- separate from the
                        # pooled signal injection actually uses, computed
                        # only at window close, below.
                        last_surprise_per_slot = surprise.detach()          # (B,)
                        last_o_t_norm_per_slot = o_t.detach().norm(dim=-1)  # (B,)
                        self._mem_stat_sums["surprise"] += last_surprise_per_slot.mean()
                        self._mem_stat_sums["o_t_norm"] += last_o_t_norm_per_slot.mean()
                        self._mem_stat_tok_count += 1

                        pending_k.append(k)
                        pending_v.append(v)
                        pending_eta.append(eta)
                        pending_theta.append(theta)
                        pending_alpha.append(alpha)
                        pending_o.append(o_t)
                        pending_surprise.append(surprise)
                        if is_window_close:
                            ks = torch.stack(pending_k, dim=0)
                            vs = torch.stack(pending_v, dim=0)
                            etas = torch.stack(pending_eta, dim=0)
                            thetas = torch.stack(pending_theta, dim=0)
                            alphas = torch.stack(pending_alpha, dim=0)
                            _, grad_norm = state.neural_memory.write(ks, vs, etas, thetas, alphas)
                            last_grad_norm_per_slot = grad_norm.detach()  # (B,)
                            self._mem_stat_sums["grad_norm"] += last_grad_norm_per_slot.mean()
                            self._accum_write_stats(state.neural_memory, alphas)
                            self._mem_stat_write_count += 1
                            pending_k, pending_v, pending_eta, pending_theta, pending_alpha = [], [], [], [], []

                            # Surprise-weighted pooling of every token's read
                            # in the window that's about to close -- this,
                            # not any single token's raw o_t/surprise, is
                            # what the injections computed above (and the
                            # next window's layers 22-40, which read
                            # state.last_o_t/last_surprise before this
                            # token's own layer-42 update) actually use. A
                            # window of 1 makes this a softmax over a single
                            # entry (i.e. weight 1.0), reproducing the
                            # original per-token o_t/surprise exactly.
                            o_stack = torch.stack(pending_o, dim=0)                  # (W, B, mem_dim)
                            surprise_stack = torch.stack(pending_surprise, dim=0)    # (W, B)
                            pool_weights = torch.softmax(surprise_stack, dim=0)      # (W, B)
                            state.last_o_t = (pool_weights.unsqueeze(-1) * o_stack).sum(dim=0)
                            state.last_surprise = (pool_weights * surprise_stack).sum(dim=0)
                            pending_o, pending_surprise = [], []

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
                        entry["grad_norm"] = (
                            last_grad_norm_per_slot[b].item() if last_grad_norm_per_slot is not None else float("nan")
                        )
                    entry["ssm_norm"] = max(state.ssm_states[i][b].norm().item() for i in self.injected_layers)
                    entry["resid_norm"] = residual[b].norm().item()
                    logs.append(entry)
                self._last_token_logs = logs
            elif self.integration == "mix" and t == seqlen - 1 and mix_beta_per_slot is not None:
                self._last_token_logs = [
                    {
                        "beta": mix_beta_per_slot[b].item(),
                        "surprise": last_surprise_per_slot[b].item(),
                        "o_t_norm": last_o_t_norm_per_slot[b].item(),
                        "grad_norm": (
                            last_grad_norm_per_slot[b].item() if last_grad_norm_per_slot is not None else float("nan")
                        ),
                        "resid_norm": residual[b].norm().item(),
                    }
                    for b in range(batch_size)
                ]

            h = self._apply_norm_f(h, residual)
            logits = self.lm_head(h)
            if self.marker_delta is not None:
                logits = self.marker_delta.head(logits, h)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1), state

    def _forward_fused(self, input_ids: torch.Tensor, state: MemoryState) -> tuple[torch.Tensor, MemoryState]:
        """Fused-kernel dispatch: processes the WHOLE chunk layer-by-layer
        (like `mamba_ssm`'s own full-sequence backbone forward) instead of
        token-by-token. For the 43 layers never in `INJECTED_LAYERS`, one
        `_mixer_span` call covers the entire chunk -- no per-token
        dependency ever exists for them. For the 21 injected layers
        (`INJECTED_LAYERS`, including `READ_LAYER`), each memory-window
        within the chunk is handled as one `_mixer_span` call over its
        `window - 1` leading tokens (skipped entirely at `window == 1`),
        followed by one `_mixer_step` call for the window-closing token --
        the only token that still needs per-token, pre-readout state access
        (see the module docstring). Windows are still processed in
        sequence (a Python loop), since `state.last_o_t`/`last_surprise`
        thread across them exactly as they do across tokens in
        `_forward_manual` -- only removes the O(seqlen) manual loop within
        each window's non-injection span, not the O(n_windows) loop itself
        (n_windows = seqlen // memory_window is small by construction, see
        the design spec's memory-window rationale).

        `_prenorm` runs once per layer over the FULL (batch, seqlen,
        d_model) hidden/residual stream rather than once per token --
        mathematically identical either way (RMSNorm/residual-add have no
        cross-token coupling; `layer_norm_fn` is already used this way in
        `mamba_ssm`'s own full-sequence `MixerModel.forward`), just computed
        in one batched call instead of `seqlen` sequential ones.

        At `READ_LAYER`, `observe()`/`_NeuralMemory.read_windowed`/
        `.surprise_windowed` replace the per-token `observe()`/`read()`/
        `.surprise()` calls, computed once per window over the residual
        entering `READ_LAYER` for every token in that window (available
        directly from `_prenorm`'s already-computed full-chunk `residual`,
        since injected/non-injected status doesn't change what `_prenorm`
        computes) -- not just the window-closing token's, preserving the
        surprise-weighted pooling semantics exactly. `_NeuralMemory.write`
        still consolidates once per window, same as `_forward_manual`. The
        ordering subtlety at `READ_LAYER` (this window's gated-delta merge
        uses the PREVIOUS window's pooled `state.last_o_t`/`last_surprise`,
        which only gets overwritten by THIS window's pooled values after
        that merge -- see the README's "layer 42 subtlety" and the module
        docstring) is preserved by computing `gated_delta` before calling
        `_mixer_step`, then updating `state.last_o_t`/`last_surprise`
        afterward, exactly the same order `_forward_manual` uses.
        """
        batch_size, seqlen = input_ids.shape
        window = self.memory_window
        n_windows = seqlen // window

        h = self.embedding(input_ids)
        if self.marker_delta is not None:
            h = self.marker_delta.embed(h, input_ids)
        residual: torch.Tensor | None = None

        # Diagnostics are only ever materialized for the chunk's LAST
        # window's closing token (== the chunk's last token, since
        # seqlen % memory_window == 0) -- same "only the last token of the
        # chunk" restriction `_forward_manual` applies, for the same reason
        # (avoid a per-injected-layer-per-window blocking .item() sync; see
        # its comment).
        token_betas: list[torch.Tensor] = []
        token_retains: list[torch.Tensor] = []
        token_cos_sims: list[torch.Tensor] = []
        last_surprise_per_slot: torch.Tensor | None = None
        last_o_t_norm_per_slot: torch.Tensor | None = None
        last_grad_norm_per_slot: torch.Tensor | None = None
        mix_beta_per_slot: torch.Tensor | None = None

        for i, layer in enumerate(self.layers):
            # "mix" integration hook -- same placement/ordering as
            # _forward_manual's (stream entering read_layer, same-token
            # landing), batched over the chunk: the per-window loop below
            # only touches front-end/M (no backbone state), so every layer
            # still runs one full-chunk _mixer_span. Reads are against the
            # window-START M (read_windowed before write), matching the
            # manual path exactly.
            if self.integration == "mix" and i == self.read_layer and self.injection_enabled:
                stream = (h + residual) if residual is not None else h  # (B, L, d_model)
                o_parts: list[torch.Tensor] = []
                s_parts: list[torch.Tensor] = []
                with torch.enable_grad():
                    for w in range(n_windows):
                        win = stream[:, w * window : (w + 1) * window, :]
                        q, k, v, eta, theta, alpha = self.front_end.observe(win)
                        q_w, k_w, v_w = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
                        o_win = state.neural_memory.read_windowed(q_w)                  # (W, B, mem_dim)
                        surprise_win = state.neural_memory.surprise_windowed(k_w, v_w)  # (W, B)
                        for wt in range(window):
                            self._mem_stat_sums["surprise"] += surprise_win[wt].detach().mean()
                            self._mem_stat_sums["o_t_norm"] += o_win[wt].detach().norm(dim=-1).mean()
                            self._mem_stat_tok_count += 1
                        _, grad_norm = state.neural_memory.write(
                            k_w, v_w, eta.transpose(0, 1), theta.transpose(0, 1), alpha.transpose(0, 1)
                        )
                        last_grad_norm_per_slot = grad_norm.detach()
                        self._mem_stat_sums["grad_norm"] += last_grad_norm_per_slot.mean()
                        self._accum_write_stats(state.neural_memory, alpha.transpose(0, 1))
                        self._mem_stat_write_count += 1
                        o_parts.append(o_win.transpose(0, 1))        # (B, W, mem_dim)
                        s_parts.append(surprise_win.transpose(0, 1))  # (B, W)
                    o_all = torch.cat(o_parts, dim=1)
                    s_all = torch.cat(s_parts, dim=1)
                    mix, mix_beta = self.mix.mix_term(o_all, s_all)
                h = h + mix.to(h.dtype)
                mix_beta_per_slot = mix_beta[:, -1].detach()
                last_surprise_per_slot = s_all[:, -1].detach()
                last_o_t_norm_per_slot = o_all[:, -1].detach().norm(dim=-1)
                self._mem_stat_sums["beta"] += mix_beta.detach().mean()
                self._mem_stat_inj_count += 1

            h, residual = self._prenorm(layer, h, residual)

            if i not in self.injected_layers or not self.injection_enabled:
                h, conv_state, ssm_state = self._mixer_span(
                    layer.mixer, h, state.conv_states[i], state.ssm_states[i]
                )
                state.conv_states[i] = conv_state
                state.ssm_states[i] = ssm_state
                continue

            conv_state, ssm_state = state.conv_states[i], state.ssm_states[i]
            out_parts: list[torch.Tensor] = []
            for w in range(n_windows):
                start, boundary = w * window, w * window + window - 1
                is_last_window = w == n_windows - 1

                if window > 1:
                    span_out, conv_state, ssm_state = self._mixer_span(
                        layer.mixer, h[:, start:boundary, :], conv_state, ssm_state
                    )
                    out_parts.append(span_out)

                # Uses state.last_o_t/last_surprise as they stand from the
                # PREVIOUS window (or the previous forward() call's final
                # window, or zero-init) -- this window's own pooled values
                # (computed below, at i == READ_LAYER) aren't visible to
                # this merge, same ordering _forward_manual uses.
                gated_delta = self.injections[str(i)].signals(state.last_o_t, state.last_surprise)
                beta, retain = gated_delta[2], gated_delta[3]
                beta_per_slot = beta[:, 0, 0, 0].detach()
                retain_per_slot = retain[:, 0, 0, 0].detach()
                self._mem_stat_sums["beta"] += beta_per_slot.mean()
                self._mem_stat_sums["retain"] += retain_per_slot.mean()
                prev_min = self._mem_stat_sums["retain_min"]
                slot_min = retain_per_slot.min()
                self._mem_stat_sums["retain_min"] = (
                    slot_min if prev_min is None else torch.minimum(prev_min, slot_min)
                )
                self._mem_stat_inj_count += 1
                if is_last_window:
                    token_betas.append(beta_per_slot)
                    token_retains.append(retain_per_slot)

                boundary_out, conv_state, ssm_state, injection_cos_sim = self._mixer_step(
                    layer.mixer, h[:, boundary, :], conv_state, ssm_state, gated_delta
                )
                out_parts.append(boundary_out.unsqueeze(1))
                if is_last_window:
                    token_cos_sims.append(injection_cos_sim)

                if i == self.read_layer:
                    with torch.enable_grad():
                        win_residual = residual[:, start : start + window, :]  # (B, W, d_model)
                        q, k, v, eta, theta, alpha = self.front_end.observe(win_residual)
                        # (B, W, ...) -> (W, B, ...): _apply_windowed's
                        # expected layout (see write()/read_windowed).
                        q_w, k_w, v_w = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
                        eta_w = eta.transpose(0, 1)
                        theta_w = theta.transpose(0, 1)
                        alpha_w = alpha.transpose(0, 1)

                        # Against the window-START M (frozen for the whole
                        # window), same as _forward_manual's per-token
                        # read()/surprise() calls -- write() (below) hasn't
                        # run yet for this window.
                        o_win = state.neural_memory.read_windowed(q_w)              # (W, B, mem_dim)
                        surprise_win = state.neural_memory.surprise_windowed(k_w, v_w)  # (W, B)
                        for wt in range(window):
                            self._mem_stat_sums["surprise"] += surprise_win[wt].detach().mean()
                            self._mem_stat_sums["o_t_norm"] += o_win[wt].detach().norm(dim=-1).mean()
                            self._mem_stat_tok_count += 1

                        _, grad_norm = state.neural_memory.write(k_w, v_w, eta_w, theta_w, alpha_w)
                        last_grad_norm_per_slot = grad_norm.detach()
                        self._mem_stat_sums["grad_norm"] += last_grad_norm_per_slot.mean()
                        self._accum_write_stats(state.neural_memory, alpha_w)
                        self._mem_stat_write_count += 1

                        # Surprise-weighted pooling -- see forward()'s
                        # docstring; identical formula to _forward_manual's.
                        pool_weights = torch.softmax(surprise_win, dim=0)  # (W, B)
                        state.last_o_t = (pool_weights.unsqueeze(-1) * o_win).sum(dim=0)
                        state.last_surprise = (pool_weights * surprise_win).sum(dim=0)

                        if is_last_window:
                            last_surprise_per_slot = surprise_win[-1].detach()
                            last_o_t_norm_per_slot = o_win[-1].detach().norm(dim=-1)

            state.conv_states[i] = conv_state
            state.ssm_states[i] = ssm_state
            h = torch.cat(out_parts, dim=1)

        if token_betas:
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
                    "min_cos_sim": min(cos_sims_b),
                }
                if last_surprise_per_slot is not None:
                    entry["surprise"] = last_surprise_per_slot[b].item()
                    entry["o_t_norm"] = last_o_t_norm_per_slot[b].item()
                    entry["grad_norm"] = (
                        last_grad_norm_per_slot[b].item() if last_grad_norm_per_slot is not None else float("nan")
                    )
                entry["ssm_norm"] = max(state.ssm_states[i][b].norm().item() for i in self.injected_layers)
                entry["resid_norm"] = residual[b, -1].norm().item()
                logs.append(entry)
            self._last_token_logs = logs
        elif self.integration == "mix" and mix_beta_per_slot is not None:
            self._last_token_logs = [
                {
                    "beta": mix_beta_per_slot[b].item(),
                    "surprise": last_surprise_per_slot[b].item(),
                    "o_t_norm": last_o_t_norm_per_slot[b].item(),
                    "grad_norm": (
                        last_grad_norm_per_slot[b].item() if last_grad_norm_per_slot is not None else float("nan")
                    ),
                    "resid_norm": residual[b, -1].norm().item(),
                }
                for b in range(batch_size)
            ]

        h = self._apply_norm_f(h, residual)
        logits = self.lm_head(h)
        if self.marker_delta is not None:
            logits = self.marker_delta.head(logits, h)
        return logits, state

    def set_memory_window(self, window: int) -> None:
        """Sets how many tokens' worth of write inputs `_NeuralMemory.write`
        consolidates into one gradient step (see forward()'s docstring).
        Must evenly divide whatever seqlen forward() is called with --
        sft/train.py validates `chunk_len % memory_window == 0` before
        calling this. 1 (the default set in __init__) reproduces the
        original exact per-token behavior."""
        assert window >= 1, f"memory_window must be >= 1, got {window}"
        self.memory_window = window

    def set_beta_anneal(self, global_step: int) -> None:
        """Updates every injected layer's beta_anneal_offset for the given
        global optimizer-step count (see BETA_BIAS_ANNEAL_START/_STEPS).
        A pure function of global_step -- not a counter incremented here --
        so calling this with a resumed run's global_step lands the offset
        exactly where it would have been had training never stopped, with no
        extra checkpoint state needed."""
        frac = min(global_step / BETA_BIAS_ANNEAL_STEPS, 1.0)
        offset = BETA_BIAS_ANNEAL_START * (1.0 - frac)
        for injection in self.injections.values():
            injection.beta_anneal_offset = offset
        if self.mix is not None:
            self.mix.beta_anneal_offset = offset

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
        bound1 = 1.0 / math.sqrt(self.mem_dim)
        bound2 = 1.0 / math.sqrt(self.mem_hidden)
        nm.w1[slot_idx].copy_((torch.rand(self.mem_hidden, self.mem_dim, device=device, dtype=mem_dtype) * 2 - 1) * bound1)
        nm.b1[slot_idx].zero_()
        nm.w2[slot_idx].copy_((torch.rand(self.mem_dim, self.mem_hidden, device=device, dtype=mem_dtype) * 2 - 1) * bound2)
        nm.b2[slot_idx].zero_()
        if not hasattr(nm, "w1_init"):
            nm.w1_init = nm.w1.detach().clone()
            nm.w2_init = nm.w2.detach().clone()
        else:
            nm.w1_init[slot_idx].copy_(nm.w1[slot_idx])
            nm.w2_init[slot_idx].copy_(nm.w2[slot_idx])
        for s in nm.momentum:
            s[slot_idx].zero_()
        state.last_o_t[slot_idx].zero_()
        state.last_surprise[slot_idx].zero_()

    def sleep_slot(self, state: "MemoryState", slot_idx: int) -> None:
        """Wipe slot slot_idx's backbone state in-place -- per-layer SSM/conv
        state back to the zeros a fresh sequence starts from -- while leaving
        the neural memory (w1/w2, momentum, drift baselines) untouched. This
        is a "sleep" boundary: context turnover that only the episodic store
        survives (see the README's three-tier design goal). last_o_t/
        last_surprise are zeroed with the backbone (they describe a read made
        against the wiped context; the first post-sleep token recomputes them
        from the persistent memory). Call only on a detached state."""
        for conv, ssm in zip(state.conv_states, state.ssm_states):
            conv[slot_idx].zero_()
            ssm[slot_idx].zero_()
        state.last_o_t[slot_idx].zero_()
        state.last_surprise[slot_idx].zero_()

    @staticmethod
    def _fresh_mem_stat_sums() -> dict[str, "float | torch.Tensor | None"]:
        return {
            "beta": 0.0, "retain": 0.0, "surprise": 0.0, "o_t_norm": 0.0, "grad_norm": 0.0,
            # w1_abs_max/w2_abs_max track a running MAX (not a sum-to-average
            # like the others) across the accumulation window, since an
            # average would smooth out exactly the kind of outlier spike
            # they exist to catch. M's w2 (unlike w1, whose pre-activation
            # gets tanh'd) has no activation bounding its output -- nothing
            # hard-clips w1/w2's magnitude directly, only the gradient used
            # to update them (see _NeuralMemory.write's soft clip), so nothing
            # stops them drifting large over many windows. Watching this is
            # meant to help confirm/rule out unbounded w1/w2 growth as the
            # source of a real crash where loss_sum stayed finite but the
            # backward pass produced a NaN gradient (see git history) --
            # write()'s create_graph=True second-order autograd through
            # this exact computation is a plausible place for that. Updated
            # with a plain .item() at each write() call site (once per
            # memory-window close, not per token -- cheap enough that the
            # sync-avoidance reasoning above doesn't apply here).
            "w1_abs_max": 0.0, "w2_abs_max": 0.0,
            # alpha is write()'s (1 - alpha) weight decay on M -- the one
            # knob that erases memory content directly (beta/retain gate the
            # ssm_state injection, not M itself), so it gets its own stat.
            # Sum-to-average over write() calls.
            "alpha": 0.0,
            # Running-MIN counterparts to the means/maxes above: retain's
            # mean can hide an episodic near-zero wipe, and w*_abs_max's max
            # can hide a mostly-zeroed matrix behind one large element.
            # w*_rms_min is the smallest per-slot RMS seen this window (a
            # healthy fresh slot sits near its init RMS, bound/sqrt(3):
            # ~0.011 for w1, ~0.0057 for w2). None = no sample yet this
            # window (reported as nan).
            "retain_min": None,
            "w1_rms_min": None, "w2_rms_min": None,
            # Mean per-slot RMS distance of M's weights from their
            # per-example random init: how much has cumulatively been
            # written this example. Sum-to-average over write() calls.
            "w1_drift": 0.0, "w2_drift": 0.0,
        }

    def _accum_write_stats(self, nm: _NeuralMemory, alphas: torch.Tensor) -> None:
        """Accumulate the write-cadence stats (see _fresh_mem_stat_sums).
        Called once per memory-window close at each write() call site --
        cheap enough for plain .item() syncs, unlike the per-token sites."""
        sums = self._mem_stat_sums
        w1 = nm.w1.detach()
        w2 = nm.w2.detach()
        sums["w1_abs_max"] = max(sums["w1_abs_max"], w1.abs().max().item())
        sums["w2_abs_max"] = max(sums["w2_abs_max"], w2.abs().max().item())
        sums["alpha"] += alphas.detach().mean().item()
        w1_rms = w1.pow(2).mean(dim=(1, 2)).sqrt().min().item()
        w2_rms = w2.pow(2).mean(dim=(1, 2)).sqrt().min().item()
        sums["w1_rms_min"] = w1_rms if sums["w1_rms_min"] is None else min(sums["w1_rms_min"], w1_rms)
        sums["w2_rms_min"] = w2_rms if sums["w2_rms_min"] is None else min(sums["w2_rms_min"], w2_rms)
        if not hasattr(nm, "w1_init"):
            nm.w1_init, nm.w2_init = w1.clone(), w2.clone()
        sums["w1_drift"] += (w1 - nm.w1_init).pow(2).mean(dim=(1, 2)).sqrt().mean().item()
        sums["w2_drift"] += (w2 - nm.w2_init).pow(2).mean(dim=(1, 2)).sqrt().mean().item()

    def pop_memory_stats(self) -> dict[str, float] | None:
        """Returns averages since the last call (None if forward hasn't run
        since then) and resets the running sums. `beta`/`retain` are the most
        direct usage signal: both start near their no-op init (beta~0.05 via
        beta_anneal_offset, retain~0.98, see _GatedDeltaInjection.__init__) so
        the backbone behaves like the unmodified pretrained model until
        training moves them -- beta staying near its init over many steps
        (well past BETA_BIAS_ANNEAL_STEPS, once the anneal offset is gone)
        means the memory is still effectively disconnected, not actually
        writing into ssm_state, regardless of how much gradient the memory
        params receive in preflight."""
        if self._mem_stat_tok_count == 0:
            return None
        # Single sync here for the whole accumulation window, instead of one
        # per injected layer per token (see the accumulation sites above).
        # _item: a sum a given integration mode never accumulates (e.g.
        # retain in "mix") is still the float 0.0 it was initialized as.
        def _item(x: "float | torch.Tensor") -> float:
            return x.item() if isinstance(x, torch.Tensor) else float(x)

        stats = {
            "beta": _item(self._mem_stat_sums["beta"] / max(self._mem_stat_inj_count, 1)),
            "retain": _item(self._mem_stat_sums["retain"] / max(self._mem_stat_inj_count, 1)),
            "surprise": _item(self._mem_stat_sums["surprise"] / self._mem_stat_tok_count),
            "o_t_norm": _item(self._mem_stat_sums["o_t_norm"] / self._mem_stat_tok_count),
            # Divided by _mem_stat_write_count, not _mem_stat_tok_count --
            # grad_norm only gets a fresh value on tokens that close a
            # memory-window (see set_memory_window), so at memory_window > 1
            # there are fewer writes than tokens.
            "grad_norm": _item(self._mem_stat_sums["grad_norm"] / max(self._mem_stat_write_count, 1)),
            # Already plain Python floats (see _mem_stat_sums init) -- a
            # running max, not a sum-to-average, so no division here.
            "w1_abs_max": self._mem_stat_sums["w1_abs_max"],
            "w2_abs_max": self._mem_stat_sums["w2_abs_max"],
            "alpha": self._mem_stat_sums["alpha"] / max(self._mem_stat_write_count, 1),
            "retain_min": (
                self._mem_stat_sums["retain_min"].item()
                if self._mem_stat_sums["retain_min"] is not None
                else float("nan")
            ),
            "w1_rms_min": (
                self._mem_stat_sums["w1_rms_min"]
                if self._mem_stat_sums["w1_rms_min"] is not None
                else float("nan")
            ),
            "w2_rms_min": (
                self._mem_stat_sums["w2_rms_min"]
                if self._mem_stat_sums["w2_rms_min"] is not None
                else float("nan")
            ),
            "w1_drift": self._mem_stat_sums["w1_drift"] / max(self._mem_stat_write_count, 1),
            "w2_drift": self._mem_stat_sums["w2_drift"] / max(self._mem_stat_write_count, 1),
        }
        self._mem_stat_sums = self._fresh_mem_stat_sums()
        self._mem_stat_inj_count = 0
        self._mem_stat_tok_count = 0
        self._mem_stat_write_count = 0
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
    extend_embeddings(model, len(tokenizer), tokenizer)
    return model


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device)).to(device)
