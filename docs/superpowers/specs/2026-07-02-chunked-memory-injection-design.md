# Chunked memory injection for mamba2_2_7b_memory

## Implementation status

- **Done**: memory-window write batching, surprise-weighted-pooled injection
  batching to the window boundary, `--memory-window` CLI flag + checkpoint
  persistence, hardware-conditional `causal-conv1d` install, and the
  fused-kernel backbone dispatch itself (`Model._forward_fused`/
  `_mixer_span`/`fused_kernel_usable` in `model.py`): the 43 layers never in
  `INJECTED_LAYERS` run through `mamba_ssm`'s native
  `mamba_chunk_scan_combined` for a whole chunk in one call; the 21 injected
  layers run per-window (one fused span over the `window - 1` non-injection
  tokens, then one manual `_mixer_step` for the window-closing token).
  Feature-detected via `fused_kernel_usable` (CUDA, not ROCm/HIP,
  `causal_conv1d` importable) — this repo's local ROCm dev box keeps using
  the original all-manual `_forward_manual` path unconditionally. Covered by
  `models/tests/test_mamba2_2_7b_memory_windowing.py`: `_NeuralMemory` and
  the pooling/gating logic directly, the windowed
  `read_windowed`/`surprise_windowed` helpers against their per-token
  equivalents, `fused_kernel_usable`'s feature-detection branches, and the
  window-span-boundary arithmetic `_forward_fused` uses — the full `Model`
  (and the fused kernel calls themselves) still can't be exercised
  end-to-end locally; see "Testing/rollout" below, which is now the
  load-bearing next step before trusting this for a real paid run.
- **Not done**: the exact intra-window scan (write still uses the coarser
  "one averaged step" approximation) — explicitly deferred, see "Non-goals"
  below for why.

## Problem

`mamba2_2_7b_memory` trains via a manual per-token Python loop over all 64
backbone layers (`Model.forward`, see `model.py`), because the gated-delta
memory merge at 21 injected layers needs per-token, pre-readout access to
each layer's `ssm_state` — a hook Mamba2's native fused/chunked kernel
doesn't expose. This pays the same asymptotic cost as autoregressive decode,
for training, on a 2.7B/64-layer model. On a rented H100 at $2.50/hr this is
the dominant cost driver, and the memory subsystem is currently so
undertrained (~165 steps into one epoch) that behavior hasn't diverged from
base-model completion.

Two causes were conflated at first and need to stay separate:

1. **Architectural**: per-token `ssm_state` injection blocks the fused
   kernel, on any hardware.
2. **Environmental**: this repo's local ROCm dev box (`gfx1102`) can't use
   `causal-conv1d`/Triton at all regardless of architecture — re-confirmed
   2026-07-02, segfaults (exit 139) via `mamba_ssm`'s own `Mamba2` module
   forward, not just synthetic calls. This is specific to that box; the H100
   training target has `causal-conv1d` working (already used there for the
   non-memory model).

This design addresses (1). (2) means the fused-kernel path this design
enables must be feature-detected (CUDA + working `causal-conv1d`), not
assumed — the ROCm box keeps the existing manual loop as a fallback.

## Scope

`mamba2_2_7b_memory` only. `mamba2_780m_memory` has been deleted (fully
superseded by the 2.7B model going forward; no longer maintained).

## Design

### Memory-update window, decoupled from BPTT chunk length

New `--memory-window` CLI arg, independent of
`--chunk-len`. Constraint: `chunk-len` must be evenly divisible by
`memory-window`, because a memory-window can never span a `chunk_loss`
backward()/detach() boundary — gradient flow through an already-freed graph
isn't possible. `memory-window` defaults to something swept empirically
(see Testing/rollout below), not guessed; `memory-window = 1` reproduces
today's exact per-token behavior as a special case, not a separate code
path.

### What stays per-token vs. what gets batched, within a window

- `q_t`, `k_t`, `v_t`: computed for every token, in parallel — plain linear
  projections, no sequential dependency, never the cost driver.
- **Read**: `o_t = M_frozen(q_t)`, computed for every token in parallel,
  using `M` as it stood at the start of the window (frozen for the window's
  duration). This is a batched forward pass through a fixed-weight MLP —
  free relative to a sequential loop. `surprise_t` likewise per-token.
- **Write (apply), consolidated once per window boundary**:
  1. `M`'s actual weight update. **Implemented as**: every token's loss
     (against the window-start `M`) is summed and differentiated in one
     `autograd.grad` call, then one momentum step is applied using the
     window's *averaged* `η`/`θ`/`α` — window=1 is exact by construction
     (see `_NeuralMemory.write`'s docstring). This is a coarser
     approximation than Titans' own exact approach (see "Non-goals" below
     for why the exact version is deferred, not just a smaller patch).
  2. Injection into `ssm_state` — one consolidated write per window,
     content-derived from a **surprise-weighted aggregation of every
     token's `o_t`/`surprise_t` in the window** (not just the last token's),
     since those per-token values are already available for free from the
     parallel read step above. **Implemented** as a softmax-over-surprise
     weighting (`Model.forward`'s pooling block); window=1 reduces to a
     softmax over one element (weight 1.0), reproducing the original
     per-token injection exactly. For the `window - 1` non-closing tokens in
     each window, every injected layer gets `gated_delta = None` — `ssm_state`
     evolves under Mamba2's own unmodified dynamics for those tokens.

### `accumulate_and_apply(window)` — one function, two call sites

Both training (window = swept value, e.g. in the teens/twenties) and
inference (window = 1, called every generated token) go through the same
function, parameterized by window size — not two separate implementations.
This mirrors the Titans paper's own `b ≥ 1` chunk formulation (confirmed via
the paper directly: Section 3.2 defines chunk size `b ≥ 1`, so `b=1`
degenerates cleanly rather than being a special case). This is what
guarantees inference never sees behavior the model wasn't trained to
produce, and directly resolves the "does the model answer before an
injection happens" concern — inference is already token-by-token regardless
of training's window size, so there's no reason to gate it to the training
window.

A training-time **legacy mode** (`memory-window = 1` during training, not
just at inference) stays available and cheap to reach, since it's the same
code path — this is intended to be used for a final fine-tuning phase after
resuming from a batched-mode run, to give the model real per-token training
signal rather than relying solely on the `b≥1` formulation transferring
cleanly.

### Fused-kernel dispatch for the un-injected spans — **implemented**

`Model._forward_fused` (`model.py`) restructures the backbone loop from
token-major to layer-major, mirroring `mamba_ssm`'s own full-sequence
`MixerModel.forward`: for the 43 layers never in `INJECTED_LAYERS`, the
whole chunk runs through one `_mixer_span` call (wrapping `causal_conv1d_fn`
+ `mamba_chunk_scan_combined`, `mamba_ssm`'s native fused/chunked forward)
instead of `seqlen` manual `_mixer_step` calls. For the 21 injected layers,
each memory-window is handled as one `_mixer_span` call over its
`window - 1` leading tokens, followed by one manual `_mixer_step` call for
the window-closing token — the only token that still needs per-token,
pre-readout state access (layer `READ_LAYER`'s own read/write additionally
needs every token's residual in the window, not just the boundary token's —
see `_NeuralMemory.read_windowed`/`.surprise_windowed`, which do that
without a sequential loop).

Dispatch is by feature detection (`fused_kernel_usable`: CUDA device type
that isn't actually ROCm/HIP, plus a successful `causal_conv1d` import at
module load), not a manual flag — so the ROCm dev box keeps working without
the caller needing to remember anything; `Model.forward` routes to
`_forward_fused` when available and to `_forward_manual` (the original,
byte-for-byte-unchanged all-manual loop) otherwise.

`sft/pyproject.toml` / `backend/pyproject.toml`'s "causal-conv1d
deliberately not installed" is already hardware-conditional (installed only
when a non-ROCm CUDA GPU is detected — see `backend/Makefile`/
`sft/Makefile`), so the H100 environment already installs it; this was
completed in a prior commit, ahead of the dispatch code that actually uses
it.

**Not independently verified end-to-end**: the kernel calls in
`_mixer_span` (conv1d state continuity across window/chunk boundaries via
manual roll-buffer reconstruction, `mamba_chunk_scan_combined`'s
`initial_states`/`return_final_states` shapes, and the layer-major
restructuring's numerical equivalence to `_forward_manual`) were built
against the installed `mamba_ssm` source directly (not guessed from
memory), matching the call conventions the library's own `Mamba2.forward`
uses — see `_mixer_span`'s docstring for the specific reasoning at each
step — but `causal-conv1d` can't be installed or exercised on this repo's
local ROCm dev box at all (see root CLAUDE.md), so none of this has run.
**Needs a real smoke test on the H100** (or equivalent CUDA host) before
being trusted for a paid training run — see "Testing/rollout" below.

### Checkpoint compatibility

None of this changes any module shapes (front-end MLP, per-layer injection
signal generators are unchanged) — only *when* accumulated results get
applied. The window size in effect for a training run is saved into
checkpoint config (same place `lora_config.json` already lives), so a
resume knows what it was trained with. Resuming the current
`epoch-1/step-163` checkpoint under the new batched code, then switching to
`memory-window=1` for a final stretch, is expected to work without
restarting from scratch.

## Non-goals / explicitly deferred

- **Exact per-token injection via a DeltaNet-style chunked/WY-representation
  formulation** (computing all `window` individual `ssm_state` merges
  exactly, in parallel, rather than one aggregated write per window) — a
  real technique that exists in the linear-attention literature, but real
  custom-kernel engineering, the same scope of work already set aside
  earlier in this project for cost reasons. Naively buffering each token's
  read and "applying them all at the window boundary" is NOT equivalent to
  this and was considered and rejected: the gated-delta merge interacts with
  Mamba2's own decay/write between tokens, so firing `window` merges
  back-to-back with no natural dynamics in between them isn't a deferred
  version of the same computation, it's a different (and likely degenerate)
  one. Flagged as a future upgrade if the current window-boundary-only
  injection turns out to hurt quality in practice, not part of this design.
- **Exact intra-window scan for the memory write** (reconstructing the true
  per-token `S_t`/`M_t` trajectory via a parallel associative scan, matching
  Titans arXiv:2501.00663 §3.2 exactly, instead of the coarser "sum losses,
  one averaged step" actually implemented). Initially estimated as a small
  addition; turned out not to be, once `MEM_HIDDEN = 4×D_MODEL = 10240` is
  accounted for — naively storing a per-token trajectory of `M`-shaped
  tensors (`(batch, hidden, dim)`) across a window is tens of GB, not
  viable. A memory-efficient version is possible by exploiting that a linear
  layer's per-example gradient is always a rank-1 outer product (the same
  trick DeltaNet-style linear attention uses to stay matrix-free), but that
  makes this comparable in difficulty/risk to the DeltaNet-style injection
  work above, not a quick PyTorch upgrade. Deferred; needs its own proper
  scoping (and validation at small `MEM_HIDDEN` before trusting it at the
  real size) rather than being rushed in alongside the window-boundary
  injection work.
- Fully dynamic (non-windowed) injection timing — incompatible with the
  parallelism goal by construction (see discussion in prior conversation);
  surprise-weighted pooling within a fixed window is the chosen middle
  ground instead.

## Testing / rollout

The real model can't be run end-to-end on the local ROCm box (OOMs at
~7.5GB base weights alone on an 8GB card) and the fused-kernel path can't be
exercised locally at all (ROCm `causal-conv1d` confirmed broken — on this
box, `Model.forward` always takes the `_forward_manual` path regardless of
this feature existing). So:

- Correctness of the windowed accumulate/apply math, checkpoint save/load,
  and legacy-mode (`window=1`) equivalence to today's behavior should be
  validated locally via `smoke_test.py` at reduced scale (small batch,
  low LoRA rank, synthetic data) — this exercises the real code path, just
  at a size that fits in 8GB. The windowed read/surprise math the fused
  path's `READ_LAYER` handling depends on (`_NeuralMemory.read_windowed`/
  `.surprise_windowed`) is covered directly by
  `models/tests/test_mamba2_2_7b_memory_windowing.py` against their
  per-token equivalents, independent of any GPU.
- **Before the first real H100 training run under this feature**: confirm
  `Model._fused_path_available()` actually returns `True` there (i.e.
  `causal-conv1d` installed successfully via the conditional `make sync`
  path), then run `make smoke-test` and diff `_forward_fused`'s logits
  against `_forward_manual`'s for the same input on a small
  synthetic/reduced-scale config (temporarily force `_fused_path_available`
  to `False` for one of the two runs) — they should match to floating-point
  tolerance. This is the one piece of this design that has not been run at
  all yet, in any form, and needs that hands-on validation before being
  trusted for a paid run — do not assume it works from the code review
  alone.
- The `memory-window` value itself should be swept cheaply (synthetic data,
  small scale) before committing real H100 hours to a full run — no
  principled reason to prefer one value in the teens/twenties over another
  without empirical signal.
- The fused-kernel dispatch is the one piece that genuinely cannot be
  validated from this environment and needs to be smoke-tested on the H100
  directly before being trusted for a real paid run.
