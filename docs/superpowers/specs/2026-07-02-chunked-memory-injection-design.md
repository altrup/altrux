# Chunked memory injection for mamba2_2_7b_memory

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
  1. `M`'s actual weight update — solved via the closed-form/associative-scan
     form of the momentum recurrence (`S_t = η·S_{t-1} - θ·∇L_t`,
     `M_t = (1-α)·M_{t-1} + S_t`), since it's linear given the per-token
     `η`/`θ` gates. The only approximation in the whole scheme is here:
     `∇L_t` for every token in the window is computed against the
     window-start `M`, not a continuously-updated one.
  2. Injection into `ssm_state` — one consolidated write per window,
     content-derived from a **surprise-weighted aggregation of every
     token's `o_t`/`surprise_t` in the window** (not just the last token's),
     since those per-token values are already available for free from the
     parallel read step above.

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

### Fused-kernel dispatch for the un-injected spans

Between memory-window boundaries, on a CUDA host with `causal-conv1d`
importable, backbone layers should route through `mamba_ssm`'s native
fused/chunked forward instead of the manual `_mixer_step` loop. Dispatch is
by feature detection (device type + successful `causal_conv1d` import/probe
at model load), not a manual flag — so the ROCm dev box keeps working
without the caller needing to remember anything. The manual per-token loop
remains, unconditionally, at the memory-window boundary itself (where the
injection happens) on every hardware target, since that's the piece that
still needs per-token state access regardless of GPU.

`sft/pyproject.toml` / `backend/pyproject.toml`'s "causal-conv1d
deliberately not installed" needs to become hardware-conditional instead of
blanket, so the H100 environment actually installs it.

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
  formulation** (computing all 24 individual `ssm_state` merges exactly, in
  parallel, rather than one aggregated write per window) — a real technique
  that exists in the linear-attention literature, but real custom-kernel
  engineering, the same scope of work already set aside earlier in this
  project for cost reasons. Flagged as a future upgrade if 24-token
  resolution turns out to hurt quality in practice, not part of this design.
- Fully dynamic (non-windowed) injection timing — incompatible with the
  parallelism goal by construction (see discussion in prior conversation);
  surprise-weighted pooling within a fixed window is the chosen middle
  ground instead.

## Testing / rollout

The real model can't be run end-to-end on the local ROCm box (OOMs at
~7.5GB base weights alone on an 8GB card) and the fused-kernel path can't be
exercised locally at all (ROCm `causal-conv1d` confirmed broken). So:

- Correctness of the windowed accumulate/apply math, checkpoint save/load,
  and legacy-mode (`window=1`) equivalence to today's behavior should be
  validated locally via `smoke_test.py` at reduced scale (small batch,
  low LoRA rank, synthetic data) — this exercises the real code path, just
  at a size that fits in 8GB.
- The `memory-window` value itself should be swept cheaply (synthetic data,
  small scale) before committing real H100 hours to a full run — no
  principled reason to prefer one value in the teens/twenties over another
  without empirical signal.
- The fused-kernel dispatch is the one piece that genuinely cannot be
  validated from this environment and needs to be smoke-tested on the H100
  directly before being trusted for a real paid run.
