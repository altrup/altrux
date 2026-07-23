# mamba2_780m_memory

The stage-2 **screening platform**: the `mamba2_2_7b_memory` Titans memory
subsystem on the 780M backbone, with an integration-point knob for the
state-injection vs token-mix A/B (`notes/DISCUSSION-20260723-780m-integration-screen.md`,
BX0/BX1). Mechanism-level comparisons run here ~3–4× faster; winners get
confirmed at 2.7B. **Never compare delta magnitudes across scales.**

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) —
48 layers, d_model 1536. Loaded in bf16 (~1.6 GB).

## Tokenizer

`EleutherAI/gpt-neox-20b` — same as every mamba2 model here (see
`models/mamba2_780m/README.md`), which is what lets the 2.7B-prepped data
`.pt` files feed this model unchanged.

## LoRA target modules

`in_proj`, `out_proj` — same choice and rationale as `mamba2_2_7b_memory`:
a fully frozen backbone is unlikely to integrate an injected memory signal,
and these projections dominate per-block parameter count.

## Special tokens

`[USER]`/`[ASSISTANT]`, bare markers, callers append the `" "` separator —
identical convention to the other models (see `models/mamba2_780m/README.md`).

## `Model` wrapper quirks

The implementation is `models/mamba2_2_7b_memory/model.py`'s `Model`, which
derives all memory geometry (nheads/headdim/d_state, `mem_dim = d_model`,
`mem_hidden = 4·d_model`) from the backbone; this package only binds the
780M layer indices and the integration mode. Everything in the 2.7B README
(manual mixer fallback on the local ROCm box, fused span dispatch on real
CUDA, windowed writes, beta anneal, sleep/reset slots) applies unchanged.

### Integration knob — `MEMORY_INTEGRATION` env var, read at construction

`MEMORY_READ_LAYER` overrides the arm's default read layer (state@32 /
mix@16) for the 2x2 cross cells — state@16, mix@32. In the mix arm the
read point is also the mix point (same-token causality).

- `state` (default): the 2.7B design with layer indices scaled by 48/64 to
  the same fractional depths — `READ_LAYER = 32` (2/3 depth, from 42/64),
  `INJECTED_LAYERS = 16..46 step 2` (from 22..62 step 2; 16 injected
  layers). Gated-delta merge of the read into `ssm_state`.
- `mix`: token-mix integration (`_TokenMixInjection`). Front-end reads the
  residual stream entering layer 16 (the layer-21/22-boundary analog at
  1/3 depth, per the DISCUSSION spec) and the gated read is **added back
  onto the same token's stream at that same point**: `h ← h + β_t·W_o(down(o_t))`,
  where `down`/`β` share `_GatedDeltaInjection`'s bottleneck/gate structure
  (128-dim `down`, β from o_t + sigmoid(surprise) + the same anneal offset)
  and `W_o` (128 → d_model) is **zero-initialized** — exact no-op at init,
  with gradient flowing to `W_o` from step 0 since β's sigmoid is nonzero.
  No `ssm_state` injections anywhere. No fixed mixing ratio: the magnitude
  equilibrium is learned via `W_o`/β under the LM loss.

**A/B purity:** the two arms share identical front-end, M, per-window
Titans writes, and gate-signal computation (o_t + surprise → sigmoid, same
anneal). The only difference is where the gated read lands: gated-delta on
`ssm_state` vs additive residual term. One unavoidable exception: the
**read point moves with the arm** (32 vs 16) — the mix arm must read where
it lands (same-token causality), while the state arm keeps the proven 2.7B
geometry. Known confound of the A/B; de-risked by the layer-21
recoverability result (L1, `notes/DISCUSSION-20260722-stage2-readout.md`:
~80% of the front-end's directional content is linearly present at 1/3
depth) and by from-scratch training (the front-end simply learns at its
layer; nothing transfers).

### Sizing decisions (team-agreed, 2026-07-23)

- Dims deriving from d_model scale down naturally: M's fast-weight MLP is
  ~19M values per sequence here vs ~52M at 2.7B — expected and fine;
  capacity is not the bottleneck.
- The 128-dim injection/read bottleneck (`BOTTLENECK_R`) is **kept at 128,
  not scaled** (~77 proportionally): relative to d_model 1536 it's a
  slightly wider straw, the safe direction for a screen — a narrower
  channel could strangle both arms and mask their difference — and
  read-out width is itself a deferred ceiling suspect we want untangled
  from the scale change. Applies to both integration modes.

### Known subtlety, noted rather than built

Residual RMS grows with depth, so the mix arm's effective memory-term ratio
shrinks as activations grow along the stack; if β saturates compensating,
the fix is an RMS-reference normalization in the mix branch. Watch β in the
training logs (`mix_w_norm` in `extra_log` tracks the zero-init `W_o`
waking up) before building anything.
