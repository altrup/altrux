# mamba2_780m_memory_state

The **state-injection arm** of the stage-2 780M screening A/B
(`notes/discussion/DISCUSSION-20260723-780m-integration-screen.md`): the
`mamba2_2_7b_memory` design on the 780M backbone, layer indices scaled to
the same fractional depths. One folder per integration arm (the token-mix
arm is `models/mamba2_780m_memory_mix/`), so `MODEL_NAME` selects the arm
and each arm's checkpoints live under its own `checkpoints/` — a
checkpoint's folder tells you its architecture. Mechanism-level comparisons
run here ~3–4× faster than at 2.7B; winners get confirmed there. **Never
compare delta magnitudes across scales.**

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) —
48 layers, d_model 1536. Loaded in bf16 (~1.6 GB).

## Tokenizer

`EleutherAI/gpt-neox-20b` — same as every mamba2 model here (see
`models/mamba2_780m/README.md`), which is what lets the 2.7B-prepped data
`.pt` files feed this model unchanged.

## LoRA target modules

`in_proj`, `out_proj` — same choice and rationale as `mamba2_2_7b_memory`.

## Special tokens

`[USER]`/`[ASSISTANT]`, bare markers, callers append the `" "` separator —
identical convention to the other models.

## `Model` wrapper quirks

The implementation is `models/mamba2_2_7b_memory/model.py`'s `Model` (which
derives all memory geometry — nheads/headdim/d_state, `mem_dim = d_model`,
`mem_hidden = 4·d_model` — from the backbone); this package only binds the
layer indices. Everything in the 2.7B README (manual mixer fallback on the
local ROCm box, fused span dispatch on real CUDA, windowed writes, beta
anneal, sleep/reset slots) applies unchanged.

- `READ_LAYER = 32` (2/3 depth, from 42/64), `INJECTED_LAYERS = 16..46
  step 2` (from 22..62 step 2; 16 injected layers). Gated-delta merge of
  the read into `ssm_state`.
- **`MEMORY_READ_LAYER` env var** (read at construction) overrides the read
  layer for the ONE conditional cross cell, state@16 — run only if mix@16
  disappoints, to separate "shallow q/k/v inadequate" from "integration
  mechanism at fault". The mix arm deliberately has no such knob (its read
  point is the mix point, fixed at 16 by that arm's thesis), so a
  checkpoint's geometry can never contradict its folder name — except under
  this override: **note `MEMORY_READ_LAYER` in the run log whenever it is
  set**, since checkpoints don't record it.
- The 128-dim read/injection bottleneck (`BOTTLENECK_R`) is kept at 128,
  not scaled (~77 proportionally): relative to d_model 1536 it's a slightly
  wider straw — the safe direction for a screen, and read-out width is a
  deferred ceiling suspect we want untangled from the scale change. Applies
  to both arms.

See `models/mamba2_780m_memory_mix/README.md` for the A/B purity contract
and the arms' known asymmetries.
