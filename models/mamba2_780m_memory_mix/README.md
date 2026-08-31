# mamba2_780m_memory_mix

The **token-mix arm** of the stage-2 780M screening A/B
(`notes/discussion/DISCUSSION-20260723-780m-integration-screen.md`, BX1 — the bet).
One folder per integration arm (the state-injection arm is
`models/mamba2_780m_memory_state/`), so `MODEL_NAME` selects the arm and
each arm's checkpoints live under its own `checkpoints/`. **Never compare
delta magnitudes across scales.**

## Source / Tokenizer / LoRA targets / Special tokens

Identical to `models/mamba2_780m_memory_state/README.md`
([`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m),
48 layers, d_model 1536, bf16; `EleutherAI/gpt-neox-20b`; `in_proj`/
`out_proj`; `[USER]`/`[ASSISTANT]`).

## `Model` wrapper quirks — the mix integration

Implementation is `models/mamba2_2_7b_memory/model.py`'s `Model` with
`integration="mix"` (`_TokenMixInjection`); this package only binds the
boundary. The front-end reads the residual stream entering **layer 16**
(the layer-21/22-boundary analog at 1/3 depth) and the gated read is
**added back onto the same token's stream at that same point**:
`h ← h + β_t·W_o(down(o_t))`, where `down`/β share `_GatedDeltaInjection`'s
bottleneck/gate structure (128-dim `down`, β from o_t + sigmoid(surprise) +
the same anneal offset) and `W_o` (128 → d_model) is **zero-initialized** —
exact no-op at init, gradient to `W_o` from step 0 since β's sigmoid is
nonzero. No `ssm_state` injections anywhere; reads/mix are per token
against the window-start M, writes per window (write semantics identical to
the state arm). No fixed mixing ratio: the magnitude equilibrium is learned
via `W_o`/β under the LM loss.

**The boundary is FIXED at 16 — deliberately no `MEMORY_READ_LAYER`
support here** (the state arm keeps that knob for its one conditional cross
cell): this arm's thesis is early entry — the whole upper stack computes
over the read — a 2/3 variant would never ship, and a fixed boundary means
a checkpoint's geometry can never contradict its folder name.

**Gradient checkpointing** comes from the shared implementation unchanged —
`train_hooks.set_grad_checkpoint` re-exports it, and `sft/training/loop.py` turns it
on per data slice so the chunk-512 cram slices pay the recompute tax and the
short-chunk slices don't. See `models/mamba2_2_7b_memory/README.md`'s
"Gradient checkpointing" for the block-size arithmetic and for why the
write step's gradient had to become closed-form first.

**A/B purity vs the state arm:** identical front-end, M, per-window Titans
writes, and gate-signal computation; the difference is where the gated read
lands (additive residual term vs gated-delta on `ssm_state`). Known
asymmetries, accepted: the read point differs (16 vs 32 — same-token
causality forces this arm to read where it lands; de-risked by the 2.7B
layer-21 recoverability result and by from-scratch training), and parameter
count differs (1 mix module vs 16 injection modules).

## Known subtlety, noted rather than built

Residual RMS grows with depth, so the mix term's effective ratio shrinks as
activations grow along the stack; if β saturates compensating, the fix is
an RMS-reference normalization in the mix branch. Watch β in the training
logs (`mix_w_norm` in `extra_log` tracks the zero-init `W_o` waking up)
before building anything.
