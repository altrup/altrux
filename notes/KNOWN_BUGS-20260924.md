# Known bugs, 2026-09-24

Found by a read-only review of the then-protected files. Unfixed because the
code is off the LAMA-CKL path (memory model, erasure, chains/cram, diagnostics)
or inside a file the person is rewriting by hand. Fixed the same day:
`sft/training/loop.py` (train mask unused, resume skipping examples, resume
losing a pending sleep).

## In the protected list (fix during the rewrite)

- `sft/experiments/dreams/distillation.py:204,347`: when the step count is
  not a multiple of `accum`, the last partial window is backpropagated but
  never stepped or zeroed, so it leaks into the next wave's first step.
- `sft/experiments/lama_ckl/protocol.py:120-136` `lama_dream_diagnostics`:
  rows sharing a subject count a correct sentence as a misbinding; correct
  bindings are a per-dream set while misbindings count per sentence; objects
  match case-sensitively, subjects do not; substring matches ("Ada" in
  "Canada"); sentences split on "." so names with periods never match;
  `contradictions` is a copy of `misbindings`.
- `sft/experiments/dreams/generation.py` `_teacher_dream_batch` ignores
  `max_turns` and hard-codes `DREAM_MAX_TURNS`. No test that batched and
  single-stream dreams agree.
- `sft/experiments/lama_ckl/upstream.py:16`: `TAALM_COMMIT` is never compared
  against the checkout; `load_official_result` assumes epoch keys start at 0.
- `sft/experiments/lama_ckl/split.py`: the LAMA source hash is recorded but
  never compared; nothing checks for the same subject across cohorts.
- `sft/experiments/lama_ckl/evaluation.py`: `rfind` assumes [Y] is the last
  slot in the template; tests use a whitespace fake tokenizer, so GPT-NeoX
  offset behaviour (leading space, BOS) is untested.

## Memory model (`models/mamba2_2_7b_memory/`)

- `model.py:1596` fused path: the outer loop is over layers, the inner over
  windows, but `state.last_o_t` only updates during the read layer's window
  loop. Layers after 42 see the chunk's last window's pooled read in every
  window (future tokens reach earlier logits); layers before 42 see the
  previous chunk's. Correct only when chunk_len == memory_window. No test
  compares fused against manual.
- `model.py:377-387,872`: training at memory_window > 1 vs inference at W=1
  decays M about W times faster per token. README says they are consistent.
- `model.py:987`: `causal_conv1d_update` is in-place with no autograd on
  CUDA; either crashes at backward or drops gradient into in_proj.
- `model.py:998` vs `:1125`: ssm_state is fp32 after `_mixer_step` but cast
  back to bf16 by `_mixer_span`.
- `model.py:822`: `MarkerDelta.delta` is a bf16 parameter; small AdamW
  updates round to zero.
- `train_hooks.py:36` comment says `DEFAULT_CHUNK_LEN=12`, value is 7.
  README line 68 says surprise is detached (it is not); line 130 calls it a
  gradient magnitude (it is the loss).

## Erasure (`sft/experiments/erasure/`)

- `operators.py:110` with `gating.py:150`: `sigma_gammas` pairs gamma i with
  spectrum index i, but the qcm basis starts at v2, so every direction gets
  the wrong strength.
- `operators.py:137`: a one-entry gamma list broadcasts silently over a
  multi-row basis.
- `probe.py:268`: the `startswith(truth + " ")` prefix match applies to all
  bystanders, not only nearcone.
- `probe.py:466`: `--deflate state-svd` deflates against the state being
  edited while the overlap panel at `:382` uses the primed state.

## Preparation (`sft/preparation/`)

- `chains.py:593`: `single_token_labels` is used but never imported;
  `make prepare-chains` raises NameError.
- `chains.py:187`: a non-QA split cut can land on an assistant marker,
  leaving an unanswered user turn. Fires only with `--split-episode-rate > 0`.
- `chains.py:329-331`: fact blocks insert consecutive user turns at
  boundaries that include assistant markers, splitting question from answer.
  `--fact-rate` still defaults to 0.3.
- `chains.py:359`: `within_episode` uses the block's episode, not the
  revision's; stats only.
- `cram.py:311`: items still pending when the budget closes a block are
  consumed without a cue.
- `filtering.py:401`: loads whatever `MODEL_NAME` is set without checking it
  is a plain backbone; `aliases()` treats any word of 4+ characters as a
  leak.

## Diagnostics (`sft/diagnostics/`)

- `recall.py:265`: gist `no-wipe-ablated` under fresh-m does not zero
  `last_o_t`/`last_surprise`; `:224` with `--n-probes 1` the distractor is
  the row's own opening.
- `reads.py:159`: random token-level train/test split over contiguous text
  inflates held-out R².
