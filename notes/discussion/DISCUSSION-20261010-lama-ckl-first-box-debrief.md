# Discussion notes — 2026-10-10: LAMA-CKL first box run debrief

Debrief of the first LAMA-CKL box session
([`EXPERIMENT_NOTES-20261010-051749`](../experiments/EXPERIMENT_NOTES-20261010-051749.md),
branch `box/20261010-041135`). It amends the protocol in
[`DISCUSSION-20260823`](DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md)
and supersedes the recap-0.5 warm start registered there. Where this note
and the 2026-08-23 note disagree, this note wins.

## 1. What the run established

No cell trained. The session stopped at cycle 1 of the first altrux cell.
Five facts are load-bearing for what follows:

1. **The recap-0.5 warm start does not reply to a bare document.** 445 to
   458 of 500 wake replies ran past the 64-token backstop; the model
   continues the document in Wikipedia style. The registered wake
   invariant (EOS before the backstop) is unsatisfiable with that adapter
   and that rendering.
2. **The recap behaviour collapsed to a self-quote.** 39 of 300 dreams were
   byte-identical copies of the synthetic recap exchange quoting itself
   (`You asked: "What did I ask you about earlier?"`). The cause is in the
   corpus: `pack_records` in `sft/preparation/conversations.py` drew a
   recap's source from every earlier pack member, recaps included. The
   runner's duplicate-dream stop then refused to train.
3. **Dreams from a 500-document wake rehearse almost nothing.** EOC-prompt
   set: 18 of 500 to-learn subjects mentioned, 4 with their object, and the
   rehearsals cite the last documents seen. The instruction-prompt set was
   worse (5 subjects, 1 binding; the instruction is echoed). This matches
   the 3 to 4 binding state capacity measured on 2026-08-04.
4. **Decode is slow.** 15.6 tok/s single stream, about 290 tok/s at batch
   30. `_mixer_step` in `models/mamba2_780m/model.py` is plain PyTorch per
   token on every device; only multi-token input uses a fused kernel on
   CUDA.
5. **Object-token accuracy wobbles at the 1% level with evaluation batch
   size** (7 of 500 retention rows flip between batch 8 and 16), and a
   greedy wake is not bit-reproducible between runs on the GH200.

Also: the box HF token has no Llama-2 access (`meta-llama/Llama-2-7b-hf`,
403), so the upstream gate did not run. The experimenter's protected edit
`634f914` (feed EOS at the backstop, count forced closes) made four smokes
pass. The experimenter also found the "registered fallback arm" text for
the instruction prompt in the 2026-08-23 note, which the 2026-10-05
amendment had withdrawn in intent but not in wording.

## 2. Reinterpretation

The run is an apparatus failure, not a result about the method. Three
things were wrong before any treatment could act: the warm start's data
(synthetic recaps, nobody chose them knowingly), the wake rendering (a
document with no reason to reply to it), and the wake size (500 documents
into a state that holds a handful). The dream prompt itself,
`<|endofconversation|>`, is not implicated: the duplicates were the recap
loop, and the loop is the data.

Dream quality (fact 3) is a real limit of the state, and it stays a
diagnostic. The protocol change is to size the wake to the state, not to
tune the dream.

## 3. Standing direction

### 3.1 Retrain the warm start

Corpus: UltraChat (`HuggingFaceH4/ultrachat_200k`, the same 1000-example
draw) plus the three context categories of `databricks/databricks-dolly-15k`
(`summarization`, `closed_qa`, `information_extraction`: a short document
and an instruction, the same shape as a LAMA evidence document), one pool.
Packing per example:

- up to 5 conversations, `--max-len 8192`;
- each slot after the first is, with probability `--repeat-rate 0.3`, a
  **verbatim repeat** of an earlier conversation in the pack, drawn
  uniformly from the earlier distinct conversations not yet repeated;
  otherwise a fresh draw from the pool. A repeat may be adjacent
  (0, 0, 1, 2, 3) or gapped (0, 2, 0, 1, 3). No source repeats twice;
- no synthetic recap. `--recap-rate` and `recap_messages` are removed;
- loss on the repeat trains as ordinary assistant turns.

Training unchanged: 800 steps, 2 epochs, lr 1e-4, chunk 512, LoRA rank 16,
alpha 32. Output `models/mamba2_2_7b/checkpoints/repeat030/epoch-2/step-800`;
record its SHA-256 in the experiment note. The recap-0.5 checkpoint, the
split cache `.cache/lama_ckl/mamba2_2_7b_recap050/`, and the `recap050`
path convention are retired.

Why repeats and not fresh-only: packing only unrelated conversations trains
the model to ignore its state at the boundary every dream starts from
(DISCUSSION-20260808 §2.10.9). A verbatim repeat gives the same
"maybe new, maybe recall" signal with no synthetic text. Known cost: it
trains copying from state, so the dream copy fraction will rise. The
diagnostics report it.

### 3.2 Wake

One fixed frame string before every document, byte-identical for every
document, arm, and seed:

```text
[USER] Remember this for later: <evidence document>
[ASSISTANT] <greedy reply>
```

The reply is greedy with a **128-token backstop**. At the backstop the
runner feeds the tokenizer EOS (the `634f914` mechanism) and counts the
forced close. Forced closes are a per-cycle diagnostic, never fatal.
`internal_eoc` stays an invariant (must be 0).

**Wake size: 10 documents per cycle** in the official frozen order, so one
published epoch (500 documents) is 50 cycles. Fallback: 5 documents, 100
cycles, chosen by the smoke rule in §3.6. Evaluation of both fact sets runs
once per epoch, after the last cycle of the epoch, as published.

Only the altrux arm wakes. Frozen, lora, and mix-review evaluate from fresh
state and never read the wake state, so their wake is cost with no effect.
The frozen arm is one evaluation.

### 3.3 Sleep

Dream prompt unchanged: the open post-wake state plus one
`<|endofconversation|>`; the model writes everything after it; stop on a
second `<|endofconversation|>` or at 512 tokens; T 0.7.

**10 dreams per wake-sleep cycle**, `--dreams-per-cycle 10`. The
experimenter may raise it (20 to 50) from the pre-grid smoke diagnostics
only, before any cell starts; the value goes in `run.json` and every cell of
the grid shares it. Always state dream counts per cycle; per-epoch totals
are cost arithmetic only.

Distillation unchanged: one pass over every realized dream, one optimizer
step per dream, KL temperature 1.0, AdamW 1e-4.

**Duplicates are a diagnostic.** The duplicate-dream stop in `runner.py` is
removed; a repeated dream trains twice. The runner writes `wake.json` and
`dreams.json` before any check, so a refused cycle leaves its artifacts.

Event sequence for one altrux cycle, to be signed off against the code:

1. Load the carried state (fresh at cycle 0 of each epoch; see 5).
2. For each of the 10 documents: feed `[USER] Remember this for later:
   <doc>`, generate greedily to EOS or 128 tokens, feed EOS if forced.
3. Save `wake.json` (every prompt and generated token, forced-close count).
4. From a copy of the open state plus `<|endofconversation|>`, generate
   10 dreams at T 0.7, 512 tokens. Save `dreams.json` and diagnostics.
5. Carried state = open state plus one `<|endofconversation|>`, fed with
   the pre-treatment weights. The state carries across cycles within an
   epoch and resets at the epoch boundary, so every epoch reads the 500
   documents into a fresh state as the published epoch does.
6. Distil the 10 dreams, one step each.
7. If this was the last cycle of the epoch: evaluate both fact sets from
   fresh state, save the checkpoint.

### 3.4 Decode kernels

On CUDA every path uses mamba-ssm's fused kernels: the chunk scan for
multi-token input (already) and `selective_state_update` plus
`causal_conv1d_update` for single-token decode (new). The manual PyTorch
path exists for the ROCm dev box only. Each manual path keeps an
equivalence test against the kernel it stands in for. Record the rule in
`models/AGENTS.md`.

### 3.5 Split and gate

Rebuild the 500/500 split on the new warm start with the same rules and
seed 42 (`make lama-ckl-split`); the sets are conditioned on what the model
already knows. The Llama-2 reproduction (`make lama-ckl-upstream-smoke`
then `-run`, `meta-llama/Llama-2-7b-hf`) runs once as a calibration check
after altrup accepts the license; a miss is recorded and discussed, it does
not block the Mamba cells.

### 3.6 Pre-grid acceptance

Before any cell, on the new warm start:

1. The warm-start format acceptance from `sft/README.md` (marker
   emissions, no mojibake, `<|endofconversation|>` opens a well-formed
   conversation).
2. One 10-document wake and one 5-document wake, 10 dreams each, from the
   first documents of the split. Forced closes must be under 20% of
   replies. **Rule, fixed here:** take 10 documents per cycle if the
   10-document dreams mention at least half of the wake's subjects (the
   diagnostics' substring count), else 5. The dream count may be raised
   here and only here.
3. Read the decoded samples.

These are diagnostics read before any benchmark score, so they are not
tuning on the outcome.

### 3.7 Grid

Seed 42 of all four arms first (frozen, lora, mix-review, altrux), 30
epochs, same split, same `--dreams-per-cycle`, same `--docs-per-wake`.
Seeds 43 and 44 only if the seed-42 curves show a difference worth
confirming. Box session order: retrain, acceptance, split, Llama-2 check,
then the cells, in one session if acceptance passes.

Cost at the current decode speed, per altrux epoch: wake about 35 to 70
min, dreams about 40 min at 10 per cycle, distillation about 10 min,
evaluation under 1 min. About 12 h for the four seed-42 cells. The fused
decode step is expected to cut most of it.

## 4. Explicitly considered and rejected

- **Deduplicate the dream set before training.** Selection on dream
  content, which §3 of the 2026-08-23 note forbids. Duplicates are a
  diagnostic; train on all.
- **Keep the synthetic recap with a fixed source draw.** Nobody chose the
  recap exchange knowingly and it is not conversation data. Verbatim
  repeats replace it.
- **Fresh-only packing.** Trains the model to ignore its state at the
  boundary.
- **The instruction-turn dream prompt as a fallback arm.** Echoed, not
  followed (293/300 unique, 1 binding). Withdrawn; the 2026-08-23 note is
  amended.
- **A `[DREAM]` token.** Needs supervised dream examples; not reached.
- **A different benchmark, or a conversational one.** The 2026-08-21
  selection stands; the rendering was the problem, and the frame string
  fixes it without touching the facts or the probe.
- **No reply in the wake (document only, one forward pass).** Cheaper and
  byte-identical exposure, but the reply is the model's own processing of
  the document into state. Kept the reply; revisit if forced closes stay
  high.
- **Several frame strings.** A sampled frame is a confound between arms.
  One string.
- **Fewer epochs or shorter dreams for the first pass.** 30 epochs and 512
  tokens stay so the curve is comparable to the paper.
- **The Llama-2 gate as a blocker.** The upstream metric check passed; the
  reproduction is a calibration, not a precondition.

## 5. Local work before the next box

Opus subagents, TDD, one commit per slice, in this order:

1. **Packer**: `--repeat-rate`, cap 5, Dolly rows in the pool, `--max-len
   8192`; remove `--recap-rate` and `recap_messages`; `make warm-start`
   and `sft/README.md` updated. Packing invariants and decoded samples at
   one repeat boundary, adjacent and gapped.
2. **Protocol** (protected; altrup commits): frame string, backstop 128
   with forced-close count, `--docs-per-wake` (default 10),
   `--dreams-per-cycle` (default 10), per-epoch evaluation, artifacts
   written before any check, duplicate stop removed, no wake for
   non-altrux arms, `REPLY_TOKENS` 128. Fold in `634f914`.
3. **Fused decode step** on CUDA with an equivalence test; `models/AGENTS.md`
   rule.
4. **Watchdog** delay-file only, scripts heartbeat it; **launch preflight**
   that `CLAUDE_CODE_OAUTH_TOKEN` works before renting.
5. Dry-run the experimenter skill against this note once 1 to 4 have
   landed, and fold in what it misreads.

## 6. Housekeeping

- `DISCUSSION-20260823` §2 carries a dated amendment withdrawing the
  instruction-turn fallback.
- `.agents/skills/altrux-debrief/SKILL.md` step 3 and the guardrails are
  reworded for a conversational debrief.
- Branch `box/20261010-041135` (notes, logs, `634f914`, README fix) is
  merged by altrup after reviewing the protected diff.
- altrup accepts the Llama-2 license on huggingface.co for the Altrup
  account.
- Open, not scheduled: the 1% evaluation-batch wobble (fix the evaluation
  batch size per grid and report it), the greedy wake's non-reproducibility
  (the saved state is the reproducible object), the TAALM environment
  deviations on ARM64 (see the experiment note), the box's re-locked
  `sft/uv.lock`.
