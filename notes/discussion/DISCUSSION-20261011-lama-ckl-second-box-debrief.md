# Discussion notes — 2026-10-11: LAMA-CKL second box run debrief

Debrief of the second LAMA-CKL box session
([`EXPERIMENT_NOTES-20261011-134419`](../experiments/EXPERIMENT_NOTES-20261011-134419.md),
branch `box/20261011-074114`, merged). It amends
[`DISCUSSION-20261010`](DISCUSSION-20261010-lama-ckl-first-box-debrief.md)
§3.1 (warm start) and §3.7 (session order) and keeps the rest of that note
in force. Where this note and the 2026-10-10 note disagree, this note wins.

## 1. What the run established

Three baseline cells and the Llama-2 calibration ran to completion. The
altrux cell did not start: pre-grid acceptance failed on the wake
invariant, and the amendment's stop-and-report rule held.

1. **The repeat-0.3 warm start closes the conversation when handed a bare
   document.** With the corpus rendering (`\n[ASSISTANT]`, no lone space),
   the greedy reply to `[USER] Remember this for later: <doc>` is
   `<|endofconversation|>` as the first token in 87 of 100 to-learn
   documents; the 11 content replies are "end of quote ]"-style closers or
   document copies. Acceptance read `internal_eoc 10/10`. Plain chat and
   Dolly-shaped prompts answer correctly, so the model is not broken; the
   frame is the one turn type its corpus never contains: an instruction
   with nothing to answer.
2. **The dream prompt has no supervision.** The train mask keeps user turns
   out of the loss (`97ecb30`, 2026-09-24). In `data/warm_start.pt` the
   token after `<|endofconversation|>` is a target 0 of 1766 times and
   `[USER]` is a target 0 of 4377 times. So the registered dream prompt
   (open state, then EOC, then the model writes `[USER] …`) was never
   trained: from an in-distribution turn, 9 of 10 dreams emit a second EOC
   at once. An altrux cycle would distill ten empty dreams even with a
   healthy wake. The 2026-10-10 recap-0.5 run dreamt only because the mask
   was not applied then.
3. **The special-token embeddings are undertrained.** `[ASSISTANT]` carries
   5–50 % next-token mass after `\n[ASSISTANT]` in several prompts (12 %
   after a plain question), loops under a question-shaped frame, and 5 of
   10 fresh-state EOC dreams fall into a `[ASSISTANT][ASSISTANT]…` run. No
   corpus line supports it. 800 LoRA steps over fresh embeddings is the
   likely cause.
4. **Baselines, seed 42, 30 epochs** (object-token accuracy; to-learn /
   not-to-forget at the to-learn peak):

   | arm | peak to-learn | epoch | retention at peak | wall |
   |---|---|---|---|---|
   | frozen | 0.004 | 0 | 0.9835 | 25 s |
   | lora | 0.064 | 15 | 0.527 | 56 min |
   | mix-review | 0.060 | 13 | 0.518 | 85 min |

   Lora's acquisition plateaus near 0.055 from epoch 4 while retention
   falls to about 0.52 by epoch 10 and holds. Mix-review's retention tracks
   lora's within 0.01 from epoch 10 on: reviewing the 500 retention
   documents every epoch did not protect their object tokens.
5. **Llama-2 calibration: a miss on two of three gate terms.** Authors'
   QLoRA recipe, single GH200, microbatch 8 × accumulation 8: peak 0.105 at
   epoch 29 with 0.897 retained, against the published 0.115 at epoch 16
   with 0.8174. Peak inside tolerance; epoch and retention outside. The
   model learns slower and forgets less than published. Candidates: the
   one-GPU accumulation and the ARM64 deviations (bitsandbytes source
   build, triton removed, peft pinned by date). Calibration only, not
   blocking (2026-10-10 §3.5).
6. **Apparatus.** Warm start 800 steps in 9 min (1.5 step/s), reproducible
   step for step across two attempts. The LR warm-up loss spike (8.45 at
   step 50 on the 08-10 curve) reads as a fault to a fresh experimenter;
   the skill now carries the curve. The wake rendering bug (lone space, no
   newline before `[ASSISTANT]`) was found by acceptance attempt 1 and fixed
   on the box (protected edit `4dee2aa`, merged).

## 2. Reinterpretation

The altrux arm is blocked by the warm start, twice over: it does not reply
to the frame, and it cannot write a conversation after a boundary. Neither
is a property of the method. Both are coverage gaps in a 2000-conversation,
800-step SFT whose user turns are masked.

The baselines are a real result about the base: on this 2.7B Mamba2 the
LoRA arm trades 46 points of retention for 6 points of acquisition, and
review does not help. That is the number altrux has to beat, and it stays
banked only if the base stays the same. It does not: §3 retrains the warm
start, so every arm reruns on the new base. The Llama-2 calibration does
not depend on the warm start and stays.

The decision of 2026-10-11 (altrup) is to fix the base rather than the
frame. A prefilled acknowledgement or a no-reply wake would unblock the
cell at zero training cost, but the end goal
([self-teaching](DISCUSSION-20261010-self-teaching-end-goal.md)) needs a
model that converses, and a base that cannot acknowledge a pasted passage
will keep producing failures of this kind. The frame, the dream prompt,
and the 10-and-10 cycle stand.

## 3. Standing direction

### 3.1 Warm start, third recipe

Pool, one shuffle, same packer (2–5 per example, `--max-len 8192`,
`--repeat-rate 0.3`, no synthetic recap):

- UltraChat `HuggingFaceH4/ultrachat_200k`, **20,000** conversations
  (was 1000);
- Dolly `databricks/databricks-dolly-15k`, **all** rows of the three
  context categories (about 4,500; was 1000);

No acknowledgement rows in the pool. The first question of this warm start
is whether general chat data alone teaches the model to acknowledge a
pasted passage. A **top-up stage** exists for the case it does not (3.3
step 1): 500 rows of user `Remember this for later: <Dolly context>` (the wake frame string, so
the top-up tokens match the wake prompt)
and a short templated acknowledgement (`preparation/dolly.py
--acknowledge`), trained for a few hundred steps by resuming from the
final checkpoint. No LAMA text; the split is model-conditioned after the
warm start, so Dolly's Wikipedia-like contexts are not a leak.

**User turns are trained** (`--train-user-turns` on the packer): the mask
is True on `[USER]` turns as well as assistant turns. This supervises the
boundary-opening `[USER]` and everything the dream prompt asks the model
to write. It departs from assistant-only SFT, and the lora baseline then
starts from a base trained on user text; every arm shares the base, so the
comparison is unchanged. Decided by altrup 2026-10-11.

Training: **1 epoch, about 11,000 steps** (26M tokens at the measured
2,400 tokens per step; about 2 h on a GH200 at 1.5 step/s), lr 1e-4, chunk
512, LoRA rank 16, alpha 32, as before. **Ten checkpoints**
(`--ckpt-every-tokens 2600000 --keep-ckpts 10`, 123 MB each with the
optimizer) instead of the last only, so the trainer resumes after a crash
with at most one interval lost and the probe below reads the curve. All
ten come home in the pull. The final checkpoint (after the top-up, if it
ran) is the warm start; the box writes its SHA-256 to
`sft/experiments/lama_ckl/warmstart.sha256`.

Not changed, and why: LoRA rank and epoch count (one variable at a time;
the arms share the LoRA config); the LR warm-up (the spike is known-shape
and recovers by step 200); the pool's breadth (UltraChat is broad in topic
and its replies are acknowledgement-shaped often enough; a second
instruction set buys task-type diversity the gate does not need); model
size (§4). The dream prompt stays as registered: EOC, then the model
writes a whole conversation, both sides; the dream is modelled on a real
dream, not on an assistant continuing its turn.

### 3.2 Probe during training

A new `make warm-start-probe ARGS="--init-adapter <checkpoint>"`
(`sft/experiments/lama_ckl/probe.py`) loads one checkpoint and reports:

- the wake frame on 50 held-out Dolly contexts (never in the pool): first
  reply token census (EOC, content, `[ASSISTANT]`, EOS), forced closes,
  reply lengths, decoded samples;
- 10 dreams from the post-reply state of one wake, T 0.7: empty-dream
  count, `[USER]` opener count, `[ASSISTANT]`-loop count, decoded text.

It reuses the protocol's wake and dream functions and touches no protected
file. The experimenter runs it in the CPU-side window on each checkpoint
as it lands while training continues; the runner's `--cycles 1` acceptance
stays the gate for the final checkpoint, because it alone verifies the pin
and the split.

Expected on a good checkpoint: EOC-first near 0 of 50, forced closes below
20 %, most dreams open a `[USER]` turn, no `[ASSISTANT]` loops.

### 3.3 Session order (replaces 2026-10-10 §3.7 order)

1. Warm start (3.1), probe on each checkpoint (3.2). The final checkpoint
   is the candidate; an earlier one is never picked. If its probe shows
   EOC-first on more than half the documents, run the top-up stage once
   and probe again. If that still fails, stop and report.
2. Pin. Split (`--lama-root ../.cache/LAMA/data`).
3. Wake acceptance, `--cycles 1`, docs 10: `internal_eoc` 0, forced closes
   below 20 %, and `epoch-01/cycle-00/dreams.json` shows non-empty dreams
   with `subjects_mentioned` above 0. Both wake sizes above 20 % forced
   closes: stop and report (2026-10-10 amendment, unchanged). **Record
   the cycle's wall time** (wake, dreams, distillation separately, from
   the log timestamps) and write the altrux ETA for 1500 cycles plus 31
   evaluations in the experiment note before the baselines start. The
   fused single-token step passed its equivalence test on the GH200
   (`ace1155`, 7 passed); this is its first timed cycle.
4. Smokes. Then frozen, lora, mix-review, altrux, seed 42, in that order,
   so the baselines are banked before the long cell. The Llama-2
   calibration is not rerun.
5. Soft cap 20 h (`--max-hours`), resume by rerunning the same command.

### 3.4 Report

`make lama-ckl-report` refuses fewer than the 12 registered cells; the
seed-42-first amendment makes a 4-cell report the normal case. The check
becomes a `--allow-incomplete`-free default of "report what exists, list
what is missing". Follow-up before the next box, not protected.

## 4. Explicitly considered and rejected

1. **Prefilled acknowledgement** (`[ASSISTANT] Noted, I will remember
   that.` teacher-forced, model finishes). Zero training cost and would
   unblock the cell now. Rejected for this program because the reply is
   scripted and the general-purpose base is the goal; it stays the
   fallback if the warm start fails the probe after the top-up.
2. **No-reply wake** (feed the document, dream from that state). Same
   objection, and it moves the method away from a conversational model.
3. **Question-shaped frame.** Rejected by the diagnostic: the model loops
   `[ASSISTANT]` (fact 3).
4. **EOC counted as a forced close.** Hides the failure inside a
   diagnostic; the dreams from an EOC-closed state are empty anyway.
5. **Longer warm start on the same recipe.** Fixes neither gap: the turn
   type is absent and the user turns are masked at any step count.
6. **A larger model.** The failure is coverage, not capacity: the same
   model answers in-distribution prompts. The stack is Mamba2-specific and
   the baselines are banked on 2.7B; scale is the step after a positive
   altrux result, not before.
7. **Masking in only the boundary `[USER]` token** instead of whole user
   turns. Enough for the dream opener, but the dream then writes user text
   it was never trained on. Whole turns, decided.
8. **Acknowledgement rows in the pool from the start.** Would hide whether
   general data suffices; kept as the top-up stage instead.
9. **A second instruction set** (Tulu 3, OpenHermes-2.5). Deferred until a
   probe or a downstream failure asks for task-type diversity.

## 5. Local work before the next box

In this order, tests first, under the standing go-ahead gate:

1. `sft/preparation/conversations.py`: `--train-user-turns`. Test: the
   mask is True on user-turn tokens with the flag, False without.
2. `sft/preparation/dolly.py`: `--acknowledge N` rows (off in the
   warm-start target; a `warm-start-topup` target uses it with
   `--resume`); `--max-examples` default covers all context rows. Test: an
   acknowledgement row carries the frame prefix and a template reply, and
   its context is drawn from rows outside the probe's held-out set.
3. `sft/experiments/lama_ckl/probe.py` and `make warm-start-probe`. Test:
   the census counters on a fake model that emits a fixed first token.
4. `sft/Makefile` `warm-start`: the counts, steps, epochs, checkpoint
   flags, and the two new flags. `sft/README.md` in the same change.
5. `.agents/skills/altrux-experimenter/SKILL.md`: the probe step, the
   top-up and stop rule of §3.3 step 1, the timed cycle and ETA of step 3,
   and the ten-checkpoint tree.
6. `sft/experiments/lama_ckl/report.py`: §3.4.
7. `scripts/lambda_launch.sh`: a missing instance type in the region
   listing is fatal on the first lookup only; after a type has been seen
   once it is "no capacity yet". Found as a transient after a lost launch
   race.

## 6. Housekeeping

- Old curves with `cycle` keys do not load in `report.py`; unchanged.
- The evaluation-batch wobble (0.9815 vs 0.9835 at the same adapter, batch
  8 vs 16) stands as recorded in the 2026-10-10 note §1.5.
- `warmstart.sha256` in `PROTECTED_PATHS`: altrup's call, still open.
- Three stale stashes on `main` (superseded notes, drop them); the local
  `altrux-1` tmux session is closed.
