# Discussion notes — 2026-08-04: consolidation-null run plan (box standing direction)

Team discussion (altrup + Claude). Standing direction for the next box
session. Builds on `DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md`
(§4 M-necessity, §5 CL evaluation) and
`DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md` (§4 ordering).

## 1. What this session is for

Run the **transcript-consolidation null** — `sft/consolidation_null.py`,
committed today, GPU-unvalidated (first run is the shakeout). It is the
field-default consolidation baseline from 07-25 §4 and it **gates every
downstream CL arm**: if brief LoRA distillation cannot install facts held
losslessly in context, neither M-replay nor SSM-replay can, and
CL-by-consolidation needs a rethink before any training budget is spent.

This is a cheap-box session (A10), not a training session. Nothing else is
on the docket: no big training run, no M arms, no data generation
(`consolidation_null.py` builds its own transcript; `LAMBDA_DATA_ARTIFACTS`
is empty on purpose).

## 2. How to run it

- `MODEL_NAME=mamba2_780m` in `sft/.env` (setup leaves it blank).
- `make consolidation-null` from `sft/`, defaults first (40 facts,
  200 distill steps, pass@10). The Makefile target tees to
  `logs/consolidation-null-<ts>.log`; per-fact results stream to the
  `--out` jsonl as produced.
- Expect a shakeout pass: the script has never touched a GPU. Fix what
  breaks, commit fixes back, then do the real run.

## 3. Read the harness checks before believing any number

1. **In-context positive control** (transcript in context, generate each
   fact) must be high — the script warns below 0.8. If it is low, the
   *harness* is broken, most likely the base 780M can't follow the probe
   format. The fix is a minimal format SFT (chat markers,
   answer-the-question behaviour) before rerunning — not a verdict about
   consolidation.
2. **Fresh-state floor** must be ~0. If the model answers without the
   transcript or distillation, codes are leaking; fix before proceeding.

## 4. Verdict semantics — pre-registered, do not move after seeing results

Constants at the top of `consolidation_null.py`:

- **PASS**: post-distill greedy exact-match rate ≥ 0.30. Consolidation
  works in our regime; the script becomes the harness and the bar for the
  M-replay arm.
- **FAIL-UNDERPOWERED**: rate < 0.30 but mean teacher-forced logprob delta
  ≥ +1.0 nat. Mechanism alive, budget too small — scale `--distill-steps`
  (×4 as the first move) and rerun, still at 780M. Do not conclude anything
  terminal from this branch.
- **FAIL-DEAD**: neither. One escalation before the consolidation-rethink
  gate fires: a single rerun at 2.7B (fact absorption scales with model
  size, so 780M-fail is confounded by scale; 780M-pass is a floor for
  2.7B). If 2.7B is also fail-dead, stop — the rethink happens off-box,
  don't burn hours iterating.

The pass@k and cue-ladder tiers are diagnostic colour (how close a miss
was), not verdict inputs.

## 5. Cost posture

A10, a few dollars/hour; the watchdog pattern now includes
`consolidation_null.py`, so the box terminates when the run stops. Total
session budget is a shakeout + one or two real runs — if iteration is
ballooning past that, stop and bring the problem home instead of debugging
on billing.

## 6. Not doing

- No training run, no M arms, no 2.7B unless §4's FAIL-DEAD branch fires.
- No threshold tuning post hoc — thresholds are the pre-registration.
- No gist/logprob headline: the headline number is behavioral
  generation-based recall (07-25 §5).
