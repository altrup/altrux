# Discussion notes — 2026-08-04: binding capacity ceiling, null re-registration, fused-path plan

Team debrief (altrup + Claude) of the 2026-08-05 box session
(`../experiments/EXPERIMENT_NOTES-20260805-002700.md`). Builds on
`DISCUSSION-20260804-consolidation-null-run-plan.md` (the plan that session
executed) and `DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md`
(§3 M-necessity). Raw numbers verified against the pulled
`sft/logs/capacity_ladder*.jsonl` and `consolidation_null*.jsonl`.

## 1. What the run established (agreed reinterpretation)

1. **Headline (solid): the 780M Mamba2 SSM holds ~3–4 arbitrary key→value
   bindings in context, set by fact count, nearly independent of transcript
   length out to ≥2.7k tokens.** The discriminator cell settles it: at a
   matched ~3k-token budget, 4 facts → 1.000 vs 24 facts → 0.042 and
   40 facts → 0.050. Absolute hits across the sweep: 4, 3, 4, 1, 2.
2. **Nuance the run notes undersell: survival skews recent at high N.**
   Hit indices per cell: 4×200 → [2,3]; 8×200 → [0,4,6,7]; 24×40 → [23]
   (the last fact written, and nothing else). Under pressure the state
   degrades toward the most recent write(s), so in a real session
   early-session facts are the first casualties. Relevant to the wake-time
   working-memory framing, not just the headline rate.
3. **This re-denominates the 07-25/07-30 M-necessity number.** "Dead by
   ~192 tokens of dense interference" was measuring binding count, not token
   distance — the horizon is ~3–4 *bindings* at 780M, length-irrelevant.
   The 07-30 §3 exchange-rate argument (sleep cadence vs interference
   horizon) survives unchanged with the unit corrected: cadence must be
   measured in facts-written, not tokens-elapsed.
4. **The consolidation null is unanswered; every number from the session is
   void.** The N=40 configs violated the premise (teacher never held the
   facts); the one valid-premise run (4×800, control 0.750) was killed at
   distill step 81/200 with no post-probes. No §4 branch of the 08-04 plan
   has fired. Do not treat anything from this session as a null verdict.
5. **Structural cap on the harness (unanticipated by the plan): the
   transcript-null's teacher only knows facts by holding them in context,
   so it is intrinsically capped at N≈4 on 780M.** High-N consolidation
   cannot be tested by this harness at all; that needs a non-context-bound
   teacher — which is itself an argument for M. Consequence: the null's
   evidential value is almost entirely in its FAIL branch (PASS at N=4 is
   compatible with brute memorisation into 9.67M LoRA params).
6. **The ceiling is soft and sits at N=4's edge.** A byte-identical 4×800
   transcript scored 1.000 (ladder) and 0.750 (null pre-phase) — GPU
   nondeterminism flipping a borderline binding (`topaz` read back as
   `osprey`'s code). Any single-run readout at this N is jittery.

## 2. Verdict semantics — re-registered (before any rerun data exists)

The 08-04 plan's thresholds were registered for N=40. At N=4, conditioned
on the ~3 facts the teacher actually holds, PASS ≥ 0.30 means one lucky
fact — nearly vacuous. Re-registration below is agreed *now*, with no
rerun data in hand; this is regime correction, not post-hoc tuning. The
numeric thresholds themselves are unchanged.

- **Config: `4x40`** (control 1.000 in the smoke run; fact tokens ~39% of
  transcript vs ~3.8% at 4×800, so a 10× denser distillation signal — the
  most-favourable-conditions config, making a FAIL decisive).
- **3 seeds** (1234, 2345, 3456), pooling per-fact results.
- **Condition per fact on `in_context_match == true`** in the pre-phase
  (recorded in the jsonl). Facts the teacher never held cannot be
  installed and must not count against (or for) consolidation.
- **PASS**: pooled post-distill greedy exact-match over conditioned facts
  ≥ 0.30 (i.e. ≥3 of ~10–12 facts installed across independent runs).
  Licenses "mechanism alive" only — not "CL works" (see §1.5).
- **FAIL-UNDERPOWERED**: pooled rate < 0.30 but pooled
  `mean_logprob_delta` over conditioned facts ≥ +1.0 nat → scale
  `--distill-steps` ×4, rerun.
- **FAIL-DEAD**: neither → one escalation run at 2.7B, at the N the 2.7B
  ladder (§3.3) says clears the 0.8 control. If that is also fail-dead,
  stop; the consolidation rethink happens off-box.
- **Adaptive early stop**: if seed 1 is unambiguous — 0 installs with flat
  or negative pooled logprob delta (clean FAIL-DEAD), or all conditioned
  facts installed — skip the remaining seeds. Boundary results get all 3.
- **Dilution bracket**: run `4x800` once *only* if 4×40 lands ambiguous
  (near a threshold), to test whether filler dilution was a confound.

## 3. Standing direction — next box session, in order

Cheap box (A10) unless the fused path is green and a bigger docket gets
approved separately.

0. **Validate the fused path first, timeboxed 45 min.** The oracle test
   written locally (§4) is `sft/tests/test_mixer_fused.py`: fused
   `_mixer_chunk` vs the per-token loop — logits, threaded `MixerState`
   across 2 chunks, LoRA gradients; fp32 tight tolerance, bf16 loose. It
   SKIPs on the ROCm dev box, so this box is the first place it ever runs:

       cd ~/altrux/sft && make test

   Green → run everything below on it. Not green → fall back to the
   per-token loop (unset nothing; the dispatch in
   `models/mamba2_780m/model.py` is automatic, so park it by reverting to
   the commit before `d6f25d7`), park the fused work, run the science
   anyway. Do not debug Triton on billing past the timebox.

   One expected non-failure: `models/tests/test_grad_checkpoint.py`'s
   plain-780M equivalence test asserts checkpointed vs un-checkpointed at
   `atol=1e-6`. On CUDA both sides now run fused with *different* kernel
   chunk splits, so a small miss there is tolerance, not a regression —
   loosen it rather than treating the fused path as broken.
1. **The null at `4x40`, per §2.** Verbatim per seed:

       cd ~/altrux/sft && make consolidation-null ARGS='--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed <SEED> --out logs/consolidation_null_4x40_s<SEED>.jsonl'

   Read the in-context control line before letting each run proceed; the
   verdict is computed from the pooled jsonls per §2.
2. **If the fused path validated: rerun 1–2 ladder cells (e.g. 4x40,
   24x40) on it** and compare to the per-token numbers before trusting any
   cross-session comparison. All existing capacity numbers are
   per-token-loop numbers; the chunked scan is not bitwise-equal in bf16
   and §1.6 shows borderline bindings flip on less.
3. **Capacity ladder at 2.7B** (`MODEL_NAME=mamba2_2_7b_memory` in
   `sft/.env`; M zero-init makes it a reasonable plain-backbone proxy):

       cd ~/altrux/sft && make capacity-ladder ARGS='--grid 4x40,8x40,16x40,24x40,40x40,4x800 --out logs/capacity_ladder_2p7b.jsonl'

   Runs on the per-token loop regardless (the fused port is 780M-only).
   The question: does the ~3–4 ceiling scale with `d_state`/`nheads`, and
   enough to matter. If 2.7B also tops out near a handful, that is close
   to decisive for M-necessity. Also sets the N for §2's FAIL-DEAD
   escalation run.

## 4. Local work (before the next box session)

**Fused/chunked fast-path for `models/mamba2_780m`**, written and
oracle-tested locally, validated on the box (the ROCm box cannot run the
kernel — that is why the per-token loop exists). Motivation: 64 ms/token
makes rented-CUDA training infeasible (~170k GPU-hours for the 230M-token
corpus) and turns each null seed from ~30 min into minutes. Sketch agreed
(from the run notes, feasibility verified on the A10 by inspection):

- Oracle test first, per-token loop as ground truth: logits, final
  `MixerState` for fresh state and two threaded chunks, gradients w.r.t.
  LoRA params. Skip cleanly when `torch.version.hip` is set (ROCm).
- `_mixer_chunk` alongside `_mixer_step`: batched `in_proj`; conv via
  `F.conv1d` with `conv_state` prepended as left padding (no
  `causal-conv1d` dependency); `mamba_chunk_scan_combined(...,
  initial_states=ssm_state, return_final_states=True, dt_softplus=True)`
  with `z=None`; gated norm applied after, matching the manual ordering.
- Dispatch: `T==1` → existing `_mixer_step` (generation stays on the
  proven path); `T>1` → fused when the kernel is importable, else the
  per-token loop (ROCm box keeps working).
- 780M only. The 2.7B port (M-wiring interleaved in the mixer) waits
  until the M arms need it; 780M-first yields a validated reference.

## 5. Explicitly considered and rejected

- **Format SFT as the low-control remedy** (08-04 plan §3). Wrong
  diagnosis: outputs were perfectly formed, just mis-bound. A low control
  means the bindings exceed SSM capacity → reduce N, never format-SFT.
- **Running the null at N<4 or chasing a 4/4 control with reruns.** N=4
  is the largest N with a valid premise; rerunning until the control is
  perfect selects on noise. Condition per fact instead.
- **High-N consolidation via this harness.** Impossible by construction
  (§1.5). A non-context-bound-teacher redesign is an open question, not
  scheduled.
- **Keeping the N=40-era thresholds unmodified at N=4.** That is
  fidelity to the letter of the pre-registration at the cost of its
  point; §2 re-registers before data instead.
- **Doing the 2.7B fused port now.** Larger piece, not needed until M
  arms train; 780M-first gives a reference implementation.
- **Finishing the killed 4×800 run as-is.** 4×40 dominates it (denser
  signal, same premise validity); 4×800 survives only as the §2
  dilution bracket.

## 6. Open questions

- Does binding capacity scale with model size (§3.3 answers this)?
- Should the null harness be redesigned around a non-context-bound
  teacher to test high-N consolidation? Deferred until the N=4 gate and
  the 2.7B ladder are in.
- Sleep-cadence exchange rate (07-30 §5b) now needs restating in
  bindings-written, not tokens — the horizon measurement should count
  facts, not filler.

## 7. Housekeeping

- `scripts/lambda_launch.sh` watchdog pattern now includes
  `capacity_ladder.py` (bit the session mid-ladder: the box's
  termination clock ran while the ladder was the only live process).
- The run notes were committed from the instance (`cd08a0e`), deviating
  from the never-commit-notes-from-the-box rule because the rsync target
  was going away; one-off, rule stands.
- The duplicate local notes commit created during banking was dropped;
  local main == origin/main at `cd08a0e`.
- Fused-path implementation started locally in this session (§4), and
  landed as `d6f25d7` — `_mixer_chunk` + `_forward_chunk` in
  `models/mamba2_780m/model.py`, oracle test in
  `sft/tests/test_mixer_fused.py`. Written but never executed: this box's
  venv had no packages installed, so not even the ROCm fallback path was
  smoke-tested locally. §3.0 is its first run of any kind.
- `.claude/commands/altrux-debrief.md` step 3 now asks for one-line claims
  to open the discussion instead of a prose block (`3e3e8e5`).
