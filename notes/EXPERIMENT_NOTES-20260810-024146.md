# Experiment notes — 2026-08-10 02:41 UTC: the §4 local pilot (steer-prefix / gate / N / rank-rule), on the teammate's local box

Run of `DISCUSSION-20260808-headline-collapse-deep-block-and-regime-bridge.md`
§4's local pilot, executed on the local ROCm machine (RX 7700S,
`HSA_OVERRIDE_GFX_VERSION=11.0.0`) by the overnight session that also closed
the launch gate. **Old adapter throughout** (`epoch-2/step-800`, trainable.pt
sha `d4bf2e35…`): every number here is PRE-§3(0)-retrain and provisional per
§2.9.3/§2.10.8 — the `<|eoc|>` steer candidates are untestable until the
retrain, and rehearsal rates may shift under the new adapter (teammate's
explicit caveat, registered before the run).

## Commands (verbatim, both candidates; second run's only deltas are
`--dream-prompt` and `--gate-threshold 0.25`)

    HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
    HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_INTERRUPT=1 MODEL_NAME=mamba2_780m \
      uv run --no-sync python -u dream_sleep.py \
      --seed 1234 --n-facts 4 --filler-tokens 40 \
      --wake-bystanders 3 --wake-nearcone 2 --wake-dialogue 2 \
      --dreams 20 --dream-tokens 512 --dream-temp 0.7 \
      --bind-min-dreams 0 --pilot-capture --build-dream-cache \
      --arm replay \
      --init-adapter /media/storage/Altrup/Code/Github/altrux/models/mamba2_780m/checkpoints/epoch-2/step-800 \
      --out logs/pilot_noprefix_s1234.jsonl

`--bind-min-dreams 0`: the pilot measures coverage, it does not gate on it
(deliberate deviation, recorded). Rich wake = 4 facts + 3 off-format + 2
near-cone + 2 ultrachat-dialogue items, 913 tokens, 62 turns; all builder
invariants 0, round-trip decode true, decoded samples read correct (log
22:46:24). §4's "extract the transcript from dream_cache_s1234.pt" clause was
superseded by the rich-wake regeneration (teammate's call, this session):
the sequencing constraint in §2.10.11 requires the rich shape and the
retrain already severed old-number pairing.

Artifacts (local only, gitignored): `sft/data/dream_set_{noprefix,instruct}_s1234.{pt,txt,pilot.pt}`,
`sft/data/dream_set_noprefix_s1234.gate_pilot.{txt,jsonl}` (+instruct),
log `sft/logs/pilot-20260809-224146.log`, scorer log `sft/logs/gate_pilot-014505.log`.

## Headline: the §4 kill-condition FIRES on both locally-testable candidates (old adapter)

Aggregate binding, 20 fresh un-spliced self-terminating dreams per candidate,
bound = code in the same sentence as its own entity:

| candidate | clove | topaz | osprey | heron | verdict at k=2, N≤16 |
|---|---|---|---|---|---|
| no-prefix (incumbent) | 1 | 1 | 1 | 4 | FAIL (3 of 4 facts below k) |
| instruction-text ("You are now dreaming…") | 3 | 0 | 0 | 0 | FAIL (3 of 4 facts at zero) |

An earlier no-prefix set (pre-acceptance-check build, same seeds):
clove 1, topaz 0, osprey 1, heron 3 — same shape.

- Most dreams rehearse **zero** facts (16/20 in the scored no-prefix set have
  no binding-scan fact read at all). Free-running dreams from the rich wake
  drift into ultrachat-style dialogue (travel logistics, competitor analysis)
  and stay there.
- The instruction prefix makes it WORSE, not better: it pulls generation into
  prompt-following (and at `--gate-threshold 1.0` its first dream gated
  0/512 positions — state reads *suppressed*; rebuilt at 0.25 to complete).
  §5's "a 780m base model has no instruction-following to prompt at" is
  confirmed from the state side.
- Per §4 this is stop-report-redesign for the no-splicing bridge as specced —
  BUT the decision is registered as a TEAM decision, because (§2.10.8) the
  three `<|eoc|>`-family candidates — including the expected winner — can only
  be tested after the box retrain, and §2.9.3 makes all pre-retrain rates
  provisional. Options at the check-in: (a) keep the box docket but open with
  a cheap post-retrain coverage re-pilot on the box before committing to the
  B4 block; (b) redesign steering now (stronger/cue-adjacent); (c) accept a
  larger N / lower k. No recommendation is frozen here; see "Team decisions".
- Watch-item feeding the same decision: dreams that DO rehearse can loop
  verbatim (the no-prefix sidecar's tail is ~15 consecutive identical
  lighthouse-keeper lines). Acceptance (below) only rejects mojibake, so
  verbatim loops pass and depress effective coverage/diversity. Candidate
  lever for the redesign discussion, not patched overnight.

## What the pilot DID deliver cleanly (less adapter-sensitive)

- **The state-dependency gate works.** Healthy dreams gate 3–86 of 512
  positions at 1.0 nats (typical 20–60); D_t median ~0.02, max 6–15. The
  all-positions-diverge failure mode did not occur.
- **Separability (test 1)**: pooled AUC **0.628** over 75 fact-read and
  10,125 other positions — above the 0.6 kill line but thin. Where a dream
  has few reads, AUC is high (0.978 at 5 reads); the heavy-rehearsal dream
  scores 0.540 — consistent with the §2.7 re-installation window: after the
  first rehearsal the fact re-enters via TOKENS, so later "fact reads" are
  legitimately state-independent and the oracle labels, not the gate, are
  what's noisy there. The binding-scan oracle overcounts state reads on
  repetitive dreams. (Registered interpretation, not proven.)
- **Cross-dream V-overlap** (per-dream erasers, §2.10.3's watch-item): raw
  mean **0.699** (max 0.993), deflated 0.689, qcm 0.289. High shared subspace
  across dreams' erasers — the per-dream-noise cost of the per-dream call
  looks small, and the qcm number says much of the overlap IS the common
  mode. Also feeds §2.10.13's union-rank-vs-budget watch-item (union will
  stay near per-dream rank at overlap ~0.7).
- **Both rank rules ran on every real spectrum**; the sweep table (appended
  below when the scorer completes) carries the rank ranges and disagreements
  per scheme. Position-0's D_t is dream-independent (same state, same
  prefix), so identical max-D_t values across dreams are expected, not bugs.
- **Per-dream acceptance** (new, this session — see "Decisions"): 1 mojibake
  dream in ~40 real dreams (2.5%) rejected and regenerated cleanly; a second
  batch had 0. Termination: every scored dream ended `max-tokens` — no dream
  emitted `<|eoc|>` (expected: the old adapter has never seen the token; its
  logit row is untrained) and none hit the turn backstop.

## Failures on the way (all fixed, committed, tested; §1.8 called every one)

1. Relative `--init-adapter` path from `sft/` cwd — operator error, absolute
   path. (No commit.)
2. `marker_delta` shape mismatch: registering `<|eoc|>` grew the delta to 3
   rows and every pre-registration checkpoint refused to load →
   `load_checkpoint` zero-pads grown marker rows, loudly (`1bdac89`).
3. CPU/GPU mismatch in the deflated variant (cached CPU queries vs cuda
   state) → deflate on the rows' device (`9b6570d`).
4. Full-mojibake dream → near-empty gate → empty qcm basis killed the build
   → per-dream acceptance + regeneration (`76e6b2a`).
5. `battery_read_queries` ran after `state_to` moved the shared wake state to
   CPU → forwards on the model's device regardless of caller order
   (`ab97598`).
6. Scorer refused every scheme when any dream lacked fact reads (15/20 did)
   → sparse coverage pools target/oracle over dreams-with-reads; only a
   capture with no reads anywhere is unscoreable (`f1f42ea`).

Four of six are the hardware-shaped-code class the standing §1.8 lesson
predicts the fake backbone cannot catch.

## Decisions taken overnight (teammate asleep; flagged for ratification)

1. **Rich-wake regeneration over cache extraction** — resolved the §4
   internal contradiction; teammate approved before bed ("regenerate with
   the new protocol").
2. **Per-dream acceptance policy** (`76e6b2a`): mojibake dream → reject +
   regenerate under bumped seed (≤3 attempts), content-free hence prod-valid
   under §2.9.1; exhausted retries stay stop-and-think. NEW mechanism
   surface — ratify or overrule.
3. **Instruct candidate rebuilt at `--gate-threshold 0.25`** after its dream
   1 gated 0/512 at 1.0 (the offline sweep re-derives thresholds from full
   D_t, so the build threshold only had to let the build complete; the two
   candidates' gate-derived numbers are therefore NOT paired — coverage is
   threshold-independent and stands).
4. **Scorer pooling** (`f1f42ea`): sparse coverage is a row property
   (`dreams_with_reads`), not a scheme failure.
5. **B4 sleep-exit carry registered** (§2.10.13, teammate's direction
   pre-sleep): union-of-erasers projection at sleep exit; implementation
   deferred to the multi-sleep docket.

## Team decisions queued for the morning check-in (nothing frozen)

1. **Kill-condition disposition** (§4): stop-redesign vs post-retrain re-pilot
   on the box vs larger-N/lower-k. The `<|eoc|>` candidates' untestability
   pre-retrain is the crux.
2. **Gate-scheme + threshold + rank-rule freeze** (§2.10.6(v)): from the
   appended sweep table; the tool recommends, the team ratifies.
3. **Ratify overnight decisions 2–4 above.**
4. **Verbatim-loop dreams**: extend acceptance (repetition clause), or treat
   as a steering/regime symptom, or leave.
5. **Push**: 10+ commits are local-only; §3's push-before-launch stands.

## Gate-close state (context for the box session)

Launch-gate §4 code items: ALL implemented, tested, committed (see
`git log 56630ff..`, 461 passing tests at `f1f42ea`, 3 further commits after
the audit). Cold dry-run audit re-run this session per §7: verdict was
"cold box session would get confused" on four blockers — pilot verdict
unrecorded (this file fixes it), pilot artifacts gitignored (this file is
the travel copy), `make warm-start` stale 400-step regime (fixed,
`d9f256f`), docket flags unpinned (fixed, `d9f256f`). Arm spellings
verified: `--arm replay | b4-raw | b4-deflated | b4-qcm`, set caches accept
exactly these four.

## Appendix: gate-scheme sweep table

### no-prefix capture (oracle base: 75 fact reads in 4 of 20 dreams)

```
gate pilot: data/dream_set_noprefix_s1234.pilot.pt  (20 dreams, set_sha b608dd44a8bd5fea154906ccad8581005713e7c1e2911ba40d339fabd6d09fc9, variant deflated, rank rule ratio-gap)
test 1 separability: pooled AUC 0.628 over 75 fact-read and 10125 other positions; per dream 0.976, nan, 0.856, nan, nan, nan, nan, nan, nan, 0.978, nan, nan, 0.540, nan, nan, nan, nan, nan, nan, nan

test 2 bake-off. THE DECISION METRIC is target vs collateral removal (sec 2.10.6);
precision/recall and oracle overlap are diagnostics -- label accuracy is a NON-goal,
and no scheme here is scored by any downstream training outcome.
rank column is the ratio-gap rule's per-layer span, /median where the rules disagree.

scheme                AUC  prec   rec oracle  stab  target  collat    ctx   batt  gated       rank
hard@q0             0.628 0.037 1.000  0.547 0.753   0.046   0.051  0.054  0.039  510.0    1-6/1-8
weighted@q0         0.628 0.037 1.000  0.588 0.545   0.048   0.038  0.037  0.038  510.0    1-7/8-8
sqrt@q0             0.628 0.037 1.000  0.619 0.676   0.056   0.043  0.043  0.042  510.0    1-7/5-8
clip@q0             0.628 0.037 1.000  0.618 0.698   0.064   0.046  0.047  0.044  510.0    1-7/3-8
hard@q50            0.628 0.042 0.850  0.539 0.729   0.046   0.051  0.053  0.043  255.0    1-8/1-8
weighted@q50        0.628 0.042 0.850  0.588 0.545   0.048   0.038  0.037  0.038  255.0    1-7/8-8
sqrt@q50            0.628 0.042 0.850  0.619 0.674   0.056   0.043  0.043  0.042  255.0    1-7/5-8
clip@q50            0.628 0.042 0.850  0.632 0.685   0.060   0.044  0.045  0.043  255.0    1-8/7-8
hard@q75            0.628 0.054 0.793  0.568 0.727   0.058   0.049  0.051  0.045  127.5    1-8/1-8
weighted@q75        0.628 0.054 0.793  0.588 0.545   0.048   0.037  0.037  0.038  127.5    1-7/4-8
sqrt@q75            0.628 0.054 0.793  0.619 0.662   0.056   0.042  0.042  0.042  127.5    1-7/1-8
clip@q75            0.628 0.054 0.793  0.628 0.651   0.058   0.043  0.043  0.043  127.5    1-8/4-8
hard@q90            0.628 0.114 0.784  0.637 0.676   0.062   0.044  0.044  0.043   51.0    1-8/1-8
weighted@q90        0.628 0.114 0.784  0.588 0.540   0.048   0.037  0.037  0.038   51.0    1-7/1-8
sqrt@q90            0.628 0.114 0.784  0.622 0.616   0.057   0.040  0.040  0.040   51.0    1-7/1-8
clip@q90            0.628 0.114 0.784  0.601 0.580   0.050   0.040  0.040  0.039   51.0    1-8/1-8
hard@q95            0.628 0.143 0.441  0.635 0.588   0.067   0.040  0.040  0.040   25.5    1-8/1-8
weighted@q95        0.628 0.143 0.441  0.579 0.519   0.047   0.037  0.037  0.038   25.5    1-8/1-8
sqrt@q95            0.628 0.143 0.441  0.607 0.554   0.056   0.038  0.038  0.038   25.5    1-8/1-8
clip@q95            0.628 0.143 0.441  0.579 0.527   0.049   0.037  0.037  0.038   25.5    1-8/1-8
hard@q99           FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
weighted@q99       FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
sqrt@q99           FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
clip@q99           FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries

The tool RECOMMENDS hard@q95 (target 0.067, collateral 0.040) by sec 2.10.6's lexicographic procedure.
The freeze is a HUMAN decision: the experimenter proposes, the team ratifies at a
check-in, and only then is it frozen for the box.
```

### instruction-text capture (oracle base: clove only, 3 dreams)

```
gate pilot: data/dream_set_instruct_s1234.pilot.pt  (20 dreams, set_sha 195a6ecd3b563b1d7a934f995491a7f17f8e26c02da30078158c911f32a345c3, variant deflated, rank rule ratio-gap)
test 1 separability: pooled AUC 0.752 over 25 fact-read and 9975 other positions; per dream 0.718, nan, nan, nan, nan, nan, nan, 0.990, 0.619, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan

test 2 bake-off. THE DECISION METRIC is target vs collateral removal (sec 2.10.6);
precision/recall and oracle overlap are diagnostics -- label accuracy is a NON-goal,
and no scheme here is scored by any downstream training outcome.
rank column is the ratio-gap rule's per-layer span, /median where the rules disagree.

scheme                AUC  prec   rec oracle  stab  target  collat    ctx   batt  gated       rank
hard@q0             0.752 0.017 1.000  0.492 0.806   0.032   0.050  0.053  0.040  500.0    1-5/1-8
weighted@q0         0.752 0.017 1.000  0.591 0.484   0.048   0.038  0.038  0.041  500.0    1-8/8-8
sqrt@q0             0.752 0.017 1.000  0.614 0.684   0.052   0.043  0.043  0.044  500.0    1-7/2-8
clip@q0             0.752 0.017 1.000  0.564 0.752   0.048   0.047  0.048  0.047  500.0    1-6/3-8
hard@q50            0.752 0.025 0.767  0.497 0.815   0.036   0.050  0.052  0.046  250.0    1-6/1-8
weighted@q50        0.752 0.025 0.767  0.591 0.484   0.048   0.038  0.038  0.041  250.0    1-8/8-8
sqrt@q50            0.752 0.025 0.767  0.614 0.681   0.052   0.043  0.043  0.045  250.0    1-7/2-8
clip@q50            0.752 0.025 0.767  0.591 0.704   0.051   0.045  0.045  0.046  250.0    1-7/4-8
hard@q75            0.752 0.036 0.667  0.529 0.802   0.043   0.049  0.049  0.047  125.0    1-8/1-8
weighted@q75        0.752 0.036 0.667  0.591 0.483   0.048   0.038  0.038  0.041  125.0    1-8/3-8
sqrt@q75            0.752 0.036 0.667  0.609 0.659   0.050   0.042  0.042  0.044  125.0    1-8/1-8
clip@q75            0.752 0.036 0.667  0.650 0.658   0.058   0.042  0.041  0.045  125.0    1-8/2-8
hard@q90            0.752 0.071 0.667  0.547 0.717   0.047   0.044  0.043  0.046   50.0    1-8/1-8
weighted@q90        0.752 0.071 0.667  0.591 0.466   0.048   0.037  0.037  0.040   50.0    1-8/1-8
sqrt@q90            0.752 0.071 0.667  0.611 0.590   0.051   0.040  0.039  0.043   50.0    1-8/1-8
clip@q90            0.752 0.071 0.667  0.595 0.519   0.043   0.039  0.038  0.041   50.0    1-8/1-8
hard@q95           FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
weighted@q95       FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
sqrt@q95           FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
clip@q95           FAILED: layer 0's qcm basis came out empty at rank 1 over 1 gated queries
hard@q99           FAILED: no gated positions
weighted@q99       FAILED: no gated positions
sqrt@q99           FAILED: no gated positions
clip@q99           FAILED: no gated positions

The tool RECOMMENDS clip@q75 (target 0.058, collateral 0.042) by sec 2.10.6's lexicographic procedure.
The freeze is a HUMAN decision: the experimenter proposes, the team ratifies at a
check-in, and only then is it frozen for the box.
```

### Reading (registered, not frozen)

- **Target removal tops out at 0.067** (hard@q95, no-prefix) against the
  goal block's (100%, 0%) ideal: at the d_state/16 address budget, an
  aggregate basis built from a dream's own gated queries removes ~7% of
  the fact readout along oracle queries, at ~4% collateral. Gate precision
  is the visible cause (0.04-0.14 across taus: most gated positions are
  real state reads of NON-fact content, so most basis directions are
  context). Whether ~7%/4% is enough consolidation pressure is exactly
  what the box's A-vs-B4 contrast measures; the pilot cannot answer it
  (sec 2.10.6's named non-goal: no mini-training outcomes).
- **The scheme families barely separate** on this capture (target 0.046-
  0.067 across everything scoreable); hard beats weighted variants
  slightly and wins ties by simplicity anyway. The tau=q99 rows die on
  1-query qcm bases -- a scheme floor, not a finding.
- **Rank rules disagree broadly** (ratio-gap spans 1-8 vs median often
  pinned at budget); per sec 2.7 the SIMPLER rule that behaves is frozen --
  on this evidence ratio-gap behaves (structure-following) and median
  mostly saturates the budget. Proposal below.
- Tool recommendations: **hard@q95** (no-prefix), clip@q75 (instruct,
  weaker oracle base). Experimenter proposal for the check-in: **freeze
  hard gating at the q95-equivalent absolute threshold printed in the
  jsonl, rank rule ratio-gap** -- and treat gate PRECISION (not scheme
  family) as the lever if the team wants higher target removal.

