# Discussion notes — 2026-07-21, post-mortem of the gist-eval / peak-erosion run

Team discussion (altrup + Claude) after `EXPERIMENT_NOTES-20260721-030057.md`.
STANDING DIRECTION for the next experimenter run — self-contained; read
alongside `DISCUSSION-20260720-gist-eval.md` (its rejected items still stand).

## Reinterpretation of the run's conclusions

1. **The gist reinterpretation is confirmed and replicated.** Gist-delta
   rises monotonically 180→435 (−0.068 → +0.0285, seed-5678 replication
   +0.0299), inverse to exact-code recall, with the right structure at 435
   (long-range dominant, thirds rising, survives a distractor episode+sleep).
   `epoch-1/step-435` is a real episodic-gist checkpoint — THE deliverable.
2. **Four continuation regimes eroded the peak with one signature**
   (long-range → recency conversion, thirds inverting, dist-delta decaying).
   Regime 4 had 435's true optimizer state, so fresh-Adam shock is refuted
   as the sole cause. Established: *naive continued training resumed at 435
   does not extend the peak, regardless of data mix.*
3. **Why regime 4 failed — three mechanisms, first two confounded in every
   test so far:**
   - (a) The loss landscape prefers recency: post-sleep tokens are mostly
     predictable from the last few hundred pre-sleep tokens, so ANY
     continued optimization converts M toward recency/always-on injection.
     Under this story 435 is a transient the optimizer passed through.
   - (b) Resume discontinuity: every failed regime changed the dataset at
     resume, which discards slot states, restarts the data stream at chain
     beginnings, and puts a shifted distribution under a converged point.
     Regime 4's +50 probe supports this a little: structure was still fully
     435-like at +50 and only inverted by +100 — perturb-then-slide, not
     immediately broken.
   - (c) Sleep density: the all-signals data had ~22.5k sleeps vs ~10.4k
     (xs), halving wake length and starving long-range signal.
   (a) vs (b) is exactly what Test 1 below separates.
4. **Slot states are gone** (no `state.pt` / mem data survives for any
   checkpoint): every future resume is weights (+optimizer where uploaded)
   with COLD slots. This makes Test 1 asymmetric — success is strong
   evidence (recipe robust even to cold start; failures pinned on the
   dataset change), failure is ambiguous (transient peak vs cold-slot/
   stream discontinuity). Both failure explanations converge on the same
   practical rule: **peaks are only reachable inside continuous runs, never
   by resuming at a peak.**
5. **Agreed signal hierarchy** for any future training mix:
   1. Cross-sleep fact queries — backbone, never removed. The only signal
      ever present during a gist rise (mid-sleeps were a no-op; xs's only
      real memory signal), and the only one with query-like amplitude: the
      answer is near-unpredictable without the fact, so its loss mass
      routes through M. Regime 1 (queries removed) collapsed gist in 75
      steps. Their function: guarantee "something must survive the sleep,
      from far back"; the LM loss shapes what — and that equilibrium
      drifted toward gist while verbatim recall faded.
   2. Split-interleaving, especially single-QA splits — the gist-shaped
      complement. Enforced distance (nothing between head and tail helps)
      but weak amplitude for conversational tails (~0.02–0.03 nats/token);
      single-QA splits (document now, answer episodes later) have
      query-like amplitude with no template. Regime 3's +49 probe had the
      best structure of any continuation attempt before sliding — right
      direction, insufficient strength.
   3. Mid-conversation sleeps — dense but recency-shaped.
   4. Sentence-boundary sleeps — same but worse distance profile. Built
      (`--sentence-sleep-rate`, e69daaf), deliberately UNUSED until the
      reproduction question is settled; only ever as an addition to a
      validated recipe, never a substitution.

## North star (agreed): what "memory working" means, and where 435 stands

The end goal is memory that still *remembers you* after a sleep. That
decomposes into three abilities; judge every run against them, not just
the next delta:

1. **Gist persistence** (topic, ongoing work, conversation shape). 435:
   measurable, not usable — gist-delta +0.0285 against a wipe cost of
   ~0.19–0.22, i.e. M recovers ~13% of what the sleep destroys.
2. **Specific-fact recall on demand** ("my name is X", post-sleep). 435:
   zero — the step-180 verbatim ability (+0.4) was traded away as gist
   grew. Under current training these compete; the end goal needs both.
3. **Durability across many sleeps/sessions.** 435: retention roughly
   halves through ONE intervening episode+sleep (dist-delta +0.016).

Staged program: (1) prove the training signal is reproducible (Tests 1–3
below); (2) find what bounds magnitude; (3) recombine — gist foundation
first, then engineered recall demands earn their way back
(DISCUSSION-20260720's phrase), plus multi-sleep durability signals
(the --split-gap-max knob points that way).

**Capacity is NOT the bottleneck — correct a misreading in the run notes:**
"218 memory params" is a TENSOR count. The trainable memory machinery is
40.8M params (front-end q/k/v + per-layer injections; LoRA is a separate
21.3M), and the memory state M itself is a Titans fast-weight MLP of ~52M
values per sequence (w1 10240x2560 + w2 2560x10240). Storage is ample.
The whisper-scale effect and zero fact-recall point instead at:
- **read-out bandwidth**: retrieval runs through 128-dim bottlenecked
  injections into alternate layers >= 22, once per memory window, gated —
  M could hold a fact and be unable to say it through that straw;
- **write fidelity**: one surprise-driven gradient step per window decides
  whether a fact lands retrievably or smears into topical bias;
- **decay**: per-token retain ~0.95, worst-case half-life ~7k writes
  (~55k tokens, model.py comment); alpha forgetting-gate dead as trained;
- **the training objective** (everything the erosion regimes showed).
These are stage-2 suspects, in that order, if Tests 1–3 validate
reproducibility but magnitude saturates at whisper scale.

## Standing direction: three tests, in order

### Test 2 first on the box (probe-only, runs while Test 1's data regens):
435 robustness

Does 435's gist generalize beyond LongAlign (which was in training data)?

- Build gist eval sets from babilong and held-out UNCAPPED ultrachat via
  prepare_data.py (ultrachat uncapped: max real length 5368, median 1121).
- Re-derive eligibility per corpus BEFORE sweeping (the C=512 lesson:
  count conversations with a boundary having ≥P before and ≥C after).
  Ultrachat needs prefix ≈2048; keep cont 512 / recent 576 / n 16 /
  seed 1234 where eligibility allows.
- Compare across checkpoints (435 vs e.g. 180) WITHIN a corpus only;
  never compare magnitudes across corpora.
- Any clearly positive gist-delta with long-range dominance on a second
  corpus = deliverable confirmed general. Flat ≈ 0 everywhere off
  LongAlign = report; 435 becomes "LongAlign-specific" and the write-up
  says so.

### Test 1: the 396 reproduction (the load-bearing question)

Does the 396→435 rise reproduce on the same data — or do peaks only exist
inside continuous runs?

- **Rebuild the original pool exactly.** Ultrachat re-prepped with
  `--max-len 1024` (the new 32768 default changes the pool 435 was trained
  on) plus the usual memory sources, then
  `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7"`
  (fact-rate 0.3 default, no split/mid-sleep flags). Verify in the regen
  log: ~10.4k sleeps, 0 mid-conversation, 0 splits. Mismatch → stop and
  investigate before training.
- **Resume `epoch-1/step-396`, `--freeze-lora`.** Expected messages:
  fresh optimizer (396's 412 MB optimizer covers the unfrozen param set —
  the b9941d8 param-match will fail; this MATCHES the original condition:
  freeze-lora began AT 396 with a fresh optimizer per
  EXPERIMENT_NOTES-20260720 08:14) and dataset-change slot reset (cold
  slots, unavoidable — see reinterpretation #4).
- **Probe at +13/+26/+39** (steps ~409/422/435): LongAlign gist config
  prefix 6144 / cont 512 / recent 576 / distractor 1536 / n 16 /
  seed 1234, gist+distractor+awake-mem. Judge the TREND, not the first
  point — cold slots may depress early steps.
- **PASS** (gist-delta ≥ ~+0.02 by +39, long-range dominant, thirds
  rising): the dataset-change-at-resume is the killer, and the recipe is
  real. Then KEEP TRAINING past +39 on the SAME data, probing every ~13
  steps — does it peak-then-erode anyway? That's the clean transient test,
  free. Then proceed to Test 3 with entry point 396.
- **FAIL** (flat/declining by +39): resumes can't reach peaks. 435 stays
  the deliverable. Proceed to Test 3 as a CONTINUOUS run (below) — that's
  the only remaining route to something better than 435.

### Test 3: regime-3-improved, as a continuous run (after Tests 1–2)

The split mechanism was right (enforced distance) but under-powered and
only ever tested as a resume-at-the-peak. Fix amplitude, run continuously.

- Data: xs base + strengthened splits, all on the `--max-len 1024` pool:
  `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7
  --split-episode-rate 0.5 --split-qa-rate 0.9 --split-gap-min 1
  --split-gap-max 4"`.
  Queries intact (backbone). NO mid-conversation or sentence sleeps — keep
  sleep density near the original cadence (mechanism (c)); check the regen
  log's single-QA split count is the majority of splits.
- Resume from step-396 (or earlier if Test 1 suggests it), `--freeze-lora`,
  `--head-weight 4.0` kept. Raise head-weight to 8.0 ONLY if a probe shows
  435-like structure but weak amplitude (structure right, magnitude low) —
  one knob at a time.
- Probe every ~13 steps, same config/seed. Stopping rule: keep the argmax
  checkpoint (peaks may be transients — that's fine, we harvest them);
  stop when 3 consecutive probes are below the best point with degraded
  structure. Archive procedure per DISCUSSION-20260720 (mv newer step dirs
  to `checkpoints/archive-<date>-<tag>/` before any non-latest resume).
- Success criterion: any checkpoint beating 435's +0.0285 with 435-like
  structure (long-range dominant, thirds rising, dist-delta positive,
  awake-mem small). That checkpoint replaces 435 as the deliverable.

## Explicitly considered and rejected

- **A fifth continuation regime resumed from 435** — four regimes, one
  erosion signature; the confound is understood and Test 1 addresses it
  directly. Don't relitigate without Test 1's answer.
- **Sentence-boundary sleeps in these runs** — capability built
  (`--sentence-sleep-rate`, e69daaf) and parked: recency-shaped distance
  profile matches the observed erosion signature, and adding it now would
  confound the reproduction question. First trial only ever as an addition
  to a validated-reproducible recipe.
- **Replacing queries with natural-continuation signals** — regime 1
  (fact-free) collapsed gist within 75 steps; queries are the only signal
  with guaranteed long-range loss mass. Natural signals complement, never
  substitute.
- **Warm-slot reconstruction for Test 1** — the mem/slot data no longer
  exists anywhere; accepted as a known deviation, handled by judging
  trends not first points.
- Carried over from DISCUSSION-20260720: chunk-len 48→64; literal episode
  repetition.

## Housekeeping (this session, committed locally)

- 42da3c5 — banked `EXPERIMENT_NOTES-20260721-030057.md`.
- e69daaf — `prepare_chains.py --sentence-sleep-rate` (sentence-boundary
  sleeps inside document turns; vocab-scan boundary detection). Built by
  agreement, unused by agreement. Tests + README + sft/CLAUDE.md updated
  (stale 1024-cap note fixed).
- 5588b6a — `prepare_chains.py --split-qa-rate` (separate rate for
  single-QA episodes) and `--split-gap-min/max` (sampled head→tail gap,
  default 2..2 = old behavior). Tests + README updated.
- altrup updated `.env`: `LAMBDA_RESUME_CHECKPOINT` = steps 180/256/295/332
  (weights only), `LAMBDA_RESUME_CHECKPOINT_FULL` = steps 396/435 (with
  optimizer.pt). All six verified present locally with optimizer.pt;
  no checkpoint has slot state.
