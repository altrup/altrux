# Experiment notes — 2026-07-23 (fresh box, GH200 96 GB)

Session start 02:35 UTC. Standing direction: `../discussion/DISCUSSION-20260722-stage2-readout.md`.
Plan: B0 dream-fidelity probe on 447-T3 (archive-20260722-test3-lineage/step-447),
B1 plateau-vs-erosion resume from archive step-476 on train_chains_split.pt to
total offset ~+120–130, probing every ~13 steps. B2a (window-4) if time allows.

Box state at boot: only `checkpoints/archive-20260722-test3-lineage/` present
(steps 396/447/476). No prepared data (expected). torch 2.11.0+cu128, CUDA OK.
MODEL_NAME=mamba2_2_7b_memory.

## Log

### 02:40 UTC — data prep launched (prep tmux)

Verbatim, chained in `prep` tmux (reproduces the prior run's pool exactly):

1. `make data-memory`
2. `make prepare ARGS="--hf-dataset HuggingFaceH4/ultrachat_200k --max-examples 20000 --max-len 1024 --output data/train.pt"`
3. `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7 --split-episode-rate 0.15 --split-qa-rate 0.9 --split-gap-min 1 --split-gap-max 4 --output data/train_chains_split.pt"`
   (Test-3 v2 data; verify: ~5498 splits, ~71% single-QA, ~15857 sleeps)
4. `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7"` → train_chains.pt
   (xs regen, for B2a later)

### 02:45 UTC — checkpoint restore

- Copied `archive-20260722-test3-lineage/step-476` → `epoch-1/step-476`
  (archive preserved). step-476 intact: lora_config.json, optimizer.pt,
  state.pt, trainable.pt. step-447 (probe target) intact: lora_config.json,
  state.pt, trainable.pt (no optimizer — fine, inference only).

### 02:49 UTC — B0 dream-fidelity launched (train tmux)

- Verbatim: `make dream-fidelity ARGS="--checkpoint ../models/mamba2_2_7b_memory/checkpoints/archive-20260722-test3-lineage/step-447"`
  (defaults: n-probes 4, prime 2048, gen 128, temp 0.8, data
  train_memory_longalign.pt, seed 1234). Log: `logs/dream-20260723-023918.log`.
- Prep tmux continuing in parallel (babilong .pt → merge → ultrachat → chains).

### 02:41 UTC — data prep DONE, verified

- All four datasets built in ~4 min (GH200 box is fast; prior box took much
  longer). NOTE: my `tee logs/prep-20260723-boot.log` silently wrote nothing
  (logs/ didn't exist yet at tee start); stats recovered from pane history.
- train_chains_split.pt: 2753 chains, 230.9M tokens, **15857 sleeps, 5498
  split-tail (3918 single-QA = 71%)** — EXACT match to prior run's accepted
  v2. train_chains.pt (xs regen): 10387 sleeps — exact match too. Both safe
  for fingerprint-matched resume.

### 02:52 UTC — B0 RESULT: dream test NEGATIVE at prime 2048

Log `dream-20260723-023918.log`, checkpoint 447-T3, defaults (n 4, prime
2048, gen 128, temp 0.8):

- Jaccard overlap (corpus-common excluded): M-primed 0.007 vs M-random 0.009,
  mean delta **−0.002** (rows: +0.000/−0.008/+0.001/+0.000). No steering.
- Decoded samples: M-primed generations are fully off-topic (math drills
  after a French-heritage doc; row 1 degenerates into a repetition loop
  "specimen is a sibling of is"). Not even topic fidelity — below the
  "gist-faithful, fact-lossy" prediction. M-random control equally off-topic
  but coherent.
- Read: with SSM wiped, backbone priors dominate and the M read injection
  doesn't steer free generation at all at this operating point. Possible
  confound: prime 2048 is below the regime where M's scored contribution
  shows (gist harness prefix 6144).

### 02:52 UTC — B0 rerun at prime 6144 (best-case variant)

- Verbatim: `make dream-fidelity ARGS="--checkpoint ../models/mamba2_2_7b_memory/checkpoints/archive-20260722-test3-lineage/step-447 --prime 6144"`
  Log `dream-20260723-024201.log`. Decision rule: still ≈0/negative delta and
  off-topic samples → dream generation is dead at this checkpoint; the
  M→weights consolidation sketch needs the transcript in the loop (collapses
  toward transcript-replay baseline). B1 starts right after this finishes.

### 02:57 UTC — B0 rerun RESULT: prime 6144 also NEGATIVE. Dream test CLOSED.

Log `dream-20260723-024201.log`: mean delta **−0.011** (rows −0.005/−0.035/
+0.004/−0.007). M-primed samples again off-topic; row 0 again collapses into
DM-mathematics-style drills — SAME attractor as the prime-2048 run. Read: the
M read injected over a wiped SSM acts as OOD noise that pushes generation into
low-entropy corpus modes; it does not steer content at any prime length tested.

**Conclusion (decision the probe was funded for): M cannot generate a faithful
dream at 447-T3. The M→weights consolidation sketch as free-dream-replay is
dead; any consolidation loop needs the transcript in the loop, i.e. it
collapses toward the transcript-replay baseline.** Discussion item for the
team: whether that baseline is still worth building, or whether stage-2
machinery work (integration point / compute depth) subsumes it.

### 03:00 UTC — B1 LAUNCHED (train tmux)

- Verbatim: `make resume ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
- **Resume point: epoch-1/step-476** (restored from
  archive-20260722-test3-lineage). Log `train-20260723-*.log` (newest).
- Dataset fingerprint MATCHED the regenerated train_chains_split.pt
  (next_ptr 31 preserved); "no saved internal state" → 8 slots restart from
  their example beginnings (mem_state.pt only exists on rotation — same as
  prior legs).
- Starting metrics (step 476, token 0): beta ~0.58, retain ~0.980, surprise
  ~0.23, o_t_norm 13.2–13.6, min_cos_sim 0.82–0.89, active 21/21. Healthy.
- Plan: train to offset ~+120–130 (abs step ~516–526), probe every ~13 steps
  (abs ~489/502/515) with the LongAlign gist config. Decision rule per
  DISCUSSION: hold ≥ ~+0.015 with long-range positive → PLATEAU REAL; slide
  < ~+0.01 or long-range negative → SLOW EROSION.

### 03:02 UTC — side probe (user asked): intact-SSM generation sanity

- Feyi asked whether the model spits gibberish generally. Scratch script
  (sanity_gen.py, scratchpad): prime 2048 SSM-INTACT (no sleep), window-1
  generation, 447-T3, n 2. Log `logs/sanity-gen-20260723.log`. Runs alongside
  B1 (96 GB card, no contention).

### 03:08 UTC — sanity gen RESULT: not gibberish, but a QA degeneration mode

Log `sanity-gen-20260723.log` (intact SSM, prime 2048, window-1 gen, 447-T3):

- Row 0 (French village doc): on-topic continuation (Loiret wiki Category
  lines). Normal-condition generation works.
- Row 1 (doc ending in a question): collapses into a `[USER] dependent
  specim` role-marker repetition loop — same row degenerated in both dream
  runs, so the failure is prompt-shaped (question→answer format), not the
  SSM wipe.
- Read: answering-mode generation at window 1 from a window-8-trained
  checkpoint is fragile — concrete supporting evidence for the train/serve
  mismatch concern attached to the deferred window-1 leg (DISCUSSION B2
  section). Worth a mention at the debrief.

### 03:15 UTC — B1 first checkpoint step-480 (+84); probe launched

- Training pace ~60 s/step; first save arrived at 480 (resume-anchored token
  counting makes the first save early, per save_checkpoint docs).
- Verbatim (probe tmux, concurrent with training):
  `make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-480"`
- Context for the read (T3 tail): +64 +0.0187, +72 ~+0.019/0.021, +80 ~+0.022.
  Decision band: ≥ ~+0.015 & long-range positive → plateau; < ~+0.01 or
  long-range negative → slow erosion.

### 03:2x UTC — step-480 probe OOM; plan revised to leg-then-sweep

- Probe concurrent with training OOMed (`probe-20260723-025050.log`): train
  process holds ~69.5 GiB of the 94.5 GiB card; the 16×6144 gist probe needs
  more than the ~24 GiB left (write working set at model.py:323). GH200 fits
  probe+train only for small probes (4×2048 dream was fine). GOTCHA for
  future GH200 runs: full gist probes do NOT run alongside training.
- Revised to the prior run's pattern: train the whole leg 476→~519 (+123),
  C-c, then sweep probes {480, 493, 506, 519} sequentially, then decide
  plateau-vs-erosion (same decision band). Wake armed on step-519 existence.

### 03:33 UTC — B1 leg complete (476→520, stopped); sweep launched

- Saves actually landed every ~4-5 steps (480/484/488/…/518), denser than
  the expected ~13 — token counting on this box differs from the prior box's
  pacing. step-518 = +122, inside the +120–130 target; C-c at step-520.
- Training was healthy the whole leg: loss 1.69–1.92, surprise 0.03–0.40
  (never flat), o_t_norm 13→23 drift, retain 0.958–0.981, alpha 0 (per-slot
  lines show beta/retain; no non-finite warnings at any point).
- Sweep (train tmux, sequential, verbatim):
  `for s in 480 492 505 518; do make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-$s"; done`
- Offsets: 480=+84, 492=+96, 505=+109, 518=+122. Decision band per
  DISCUSSION: hold ≥ ~+0.015 w/ long-range positive → PLATEAU; < ~+0.01 or
  long-range negative → SLOW EROSION.

### Sweep results (LongAlign gist, n 16, seed 1234)

| step | offset | gist-delta (SEM) | long-range | recency | dist-delta |
|------|--------|------------------|------------|---------|------------|
| 480 | +84 | +0.0134 (0.0045) | +0.0089 | +0.0045 | +0.0105 |
| 492 | +96 | +0.0193 (0.0070) | +0.0262 | −0.0068 | +0.0210 |
| 505 | +109 | +0.0125 (0.0050) | +0.0145 | −0.0020 | +0.0137 |
| 518 | +122 | +0.0141 (0.0057) | +0.0193 | −0.0053 | +0.0165 |
| 544 | +148 | +0.0211 (0.0049) | +0.0109 | +0.0102 | +0.0102 |

- 544 (log probe-20260723-043455): headline strongest since +96; mild
  recency tilt (recency +0.0102, dist-delta down to +0.0102) but long-range
  still positive. ANOMALY: awake-mem +0.3257 (SEM 0.3126) — one probe row
  blew up in the no-wipe branch; headline branches normal. Watch whether it
  recurs at 557.

### 03:55 UTC — B1 SWEEP VERDICT: PLATEAU (provisional); extension leg launched

- Four points flat around ~+0.0148 mean; long-range positive at every point
  (+0.0089…+0.0262); recency ≈0/negative; dist-delta stable (ends +0.0165).
  NO erosion signature. Headline straddles the ≥+0.015 band edge but the
  structure criterion is cleanly met. Shelf is lower than the prior leg's
  ~+0.019–0.022 tail (the resume knocked it down a notch), but it is NOT
  sliding: split data looks erosion-resistant.
- Probe logs: 480 probe-…-033122, 492 -033702, 505 -034239, 518 -034814.
- Per decision rule (+ Feyi's live request to extend if good): resumed
  verbatim from step-518 (next_ptr 41, fingerprint matched, slots restart
  from example beginnings as before) to ~step-558 (+162); then probe
  {~538, ~558}. Confirmation bar: both ≥ ~+0.012 with long-range positive
  → PLATEAU REAL goes in the debrief; either collapsing → note the shelf
  as fragile.

- 480 log: probe-20260723-033122.log. Awake-mem +0.0177. Thirds
  +0.0009/+0.0190/+0.0203. Prior leg's +80 was ~+0.022 → some slide across
  the resume, but long-range positive and structure intact. Between decision
  bands; trend decides.

### 04:40 UTC — extension leg complete (518→557, +161); confirmation probes

- Resume point: epoch-1/step-518 (next_ptr 41, fingerprint matched). Trained
  518→557, stopped via C-c. Healthy throughout (retain dipped to 0.929–0.936
  a few times, recovered each time; alpha 0.0001; w1_abs_max 0.0824 stable;
  no non-finite).
- Confirmation probes (train tmux, verbatim):
  `for s in 544 557; do make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-$s"; done`
  Offsets: 544=+148, 557=+161. Bar: both ≥ ~+0.012 w/ long-range positive →
  PLATEAU REAL.

### 04:50 UTC — B1 VERDICT: PLATEAU REAL; new late-checkpoint anomaly

| 557 | +161 | +0.0210 (0.0069) | (contaminated) | (contaminated) | +0.0125 |

- Full series +84→+161: 0.0134 / 0.0193 / 0.0125 / 0.0141 / 0.0211 / 0.0210.
  Flat-to-rising over 77 steps, wiped-branch structure stable (dist-delta
  0.0102–0.0210). **Split data is erosion-resistant: B1 decision → PLATEAU
  REAL; split recipe becomes the base for stage-2 harness runs.** Shelf sits
  at ~+0.015–0.021, ~35–55% below T1's transient +0.0344 peak but durable.
- **NEW ANOMALY (real finding): no-wipe branches destabilize with continued
  training.** awake-mem: +0.0137 (518) → +0.33±0.31 (544) → +3.07±0.68 (557);
  recency/long-range at 557 contaminated (±0.39). Fixed seed/rows across
  probes → checkpoint-driven. Positive sign = the no-wipe-ABLATED branch
  (M removed, SSM never wiped) collapses: the awake model is becoming
  hard-reliant on M over long unwiped horizons. Consistent with o_t_norm
  drift 13→27 over the leg. Two readings: (a) M is now load-bearing awake
  (deep integration — arguably the goal!), (b) norm drift toward instability
  that M happens to compensate. Checks running: seed-999 rerun @557
  (row-dependence) + held-out ultrachat @557.
- 557 log: probe-20260723-044035.log.

### 05:00 UTC — seed-999 @557: anomaly is systematic, not row-specific

Log probe-20260723-044710.log: gist-delta +0.1770 (SEM 0.1508), dist-delta
+0.4318 (SEM 0.4127), awake-mem **+4.4619 (SEM 0.5808)** — most rows
elevated; on this row set even WIPED-branch ablations blow up.

- Mechanism read: with --freeze-lora only memory params train; injection
  magnitude grew all leg (o_t_norm 13→27). The backbone's state dynamics
  co-adapt to injected reads; ablating M at late checkpoints leaves the
  backbone OOD → logprob collapse. The harness metric shifts meaning from
  "M adds recall" to "M is load-bearing for stability."
- Caveats on tonight's verdict: PLATEAU is clean through +122 (all branches
  sane at seed 1234); at +148/+161 seed-1234's wiped rows stayed clean
  (+0.0211/+0.0210) but seed-999's do not. Treat step-518 as the last
  fully-clean checkpoint; 544/557 headline numbers valid on their measured
  rows but the instrument is saturating.
- Debrief items: (a) does "ablated collapses" count toward the goal (M
  earning its place) or is it a confound to control (e.g. norm-matched
  ablation, or co-training a no-M path)? (b) if a durable deliverable is
  wanted from this lineage, prefer step-518 (or re-measure 544/557 with an
  ablation that keeps the backbone in-distribution).

### 05:05 UTC — held-out ultrachat @557: BEST HELD-OUT RESULT TO DATE

Log probe-20260723-045248.log (prefix 2048, n 16, seed 1234):

- gist-delta **+0.0895 (SEM 0.0157)** vs prior best +0.0120 (435-orig);
  long-range **+0.0456** vs prior best +0.0086 (447-T3); recency +0.0439;
  dist-delta +0.0197; awake-mem +0.0781 (elevated, not collapsed).
- Interpretation discipline: the raw gist-delta is inflated by the
  M-dependence confound (ablated backbone weakened). But long-range =
  intact − recent compares branches that BOTH carry M, so the confound
  largely cancels — the +0.0456 long-range is the most defensible headline:
  ~5× the prior best held-out long-range signal. Split-data plateau
  checkpoints generalize off-corpus far better than any peak harvested
  before.
- Debrief framing: tonight's lineage trades the LongAlign transient peak
  (+0.0344, corpus-fit) for a durable shelf that transfers (+0.0456
  held-out long-range at +161 and still climbing on that axis earlier in
  the sweep). Candidate deliverable: step-557 (with the dependence caveat)
  or step-518 (fully clean instrument).

### 05:10 UTC — B2a LAUNCHED (window-4 densification A/B)

- Checkpoint housekeeping: moved epoch-1/{step-476, 480…557} (20 dirs) →
  `archive-20260723-b1-extension/` (step-476 there duplicates
  archive-20260722-test3-lineage/step-476). Restored
  archive-20260722-test3-lineage/step-396 → epoch-1/step-396.
- Verbatim: `make resume ARGS="--data data/train_chains.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 4 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
- Resume point: epoch-1/step-396, xs data (fingerprint matched, next_ptr 34).
  Train to ~step-447 (+51); then probe sweep nearest saves to +12/+24/+51 vs
  T1 window-8 trajectory (+0.0144/+0.0186/+0.0344). Lift → densification
  works (fund window-1 next run); flat/mixed → suspect write fidelity /
  width; worse → window-8 pooling is load-bearing.

### 05:15 UTC — B2a OOM at batch 8; relaunched at batch 4

- First B2a launch OOMed immediately: window 4 doubles retained
  write/injection graphs per chunk-48 (12 vs 6); 94.5 GB card full; failing
  alloc was the injection_cos_sim diagnostic (model.py:822), i.e. pure
  pressure, not a leak. GOTCHA: window-4 training does NOT fit at batch 8 /
  chunk 48 on 96 GB.
- Relaunch (verbatim): `make resume ARGS="--data data/train_chains.pt --eos-weight 32 --batch-size 4 --chunk-len 48 --memory-window 4 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
  **DEVIATION from DISCUSSION spec: --batch-size 4 (was 8).** accum-tokens
  unchanged → same tokens/optimizer step; per-token semantics unchanged; but
  slot parallelism halved → the A/B vs T1's batch-8 trajectory is no longer
  single-knob. Read B2a's result with that caveat.

### 06:35 UTC — B2a leg complete (396→~447 at window 4, batch 4); sweep launched

- Healthy throughout (~105 s/step, ~1.75× window-8 pace; o_t_norm 13→22,
  surprise never flat, retain 0.969–0.981, no non-finite). Saves:
  404/412/420/428/437/445.
- Sweep (verbatim): `for s in 412 420 445; do make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-$s"; done`
  Offsets +16/+24/+49 vs T1 window-8 (+12 +0.0144 / +24 +0.0186 / +51 +0.0344).

### B2a sweep results (window 4, LongAlign gist, n 16, seed 1234)

| step | offset | gist-delta (SEM) | long-range | recency | dist-delta | T1 w8 @≈offset |
|------|--------|------------------|------------|---------|------------|-----------------|
| 412 | +16 | +0.0267 (0.0038) | +0.0372 | −0.0105 | +0.0226 | +0.0144 (+12) |
| 420 | +24 | +0.0326 (0.0042) | +0.0469 | −0.0143 | +0.0207 | +0.0186 (+24) |

- 420 log: probe-20260723-063115. Awake-mem +0.0402. Near T1's peak value at
  half the offset.

| 428 | +32 | +0.0324 (0.0048) | +0.0278 | +0.0046 | +0.0192 | (+38: +0.0304 T3) |
| 445 | +49 | +0.0296 (0.0036) | +0.0140 | +0.0156 | +0.0087 | +0.0344 (+51, T1 peak) |

- 428 log: probe-20260723-064245. Headline flat vs 420; structure softening.

| 437 | +41 | +0.0336 (0.0054) | +0.0244 | +0.0092 | +0.0130 | — |

- 437 log: probe-20260723-064820. **Window-4 argmax.**

### 06:55 UTC — B2a VERDICT: densification scales the clock, NOT the ceiling

- w4 curve: +16 +0.0267 / +24 +0.0326 / +32 +0.0324 / +41 **+0.0336** (argmax)
  / +49 +0.0296 (eroding). w8 (T1): +12 +0.0144 / +24 +0.0186 / +38 +0.0304 /
  +51 **+0.0344** (argmax).
- Peak heights equal within SEM (+0.0336 vs +0.0344); peak arrives ~20%
  earlier in steps (~2.5× fewer wall-clock-normalized tokens-per-write);
  matched-offset "lifts" are pace effects. Erosion also arrives
  proportionally earlier. **The ~+0.034 LongAlign ceiling is invariant to
  injection/write cadence 8→4.**
- Consequences per DISCUSSION's decision rules: window-1 leg and the learned
  sparse gate are MOOT for peak height (the gate remains at most an
  efficiency device). Ceiling suspects now: integration point (MAL→MAC/MAG),
  compute depth over the read, write fidelity / bandwidth width — the
  shape-changing tier.
- Caveat: batch 4 (not 8) per the OOM deviation; peak-equality across
  cadences is an invariance result and unlikely to be a batch artifact, but
  a purist rerun at batch 8 / chunk 24 could close that hole.
- Held-out ultrachat probe on 437 launched (verbatim, prefix-2048 config,
  seed 1234) for the generalization record.

### Planned next

- 445 log: probe-20260723-063650. Recency-conversion signature already in
  (recency positive, long-range fallen, dist-delta collapsed) — window 4
  compresses the whole transient: climbs ~2× faster AND erodes earlier.
  Argmax bracketing probes launched on saves 428 (+32) and 437 (+41),
  verbatim same probe command.

- 412 log: probe-20260723-062549. Awake-mem +0.0280 (instrument clean).
  ~2× the window-8 headline at the matched early offset, long-range
  dominant. Densification lift showing immediately.

### 07:00 UTC — held-out ultrachat @437 (w4 argmax) + CLOSE

- Log probe-20260723-065440: gist-delta +0.0124 (SEM 0.0051), long-range
  +0.0104, dist-delta +0.0018. Transfers like the old w8 xs peaks (435:
  +0.0120) — nothing special. **B1's step-557 (+0.0895 / long-range +0.0456
  held-out) remains decisively the best generalizer of anything measured.**

## CLOSING SUMMARY (run over — shutdown checklist executed)

What this run established, in order of importance:

1. **B1: PLATEAU REAL.** Split-data lineage holds gist-delta ~+0.015–0.021
   from +84 through +161 (six probes, no erosion signature in wiped
   branches). Split recipe = base recipe for stage-2 going forward.
2. **Best held-out result to date: step-557** (archive-20260723-b1-extension/):
   ultrachat gist-delta +0.0895, long-range +0.0456 (~5× prior best; the
   long-range number is dependence-confound-resistant since both branches
   carry M). Candidate deliverable, with the caveat below.
3. **NEW: M-dependence / instrument saturation at late checkpoints.**
   Continued memory-only training makes the backbone co-adapted to M:
   ablating M collapses logprobs on a growing set of rows (awake-mem
   +0.0137@518 → +3.07@557 seed-1234; seed-999 shows wiped branches also
   affected). Ablation deltas at 544/557 partly measure dependence, not
   recall. step-518 = last fully-clean checkpoint. Debrief question: is
   deep reliance the goal achieved, or a confound needing a norm-matched
   ablation control?
4. **B2a: densification scales the clock, not the ceiling.** Window-4 peak
   +0.0336@+41 ≈ window-8 peak +0.0344@+51; lifts at matched offsets are
   pace effects; erosion also arrives earlier. Window-1 leg and learned
   sparse gate are moot for peak height. Ceiling suspects → integration
   point / compute depth / width. (Caveat: batch 4 not 8, OOM-forced.)
5. **B0: dream test NEGATIVE (both prime 2048 and 6144).** M cannot steer
   free generation over a wiped SSM (deltas ≈0/negative; samples off-topic;
   M-injection pushes into low-entropy corpus attractors). Free-dream
   M→weights consolidation is dead; any consolidation loop needs the
   transcript (collapses toward transcript-replay baseline).
6. **Side finding (Feyi's request): generation sanity.** Intact-SSM
   continuation is coherent/on-topic for documents; documents ending in a
   question tip the model into role-marker repetition loops at window-1
   serving — concrete evidence on the train/serve mismatch.

Box gotchas discovered (GH200 96 GB): full gist probes do NOT fit alongside
training (OOM); window-4 training needs batch 4 at chunk 48. Save cadence
lands every ~4-5 steps with these args, not ~13.

Checkpoint layout at termination: epoch-1/step-396 (copy);
archive-20260722-test3-lineage/{396,447,476} (untouched);
archive-20260723-b1-extension/{476, 480…557} (20 dirs; 557 = star);
archive-20260723-b2a-window4/{404…445} (437 = w4 argmax).

Suggested next-run agenda: (a) decide dependence-vs-achievement on the
ablation harness (norm-matched ablation control?), (b) held-out-first
harvesting from the split lineage (557 wasn't even selected on held-out —
the argmax there may be higher), (c) integration-point/compute-depth bets,
(d) purist B2a rerun at batch 8 / chunk 24 only if the invariance result is
doubted. Datasets regenerate exactly from the verbatim commands above.
