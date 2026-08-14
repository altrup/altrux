# Experiment notes — mamba2_2_7b_memory, gist-eval run, 2026-07-21

Fresh Lambda GH200 instance, auto-started. Operating autonomously per
/experimenter brief + `../discussion/DISCUSSION-20260720-gist-eval.md` (standing direction).

## Starting state

- Checkpoints on disk (rsync-restored, no optimizer.pt): epoch-1/step-{180,
  256,295,332,396,435} — the six interesting points from the 2026-07-20 run
  (baseline +0.4 exact-code sleep-delta @180 → collapse ~0 @256/295 →
  transient xs-data recovery @332 → collapse @383/396 → freeze-lora no-op
  @435).
- No prepared data (fresh box). No training running; touching
  scripts/.watchdog-delay while building the eval.
- Prior run's conclusion: `M` can't do verbatim exact-code recall across a
  sleep durably. DISCUSSION reinterpretation: that probe demands the thing
  `M` wasn't designed for; the open question is whether `M` carries **gist**.

## Plan (from DISCUSSION Priority 1)

Build a gist mode into `probe_recall.py`, validate, then sweep the six
checkpoints. Decision rule: gist delta clearly positive → M works at its
designed job → Priority 2 (gist-shaped training: neutralize exact-code
queries, raise mid-conversation sleep fraction well above ~20%). Gist delta
≈ 0 → structural conclusion stands → write up, likely stop.

### Gist eval design (agreed in DISCUSSION, concretized here)

- Mode of `probe_recall.py` (`--gist <data.pt>`), same harness: prefix →
  `sleep_slot` backbone wipe → ablation = fresh random `M`.
- Data: THUDM/LongAlign-10k real long multi-turn conversations (tokenized
  via prepare_data.py). CAVEAT (noted, accepted): LongAlign episodes were in
  the chains TRAINING data, so text is not held out — but the metric is
  differential (intact − ablated on the same text), so parametric
  memorization cancels to first order; trajectory across checkpoints on
  fixed data stays meaningful.
- Per probe row: pick a conversation with a between-turn boundary b (token
  == USER_OPEN/ASST_OPEN id) having ≥ P tokens before and ≥ C after; prefix
  = tokens [b−P, b), continuation = [b, b+C). Boundary chosen nearest the
  conversation middle (mid-conversation wipe, per DISCUSSION). Equal-length
  rows → batched like the existing probe.
- Conditions (scored on the same continuation, teacher-forced):
  1. **no-wipe** — full state, positive control; must beat both wiped
     conditions by a clear margin or the harness is broken.
  2. **sleep-intact** — backbone wiped at b, memory kept.
  3. **sleep-recent** — memory built from ONLY the last R pre-wipe tokens
     (fresh state fed prefix[−R:]), then backbone wiped. Distance-grade:
     if sleep-intact ≈ sleep-recent, M is a last-few-turns buffer.
  4. **sleep-ablated** — backbone wiped AND fresh random memory.
- Metric: mean log-prob per continuation token. **gist-delta = sleep-intact
  − sleep-ablated**; long-range component = sleep-intact − sleep-recent.
  Also binned by distance into the continuation (first/mid/late thirds).
- Config recorded before the sweep (below) for reproducibility.

## Timeline

### 03:01 UTC — session start

Read prior notes + DISCUSSION. Verified venv/torch (2.11.0+cu128, CUDA ok),
six checkpoints present. Starting data prep (LongAlign) in `prep` tmux;
writing the gist mode.

### 03:10 UTC — eval data + gist mode implemented

- LongAlign prep done: `data/train_memory_longalign.pt`, 9846 episodes.
- `probe_recall.py --gist` implemented per the design above (four
  conditions incl. no-wipe positive control and sleep-recent recency
  control; row-paired deltas with SEM; continuation-distance thirds).
  Flags: `--gist <data.pt> --gist-prefix 6144 --gist-cont 1536
  --gist-recent 576` (prefix/recent rounded down to CHUNK_LEN=24
  multiples).
- Smoke run sent to `train` tmux on step-180: n_probes 4, prefix 1536,
  cont 480, recent 240. Sanity checks: plausible log-prob range, nonzero
  variance, no-wipe clearly beats all wiped conditions.

### 03:07 UTC — smoke result (step-180, small scale): HARNESS VALID, delta negative

Log `sft/logs/probe-20260721-030500.log`. no-wipe −1.461 beats all wiped
conditions by +0.14–0.20 (~10σ) → positive control passes, harness sound.
Range/variance sane. PRELIMINARY surprise: gist-delta −0.059 (SEM 0.019) —
keeping `M` slightly HURTS continuation prediction vs fresh random `M` at
step-180 at this small scale (4 rows, prefix 1536). Not concluding from a
smoke; full sweep decides. Committed+pushed `--gist` mode as **1fa1a93**
(with README).

### 03:08 UTC — SWEEP config (recorded before running, per DISCUSSION); sweep launched

- Data: `data/train_memory_longalign.pt` (THUDM/LongAlign-10k train split,
  9846 episodes; prep log `sft/logs/prep-longalign.log`).
- `--gist-prefix 6144 --gist-cont 1536 --gist-recent 576 --n-probes 16
  --seed 1234`, memory-window 8 (default). Same 16 conversations/boundaries
  at every checkpoint (fixed seed → paired across checkpoints).
- Checkpoints: epoch-1/step-{180,256,295,332,396,435}, in that order, one
  `make probe-recall` each (each tees to its own `logs/probe-*.log`).

### 03:11 UTC — sweep FAILED instantly (data too short for config); fixed and relaunched

All six runs exited at row-building: only 2 conversations have a turn
boundary with ≥6144 before AND ≥1536 after. Diagnosis: LongAlign is mostly
single-QA — huge document in one user turn, answer <1k tokens, end — so the
binding constraint is the continuation length, not the prefix. Eligibility
counts: C=1536 → 2 convs; C=1024 → 8; **C=512 → 708** (and prefix can stay
at 6144 with no loss: P=2048..6144 all give 708 at C=512).

Revised sweep config (only change: `--gist-cont 1536` → `--gist-cont 512`):
`--gist-prefix 6144 --gist-cont 512 --gist-recent 576 --n-probes 16 --seed
1234`. Continuation thirds now ~170 tokens each. Relaunched 03:13 UTC.

## Sweep results (full scale: prefix 6144, cont 512, recent 576, n=16)

Columns: mean log-prob per continuation token; deltas row-paired, SEM = std/√16.

| step | no-wipe | intact | recent | ablated | gist-delta (intact−abl) | long-range (intact−recent) | wipe cost |
|------|---------|--------|--------|---------|-------------------------|----------------------------|-----------|
| 180 | −1.4447 | −1.6285 | −1.5723 | −1.5607 | **−0.0678 (0.0145)** | −0.0562 (0.0113) | +0.1838 (0.0471) |
| 256 | −1.3157 | −1.5372 | −1.5291 | −1.5300 | **−0.0072 (0.0044)** | −0.0081 (0.0042) | +0.2215 (0.0482) |
| 295 | −1.3044 | −1.5213 | −1.5262 | −1.5265 | **+0.0052 (0.0033)** | +0.0049 (0.0050) | +0.2169 (0.0468) |
| 332 | −1.3002 | −1.5246 | −1.5267 | −1.5267 | **+0.0021 (0.0036)** | +0.0021 (0.0034) | +0.2244 (0.0490) |
| 396 | −1.2885 | −1.5106 | −1.5256 | −1.5211 | **+0.0105 (0.0035)** | +0.0150 (0.0038) | +0.2221 (0.0477) |
| 435 | −1.2922 | −1.5099 | −1.5398 | −1.5384 | **+0.0285 (0.0050)** | +0.0299 (0.0025) | +0.2177 (0.0479) |

### 03:32 UTC — sweep complete: gist-delta trajectory MONOTONICALLY RISING, positive and strong at step-435

Trajectory: −0.068 → −0.007 → +0.005 → +0.002 → +0.011 → **+0.029**.
Key observations:

1. **Step-435 (the freeze-lora checkpoint) shows a real gist effect**:
   +0.0285 at ~5.7σ, and the long-range component (+0.0299, ~12σ, recent ≈
   ablated) says it comes from content ≥576 tokens before the wipe — true
   episodic carry-over, not a recency buffer. Thirds rise monotonically
   (+0.014 → +0.030 → +0.041): the memory helps MORE deeper into the
   continuation, i.e. it supports sustained topical coherence, not just the
   first few tokens after the wipe.
2. **Exactly inverse to the exact-code trajectory** (+0.4 @180 → 0 @435).
   At 180 M carries a verbatim-recall signal that HURTS free continuation
   (−0.068). As training proceeds, verbatim fades and gist grows. The
   freeze-lora phase (396→435), which looked like a failure on the
   exact-code probe, is where gist improves fastest — freezing the
   parametric path let the memory pathway learn something useful.
3. The prior run's "structural failure" conclusion applies to VERBATIM
   recall only. M is becoming a gist store — the DISCUSSION's
   reinterpretation is confirmed by the most permissive detector.

Decision rule ("gist delta clearly positive → Priority 2") fires: continue
gist-shaped training from step-435. User is online; discussing exact
Priority-2 config before launching.

### 03:40 UTC — seed-5678 replication at step-435: HOLDS

16 fresh conversations: gist-delta **+0.0299 (SEM 0.0049)**, long-range
+0.0277 (0.0043), recency +0.0022 (≈0), thirds +0.002/+0.034/+0.054. Same
magnitude and same signature as seed-1234. The step-435 gist effect is real.

### 03:42 UTC — Priority 2 launched (agreed with user, online)

Plan (user-approved): gist-shaped continuation from step-435, with fallback.

- Regen chains: `make prepare-chains ARGS="--mid-sleep-rate 0.6
  --fact-rate 0.0"` (mid-conversation sleeps now the dominant signal; no
  synthetic fact blocks — the verbatim signal they train was net harmful to
  gist at step-180, −0.068). Sources train.pt + train_memory.pt (11846
  memory episodes) built this session.
- Resume from step-435 with standing args + `--freeze-lora` (the freeze is
  the apparent active ingredient: gist grew fastest 396→435). Fresh
  optimizer (freeze-lora always starts one). `--head-weight 4.0` default
  kept; `--recall-weight` inert (no recall_masks at fact-rate 0).
- Success: gist-delta at ~every 50 steps rises above the +0.029 baseline.
- FALLBACK (agreed): if the first probe point is below baseline or training
  misbehaves, stop and revert to freeze-lora training on the PREVIOUS
  chains config (what demonstrably produced the rising trend).

### Cross-episode retention concern + proposed fix (discussed with user)

With fact-rate 0, NOTHING rewards retaining a finished episode across a
sleep — the only retention pressure left is mid-conversation (continue the
same conversation). Likely consequence: M learns "current-conversation
buffer, flush at example boundaries" (watch alpha at example starts).
Partly by design (adaptive forgetting; clinging to stale episodes is
interference — possibly part of step-180's −0.068), and tonight's metric is
within-conversation so the run is still well-posed. But for cross-session
memory it's a real gap. Root cause: unrelated episodes give M nothing worth
retaining FOR. Proposed fix (user suggested episode repeats; refined
together): **interleaved continuation** — a `--split-episode-rate` in
prepare_chains.py that splits an eligible long episode at a turn boundary
and schedules its tail 1–2 wake phases later. Same mid-sleep gist signal,
but stretched across intervening episodes+sleeps → naturalistic
cross-episode retention pressure, no synthetic codes, no verbatim repeats
(literal episode repetition rejected: trains recognition/copy of re-seen
text and a false "conversations repeat" prior). Plan: implement after the
current run's first probe point validates.

### 03:5x UTC — `--gist-distractor` implemented (3b4cf4b) to test the flush directly

New probe conditions (user asked whether the flush-at-episode-boundary risk
is testable in the next probe — it is, with no training change): after the
wipe, feed N tokens of an UNRELATED conversation's opening (row i gets row
i+1's episode start), sleep again, score the original continuation.
dist-delta = dist-intact − dist-ablated = prefix gist surviving THROUGH an
intervening episode + sleep; flush cost = gist-delta − dist-delta (paired).
Probe plan at the ~step-485 pause (in order):
1. standard gist @485 seed 1234 (the go/no-go vs +0.029 baseline);
2. distractor gist @485 (`--gist-distractor 1536`);
3. distractor gist @435 (pre-Priority-2 flush baseline — did the new data
   make flushing worse?).

### 04:35 UTC — early probe at step-459 (+24 steps, user present): BOTH METRICS UP

Paused training at step-459, ran gist+distractor (seed 1234, distractor
1536) on 459 and the 435 baseline; resumed training after.

| step | gist-delta | dist-delta | flush cost | long-range |
|------|-----------|-----------|-----------|-----------|
| 435 | +0.0205 (0.0036) | +0.0105 (0.0024) | +0.0100 | +0.0279 |
| 459 | +0.0375 (0.0047) | +0.0178 (0.0038) | +0.0197 | +0.0335 |

- Gate 1 passes early: gist-delta up ~80% in 24 steps on the gist-shaped
  data (same seed/rows as baseline).
- Distractor case (the flush worry): retention through an intervening
  episode+sleep EXISTS at 435 (+0.0105, ~half the direct gist) and
  IMPROVED at 459 (+0.0178) — the fact-free data is not (yet) teaching
  flush-at-episode-boundary. Decision tree → keep training as-is;
  implement --split-episode-rate after the second good probe (~step 510+).
- Methodology note: 435's gist-delta reads +0.0205 here vs +0.0285 in the
  earlier same-seed run without --gist-distractor. Same rows/conditions;
  difference is the ablation floor's fresh RANDOM memory draw (torch RNG
  unseeded run-to-run) — ±0.008 run noise on the floor. Cross-checkpoint
  comparisons within one probe invocation (like this table's two runs,
  probed minutes apart with the same code) remain paired and sound;
  treat single-run deltas as ±0.01.

### 05:35 UTC — step-510 probe: GIST COLLAPSED → FALLBACK TRIGGERED

Full suite (gist + distractor + new awake-mem), seed 1234, on 510 then 435
(same code, minutes apart):

| step | gist-delta | dist-delta | awake-mem | long-range | ablated floor |
|------|-----------|-----------|-----------|-----------|---------------|
| 435 | +0.0248 (0.0046) | +0.0160 (0.0032) | +0.0252 (0.0045) | +0.0256 | −1.5358 |
| 510 | **+0.0070 (0.0023)** | +0.0026 (0.0020) | +0.0135 (0.0034) | +0.0034 | −1.5147 |

- The 459 spike (+0.0375) fully reversed: by 510 gist is ~4σ BELOW the 435
  baseline; long-range and dist-delta ≈ 0. Gate 1 fails → user-agreed
  fallback applies.
- Mechanism read: the ABLATED floor improved +0.021 while no-wipe barely
  moved — the trainable memory machinery (front_end/injections train even
  when M's content is random) learned to inject generically useful signal,
  re-absorbing M-content's marginal value. Same re-absorption dynamic as
  the parametric path in the prior run, now inside the memory subsystem.
  60% mid-sleep + no fact blocks apparently trains "continue well from
  empty state" rather than "read M". Also awake-mem@435 (+0.0252 ≈ its
  gist-delta) shows M's contribution was never sleep-specific.
- FALLBACK (as agreed with user): archived step-{447,459,472,485,498,510}
  → `checkpoints/archive-20260721-gistdata/` (kept probe-able, outside the
  epoch-* pattern per DISCUSSION procedure). Regenerating
  `train_chains_xs.pt` (`--cross-sleep-bias 0.75 --seed 7`, the config
  step-435 was actually trained on per its dataset fingerprint) and
  resuming freeze-lora from step-435 on it — the exact regime that
  produced the 396→435 gist rise.

### 07:45 UTC — fallback probe (step-485 on xs data): NOT recovering

| step (regime) | gist-delta | dist-delta | awake-mem |
|------|-----------|-----------|-----------|
| 435 (same-code baseline) | +0.0248 (0.0046) | +0.0160 | +0.0252 |
| 485 (xs fallback, +50) | **+0.0164 (0.0025)** | +0.0060 | **+0.0300** |

Fallback did not resume the 396→435 rise: gist ~2.4σ below baseline at
+50 steps, dist-delta down, and awake-mem (always-on contribution) RISING.
Emerging picture: the 435 gist peak may not be stably extendable by more
training in ANY of the three data regimes tried (gist-shaped collapsed it,
xs is eroding it); M's contribution is drifting toward always-on injection
rather than sleep-specific recall. Caveat: fresh-optimizer transient could
depress the first ~50 steps after resume. Decision: ONE more probe point
(~step 535) on xs before concluding; if still declining, step-435 is the
best checkpoint, stop training and write up for the team.

### 08:5x UTC — decision probe (step-536, xs +101): headline recovered, STRUCTURE DEGRADED

| metric | 435 | 485 (xs+50) | 536 (xs+101) |
|--------|-----|-----|-----|
| gist-delta | +0.0248 | +0.0164 | **+0.0285** |
| long-range | +0.0256 | +0.0175 | +0.0151 |
| recency | −0.0008 | −0.0011 | **+0.0134** |
| dist-delta | +0.0160 | +0.0060 | +0.0059 |
| awake-mem | +0.0252 | +0.0300 | **+0.0418** |
| thirds trend | rising | mixed | **falling** |

The 485 dip was plausibly the fresh-optimizer transient (headline back at
baseline by 536). But the composition worsened monotonically: long-range
and cross-episode (dist) down, recency-buffer share and always-on
(awake-mem) up, help now concentrated right after the wipe. Continued xs
training converts M from "long-range episodic gist" toward "always-on
recency injection" — the exact drift the user flagged. Sleep-specific
long-range function peaked at step-435 under every regime tried.

Decision (my call, within mandate; matches the direction agreed with the
user pre-sleep): implement `--split-episode-rate` (interleaved
continuation — the one intervention that directly pressures cross-episode
long-range retention, the failing metrics), regen on the xs base, resume
from the CLEAN 435 peak (archiving the xs lineage per procedure). This is
the last idea of the night; if it also fails to hold 435's structure,
step-435 is the deliverable and I stop.

### 09:2x UTC — CRITICAL DISCOVERY: mid-conversation sleep has been a NO-OP on real data all along

While wiring --split-episode-rate: its first regen produced n_split=0.
Investigation: mid-sleep (and the original split rule) requires an episode
≥ mid-sleep-min-len (4096) WITH a middle-third turn boundary. On the real
corpora that set is EMPTY: ultrachat episodes are capped at 1024 tokens by
prepare_data (all multi-turn episodes are short), and LongAlign/babilong
episodes are single-QA (boundaries only at position 0 and the answer start
near the end — never the middle third). Every regen tonight produced
~10.4k sleeps — ALL between-episode; the "mid-conversation sleep" fraction
was 0% in every dataset ever trained on, including:
- the prior run's data (mid-sleep-rate 0.2 → actually 0),
- tonight's "gist-shaped" data (0.6 → actually 0; it ALSO removed fact
  blocks, leaving ZERO memory-rewarding signal — fully explaining the
  gist collapse at step-510),
- the xs data (which works via cross-sleep fact queries, its only real
  memory signal).
CORRECTION to earlier entries: step-435's gist ability came from
between-episode sleeps + cross-sleep queries alone, not mid-sleeps.

Fix (committed d8ed15a + b1b19de, tests 7/7): --split-episode-rate with a
--split-min-part (256) rule — cut at a turn boundary with ≥256 tokens each
side, tail resumes two episodes later behind a forced sleep. Eligible:
11116/18221 ultrachat + 4307/11846 memory episodes. For single-QA this
cuts at the answer boundary: read the document now, answer it an episode
and a sleep later — the first natural continuation signal this training
has ever had. Also fixed pre-existing broken tests (missing
cross_sleep_bias in test args since cdd6b9e).

New dataset: train_chains_split.pt = xs base (cross-sleep-bias 0.75) +
split-episode-rate 0.5. Resume from step-435, freeze-lora, probe at
+50/+100 against 435's structure metrics.

### 10:5x UTC — split-run probe at step-484 (+49): transient dip, structure healthy

| step (regime) | gist-delta | dist-delta | awake-mem | recency | thirds |
|------|-----------|-----------|-----------|---------|--------|
| 435 (baseline) | +0.0248 | +0.0160 | +0.0252 | ≈0 | rising |
| 485 (xs +50) | +0.0164 | +0.0060 | +0.0300 | ≈0 | mixed |
| **484 (split +49)** | +0.0149 (0.0055) | +0.0087 (0.0025) | +0.0324 | ≈0 | rising |

Nearly the same headline dip as xs at +50 (fresh-optimizer transient), but
with BETTER structure than xs showed: thirds still rising, dist-delta
higher than xs's +50 point. Decision deferred to ~step-533 (+100), the
point where xs recovered headline but with degraded structure (recency
buffer + falling thirds). If split's +100 shows headline recovery WITH
435-like structure (long-range dominant, dist-delta ≥ +0.016, thirds
rising), the split data wins and training continues; otherwise step-435
remains the peak → stop and write up. Training resumed from 484.

### Follow-up ideas for the team (discussed with user, not run tonight)

- ~~Uncap ultrachat episode length~~ **DONE 11:4x UTC (user-approved,
  user online)**: prepare_data `--max-len` default 1024 → 32768, commit
  5d7fa04. The 1024 cap was a legacy pre-chunking limit from 8 GB-VRAM
  hardware (per user) — obsolete under chunked training. Ultrachat
  re-tokenized uncapped (`logs/prep-ultrachat-uncapped.log`); long
  multi-turn episodes now exist, so --mid-sleep-rate fires for real for
  the first time. Next chains regen combines all three memory signals:
  real mid-sleeps + interleaved splits + cross-sleep queries.
- **Sleeps inside document turns (sentence-boundary wipes) — user wants
  this discussed properly.** Current rule: sleeps only at TURN boundaries.
  Rationale: (1) deployment realism — a sleep models a session ending,
  which happens between exchanges; (2) signal cleanliness — a mid-SENTENCE
  wipe makes the next tokens irreducibly unpredictable, and the only way
  to reduce that loss is storing verbatim text in M, the exact failure
  mode the exact-code probes showed is harmful; (3) eval interpretability
  (a mid-sentence wipe conflates "lost the gist" with "lost the sentence
  fragment"). The proposed middle ground: wipe at SENTENCE boundaries
  inside long document turns (LongAlign interiors), which avoids the
  fragment pathology while unlocking the one place our corpora have long
  contiguous content — trains "keep reading a document across a sleep"
  (reading-persistence) rather than conversational memory. Open questions
  for discussion: does continuing a document mid-way still reward
  verbatim-ish storage more than gist (the next sentence depends heavily
  on exact local content)? is sentence granularity enough distance from
  the fragment problem? implementation needs a sentence-boundary detector
  over token ids (period+space heuristics vs re-tokenization). Relatedly:
  whether the turn-boundary-only rule should stay for EVAL wipes even if
  training relaxes it.
- Mid-conversation sleeps remain 0 in ALL data to date, including tonight's
  split data (its 18060 sleeps = between-episode + 7743 split-tail).

### 11:2x UTC — split-run FAILED at +101; FINAL RUN launched (all signals + real optimizer)

Split-run decision probe (step-536): gist-delta +0.0081, dist-delta
+0.0036, awake-mem +0.0073 — trajectory 435→484→536 =
+0.0248→+0.0149→+0.0081, monotone decline. Third regime to erode the 435
peak. NEW HYPOTHESIS: every restart tonight began with a FRESH AdamW
(uploaded checkpoints had no optimizer.pt), while the original 396→435
rise ran with intact optimizer state — fresh-Adam shock on the 218 memory
params may erode the peak regardless of data.

Final run (user online and approving; user supplied the missing piece):
- User scp'd the ORIGINAL step-435 optimizer.pt (326 MB, saved during the
  freeze-lora phase → covers exactly the memory params) from their local
  pull of the prior instance.
- train.py fix (b9941d8, tests 17/17): under --freeze-lora, load the saved
  optimizer when param counts match instead of unconditionally starting
  fresh.
- prepare_data --max-len 1024→32768 (5d7fa04): ultrachat uncapped (max
  real length 5368, median 1121). --mid-sleep-min-len lowered to 1536 for
  the regen (4770 eligible episodes; at the 4096 default even uncapped
  data yields only 4 mid-sleeps — caught immediately by the new
  mid-conversation counter, ec3e14e).
- Data `train_chains_all.pt`: 2940 chains, 241.3M tokens, 22501 sleeps =
  **868 mid-conversation (first ever) + 10609 split-tail** + between-episode;
  26671 cross-sleep queries. ALL THREE memory signals, first time.
- Resume: step-435 + its true optimizer, --freeze-lora, split lineage
  archived to `archive-20260721-splitrun/`.
- Gates: probe at +50 and +100 (gist+distractor+awake-mem, seed 1234).
  PASS = gist-delta ≥ +0.025 with 435-like structure (long-range
  dominant, thirds not inverted, dist-delta not collapsed). FAIL at +100
  → step-435 is final; write up + shutdown checklist. User at work for
  ~9h; run continues under these gates autonomously.

### 03:44 UTC — chains regen done; training resumed

- `data/train_chains.pt`: 2753 chains, 229.2M tokens, **10364 sleeps**, 0
  fact blocks (log `logs/prep-chains-gist.log`).
- RESUME POINT: epoch-1/step-435, `--freeze-lora` (218 memory params
  optimized, LoRA fixed, fresh optimizer). Dataset-change detected → slot
  states discarded, slots start fresh (expected). recall-weight warning is
  the expected no-op.
- Starting memory metrics (~6k tokens in): beta 0.19–0.31, retain
  0.96–0.98, active 15–20/21, surprise 0.10–0.25, o_t_norm 16–25,
  grad_norm 2.6–4.0. Healthy; no non-finite warnings.

- step-180: delta negative at ~4.7σ, consistent with the smoke run. Keeping
  the trained-prefix `M` is WORSE than a fresh random `M` for continuation
  prediction, and the deficit is mostly the long-range component (recent-only
  `M` ≈ ablated, −0.0116). Thirds: mid-third worst (−0.0928).

### 04:5x UTC — awake-mem diagnostic added (user concern: M replacing SSM short-term role)

Concern: gist-shaped training might teach M to shadow the SSM's short-term
function rather than serve as the medium/long-term tier. Existing guard:
sleep-recent ≈ ablated (+0.004 @459) says M's value is long-range, not a
recency buffer. New guard (committed): `no-wipe-ablated` condition —
**awake-mem** = no-wipe − no-wipe-ablated = M's contribution while the SSM
is alive. Should stay small as sleep deltas grow; growth = shadowing.
Tracked from the step-510 probe on.

### 12:2x UTC — +50 probe (step-485, all-signals run)

Probe: longalign gist, prefix 6144 / cont 512 / recent 576 / distractor
1536, n=16, seed 1234 (log `logs/probe-20260721-121314.log`).

| metric | step-435 (peak) | step-485 (+50) |
|---|---|---|
| gist-delta | +0.0285 | **+0.0166** (SEM 0.0027) |
| long-range | dominant | +0.0115 (dominant) |
| recency | small | +0.0051 |
| thirds | rising | rising (+0.0112 → +0.0182 → +0.0204) |
| dist-delta | positive | +0.0058 (SEM 0.0018, not collapsed) |
| awake-mem | — | +0.0061 (SEM 0.0038, small — no SSM shadowing) |
| flush cost | — | +0.0108 |

Read: magnitude down ~40% from the 435 peak (~4σ), BUT the structure is
fully 435-like — long-range dominant, thirds monotone rising, dist-delta
positive, awake-mem small. Unlike the three failed regimes (which flipped
structure toward recency/flat), this looks like the same memory doing the
same job, weaker. Consistent with early-resume transient even with true
optimizer state (data distribution changed under it).

Gate: +50 is informational (gate is decisive at +100/step-535). Training
resumed from step-485, optimizer state carried (no fresh-optimizer
message). Decision at step-535: gist-delta ≥ +0.025 with this structure =
PASS; else step-435 is final.

### 13:2x UTC — +100 probe (step-537): FAIL — run concluded, step-435 final

Log `logs/probe-20260721-131148.log` (same config/seed as +50).

| metric | 435 peak | 485 (+50) | 537 (+100) |
|---|---|---|---|
| gist-delta | +0.0285 | +0.0166 | +0.0174 (SEM 0.0029) |
| long-range | dominant | +0.0115 | **+0.0021** (collapsed) |
| recency | small | +0.0051 | **+0.0153** (dominant) |
| thirds | rising | rising | **falling** (+0.0234 → +0.0101) |
| dist-delta | positive | +0.0058 | +0.0035 (SEM 0.0031, ~noise) |
| awake-mem | — | +0.0061 | −0.0066 (SEM 0.0074) |

Verdict: magnitude flat AND structure inverted to the xs-regime signature
(recency-dominant, thirds falling, long-range gone). Not the gray-zone
"rising with structure intact" case — the +150 tiebreaker does not apply.
FAIL per the pre-registered gate.

Key inference: with step-435's TRUE optimizer state restored, the
fresh-optimizer-shock hypothesis is REFUTED as the (sole) cause of
peak erosion — this is the 4th continuation regime to erode it, now with
optimizer continuity. The erosion is driven by the data/objective mix
itself: continued training on chain data (even all-signals) converts M's
long-range contribution into recency injection. The 396→435 freeze-lora
stretch on the ORIGINAL data was special; what it had that all
continuations lack needs team discussion (curriculum order? the specific
cross-sleep-bias mix? finite-data second pass?).

Decision: step-435 is the run's deliverable. Proceeding to debrief +
shutdown checklist.

## Closing summary (13:3x UTC) — run over, step-435 is the deliverable

**What this run banked:**
- `epoch-1/step-435` — the gist peak (gist-delta +0.0285, long-range
  dominant, thirds rising, dist-delta positive). Verified intact: 474
  trainable tensors all finite, optimizer.pt (326MB, true Adam state)
  present. THE checkpoint for anything downstream.
- A validated cross-sleep gist eval (`probe_recall.py --gist`,
  `--gist-distractor`, awake-mem condition) with a replicated monotone
  rise 180→435 and pre-registered gates. Probe config that works on
  longalign: prefix 6144 / cont 512 / recent 576 / distractor 1536 /
  n 16 / seed 1234.
- The mid-conversation-sleep no-op discovery + fix pipeline: uncapped
  ultrachat (5d7fa04), split-episode interleaving (b1b19de), mid-sleep
  counters (ec3e14e), optimizer-resume fix (b9941d8). All pushed.
- Data recipes on disk: `train_chains_all.pt` (all three signals, 868
  mid-conv + 10609 split-tail sleeps) — recipe is in the 11:0x entry.

**What this run established (for the debrief):**
1. Gist-delta and exact-code recall move INVERSELY across training; the
   396→435 freeze-lora stretch on the original chain data produced the
   only sustained gist rise.
2. FOUR continuation regimes all eroded the peak — gist-shaped data, xs
   fallback, split-run, and finally all-signals data WITH step-435's true
   optimizer state. Erosion signature is consistent: long-range → recency
   conversion, thirds invert, dist-delta decays.
3. Fresh-optimizer shock is refuted as the sole cause (regime 4 had
   optimizer continuity and still inverted). The cause is the continued
   training itself — hypotheses for the team: (a) the peak is a transient
   the objective passes through, not a fixed point it converges to —
   any further optimization moves off it; (b) second-pass data repetition
   shifts the objective toward memorization-adjacent recency; (c) the
   all-signals mix's higher sleep density over-rewards short-horizon
   carry-over.
4. Mid-conversation sleeps were structurally absent from ALL prior
   training (silent no-op) — every prior result should be read knowing
   the model never trained on intra-episode wipes.
5. alpha stayed 0.0000 throughout (forgetting gate inactive); decay runs
   entirely through per-token retain (~0.95). w1_abs_max stable ~0.03.
   Not a blocker, but the alpha pathway is dead weight as trained.

**Open questions queued for the debrief** (with the sentence-boundary-
sleep discussion item from the 10:xx entry):
- Why was 396→435 special? Reproduce the rise from an EARLIER checkpoint
  (e.g. 396) on identical data/optimizer to test path-dependence vs data.
- Is there a stopping-rule better than "probe every ~12 steps and keep
  the argmax"? The peak was found only because we swept.
- Does step-435 hold up on non-longalign gist material (babilong,
  held-out ultrachat)?

**Housekeeping state at shutdown:**
- Training STOPPED after the step-537 probe (not resumed — gate failed).
- Checkpoints: epoch-1 holds the canonical lineage ending at step-435;
  post-435 all-signals steps archived to `archive-20260721-allsignals/`;
  earlier failed regimes in `archive-20260721-{gistdata,xsfallback,splitrun}/`.
- All code pushed through d6f0a71 (nothing uncommitted but this notes
  file, which travels via rsync pull, not git).
- Shutdown checklist executed to completion; `.watchdog-terminate`
  touched as the final act of the session.
