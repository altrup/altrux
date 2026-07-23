# Discussion notes — 2026-07-22, post-run debrief: reproducibility settled, stage 2 begins

Team discussion (altrup + Claude) after `EXPERIMENT_NOTES-20260721-232250.md`.
STANDING DIRECTION for the next experimenter run — self-contained; read
alongside `DISCUSSION-20260721-peak-reproducibility.md` (its north-star
section and rejected items still stand) and `DISCUSSION-20260720-gist-eval.md`
(ditto).

## Reinterpretation of the run's conclusions

1. **The recipe is real and reproducible (Test 1 PASS).** The 396→435 rise
   reproduced on exactly-regenerated xs data through cold slots, overshooting
   the original: peak +0.0344 at step-447 (+51) vs original 435's +0.0285.
   The four prior erosion regimes are explained: dataset-change-at-resume
   was the killer. Practical consequence: **we now have a validated A/B
   harness** — resume `epoch-1/step-396` on regenerated xs data
   (`train_chains.pt`), train ~+51 steps (~1 h box time), probe LongAlign
   gist (prefix 6144 / cont 512 / recent 576 / distractor 1536 / n 16 /
   seed 1234). Baseline to beat: **+0.0344**. Every stage-2 change gets
   measured on this harness, one knob at a time.
2. **Peaks are optimization transients — mechanism (a) confirmed.** Erosion
   occurred with no discontinuity whatsoever (same data, same stream, warm
   slots): +0.0344 → 0.0206 → 0.0150 → 0.0104, classic long-range→recency
   conversion. "Harvest the argmax, never train through it" is now settled
   policy, not a hunch.
3. **Generalization holds directionally (Test 2 PASS).** Held-out ultrachat
   test_sft: 435 +0.0120 vs 180 −0.0364. But the ultrachat-2048 ruler is a
   weak instrument (459 eligible convs, prefix capped at 2048 by corpus
   length, headline recency-flavored by construction). babilong is
   structurally ineligible for the gist harness (verified zero eligibility
   down to cont 64).
4. **Split data changes the shape, not the height (Test 3).** Same peak
   location (~+51), lower peak (+0.0304 vs +0.0344), best-of-night mid-climb
   durability (dist-delta +0.0296 at +38), and — after the log-misattribution
   correction — an erosion tail that plateaus (~+0.019/0.021/0.022 at
   +64/+72/+80) instead of collapsing.
5. **Plateau vs slow erosion is UNRESOLVED — the run's open question.**
   Points against reading it as solved transience: (a) at the matched offset
   +64 the two runs are equal (T1 +0.0206, T3 +0.0187) and T3 was simply
   stopped earlier (+80) than T1 was run (+90); (b) T3's tail shows the
   erosion signature in progress (recency +0.0092 and climbing, dist-delta
   fallen +0.0296→+0.0110); (c) even a real plateau sits ~30% below T3's own
   peak. Best case Test 3 bought a softer landing, not a cure.
6. **The peak-height ceiling (~+0.030–0.034 across two quite different data
   recipes) says the magnitude bottleneck is not data composition.** Against
   the north star (wipe cost ~0.19–0.22, best recovery ~15%), the ceiling is
   the problem. Stage 2 — the machinery/objective — is where the program
   goes now, per the north-star section's suspect list (read-out bandwidth
   first).

## Deliverable: 447-T3 (agreed)

Held-out ultrachat comparison: 435-orig +0.0120 | 447-T1 +0.0024 |
447-T3 +0.0078 | 434-T3 +0.0047. 447-T1 — the best LongAlign peak —
transfers worst (LongAlign was in training data; its argmax partly
corpus-fits). 435's held-out headline win is mostly its recency term; on
held-out long-range 447-T3 wins (+0.0086 vs +0.0038) with a balanced
profile from the recipe with the durability advantages.

- **`archive-20260722-test3-lineage/step-447` is THE deliverable.**
- 435-orig (`archive-20260722-test1/step-435`) stays banked as the
  held-out-headline reference. 447-T1 stays banked. Nothing is deleted.

## Stage-2 architecture direction (agreed framing)

End state we're steering toward: remove the fixed memory window; the model
chooses when to read/inject (and eventually when to flush writes), with
reads gated on the memory's own output. Key code facts established during
this discussion (all verified in `models/mamba2_2_7b_memory/model.py`):

- **Reads already happen at EVERY token** (`_forward_manual`'s READ_LAYER
  block; `read_windowed` on the fused path): observe + M.read + surprise
  run per-token unconditionally. What is window-gated is (a) the WRITE (the
  window's tokens batched into one gradient step, Titans chunk-b form) and
  (b) the INJECTION EVENT, which consumes a surprise-weighted pool of the
  window's per-token reads — i.e. soft within-window read selection already
  exists. The novelty of the stage-2 gate is therefore sparse *injection
  timing*, not per-token reading. (Corrected during this discussion; an
  earlier draft claimed reads were per-window.)
- Injections touch layers 22, 24, … 62 (`INJECTED_LAYERS`), including
  layers below READ_LAYER=42 (they receive the read one token late). The
  only injection-independent stream is **layers 0–21**.
- The fused path (`_mixer_span`) already handles arbitrary injection
  boundaries: fused kernels between injection points, manual `_mixer_step`
  only at them. Sparse trigger-chosen fire points are compatible with the
  causal-conv1d/chunked-scan route **by construction**, provided fire
  points are computable from the injection-free lower stack (layers ≤21):
  run layers 0–21 fused over the chunk, decide fire points, then run
  layers 22+ with span dispatch. No rewinding.
- q/k/v are RMS-normalized to unit scale before reaching M, but
  **o_t = M(q) is NOT normalized** — since q is unit-rms, ‖o_t‖ is purely
  the accumulated content of M along the query direction, i.e. "how much
  memory has to say here". Legitimate gate signal, not a hack.
- alpha (M's decay) is dead as trained (≈0) — that is M declining to
  forget, NOT a retention problem; targeted forgetting is the
  delta-overwrite path. Do NOT raise ALPHA_CAP (a real run walked alpha up
  20× and wiped 80% of M — 2026-07-17 notes; its gradient is structurally
  short-horizon and will abuse any headroom). The ~0.976 retain is a
  different gate (mamba ssm_state decay at injection) and is healthy.
  Decay is not the bottleneck; both stay untouched.

**The ordering principle (settled late in the discussion): test the
densification ceiling with dumb fixed cadences BEFORE building anything
trainable.** A learned sparse gate has a real training problem — straight-
through gives zero gradient to non-fired tokens, so "never fire" is
self-reinforcing; the fixes (dense would-be injections in training, or
stochastic firing) are costly or high-variance. But the question the gate
would answer — does injecting/writing more often raise the peak? — is
testable with NO new machinery: `--memory-window` is an existing flag, and
window 4/1 legs are plain resumable harness A/Bs. If maximal densification
(window 1) doesn't lift the peak, no timing gate ever will and the whole
gate line is moot. The gate is an EFFICIENCY device to be built only after
densification proves there is a lift to recover sparsely.

Also relevant: window 1 is the prod-serving target anyway (a window-8-
trained model served at window 1 is a train/serve mismatch — beta gates
trained on 8-token surprise-pooled reads would see raw per-token ones), so
the window-1 leg doubles as the prod-config test.

Groundwork banked for the eventual gate (from L1 + this discussion):

- Layer-21 recoverability: GO (see L1 results below) — front-end could move
  below the injection floor, making reads/gate precomputable and keeping
  the fused path with trigger-chosen boundaries.
- Gate = small learned scorer trained straight-through against LM loss,
  inputs o_t (and optionally surprise — already consumed by today's
  injection gate; no claim either is individually predictive, and L1 shows
  raw ‖o_t‖ magnitude is confounded by domain). Budget ≤1 fire per stretch
  + window-close fallback, so always/never is handled structurally.

Deferred within stage 2 (only after the densification answer):
the learned gate + front-end-at-21 move; dynamic write-flush
(accumulate-since-last-fire); widening the 128-dim injection bottleneck
(shape-changing → can't resume 396, 10–50× costlier to test).

**Two ceiling suspects added this discussion (see
`RESEARCH-20260722-memory-consolidation-landscape.md` for the literature):**

- **Integration point.** Our SSM-state injection at layers ≥22 is the
  MAL-analog (Titans' weakest variant; MAC/MAG beat it). Lever: inject the
  read LOWER (MAC-analog for a mamba stack — more layers compute over it), or
  a MAG-style separate long-term branch + learned gate instead of fusing M
  into the SSM state that's already a short-term associative memory.
- **Compute-depth over the read**, distinct from bandwidth-WIDTH: the
  ceiling may be too-shallow computation over the read (one gated lookup),
  not too-narrow a channel. Fix = more passes / inject-lower / longer sleep,
  not widening. Test the depth story before spending on width.

**Reframed near-term objective for M** (supersedes "beat +0.0344 gist-delta"
as the north star, though the harness metric stays): M's job is to be a
faithful, durable STORE (hold gist AND specific facts, across sleeps), NOT to
do single-shot read-out reasoning — that's a compute-depth problem for the
architecture/inference, not something to train M harder for. If sleep-time
weight-consolidation (research note) carries the durable load, M's job
shrinks further to "generate a faithful dream." The gist-vs-verbatim
competition (facts collapse as gist rises) is the sharpest near-term target
and appears novel.

## Next steps, prioritized

### Local (this box, before the next rental)

**L1. Per-token read diagnostic on 447-T3 — RUN, results in.**
`sft/read_diagnostic.py` (`make read-diagnostic`), 4 × 4096 LongAlign
tokens, log `sft/logs/read-diag-20260722-*.log`, ~35 min wall on the 8 GB
box (`--chunk-len 8` mandatory; 48 OOMs):

- **Gate signal: raw ‖o_t‖ magnitude REJECTED as a threshold.** CV 0.15
  over sequences but perfectly flat by window position (18.06–18.11), and
  the top-norm tokens all cluster in LongAlign's Chinese-document regions —
  the norm tracks domain/content shift, not retrieval-worthy moments.
  Surprise has 3× the relative variance (CV 0.48) and is uncorrelated with
  ‖o_t‖ (−0.07). Consequence: any future gate is a learned scorer, not a
  norm threshold.
- **Layer-21 recoverability: GO (cautious).** Ridge res21→res42, 13.1k
  train / 3.3k held-out tokens: residual R² 0.59; mean-centered cosine in
  q/k/v space (the front-end-relevant metric) medians 0.80–0.83 vs
  shuffled control ≈ 0.00. ~80% of the per-token directional content the
  front-end consumes at 42 is linearly available at 21. Permission, not
  proof — the harness A/B remains the real test if/when the front-end
  moves.

**L2. M readout / dream-fidelity probe on 447-T3** — `sft/dream_fidelity.py`
(`make dream-fidelity`), BUILT. Tests the precondition for the M→weights
consolidation sketch (research note): *can M generate a faithful dream of what
it stored?* Generation-only (no teacher-forced scoring). **Runs on the BOX,
not locally** — priming the 2.7B + the memory write's transient working set
needs ~17 GB (same wall as probe_recall); confirmed constant-memory (no leak —
`state.detach()` cuts the graph each step, M is fixed-size), the 8 GB card is
just ~100 MB short at the write's fp32 temps. Box defaults already match the
real run: `make dream-fidelity ARGS="--checkpoint <447-T3 dir>"` (n-probes 4,
prime 2048, gen 128, temp 0.8; priming uses the trained window + fused path).
The overlap/sampling logic is unit-tested locally. Design —
the naive "prime, wipe SSM, read M's output" is dominated by generic backbone
priors, so **contrast, don't inspect**:

- Prime a real text (with specific facts) through 447-T3; wipe SSM
  (`sleep_slot`), keep M.
- (a) **Scoring** (reuse probe_recall gist machinery): teacher-force the
  primed content, per-token logprob M-intact vs M-ablated → did M store it.
- (b) **Generation = the dream test** (NEW — we have never generated from M,
  only scored): sample under M-primed vs M-random, same wiped SSM; measure
  topic/entity/fact overlap with the priming text.

Prediction (from gist-vs-verbatim): topically steered, factually generic
(gist-faithful, fact-lossy). If so, the consolidation loop needs the
transcript in the loop → collapses toward the transcript-replay baseline.
Decision-relevant for whether the M→weights sketch is worth building.

### Box (next rental, in order)

**B1. T3 plateau-vs-slow-erosion test** (cheap, independent, first):
resume `archive-20260722-test3-lineage/step-476` on unchanged
`train_chains_split.pt` (restore lineage dirs to epoch-1 or point resume at
the archive per the established archive procedure), same verbatim resume
args as Test 3, train to total offset ~+120–130, probe every ~13 steps
(LongAlign config above). Decision rule:

- gist-delta holds ≥ ~+0.015 with long-range positive through 3 more
  sweeps → PLATEAU REAL: split data is erosion-resistant and becomes the
  base recipe for subsequent stage-2 harness runs.
- slides below ~+0.01 or long-range goes negative → SLOW EROSION: transience
  is unsolved by data composition; strengthens stage-2-first; xs data stays
  the harness recipe (faster to its peak).

**B2a. Window-4 densification A/B** (~2 h): resume
`epoch-1/step-396`, xs data, baseline args but `--memory-window 4`
(existing flag, zero code): verbatim
`make resume ARGS="--data data/train_chains.pt --eos-weight 32
--batch-size 8 --chunk-len 48 --memory-window 4 --accum-tokens 1536
--ckpt-every-tokens 49152 --freeze-lora"`.
Train ~+51, probe (LongAlign config) at ~+12/+24/+51 vs the window-8
trajectory (+0.0144/+0.0186/+0.0344). Fused path retained (spans of 4);
~2× step time expected. This is the middle point of the dose-response
curve 8 → 4 → 1.

Reading B2a against the window-8 baseline (+0.0144/+0.0186/+0.0344 at
+12/+24/+51):

- Clear lift → densification works; the window-1 leg (below, deferred)
  gets funded next run, and B3 attribution follows.
- Flat/mixed → densification looks unpromising; next suspect is write
  fidelity via multi-step writes (or bandwidth width, the shape-changing
  tier). The window-1 leg stays deferred.
- Clearly worse → window-8 pooling is load-bearing (surprise-weighted
  read selection matters); record and move on.

**DEFERRED this run (team decision): the window-1 leg** — the maximal
densification and prod-config test (`--memory-window 1`, same command).
Fully manual path; expect ~6–8× step time (2026-07-20 notes: GPU util 23%
at window 8, batch parallelism NOT the bottleneck — sequential per-window
ops dominate, and window 1 multiplies exactly those by 8), i.e. an
overnight ~6–8 h leg. When it does run: **early-abort rule — probe at
~+12 (~1.5–2 h in); negative gist-delta or recency-shaped structure →
abort, bank the answer.** Also note the standing prod question it would
answer: serving at window 1 with a window-8-trained model is a
train/serve mismatch (beta gates trained on 8-token surprise-pooled
reads would see raw per-token ones).

**B3 (conditional on a B2a lift, likely next run). Cadence decoupling:**
small code change separating injection cadence from write cadence (e.g.
write every 8, inject every 1, pooling reads since last injection),
harness A/B. Attributes a densification lift to reads vs writes. Only
after this — and only if injection timing specifically is what pays —
does the learned sparse gate (+ front-end-at-21) get built, using the
banked L1 groundwork.

Also on any box run where it's cheap: probe the winning candidate on the
held-out ultrachat config for the generalization record (and see open
question on a better ruler below).

## Open questions (carried, not resolved)

- **Better held-out ruler.** ultrachat-2048 is weak (459 convs, prefix
  capped, recency-flavored). Want a long-document corpus never in training
  with eligibility at prefix ≥4096. Candidate work item for a local
  session; not blocking B1–B3.
- **Gate trainability (if B3 ever selects it).** Beyond the short-horizon
  issue: straight-through gives zero gradient to non-fired tokens, so
  "never fire" is self-reinforcing; the candidate fixes (dense would-be
  injections during training — kills the fused path — or stochastic firing
  with ST through samples) are costly or high-variance. Unsolved; one
  reason the densification-first ordering was chosen.
- **Layer-21 semantic adequacy** — L1's recoverability is permission, not
  proof; the real test is a harness A/B if the front-end ever moves.
- **Whether surprise belongs in a gate at all** — it's the WRITE signal
  (‖M(k)−v‖², "should I learn this"), though today's injection gate and
  pooling already consume it. High surprise can mean "context shift,
  memory needed" or "novel content, memory has nothing". Untested either
  way; a learned scorer can zero it out.

## Explicitly considered and rejected (this discussion)

- **Raising ALPHA_CAP to help retention** — backwards: alpha≈0 means M
  isn't forgetting; headroom gets abused into erosion (2026-07-17 evidence);
  decay isn't the bottleneck.
- **Soft per-token gate + λ·firing-rate penalty** — needs injection
  computed every token in training (kills the fused path) and λ is exactly
  the always/never knife-edge. Budget-by-construction instead.
- **Top-k fire-point selection within a window/chunk** — non-causal at
  streaming inference; first-crossing threshold instead.
- **Trigger/gate computed from any layer ≥22** — creates a
  decide-then-affects-your-own-input cycle; requires speculative
  run-and-rewind, breaking the fused path. Everything gate-related reads
  layers ≤21 only.
- **Raw ‖o_t‖-norm threshold as the gate** — L1 measured it: flat by
  window position, peaks track document domain (Chinese-text regions), not
  retrieval moments. Any gate is a learned scorer.
- **Building the learned gate before the densification ceiling test** —
  the gate has an unsolved training problem (see open questions) and its
  ceiling is measurable for free with fixed window flags; reversed
  ordering adopted instead (B2a/B2b before any gate work).
- **Front-end-at-21 move as a near-term A/B** — demoted off the critical
  path: fixed cadences need no precomputation, so the move only matters
  if/when the learned gate gets built. L1's GO result is banked for then.
- Carried from prior notes: fifth continuation regime from 435;
  sentence-boundary sleeps in these runs; replacing queries with natural
  continuation signals; warm-slot reconstruction; chunk-len 48→64; literal
  episode repetition.

## Housekeeping

(to be filled as items are agreed and committed)

- Banked the run: b676d2c (`EXPERIMENT_NOTES-20260721-232250.md`).
- `sft/read_diagnostic.py` + `make read-diagnostic` (tees to
  `sft/logs/read-diag-*.log`): the L1 diagnostic — per-token ||o_t|| /
  surprise stats, peak-token decode, and the layer-21→42 ridge
  recoverability check scored in q/k/v space (raw + mean-centered cosine;
  raw is inflated ~0.9 by the projections' shared mean direction). Local
  gotcha honored: `--chunk-len 8` on the 8 GB box (48 OOMs, same cap as
  smoke_test.py documents).
