# Discussion notes — 2026-07-23, post-run debrief: plateau banked, pivot to the 780M integration screen

Team discussion (altrup + Claude) after `EXPERIMENT_NOTES-20260723-023554.md`.
STANDING DIRECTION for the next experimenter run — self-contained; read
alongside `DISCUSSION-20260722-stage2-readout.md` (its north-star framing and
rejected items still stand except where superseded below) and
`RESEARCH-20260722-memory-consolidation-landscape.md` (the MAC/MAL and
compute-depth arguments this direction acts on).

## Reinterpretation of the run's conclusions

1. **B1: PLATEAU REAL — accepted.** Six probes +84→+161, wiped branches
   flat-to-rising (~+0.013–0.021), long-range positive at every clean point,
   recency ≈0/negative, dist-delta stable. Split data is erosion-resistant;
   **split recipe (`--split-episode-rate 0.15 --split-qa-rate 0.9
   --split-gap-min 1 --split-gap-max 4` on the cross-sleep-bias-0.75 pool) is
   the base recipe going forward.** Caveats: the shelf sits ~35–45% below the
   xs transient peak (+0.0344), and each resume knocks it down a notch
   (~+0.022 → ~+0.015 across the 518 restart) before re-stabilizing.
2. **The M-dependence anomaly is REREAD as (probably) an instrument
   artifact, not "deep integration achieved."** The ablation control is a
   FRESH RANDOM M (`model.py` NeuralMemory init), not "no memory": under
   `--freeze-lora` the injection machinery (beta gates, projections) keeps
   firing on content-free reads, and as trained injection magnitude grows
   (o_t_norm 13→27 over the leg) the ablation branch increasingly injects
   junk through gates tuned for signal. Supporting evidence: the collapse
   scales with unwiped horizon — held-out prefix-2048 awake-mem elevated but
   sane (+0.078) while LongAlign prefix-6144 collapsed (+3.07) at the same
   checkpoint. "Ablated collapses" may measure the CONTROL going OOD, not M
   being load-bearing. Decisive, cheap test: a **zero-injection ablation
   mode** (skip injection entirely). Until it runs, dependence-vs-achievement
   is UNDECIDED and 544/557's headline numbers are provisional.
3. **Held-out @557 is the best transfer result to date, but one headline was
   oversold.** gist-delta +0.0895 (SEM 0.0157, ~5.7σ) is real but carries
   the fresh-M confound above. The "long-range +0.0456, 5× prior best" claim
   has SEM 0.0287 (~1.6σ) and recency (+0.0439 ± 0.0258) is the same size —
   "long-range dominant" is NOT statistically supported at 557 held-out, and
   the thirds are front-loaded (+0.133 → +0.046, recency-flavored shape).
   Defensible claim: split-lineage late checkpoints transfer off-corpus far
   better than any harvested peak. **Deliverable selection (518 vs 557 vs
   held-out argmax) is deferred until the clean-instrument re-probe (F1).**
4. **B2a: densification scales the clock, not the ceiling — accepted.**
   w4 peak +0.0336@+41 ≈ w8 peak +0.0344@+51; transient compressed
   ~proportionally; batch-4 caveat doesn't threaten an invariance result.
   Window-1 leg and the learned sparse gate are MOOT for peak height. The
   cadence/timing branch of the suspect tree is CLOSED. Live ceiling
   suspects: **integration point, compute depth over the read, write
   fidelity / bandwidth width** — in that order per the research note.
5. **B0 dream: negative as measured, but the verdict is REREAD as possibly
   about the wipe, not about M.** Generation over a zeroed SSM is itself OOD
   — the backbone has never generated from that state; its priors collapse
   into low-entropy attractors and M's injections read as noise (the
   sanity-gen side probe showed intact-SSM generation is coherent). The
   free-dream verdict stands for the wiped-SSM design; a **neutral-snapshot
   rerun** (below) decides whether dreaming is dead or the operating point
   was wrong. Deployment-time dreaming would use periodic SSM snapshots
   (~170 MB each at 2.7B) anyway; for ATTRIBUTION the snapshot must be
   content-neutral or the dream can echo the snapshot instead of M.

## The pivot (agreed): 780M screening platform + integration-point A/B

B2a closed the cadence branch; the remaining suspects are structural
(shape-changing, 10–50× costlier to test at 2.7B). Team decision: **stand up
a 780M memory model as the screening platform and make the next box run an
A/B of the two integration designs.** Rationale:

- Mechanism-level, qualitative comparisons (integration point, unfreeze
  dynamics, erosion signatures) are what small models are for; the
  literature settles these at 170M–760M. Absolute deltas and recipe
  constants will NOT transfer — never compare magnitudes across scales.
- ~3–4× faster steps (48×1536 vs 64×2560), bigger batches, probes fit
  alongside training on a 96 GB card, and local smoke tests become possible
  on the 8 GB box. One rental buys 4–6 experiments instead of one.
- Not below 780M: 370M/130M backbones are weak enough that the
  continuation-prediction task changes character.

**The A/B: state-injection (current MAL-analog) vs input token-mix
(MAC-analog).** Current design injects M's read into the SSM state at
layers ≥22 — the Titans MAL analog, their weakest variant, sharing substrate
with mamba's own associative state (the awake-mem competition), with only
the top third of the stack computing over the read. The alternative:
**additive gated mix of o_t into the NEXT token's input embedding** —
the full stack computes over the retrieval, attacking the integration-point
and compute-depth suspects at once.

Token-mix spec (agreed; revised during this discussion from an
embedding-level mix to the layer-21/22 boundary):

- **Mix point = the layer-21/22 boundary, front-end at layer 21,
  same-token.** Token t's layers 0–21 run injection-free; q/k/v form at 21;
  o_t mixes (additive, gated, gate zero-init) into the same token's
  layer-22 input. NOT inserted memory tokens — sequence length preserved.
- Why not the embedding: mixing o_t into token t+1's embedding makes each
  token's forward depend on the previous token's mid-stack output — a
  serial dependency that kills the chunked/fused path for the WHOLE stack.
  Mix-at-21 keeps 0–21 fully fused, batches reads per window (M is fixed
  within a window), adds o_t to layer-22 inputs, and runs 22+ fused between
  write boundaries. It also removes the one-token delay.
- Cost accepted: layers 0–21 never compute over the read (~2/3-depth MAC,
  not full-depth). L1's result — ~80% of the front-end's directional
  content is linearly present at 21 — plus from-scratch training at 780M
  (front-end simply learns at 21; nothing transfers) make this the right
  trade.
- Loop alignment: this is the prelude(0–21) / recurrent-core(21→42) /
  coda(42+) structure of the internal-looping roadmap — the mix point sits
  at the loop entry, where per-iteration re-read wants to live later.
- No loops, no per-iteration re-read yet — single pass, one knob vs the
  state-injection arm.

**Internal-thinking roadmap decision (recorded, nothing built):** recurrent
depth over Coconut-style continuous latents. Coconut generates latents
sequentially even in training (kills teacher-forced parallelism), needs a
CoT-distillation curriculum we don't have, and is partially redundant with
mamba's own carried state. Recurrent depth keeps standard parallel training,
adds a test-time compute dial, and composes with token-mix: fresh query +
re-read of M at each loop boundary = multi-hop retrieval, i.e. the
compute-depth lever. This synergy is a further reason token-mix is the right
integration bet; state-injection has no clean per-iteration semantics under
loops.

## Next steps, prioritized

### Local (this box, before the next rental)

**L1. Build `models/mamba2_780m_memory`** — port the 2.7B memory machinery
(front-end, NeuralMemory, gated-delta injection, manual mixer + fused span
dispatch) parameterized to the 780M backbone (48 layers, d_model 1536;
scale READ_LAYER/INJECTED_LAYERS proportionally, document choices in its
README), with an integration knob selecting **state-injection vs
input-token-mix** (env/config read at model construction, so both A/B arms
run from one model dir). Unit tests for shape/causality of both paths;
smoke locally. This is the gating work item for the next rental.

**L2. `probe_recall.py --ablation {fresh-m,none}`** — `none` skips
injection entirely (not zero-M: a zeroed M still fires injection events and
their retain-decay on the SSM state). Default stays `fresh-m` so existing
numbers remain comparable; probes of record run BOTH.

**L3 (smaller). `dream_fidelity.py` neutral-snapshot mode** — prime the
generation SSM with a content-neutral snapshot (state from unrelated/
session-start text) instead of a zeroed state; keep the M-primed vs
M-random contrast, which still isolates M while generation stays
in-distribution.

### Box (next rental, in order)

**BX0. 780M calibration = the state-injection arm.** Train
`mamba2_780m_memory` (state-injection mode) from scratch on the split
recipe (same data .pt files — tokenizer is shared across the mamba2 family;
regen commands verbatim in EXPERIMENT_NOTES-20260723 02:40). Start from the
2.7B-proven args, scale batch up to the card — batch is near-free
throughput here (step time is dominated by sequential per-window ops that
amortize across rows; GPU util was ~23% at 2.7B batch 8), so expect batch
32+ at 780M. Fix batch/chunk/accum once at launch, document them, and use
the IDENTICAL values for BX1 so the A/B stays one knob; probe with the LongAlign gist
config (prefix 6144 / cont 512 / recent 576 / distractor 1536 / n 16 /
seed 1234) every few hundred M tokens, BOTH ablation modes.

- Decision rule: a clearly positive gist-delta with long-range-dominant
  structure emerges (any magnitude — do not compare to 2.7B numbers) →
  **SCREEN VALID**, proceed to BX1. Optional strengthener if time: an
  xs-data leg reproducing the erosion signature as contrast.
- No such signature after a solid training budget → **SCREEN DEAD**: 780M
  can't express the effect; bank that, revert the program to 2.7B and fund
  the deferred 2.7B items below instead.

**BX1. Token-mix arm.** Same recipe, same probes, integration knob flipped.
Read the A/B on trajectory shape and peak with long-range structure:

- Token-mix clearly better → integration point is (part of) the ceiling;
  next run ports token-mix to 2.7B (resume-compatible via zero-init gate)
  and re-runs the harness A/B there.
- ≈ Equal → integration point is not the binding constraint at this scale;
  next suspect is compute depth (multi-pass read) or write fidelity.
- Token-mix fails to train (gate stuck at 0, no delta) → diagnose gate
  dynamics before concluding anything; a dead gate is a bug, not a result.

**BX2 (the cross cells — the A/B is a 2×2, adaptively gated).** Full
design: {state, mix} × {read @16, @32}. BX0 = state@32, BX1 = mix@16 (the
max-contrast pair, always run in that order). The cross cells — state@16
and mix@32, both config-only (read_layer is a ctor param; mix boundary
configurable) — run ONLY if the first pair shows a difference worth
attributing or a mix failure worth diagnosing; if mix@16 ≈ state@32 the
headline is "integration point doesn't matter at this scale" and the cross
cells are optional. What the factorial buys: the known confound (arms
differ in mechanism AND read point) becomes two separately-measured
effects; state@16 vs state@32 isolates read depth ("shallow q/k/v
inadequate" vs "integration at fault"), and mix@32 tests whether the
bypass property alone helps even with shallow compute-over-the-read.
Same recipe/args for every cell. Related post-BX0 cheap diagnostic: port
`read_diagnostic.py` to MODEL_NAME resolution and run ridge res16→res32 in
the TRAINED state arm's q/k/v space (the 780M parallel of the 2.7B
layer-21 GO result; needs a trained front-end, so post-BX0 only). No
smaller-model shortcut: nothing below 780M in the family fits this, and
780M diagnostics run on the local 8 GB card anyway.

**F1 (filler, 2.7B, probe-only — runs alongside 780M training; a 780M train
job + 2.7B probe should coexist on 96 GB, verify before relying on it).**
Re-probe `archive-20260723-b1-extension/{step-518,step-544,step-557}` with
`--ablation none` AND `--ablation fresh-m`, LongAlign + held-out ultrachat
configs. This settles dependence-vs-achievement (#2 above) and picks the
lineage deliverable.

**F2 (filler, probe-only).** Held-out-first harvest: sweep the held-out
ultrachat config across the banked `archive-20260723-b1-extension`
checkpoints (~480→557). 557 was never SELECTED on held-out; the argmax may
be elsewhere.

### Deferred (agreed worth doing, not this run)

- **Unfreeze-LoRA A/B** — agreed worth trying; run as a controlled A/B,
  never a blanket recipe change: resume `archive-20260723-b1-extension/
  step-518` WITHOUT `--freeze-lora` vs the frozen extension as control,
  split data unchanged, ~+40–50 steps, clean instrument. Can be screened at
  780M post-BX0 instead if the screen validates. Motivations: cheapest
  probe of compute-over-the-read (frozen upper layers cannot learn to use
  the read), and the anomaly's one-sided co-adaptation. Risk: 21M params
  re-opened to the recency/corpus-fit shortcut. Note: gist rose 180→396
  with LoRA unfrozen, so the freeze is not established as load-bearing.
- Dream neutral-snapshot rerun at 2.7B (after L3).
- 2.7B split-lineage extension past +161 (was still rising on both rulers).
- Window-1 leg; purist B2a rerun at batch 8 / chunk 24 (only if the
  invariance result is ever doubted).

## Explicitly considered and rejected (this discussion)

- **Blanket-unfreezing LoRA as a recipe change** — changes the condition
  under which the plateau was just established; loses attribution. A/B only.
- **Reading the M-dependence collapse as "deep integration achieved"** —
  premature until the zero-injection control separates dependence from
  junk-sensitivity of the fresh-M ablation.
- **"Long-range 5× prior best" as the held-out headline** — SEM 0.0287 on
  +0.0456 (~1.6σ), recency term the same size; use the transfer claim, not
  the decomposition, until F1 re-measures.
- **Content-bearing SSM snapshots for dream ATTRIBUTION** — the dream can
  echo the snapshot instead of M; neutral snapshots only (content-bearing
  snapshots are fine for deployment-time dreaming, where attribution isn't
  the question).
- **Coconut-style continuous latents** (for this program, now) — sequential
  latent generation breaks teacher-forced parallel training, needs a
  CoT-distillation curriculum, partially redundant with mamba's carried
  state. Recurrent depth is the recorded roadmap choice.
- **Building loops / per-iteration re-read now** — roadmap synergy noted;
  near-term prototype stays single-pass so the A/B is one knob.
- **Screening below 780M (370M/130M)** — backbone too weak; the
  continuation-prediction task changes character.
- **Consolidation/transcript-replay baseline this run** — B0 (as measured)
  plus B2a both point at read-out/integration as the binding constraint;
  replay doesn't touch it. Revisit after the dream neutral-snapshot rerun.
- Carried from prior notes: raising ALPHA_CAP; soft per-token gate;
  top-k fire points; gate reading layers ≥22; norm-threshold gate;
  gate-before-densification (now moot entirely per B2a); fifth continuation
  regime from 435; sentence-boundary sleeps; replacing queries with natural
  signals; warm-slot reconstruction; chunk-len 48→64; literal episode
  repetition.

## Housekeeping

- Banked the run: 5a1f8bf (`EXPERIMENT_NOTES-20260723-023554.md`).
- **L2 DONE (3e78b3b):** `probe_recall.py --ablation {fresh-m,none}`.
  `none` = `Model.injection_enabled` kill switch — skips injections AND the
  front-end read/write machinery in both forward paths, so the ablated
  branch is exactly the plain backbone. `fresh-m` stays the default
  (historical comparability). Probes of record run BOTH modes.
- **L1 DONE (0703779, + fe4692f):** `models/mamba2_780m_memory/` built on a
  parameterized shared machinery (memory geometry derived from the
  backbone; `read_layer`/`injected_layers` ctor params). Integration knob:
  **`MEMORY_INTEGRATION=state|mix`** env var (in `sft/.env.example`).
  Full suite green (78 tests incl. tiny-backbone GPU tests: none-mode
  invariance, mix exact no-op at zero-init, causality); both arms smoked
  on the real 780M backbone locally (state arm 21.7M trainable, mix 12.3M).
  Implementation decisions (details in the model README):
  - Front-end/M/writes IDENTICAL between arms (purity); the 128 bottleneck
    is the shared `down` projection, `W_o: 128→d_model` zero-init, β from
    the same o_t+surprise+anneal signals. Bottleneck kept at 128 unscaled.
  - Scaled indices: state arm READ_LAYER=32, INJECTED_LAYERS=16..46 step 2;
    mix arm boundary at layer 16. **Known A/B confounds, accepted:** the
    read point necessarily differs between arms (32 vs 16 — same-token
    causality forces the mix arm to read where it lands), and parameter
    count is asymmetric (16 injection modules vs 1 mix module).
  - Mix cadence: read+mix per token against window-start M; writes per
    window — write semantics identical to the state arm.
  - Checkpoints do NOT record their integration arm — set
    `MEMORY_INTEGRATION` to match when probing/resuming (mismatch now
    warns loudly via the fe4692f unexpected-key guard).
- **BX0/BX1 box addendum:** the fused-path mix hook could not execute
  locally (no causal_conv1d on ROCm) — **first thing on the box, smoke
  `MEMORY_INTEGRATION=mix` with a short fused-path run** before committing
  to the BX1 leg. `DEFAULT_CHUNK_LEN=16` in the 780m hooks is a guess;
  pass `--chunk-len` explicitly.
- L3 (dream neutral-snapshot mode) remains open — the only unbuilt local
  item; optional for this rental.
