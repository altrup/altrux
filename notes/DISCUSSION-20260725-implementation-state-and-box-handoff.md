# Discussion notes — 2026-07-25 (later): implementation state and box handoff

Team session (altrup + Claude). The plan of
`DISCUSSION-20260724-next-run-plan.md` §1.3–1.5 is now implemented and
pilot-validated; this note is the state-of-the-world bridge for the box
session. The 07-24 plan's §6 gates remain the go/no-go authority; the same
morning's `DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md` holds
the CL-side decisions.

## 1. What is implemented and committed

- **Trainer** (`aaa39c7`, `47502cd`, `6707ea5`): repeatable `--data
  path[,key=value]*` with per-slice `share`/`chunk-len`/`batch-size`/
  `grad-checkpoint`/`shuffle`; same-config slices interleave example-by-example
  by token share, differing-config groups alternate in `--mix-segment-tokens`
  segments with drain (not cut) handovers; recall-weight ramp 1→`--recall-weight`
  over `--recall-ramp-steps` (default 32 = the beta-anneal window);
  `--grad-ckpt-block` plumbs checkpoint block size to the model hook.
- **Gradient checkpointing** (`60e1cde`, `83b7cef`): block-wise
  (`models/common.py:blockwise_checkpoint`), both manual and fused paths via
  the shared dispatch wrapper. Required replacing the write's
  `autograd.grad(create_graph=True)` with a closed-form gradient (naive
  checkpointing *raised* VRAM — every window's write replayed its block during
  forward). Equivalence pinned by tests incl. second differentiation; **fp
  rounding changes in every run**, checkpointed or not — bit-continuity with
  BX checkpoints is gone (fresh-LoRA plan makes this free).
- **Generators** (`333a5d1`, `e34ecb4`, `6e99b5e`): `prepare_cram.py`
  (Wikipedia IMR, data-side gap curriculum — geometric ceilings, per-item
  log-uniform gaps, `shuffle=0` realizes it), `prepare_needles.py` (babilong
  needles, one bAbI story per block, Wikipedia filler). NER via
  `transformers` (no cp314 spaCy wheels — see `sft/CLAUDE.md`).
- **Filter** (`d7e20e1`, `0e3e6fc`, `9ad26ae`, `4ab5066`): three-test A/B/C
  battery + string leak-check; raw scores stored per item so `--rescore`
  re-verdicts without a GPU pass; `--leak-only` mode; samples print
  immediately and the fail composition rides the progress line.

## 2. Pilot results (this box, 300/600 articles)

- **Cram** (`data/pilot_cram.pt`, 18 blocks / 401 items / 0.10M tokens): all
  five zero-expected structural invariants zero; credit 1.14% of tokens; gap
  floor holds. Decoded samples read correct (entity-substituted sources,
  completed-sentence answers, credit on entity span only, varied cue
  positions). Quirk: realized gaps can overshoot the recorded ceiling by
  passage granularity (`gap` vs `target_gap` both recorded).
- **Filter calibration on cram**: shipped defaults kept 26%. Diagnosis from
  stored scores: the −0.7 A floor sat above the median item (substituted
  spans are implausible *by design*, so the backbone's prior fights the copy
  even with source visible — fail_a median A −1.4 at A−B margins of 6+
  nats), and margin-over-`max(B,C)` double-counted C. Calibrated defaults
  (`--a-min -2.0`, `--min-margin 4.0` over B only): **194/401 kept (48%)**,
  every fail category doing its designed job — fail_b 0 (substitution fully
  beats pretraining leakage), fail_a 62 (genuinely bad clozes, e.g.
  section-heading blanks), fail_leak 30, fail_c 96 (see §3.1).
- **Needles** (`data/pilot_needles.pt`, 172 blocks / 172 items / 0.20M
  tokens): invariants zero. The A/B/C battery is **not valid for this
  slice**: the base backbone cannot express bAbI's cold bare-entity answers
  after a role marker (A −12..−18 in *every* context; cram answers ride a
  long copied-sentence runway, needles have none), so scoring measures format
  competence, not item quality. bAbI items are well-posed and leak-proof by
  construction → **`--leak-only` for needle artifacts**: 166/172 kept, 6
  leak discards. Recorded option if per-item SSM-verification ever matters:
  rescore in a base-model-native `Q:/A:` format.

## 3. Findings that change box behavior

1. **Prose interference is weak.** fail_c concentrates at short gaps (median
   435 vs 758 for kept): ~400 tokens of flowing Wikipedia prose does *not*
   defeat the SSM, though 192 tokens of dense labeled facts did
   (`RESEARCH-20260724-local-diagnostics.md` §1). The keystone gate stands,
   but the early curriculum (short gaps, inside BPTT reach) loses ~a quarter
   of its items to fail_c. Lever for full-scale generation: **denser
   interference in early blocks** (more items, less prose per block) so
   short-gap items can be memory-required; fail_c rate is the gauge.
2. **Within-window credit is probabilistic.** Fixed chunking gives an item
   an unbroken write→read credit path only when source and answer share a
   chunk — ~(chunk−gap)/chunk, i.e. ~12–40% for early-curriculum items at
   chunk 512. Consider chunk 640–768 for cram in the gate-5 sweep: raises
   early-item credit coverage to ~42–58% for ~1 GB/slot more retained state.
3. **VRAM arithmetic, confirmed the hard way.** The write graph
   (windows-per-block × ~225 MB/slot at 780m geometry) plus the backbone's
   per-token live graph dominate. The **default checkpoint block (64) exceeds
   this local card's ~52-token live-graph ceiling** — full-geometry cram
   config cannot train on 8 GB at any chunk length (four OOMs at an invariant
   ~7.6 GiB proved it). Box (80 GB): block 64 default stands; sweep batch
   from 8.
4. **`--memory-window` must be passed explicitly (8).** The model default is
   a write every token — 8× the write compute and write-graph VRAM.
   Omitting it was one of this session's OOM causes. The box invocation
   below carries every constant explicitly.
5. **Slice sizing**: emitted token counts must be checked against the
   35/15/35/15 shares before launch (a slice that exhausts early re-normalizes
   the rest and the run tail drifts off-share). Cram needs ~2× candidate
   oversampling at the measured 48% keep; needle filler wants tens of
   thousands of articles (pilot wrapped at 1.1×).

## 4. Box invocation sketch (constants explicit; sweep before locking)

```
--data 'data/train_cram.pt,share=35,chunk-len=512,batch-size=<sweep from 8>,grad-checkpoint=1,shuffle=0' \
--data 'data/train_needles.pt,share=15,chunk-len=512,batch-size=<same>,grad-checkpoint=1,shuffle=0' \
--data 'data/train_chains.pt,share=35,chunk-len=48,batch-size=24' \
--data 'data/train_ballast.pt,share=15,chunk-len=48,batch-size=24' \
--memory-window 8 --accum-tokens 1536 --recall-weight 16 --eos-weight 32
```

Gate-5 sweep on the rented card before locking: batch (from 8), chunk
640/768 vs 512 (finding §3.2), block via `--grad-ckpt-block` if needed.
Recall ramp defaults are already aligned to the anneal window.

## 5. Integration status, and a standing rule

The multi-slice pipeline was smoke-tested locally at reduced dims (chunk 64,
block 16, real pilot artifacts): **three optimizer steps, finite decreasing
losses, checkpoint recomputes live** — schema→loader→per-slice-config→
checkpoint-hook→closed-form-write→recall-weighting all verified. Not
exercised live: the config-group handover (CPU tests only) and anything at
real dims (§3.3).

**Standing rule (owner decision, this session): no training-shaped work on
the local box.** Local GPU is for inference/scoring only. The box's first
act, before full-scale generation or training: rerun this smoke at real
config on tiny slices (1-block artifacts, minutes, catches everything the
local card cannot).

## 6. Queued, in order

1. **Box step 0**: real-config integration smoke (§5), then gate-5 sweep (§4).
2. **Full-scale generation on the box**: cram ~2× oversampled with denser
   early-block interference (§3.1), needles with a large article pool, chains
   + ballast regenerated with the repaired pipeline (`make data` is already
   uncapped). Read the decoded samples before training — the rule held twice
   this session (caught `str.replace` mangling and NER subword junk in
   review, and the pilot read validated the shape).
3. **Full filter on the box** (A/B/C cram, `--leak-only` needles); check the
   calibrated thresholds' keep-rate and composition at scale via `--rescore`
   sweeps (free) before regenerating anything.
4. **Transcript-consolidation null** — still queued, unowned, must run
   before the training budget is committed (see the 07-25 morning note §4).
5. The run itself, under the 07-24 plan's §6 gates — NaN watch (§6.8)
   unchanged and sharpened by §3.4 here: watch `GRAD_NORM` /
   `[nonfinite-write]` in the first hours.

Smoke artifacts (`sft/data/smoke_*.pt`, `pilot_*`-series) are disposable;
`models/mamba2_780m_memory_mix/checkpoints/epoch-1/` predates this session
(2026-07-24 01:37–02:42 local run) and was deliberately left untouched.

---

# Part 2 — same day, evening: data regenerated, filter re-economized, ops loop closed

Everything below happened after Part 1, same team session. Part 1's §6 queue
is superseded by §12 here.

## 7. All training artifacts regenerated and verified (this box, tonight)

- **Base corpora** (`make data` + `make data-memory`): ultrachat uncapped
  24.4M tokens / LongAlign 203.7M / babilong 11.1M with **all 2000 episodes
  carrying `question_offsets`**. Every artifact in `sft/data` now postdates
  every pipeline repair.
- **Chains** (`data/train_chains.pt`, 35.6M tokens, 523 chains): regenerated
  with `--sources data/train.pt data/train_memory_babilong.pt --split-qa-rate
  0.5 --mid-sleep-min-len 1536 --fact-rate 0`. Result: **0 malformed
  transitions** (validator now hard-fails on any), 78 split-QA suspensions
  (gaps 3.2k–7.8k), 68 mid-conversation sleeps, 8.4% sleeping chains
  (`--sleep-chain-rate 0.1`, the plan's ~10% attribution fraction —
  `f64488b`). LongAlign is excluded from chains; it is consumed directly as
  the ballast slice.
- **Two more latent generators bugs found by the read-the-log rule** and
  fixed: (a) **fact blocks emit structurally illegal role sequences**
  (consecutive unanswered `[USER]` fact turns; query pairs spliced between a
  question and its answer) — 14,716 malformed transitions in one regen, all
  inside fact-bearing chains. Standing decision: `--fact-rate 0` for chains
  (plan-conformant — retention lives in cram); the fact-block splicer itself
  is unrepaired. (b) `--split-min-part` applied to both sides of a QA cut,
  and babilong's dataset-authored tails are 8–12 tokens — **split-QA had
  been silently zero in every dataset that ever passed a rate**. Fixed
  head-only (`25a7d84`); a zero split-tail counter now means a constraint
  bound, not luck.
- **Legacy `train_memory.pt` deleted** (merged single-`--data`-era artifact;
  nothing in the plan consumes it; regenerable via `merge_data.py`).

## 8. Density result (pilot2) and the C decision

Densified cram generation (`efedc51`: one source passage can answer several
items) measured against the baseline pilot, same articles, same calibrated
thresholds:

- **Yield is the real win**: 590 items vs 401 (+47%), 288 kept vs 194
  (+48%), short-gap kept items 81 vs 57 (+42%) — ~1.5× recall credit per
  article/NER-dollar.
- **Short-gap fail_c improved 41% → 31% but is nowhere near zero** → density
  cannot be trusted open-loop, a gap threshold cannot replace test C (fail_c
  spans the whole gap range), and **per-item C stays for this run**. Its
  cost concern is answered by batching (§9), not by dropping it.
- **C's verdict is a t=0 lower bound that erodes as training improves
  SSM-carry** (owner's observation; BX runs demonstrably improved SSM-range
  recall). Anchors slowing the decay: capacity is information-theoretic once
  interference volume exceeds the state (selectivity has no signal — probe
  identity is unpredictable at read time), and the curriculum sends the
  hardest items to the most-trained model. Standing decision: **post-run
  C-rescore against the final checkpoint** (~minutes, batched) as the
  erosion diagnostic; mid-run C-refreshes only if that measurement says so.
  A/B/leak verdicts are properties of the text and do not decay.

## 9. Filter economics after batching (`11d4d8c`)

All three passes run batched (`--score-batch`, default 16 sized for this
8 GB card — **raise it on the GH200**); progress line carries tok/s (pad
excluded) + ETA, every line timestamped. Cost model, GH200, full-scale cram
(~50M candidate tokens, ~200k items): B+C are block-amortized long passes
(~minutes); A is per-item and the only real bill (~15–20 min batched).
Whole filter ≈ **tens of minutes**, smaller than the NER pass (~30–60 min
batched). Measured-trigger fallbacks if a box pilot disagrees: skip C above
a gap threshold several× the interference capacity, sample-audit A.
First box act for the filter regardless: a pilot-sized run (minutes) to
replace every number above with a measurement.

**Scoring-regime caveat**: serial vs batched scores differ beyond matmul
noise (median |Δ| 0.04 nats, max ~0.5 late in long blocks — bf16 recurrence
divergence compounding with position; verdicts held on the equality check,
and the CPU fake-model test pins index-correctness bitwise). Scores are
consistent *within* one run; treat cross-regime borderline flips as noise
and calibrate any threshold change on the box's own batched scores via
`--rescore`.

## 10. Ops loop (scripts/, all committed)

- **Launch uploads local data artifacts** (`train*.pt eval_*.pt` in
  `sft/data`, sizes on the confirm screen; `LAMBDA_DATA_ARTIFACTS`
  overrides). Today's set ≈ 2.4 GB, all load-bearing, all regenerated
  tonight. Pilots/smokes/raw intermediates never sync.
- **`lambda_pull.sh` pulls the same set home** (plus checkpoints), `--follow`
  runs continuously in the local watch window — box-generated artifacts and
  their embedded filter scores archive here automatically; pull explicitly
  after the filter, before training starts. Generate once, reuse forever:
  a future launch uploads instead of regenerating.
- **`work` tmux session** on the box for everything that isn't training
  (one named window per task); local `altrux` session gets a live attach
  window to it. The `train` session's history stays a pure training log.
- **MODEL_NAME**: launch tooling never chooses a model — setup blanks it on
  the box (unnamed runs fail at import); the experimenter passes
  `MODEL_NAME=<arm>` inline on every command (inline beats `.env`).
- **Data sanity gate** (`make sanity-sample` + a haiku-class subagent reads
  the decoded windows) before the first training start and after any on-box
  regeneration — wired into the experimenter skill.
- Root CLAUDE.md logging rules added: results stream as produced (never
  end-of-run-only), every log line timestamped.

## 11. Run sizing (deciding data: tonight's inventory)

~70M-token run: chains 24.5M (pool ceiling ~35M — comfortable) / cram 24.5M
kept (≈50M candidates at the measured ~49% keep → ~150k articles through
NER) / needles 10.5M (large filler pool; pilot wrapped at 1.1×) / ballast
10.5M (LongAlign, abundant). Check emitted per-slice token counts against
35/15/35/15 before launch — an exhausted slice re-normalizes the rest.

## 12. The box sequence (supersedes §6)

1. Launch (uploads chains/ballast/pools; confirm screen shows the set).
2. Integration smoke at real config on tiny slices (§5) — the one seam
   never live-tested is the config-group handover.
3. **Transcript-consolidation null — DEFERRED out of this run's critical
   path** (owner-approved reasoning, launch night): no harness exists, and
   building one unsupervised on billed time is worse than deferring. This
   run's value doesn't depend on it — it trains wake-time memory (M's one
   non-substitutable role), while the null gates the *CL design*. Run it
   next session (or in a quiet stretch late in this one if training is
   healthy and attended); it stays mandatory before any CL implementation
   work.
4. Gate-5 sweep: batch from 8, chunk 640/768 vs 512 (§3.2), filter
   `--score-batch` raise, pilot-sized filter+NER runs for real throughput
   numbers.
5. Full-scale generation (`prepare_cram` densified defaults, ~2×
   oversample; `prepare_needles` with a large `--articles`) →
   `make sanity-sample` + cheap-subagent read → batched filter (A/B/C cram,
   `--leak-only` needles) → **pull artifacts home** → train, per the §4
   invocation with `--recall-weight 16` explicit and `MODEL_NAME` inline.
6. NaN watch unchanged (§6.8 of the 07-24 plan): `GRAD_NORM` /
   `[nonfinite-write]` in the first hours; post-run C-rescore (§8) at the
   end.
