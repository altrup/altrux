# Experiment notes — mamba2_2_7b_memory, reproducibility run, 2026-07-21/22

Fresh instance, auto-started session. Standing direction:
`DISCUSSION-20260721-peak-reproducibility.md` (Tests 1–3, Test 2 first on the
box while Test 1's data regenerates), with `DISCUSSION-20260720-gist-eval.md`'s
rejected items still standing.

State on arrival: all six checkpoints present under
`models/mamba2_2_7b_memory/checkpoints/epoch-1/` (steps 180/256/295/332/396/435;
396+435 with optimizer.pt per the .env note), venv OK, NO prepared data
(expected on a fresh box). tmux sessions `train` and `experimenter` exist.
Persistent monitor armed (train-log error grep + 300s heartbeat).

## Plan for this session

1. **Test 2 (435 robustness, probe-only)** — gist eval on corpora outside the
   training pool: ultrachat_200k `test_sft` split (held out; training used
   train_sft), tokenized UNCAPPED. babilong: prepare_babilong.py emits
   single-QA rows (user = doc+question, asst = short answer), so the only
   usable turn boundary has only answer-length tokens after it → expected
   structurally INELIGIBLE for gist-cont 512; will verify with an eligibility
   count and report rather than assume. Re-derive eligibility per corpus
   before sweeping; compare 435 vs 180 within-corpus only.
2. **Test 1 (396 reproduction)** — rebuild the original pool exactly
   (ultrachat train_sft, max-examples 20000, --max-len 1024 → train.pt;
   data-memory as-is → train_memory.pt; prepare-chains --cross-sleep-bias 0.75
   --seed 7, no split/mid-sleep flags). Verify regen log ~10.4k sleeps /
   0 mid-conversation / 0 splits. Resume step-396 --freeze-lora, probe at
   +13/+26/+39, judge the trend.
3. Test 3 after, per its own section in the discussion notes.

## Log

### 23:22 UTC — session start
- Monitor armed, watchdog-delay touched, notes read.

### 23:30 UTC — data prep launched
- `train` tmux: eval ultrachat (test_sft, uncapped 32768 default) →
  data/eval_ultrachat.pt, then eval babilong → data/eval_babilong.pt.
- `prep` tmux (45s stagger to avoid HF-cache download races):
  babilong 16k eval raw → make data-memory → ultrachat train_sft
  max-len 1024 (original pool) → prepare-chains --cross-sleep-bias 0.75
  --seed 7. Logs under sft/logs/prep-*.log.

### 23:37 UTC — fresh-box gotcha: sft/logs/ didn't exist
- Both prep chains stalled after their first command: `tee logs/...` failed
  (fresh instance has no sft/logs/), making each pipeline exit nonzero and
  stopping the `&&` chain. mkdir'd sft/logs/ and restarted the remaining
  steps. eval_ultrachat.pt (23110 convs) and babilong_eval_raw.jsonl (1000
  ex, 16k config) had already completed.

### 23:32 UTC — prep complete; eligibility derived; Test 2 probes launched
- Test 1 pool regen VERIFIED: train_chains.pt = 2753 chains, 230.9M tokens,
  10387 sleeps, 0 mid-conversation, 0 split-tail (target ~10.4k/0/0). Also
  0 sentence-boundary. Queries: 3729 within-ep, 2707 cross-ep, 21573
  cross-sleep; 213.9k recall-answer tokens.
- Eligibility (eligibility_check.py, new script):
  - eval_ultrachat.pt (test_sft, held out, 23110 convs, median 1114, max
    4774): prefix2048/cont512 → 459 eligible; prefix3072 → 24; prefix4096 → 0.
    Config chosen: prefix 2048 / cont 512 / recent 576 / distractor 1536 /
    n 16 / seed 1234.
  - eval_babilong.pt (16k config, 1000 convs): ZERO eligible at every
    (prefix, cont) down to cont 64 — single-QA shape has no boundary with a
    long prefix before AND real continuation after. babilong is structurally
    unusable for the gist harness; Test 2 rests on ultrachat test_sft.
- Test 2 probes launched in train tmux: step-435 then step-180, same config,
  done-markers + background waiters for instant wake.

### 23:37 UTC — Test 2, step-435 on held-out ultrachat (test_sft)
Config: prefix 2048 / cont 512 / recent 576 / distractor 1536 / n 16 / seed
1234. Log: sft/logs/probe-20260721-233231.log.
- gist-delta +0.0120 (SEM 0.0060); dist-delta +0.0054 (SEM 0.0019);
  recency +0.0083 (SEM 0.0069); long-range +0.0038 (SEM 0.0050);
  awake-mem +0.0086 (SEM 0.0049); wipe cost +0.70.
- Positive (~2 SEM) on a corpus never in training; distractor-survival
  positive at ~2.8 SEM. Recency-leaning rather than long-range-dominant —
  but prefix is 2048 (vs 6144 on LongAlign; recent window covers 28% of
  prefix), so long-range room is structurally small; do NOT read structure
  across corpora. Thirds flat (+0.0118/+0.0128/+0.0115), i.e. delta does
  not decay over the continuation.
- Verdict pends the within-corpus 180 contrast (running).

### 23:52 UTC — Test 2 COMPLETE: 435's gist effect GENERALIZES; verdict PASS
step-180, same config (log probe-20260721-233730.log):
- gist-delta −0.0364 (SEM 0.0100); long-range −0.0588; dist-delta −0.0233;
  thirds WORSENING with distance (−0.0100/−0.0345/−0.0646). Memory at 180
  actively hurts held-out continuation prediction.
- Within-corpus ordering replicates: 435 (+0.0120) vs 180 (−0.0364),
  separation ~0.048 >> joint SEM (~0.012). Same trajectory direction as
  LongAlign (−0.068 → +0.0285). Deliverable confirmed general (with the
  caveat: recency-leaning structure on this corpus, prefix only 2048 by
  eligibility — structure not comparable across corpora).
- babilong leg: structurally ineligible (see 23:32) — reported, not run.

### 23:55 UTC — Test 1 LAUNCHED: 396 reproduction
- Archived epoch-1/step-435 → checkpoints/archive-20260722-test1/ (keeps it
  probe-able; makes 396 latest for resume).
- Resume command: baseline args + --freeze-lora, with --ckpt-every-tokens
  49152 (⅓ baseline → checkpoint every ~13 steps to serve the +13/+26/+39
  probe schedule; save cadence only, no dynamics change).
- Expected at startup: resuming from step-396; optimizer param-match FAIL →
  fresh optimizer (matches original condition); cold slots (no slot state
  survives — judge trends, not first points).

### 23:43 UTC (log train-20260721-234322.log) — Test 1 resume point RECORDED
- resuming from epoch-1/step-396; --freeze-lora active (218 memory params,
  LoRA fixed); fresh optimizer (saved state covers 474 params, current 218 —
  expected, matches original condition); slots discarded (dataset regen →
  fingerprint mismatch, cold slots as anticipated).
- First memory metrics pending first step lines — record on next check.

### 23:50 UTC — Test 1 starting memory metrics (train-20260721-234322.log)
- First memory lines: beta 0.43→0.32, retain ~0.976, alpha 0.0000,
  surprise 0.21→0.17 (not flat at 1.0), o_t_norm 16.8→18.8 (strong; original
  leg ran 20–24), w1_abs_max ~0.023 stable, w2_abs_max ~0.014–0.017.
  Healthy start; beta adjusting downward as in the original freeze-lora leg.
- First checkpoint saved: step-400 (early save after resume is expected —
  token counter anchors at resume). Probe targets remain 409/422/435.

### 00:0x UTC — training stopped on user request at ~step 410
- User asked to stop; interrupted cleanly. Latest checkpoint step-408 (+12
  from 396). This leg's checkpoints: 400/404/408 (cadence ~4 steps, ~1
  step/min). Metrics healthy throughout (alpha 0, o_t_norm ~18, surprise
  0.09–0.17). Awaiting user direction; watchdog-delay maintained meanwhile.

### 00:15 UTC — verbatim commands run so far (new standing rule: always log these)
- Eval prep: `make prepare ARGS="--hf-dataset HuggingFaceH4/ultrachat_200k --hf-split test_sft --output data/eval_ultrachat.pt"`
- babilong eval: `uv run --no-sync python prepare_babilong.py --configs 16k --max-per-split 100 --output data/babilong_eval_raw.jsonl` then `make prepare ARGS="--input data/babilong_eval_raw.jsonl --max-len 100000 --output data/eval_babilong.pt"`
- Test 1 pool: `make data-memory` ; `make prepare ARGS="--hf-dataset HuggingFaceH4/ultrachat_200k --max-examples 20000 --max-len 1024 --output data/train.pt"` ; `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7"`
- Test 2 probes (each): `make probe-recall ARGS="--gist data/eval_ultrachat.pt --gist-prefix 2048 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-435"` (and same with step-180)
- Test 1 training: `make resume ARGS="--data data/train_chains.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
  (ckpt-every-tokens 49152 vs baseline 147456 is deliberate: denser saves for
  the +13/+26/+39 probe schedule; save cadence only. NOTE: the original
  396→435 leg's verbatim command is NOT recorded in prior notes — only
  reconstructable (xs data + --freeze-lora + baseline args). Hence this rule.)

### 00:2x UTC — arg provenance settled; training resumed
- User challenge on arg match resolved: prior notes quote baseline verbatim
  (identical incl. eos-weight 32 / batch 8 / chunk 48 / window 8 / accum
  1536); state.pt in 396+435 records dataset train_chains_xs.pt n=2753
  batch 8; regen pool = 2753 examples. Only unrecorded residue: hypothetical
  unmentioned extra flag on the original leg.
- Resumed (verbatim): `make resume ARGS="--data data/train_chains.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
  Expect: resuming from step-408, fingerprint MATCH this time (same file) →
  slots continue (mem_state on newest ckpts) or restart per train.py.

### 00:35 UTC — reached +38 (step-434); stopped; Test 1 probe sweep launched
- Training C-c'd cleanly after step-434 save (+38 of +39 target; saves at
  400/404/408/412/416(?)/420/425/429/434 — ~4-step cadence).
- Sweep (verbatim, per step in {408, 420, 434}):
  `make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-$s"`
  (LongAlign corpus + config/seed identical to the original 435
  characterization; probe points nearest +13/+26/+39.)
- PASS bar (from DISCUSSION): gist-delta ≥ ~+0.02 by the +39 point,
  long-range dominant, thirds rising; judge the TREND across the three.

### 00:40 UTC — Test 1 first probe point, step-408 (+12), LongAlign
(log probe-20260722-003427.log)
- gist-delta +0.0144 (SEM 0.0049); long-range +0.0198 (dominant; recency
  −0.0054); dist-delta +0.0217 (SEM 0.0063); awake-mem +0.0176.
- 435-like structure already present at +12. Awaiting 420 and 434.

### 00:47 UTC — step-420 (+24): rise continues
(log probe-20260722-004006.log)
- gist-delta +0.0186 (SEM 0.0066); long-range +0.0294; recency −0.0109;
  dist-delta +0.0317 (SEM 0.0064); awake-mem +0.0211.
- Monotone 408→420 on gist/long-range/dist. PASS bar (~+0.02 by +39) within
  reach; step-434 decides.

### 00:55 UTC — TEST 1 = PASS: the 396→435 gist rise REPRODUCES
step-434 (+38), log probe-20260722-00xxxx.log (newest):
- gist-delta +0.0304 (SEM 0.0050) — above the +0.02 bar AND above original
  435's +0.0285; long-range +0.0331 (dominant, recency −0.0027); dist-delta
  +0.0202; awake-mem +0.0269; thirds RISING (−0.0052/+0.0354/+0.0606).
- Full trajectory: +12 → +0.0144, +24 → +0.0186, +38 → +0.0304. Monotone,
  right structure at every point, despite cold slots at the 396 resume
  (success = strong evidence per the asymmetry note).
- CONCLUSION: dataset-change-at-resume was the killer; the recipe is real
  and robust even to a cold-slot resume. The four erosion regimes are
  explained by mechanism (b) (+possibly (c)); (a) transient-peak now tests
  free by continuing.
- Proceeding per PASS branch: continue training past +39 on the SAME data,
  probe every ~13 steps (every ~3rd checkpoint), watch for peak-then-erode.
  Resumed (verbatim): `make resume ARGS="--data data/train_chains.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`

### 01:32 UTC — transient-test leg 1 done (434→472); sweep launched
- Trained 434→472 on unchanged data (verbatim resume command as at 00:55;
  resume confirmed "from step-434", internal state loaded). Metrics healthy
  throughout (o_t_norm 17–26, alpha 0, surprise 0.04–0.26).
- Sweep (verbatim, per step in {447, 460, 472}):
  `make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-$s"`
- Question: does gist keep rising past the original peak span, hold, or
  peak-then-erode with NO data change? Stopping rule (from DISCUSSION Test
  3, applied here): keep argmax; stop after 3 consecutive probes below best
  with degraded structure.

### 01:40 UTC — step-447 (+51): NEW BEST, rise continues past original peak span
(log probe-20260722-013424.log)
- gist-delta +0.0344 (SEM 0.0050); long-range +0.0398; dist-delta +0.0304.
- Beats original 435 (+0.0285) and this run's 434 (+0.0304). No transient
  yet.

### 01:47 UTC — step-460 (+64): first decline, structure intact
(log probe-20260722-013959.log)
- gist-delta +0.0206 (SEM 0.0026); long-range +0.0293; dist-delta +0.0190.
- Down from 447's +0.0344 (~2.4σ joint) but structure still 435-like.
  Possible transient peak near ~447 even with NO data change; 472 next.

### 01:55 UTC — step-472 (+76): EROSION CONFIRMED ON UNCHANGED DATA
(log probe-20260722-01xxxx.log, newest)
- gist-delta +0.0150 (SEM 0.0041); long-range +0.0097; recency +0.0053 (now
  positive/growing); dist-delta +0.0058 (SEM 0.0021, collapsed from 0.0304);
  thirds flattened. Classic erosion signature.
- Trajectory this run: +12 → +0.0144, +24 → +0.0186, +38 → +0.0304,
  +51 → +0.0344 (PEAK, step-447), +64 → +0.0206, +76 → +0.0150.
- MECHANISM (a) CONFIRMED: peak-then-erode with NO dataset change, no resume
  discontinuity (continuous stream, exact slot state). Peaks are optimization
  transients; the recipe produces them reliably (Test 1 PASS) but continued
  training always converts gist → recency. Practical rule: HARVEST the
  argmax; don't train through it.
- Resumed for one short leg to ~485 (verbatim command as before) to complete
  the 3-consecutive-below-best stopping rule; then held-out check of 447.

### 02:05 UTC — Test 3 data build launched (prep tmux)
- Verbatim: `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7 --split-episode-rate 0.5 --split-qa-rate 0.9 --split-gap-min 1 --split-gap-max 4 --output data/train_chains_split.pt"`
  (explicit --output to preserve train_chains.pt; DISCUSSION's spec otherwise
  verbatim). Will verify: sleep density near ~10.4k, single-QA splits the
  majority of splits.
- Decision (user asked): Test 3 enters from step-396, NOT step-447 — resuming
  a peak under changed data is the established killer; 447 stays banked.

### 02:1x UTC — Test 3 data v1 REJECTED, rebuilding with lower split rate
- v1 (spec-verbatim flags): 19733 sleeps (vs xs 10387 — tokens/wake halved
  22.2k → 11.7k, ≈ the all-signals density mechanism (c) blames), and
  single-QA splits only 3871/9374 = 41% (spec requires majority). The spec's
  flags and its own density/majority constraints are mutually inconsistent —
  each split adds one sleep, so density scales with split count.
- Rebuild (verbatim): `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7 --split-episode-rate 0.15 --split-qa-rate 0.9 --split-gap-min 1 --split-gap-max 4 --output data/train_chains_split.pt"`
  Expected ≈5.5k splits (~70% single-QA), ≈15.9k sleeps, ~14.5k tokens/wake.
  Rationale: conversational splits are the weak-amplitude component; cutting
  them buys density without losing the single-QA signal the notes rank as
  the active ingredient.

### 02:20 UTC — Test 3 data v2 VERIFIED; closing probes launched
- v2: 5498 splits (3918 single-QA = 71% majority), 15857 sleeps, ~14.6k
  tokens/wake. Accepted as Test 3 data.
- Training stopped at step-486 (+90). Closing probes (verbatim):
  1. LongAlign gist @486 (3rd-consecutive-below-best check):
     `make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-486"`
  2. Held-out ultrachat @447 (deliverable generalization):
     `make probe-recall ARGS="--gist data/eval_ultrachat.pt --gist-prefix 2048 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-447"`

### 02:35 UTC — lineage closed; TEST 3 LAUNCHED from step-396 on split data
- step-486 LongAlign: gist-delta +0.0104, long-range +0.0041, dist-delta
  +0.0019 (log probe-20260722-020701.log). THIRD consecutive below-best with
  degraded structure → stopping rule complete. Lineage argmax: step-447
  (+0.0344).
- step-447 on held-out ultrachat (log probe-20260722-021243.log): gist-delta
  +0.0024 (SEM 0.0045, ≈0) BUT long-range +0.0073 and dist-delta +0.0097 —
  both better than 435's (+0.0038/+0.0054); headline lower than 435's
  +0.0120 because 447's recency term is negative (−0.0050). 447 =
  longer-range/more-durable shape; 435 = higher held-out headline. BOTH
  banked (447 in archive-20260722-test1-lineage/, 435 in
  archive-20260722-test1/). Deliverable choice = team discussion item.
- Archived steps 400–486 (21 dirs) → archive-20260722-test1-lineage/;
  epoch-1 back to 180/256/295/332/396.
- Test 3 resumed (verbatim): `make resume ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
  Expect: resuming from step-396, fresh optimizer, slots reset (new dataset).
  Plan: probe every ~13 steps from ~+38 on; success bar = beat +0.0344 with
  435-like structure; stopping rule = 3 consecutive below best w/ degraded
  structure. Watch for LATER peak (single-QA split signal may climb slower —
  its amplitude arrives via document answers scheduled 1–4 wakes out).

### 02:58 UTC — Test 3 first sweep launched (steps 408/420/434)
- Trained 396→434 on train_chains_split.pt, healthy throughout (o_t_norm
  15.5–19.2, alpha 0, surprise 0.07–0.21).
- Sweep per step in {408, 420, 434} (verbatim):
  `make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --checkpoint ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-$s"`
- Read: vs Test 1's trajectory at same offsets (+12 +0.0144 / +24 +0.0186 /
  +38 +0.0304). Structure right + amplitude weak → head-weight 8 next leg.

### 03:05 UTC — Test 3 step-408 (+12): slightly ahead of Test 1 at same offset
(log probe-20260722-025927.log)
- gist-delta +0.0161 (SEM 0.0049) vs Test 1 +0.0144; long-range +0.0221 vs
  +0.0198; dist-delta +0.0234 vs +0.0217. Structure right, trending ≥ Test 1.

### 03:12 UTC — Test 3 step-420 (+24)
(log probe-20260722-030506.log)
- gist-delta +0.0180 (Test 1: +0.0186, even); long-range +0.0347 (vs
  +0.0294); dist-delta +0.0331 (vs +0.0317). Distance/durability ahead,
  headline even — the split signal's intended fingerprint.

### 03:20 UTC — Test 3 step-434 (+38); leg 2 resumed
(log probe-20260722-03xxxx.log, newest)
- gist-delta +0.0269 (SEM 0.0054) vs Test 1 +0.0304; long-range +0.0387 (vs
  +0.0331); dist-delta +0.0296 (vs +0.0202!); recency −0.0117; awake-mem
  +0.0362 (elevated — watch for always-on drift, but long-range dominance
  says not erosion-shaped yet); thirds −0.0009/+0.0292/+0.0524.
- Read: headline a touch behind Test 1, distance+durability clearly ahead —
  split signal doing its intended job. No knob change (amplitude fine).
- Resumed leg 2 (verbatim, same command):
  `make resume ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 49152 --freeze-lora"`
  Next sweep at ~447/460 (Test 1's peak window).

### 03:44 UTC — Test 3 leg 2 done (434→460); peak-window sweep launched
- Sweep steps {447, 460}, same verbatim probe command as 02:58 entry.
- Comparison: Test 1 peaked at 447 (+51) with +0.0344.

### 03:55 UTC — Test 3 step-447 (+51)
(log: 2nd-newest probe log)
- gist-delta +0.0304 (SEM 0.0060); long-range +0.0251 (down from +38's
  +0.0387); dist-delta +0.0180 (down from +0.0296); recency +0.0053 (now
  positive). vs Test 1 @447: +0.0344/+0.0398/+0.0304.
- Headline still rising (0.0269→0.0304) but structure softening — possible
  earlier erosion onset than Test 1. 460 decides.

### 04:05 UTC — Test 3 step-460 (+64): sliding; final leg launched
(newest probe log)
- gist-delta +0.0187 (SEM 0.0038); long-range +0.0143; dist-delta +0.0102;
  recency +0.0044. First below-best strike (best: 447-T3 +0.0304).
- Emerging verdict: Test 3 peak = SAME LOCATION (~+51) but LOWER headline
  than Test 1 (+0.0304 vs +0.0344); its distinctive win is mid-climb
  durability (dist +0.0296 @+38, night's best). Split signal changes the
  shape, not the peak height; erosion unbeaten.
- Final leg resumed (same verbatim command) to ~473 for strikes 2/3.

### 04:14 UTC — Test 3 stopped at step-476 (+80); closing probes queued
- Probes (verbatim patterns as before): LongAlign @468, @476 (stopping-rule
  strikes 2/3); held-out ultrachat @447-T3 and @434-T3 (argmax candidates:
  headline peak and durability peak respectively).

### 04:35 UTC — Test 3 CLOSED by stopping rule
- LongAlign trajectory (offsets from 396): +12 +0.0161 / +24 +0.0180 /
  +38 +0.0269 / +51 +0.0304 (PEAK, step-447-T3) / +64 +0.0187 /
  +72 +0.0208 / +80 +0.0078. Three consecutive below-best with degraded
  structure (long-range +0.0086, dist +0.0052 at 476).
- VERDICT: strengthened-split data does NOT raise the peak (+0.0304 <
  Test 1's +0.0344); same peak location (~+51); it does buy the night's best
  mid-climb durability (dist-delta +0.0296 at +38). Erosion undefeated by
  data composition — consistent with mechanism (a) being about the LM loss
  itself, not the mix.

### 04:40 UTC — CORRECTION to the 04:35 entry (log misattribution)
- The "+0.0078 @476" I logged was actually 447-T3 on ULTRACHAT
  (probe-...042555.log). Correct assignments, from checkpoint lines in logs:
  - 476-LA (probe-...042017.log): gist-delta +0.0219 (SEM 0.0046),
    long-range +0.0128, recency +0.0092, dist-delta +0.0110.
  - 447-T3 on ultrachat (042555): gist-delta +0.0078 (SEM 0.0044),
    long-range +0.0086, recency −0.0008, dist-delta +0.0052.
  - 434-T3 on ultrachat (043019): gist-delta +0.0047 (SEM 0.0034),
    long-range +0.0077, recency −0.0030, dist-delta +0.0066.
- Corrected Test 3 LongAlign trajectory: ... +51 +0.0304 (peak) /
  +64 +0.0187 / +72 +0.0208 / +80 +0.0219. Still 3 consecutive below best →
  stopping rule holds, Test 3 stays closed. BUT the shape differs from
  Test 1: not a collapse toward ~0.01, a PLATEAU at ~+0.02 with long-range
  still positive and recency creeping up. Split data may flatten the erosion
  tail even though it doesn't raise the peak — worth a probe-eval in the
  next run before dismissing.
- Held-out comparison across candidates (ultrachat, same config):
  435-orig +0.0120 | 447-T1 +0.0024 | 447-T3 +0.0078 | 434-T3 +0.0047.
  435 keeps the best held-out headline; T3's candidates sit between, with
  long-range ~+0.008 and near-zero recency (long-range-shaped like T1's).

## CLOSING SUMMARY — run of 2026-07-21/22 (~5.5 h)

### What was established
1. **Test 2 PASS — 435's gist generalizes.** Held-out ultrachat test_sft:
   435 +0.0120 vs 180 −0.0364 (same ordering as LongAlign). babilong is
   structurally ineligible for the gist harness (single-QA shape; verified
   zero eligibility down to cont 64 via new sft/eligibility_check.py).
2. **Test 1 PASS — the 396→435 rise REPRODUCES** on exactly-rebuilt xs data
   (2753 chains / 10387 sleeps, matches checkpoint fingerprints): +12
   +0.0144 → +38 +0.0304, right structure, despite cold slots. The four
   prior erosion regimes are explained by dataset-change-at-resume.
3. **Transient confirmed — erosion needs NO discontinuity.** Continuing
   Test 1's run unchanged: peak +0.0344 at step-447 (+51), then 0.0206 /
   0.0150 / 0.0104 with long-range→recency conversion. Peaks are
   optimization transients; harvest the argmax, never train through it.
4. **Test 3 (71% single-QA splits, ~14.6k tok/wake): peak NOT raised**
   (+0.0304 at +51, vs +0.0344) but (a) mid-climb durability best of night
   (dist-delta +0.0296 at +38) and (b) the erosion tail PLATEAUS ~+0.02
   instead of collapsing (+64/+72/+80 = 0.0187/0.0208/0.0219). Split data
   changes the shape, not the height.

### Deliverable candidates (all banked, all probe-able)
| ckpt | where | LongAlign gist | held-out UC gist | UC long-range | notes |
|---|---|---|---|---|---|
| 435-orig | archive-20260722-test1/ | +0.0285 | +0.0120 | +0.0038 | best held-out headline |
| 447-T1 | archive-20260722-test1-lineage/ | **+0.0344** | +0.0024 | +0.0073 | best LongAlign peak |
| 447-T3 | archive-20260722-test3-lineage/ | +0.0304 | +0.0078 | +0.0086 | balanced; best UC long-range |
| 434-T3 | archive-20260722-test3-lineage/ | +0.0269 | +0.0047 | +0.0077 | best dist-delta (+0.0296 LA) |
Which is THE deliverable = team call (headline vs long-range/durability).

### Discussion items for the team
- T3's erosion PLATEAU (~+0.02 with structure half-intact) vs T1's collapse:
  if real, split data buys erosion-resistance — test by training T3 lineage
  further (from 476, SAME data) and probing the tail, before any new recipe.
- Peak height seems capped ~+0.03–0.034 across recipes → stage-2 bottleneck
  hunt (read-out bandwidth first) per the north-star notes.
- Data-spec inconsistency found: each split adds a sleep, so the DISCUSSION's
  Test 3 flags (rate 0.5/qa 0.9) give 19.7k sleeps + 41% single-QA — I ran
  rate 0.15 instead (15.9k sleeps, 71% single-QA). v1 build rejected, logged.
- Deliverable choice above; also whether ultrachat prefix-2048 headline is
  the right held-out ruler (only 459 eligible convs; corpus-capped prefix).

### Housekeeping
- Pushed: a2e9276 (experimenter brief: verbatim-command rule), 14948ec
  (sft/eligibility_check.py). Monitor recipe upgrade pulled from origin
  (0951fae) and used all night — completion wakes worked well.
- Data files on box (not synced, rebuildable from commands above):
  train_chains.pt (xs regen), train_chains_split.pt (v2), eval_ultrachat.pt,
  eval_babilong.pt, train_memory*.pt, train.pt.
- epoch-1 left at 180/256/295/332/396; lineages in archive-20260722-test1/,
  -test1-lineage/ (21 dirs), -test3-lineage/ (19 dirs).
