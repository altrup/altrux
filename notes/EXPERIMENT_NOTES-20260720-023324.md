# Experiment notes — mamba2_2_7b_memory, episodic-chains run, 2026-07-20

Fresh Lambda instance, auto-started by setup script. Owner AFK. Operating
autonomously within the /experimenter brief.

## Starting state (read from prior run + disk)

- **Prior run**: `notes/EXPERIMENT_NOTES-2026-07-16.md` (GH200). Summary: fixed a
  non-finite crash (unnormalized gated-delta write key), then found the core
  problem — the neural memory `M` contributes ~0 to recall because every recall
  demand in the training data sits within the SSM's own capacity, so the
  optimizer routes recall through the backbone and suppresses `M` (alpha
  climbing, o_t_norm shrinking, ablation deltas ~0). Interference data narrowed
  but didn't close the gap; the SSM kept re-absorbing the niche.
- **New since then (episodic-chains track, `docs/.../2026-07-17-episodic-chains-design.md`)**:
  makes the niche *structural*. A "chain" concatenates many episodes into one
  long example with **SSM-state wipes ("sleep")** at random between-episode
  boundaries; recall queries that cross a sleep are answerable ONLY through `M`
  — no gradient path lets the backbone absorb them. Alpha ceiling cut 0.1 → 1e-4
  (sigmoid multiplier) so erosion can't decay `M` away over long chains. Data:
  `sft/data/train_chains.pt` (2.26 GB, on disk). Weighted objective is now the
  default (`--recall-weight 8 --head-weight 4 --head-tokens 1024`).
- **Checkpoint on disk**: only `epoch-1/step-168` (prior run reached step-308+
  in probes; only 168 was rsynced/restored here). Its `dataset_fingerprint` is
  the old v2 dataset, `state_batch_size=16`, `total_tokens=4,132,742`. Trainable
  = 474 keys (LoRA + front_end.knob_proj etc.). Predates the alpha-ceiling and
  sleep-slot changes, but those are training mechanics / a smaller sigmoid
  multiplier — no param-shape change, so it warm-starts cleanly.

## Plan for this run

- Baseline command (from brief): `make resume ARGS="--data data/train_chains.pt
  --eos-weight 32 --batch-size 8 --chunk-len 48 --memory-window 8
  --accum-tokens 1536 --ckpt-every-tokens 147456"`. Resumes step-168 weights +
  optimizer; slots start fresh against train_chains (fingerprint won't match).
  Weighted-loss defaults apply.
- Goal: get `M` to earn its place — a positive ablation delta in the
  **cross-sleep** regime (the new only-`M`-can-answer probe condition), holding
  across checkpoints.
- Monitor per brief: every ~5 min first hour, then ~15 min. Probe checkpoints
  as they land.

## Timeline

### 02:33–02:36 UTC — training started (log train-20260720-023345.log)

- **RESUME POINT**: `resuming from .../checkpoints/epoch-1/step-168`, loaded
  474/474 trainable tensors, resumed at step 168 / epoch 1. Dataset changed
  (train_memory_v2 → train_chains, 2753 chains) → slot_states/next_ptr
  discarded, slots start FRESH against train_chains. 62,097,965 trainable params
  (128 LoRA adapters). GPU: NVIDIA GH200 480GB, 94 GB VRAM free.
- **Starting memory metrics** (first chunks, slots at token 0 of chains sized
  40k–156k tokens):
  - o_t_norm ≈ **13.5** across all 8 slots — the memory read magnitude. Strong
    and uniform. (Contrast: prior run collapsed this to ~2.6 under suppression;
    13.5 means reads are contributing strongly at warm-start.)
  - beta ≈ 0.62, retain ≈ 0.982 (write gate open, inherited from step-168; no
    re-anneal hump — loss starts at ~1.3–2.7/chunk, not the 8.9 fresh-start
    spike).
  - surprise 0.10–0.31 (NOT flat at ~1.0 → memory writes are meaningful).
  - min_cos_sim 0.67–0.93 (NOT near −1 → no anti-parallel write blow-up, the
    prior crash signature).
  - ssm_norm 12–21 (bounded), grad_norm 2–5, resid_norm ~3–6k. Zero non-finite.
  - alpha/w1_abs_max not in the current chunk_extra_log; alpha is now
    ceiling-capped at 1e-4 (episodic-chains change) so erosion is structurally
    bounded. Will measure via `measure_knobs.py` at the first probe pause.
- Assessment: clean, healthy warm-start. The suppression-era step-168 read
  strength (o_t_norm 13.5) survived; now training on structural-niche chain data
  should either keep it earning its place (cross-sleep recall) or reveal whether
  the SSM-wipe regime changes the picture. Watching.

### 02:41 UTC — batch-size 8 → 12 (owner-prompted throughput tune)

- Owner (back briefly) noted only ~70 GB / 98 GB VRAM used. Checked nvidia-smi:
  **GPU util only 23%** — the run is compute-underutilized, not memory-bound, so
  the real win from more slots is throughput/utilization (faster
  checkpoints/probes = less billed time), not "using more RAM." Cheapest possible
  moment to change it: slots had only ~5k tokens of progress.
- Stopped, restarted with `--batch-size 12` (all else identical). At the C-c,
  slot 7 ssm_norm had crept to ~39 (from 12–21 at start) — still bounded, not a
  runaway; o_t_norm still ~14. Watching VRAM settle after restart; will drop
  back to 10 if it approaches the cap. **Current run args: batch-size 12**,
  everything else as the baseline command.
- Flagged to owner as an alternative use of the headroom: `--chunk-len 48 → 64`
  (the spec §2 names chunk-len as the hard limit on the memory's retention
  gradient — BPTT truncates there). Went with batch per their ask; chunk-len is
  a candidate follow-up.

### 02:41 UTC — batch 12 OOMed → reverted to batch 8

- **Two findings that settle the batch question:**
  1. GPU util stayed at **23%** going 8→12 slots — batch parallelism is NOT the
     throughput bottleneck (it's sequential: likely the per-write memory
     recurrence / chunk driver). So more batch buys little wall-clock.
  2. Batch 12 **OOMed** ~90s in, at `ssm_state = retain*forgotten + written`
     (_mixer_step, model.py:818). The 91 GB steady-state reading was misleading —
     peak activation as multiple slots hit their compute-heavy boundary steps
     together climbed past the 94.5 GB cap (OOM needing just 30 MiB more).
- Verdict: the team's **batch 8 / chunk-len 48** is the right operating point;
  reverted to it. The ~24 GB of "free" VRAM is peak-allocation headroom the run
  genuinely needs, not slack. Not spending it on batch. (chunk-len 64 would also
  eat peak activation — would need to drop batch to try it; parked.)
- This was a self-inflicted OOM from my batch change, not a code bug — reverting
  and continuing, not a stop-condition.

### 02:42 UTC — resumed at batch 8 (baseline), stable

- Log `train-20260720-024115.log`, resumed from step-168. VRAM back to
  **69 GB / 98 GB** (29 GB margin), util ~21%. o_t_norm ~13.5, surprise 0.16–0.31,
  min_cos_sim 0.73–0.92, ssm_norm 12–13 — all nominal, matching the first
  batch-8 start. Persistent failure/checkpoint monitor re-armed on this log.
- **This is the settled operating point for the run.** First checkpoint expected
  ~147k new tokens in (~13 min). Will probe cross-sleep recall on the first
  couple checkpoints.

### 02:47 UTC — verified the cross-sleep regime is genuinely active (owner asked)

- Data (`train_chains.pt`): 2580/2753 chains carry 1–20 backbone resets each;
  213,511 flagged query tokens; **2011/2051 query-bearing chains have ≥1 query
  landing AFTER a sleep** — the only-`M`-can-answer case. 702 chains have no
  engineered queries (natural-continuation signal).
- Loop wiring end-to-end: train.py reads `sleep_positions` (:1001), passes into
  every `_Slot` (:532/581/796), and fires the wipe at :636–640 when
  `slot.pos >= slot.sleeps[i]` via `sleep_slot_fn` → train_hooks `sleep_slot`
  (:103) → `model.sleep_slot` (backbone zeroed, memory persists). The
  "sleeps will be ignored" guard (:1003) did NOT fire — hook present. Resets are
  silent (no log line), so absence from logs ≠ inactive.
- Trend @02:47: loss ~1.7 (from 2.7), o_t_norm ~14.5, depth 8k, 0 non-finite,
  VRAM 70 GB, util ~17–45%. Healthy.

### 02:52 UTC — added reset logging + answered owner's two questions

- Owner asked: (1) are sleeps logged? — NO, they fired silently. (2) does the
  per-slot `token N/TOTAL` count the whole chain or one episode? — the WHOLE
  chain (TOTAL = full chain length, 30k–202k tokens, mean 84k; the counter runs
  continuously chain-start→end, only the backbone STATE resets at sleeps).
- Fix (commit **0ca94ff**, pushed): train.py now prints
  `[sleep] slot N: backbone wiped at token X/LEN (example …, sleep i/total)` per
  fired sleep. Safe observability change (a print, no logic change; py_compile
  OK). Takes effect at the next restart — folding into the first-probe
  pause/resume, so no dedicated restart. Git identity was unset on the fresh box
  (set repo-local to altrup); had to rebase onto 2 incoming scripts/ commits.
- **Note for next check**: the currently-running process predates 0ca94ff, so
  sleeps stay silent until the first-probe resume. After that resume, expect
  `[sleep]` lines — their appearance is live confirmation the wipes fire.

### 02:55–03:0x UTC — first checkpoint (step-180) + baseline probe

- First save landed at **step-180** = step-168 + 12 optimizer steps. Cadence
  math: `--accum-tokens 1536` is PER SLOT, so real tokens/step = 1536×8 = 12,288,
  and 147,456 / 12,288 = **12 steps/save** (NOT 96 — `accum_tokens` is per-slot,
  ×batch_size; see train.py:445-446, :906). Saves at step-192, -204, … ~13 min
  each. (Corrected an earlier wrong "96 steps" answer to owner.)
- Owner request: run a probe NOW (before more training) for a before/after
  comparison. Paused training at step-180 (≈ warm-start baseline, only 12
  chains-steps in), touched watchdog-delay, ran probe in the `train` tmux window
  (owner: GPU/long commands go through train tmux, not a shell I own; also
  enforced by commit a18f41d).
- **Probe #1 (16 probes) OOMed** — a *harness* bug, not training. The new
  `--sleep` path held 4 full batched states alive at once (prefix + ablated +
  sleep-intact + sleep-ablated), each with a ~1.5 GB neural-memory copy; at
  n_probes 16 it hit 93 GB and died allocating sleep-ablated's fresh memory
  (probe_recall.py:275). Fix (commit **ebca934**, pushed): `del` + `empty_cache`
  each condition's state once scored — scoring is sequential, so peak drops from
  ~4 states to ~2. Re-running at 16 probes.
- Probe config (standard for this run's before/after): `--sleep --gaps 1024
  --n-facts 64,128,256 --n-probes 16 --seed 1234`. Success metric = **sleep-delta
  = sleep-intact − sleep-ablated** (both backbone-wiped; memory kept vs replaced).
- Pulled owner commits mid-session (a18f41d, 8839d25, 313da5e) — clean rebase.

### 03:06 UTC — BASELINE probe (step-180), 16 probes, seed 1234 (the "before")

Full output in `notes/probe-step180.out`. Log-prob per code digit (higher=better):

| facts | intact | ablated | floor | mem-delta | slp-int | slp-abl | **slp-delta** |
|---|---|---|---|---|---|---|---|
| 64  | −1.479 | −1.598 | −5.352 | +0.119 | −4.929 | −5.343 | **+0.413** (±0.162) |
| 128 | −2.247 | −2.335 | −5.198 | +0.089 | −4.818 | −5.238 | **+0.420** (±0.103) |
| 256 | −2.417 | −2.567 | −5.035 | +0.150 | −4.681 | −5.023 | **+0.343** (±0.116) |

- **All three cross-sleep deltas strongly positive** (~+0.34–0.42; SEM=std/4, so
  ~10–16σ). The memory ALREADY contributes across a backbone wipe at warm-start.
- **sleep-ablated ≈ floor** at every fact count (256: −5.023 vs −5.035) — clean
  confirmation that backbone-wipe + memory-replace = at floor (SSM alone carries
  nothing across a wipe, exactly the structural niche). The +delta is entirely
  the kept memory.
- Absolute cross-sleep recall is tiny though: sleep-intact ~0.7–0.9%/digit vs
  ~0.5% floor. So the memory carries a real but faint gist. **The run's success
  criterion: grow sleep-intact (and thus sleep-delta) with chains training,
  holding across checkpoints.** Baseline to beat: slp-delta +0.34–0.42.
- No-sleep mem-delta is also modestly positive here (+0.09–0.15), unlike the
  prior run's ~0 — plausibly the warm-start's strong reads (o_t_norm 13.5).
- **Next**: resume training, let it accumulate (~step-300+), re-probe with the
  SAME config for the after. Watch whether slp-delta and slp-intact rise.

### 03:07 UTC — resumed from step-180 (post-probe), reset-logging live

- Log `train-20260720-030720.log`, resumed from step-180. Confirmed we're running
  the weighted objective via DEFAULTS (didn't pass flags): `--recall-weight 8`,
  `--head-weight 4`, `--head-tokens 1024` (train.py:914-916). Head-weight applies
  after example-start AND each sleep (verified with owner).
- @03:18: **`[sleep]` reset-logging works** — 6 wipes fired, e.g. `slot 1:
  backbone wiped at token 30528/93569 (example 1520, sleep 2/5; memory persists)`.
  Live proof the mid-chain backbone resets happen. o_t_norm **rising** 13.5 → 16.6
  (reads strengthening, anti-suppression), ssm_norm max ~68 (bounded; sleeps
  reset it), loss ~2.3–2.7/chunk (noisier — slots now hitting 4–8× weighted
  regions), 0 non-finite, util 69%. Healthy. Checkpoints step-168+step-180 on
  disk (keep=2). Target next probe ~step-300.
- @03:34 (≈step-205, depth 56k): loss back to ~1.7. Clean metric distributions
  (last 64 slot-lines): **surprise** min .027/med .107/max .271 (healthy spread,
  not collapsing), **o_t_norm** min 16.4/med 18.7/max 21.8 — **steadily rising
  from 13.5** = memory reads strengthening (anti-suppression, the trend we want).
  beta ~0.49–0.55 (down from 0.62 → writes more selective as reads grow).
  0 non-finite, 12 sleeps fired. Encouraging trajectory into the after-probe.

### 04:42 UTC — AFTER probe #1 (step-256, 76 steps past baseline) — MIXED/CONCERNING

Full output `notes/probe-step256.out`. Same config/seed as baseline.

| facts | slp-delta @180 | **slp-delta @256** | slp-int 180→256 | slp-abl 180→256 | floor 180→256 |
|---|---|---|---|---|---|
| 64  | +0.413 | **−0.013** (±0.160) | −4.93→−3.74 | −5.34→−3.73 | −5.35→−3.71 |
| 128 | +0.420 | **+0.086** (±0.124) | −4.82→−3.55 | −5.24→−3.64 | −5.20→−3.63 |
| 256 | +0.343 | **+0.101** (±0.113) | −4.68→−3.46 | −5.02→−3.56 | −5.04→−3.57 |

- **The cross-sleep delta SHRANK** toward ~0 (64-fact gone; 128/256 down to weak
  ~+0.09–0.10, only ~3σ vs ~10–16σ at baseline). This is the wrong direction for
  the goal.
- BUT absolute recall rose ~3× (sleep-intact 0.9%→3.1% @256) — AND so did the
  **floor** (0.5%→2.5%). sleep-ablated still ≈ floor everywhere. Reading: chains
  training sharply improved the model's **parametric prior** on the code-query
  task (floor up), which re-absorbed the memory's measurable edge. Same
  re-absorption the prior run hit in the no-sleep regime — here it comes via the
  PARAMETRIC path (LoRA/head), since the SSM path is wiped. The cross-sleep
  regime blocks SSM substitution but NOT parametric substitution.
- **Why not panic-intervene yet (positive reason, per brief):** only 2 points,
  and a specific transient is plausible — the +0.4 baseline came from step-168,
  which was tuned on the OLD interference data for exactly this recall; chains
  training (new objective + alpha ceiling 1e-4) is mid-transition to a new
  equilibrium and may dip-then-recover. A 3rd point resolves transient-dip vs
  monotonic-decline; it's cheap (~5 min). So: TIGHTER cadence — probe again at
  ~step-296 (~30 min), then ~step-330.
- **If decline confirmed → intervention hypotheses (getting ready now):**
  1. Most queries (within-episode, cross-episode-within-wake) are answerable
     WITHOUT the memory; only cross-sleep queries need it. If those are a
     minority of the recall_masks tokens, the recall-weight-8 gradient is
     dominated by memory-unnecessary queries → memory role atrophies. Fix:
     regenerate data weighting/increasing CROSS-SLEEP queries specifically, or
     add a separate higher weight for cross-sleep query tokens (needs
     prepare_chains.py to mark them distinctly).
  2. o_t_norm stayed high (~18) through training yet delta shrank — reads are
     strong but not carrying fact-specific gist across the wipe. Points at the
     WRITE side (gist quality) more than the read side.
- Resumed training from step-256 at 04:44.

### 04:50 UTC — data-composition analysis (prep for possible intervention)

Testing hypothesis 1 (memory-requiring gradient is diluted). Findings:
- Of 213,511 recall-flagged query tokens: 86.6% occur after ≥1 sleep, 12.9%
  within the first wake, 0.5% in zero-sleep chains. So NOT dominated by early
  memory-unnecessary queries at the coarse level.
- BUT the true three-distance split (prepare_chains.py:194-211): each query's
  distance is chosen **uniformly at random** among the available
  {within_episode, cross_episode, cross_sleep} (line 210). So true **cross-sleep**
  queries — the ONLY ones that require the memory (SSM wiped between fact and
  query) — are only **~1/3** of engineered queries. The other ~2/3
  (cross_episode-within-wake, within_episode) are SSM-answerable. `--recall-weight
  8` amplifies all equally, so ~2/3 of the boosted recall gradient trains
  SSM/parametric recall, not memory. **Partial dilution confirmed (~1/3
  memory-requiring), consistent with the floor-rises-faster-than-delta result.**
- **Intervention lever (if point 3 confirms decline):** bias the memory-requiring
  gradient up — either regenerate with distance selection weighted toward
  cross_sleep (e.g. 60% vs uniform 33%), or add a separate higher weight for
  cross_sleep query tokens (prepare_chains already classifies them; would emit a
  distinct mask). Moderate cost (data regen ~3 min + optional train.py flag).
- Still holding: 2 probe points don't justify the intervention. Get point 3
  (~step-296) first — transient-dip vs monotonic-decline.

### 05:35 UTC — AFTER probe #2 (step-295): DECLINE CONFIRMED (3 monotonic points)

| facts | slp-delta @180 → @256 → @295 | floor @180→256→295 | slp-int @180→256→295 |
|---|---|---|---|
| 64  | +0.413 → −0.013 → **−0.015** | −5.35→−3.71→−3.34 | −4.93→−3.74→−3.38 |
| 128 | +0.420 → +0.086 → **+0.022** | −5.20→−3.63→−3.30 | −4.82→−3.55→−3.27 |
| 256 | +0.343 → +0.101 → **+0.035** | −5.04→−3.57→−3.19 | −4.68→−3.46→−3.16 |

- **Monotonic decline to ~0** on all fact counts. floor rises in lockstep with
  sleep-intact → parametric re-absorption. Not a transient. The chains objective
  (as configured) trains the memory OUT of the cross-sleep loop.
- **INTERVENTION 1 launched (05:38):** regenerate data with
  `--cross-sleep-bias 0.75` (commit **cdd6b9e**) → `data/train_chains_xs.pt`.
  Concentrates the ~1/3 memory-requiring cross-sleep queries up to ~majority, so
  the shared read machinery isn't pulled toward SSM-reliance by the 2/3
  SSM-answerable queries. Original `train_chains.pt` kept intact for revert.
  Regen runs in a `prep` tmux session (CPU-only, concurrent with training on old
  data — no idle). When ready: swap training to the xs data (fresh slots, weights
  continue from latest ckpt), probe after ~1h.
- **Hypothesis / kill criterion:** if the memory-requiring gradient is the fix,
  slp-delta should stop declining and recover above ~+0.1, ideally toward the
  +0.3–0.4 baseline, over ~1–2h on xs data. If it stays ~0, dilution is NOT the
  bottleneck → the limit is write-side (BPTT truncation severs credit assignment
  from cross-sleep query back to the fact-write; memory relies on the local
  delta-rule write, which may lack capacity/precision for specific codes) — a
  structural finding for the team, not a data fix.
- Training resumed on OLD data at 05:37 (no idle) pending the swap.

### 05:40 UTC — SWAPPED to cross-sleep-biased data (intervention 1 live)

- Regen result: `train_chains_xs.pt`, 230.9M tokens, queries now **cross_sleep
  21573 / within 3729 / cross_episode 2707 = 77% cross-sleep** (was ~33%). Same
  ~214k recall-answer tokens. Original `train_chains.pt` untouched (revert path).
- Swapped: stopped at step-295, `make resume --data data/train_chains_xs.pt`
  (all other args identical). Resumed step-295 weights+optimizer; dataset change
  → fresh slots against xs. Log `train-20260720-053948.log`. Healthy start
  (VRAM 69 GB, o_t_norm 13.5 @token0, loss 2.23, 0 non-finite). Monitor re-armed
  (ban5c3tet). **Current run args: --data data/train_chains_xs.pt, batch 8, else
  baseline.**
- Plan: 15-min trend checks; **first intervention probe ~06:40 (~step-345, ~1h
  of xs data)**, same probe config. Watch slp-delta: recover = dilution was it;
  flat ~0 = write-side/structural limit (report for team).

### 06:31 UTC — INTERVENTION PROBE (step-332, 37 steps of xs data): IT WORKED

Full output `notes/probe-step332.out`. Same config/seed.

| facts | slp-delta @180 → @256 → @295 → **@332** | slp-int/slp-abl/floor @332 |
|---|---|---|
| 64  | +0.413 → −0.013 → −0.015 → **+0.048** | −3.418 / −3.465 / −3.493 |
| 128 | +0.420 → +0.086 → +0.022 → **+0.137** | −3.399 / −3.536 / −3.557 |
| 256 | +0.343 → +0.101 → +0.035 → **+0.191** | −3.281 / −3.473 / −3.464 |

- **The decline REVERSED.** After only 37 steps on cross-sleep-biased data, the
  256-fact delta went +0.035 → **+0.191** (~6σ; SEM≈std/4≈0.03), 128 → +0.137
  (~4.5σ), 64 back positive. sleep-ablated ≈ floor everywhere, so this is the
  memory's genuine contribution. Diagnosis confirmed: **dilution of the
  memory-requiring gradient was a real driver of the re-absorption.** The
  intervention (77% cross-sleep queries) fixed the *direction*.
- Not yet back to the +0.34–0.42 baseline, but climbing fast in 37 steps and
  now trending UP. Need to confirm it keeps climbing / holds across more
  checkpoints (that's the brief's success criterion: positive delta in the
  high-fact/cross-reset regime, holding across checkpoints).
- Caveat: 1 post-intervention point. The recovery is strong + consistent across
  all fact counts, but next probe (~1h) confirms durability vs one-off.
- Resumed xs-data training at 06:32. **Next probe ~07:30 (~step-390)** to check
  the delta keeps rising / holds. If it holds ≥ baseline across 2 probes → that's
  the success condition; bank it and stop.

### 07:38 UTC — CONFIRMATION probe (step-383): recovery did NOT hold

Full output `notes/probe-step383.out`.

| facts | slp-delta @295→332→383 | floor @295→332→383 | slp-int≈abl≈floor @383? |
|---|---|---|---|
| 64  | −0.015 → +0.048 → **−0.046** | −3.34→−3.49→−3.07 | −3.13/−3.09/−3.07 yes |
| 128 | +0.022 → +0.137 → **−0.001** | −3.30→−3.56→−2.97 | −3.00/−3.00/−2.97 yes |
| 256 | +0.035 → +0.191 → **−0.007** | −3.19→−3.46→−2.70 | −2.74/−2.73/−2.70 yes |

- **The step-332 recovery (+0.19) COLLAPSED back to ~0 by step-383.** Non-monotonic:
  +0.035 → +0.19 → ~0. The floor kept climbing hard (256: −3.46→−2.70), and
  sleep-intact ≈ sleep-ablated ≈ floor again — parametric prior re-absorbed the
  memory a second time, now even on 77%-cross-sleep data.
- **Read:** the data-composition intervention gave only a TRANSIENT blip, not a
  durable fix. Strong evidence the bottleneck is NOT gradient dilution but
  structural — the parametric prior keeps improving and re-absorbing regardless
  of how memory-heavy the query mix is. Consistent with the write-side hypothesis
  (BPTT truncation severs credit assignment from cross-sleep query → fact-write;
  memory can't learn to store specific codes that beat the improving prior).
- **Before concluding:** getting ONE more point (~step-410) to rule out that
  step-383 is a noisy down-swing vs genuine collapse. step-383's consistency
  (intact≈abl≈floor on all 3 fact counts) already argues genuine ~0. Resumed
  xs training at 07:39.
- **Leaning toward:** if step-410 ≈ 0, conclude the data fix is insufficient →
  the remaining levers are structural (memory capacity / write mechanism /
  credit assignment), which are expensive + low-confidence + a team-design
  discussion per the brief → write up thoroughly and STOP rather than burn hours
  guessing at architecture overnight.

### 08:09 UTC — deciding probe (step-396): COLLAPSE confirmed, not oscillation

| facts | slp-delta @332 → @383 → @396 | floor @396 |
|---|---|---|
| 64  | +0.048 → −0.046 → **−0.023** | −2.569 |
| 128 | +0.137 → −0.001 → **−0.049** | −2.555 |
| 256 | +0.191 → −0.007 → **−0.028** | −2.401 |

Two consecutive ~0 points (383, 396); floor now near chance (256: 9.1%/digit),
intact≈abl≈floor. step-332's +0.19 was a transient blip. **Data intervention
does not durably work.** Bottleneck is structural: the parametric prior keeps
improving and re-absorbing regardless of query mix.

### 08:14–08:23 UTC — INTERVENTION 2: --freeze-lora (memory-only training)

- Rationale ("one more real idea," targeted at the confirmed mechanism): if the
  parametric (LoRA) path is what re-absorbs the niche, **freeze it and train only
  the memory** (front_end projections + injection modules, 218 params) — removes
  the re-absorption route so the memory alone must carry cross-sleep recall, or
  fail cleanly. Either outcome is decisive.
- Implemented `--freeze-lora` (commit **94ce514**): keeps requires_grad on all
  params (checkpoints stay complete → probes/resume unaffected), shrinks only the
  optimizer's set to the 218 non-LoRA params. First launch crashed (optimizer
  state param-group size mismatch); fixed by skipping the saved optimizer state
  under --freeze-lora (fresh optimizer, commit pushed).
- Live from 08:23, log `train-20260720-082324.log`, resumed step-396 on xs data
  (slots continued — same-fingerprint resume). Healthy: o_t_norm 20–24 (strong),
  **beta dropped 0.5 → ~0.21** (write gate adjusting now the memory trains alone),
  0 non-finite. Monitor bermb8d9w.
- **Hypothesis / decision rule:** probe ~09:05 (~45 min). If slp-delta GROWS and
  holds → parametric re-absorption was suppressing a capable memory (huge — points
  to a training-schedule fix). If it stays ~0 → the memory genuinely cannot carry
  cross-sleep exact-code recall (capacity / credit-assignment limit) → conclusive
  structural finding, STOP + hand off. **This is the last planned experiment;
  either result concludes the run.**

### 09:08 UTC — FREEZE-LORA PROBE (step-435, 39 steps memory-only): NO recovery

| facts | slp-delta @396 (pre-freeze) -> @435 (memory-only) | floor @435 |
|---|---|---|
| 64  | -0.023 -> **-0.011** | -2.792 |
| 128 | -0.049 -> **-0.031** | -2.721 |
| 256 | -0.028 -> **+0.024** | -2.616 |

All ~0. Freezing the parametric path did NOT unlock a memory contribution
(256-fact +0.024 ~ 0.7 sigma, noise). Concludes the run.

---

## CLOSING SUMMARY (2026-07-20, ~09:10 UTC) - for the team

### What this run did
Warm-started from the prior run's step-168 onto the NEW episodic-chains regime
(`train_chains.pt`, SSM wipes at sleeps -> cross-sleep recall answerable only by
the neural memory `M`). Goal: get `M` to earn a positive, durable cross-sleep
ablation delta at high fact counts. Batch 8 (batch 12 OOMs; util-bound anyway).

### Full cross-sleep delta trajectory (slp-delta = sleep-intact - sleep-ablated, log-prob/digit; SEM ~ std/4 at 16 probes)

| step | data / mode | 64 | 128 | 256 | note |
|---|---|---|---|---|---|
| 180 | chains (warm-start baseline) | **+0.413** | **+0.420** | **+0.343** | strong (~10-16 sigma) |
| 256 | chains | -0.013 | +0.086 | +0.101 | declining |
| 295 | chains | -0.015 | +0.022 | +0.035 | ~0 |
| 332 | chains_xs (77% cross-sleep) | +0.048 | +0.137 | **+0.191** | transient recovery |
| 383 | chains_xs | -0.046 | -0.001 | -0.007 | collapsed back |
| 396 | chains_xs | -0.023 | -0.049 | -0.028 | ~0 |
| 435 | chains_xs + `--freeze-lora` | -0.011 | -0.031 | +0.024 | no recovery |

`floor` (never-saw-it recall) rose monotonically 0.5% -> ~9%/digit as the model
learned the code-query format parametrically; sleep-ablated ~ floor at every
probe (SSM wiped + memory replaced = at floor, as designed).

### Core finding
1. **A cross-sleep memory contribution CAN exist** - the warm-start (from prior
   interference training) showed +0.34-0.42 robustly. The read/inject path can
   surface `M` across a wipe.
2. **Chains training re-absorbs it.** The parametric prior (LoRA + base LM) keeps
   improving at the code task (floor rises) and the memory's measurable edge
   decays to ~0. The cross-sleep design blocks *SSM* substitution but NOT
   *parametric* substitution.
3. **Data composition is NOT the bottleneck.** 77% cross-sleep queries
   (`--cross-sleep-bias 0.75`) gave only a transient blip (+0.19 @332) that
   collapsed by @383.
4. **Parametric competition is NOT the whole story either.** Freezing LoRA +
   memory-only training (218 params, 39 steps) did not recover the delta.
   => **Structural bottleneck: `M` as architected/trained cannot reliably store &
   retrieve specific facts (exact 5-digit codes) across an SSM wipe.** Likely
   mechanism: the write gets no long-range BPTT gradient (truncated at
   `--chunk-len` 48; a cross-sleep query is thousands of tokens + a sleep after
   its fact), so `M`'s contents come from the LOCAL delta-rule write, which
   appears to lack the capacity/precision for many exact codes.

### Recommended next experiments (team decision; structural, GPU-heavy)
- **Freeze at an EARLIER checkpoint (pre-re-absorption) + memory-only** - tests
  whether the +0.4 delta can be *preserved/grown* vs *recovered from collapse*.
  Cheapest next test; still uncertain.
- **Increase memory capacity/precision** (`mem_hidden` / write rule). Param-shape
  change -> fresh training. Targets the storage-capacity hypothesis directly.
- **Auxiliary local write supervision** - an in-chunk reconstruction/retrieval
  loss on the just-written fact, so `M` learns to store retrievably without
  long-range BPTT.
- **Resolve a probe-design question FIRST:** the cross-sleep probe tests EXACT
  5-digit-code recall, but `M` is designed for *gist*. `M` may store gist the
  exact-code probe can't see. Try a gist-shaped cross-sleep eval (topic/entity
  recall, or natural-continuation perplexity delta after a sleep) before
  concluding `M` fails - it may be succeeding at what it's for.

### Code changes this run (all pushed to main)
- `0ca94ff` reset-logging (`[sleep]` lines).
- `ebca934` probe_recall OOM fix (free each condition's state; enables n_probes 16).
- `cdd6b9e` `prepare_chains --cross-sleep-bias`.
- `94ce514` (+ optimizer-load fix) `train --freeze-lora`.
- READMEs updated for both new flags.

### Artifacts (rsync'd, not git)
- `notes/probe-step{180,256,295,332,383,396}.out`,
  `notes/probe-step435-frozenlora.out` - raw probe outputs.
- Checkpoints step-168..435 on disk (keep-50). Regenerate the biased data with
  `make prepare-chains ARGS="--cross-sleep-bias 0.75 --output data/train_chains_xs.pt --seed 7"`.

### Why stopping (not grinding)
Two honest, well-reasoned interventions targeting the confirmed mechanism, both
negative; delta flat ~0 with no positive trend; remaining levers are structural
architecture changes (fresh multi-hour training, low confidence, team-design
decisions). Per the brief's GOING NOWHERE clause, releasing the billed box is
correct - checkpoints + these notes carry everything forward.
