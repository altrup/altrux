# Experiment notes — 2026-07-24 01:48 UTC — 780M three-cell integration screen

Fresh GH200 96 GB instance. Repo at 1d965cb, clean. Standing direction:
`../discussion/DISCUSSION-20260723-780m-integration-screen.md` — this run is the 780M
screen ONLY: BX0 (state@32, calibration), BX1 (mix@16), BX2 (state@16 via
`MEMORY_READ_LAYER=16`), in that order, all three unconditional. First thing
on the box: smoke the mix arm's fused path (couldn't run locally on ROCm).
F1/F2 deferred per scope decision.

Session plan:
1. Data prep (prep tmux): data-memory, ultrachat train.pt, split chains.
2. Smoke `MODEL_NAME=mamba2_780m_memory_mix` short fused-path run.
3. BX0: train `mamba2_780m_memory_state` from scratch, split recipe,
   batch scaled to card; probe LongAlign gist config every few hundred M
   tokens, BOTH ablation modes (fresh-m + none).
4. BX1 mix@16, BX2 state@16 — identical args.

## Log

### 01:48 UTC — boot

- Monitor armed (persistent, HB=300). Watchdog-delay touched.
- No data, no checkpoints (expected — fresh box; 780M cells train from
  scratch, no checkpoint restore needed).

### 01:52 UTC — data prep launched (prep tmux)

- First launch failed instantly: fresh `sft/.env` ships `MODEL_NAME=` empty →
  `ModuleNotFoundError: No module named 'models.'`. Fixed by setting
  `MODEL_NAME=mamba2_780m_memory_state` in `sft/.env` (session default =
  the BX0 arm; mix runs will pass MODEL_NAME explicitly).
- Verbatim, chained in `prep` tmux (reproduces prior run's pool exactly),
  teed to `logs/prep-20260724-boot.log`:
  1. `make data-memory`
  2. `make prepare ARGS="--hf-dataset HuggingFaceH4/ultrachat_200k --max-examples 20000 --max-len 1024 --output data/train.pt"`
  3. `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7 --split-episode-rate 0.15 --split-qa-rate 0.9 --split-gap-min 1 --split-gap-max 4 --output data/train_chains_split.pt"`
     (verify: ~5498 splits, ~71% single-QA, ~15857 sleeps)
  3. `make prepare-chains ARGS="--cross-sleep-bias 0.75 --seed 7"` → train_chains.pt (xs pool, for optional erosion-contrast leg)

### 01:56 UTC — mix-arm fused-path smoke launched (train tmux, parallel with prep)

- Gating check for BX1 (fused mix hook never ran on real CUDA; local ROCm box
  can't). Verbatim:
  `MODEL_NAME=mamba2_780m_memory_mix make smoke-test ARGS="--batch-size 32 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --eos-weight 32 --length 385"`
  → `logs/smoke-mix-20260724.log`. Args = the planned BX constants (batch 32
  scaled to the GH200 per the discussion note; chunk/window/eos from the
  2.7B-proven recipe; accum-tokens 1536 = one micro-step at batch 32,
  keeping tokens-per-update identical to 2.7B). `--length 385` = 8 chunks so
  the run crosses multiple window boundaries / write events, not one chunk.
- First attempt crashed at call time: train.py's `run_training` grew a
  `train_sleeps` positional (sleep_positions support) and smoke_test.py was
  never updated. Fixed (second `[None]*len(train_ids)` placeholder),
  committed+pushed f312441, relaunched same command.
- Second stale-smoke crash: run_args Namespace missing `lr`/`warmup_steps`
  (run_training reads both for its warmup schedule). Added (lr from the
  smoke CLI, warmup_steps=0), committed+pushed 3973d6a, relaunched.
  smoke_test.py had clearly not been run since run_training's signature
  last changed; no further drift expected (checked run_training's full
  args-attribute usage against the Namespace).

### 01:53 UTC — mix smoke PASSED; data prep DONE and verified

- Mix-arm fused-path smoke (third launch) PASSED: one optimizer step at
  batch 32 / chunk 48 / window 8 / length 385; "LoRA params received
  gradients: True (192), other trainable params: True (14)". Mid-run
  metrics sane (beta 0.094, surprise 0.26, o_t_norm 10.5, finite grads).
  **BX1 unblocked — the fused mix hook works on real CUDA.**
- Data prep verified against prior pool: train_chains_split.pt = 2753
  chains, 230.9M tokens, 15857 sleeps, 5498 split-tails (3918 = 71%
  single-QA) — exact match to EXPERIMENT_NOTES-20260723. train_chains.pt
  (xs), train.pt, train_memory*.pt all present.
- Launched the same smoke for the STATE arm (insurance — its fused span
  dispatch at 780M indices never ran on CUDA either):
  `MODEL_NAME=mamba2_780m_memory_state make smoke-test ARGS="--batch-size 32 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --eos-weight 32 --length 385"`
  → `logs/smoke-state-20260724.log`.

### 01:56 UTC — state smoke OOM at batch 32; stepping down

- State arm OOM'd at batch 32 (93 GiB allocated, first chunk) where mix
  passed — the 16 injection modules + per-span gradient state cost far more
  than mix's single module. The screen's constants must fit the STATE arm
  (identical args across all three cells), so batch scales to its ceiling.
- Retrying batch 16 (accum-tokens 1536 → 2 micro-steps, tokens/update
  unchanged): `MODEL_NAME=mamba2_780m_memory_state make smoke-test
  ARGS="--batch-size 16 --chunk-len 48 --memory-window 8 --accum-tokens
  1536 --eos-weight 32 --length 385"` → `logs/smoke-state-b16-20260724.log`.

### 01:58 UTC — smoke ladder done; screen constants LOCKED; BX0 LAUNCHED

- State smoke: batch 16 PASSED, batch 24 PASSED (~74 GiB peak sampled),
  batch 32 OOM. Both passes: LoRA 192 + other 168 params all got grads.
- **Constants locked for all three cells: batch 16 / chunk 48 / window 8 /
  accum-tokens 1536 / eos-weight 32.** Batch 16 over 24 because
  16×48=768 → exactly 2 micro-steps = 1536 tokens/update, identical to the
  2.7B-proven recipe (24 would round accum up to 2304, changing
  optimizer-dynamics-per-token); and it leaves ~35+ GiB headroom so probes
  can run alongside training as the plan assumes. Discussion's "expect
  batch 32+" was wrong for the state arm on 96 GB — the 16 injection
  modules' per-span gradient state dominates; noted for the team.
- **BX0 launched (train tmux), from scratch** — verbatim:
  `MODEL_NAME=mamba2_780m_memory_state make train ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 16 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 147456"`
  → `logs/train-bx0-20260724.log`. Resume point: N/A (starting fresh —
  record the "starting fresh" line + first alpha/w1_abs_max/o_t_norm when
  they appear).
- CORRECTION to my batch-16 rationale above (semantics checked against
  EXPERIMENT_NOTES-20260720 line ~150): `--accum-tokens` is PER SLOT, so
  tokens/update = 1536 × batch — 24,576 at batch 16 vs 12,288 at 2.7B
  batch 8. Tokens/update scales with batch under any accum setting, so no
  batch choice preserves the 2.7B constant; 16 is still the pick (state-arm
  VRAM ceiling + probe headroom), just not for the exact-match reason
  stated. Batch 16 = 2.7B's later batch-12 tune scaled ~1.3×, same order.

### 02:00 UTC — BX0 start state (fresh confirmed)

- Log header: GH200 94.5 GB free, `state-spaces/mamba2-780m`, 96 LoRA
  adapters, read_layer 32, trainable 26,514,467, 2753 chains, 1 epoch,
  batch 16. Fresh start (no resume line).
- First-chunk metrics: avg_loss 3.2040; beta ~0.081, retain ~0.982,
  active 0/16, surprise 0.3–0.65, o_t_norm ~10.5, grad_norm 4–8,
  ssm_norm 250–380. All sane for step 0.

### 02:03 UTC — throughput measured (owner asked)

- 88,320 tokens in 184 s (per-slot 0→5520 from the 01:59:43 first-loss
  line) ≈ **480 tok/s** at batch 16. 2.7B reference: ~60 s/step ×
  12,288 tokens (batch 8) ≈ **205 tok/s** → **~2.3× faster end-to-end**.
- Per-CHUNK wall time is nearly unchanged (1.6 s vs 1.875 s despite 2×
  rows and 3.5× fewer params) — confirms the sequential per-window ops
  dominate step time and model size is almost free; the speedup is
  essentially all from the bigger batch. Corollary: batch 24 would have
  been ~3.4×; accepted cost of the VRAM-safety/probe-headroom choice.
- Horizon math: 2.7B xs transient peak was ~+41 steps ≈ 0.5M tokens;
  at 480 tok/s that's ~20 min of training — but split-data trajectories
  build slower (plateau regime), so first probe at ~2–3M tokens
  (~1.5 h), then every ~2M.

### 02:12 UTC — parallel-cells question (owner) — decision: sequential

- Owner asked about running two cells in parallel. GPU util is 23%
  (latency-bound sequential window ops) so compute-wise it would work;
  BX0 sits at 51.8 GiB leaving ~42. Risk: shared-VRAM CUDA OOM can kill
  the OTHER process — a parallel BX1 could dirty the BX0 calibration cell
  hours in. Decision (owner concurred): strictly sequential.
- Owner also ok'd dropping batch toward 2.7B's 8 if needed; not needed —
  16 is stable so far.

### 02:17 UTC — harness-validation probe launched (prep tmux)

- ~463k tokens @ ~483 tok/s confirmed. First probe-of-record still planned
  ~2.5M tokens (~03:25 UTC), then every ~2M.
- probe_recall.py has never executed against a 780M arm checkpoint (same
  stale-code risk smoke_test just demonstrated twice), so running a
  THROWAWAY validation probe now on step-18 — numbers meaningless, purpose
  is (a) tooling works end-to-end incl. the new --ablation none path,
  (b) real probe VRAM footprint measured alongside training. Verbatim
  (prep tmux, both modes chained, CKPT=.../epoch-1/step-18):
  `make probe-recall ARGS="--gist data/train_memory_longalign.pt --gist-prefix 6144 --gist-cont 512 --gist-recent 576 --gist-distractor 1536 --n-probes 16 --seed 1234 --ablation fresh-m --checkpoint $CKPT"`
  then the same with `--ablation none`.

### 02:25 UTC — validation probe DONE: harness works; instrument baseline

- Both modes ran clean on the 780M state arm (360/360 trainable tensors
  loaded; logs `probe-20260724-021625.log` fresh-m, `-022103.log` none).
  Probe footprint 20.3 GiB; combined with training 72/94.5 GiB — probes
  coexist with training as the plan assumed (and a parallel second cell
  would NOT have left room; sequential decision vindicated).
- step-18 (~0.4M tokens, throwaway numbers, expected negative — memory
  is untrained noise the backbone hasn't learned to ignore or use):
  - fresh-m: gist-delta -0.0377 (SEM 0.0045), recency -0.0009,
    dist-delta -0.0369, flush -0.0009.
  - none: gist-delta -0.0505 (SEM 0.0046), recency -0.0195,
    dist-delta -0.0603, flush +0.0098.
  - Note none < fresh-m here (plain backbone beats intact-M by MORE than
    the fresh-m control does): at this stage injections are pure noise, so
    "none" is the cleaner floor, consistent with the debrief's junk-
    injection reading of fresh-m. Useful baseline pair for later probes.

### 02:57 UTC — probe @ step-51 (~1.6M tokens; owner-requested early look)

- Same config/commands as the validation pair, CKPT=epoch-1/step-51.
  Logs: `probe-20260724-024904.log` (fresh-m), `-025337.log` (none).
- fresh-m: gist-delta -0.0010 (SEM 0.0021), recency -0.0022,
  dist-delta -0.0004, flush -0.0006. Thirds: +0.0017 / -0.0027 / -0.0019;
  long-range +0.0070 / -0.0018 / -0.0016.
- none: gist-delta -0.0103 (SEM 0.0023), recency -0.0096,
  dist-delta -0.0094, flush -0.0009. Thirds: -0.0066/-0.0101/-0.0141.
- Read: harm has washed out — both modes moved sharply toward zero from
  step-18 (-0.038→-0.001 fresh-m, -0.051→-0.010 none). Injections are no
  longer net noise; nothing positive yet (expected this early). none-mode
  recency also negative (-0.0096) suggests intact runs still slightly
  hurt by injections vs plain backbone across the board, shrinking.
  Trajectory direction is right; next probe ~3.5–4M tokens.

### 04:05 UTC — probe @ step-119 (~2.9M tokens)

- OPS NOTE first: the intended 03:50 launch NEVER STARTED — a stray `w`
  left on the prep prompt (visible in an earlier pane tail; typed into the
  attached pane, not by me) turned `cd` into `wcd`, instant
  command-not-found, and the monitor can't see a launch that produces no
  process transition. Owner caught it. New rule for this run: after every
  tmux send-keys, capture the pane and confirm the process is up before
  reporting it running. Actual probe launched 03:55 (C-c/C-u first),
  logs `probe-20260724-035539.log` (fresh-m), `-040007.log` (none).
- fresh-m: gist-delta +0.0013 (SEM 0.0025), recency +0.0014,
  dist-delta +0.0004, flush +0.0009. Thirds +0.0058/+0.0007/-0.0025.
- none: gist-delta -0.0080 (SEM 0.0024), recency -0.0079,
  dist-delta -0.0070, flush -0.0011. Thirds -0.0024/-0.0076/-0.0141.
- Read: fresh-m crossed zero (statistically ~0, not signal yet). none-mode
  stuck at ~-0.008 with recency matching gist-delta exactly — the residual
  intact-vs-backbone penalty is uniform, not memory-shaped; the modes now
  DISAGREE by ~+0.009: injections still cost vs no-injection, while
  intact-M beats junk-M. Watch whether none-mode follows fresh-m toward
  zero over the next probes; if it never does, the arm is learning to
  tolerate its own injections rather than profit from them — that pattern
  across the whole leg would be a real (negative-ish) BX0 shape.

### 04:25 UTC — batch 16 → 24 mid-leg (owner call); plan compressed to fit window

- Owner wants all three cells inside ~7–8 h and pushed back on my restart
  assumption — mid-leg batch change via resume is fine (2.7B precedent:
  the 8→12 tune). Stopped BX0 at ~step-140, resumed batch 24. Verbatim:
  `MODEL_NAME=mamba2_780m_memory_state make resume ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 24 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 147456"`
  → `logs/train-bx0-b24-20260724.log`. Resume point: step-140, epoch 1,
  next_ptr 49; mem_state batch mismatch → slot states re-seeded from
  example starts (expected; weights/optimizer/token count carry).
- Consequences accepted & noted: tokens/update now 36,864 (was 24,576) —
  mid-leg optimizer-clock change, BX1/BX2 will run batch 24 THROUGHOUT so
  the first ~3.4M of BX0 is the mismatched stretch; probes move to
  END-OF-CELL parallel sweeps over banked checkpoints (74 GiB @ b24 + 20
  GiB probe doesn't fit); OOM risk covered by monitor (VRAM now in
  heartbeat, HB=300 for the post-change hour) + resume-from-checkpoint.
- MUST DO before BX2: archive BX0's checkpoints out of
  `mamba2_780m_memory_state/checkpoints/epoch-1/` (BX2 writes there too);
  also protect matched-point checkpoints from rotation for post-hoc probes.
- Cell budget: ~6M tokens each (readable-verdict anchor from 2.7B clock),
  end-of-cell probe sweeps at matched checkpoints ~1.6M/2.9M/4.4M/6M.
- Batch-24 measured: **648 tok/s** (per-slot 384→6864 in 240 s), 1.34× over
  batch 16's 483. VRAM 75.1 GiB steady, ~19.5 GiB headroom. ckpts now every
  4 steps. Schedule: BX0 cut at step-212 (~6.09M) ~05:42; BX1 ~05:45→08:20;
  BX2 ~08:25→10:55; final sweep ~11:20. Owner asleep as of ~04:40 —
  autonomous from here per the brief.

### 05:18 UTC — BX0 CELL COMPLETE at step-212 (~6.09M tokens)

- Effective rate ~590 tok/s (slightly under the 648 clean measure); cut
  slipped to 05:18. Stopped training, step-212 verified intact (incl.
  mem_state.pt), **checkpoints archived to
  `mamba2_780m_memory_state/checkpoints/archive-bx0-state32/`** (protects
  from BX2 reuse of the folder; all probe paths now point at the archive).
- BX0 leg summary: fresh start → step-140 @ batch 16 (~3.44M tokens),
  step-140→212 @ batch 24 (owner-approved mid-leg change, 2.7B precedent).
  Healthy throughout: no non-finite, VRAM plateaued 77.3 GiB, beta ~0.5,
  retain ~0.98, o_t_norm mid-20s, sleeps firing.
- End-of-cell sweep (prep tmux, parallel): step-212 pair running
  (fresh-m + none); step-166 didn't exist (irregular ckpt numbering) →
  step-168 pair queued behind it. Logs: `logs/probe-bx0-step{168,212}-{fresh-m,none}.log`.
  Matched-point probes already done live: step-51 (~1.6M), step-119 (~2.9M).

### 05:40 UTC — BX0 step-212 (~6.09M) probe results

- fresh-m: gist-delta +0.0013 (SEM 0.0019) — flat at zero since step-119.
  recency -0.0028. ANOMALY: dist-delta +0.3265 (SEM 0.3253) — one
  conversation's distractor-ABLATED branch collapsed (fresh-m junk
  injection going OOD, the same instrument failure mode the debrief flagged
  at 2.7B prefix-6144); treat dist/flush at this point as unusable, the
  gist-delta itself looks clean (SEM normal).
- none: gist-delta -0.0090 (SEM 0.0025), recency -0.0144. Notable
  structure: long-range components POSITIVE in thirds 1–2 (+0.0076,
  +0.0083) while overall gist stays negative — the injection cost is
  recency-flavored, but a small genuine long-range contribution exists
  under the clean control.
- BX0 emerging verdict at 6M: **no clear positive ablation delta** (the
  screen signature did not emerge in the state@32 cell at this budget).
  fresh-m plateaued at ~0 from 2.9M→6.1M; none-mode penalty persists
  ~-0.009 (tolerance-not-profit pattern) BUT with real long-range
  positives in the none decomposition. Per standing direction: null BX0
  gates nothing; BX1/BX2 proceed. The none-mode long-range structure is
  the thing to compare against the mix arm.
- step-168 (~4.5M) pair confirms the plateau: fresh-m +0.0009 (SEM
  0.0024), none -0.0074 (SEM 0.0022) — same shape as 212, no anomaly.
  Full BX0 trajectory (fresh-m / none): 0.4M -0.038/-0.051 → 1.6M
  -0.001/-0.010 → 2.9M +0.001/-0.008 → 4.5M +0.001/-0.007 → 6.1M
  +0.001/-0.009. Clean monotone wash-out then hard plateau at ~0/-0.008.

### 05:34 UTC — BX1 LAUNCHED (mix@16, from scratch, batch 24 throughout)

- Verbatim (train tmux):
  `MODEL_NAME=mamba2_780m_memory_mix make train ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 24 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 147456"`
  → `logs/train-bx1-mix-20260724.log`. Fresh start; trainable 17,152,772
  (mix arm: 1 mix module vs state's 16 injection modules). Budget: 6M
  tokens ≈ step-163 at 36,864 tok/step. Note asymmetry vs BX0 (BX0's
  first 3.4M ran batch 16): accepted, batch is a throughput knob here.
- Cell-end watcher: step-164 (~6.05M).
- Mix VRAM 55.9 GiB → live matched-point probes restored for BX1
  (sequential, one 20 GiB probe at a time; MODEL_NAME=mamba2_780m_memory_mix
  passed explicitly — .env defaults to the state arm).
- Anneal transient: beta pegged 1.0 with o_t_norm spike to 44.8, settled
  back ~30 within 10 min. Later beta samples (n=200): min 0 / med 0.06 /
  p75 0.37 / max 0.98 — SELECTIVE surprise-gated writes, not a stuck gate.

### 05:54 UTC — BX1 probe @ step-43 (~1.6M) — NO harm phase in the mix arm

- fresh-m: gist-delta +0.0006 (SEM 0.0010); thirds +0.0019/-0.0001/-0.0000.
- none: gist-delta +0.0020 (SEM 0.0012, 1.7σ); recency -0.0001;
  thirds +0.0002/+0.0014/+0.0045 (3rd third 2.6σ); long-range positive in
  ALL thirds (+0.0026/+0.0013/+0.0025).
- Matched-budget contrast vs BX0 @1.6M (fresh-m/none): state -0.001/-0.010
  vs mix +0.001/+0.002. The state arm's early "injection tax" (and its
  slow wash-out) is ABSENT in the mix arm; clean control already at/above
  zero with late-third-dominant, long-range-positive structure. Early and
  small, but the sign pattern matches the integration-point hypothesis
  (zero-init residual mix has no harm phase to unlearn).
- Next: matched probe ~2.9M (step ~79), then cell end ~6M.

### 06:10 UTC — BX1 probe @ step-80 (~2.9M)

- fresh-m: gist-delta -0.0004 (SEM 0.0015). none: +0.0009 (SEM 0.0016);
  third-3 +0.0060 (SEM 0.0029, ~2σ) with long-range +0.0054 — the
  late-third long-range structure persists; headline hovers at zero
  (softer than step-43's +0.0020).
- Matched 2.9M contrast (none): state -0.008 vs mix +0.001. Mix still
  clear of the state arm's tax; no strong growth yet. Watch whether the
  late-third long-range component compounds by 4.4M/6M or stays flat.
- Added a ~4.4M matched probe (step ~120) for trajectory resolution.

### 06:28 UTC — BX1 probe @ step-120 (~4.4M) — late-third structure did NOT compound

- fresh-m: +0.0004 (SEM 0.0016). none: +0.0007 (SEM 0.0012);
  recency +0.0036 (2.3σ). Structure INVERTED vs 1.6M/2.9M: third-1 now
  strongest (+0.0049) with long-range NEGATIVE everywhere (-0.0044/
  -0.0027/-0.0017) — drift toward recency flavor, away from the
  long-range signature.
- BX1 none-mode trajectory: +0.0020 → +0.0009 → +0.0007 (flat-zero,
  softening), structure front-loading over time. o_t_norm meanwhile
  climbing 23→42 (bigger reads, not more useful ones).
- Interim A/B read at 4.4M: mix avoids the state arm's injection tax
  entirely but is NOT building the positive long-range delta either —
  both arms currently null on the actual screen signature. Cell-end 6M
  probe decides BX1's line in the table.

### 06:46 UTC — BX1 CELL COMPLETE at step-166 (~6.12M); END PROBE POSITIVE

- Stopped at step-166 (intact, incl. mem_state.pt), archived to
  `mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/`. End pair run
  in parallel post-stop. Logs `probe-bx1-step-166-{fresh-m,none}.log`.
- none: gist-delta **+0.0048 (SEM 0.0017, ~2.8σ)**; dist-delta +0.0059
  (~3.1σ — survives distractors); recency +0.0041; thirds
  +0.0092/+0.0027/+0.0024; long-range +0.0015/-0.0009/+0.0016.
- fresh-m: +0.0038 (SEM 0.0019, 2σ), same shape.
- Read: FIRST clearly-positive clean-control delta of the screen, still
  RISING at the cut (+0.0020→+0.0009→+0.0007→+0.0048). BUT recency-sized
  and front-loaded — not long-range-dominant; not (yet) the screen
  signature. 6M catches the mix arm mid-rise.
- Matched-6M A/B headline (none): state@32 -0.0090 vs mix@16 +0.0048 —
  ~0.014 separation at ~0.002 SEMs. On integration point, the mix arm
  wins decisively at this scale/budget.

### 06:51 UTC — BX2 LAUNCHED (state@16 via MEMORY_READ_LAYER=16)

- Verbatim (train tmux):
  `MODEL_NAME=mamba2_780m_memory_state MEMORY_READ_LAYER=16 make train ARGS="--data data/train_chains_split.pt --eos-weight 32 --batch-size 24 --chunk-len 48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 147456"`
  → `logs/train-bx2-state16-20260724.log`. **read_layer: 16 confirmed in
  the log header** (env override, not recorded in checkpoints — this note
  is the record). Fresh start, trainable 26,514,467, batch 24.
- State arm @ b24 = ~77 GiB → NO live probe headroom; BX2 matched points
  (steps ~43/80/120/end) probed post-hoc in the end sweep from banked
  checkpoints (keep-ckpts 50 covers the ~41 produced).

### 09:15 UTC — timestamp corrections + mix-arm SPEED finding

- All section timestamps 05:34→06:51 above were reconstructed from log
  timestamps / probe-log mtimes after my wall-clock labels drifted ~2 h
  ahead (twice this session; lesson recorded: stamp from `date -u`, not
  from a running sense of time).
- The drift exposed a real finding: **the mix arm trains ~2.3× faster
  than the state arm** — BX1 did 6.12M tokens in 69 min (~1478 tok/s) vs
  state's ~650 at the same batch/chunk. The state arm's 16 per-window
  injection modules dominate its step time; the mix arm's single
  boundary-16 hook (with fused 0–15 and per-window batched reads) is far
  cheaper. BX2 (state@16) runs ~694 tok/s — reading shallower helps the
  state arm only marginally. Cost-of-mechanism now favors mix on BOTH
  quality-at-matched-tokens AND wall-clock (~2.3×) at this scale.

### 09:25 UTC — BX2 CELL COMPLETE at step-166 (~6.12M); sweep OOM + retry

- BX2 stopped at step-166 (intact incl. mem_state.pt), archived to
  `archive-bx2-state16/`. Matched steps existed exactly (43/80/120/166 —
  same token cadence as BX1).
- Sweep lesson: 4 parallel STATE-arm probes OOM (state probe ≈ 24 GiB,
  not the mix arm's 20; 4×24 > 94.5) — 6 of 8 died, only step-80 fresh-m
  (-0.0038) and step-166 fresh-m (+0.0029) survived wave order. Retrying
  the 6 in waves of 3 (`BX2-R1-DONE`/`BX2-RETRY-DONE` markers), all with
  MEMORY_READ_LAYER=16 (verified in probe log headers).
- 3-wide ALSO OOM'd (1 survivor/wave — allocator contention under
  expandable_segments, not per-probe footprint: solo probes run 22.7 GiB).
  Reran the missing 5 SEQUENTIALLY — clean. Ops rule going forward: max
  2-wide probe parallelism.

### 10:40 UTC — FULL THREE-CELL TABLE; A/B VERDICT; BX1 EXTENSION LAUNCHED

None-mode (clean control) gist-delta at matched tokens:

| tokens | BX0 state@32 | BX1 mix@16 | BX2 state@16 |
|--------|--------------|------------|--------------|
| 1.6M   | -0.0103      | +0.0020    | -0.0162      |
| 2.9M   | -0.0080      | +0.0009    | -0.0077      |
| 4.4M   | -0.0074      | +0.0007    | -0.0141      |
| 6.1M   | -0.0090      | **+0.0048**| -0.0141      |

(fresh-m rows track the same ordering; BX2 fresh-m ends +0.0029 vs none
-0.0141 — the fresh-m control flatters the state mechanism, consistent
with the debrief's junk-injection critique. BX2 step-166 thirds show a
+0.0114 long-range FIRST third under none — front-loaded, recency-adjacent.)

Verdict (mechanism A/B — what the screen was for):
1. **Integration point is (part of) the ceiling; the LANDING MECHANISM is
   the payer.** Matched-front-end contrast (both read+land @16):
   mix +0.0048 (rising) vs state -0.0141 (worst cell). Read-depth
   contrast within the state mechanism (32 vs 16): shallower is equal-or-
   worse → "shallow front-end inadequate" is refuted as the blocker.
2. Mix never had a harm phase; state cells never escaped theirs at this
   budget. Plus mix trains ~2.3× faster (structural, not incidental).
3. NOT yet shown: long-range-dominant structure (mix's positive is
   recency-flavored; screen signature incomplete). NOT SCREEN DEAD: a
   real positive emerged, so 780M can express the effect.
4. Per the discussion's decision rule ("token-mix clearly better → port
   to 2.7B and re-run the harness A/B there") the port is now funded on
   this evidence; the CONDITIONAL "pooled mix" attribution cell is also
   now live (mix won).

- **BX1 EXTENSION launched** (the one cheap open question: was 6M a
  mid-rise cut? does long-range structure emerge?): resumed step-166,
  verbatim: `MODEL_NAME=mamba2_780m_memory_mix make resume ARGS="--data
  data/train_chains_split.pt --eos-weight 32 --batch-size 24 --chunk-len
  48 --memory-window 8 --accum-tokens 1536 --ckpt-every-tokens 147456"`
  → `logs/train-bx1ext-mix-20260724.log`. Resume point: step-166, "loaded
  saved internal state" (slots continue exactly). Target ~step-330
  (~12.2M, ~70 min); end pair probe after.
  (Housekeeping: `archive-bx1-mix16` temporarily moved back to `epoch-1`
  for the resume; re-archive at cut.)

### 12:05 UTC — BX1 EXTENSION RESULT (step-333, ~12.2M) + RUN CLOSE

- Stopped at step-333 (intact), re-archived to `archive-bx1-mix16/`
  (now holds the full 0→12.2M lineage). Probe pair (2-wide):
  `probe-bx1-step-333-{fresh-m,none}.log`.
- none: gist-delta **+0.0037 (SEM 0.0012, ~3.1σ)**; dist-delta +0.0037
  (flush cost 0.0000 — fully distractor-proof); recency +0.0036; thirds
  +0.0114/+0.0003/-0.0006; long-range ~0 (+0.0004/+0.0010/-0.0012).
- fresh-m: +0.0031 (SEM 0.0012), same shape.
- Read: the 6M positive was a PLATEAU ARRIVAL, not mid-rise — doubling
  the budget held the delta (+0.0048→+0.0037, within noise) but
  long-range-dominant structure did NOT emerge; the mix arm's gain at
  780M is a robust, distractor-proof, recency-flavored gist benefit.

## Run close — what this run established

1. **Screen platform works.** 780M + split recipe + LongAlign probe
   harness reproduces interpretable memory dynamics at 4–6× the 2.7B
   experiment rate; both ablation modes ran; instrument anomalies
   (fresh-m OOD collapse) reappear here exactly as the debrief predicted,
   validating the `--ablation none` addition.
2. **Integration A/B: mix@16 decisively beats state@32 and state@16** on
   clean-control gist-delta at every matched budget (table at 10:40), has
   no harm phase, and trains ~2.3× faster. Matched-front-end contrast
   pins the win on the LANDING MECHANISM (transient per-token residual
   add vs persistent gated-delta injection); read-depth contrast refutes
   shallow-front-end-inadequacy. Per the standing decision rule, the
   2.7B token-mix port is funded; the "pooled mix" attribution cell is
   now live.
3. **Not established: the long-range signature.** No cell produced
   long-range-dominant structure by 6M (mix) / 12.2M (mix extension).
   NOT SCREEN DEAD (a real 3σ positive emerged) — but the three-tier
   goal's defining regime remains undemonstrated at 780M. Open question
   for the team: is long-range structure scale-gated (2.7B showed it),
   data-gated (more/longer split chains), or mechanism-gated?
4. Ops lessons: smoke_test drift (2 fixes pushed: f312441, 3973d6a);
   verify-after-send rule for tmux; max 2-wide probe parallelism;
   timestamps from `date -u` only; state-arm b32 OOM / b24=77 GiB /
   b16=52 GiB; accum-tokens is per-slot.

Discussion items for the team: (a) port token-mix to 2.7B and rerun the
harness A/B there (decision-rule-funded); (b) pooled-mix cell at 780M
(cheap, mix-arm speed); (c) chase long-range: longer/denser split chains
at 780M vs going straight to 2.7B; (d) F1/F2 (deferred 2.7B probe-only
items) remain open.

Artifacts: checkpoints `archive-bx0-state32/` (…step-212),
`archive-bx1-mix16/` (…step-333, full lineage), `archive-bx2-state16/`
(…step-166); all probe logs `sft/logs/probe-bx{0,1,2}-*`; train logs
`train-bx0*`, `train-bx1*`, `train-bx2*`; data regen commands at 01:52.
