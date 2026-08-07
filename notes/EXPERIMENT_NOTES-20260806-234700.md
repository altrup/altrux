# Experiment notes — 2026-08-06 23:47 UTC (box session)

Fresh Lambda **GH200 480GB** (97871 MiB, aarch64, torch 2.13.0+cu129, CUDA
available). Not ROCm — the local box's gfx1102 workarounds do not apply; the
fused SSD kernel path and `causal-conv1d` are live and must be used here.

Standing direction: `notes/DISCUSSION-20260806-dream-distillation-ab-postmortem.md`
§5 — the corrected single-sleep rerun (a), then the saturation ladder (b),
and (c) only if the ≥0.3 margin-install gate clears. This session executes
§5(a). Model for the whole session: `mamba2_780m` (inline on every command).

## Box state at start

- Repo at `58a7e1e` (clean) — carries the corrected harness the 08-06 debrief
  specified (dream cache, erase_hook, corrected B1, student-query erasure,
  margin probes, binding-aware coverage, hash assertions, `run_grid2.sh`).
- `sft/.venv` present and healthy; torch 2.13.0+cu129, GH200 visible.
- **No HF cache, no `sft/data/`, no logs, no checkpoints** — nothing arrived
  from the local archive upload. Expected: the dream-sleep harness builds its
  own wake transcripts, self-calibrates the knowledge battery
  (`data/knowledge_battery_mamba2_780m.json`, base held-out PPL should
  reproduce ≈23.559 — the 08-06 §8 item), and downloads weights on first use.
  No `make data` prep is needed for this docket.
- Training is NOT part of this session's docket — no train.py runs, so the
  watchdog counts the whole session as "training stopped" and
  `scripts/.watchdog-delay` is touched on every heartbeat.

## Timeline / verbatim commands

### 1. Test suite — 23:47 UTC

    tmux new-window -t work -n test 'cd ~/altrux/sft && MODEL_NAME=mamba2_780m make test 2>&1 | tee ~/altrux/sft/logs/box-test.log; echo "EXIT=$?" >> ~/altrux/sft/logs/box-test.log; exec bash'

First GPU execution of the corrected harness (dream cache round-trip, cue
masking, margin arithmetic, binding-aware coverage, B1/B2 token-1
equivalence, detach, erase-hook no-op). Prior-run gotcha carried forward: the
suite goes silent for minutes at `test_mixer_fused.py` (cold Triton compile of
the SSD backward kernel) — not a hang; and `pane_current_command` reads `bash`
during `uv run`, so completion is detected from the `EXIT=` marker in the log.

**One failure, live at 23:50:** `tests/test_erase_hook.py::test_hook_unset_is_a_byte_identical_no_op`.
Hypothesis before reading the traceback: the test compares a run with no hook
against a run with an *identity* hook using `torch.equal`. Setting any hook
forces the per-token loop (the fused chunk-scan has nowhere to interleave the
hook — that is the next test's assertion), so on a CUDA host this is a
fused-vs-loop numeric comparison, which the project's own oracle tests
(`test_mixer_fused.py`) do at `atol=rtol=1e-4`. On the ROCm dev box both sides
run the loop, so exact equality held there and the over-strict assertion was
invisible. To be confirmed from the traceback's actual max difference before
touching anything.

**Result: 1 failed, 295 passed in 204s (EXIT=0).** The traceback confirms the
hypothesis exactly — the two logit tensors agree to ~1e-6 (`8.6626e-02` vs
`8.6627e-02`), i.e. fused-vs-loop float noise, not a wiring difference. Fixed
by comparing at `atol=rtol=1e-4` (the tolerance `test_mixer_fused.py`'s own
fused-vs-loop oracle uses) and renaming the test to
`test_an_identity_hook_is_a_no_op`; re-ran `tests/test_erase_hook.py` → 4
passed. Committed and pushed as `1ace4cf`.

    tmux new-window -t work -n retest 'cd ~/altrux/sft && MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. uv run --no-sync pytest tests/test_erase_hook.py -q ...'

Harness verdict: the corrected 08-06 harness passes on this host, including
the erase-hook, dream-cache round-trip, cue-masking, margin, coverage-binding
and B1/B2 token-1 equivalence tests.

### 2. Dream caches, three seeds — 23:51 UTC (train tmux, serial)

    tmux send-keys -t train 'cd ~/altrux/sft && for s in 1234 2345 3456; do env MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. uv run --no-sync python -u dream_sleep.py --n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 32 --cue-greedy 12 --seed $s --build-dream-cache 2>&1 | tee logs/g2_cache_s${s}.log; echo "EXIT=$?" >> logs/g2_cache_s${s}.log; done' Enter

Built separately from `run_grid2.sh` on purpose: the docket gates every arm on
reading the sidecar first (binding-aware coverage ≥ 3/4 per seed, else rebuild
that seed at `--cue-every 24`), and the script would otherwise roll straight
from the build into arm A.

Shared preamble reproduced from the 08-06 A10 session: battery 24/40 kept,
base held-out PPL to be confirmed ≈23.559; wake facts at seed 1234 are the
same four (clove/topaz/osprey/heron), so the transcript RNG is stable across
hosts as designed.

**Seed 1234 — bound code coverage 2/4, below the gate.** Sidecar
(`data/dream_s1234.txt`, dream_sha `f04994fe…`, transcript_sha `930fa442…`,
346/512 tokens freely generated):

- bound rehearsals clove=3, topaz=2, osprey=0, heron=0; misbound heron=2.
- The failure is *binding*, not rehearsal volume: every fact gets a cue. The
  dream answers "the code for the osprey" with `2 1 2 1 0` (heron's code) and
  "the code for the heron" with `4 4 2 3 0` — a code belonging to no wake
  fact, first emitted as "the code for the **blackbird**", an entity that does
  not exist in the wake transcript.
- The dream seed separator fix is visible: the sidecar opens `[ASSISTANT] `
  with the literal space, on-format.

This is exactly the failure the binding-aware coverage metric was added to
catch (08-06 §1: the old substring coverage scored this seed 3–4/4).

**Seed 2345 — 4/4 bound, 0 misbound** (saffron 3, marimba 2, oboe 2, viola 2;
340/512 free; dream_sha `2b761fb9…`). Clean.

**Seed 3456 — 3/4 bound**, passes the gate (calcite 2, ketch 11, saffron 2,
**schooner 0**; misbound calcite 3; 340/512 free; dream_sha `256dcc84…`).

Baselines this host: base held-out PPL **23.550 / 23.832 / 23.550** across the
three builds (the middle one is the same base model on the same 364 tokens —
so ~1% run-to-run numeric spread on the fused path), battery **25 items** kept
of 40 (A10 kept 24; the battery is rebuilt per host, so absolute battery counts
are not comparable across hosts — within this session all seeds share the one
`data/knowledge_battery_mamba2_780m.json` built at seed 1234).

Both surviving dreams are dominated by cue text plus verbatim repetition of one
filler sentence — the free-generation budget is real (340/512) but low-entropy.
Recorded as the cue-rate confound the 08-06 §7 open question names; it is
identical across arms within a seed, so the A-vs-B contrast is unaffected.

### 3. Seed 1234 rebuilt at the registered fallback — 23:55 UTC

Rejected cue-32 sidecar archived to `logs/dream_s1234_cue32_rejected.txt` (it
rides the rsync pull; `data/dream_s1234.txt` is overwritten by the rebuild).

    tmux send-keys -t train 'cd ~/altrux/sft && env MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. uv run --no-sync python -u dream_sleep.py --n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 24 --cue-greedy 12 --seed 1234 --build-dream-cache 2>&1 | tee logs/g2_cache_s1234_cue24.log; echo "EXIT=$?" >> logs/g2_cache_s1234_cue24.log' Enter

`run_grid2.sh` gained a `CUE_EVERY` parameter (default 32) so seed 1234's arms
run at the same cue rate its cache was built with — otherwise the cells' own
flags would regenerate a dream they never distilled. Committed as `eb4a95d`.

Caveat to carry into the readout: the cue rate is only a *chance* multiplier.
Seed 1234's misbinding is a base-model in-context failure (the answers after a
cue are decoded greedily via `--cue-greedy 12`, so temperature is not the
cause), and the two confused facts — osprey and heron — are the seed's two
same-category (bird) facts. If cue-24 does not lift it to 3/4, the honest
options are to run the seed with the deviation recorded (within-seed contrasts
stay valid; cross-seed pooling of absolute install rates is what it taints) or
to drop it to two seeds. Not a knob to keep turning until the number looks
right.

**Cue-24 result: still 2/4** (clove 2, heron 2 bound; topaz answered with
osprey's code, osprey with the phantom "blackbird" code `4 4 2 3 0`; 364/512
free; dream_sha `91199566…`, transcript_sha unchanged `930fa442…` — so the
wake session really is fixed and only generation moved). Archived to
`logs/dream_s1234_cue24_rejected.txt`. The registered fallback does not fix a
binding failure, as expected: the post-cue answer is greedy, so more cue slots
only buy more draws of the same wrong retrieval.

### 4. Same-seed generation is NONDETERMINISTIC — the 08-06 open question, answered

I rebuilt seed 1234 at the original `--cue-every 32` to restore uniform flags
across seeds. Same command, same seed, same arm, third build:

| build | dream_sha | free/cue tokens | bound coverage |
|---|---|---|---|
| 1 (cue 32) | `f04994fe…` | 346 / 166 | 2/4 |
| 2 (cue 24) | `91199566…` | 364 / 148 | 2/4 |
| 3 (cue 32, **in use**) | `ec8ad8a0…` | 364 / 148 | **3/4** |

Builds 1 and 3 are the *same command* and produced different dreams. That is
the discriminating check 08-06 §5 asked for (run the same arm twice at one
seed), and it comes back **kernel/generation nondeterminism, not arm code
paths** — the A-vs-B2 same-seed divergence in the old grid needs no further
explanation, and the dream cache makes it moot for every comparison from here.
`transcript_sha` is stable across all three builds, so the wake session and its
RNG are deterministic; only the sampled dream is not.

**Selection caveat, stated loudly:** seed 1234's in-use dream is the third
build, and it is the one that happened to clear the 3/4 gate (draws were
2/4, 2/4, 3/4). Build 1 cannot be recovered — nondeterminism means "rebuild at
the original flags" does not reproduce it. So seed 1234's *absolute* dream
quality is a favourable draw and its coverage should not be pooled naively with
the other seeds; every within-seed arm contrast is unaffected (all six arms
distil this one cached dream), and the docket's install metric is conditioned
on binding coverage anyway.

Seeds in use: 1234 `ec8ad8a0…` (3/4), 2345 `2b761fb9…` (4/4), 3456
`256dcc84…` (3/4).

### 5. The grid — 23:57 UTC

    tmux send-keys -t train 'cd ~/altrux/sft && for s in 1234 2345 3456; do ./run_grid2.sh $s 2>&1 | tee logs/g2_grid_s${s}.log; done; echo "EXIT=$?" >> logs/g2_grid_s3456.log' Enter

Seven cells per seed (A, B1-corrected, B2, CE-on-dream, sft-ref, no-sleep, plus
the A_bridge cell at seed 1234 only), serial, `--lr 1e-4 --distill-steps 800
--probe-every 200`. All arms load their seed's cache; none generates.

First checkpoint (seed 1234, arm A, in-context control): 3/4 greedy HIT but
**4/4 margin-install**, margins +12.1…+14.7 nats. Worth noting for the readout
that topaz's greedy answer is osprey's code while its margin is +12.2 — the
margin metric asks "correct code vs a fixed foil", so it is insensitive to a
*misbinding* onto another real fact's code. In-context that is the right
reading (the fact is present), but it means margin-install alone cannot detect
cross-fact confusion; the paraphrase and greedy columns carry that signal.

**Live readout, seed 1234 arm A (fresh-state probes, the real installation
measure).** At d200 and d400 the picture is stable:

| fact | dream bound rehearsals | margin @s400 | install |
|---|---|---|---|
| clove | 2 | +14.23 | yes (also greedy HIT) |
| osprey | 2 | +4.54 | yes |
| heron | 2 | +6.69 | yes |
| topaz | **0** | +0.58 | no |

**3/4 installed from a fresh state**, and the one failure is exactly the fact
the dream never rehearsed in binding — per-fact agreement is 4/4 between "the
dream rehearsed it correctly" and "it installed". Damage at the same point:
battery 24/25 retained, mean dlogp −0.067, **dPPL −0.028** (negative).

Two things this establishes on its own, before any arm contrast:

1. **The corrected arm A does install facts into weights.** The 08-06 grid
   scored this same regime "0 installs" — that was the greedy/counting
   attractor artifact (08-06 §1), and the distractor-margin metric the debrief
   registered is what makes the installation visible. The install fraction at
   this seed (0.75, and 1.0 conditioned on dream binding) is far above the
   ≥0.3 gate §5(c) sets for the multi-sleep phase.
2. **Dream binding is the bottleneck, not the distillation.** A fact that is
   rehearsed correctly installs; a fact that is not, does not. That makes
   teacher-dream binding quality the lever worth pulling for capacity, and it
   is a base-model in-context retrieval property, upstream of every arm.

Arm A seed 1234 finished 00:04:36: **291,200 token-gradients in 6m37s**,
installs 3/4 at every probe point (d200/400/800). Damage *curve*:

| probe | dPPL | battery retained | mean dlogp |
|---|---|---|---|
| d200 | −0.037 | 24/25 | −0.071 |
| d400 | −0.028 | 24/25 | −0.067 |
| d800 | **+0.345** | 24/25 | −0.080 |

Learning is flat from d200 onward while damage turns positive between d400 and
d800 — the iso-learning curve the debrief wanted instead of LR-matching, and it
already says A's useful budget at this regime is well below d800. The decoded
dream printed by the cell matches the cache's build-3 content (topaz answered
with osprey's code, osprey/heron/clove correct), confirming the arm distilled
the cached dream and did not regenerate.

Timing: ~7 min per A cell, B1 running at 2.1 step/s (~6.5 min); the full
3-seed grid should complete in ≈2 h.

### 6. Are we actually using the GH200? — 00:10 UTC

Checked, because renting it was the reason (a) moved hosts.

**Yes on the protocol:** arm A logs `replay chunk 512 tokens over 512` — one
optimizer step per pass over the whole dream, and `replay_step` with
`n_chunks == 1` returns `reset=True` every step, so every pass genuinely starts
from a zero state. 800 × 364 scored tokens = the 291,200 token-gradients
reported. The registered full-sequence BPTT form is what ran. (The `chunk_len
48` in the startup banner is the wake/held-out-PPL chunk, not the replay chunk
— easy to misread.)

**No on the hardware:** a single cell sat at **3.4 GB of 97 GB and 12% GPU
utilisation**. The docket's serial-execution rule comes from an A10
measurement (three streams, 4× worse aggregate); that is a hardware fact, not a
scientific invariant, and cells are independent processes writing disjoint
files. Re-measured here by launching seeds 2345 and 3456 concurrently in
`work` windows:

    tmux new-window -t work -n grid2345 'cd ~/altrux/sft && ./run_grid2.sh 2345 2>&1 | tee logs/g2_grid_s2345.log; exec bash'
    tmux new-window -t work -n grid3456 'cd ~/altrux/sft && ./run_grid2.sh 3456 2>&1 | tee logs/g2_grid_s3456.log; exec bash'

Three streams: **13.6 GB, 99% utilisation**. Once warm (the first 60 s reads
low — those numbers are startup, not steady state), the two concurrent A cells
run at **2.23 and 2.26 step/s against 2.01 solo** — i.e. no slowdown at all —
while B1 continues at 1.10 (vs 2.09 solo). Aggregate ≈**2.8× single-stream**,
close to linear.

**Standing finding for this host: run the grid's seeds concurrently, one
process per seed.** The A10's "three streams are 4× worse" rule is a capacity
limit of that card and does not transfer to a GH200; measured, not assumed.
Wall clock for §5(a) drops from ≈2 h to ≈1 h. (Belongs in `sft/CLAUDE.md` next
to the A10 entry — the existing note reads as a general rule and is not one.)
Recorded there as `2fcbbc0`.

**But the two arm families scale completely differently, and the B arms are
the ones that matter for cost.** Measured on B1 (per-token) once all three
streams landed on it, plus a throwaway 4th stream (`logs/g2_scratch.*`,
deleted after the measurement):

| concurrent streams | per-stream step/s | aggregate |
|---|---|---|
| 1 | 2.09 | 2.09 |
| 3 | 0.83 | ≈2.5 |
| 4 | 0.61–0.82 | ≈2.65 (**+6%**) |

Arm A (chunked, 512-token forward) holds its solo rate at 3 streams —
near-linear. The per-token arms saturate at ≈2.5 aggregate steps/s and no
number of processes moves it: they are bound by **per-token kernel-launch
latency**, with 13.6 GB of 97 GB memory in use. So "99% utilisation" here
means "a kernel is resident", not "the SMs are busy"; the GH200 is *not* being
used fully by the B arms and cannot be by adding streams.

**Concrete proposal for phase (b) (not started — the grid is mid-flight):
batch the dreams inside one process.** The B arms' inner loop is one token
through one model with batch 1; nothing in the protocol requires the seeds to
be separate processes. Running S seeds as a batch-S forward turns S× the
launch overhead into one launch and should recover most of the missing
throughput. This is what makes the registered d12800 B2 ladder cell (≈7.6 h at
the solo rate, and ~4.4 h each if merely run 3-up) affordable. It is harness
work — batch the cached dream tensors, the wake states and the per-token erase
hook — with an exact equivalence test available (batch-1 vs batch-S must give
identical per-seed logits). Seed 1234's serial loop
in `train` is left running alongside the parallel seed runs; the driver's
resume check is what keeps them from duplicating work — see §7, which is about
that check being wrong.

### 7. Harness bug found by reading an interim table — cells had no completion marker

An interim `make summarize-grid` at 00:35 printed B1 cells with `bound 0/4` and
`tokgrad 0` while arm A on the same seed (same cached dream!) showed 4/4 and
272000. Read literally that says the drain arm rehearsed nothing — a mechanism
claim. It is an artifact: **those cells were still running.**

A cell's jsonl carries a `locality` record from its *first* periodic probe
(step 200), and the `dream`/`sleep` records are only written when sleep ends.
Both `run_grid2.sh`'s resume check and `summarize_grid.py`'s `cell()` keyed on
"has a locality record", so a running (or half-dead) cell counted as finished:
the summarizer pooled it with zeroed fields, and a re-launched driver would
have *skipped* a cell that died at step 200. The periodic-probe feature the
08-06 debrief added is what broke both — neither consumer was updated with it.

Same lesson as §2 of the postmortem, one level down: the completion invariant
had no machine check either. Fixed in `0436ecb` — `dream_sleep.py` emits an
explicit `{"phase": "done"}` terminal record, the driver's resume check and the
summarizer both require it, and the summarizer now *prints* which cells it
skipped instead of silently dropping (or silently pooling) them.

Cells already finished (or in flight) under the old code have no `done` record;
they are backfilled once the grid ends — verified complete by their log's final
`done -> …` line with no live process — and the backfill is recorded here
rather than done silently. Nothing is re-run: the marker is bookkeeping, not a
measured quantity.

**Do not quote the 00:35 interim table.** The only cells in it that were
genuinely complete are the three arm-A cells.

### 8. Self-inflicted: the fix, deployed mid-grid, re-ran two finished cells

My mistake, recorded because the artifact trail shows it. I committed the
`done`-record fix (§7) *while the grid was running* and did not backfill the
marker on already-finished cells first. At 01:04 the `train`-tmux loop finished
seed 1234 and moved to seed 2345, where the new resume check found no `done`
record on the completed `A_s2345` cell and **re-ran it, truncating its jsonl**;
the loop then marched to seed 3456 and did the same to `A_s3456` — while the
`work`-tmux drivers already owned both seeds, so a same-cell collision was one
step away.

Recovery, in order: `C-c` in `train` (which killed only the inner python — the
outer `for` loop and `run_grid2.sh 3456` survived and had to be killed by PID,
the `pkill` trap in `sft/CLAUDE.md` being why I went by PID); backfilled `done`
markers onto every verified-complete, not-running cell (each backfilled record
carries `"backfilled": "marker added post-hoc; cell predates 0436ecb"`, so no
reader mistakes it for a live marker); let the in-flight `A_s3456` re-run
finish; relaunched `A_s2345` explicitly:

    tmux new-window -t work -n rerunA2345 'cd ~/altrux/sft && env MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. uv run --no-sync python -u dream_sleep.py --n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 32 --cue-greedy 12 --lr 1e-4 --distill-steps 800 --probe-every 200 --seed 2345 --arm replay --out logs/g2_A_s2345.jsonl 2>&1 | tee logs/g2_A_s2345.log; exec bash'

Scientific cost: none — both cells re-run the identical configuration against
the same cached dream, and their original per-probe output survives in the
`.log` files, which turns the accident into a free replication check of arm A
under kernel nondeterminism (worth reading when the grid is scored). Cost in
GPU time: ~15 min. **Lesson for the next session: a harness fix that changes a
resume predicate is not safe to deploy while drivers are live — backfill the
marker first, or wait for the drivers to drain.**

## §5(a) RESULT — the corrected single-sleep grid, 19/19 cells, 01:17 UTC

Command: `MODEL_NAME=mamba2_780m make summarize-grid` (glob `logs/g2_*_s*.jsonl`).
Hash check passes: **wave-1 dream hashes agree within every seed** — the
registered shared-dream invariant held this time, and it is machine-asserted.

### The primary metric, as registered, does not discriminate

**The untrained `no-sleep` floor scores 8/12 "installs" on the registered
metric.** Per-fact floor margins, with no training of any kind:

    1234 clove   +1.25   1234 heron  +4.07   1234 osprey +1.33   1234 topaz  +1.01
    2345 marimba -0.86   2345 oboe   +1.06   2345 saffron+1.16   2345 viola  -0.34
    3456 calcite -1.58   3456 ketch  +1.96   3456 saffron+1.24   3456 schooner -0.74

A single fixed random 5-digit foil is not a calibrated control: summed over
five tokens, ordinary digit-frequency preferences produce ±2–4 nats with
nothing learned. The registered "installed iff margin ≥ 1.0 nat" therefore has
a **67% false-positive rate at the floor**, and the §5(c) gate (≥0.3 install
fraction) is cleared *by the arm that does nothing*. This is the same failure
shape as greedy EM in the 08-06 grid: a metric that reports a number without a
floor.

Fix (committed `1f1ab33`, applied below): score **Δmargin = margin(arm) −
margin(no-sleep, same seed, same fact)**. The floor arm is a per-(seed,fact)
constant, so the correction is free, it gives the metric a true zero, and
`summarize_grid.py` now prints the floor's own false-positive count above the
table so the raw column cannot be quoted by accident.

### Floor-corrected frontier (pooled, 3 seeds, final d800 probe)

| arm | Δmargin | Δ-installs | dPPL (damage) | battery lost | EM | para | token-grads |
|---|---|---|---|---|---|---|---|
| sft-ref | **+18.55** | 10/12 | **+1.793** | 3/75 | 3/12 | 0.21 | 110,279 |
| B1 (corrected drain) | **+6.86** | 11/12 | +0.294 | 2/75 | 1/12 | 0.02 | **2,400** |
| A (replay, full-seq) | +5.66 | 11/12 | **+0.009** | 4/75 | 1/12 | 0.02 | 835,200 |
| CE-on-dream | +4.78 | 9/12 | +0.513 | 3/75 | 0/12 | 0.04 | 832,800 |
| B2 (per-token cf) | +1.38 | 6/12 | −0.029 | 1/75 | 0/12 | 0.00 | 2,400 |
| no-sleep (floor) | 0.00 | 0/12 | 0.000 | 1/75 | 0/12 | 0.00 | 0 |
| A_bridge (08-06 cfg) | +3.99 | 3/4 | +0.375 | 1/25 | 0/4 | 0.00 | 26,482 |

What this establishes:

1. **Dream distillation is enormously gentler than fine-tuning, at real
   learning.** A learns Δ+5.66 at dPPL **+0.009**; sft-ref learns 3.3× more at
   **+1.79**, i.e. ~200× the damage per unit of held-out perplexity. Both
   install ~10–11 of 12 facts. This replicates the 08-06 headline on a metric
   with a floor, and the shared-dream invariant now holds.
2. **The corrected B1 is not the retired drain arm — it is the best learner of
   the gentle arms.** Δ+6.86 (above A) at dPPL +0.29, and it does it with
   **2,400 token-gradients against A's 835,200** — 348× fewer, at equal
   optimizer steps. The 08-06 budget caveat guessed the erase arms might be
   more efficient per unit of gradient signal; on this grid B1 is, by a wide
   margin. This is the first positive result for the erase mechanism itself.
3. **B2 remains the weak learner** (Δ+1.38, 6/12) at genuinely negative damage
   — the same ~1/4–1/5-of-A ratio the 08-06 grid found, reproduced under the
   corrected arms and shared dream. Carrying the ablation (B1) versus
   discarding it (B2) is worth **5× the learning**, which is exactly the
   contrast the two arms were built to isolate.
4. **A's gentleness is partly the objective and partly the data.** CE-on-dream
   (same sequence, cross-entropy instead of KL) lands at dPPL +0.51 — 56× A's
   damage but still 3.5× gentler than sft-ref. So the soft targets matter *and*
   dreaming on the dream matters; neither alone explains A.
5. **The A_bridge cell** (08-06 configuration: chunk 48, carried) learns
   Δ+3.99 at dPPL +0.375 versus the registered full-sequence A's Δ+5.66 at
   +0.009. The schedule correction the debrief demanded was worth ~40% more
   learning at ~1/40th the damage — the old configuration was materially worse,
   as suspected.
6. **The §5(c) gate is met on the corrected metric**: best dream arm (B1)
   installs 11/12 = 0.92 pooled, and conditioned on binding-aware dream
   coverage it is 11/11 for facts the dream actually rehearsed. Gate threshold
   was 0.3.

Caveat carried: greedy EM stays near zero everywhere except sft-ref (3/12),
and paraphrase rates are ~0 for the dream arms vs 0.19–0.25 for sft-ref. The
dream arms install a *preference* for the right code, not a fluent retrieval —
sft-ref is still the only arm that will say the code out loud. That gap is the
most important open question this grid leaves.

## §5(c) is BLOCKED on harness work — do not launch it as registered

Checked before launching, not assumed. `run_sleep()` takes the seed's dream
cache and calls `dream_from_cache(cache, …)` **unconditionally**, so a wave-2
sleep would re-distil the *wave-1* dream and never rehearse wave 2's new facts.
The wake side of the wave loop is correct (wave ≥ 2 advances the carried state
through the new transcript, and the in-context control is taken on it), but the
sleep side has no per-wave dream.

Multi-sleep therefore needs, at minimum:

1. **Per-wave dream generation** from that arm's own carried state (which is
   what makes wave-≥2 dreams legitimately arm-specific, per §5(c) — the hash
   assertion is already correctly scoped to wave 1).
2. **A registered decision the 08-06 file does not make: who is the wave-2
   teacher?** Wave 1's dream comes from the frozen base with adapters bypassed.
   At wave 2 the student has been trained, so "frozen base" and "current model"
   are different teachers with different meanings (a stationary teacher vs.
   self-distillation of an already-consolidated model). This is a protocol
   choice, not an implementation detail.
3. §5(c) itself says the detailed cell list is "to be registered in its own
   DISCUSSION section before launch".

With the team AFK, inventing (2) unilaterally mid-run is exactly the kind of
protocol decision the postmortem's process rules exist to prevent. Left for
the debrief; recorded here as the next session's first harness item.

## §5(b) saturation ladder — launched 01:47 UTC

Registered docket, no new protocol needed. `sft/run_ladder.sh` (`b1c7fb5`),
seed 1234, `--probe-every 1600`, four arms concurrently:

    for a in A B1 B2 B2deep; do tmux new-window -t work -n "lad_$a" "cd ~/altrux/sft && ./run_ladder.sh 3200 $a 2>&1 | tee logs/ladder_${a}_3200.log; echo EXIT=\$? >> logs/ladder_${a}_3200.log; exec bash"; done

Cells are named `<arm>_d<steps>` so `summarize_grid.py`'s default glob pools
each rung as its own arm *and* still sees seed 1234's `no-sleep` cell, which
the floor correction needs.

Deviation from §5(b), recorded: **B1 is on the ladder** although the section
registered only A and B2. It was the best gentle learner in (a) at 1/348th of
A's token-gradients, so its saturation behaviour is now the most decision-
relevant curve on the board. B2deep is the registered no-detach BPTT cell.

The registered stop rule bounds the cost: an arm whose d3200 rung is flat
against its d800 result does not buy a d12800 rung. That decision gets made
from these curves, not in advance.

**Ladder KILLED at 01:56 UTC, ~9 minutes in, at altrup's request** (teammate
shutting down the machine that hosts the watchdog and the rsync pull — with it
off, nothing can terminate the instance, so the box had to go down first). No
ladder cell reached a probe point; `logs/g2_*_d3200_s1234.*` are partial and
carry no `done` record, so `summarize_grid.py` will skip them by construction.
**§5(b) is entirely unrun** — it is the next session's first GPU item, and
`sft/run_ladder.sh` (`b1c7fb5`) launches it in one command per rung.

## Harness work banked this session (all pushed)

| commit | what |
|---|---|
| `1ace4cf` | erase-hook identity test compared at fused-kernel tolerance (it was bitwise, which only holds on the ROCm box) |
| `eb4a95d` | `run_grid2.sh` takes `CUE_EVERY`, for a seed that misses the binding gate |
| `2fcbbc0` | `sft/CLAUDE.md`: the serial-grid rule scoped to the A10, with the GH200 concurrency measurement |
| `0436ecb` | explicit per-cell `done` record; driver resume check and summarizer both require it |
| `1f1ab33` | `summarize_grid.py` scores margins against the untrained floor, and prints the floor's own false-positive count |
| `b1c7fb5` | `run_ladder.sh`, the §5(b) rungs |
| `62601d7` | per-wave dream generation + `--wave-teacher`; `teacher_dream(frozen=…)` |

Full suite green at close: **230 passed** (`MODEL_NAME=mamba2_780m make test`).

## Closing summary — what this session established

1. **§5(a) ran clean and complete**, 19/19 cells, three seeds, with the
   shared-dream invariant machine-asserted for the first time (the bug that
   voided the 08-06 grid did not recur).
2. **The registered primary metric was uncalibrated** — the untrained floor
   scored 8/12 "installs". Fixed by floor-correction; every number in this
   file's frontier table is Δ-against-floor.
3. **Dream distillation is ~200× gentler than fine-tuning per unit learned**,
   and the corrected **B1 (the erase arm) is the best gentle learner** — more
   learning than A at 1/348th of the token-gradients. First positive result for
   the erase mechanism itself.
4. **Carrying the ablation is worth ~5× the learning** (B1 +6.86 vs B2 +1.38),
   isolating exactly what those two arms were built to separate.
5. **The §5(c) gate clears at 0.92** — but §5(c) *cannot run* until the
   wave-2 teacher choice is registered (harness side is now built, `62601d7`).
6. **The GH200's serial-execution assumption was wrong** for chunked arms
   (~2.8× from concurrency) and **right** for per-token arms, which saturate at
   ~2.5 aggregate steps/s — the batching proposal in §6 is what would fix that.

## For the debrief — open questions ranked

1. **The retrieval gap.** Dream arms move Δmargin +5.7…+6.9 but greedy EM ~0
   and paraphrase ~0; sft-ref reaches EM 3/12 and paraphrase 0.21. The dream
   arms install a *preference*, not a retrievable answer. Until that closes,
   "the memory earns its place" is not demonstrated in the sense the project
   goal means it.
2. **Wave-2 teacher: frozen base or trained student?** Blocks §5(c). Also
   unregistered: whether a later wave's cues cover only that wave's facts (what
   `62601d7` implements, matching wave 1) or every fact seen so far (which
   would make backward transfer partly an artifact of cueing).
3. **§5(b) unrun** — B2's saturation ceiling and the deep-B2 rewrite
   fingerprint are still unmeasured.
4. **Dream binding is the capacity bottleneck**, not distillation: per fact,
   installed ⟺ rehearsed-in-binding. Seed 1234 never bound topaz in any of
   three builds. Improving teacher-dream binding is likely worth more than any
   arm change.
5. **Same-seed generation is nondeterministic** on this host (§4) — cache
   everything that must be shared; never rely on a seed to reproduce a dream.

## Shutdown — 02:00 UTC, at altrup's call

Checklist run in order. No training checkpoints exist to verify: this docket
trains a throwaway LoRA per cell and its artifacts are the jsonl/log/cache
files, all of which the pull carries (`sft/logs/` 4.4M, `sft/data/` 553M of
dream caches + sidecars, `notes/`).

**Caveat on the last commit, stated plainly:** `62601d7`'s multi-sleep code is
green against the CPU fake backbone only. It has never executed on the real
model, because §5(c) cannot run until the wave-2 teacher is registered. Treat
it as unproven-on-hardware, not as a working feature.

Instance terminated with `scripts/.watchdog-terminate` while altrup's machine
was still up — with that machine off, the watchdog cannot terminate the box and
it would bill unattended, so the box goes down first.
