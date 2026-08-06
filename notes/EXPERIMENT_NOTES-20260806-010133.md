# Experiment notes — 2026-08-06 01:01 UTC (box session)

Fresh Lambda A10 (24 GiB, CUDA 12.9, torch 2.13.0+cu129 — **not** ROCm, so
none of the local box's gfx1102 workarounds apply here; `causal-conv1d` and
the fused SSD path are live).

Standing direction: `notes/DISCUSSION-20260805-dream-distillation-cl-ab.md`
§5 "Box (next session, A10, in order)". This session executes that docket:
test suite → dream-sleep smoke (read the dream!) → single-sleep primary grid
→ two-wave only if the install gate clears. No 2.7B, no M, no ladder, no null.

Model for the whole session: `mamba2_780m` (passed inline on every command).

## Box state at start

- Repo at `482c72a` (clean). `sft/.venv` present and healthy.
- **No HF cache, no checkpoints, no logs** — nothing arrived from the local
  archive upload. The dream-sleep harness builds its own synthetic wake
  transcripts and self-calibrates its knowledge battery, so no `make data`
  prep is needed for this docket; the model weights download on first use.
- `sft/logs/` did not exist; created (first `make test` invocation's `tee`
  failed on it — the pytest run itself was unaffected, output in the pane).

## Timeline / verbatim commands

### 1. Test suite (docket step 1) — 01:01 UTC

    tmux new-window -t work -n test 'cd ~/altrux/sft && MODEL_NAME=mamba2_780m make test 2>&1 | tee ~/altrux/sft/logs/box-test.log; echo EXIT=$?; exec bash'

First GPU execution of the dream_sleep + probes_common tests. 250 items
collected.

**Gotcha (cost me ~15 min):** the suite appears to *die silently* at
`tests/test_mixer_fused.py ..` — no traceback, no further output. It has not
died: the third test (`test_fused_lora_gradients_match_loop`) triggers a
**cold Triton compile of the SSD backward kernel**, and `ptxas` sits at ~90%
CPU for many minutes with pytest emitting nothing. `dmesg` shows no OOM kill;
`ps` shows the pytest process alive with a `triton/backends/nvidia/bin/ptxas`
child. On a fresh instance the Triton cache is empty, so budget several
minutes for the first fused-backward test. Second gotcha: on this box
`tmux display-message -p '#{pane_current_command}'` reads `bash` while
`uv run python` is actually working, so **completion detection must use a
file marker** (`echo EXIT=$? >> log`), not the pane command — the monitor's
"command finished" events are unreliable for `uv run` panes here.

**Result: 250 passed, 109 warnings in 116.45s. EXIT=0.** Includes the
dream_sleep (16) and probes_common tests' first GPU execution, and all 5
`test_mixer_fused.py` fused-SSD oracle tests (re-confirming the fused path on
a second CUDA host). Nothing to fix; docket step 1 done.

### 2. Dream-sleep smoke, Arm A (docket step 2) — 01:38 UTC

    tmux new-window -t work -n smoke 'cd ~/altrux/sft && MODEL_NAME=mamba2_780m make dream-sleep ARGS="--arm replay --n-facts 4 --filler-tokens 40 --dream-tokens 512 --distill-steps 200 --seed 1234 --out logs/dream_smoke_replay.jsonl" > ~/altrux/sft/logs/smoke_replay.log 2>&1; echo "EXIT=$?" >> ~/altrux/sft/logs/smoke_replay.log; exec bash'

This also builds the two shared artifacts every later arm/seed reuses: the
self-calibrated knowledge battery (`data/knowledge_battery_mamba2_780m.json`)
and the base-model held-out PPL. Per the docket: **read the decoded dream and
the fact-rehearsal fraction before running anything else**; if rehearsal is
~0, apply §4's `--dream-prompt` seeding rule before concluding anything about
the mechanism.

**Result (EXIT=0, ~4 min end to end).** Everything upstream of the dream is
healthy:

- Transcript: 4 facts / 20 turns / 264 tokens, **all four invariants 0**
  (role adjacency, facts-stated-once, entities-mentioned-twice, digits in
  filler), round-trip decode == source. Decoded samples read correctly.
- Knowledge battery self-calibrated to **24 kept of 40 candidates** →
  `data/knowledge_battery_mamba2_780m.json` (shared by every later run).
- Base held-out PPL **23.559** over 364 tokens.
- Facts this seed: clove (spice), topaz (mineral), osprey (bird), heron
  (bird).

| phase | result |
|---|---|
| fresh-state floor | 0/4, lp ≈ −3.7…−4.4 |
| in-context control | **3/4 HIT** (lp −0.11…−0.36) |
| post-sleep fresh probe | **0/4**, but lp deltas **+0.94 / +1.12 / +1.54 / +1.36** |
| paraphrase rate | 0.00 everywhere |
| battery | **24/24 retained**, mean dlogp +0.0446 |
| held-out PPL | 23.592, **dPPL +0.0328** |

Two observations worth carrying:

1. **The in-context miss is topaz, and it returned osprey's code
   (`5 9 7 9 7`).** This is the *same* topaz↔osprey entanglement §3a found on
   the local box at this seed — independent reproduction on different
   hardware, and a reminder that the in-context control is the conditioning
   denominator for a reason.
2. **`rehearsal_fraction = 0.00`.** The dream is fluent but wanders straight
   into fiction (`'the guardbox serve to Nyoken Makiyama! She moves all an'`,
   `'a pile of radios that causes'`). It never emits an entity or a code.

**Interpretation.** The uniformly positive lp deltas are *not* evidence of
installation and must not be read as such: with rehearsal at 0 the dream
never touched a binding, so there is nothing for the state-reads to distil.
What moved is the answer-slot **format prior** — post-sleep generations
became digit-shaped (`' 1.'`, `' 1 2 3 4 5 6 7 8...'`) where the floor
generations were C code (`'#include <stdio.h>'`). A +1 nat gain on a 5-digit
target is what you get from learning "digits go here", uniformly, for facts
the dream never mentioned. This is exactly why §4 made rehearsal a
conditioning variable rather than a footnote.

Docket §4's decision rule therefore fires: **seed the dream before concluding
anything about the mechanism.** Not doing so would burn the whole 30-run grid
on dreams that cannot test the hypothesis.

Useful side note: KL loss shows a clean sawtooth with period ~11 steps
(512 dream tokens / chunk_len 48 ≈ 11 chunks) whose peaks decay monotonically
(2.54 → 2.34 → 2.03 → … → 0.24), i.e. the replay distillation itself is
working fine — the problem is purely the dream's *content*.

### 3. Rehearsal-cue sweep (§4 decision rule) — 01:14 UTC

`sft/cue_sweep.sh`, committed. 3 temperatures × 3 cues, dream-only
diagnostic (`--distill-steps 1`, so cost is the dream not the training),
seed 1234 fixed so the wake state and facts are identical across cells:

    for temp in 1.0 0.7 0.4; do for cue in "" "Let me go back over what I was just told." "Here is a summary of the codes I was given."; do
      MODEL_NAME=mamba2_780m ... uv run --no-sync python -u dream_sleep.py \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens 512 \
        --distill-steps 1 --seed 1234 --dream-temp "$temp" --dream-prompt "$cue" \
        --out "logs/cue_sweep_$i.jsonl"

Temperature is in the sweep because it is the other obvious lever on
wandering: at temp 1.0 the dream drifts into fiction within a few tokens, and
a lower temperature should keep it nearer the state's strongest content. The
cues are deliberately **category-free and entity-free** — they cue the
*register* of the wake session, not its content, so no arm gets a prompt-side
leak of the thing being scored.

**Result — temperature is the lever; the cues are not.**

| temp | cue: none | "go back over what I was just told." | "summary of the codes I was given." |
|---|---|---|---|
| 1.0 | 0.014 | 0.000 | 0.000 |
| 0.7 | **0.125** | 0.086 | 0.066 |
| 0.4 | 0.312 | 0.148 | 0.182 |

Cues make rehearsal *worse* at every temperature where it exists at all — they
push the dream into a summarizing register that then drifts, whereas the
uncued dream lets the wake state's strongest content dominate the sampling.
The registered `--dream-prompt` fallback is therefore **not** the fix; the
fix is the sampling temperature, which the docket never pinned.

**But the highest rehearsal fraction is a trap, and only reading the dream
shows it.** Decoded cell 7 (temp 0.4, frac 0.312, the sweep's best number):

    '[ASSISTANT] The code for the heron is 2 1 2 1 0.[ ...filler... ]
     [ What is the code for the heron?] The code for the heron is 2 1 2 1 0.
     [ What is the code for the heron?] The code for the heron is 2 1 2 1 0.
     [ What is the code for the heron?] The code for the heron is 2 1 2 1 0. ...'

— a degenerate loop on **one** fact, repeated to the token budget. It scores
0.312 because `rehearsal_fraction` counts needle *tokens*, and can only ever
install heron. Decoded cell 4 (temp 0.7, frac 0.125) is genuine session
rehearsal: it questions/answers heron, drifts through filler, then asks **and
correctly answers topaz from state** (`The code for the topaz is 5 3 0 0 0.`)
— note the dream recalls topaz correctly even though the in-context probe
returned osprey's code for it.

**Consequence for the A/B: `rehearsal_fraction` is the wrong conditioning
variable.** The quantity that bounds how many facts a sleep can possibly
install is **distinct-fact coverage** — how many of the wave's entities and
codes appear at all. §4 conditions installs on rehearsal; conditioning on a
metric that a one-fact loop maximizes would systematically mis-score every
arm. The existing `needle_counts` field already carries what's needed
(per-entity and per-code hits), so coverage is computable from the logs
without a harness change — but it should become a first-class logged field
before the grid runs.

### 4. Dream-length sweep (`sft/cue_sweep2.sh`, commit `cf573fd`) — 01:29 UTC

Temperature 0.7/0.55 × 512/1024/2048 tokens, no cue, scored on coverage:

| temp | tokens | frac | entities | codes |
|---|---|---|---|---|
| 0.7 | 512 (sweep 1 cell 4) | 0.125 | 3/4 | 4/4 |
| 0.7 | 1024 | 0.185 | 2/4 | 1/4 |
| 0.7 | **2048** | 0.137 | **3/4** | **4/4** |
| 0.55 | 1024 | 0.002 | 1/4 | 0/4 |
| 0.55 | 2048 | 0.156 | 1/4 | 1/4 (heron ×157) |

temp 0.55 collapses into the same single-fact loop as 0.4 — **below ~0.7 the
dream mode-collapses**, so 0.7 is a genuine sweet spot rather than "lower is
better". Dream length is *not* cleanly monotonic: 512 and 2048 tie at 3/4 +
4/4 while 1024 scores worse, which at n=1 per cell is noise, not signal.

### 5. Seed-robustness of the dream setting (`sft/cue_sweep3.sh`) — 01:33 UTC

Deciding the A/B's dream config on one seed's sample would be exactly the
kind of n=1 tuning that taints the grid it feeds. 512 vs 2048 tokens × seeds
1234/2345/3456 at temp 0.7, coverage-scored:

    for seed in 1234 2345 3456; do for toks in 512 2048; do
      MODEL_NAME=mamba2_780m ... uv run --no-sync python -u dream_sleep.py \
        --arm replay --n-facts 4 --filler-tokens 40 --dream-tokens "$toks" \
        --distill-steps 1 --seed "$seed" --dream-temp 0.7 \
        --out "logs/seed_sweep_${seed}_${toks}.jsonl"

Standing bias for the decision: deviate from the docket's registered
parameters as little as the evidence forces. Temperature 1.0 → 0.7 is forced
(0.00 rehearsal makes the experiment untestable). Dream length 512 → 2048 is
**not** forced unless it buys coverage across seeds, and it costs 4× the
generation time on every one of the ~30 grid runs.

**Result — the dream setting is NOT seed-robust, and this blocks the grid.**

| seed | 512 tok | 2048 tok |
|---|---|---|
| 1234 | 1/4 ent, 2/4 codes | 3/4 ent, 4/4 codes |
| 2345 | **0/4, 0/4** | **0/4, 0/4** |
| 3456 | 1/4 ent, 0/4 codes | 1/4 ent, 0/4 codes |

Free generation rehearses usably at **one of three seeds**. Length is not the
lever (2345 is zero at both).

### 6. Retrieval-stem cues (`sft/cue_sweep4.sh`) — 01:52 UTC

Sweep 1's cues were declarative and cued a *summarizing* register. This one
seeds the wake session's question stem and stops mid-phrase, so continuing
requires completing the entity from state and answering it — cue supplies
format, state supplies content. Entity-free and code-free, so no leak.

| seed | `[USER] What is the code for the` | `Let me go back over the codes from earlier.` + stem |
|---|---|---|
| 1234 | 0/4, 0/4 | **0/4, 0/4** (was 3/4+4/4 uncued!) |
| 2345 | 0/4, 0/4 | 2/4 ent, 3/4 codes (was 0) |
| 3456 | 1 ent hit | 1/4 ent, 1/4 code |

The bare stem degenerates because the seed becomes `[ASSISTANT] [USER] …`, an
immediate role flip. More important: **cueing rescues 2345 and destroys
1234.** There is no fixed dream configuration that gives usable coverage at
all three seeds. Picking the best config per seed would be tuning on the
outcome variable and would taint every comparison built on it.

## Decision: the registered A/B could not be run as specified

Free-generation rehearsal is a **lottery** — it depends on which entity
happens to dominate the wake state at that seed. §4 treats "does the dream
rehearse?" as a smoke-test question with a one-line fallback; measured across
seeds it is the binding constraint on the whole experiment. Running the
~25-cell grid on free dreams would have spent hours producing cells that
structurally cannot test their own hypothesis, and the pooled
rehearsal-conditioned install rate that gates the two-wave phase (§4, ≥0.3)
would have been computed over a denominator of ~1 seed.

So I built the fix rather than run the grid blind (commit `719dc2a`).

### 7. Cued rehearsal — `--cue-every` / `--cue-greedy`

Two new flags on `dream_sleep.py`, **off by default** so the registered
free-generation protocol stays reproducible:

- `--cue-every N` forces each wave fact's own wake-session question stem
  (`[USER] What is the code for the {entity}?[ASSISTANT] The code for the
  {entity} is`) into the dream every N tokens, cycling the facts. The cue
  names *which* fact to recall; the state still supplies the digits.
- `--cue-greedy K` decodes the K tokens after each cue at temperature 0.

The second flag exists because of a diagnostic worth recording: with cues
alone, entity coverage went to 4/4 at every seed but **code** coverage stayed
at 0/4 for seed 3456 — the dream asked about every fact and answered none.
In-context recall at that seed is 3/4, so the state *did* hold the facts. The
culprit is sampling: a 5-digit code needs five correct digits in a row, which
temperature-0.7 sampling almost never produces even when each digit is
individually likely. Greedy answer spans make the rehearsed code the model's
actual best recall from state — which is what "rehearsing the binding" means,
and matches the probes' own greedy convention.

TDD as required: 4 new tests in `tests/test_dream_sleep.py` (cue rotation,
answer slot still sampled, greedy span == teacher argmax, cues-off is a
no-op), failing first, then implemented. Full suite **20 passed**.

**Coverage after the fix (`--cue-every 64 --cue-greedy 12`, 512 tok, 0.7):**

| seed | entities | codes | (free-generation was) |
|---|---|---|---|
| 1234 | **4/4** | 3/4 | 3/4, 4/4 |
| 2345 | **4/4** | **4/4** | 0/4, 0/4 |
| 3456 | **4/4** | 3/4 | 1/4, 0/4 |

Code coverage now tracks each seed's **in-context recall ceiling** (3/4, 3/4,
3/4) — the most any dream can possibly rehearse, since a fact the state
cannot recall cannot be dreamt. `--cue-every 64` beats 128 (more firings per
512 tokens). Decoded sample read before trusting it (seed 2345), and it is
correct rehearsal material: filler drift, then `[USER] What is the code for
the saffron?[ASSISTANT] The code for the saffron is 3 0 4 1 6.`, then
`marimba is 5 4 6 6 9` — both matching the fact list.

**Two caveats for the debrief, flagged deliberately:**

1. **Cued dreams move Arm A toward SFT-ref.** With every fact cued in its
   wake phrasing, the dream's rehearsal segments approach the wake transcript,
   so generative replay converges on the CE-on-raw-text reference. This is
   arguably what generative replay *is* in the literature (the model
   regenerates its own training data), but it narrows the A-vs-SFT-ref
   contrast, and the team should decide whether that is acceptable or whether
   Arm A wants a lower cue rate than the B arms. It does **not** affect the
   A-vs-B contrast, which is the registered primary: A, B1 and B2 share the
   identical cued dream at a seed, and state deprivation remains the only
   manipulated variable.
2. The cue injects the answer *format*, which inflates the format-prior effect
   measured in §2. `--no-sleep` and the fresh-state floor bound it.

### 8. The single-sleep primary grid — 02:52 UTC

`sft/run_grid.sh SEED`, three parallel streams (one tmux window per seed).
The A10 ran one job at ~2.3/23 GiB and ~20% util, so the grid is latency-bound
rather than memory-bound; three concurrent streams put it at 7 GiB / 100%.

    common="--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 --cue-every 64 --cue-greedy 12"
    for steps in 200 800; do for arm in replay drain counterfactual; do
      uv run --no-sync python -u dream_sleep.py $common --seed $seed --arm $arm --distill-steps $steps --out logs/dream_${arm}_d${steps}_s${seed}.jsonl
    done; done
    # then: --sft-ref at 200 and 800, --no-sleep, and --arm drain-live at seed 1234 only

25 cells: 3 dream arms × 3 seeds × 2 budgets (18) + sft-ref × 3 seeds × 2
budgets (6) + no-sleep × 3 seeds (3) + drain-live at seed 1234 (1). Deviations
from the registered invocation are exactly `--dream-temp 0.7` and the two cue
flags, both forced by §§3–7 above and annotated in the script itself.

**Parallel streams are a net loss on this GPU — run the grid serially.** The
A10 looked idle on one job (2.3/23 GiB, ~20% util), so three seed streams
seemed free. Measured: each stream managed **0.11 step/s** (60 KL steps in
9 min), i.e. **0.33 step/s aggregate against 1.4 step/s solo** — three-way
concurrency is ~4× *worse* in total throughput, not 3× better. Whatever the
util number reflects, it is not headroom this workload can use; the sleep loop
is one-token-at-a-time and latency-bound, and three CUDA contexts on one A10
serialize badly. Killed and relaunched as a single serial stream over all
three seeds: `replay_d200` then completed its 200 steps in ~3 min.

Two harness fixes made at the same time, both worth keeping:

- `run_grid.sh`'s resume check tested `[ -s "$out" ]`, but a cell's jsonl is
  non-empty from its first probe record — a killed cell would have been
  skipped as "done" and silently missing from the grid. It now requires the
  cell's terminal `"phase": "locality"` record.
- The per-cell `grep` filter had no `--line-buffered`, so with output
  redirected to a file the grid logs showed nothing until a cell finished
  (they looked hung while training fine). This is the project's live-progress
  rule; fixed.

**Interim result, seed 1234 (4 cells), scored by `sft/summarize_grid.py`:**

| arm | rehrs | code cov | in-ctx | installs | mean dlogp | para | battery lost | dPPL |
|---|---|---|---|---|---|---|---|---|
| replay d200 | 0.213 | 3/4 | 3/4 | **0/4** | **+1.818** | 0.00 | 0/24 | +0.383 |
| replay d800 | 0.240 | 3/4 | 4/4 | **0/4** | +1.759 | 0.12 | 0/24 | +0.074 |
| counterfactual d200 | 0.234 | 3/4 | 4/4 | **0/4** | +0.077 | 0.00 | 0/24 | +0.010 |
| drain d200 | 0.168 | **1/4** | 4/4 | **0/4** | −0.066 | 0.00 | 0/24 | +0.170 |

Two things read cleanly even this early:

1. **The arm ordering matches the registered hypothesis.** A moves far more
   than either B arm (+1.82 vs +0.08 / −0.07) and damages more (dPPL +0.38 vs
   +0.01 / +0.17). B being slower to install is exactly what altrup
   pre-registered.
2. **B1's 1/4 code coverage is the mechanism, not a bug.** The drain erases
   bindings *as the teacher generates*, so cued questions later in the dream
   can no longer be answered from the drained state. Consumption is visible
   in vivo, which is what §4's carried-state diagnostic column was for.

**But every arm installs 0/4, including Arm A at 4× the budget.** A frontier
cannot be drawn through zeros, and §4's two-wave phase is gated on a pooled
conditioned install rate ≥ 0.3. Completing the remaining ~21 cells would have
spent ~2.5 h to produce a table of zeros.

### 9. Install-feasibility probe (grid paused) — 03:12 UTC

Diagnosis before knob-turning: `consolidation_null.py` installs at this *same*
LR (1e-4) and step count (200), so the learning rate is not the obvious
difference. What differs is **what the replayed material is made of** — the
null replays a fact-bearing transcript, while this dream is ~80% filler
(rehearsal fraction 0.21, i.e. ~8 cue firings × ~25 answer tokens out of 512).
So the knobs under test are rehearsal *density* and *total rehearsal count*.

`sft/install_probe.sh`, Arm A only (the arm that moved most, hence the cheapest
test of whether installation is reachable here at all):

    probe dense32_d800   --cue-every 32 --dream-tokens 512  --distill-steps 800
    probe long2048_d1600 --cue-every 64 --dream-tokens 2048 --distill-steps 1600
    probe dense32_lr3e4  --cue-every 32 --dream-tokens 1024 --distill-steps 1600 --lr 3e-4

Decision rule set before running: if some cell installs, re-run the grid at
that setting (the arm comparison is only meaningful above the install floor).
If none does, the honest finding is that dream-distillation does not reach
installation at 780M within a sleep-sized budget, and that — with the arm
ordering above — is what goes to the debrief.

The four completed grid cells are kept; `run_grid.sh` skips cells that already
carry a `locality` record, so the grid resumes rather than restarts.

**Result — and it changes how the whole grid must be scored.** Arm A, seed
1234, per-fact teacher-forced logprob (`lp`), delta vs floor (`d`), and
paraphrase rate:

| cell | clove | topaz | osprey | heron | battery | dPPL |
|---|---|---|---|---|---|---|
| dense32 d800 | −2.24 (+1.78) | −2.72 (+1.17) | −3.81 (+0.68) | **−0.70 (+2.91)** | 24/24 | −0.138 |
| long2048 d1600 | −1.86 (+1.94) | −2.23 (+1.56) | −3.55 (+0.82) | **−0.52 (+3.14), para 0.50** | 24/24 | +0.155 |
| dense32 lr3e-4 | −1.64 (+2.16) | −1.14 (+2.65) | −3.69 (+0.68) | **−0.38 (+3.28), para 0.25** | 24/24 | +0.072 |

Greedy exact match is **0/4 in every cell** — but that metric is lying:

- heron reaches logprob −0.38, i.e. **p ≈ 0.68 on the entire 5-digit code**.
  That is an installed fact by any reasonable reading.
- At `long2048_d1600` heron's **paraphrase rate is 0.50** — it emits the
  correct code on 2 of 4 *reworded* prompts while scoring `miss` on the
  trained phrasing. Generality passing where reliability fails inverts the
  usual expectation and is the tell.
- The failing generations are a **digit-counting attractor**
  (`' 1 2 3 4 5 6 7 8 9 0.'`) or one-digit misses (heron's true code is
  `2 1 2 1 0`; greedy emits `1 1 2 1 0`), and osprey's probe emitted heron's
  code verbatim.

Mechanism: the dream rehearses the *trained phrasing* many times, each time
followed by digits, so distillation makes that exact context the counting
attractor's home. The paraphrase prompts sit outside the over-rehearsed
context and recover the fact. Raising the LR to 3e-4 buys the best logprobs
and the worst paraphrase rate — consistent with sharpening the attractor.

**Consequence: the grid's primary axis becomes per-fact logprob delta (and
paraphrase rate), not greedy exact match.** This is not a post-hoc rescue —
§4 already tracks per-fact logprob deltas "throughout" precisely because "the
null taught us installs are too rare to carry significance alone". The
registered install-vs-ΔPPL frontier becomes a **dlogp-vs-ΔPPL** frontier,
which is computable and already shows clean arm separation. §4's ≥0.3
conditioned-install gate for the two-wave phase is **not met on greedy EM**,
and I am not reinterpreting that gate to unlock two-wave — that call belongs
to the team.

Grid resumed at the **registered** budgets (200/800, `--cue-every 64`,
512 tokens) rather than at these larger ones: the registered design is what
the frontier was specified on, it costs ~2.5 h instead of ~6, and the higher
budgets are recorded here as separate frontier points.

**COMPLETE — all 25 registered cells, 3 seeds** (28 jsonl files incl. the 3
install-probe cells). Scored with `sft/summarize_grid.py`; full per-cell table
in `sft/logs/`, pooled below.

| arm | n | installs | mean dlogp | battery lost | dPPL |
|---|---|---|---|---|---|
| **replay d800 (A)** | 3 | 0/12 | **+1.830** | 1/72 | **+0.145** |
| replay d200 (A) | 3 | 0/12 | +1.688 | 2/72 | +0.609 |
| **sft-ref d200** | 3 | **3/12** | +0.943 | 1/72 | **+2.190** |
| **sft-ref d800** | 3 | **3/12** | +0.612 | 2/72 | **+2.177** |
| counterfactual d800 (B2) | 3 | 0/12 | +0.346 | 0/72 | **−0.163** |
| counterfactual d200 (B2) | 3 | 0/12 | +0.176 | 0/72 | +0.069 |
| drain d800 (B1) | 3 | 0/12 | +0.048 | 0/72 | −0.089 |
| drain d200 (B1) | 3 | 0/12 | −0.003 | 0/72 | +0.094 |
| drain-live (1 seed) | 1 | 0/4 | +0.030 | 0/24 | −0.042 |
| no-sleep (floor) | 3 | 0/12 | **+0.000** | 0/72 | **+0.0000** |

## Findings

**1. Generative replay dominates conventional fine-tuning on the frontier.**
Arm A at 800 steps beats SFT-ref on *both* axes simultaneously: **3× the mean
logprob gain (+1.830 vs +0.612) at 1/15th the perplexity damage (+0.145 vs
+2.177)**. This is the registered primary question — "is dream-distillation
sleep better than conventional CL fine-tuning, and does it suffer the same
catastrophic forgetting" — and the answer on this battery is yes and no
respectively. It reproduces at all three seeds (A's dPPL 0.036/0.074/0.325
against SFT's 0.71/2.84/2.98; no overlap).

**2. The erase arms did NOT beat generative replay — the registered
hypothesis is not supported.** B1/B2 are the *gentlest* arms by a wide margin
(0 battery items lost in 72 across every cell, dPPL ≤ 0 at the higher budget)
but they barely learn: +0.35 (B2) and +0.05 (B1) against A's +1.83. altrup's
pre-registered bet was that B would be slower to install but own a better
learned/forgotten *ratio*. B is indeed slower; it does not own the frontier,
because A already achieves near-zero damage. There is no budget in this grid
where B's curve dominates A's.

**3. B1's consumption is real and is also its ceiling.** B1's dream code
coverage is **1/4 at every seed and both budgets** (vs 3–4/4 for A and B2),
because the erase consumes bindings *as the teacher generates*, so cued
questions later in the dream can no longer be answered. The mechanism does
what §3 says it does — and that is exactly why it has almost nothing left to
distil. Worth the team's attention: consumption and installation are in direct
tension in the single-sleep form.

**4. Greedy exact match is the wrong reliability metric here, and it inverted
the arm ranking.** SFT-ref is the *only* arm with greedy installs (3/12, i.e.
1 fact per seed) while showing lower logprob gains than A. Meanwhile A reaches
per-fact logprob −0.38 (p ≈ 0.68 on the whole 5-digit code) and paraphrase
rates up to 0.50 while scoring 0/12. The failure mode is a **digit-counting
attractor** (`' 1 2 3 4 5 6 7 8 9 0.'`) at the *trained phrasing*, which the
dream itself trains in by rehearsing that phrasing followed by digits; the
paraphrase probes sit outside it and recover the fact. Generality passing
where reliability fails is the diagnostic. **§4's ≥0.3 conditioned-install
gate for the two-wave phase is not met on greedy EM (0/24 conditioned), so I
did not run two-wave** — reinterpreting a pre-registered gate to unlock the
next phase is the team's call, not mine.

**5. The floor validates the harness.** `no-sleep` returns exactly +0.000
dlogp and +0.0000 dPPL at all three seeds — the no-op arm is a true no-op, so
the deltas above are training, not measurement drift.

## What I'd want to discuss

1. **Replace or supplement greedy EM as the reliability metric** before any
   further grid work. Options: score the code's teacher-forced logprob against
   a threshold; constrain decoding to the digit-answer format; or promote the
   paraphrase battery to primary. Every "0 installs" number above is a
   metric artifact of the attractor, and the two-wave gate depends on it.
2. **Does the erase mechanism survive finding 2?** B is safe but inert here.
   The 07-25 "verified erase" follow-up in §7 (install via B2, then one
   end-of-sleep real erase gated on a fresh-state check) now looks more
   attractive than B1's generate-while-draining, precisely because B1's
   coverage collapse is structural.
3. **Cued rehearsal moves Arm A toward SFT-ref** (§7 caveat 1). A won the
   frontier *with* cues; the team should decide whether that comparison is
   clean enough, or whether A wants a lower cue rate than the B arms.
4. **Free-generation rehearsal is a lottery** (§5–6). Any future protocol
   relying on unprompted dreams needs this handled; `--cue-every` is one
   answer, not necessarily the right one.

## Session summary

Docket §5 steps 1 and 2 are **done**; step 3 (two-wave) is **not run**, gate
not met (finding 4). Step 4 is "nothing else", so the run is complete.

- Tests: 250 → 254 passed (4 new for cued rehearsal), incl. first GPU
  execution of dream_sleep/probes_common and all 5 fused-SSD oracle tests.
- Artifacts for the pull: 28 `sft/logs/dream_*.jsonl` + `inst_*.jsonl`, the
  grid/sweep logs, and `sft/data/knowledge_battery_mamba2_780m.json` (24
  self-calibrated items, base PPL 23.559 — reuse it so future runs score
  identical items).
- Commits pushed: `cf573fd` (sweeps), `719dc2a` (cued rehearsal + tests),
  `bde8a71` (grid driver), `54635b9` (summarizer + driver fixes), `f7d24b2`
  (install probe).
- No training run was started this session and no checkpoint was written —
  this was an eval/probe session throughout, as the DISCUSSION docket called
  for. Nothing to resume from.
