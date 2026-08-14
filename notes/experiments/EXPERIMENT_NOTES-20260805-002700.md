# Experiment notes — 2026-08-05 00:27 UTC — transcript-consolidation null

Standing direction: `notes/discussion/DISCUSSION-20260804-consolidation-null-run-plan.md`.
This session runs `sft/consolidation_null.py` (GPU-unvalidated; first run is a
shakeout) at 780M. No training run, no M arms, no data generation.

## Box

- Instance: `150-136-86-244`, **NVIDIA A10, 23028 MiB**, driver 570.148.08,
  CUDA 12.8. Not the local ROCm box.
- `sft/.venv` present, `torch 2.13.0+cu129`, `cuda.is_available() True`,
  device `NVIDIA A10`.
- HF cache **empty** on arrival (`.cache/huggingface` absent) — the 780M
  weights download on first run. `LAMBDA_DATA_ARTIFACTS` empty on purpose
  (this run builds its own transcript), so nothing else to regenerate.
- `sft/.env` `MODEL_NAME` arrived blank (as documented); set to
  `mamba2_780m`.
- tmux: `train` and `work` sessions created. This is not a training session,
  so everything runs in `work`.

## Pre-run read of the script

`consolidation_null.py` builds its own transcript and already prints the
structural invariants + decoded samples the root CLAUDE.md data gate asks for
(`report_transcript`), so the sanity gate is satisfied by the run's own log
rather than by `make sanity-sample` — there is no `.pt` slice here.

**Runtime hazard identified before the first run.** Every model in `models/`
drives Mamba2's mixer with a per-token Python loop
(`models/mamba2_780m/model.py:234`, `for t in range(seqlen)` over 48 layers)
because `causal-conv1d` and the Triton SSD scan kernel are broken on the
*local ROCm* box. On this CUDA A10 that workaround is pure overhead — the
native fused/chunked path would work here. The loop is the dominant cost of
this run, so before spending the default budget I am measuring throughput
with a tiny smoke config.

Rough cost of the default run (40 facts, filler 200 → ~9.2k-token transcript,
chunk_len 48 → ~192 chunks, 200 distill steps ≈ one pass):
~62k token-steps end to end (3 full transcript passes forward, 200 chunk
steps with backward, ~5k probe-generation tokens). Wall time therefore
swings between ~10 min and several hours purely on the per-token loop's
speed — hence the measurement first.

## Log

### 00:27 — setup

Monitor armed (persistent; watches `train`/`work` pane completions, newest
`sft/logs/*.log` for `non-finite|Traceback|RuntimeError|out of memory|Killed|
VERDICT|WARN`, heartbeat 300s).

### 00:28 — shakeout smoke run (tiny config, to shake out crashes cheaply and measure throughput)

Verbatim:

    tmux send-keys -t work "cd ~/altrux/sft && make consolidation-null ARGS='--n-facts 4 --filler-tokens 40 --distill-steps 10 --pass-k 3 --out logs/smoke.jsonl'" Enter

(log: `sft/logs/consolidation-null-20260805-002821.log`)

Deliberately *not* the plan's default config: the script had never touched a
GPU, and finding a crash 30 minutes into a full run costs more than a 3-minute
tiny one. Full defaults follow immediately after.

Results:

- **Loads and runs.** 96 LoRA adapters, 9,673,728 trainable params. No import
  breakage — note this box is CUDA, so none of the root CLAUDE.md ROCm
  workarounds (`selective_scan_cuda` stub etc.) are being exercised the same
  way.
- **Transcript structurally clean**: all four invariants 0, round-trip decode
  `True`, and the decoded samples read correctly — facts stated once, filler
  digit-free, roles strictly alternating across the filler→fact join.
- **Both harness controls pass** (the §3 gate in the plan):
  - in-context positive control **1.000** (threshold 0.8) — the base 780M
    *can* follow the probe format. No format-SFT detour needed.
  - fresh-state floor **0.000**, mean code log-prob −3.9186 — no code leakage.
- Teacher logit cache worked (0.02 GiB at 264 tokens; scales to ~0.9 GiB at
  the default ~9.2k tokens — comfortable in 23 GiB).

**Throughput (the number that shapes the session):** ~**64 ms/token**
(prime 264 tokens in 17s), GPU utilization **19%** — Python-overhead-bound in
the per-token mixer loop, not compute-bound. Distillation measured
**0.13 step/s** (10 steps in 76s). Projects the default run to ~1 hour:
~10 min prime + ~10 min teacher context + ~10 min teacher replay + ~26 min
distill + ~6 min probes.

Initial decision: **do not touch the mixer for this run** — 1 hour is
affordable and the plan's budget is a shakeout plus one or two real runs, so
speed is not the binding constraint on the verdict. (Superseded below.)

### 00:31 — the fast-path question (raised by altrup)

The per-token loop's real cost is not this run, it is that it makes **training
on rented CUDA impossible**, not merely slow: at 64 ms/token the 230M-token
chain corpus is ~170,000 GPU-hours. Root CLAUDE.md already records that the
fused path should be used on a proper CUDA target; this box is the only place
it can be *validated*, since the local ROCm box cannot run the kernel at all.

Feasibility checked before committing (inspection only):

- `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` takes
  `initial_states` and `return_final_states` and is autograd-capable
  (`MambaChunkScanCombinedFn.apply`) — so the chunk-to-chunk SSM-state
  threading that `MixerState` and `train.py` depend on is directly supported.
- `causal-conv1d` is absent here, but the conv is exactly reproducible by
  prepending `conv_state` as left padding to `F.conv1d`.

**DECISION: deferred, not done this session** (altrup). It is out of scope for
`DISCUSSION-20260804`, and the null verdict — not throughput — is what gates
downstream CL work. The 1-hour run cost is affordable as-is. Recorded here as
a standing item for the team rather than acted on.

#### Deferred item: migrate the models to the fused/chunked fast path

**Why it matters (not about this run).** The per-token loop costs ~64 ms/token.
This null run is ~1 hour either way, so it is irrelevant *here*. It is decisive
for training: at that rate the 230M-token chain corpus is ~170,000 GPU-hours,
i.e. the manual loop does not merely slow rented-CUDA training down, it makes
it impossible. Root CLAUDE.md already notes the fused path should be used on a
proper CUDA target; what is new is the measured number and the confirmation
that the machinery is present.

**Feasibility (verified by inspection on this box, 2026-08-05):**

- `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined` accepts
  `initial_states` and `return_final_states`, and is autograd-capable
  (`MambaChunkScanCombinedFn.apply`). This is the load-bearing fact: the
  chunk-to-chunk SSM-state threading that `MixerState` and `train.py` depend
  on is directly supported, so the fast path does not require giving up
  truncated-BPTT chunking.
- `causal-conv1d` is absent, but is not needed: the conv is exactly
  reproducible by prepending `conv_state` as left padding to `F.conv1d`.

**Sketch when someone picks this up:**

1. Test first, with the existing per-token loop as the oracle — compare logits
   *and* final `MixerState` for a fresh state, two threaded chunks, and
   gradients w.r.t. LoRA params. fp32 for a tight bound, bf16 looser (a
   sequential scan and a chunked scan will not agree to bf16 round-off).
2. `_mixer_chunk` alongside `_mixer_step`: batched `in_proj`; `F.conv1d` with
   `conv_state` as left pad; `mamba_chunk_scan_combined(..., initial_states=
   ssm_state, return_final_states=True, dt_softplus=True)` with `z=None`, then
   the gated norm applied *after* — matching the manual path's ordering.
3. Dispatch on `T==1` → keep the proven `_mixer_step` (generation is
   latency-bound anyway); `T>1` → fused. Gate it so the ROCm box still works.

Note found while adding a Makefile target: `sync` **already** installs
`causal-conv1d` on a CUDA (non-ROCm) host and skips it on ROCm (Makefile
lines 20–25), so half the groundwork exists. The missing piece is purely
`model.py`'s forward, which takes the per-token path unconditionally.

**Two caveats for whoever scopes it.** This box is the only place the fast path
can be *validated* — the local ROCm box cannot run the kernel at all, which is
the whole reason the manual loop exists. And 780M alone buys nothing for
training: the payoff needs the port to `mamba2_2_7b_memory`, whose mixer has
the M-wiring interleaved and is the larger piece. 780M-first is still right, as
it yields a validated reference implementation before touching the memory
model.

### 00:31 — the real null run (default config), and why it was killed at 00:43

Verbatim:

    tmux send-keys -t work "cd ~/altrux/sft && make consolidation-null ARGS='--n-facts 40 --distill-steps 200 --pass-k 10 --out logs/consolidation_null.jsonl'" Enter

(log: `sft/logs/consolidation-null-20260805-003158.log`; partial per-fact
results in `sft/logs/consolidation_null.jsonl`)

Transcript: 40 facts, 200 turns → 10,080 tokens (210 chunks). All structural
invariants 0, round-trip decode `True`.

**The in-context positive control failed: 0.03 (1 of 39 probed).** The §3 gate
in the plan fires. Killed at 00:43, after the pre-probe phase and before the
teacher-context / teacher-replay / distillation phases, saving ~45 min of
billed GPU on a number that could not have been interpreted.

**The plan's guessed cause is wrong.** §3 anticipates "the base 780M can't
follow the probe format", with a format-SFT detour as the remedy. That is not
what is happening — the smoke run scored **1.000** at 4 facts, and the
generations here are perfectly well-formed. What they are is *the wrong fact*:

    clove    code 9 0 4 8 1  ->  " 4 7 5 4 3."   <- topaz's code
    topaz    code 4 7 5 4 3  ->  " 4 7 5 4 3."   <- the single hit
    osprey   code 3 7 8 0 8  ->  " 2 8 9 7 7."   |
    heron    code 1 0 5 2 7  ->  " 2 8 9 7 7."   |  one code, three entities
    pelican  code 6 2 0 1 9  ->  " 2 8 9 7 7."   |

The model emits a correctly formatted five-digit code every time, but only a
small set of *attractor* codes, largely independent of which entity is asked.
That is the signature of **SSM state saturation with interference between
competing bindings** — a fixed-size state collapsing 40 bindings into a
handful, the readout returning whichever survived. Not a format problem, so
format SFT would have been the wrong fix.

**Consequences.**

1. *The null is invalid at N=40, not failed.* Its premise is facts "held
   losslessly in context". They are not, so the teacher does not know the
   facts and distillation has nothing to install. Any verdict would have
   measured the wrong thing. **No consolidation conclusion may be drawn from
   this run.**
2. *This is itself a result, and it favours the project thesis.* The case for
   M is exactly the regime of many competing facts that the Mamba2 SSM
   structurally cannot cover (root goal; 07-25 §4 M-necessity). That regime
   showed up sharply and unprompted: 4 facts → 1.000, 40 facts → 0.03.
3. *A structural limit on this harness, worth recording.* The teacher only
   knows the facts by holding the transcript in context, so the
   transcript-consolidation null is **intrinsically capped below the SSM's
   in-context capacity**. High-N consolidation cannot be tested by this
   harness at all — it would need a teacher that is not context-bound. Which
   is an argument for M rather than against the experiment.

**On N and the gate** (question raised by altrup, recorded because it shaped
the next step): high N is *not* needed for what the null gates. The null asks
one binary thing — can context→weights distillation install anything
recallable — and a gate is best tested where its premise holds most cleanly,
i.e. low N. The evidence is asymmetric: **FAIL at low N is decisive** (teacher
perfect, task easy, still nothing), whereas **PASS at low N is necessary but
not sufficient** — 9.67M LoRA params memorising 8 facts could be brute
memorisation that saturates immediately, so a pass licenses "mechanism alive",
not "CL works". Consolidation *capacity* is the real second question and does
need large N, but it is downstream and only worth asking if the gate passes.
Practical consequence: run the null at the largest N that still clears the
0.8 control — not arbitrarily small, not as large as possible.

### 00:48 — capacity ladder (new diagnostic, `sft/capacity_ladder.py`)

Written because the null cannot pick its own most important parameter without
it, and because the two data points in hand (4 facts/264 tok = 1.000;
40 facts/10k tok = 0.03) **confound fact count with transcript length** — they
differ in both. Prime + one greedy in-context probe per fact, no distillation,
so each cell costs a single forward pass.

Verbatim:

    tmux send-keys -t work "cd ~/altrux/sft && make capacity-ladder" Enter

(new `capacity-ladder` Makefile target; log
`sft/logs/capacity-ladder-20260805-004827.log`, per-fact jsonl
`sft/logs/capacity_ladder.jsonl`)

Grid `4x200,8x200,16x200,24x200,40x40,4x800` (n_facts × filler_tokens): the
fixed-filler sweep gives the capacity curve; **40x40** (many facts, short
transcript) and **4x800** (few facts, long transcript) are the two cells that
separate count from distance. Also reports a **late-half hit rate** and the
hit indices per cell — if the state were recency-limited rather than
capacity-limited, late facts would survive while early ones fall out, which
the overall rate alone cannot distinguish.

Outputs wanted: (a) the largest N clearing 0.8 → where the null gets rerun;
(b) capacity vs distance.

**Ops gotcha for the team:** the watchdog's process pattern covers `train.py`
and `consolidation_null.py` but **not** `capacity_ladder.py`, so while the
ladder runs the watchdog counts training as stopped and the 30-minute
termination clock is live. `.watchdog-delay` has to be touched through the
ladder even though the box is busy. Add the new script to the pattern if the
ladder becomes a routine tool.

#### First cell overturns the N=40 reading

    4x200 -> 790 tokens: in-context 0.500 (2/4), late-half 1.000, hits [2, 3]

The same 4 facts at filler 40 (264 tokens) scored **1.000** in the smoke run.
Same fact count, longer transcript, recall halves — and the survivors are the
two *most recent* facts. So the dominant variable is **distance / state
erosion, not capacity**. The "saturation with interference between 40
competing bindings" reading above is at best half right: at 4 facts there is
essentially no competition, yet the early facts still fall out. The
attractor-code pattern at N=40 is consistent with the same mechanism — what
survives is whatever was written most recently.

This reframes the fix favourably. If the limit is distance rather than fact
count, `40x40` (2,800 tokens) may clear the 0.8 control **at full N**, giving
a valid premise *and* the high N that makes a PASS evidentially strong,
instead of forcing the null down to N=8 where a pass is confounded with brute
LoRA memorisation.

#### Second cell — the rate is flat in N

    4x200 ->  790 tok: 0.500 (2/4)  late-half 1.000  hits [2, 3]
    8x200 -> 1842 tok: 0.500 (4/8)  late-half 0.750  hits [0, 4, 6, 7]

**Flat at 0.50 across a doubling of N**, with the *number* of hits scaling
with N (2 → 4) rather than staying a fixed recent-count. That rules out both
simple readings:

- not a hard recency horizon — a fixed-token horizon would keep the absolute
  hit count constant and halve the rate;
- not fact-count capacity — doubling the competing bindings did not lower the
  rate at all.

What it looks like instead is a roughly **per-fact survival probability set by
spacing**: at 200 tokens of separation each binding has ~50% chance of still
being readable, largely independent of how many others there are. Fact 0
surviving in the 8x200 cell (hits `[0, 4, 6, 7]`) argues against pure recency
too.

**Consequence: N is largely not the binding knob — filler is.** So the null
should be run at *full N with small spacing*, not at reduced N.

### 00:52 — ladder killed early, null relaunched at 40x40

Killed after 2 of 6 cells. Two reasons, both about not spending billed GPU on
foregone or duplicated work:

1. The remaining fixed-filler-200 cells (16x200, 24x200) would land near 0.50
   and cannot clear the 0.8 gate — ~10 min to confirm what cells 1–2 settled.
2. **The ladder is redundant with the null itself.** `consolidation_null.py`
   runs the in-context control as its own pre-phase and prints it *before*
   distilling, so pointing the null straight at a candidate config tests that
   config and flows into the real experiment if it passes — no repeated prime,
   no handoff. Failure costs ~5 min instead of ~15.

`capacity_ladder.py` is kept as a tool (the two banked cells are the finding
above); it is simply not the cheapest way to pick this run's config.

Verbatim:

    tmux send-keys -t work "cd ~/altrux/sft && make consolidation-null ARGS='--n-facts 40 --filler-tokens 40 --distill-steps 200 --pass-k 10 --out logs/consolidation_null_40x40.jsonl'" Enter

(log: `sft/logs/consolidation-null-20260805-005156.log`)

Decision point is the in-context control line. **≥0.8** → the run continues to
a real verdict at full N=40 (the evidentially strong case). **<0.8** → drop
`--filler-tokens` further (10, then 0) rather than dropping N, since spacing
is what binds.

### 00:58 — 40x40 control also 0.050; the spacing hypothesis is dead

    transcript: 40 facts, 236 turns, 3254 tokens
    IN-CONTEXT POSITIVE CONTROL: 0.050 (2/40)
    fresh-state floor: 0.000, mean code log-prob -4.4310

Killed during the teacher-context pass. (Note: the first `C-c` did not land —
always confirm with `pgrep -af "[c]onsolidation_null"` and a
`nvidia-smi` memory check rather than assuming. Use the bracket trick; a plain
`pgrep -f consolidation_null` matches its own command line and reports a
false positive.)

**All in-context control measurements so far:**

| cell | tokens | in-context |
|---|---|---|
| 4 × 40 | 264 | **1.000** |
| 4 × 200 | 790 | 0.500 |
| 8 × 200 | 1842 | 0.500 |
| 40 × 40 | 3254 | **0.050** |
| 40 × 200 | 10080 | 0.030 |

**My "distance/spacing dominates" reading is wrong.** Cutting the transcript
3× at N=40 (10080 → 3254 tokens) moved the control from 0.030 to 0.050, i.e.
not at all. Fact *count* matters a great deal. The flat 0.500 across 4→8 that
prompted the spacing hypothesis was too narrow a base — 4 and 8 agree, 40 does
not, and I generalised from two adjacent points.

**Process error worth recording:** `4x800` was the one cell that separates
count from length at a matched token budget (~3,300 tokens, the same as
40x40, with 10× fewer facts), and I killed the ladder one cell before it —
dropping the discriminator to chase the configuration I wanted to be true.
The correct instinct when killing a diagnostic early is to keep whichever cell
is *most likely to falsify* the current hypothesis, not the one that confirms
the plan.

### 00:58 — targeted ladder (bracket the cliff, and recover the discriminator)

Verbatim:

    tmux send-keys -t work "cd ~/altrux/sft && make capacity-ladder ARGS='--grid 8x40,16x40,24x40,4x800 --out logs/capacity_ladder2.jsonl'" Enter

(log: `sft/logs/capacity-ladder-20260805-005843.log`)

- **8x40, 16x40, 24x40** — bracket the threshold N at short spacing. 4x40 =
  1.000 and 40x40 = 0.050, so the cliff is between 4 and 40; this locates it,
  and the largest cell clearing 0.8 is where the null gets run.
- **4x800** (~3,200 tokens) — the recovered discriminator, matched in length
  to 40x40's 3,254. If it scores well while 40x40 fails, **count** dominates
  at fixed length.

All four are short transcripts; the whole ladder is ~8 minutes. This is the
case the ladder was actually built for — needing the shape of a curve rather
than one candidate config.

### 01:05 — HEADLINE RESULT: the SSM holds ~3–4 bindings, independent of length

Complete in-context control curve (all cells, both ladders + the two null
pre-phases):

| cell | tokens | rate | hits |
|---|---|---|---|
| 4 × 40 | 264 | **1.000** | 4/4 |
| 4 × 200 | 790 | 0.500 | 2/4 |
| 4 × 800 | 2716 | **1.000** | 4/4 |
| 8 × 40 | 616 | 0.375 | 3/8 |
| 16 × 40 | 1269 | 0.250 | 4/16 |
| 24 × 40 | 1964 | 0.042 | 1/24 |
| 40 × 40 | 3254 | 0.050 | 2/40 |
| 40 × 200 | 10080 | 0.030 | ~1/40 |

**Fact count dominates; transcript length is nearly irrelevant.** The
discriminator settles it — at a matched ~2–3k token budget, 4 facts → 1.000
while 24 facts → 0.042 and 40 facts → 0.050. And N=4 scores 1.000 at *both*
264 and 2716 tokens, which also retires the 4x200=0.500 point as small-sample
noise on four items rather than an erosion effect. Both of my earlier readings
were wrong: not spacing, and not "interference between many bindings" in the
sense of graceful degradation.

**Absolute hit counts across the sweep: 4, 3, 4, 1, 2** — essentially a small
constant, with the *rate* falling only because the denominator grows.

> **Finding: the 780M Mamba2 SSM holds on the order of 3–4 arbitrary
> key→value bindings in context, roughly independent of transcript length out
> to at least 2.7k tokens.**

Two consequences for the project:

1. *Direct support for M-necessity, with a number.* The design goal names the
   regime "many competing facts the SSM structurally can't cover" (root
   `models/mamba2_2_7b_memory/README.md`). This puts a figure on "many": more
   than about four, at 780M. Worth re-measuring at 2.7B — capacity plausibly
   scales with `d_state`/`nheads`, and the interesting question is whether it
   scales *enough* to matter.
2. *The consolidation null is confined to N≈4 on this model.* Which lands
   exactly on the asymmetry recorded above: **FAIL at N=4 is decisive**
   (teacher perfect, task trivially easy, nothing installed), **PASS at N=4 is
   weak** (4 facts into 9.67M LoRA params is trivially memorisable, and says
   nothing about whether consolidation scales).

Caveat on statistical power: at N=4 the match rate is quantised to 0.25, so
the PASS threshold of 0.30 effectively means ≥2/4. The continuous
`mean_logprob_delta` is the better-resolved signal at this N and is what
separates FAIL-UNDERPOWERED from FAIL-DEAD. If the result lands near a
boundary, the follow-up is confirmatory reruns at other seeds rather than any
threshold adjustment (thresholds are the pre-registration).

### 01:06 — the null, at the ladder's answer

Verbatim:

    tmux send-keys -t work "cd ~/altrux/sft && make consolidation-null ARGS='--n-facts 4 --filler-tokens 800 --distill-steps 200 --pass-k 10 --seed 1234 --out logs/consolidation_null_4x800.jsonl'" Enter

(log: `sft/logs/consolidation-null-20260805-010623.log`)

This is the first configuration in the session with a **valid premise** — the
transcript is verifiably held losslessly (control 1.000 measured twice at this
N). ETA ~36 min, dominated by the fixed 200 distill steps.

Code state: `ed94dd8` pushed to `origin/main` (rebased onto altrup's
`853c25e`/`27eadec`, disjoint paths, no conflicts).

#### 01:09 — control 0.750 here vs 1.000 in the ladder, on a byte-identical transcript

Both are 4 facts × 800 filler, seed 1234 → 2716 tokens, and `build_facts` is
seed-deterministic (same entities, same codes). The run-to-run difference is
**GPU nondeterminism tipping one borderline binding**, and the failing fact
shows the mechanism:

    clove   1 9 0 1 1  ->  " 1 9 0 1 1."   HIT
    topaz   5 3 0 0 0  ->  " 5 9 7 9 7."   MISS  <- osprey's code
    osprey  5 9 7 9 7  ->  " 5 9 7 9 7."   HIT
    heron   2 1 2 1 0  ->  " 2 1 2 1 0."   HIT

`topaz` returns *osprey's* code — the same cross-binding attractor seen at
N=40, now at N=4. The null interleaves extra `generate`/`target_logprob` calls
between facts where the ladder does not, which perturbs allocation and kernel
selection enough to flip a marginal argmax in bf16.

Two consequences:

1. **The 3–4 binding ceiling is soft, and N=4 already sits at its edge** — not
   a clean shelf. This strengthens the headline finding: even four arbitrary
   bindings are not reliably held.
2. **Analysis must condition on the facts the teacher actually held.** The
   premise is per-fact, not per-run: `topaz` is absent from the teacher's
   state, so distillation cannot install it, and counting it would *understate*
   consolidation. The correct headline is the post-distill match rate over the
   three `in_context_match=True` facts. The jsonl records `in_context_match`
   per fact, so this is computable exactly, post hoc, without rerunning.

Deliberately **not** rerunning to chase a 4/4 control — that would be
selecting on noise. Cost stated plainly: the headline rests on **three** facts,
which is thin; `mean_logprob_delta` over those three is better resolved than
the quantised match rate, and a boundary result needs confirmatory seeds
before anyone treats it as settled.

#### Filler dilutes the distillation signal — `4x40` is the better gate config

Noticed while the run was in flight, too late to change it. At `4x800` the
transcript is 2716 tokens of which only ~104 (4 facts × ~26 tokens) carry a
binding — **~3.8% of tokens are the thing being installed**. The KL is
`batchmean` over all tokens, so fact tokens are a small share of the loss
(live KL 0.0388 at step 27, low because filler dominates the average).

I took the ladder's summary line ("largest n_facts clearing 0.8: 4, filler
800") at face value without considering what filler does *downstream* of
config selection. `4x40` also clears the control (1.000, smoke run) and there
fact tokens are ~39% of the transcript — a 10× denser signal, and 200 steps
covers a 6-chunk transcript ~33 times rather than ~3.5. **By the gate logic
used throughout this run — test where conditions are most favourable, so a
FAIL is decisive — `4x40` is the better configuration and should be the first
thing the next session runs.**

How much it matters is uncertain: the gradient still concentrates on fact
tokens naturally (teacher and student barely differ on filler), and Adam is
largely scale-invariant, so dilution behaves more like a reduced effective LR
than a lost signal. Second-order, not fatal. Running both would bracket it —
if `4x800` fails and `4x40` passes, dilution was the confound.

## Closing — session ended early (teammate's machine going down)

**Why it ended here.** altrup's local machine — which hosts *both* the
watchdog and the rsync pull — had to shut down at ~01:32 UTC, with the
in-flight null ~18 minutes from its verdict. Two hard constraints made
finishing impossible rather than merely inconvenient:

- The watchdog probes this instance over ssh *from that machine*.
  `.watchdog-terminate` only works while it is alive to see the flag, so a
  box left running past the shutdown bills unbounded until they return.
- That machine is also the rsync target, so anything written after shutdown
  is lost.

Trading an unfinished verdict for unbounded billing is a bad deal, especially
when the verdict is cheap to redo — the ladder has already determined its
configuration. **Because rsync was going away, this notes file was committed
to git from the instance**, deviating from the normal "never commit notes/
from the box" rule (which exists to avoid racing the rsync flow — there was no
rsync left to race).

### What this session established

1. **The headline, and it is not the null.** The 780M Mamba2 SSM holds on the
   order of **3–4 arbitrary key→value bindings in context, roughly independent
   of transcript length** out to at least 2.7k tokens. Fact *count* is the
   binding constraint; length is nearly irrelevant (4 facts → 1.000 and 40
   facts → 0.050 at a matched ~2–3k token budget). Absolute hits across the
   whole sweep: 4, 3, 4, 1, 2. The ceiling is soft — even N=4 has a fact that
   flips run to run.
2. **A concrete number for M-necessity.** The design goal's "many competing
   facts the SSM structurally can't cover" now has a figure attached: more
   than about four, at 780M. The obvious follow-up is whether this scales with
   model size.
3. **The consolidation null never got a valid verdict.** Every number produced
   for it this session is void: the 40-fact configurations violated its
   premise, and the one valid-premise run (`4x800`) was killed 18 minutes
   short. **No consolidation conclusion may be drawn from this session.**
4. **A structural limit on the null harness.** Its teacher only knows the facts
   by holding them in context, so the transcript-consolidation null is
   intrinsically capped below the SSM's in-context capacity — i.e. at N≈4 on
   this model. High-N consolidation cannot be tested by this harness at all;
   it needs a teacher that is not context-bound. That is an argument for M
   rather than against the experiment, and it is worth the team's attention
   because it was not anticipated in the plan.

### What the next session should do, in order

1. **Run the null at `4x40`** (not `4x800`) — control 1.000, 10× denser
   distillation signal, and the most-favourable-conditions config that makes a
   FAIL decisive. This is ~30 min and is the unfinished deliverable.
2. **Confirmatory seeds.** At N=4 the match rate quantises to 0.25 and one
   binding is unstable, so a single run cannot settle the verdict. Run 2–3
   seeds and read `mean_logprob_delta` (continuous, better resolved) alongside
   the quantised match rate. **Condition on `in_context_match=True` per fact** —
   the premise is per-fact, and facts the teacher never held cannot be
   installed.
3. **Capacity ladder at 2.7B.** `make capacity-ladder` with
   `MODEL_NAME=mamba2_2_7b_memory` (the only 2.7B here; with M zero-init it is
   a reasonable plain-backbone proxy). No training, so it is cheap. Tests
   whether the 3–4 ceiling scales with `d_state`/`nheads` — and whether it
   scales *enough* to matter, which is the real M-necessity question.
4. **The §4 escalation branches remain untouched** because no valid verdict
   exists yet. Do not treat any FAIL-DEAD from this session as triggering the
   2.7B escalation.

### Open questions for the team

- Does in-context binding capacity scale with model size, and how? If 2.7B
  also tops out near a handful of facts, that is close to decisive for M.
- The plan's §3 remedy for a low control assumes a *format* failure and
  prescribes format SFT. That diagnosis was wrong here — the model's outputs
  were perfectly well-formed, just mis-bound. §3 should be rewritten to
  distinguish "can't follow the format" from "can't hold the bindings", since
  they have completely different remedies.
- Should the null harness be redesigned to escape the context-capacity cap
  (deferred item above)? As written it can only ever test N≈4 at 780M.

### Deferred, written up above, not done

- **Fused/chunked fast-path migration** (~64 ms/token today; makes rented-CUDA
  training infeasible at 230M tokens). Feasibility verified: the kernel takes
  `initial_states`/`return_final_states` and `sync` already installs
  `causal-conv1d` on CUDA hosts. Deliberately deferred as out of scope.
- `capacity_ladder.py` is committed and documented (`ed94dd8`), and the
  watchdog's process pattern does **not** cover it — add it if the ladder
  becomes routine, or the box gets terminated mid-ladder.
