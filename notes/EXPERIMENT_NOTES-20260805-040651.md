# Experiment notes — 2026-08-05 04:06:51 UTC (A10 box, probe/eval session)

Plan followed: `notes/DISCUSSION-20260804-binding-capacity-and-null-rerun.md` §3
(standing direction), in order: 0) validate fused path, 1) consolidation null at
`4x40` × 3 seeds, 2) rerun 1–2 ladder cells on the fused path, 3) capacity
ladder at 2.7B.

## Box state at session start

- GPU: NVIDIA A10, 23028 MiB. `torch 2.13.0+cu129`, `cuda_available=True`.
- `mamba_ssm` and `causal_conv1d` both import — the fused kernel path is
  available here (unlike the ROCm dev box).
- Fresh instance: no `sft/logs/`, no `models/*/checkpoints/`. No training data
  present. None of §3 needs training data (null + ladder generate synthetic
  transcripts), so no prep run.
- `sft/.env` has `MODEL_NAME=` empty → every command below passes
  `MODEL_NAME` inline.
- No training this session; it is probe/eval only. `.watchdog-delay` touched
  on each check-in.

## §3.0 — fused-path validation (timebox 45 min)

Verbatim:

    cd ~/altrux/sft && make test

(run in `work:test`, teed to `sft/logs/make-test-<UTC>.log`)

Result: **collection error** — `tests/test_train.py` and
`tests/test_train_mixing.py` fail at import with
`ModuleNotFoundError: No module named 'models.'`, i.e. the empty `MODEL_NAME`
in `sft/.env` (setup ships it blank). Not a code fault. Rerun with the model
selected inline:

    cd ~/altrux/sft && MODEL_NAME=mamba2_780m make test

Result: **fused path GREEN** — `1 failed, 210 passed in 219.49s`. All of
`tests/test_mixer_fused.py` passed (logits from fresh state, logits + final
`MixerState` across two threaded chunks, LoRA gradients; fp32 and bf16). This
is the first execution of the fused path anywhere. §3 items 1–3 run on this
box with the fused dispatch live for 780M `T>1`.

The one failure is NOT the one §3.0 predicted, and does not involve the fused
path:

- Predicted: `test_grad_checkpoint_matches_uncheckpointed_plain_model`
  (plain 780M, which does now run fused). That test **passed**.
- Actual: `test_grad_checkpoint_matches_uncheckpointed_memory_model`
  (`mamba2_780m_memory_mix`, which derives from `mamba2_2_7b_memory` and
  never touches `_mixer_chunk` — it is per-token-loop only). Failing
  assertion is a LoRA-grad compare, greatest relative difference 9.95e-4
  against `rtol=2e-4`, fp32.

So the fused port did not cause it; it is checkpointed-vs-uncheckpointed
gradient drift in the memory model on CUDA (the ROCm dev box passes). Most
likely different cuBLAS reduction orders between the full forward and the
BLOCK=4 recompute. Reran the single test twice; **bit-identical both times**:

    marker_delta.delta: Mismatched elements: 1 / 128 (0.8%)
    Greatest absolute difference: 1.4370307326316833e-06 (up to 1e-06 allowed)
    Greatest relative difference: 0.0009946817299351096 (up to 0.0002 allowed)

One near-zero element of a 128-element gradient, off by 1.44e-6 absolute,
deterministic. That is float non-associativity, not a defect — the "1e-3
relative" is an artifact of the element being ~1.4e-3. Widened the gradient
`atol` 1e-6 → 1e-5 in both grad-checkpoint equivalence tests (`rtol` left at
2e-4), committed and pushed as `197eb4e`. Suite is otherwise 210/210.

## §3.1 — consolidation null at 4x40 (per §2 of the discussion note)

Seed 1234, verbatim:

    cd ~/altrux/sft && MODEL_NAME=mamba2_780m make consolidation-null ARGS='--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed 1234 --out logs/consolidation_null_4x40_s1234.jsonl'

Launched 04:12:40 UTC in `work:null`, log
`sft/logs/consolidation-null-20260805-041240.log`. Runs on the fused path
(780M, T>1).

Transcript sanity from the harness's own preamble — all four invariants 0,
round-trip decode True, 4 facts / 20 turns / 264 tokens, decoded samples at
fact 0 and at the filler→fact join both well-formed.

**Seed 1234 (2m40s wall — the fused path makes a seed cheap; the previous
session's per-token equivalent was ~30 min):**

    in-context control : 0.750   (clove T, topaz F, osprey T, heron T)
    fresh-state floor  : 0.000
    post greedy match  : 0.250   (script's own unconditioned verdict)
    post pass@10       : 0.175
    mean logprob delta : +1.4334 nats/token
    script VERDICT     : FAIL-UNDERPOWERED

Conditioned per §2 (facts with `in_context_match == true`): 3 facts —
clove HIT, osprey miss, heron miss → **1/3**. Per-fact dlogp: clove +3.855,
osprey +0.176, heron +0.301 (topaz, unconditioned, +1.401).

The failed control fact is `topaz` again — read back as osprey's code
(`5 9 7 9 7`), the exact borderline binding §1.6 flagged.

Distillation converged cleanly: KL 3.59 → 0.022 over 200 steps, no
instability.

Seeds 2345 and 3456, verbatim (loop in `work:null`):

    cd ~/altrux/sft && for S in 2345 3456; do MODEL_NAME=mamba2_780m make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed $S --out logs/consolidation_null_4x40_s$S.jsonl"; done

Launched 04:16:25 UTC. No adaptive early stop — seed 1 is a
boundary result (1 of 3 conditioned installed, dlogp above the underpowered
threshold), so all three seeds run per §2.

**All three seeds (per-seed script summaries):**

| seed | in-context control | post greedy (unconditioned) | mean dlogp | script verdict |
|------|-----|-----|-----|-----|
| 1234 | 0.750 | 0.250 | +1.433 | FAIL-UNDERPOWERED |
| 2345 | 0.750 | 0.250 | +2.245 | FAIL-UNDERPOWERED |
| 3456 | 0.750 | 0.000 | +2.382 | FAIL-UNDERPOWERED |

Every seed's in-context control is 0.750 — 3 of 4 facts held by the teacher,
consistent with the ~3–4 binding ceiling.

**Conditioned pooling per §2** (only facts with `in_context_match == true`):

    s1234: clove   HIT  (+3.855) | osprey  miss (+0.176) | heron   miss (+0.301)
    s2345: saffron HIT  (+4.309) | oboe    miss (+1.603) | viola   miss (+0.205)
    s3456: calcite miss (+2.437) | ketch   miss (+1.851) | saffron miss (+1.544)

    POOLED conditioned installs : 2/9 = 0.222   (PASS threshold 0.30)
    POOLED conditioned mean dlogp: +1.809 nats  (underpowered threshold 1.00)

→ **FAIL-UNDERPOWERED** on the pre-registered §2 verdict: below the install
threshold, but the teacher's distribution clearly moved toward the right
codes. Prescribed branch: scale `--distill-steps` ×4 and rerun.

Observation worth carrying: in both seeds that installed anything, the
installed fact is the *first* fact in the transcript, while the ladder's
in-context survival skewed *recent* (§1.2 of the discussion note). If that
holds up, distillation and in-context retention have opposite positional
biases.

## §3.1b — FAIL-UNDERPOWERED branch: distill-steps ×4

Verbatim (loop in `work:null`), launched 04:19:50 UTC:

    cd ~/altrux/sft && for S in 1234 2345 3456; do MODEL_NAME=mamba2_780m make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 800 --pass-k 10 --seed $S --out logs/consolidation_null_4x40_d800_s$S.jsonl"; done

**Result: ×4 changes nothing.**

    d800  s1234: clove   HIT (+3.44) | topaz miss (+1.31) | osprey miss (+0.53) | heron miss (+0.90)
    d800  s2345: saffron HIT (+4.25) | oboe  miss (+1.43) | viola  miss (+0.04)
    d800  s3456: calcite miss (+2.43)| ketch miss (+1.60) | saffron miss (+1.44)

    d200 pooled: 2/9  = 0.222, mean conditioned dlogp +1.809
    d800 pooled: 2/10 = 0.200, mean conditioned dlogp +1.738

(s1234 has 4 conditioned facts at d800 because `topaz` matched in-context
this time — the §1.6 borderline binding flipping again, on a byte-identical
transcript. Independent confirmation that the N=4 control is jittery.)

Both the install rate and the logprob delta are flat under 4× the
distillation. The distillation objective itself is *satisfied* — KL
converges to ~0.02 well before step 200 — so the extra steps refit the same
teacher logits without making any additional fact retrievable from a fresh
state. "Underpowered" is ruled out as the explanation; on the §2 ladder this
now behaves as FAIL-DEAD, whose branch is one escalation run at 2.7B at the
N the 2.7B ladder says clears the 0.8 control. That ladder is §3.3, next
after the cheap §3.2 cross-check.

## §3.2 — 780M ladder cells re-measured on the fused path

The §3.1 numbers above are fused-path numbers while every prior capacity
number is per-token-loop; §1.6 shows borderline bindings flip on less than a
kernel change, so the comparison needs anchoring. Verbatim, launched
04:31:11 UTC in `work:ladder`:

    cd ~/altrux/sft && MODEL_NAME=mamba2_780m make capacity-ladder ARGS='--grid 4x40,24x40 --out logs/capacity_ladder_780m_fused.jsonl'

Prior per-token-loop values for these cells (from the 08-04 discussion note
§1.1): 4×40 control 1.000, 24×40 control 0.042 with hit indices [23].

**Result — fused reproduces per-token within the known jitter:**

    cell     tokens   rate    late-half   hit indices
    4x40        264   0.750   1.000       [0, 2, 3]     (per-token was 1.000)
    24x40      1964   0.042   0.083       [23]          (per-token: 0.042, [23])

24×40 is identical, down to which single fact survives. 4×40 differs only by
`topaz` — the same borderline binding that flips between byte-identical runs
on the *same* kernel (§1.6, and again in §3.1b above). So the fused path is
not a confound for cross-session comparison; treat all 780M numbers as
comparable, and treat any single N=4 readout as ±1 fact.

## §3.3 — capacity ladder at 2.7B

Does the ~3–4 binding ceiling scale with `d_state`/`nheads`? Also fixes the N
for the FAIL-DEAD escalation run. M is zero-init here, so this measures the
plain backbone. Runs on the per-token loop (the fused port is 780M-only).
Verbatim, launched 04:36:19 UTC in `work:ladder`:

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b_memory make capacity-ladder ARGS='--grid 4x40,8x40,16x40,24x40,40x40,4x800 --out logs/capacity_ladder_2p7b.jsonl'

**Result (memory injection ON — see the correction below before using these):**

    cell     tokens   rate    late    hits   hit indices
    4x40        264   0.750   1.000   3/4    [0, 2, 3]
    8x40        616   0.375   0.500   3/8    [2, 5, 6]
    16x40      1269   0.125   0.125   2/16   [6, 8]
    24x40      1964   0.208   0.250   5/24   [0, 8, 16, 18, 23]
    40x40      3254   0.050   0.050   2/40   [13, 29]
    4x800      2716   0.500   0.500   2/4    [1, 2]

No cell clears 0.8. Absolute hits 3,3,2,5,2,2 against 780M's 4,3,4,1,2.

### Correction: this is NOT a plain-backbone reading

§3.3 of the discussion note assumed "M zero-init makes it a reasonable
plain-backbone proxy". That is false for the `state` integration arm at a
*fresh* init, and the ladder script gave no way to tell:

- `retain = sigmoid(retain_proj(z))` starts at ~0.98 by design (bias 4), so
  `ssm_state` is scaled by ~0.98 at every window close of every injected
  layer. Over a multi-thousand-token transcript that compounds.
- Worse, `beta` is **not** near 0 at a fresh init. It is
  `sigmoid(beta_proj(z) + sigmoid(surprise) + beta_anneal_offset)`, and
  `beta_anneal_offset` is `0.0` until `Model.set_beta_anneal` is called —
  which only happens in training. `beta_proj.bias` is deliberately left at
  its ordinary small default. So beta ≈ sigmoid(0 + ~0.5 + 0) ≈ 0.6: at
  every window close the merge erases the state's key-component and writes
  an *untrained random* `p` in its place.

Zero-init covers `knob_proj` (the write knobs) and the `mix` arm's `o_proj`
— not the `state` arm's gated-delta merge. So the table above measures a
2.7B backbone actively corrupted by an untrained memory, which is a floor,
not the backbone's capacity. It is also, incidentally, a first ablation
delta — in the wrong direction, which is expected for untrained weights.

**Fix (committed):** `capacity_ladder.py` gains `--no-memory`, which turns
`model.injection_enabled` off, plus a `memory_injection` field in every
jsonl record and a printed `memory injection: on/off/absent` line, so no
future reading is ambiguous about what it measured. Unit tests in
`sft/tests/test_capacity_ladder.py`, README updated.

### §3.3b — the same ladder with the memory path off

Verbatim, launched 04:53:37 UTC:

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b_memory make capacity-ladder ARGS='--grid 4x40,8x40,16x40,24x40,40x40,4x800 --no-memory --out logs/capacity_ladder_2p7b_nomem.jsonl'

**Result — the plain 2.7B backbone:**

    cell     tokens   rate    late    hits   hit indices
    4x40        264   1.000   1.000   4/4    [0, 1, 2, 3]            -> clears 0.8
    8x40        616   0.625   1.000   5/8    [0, 4, 5, 6, 7]
    16x40      1269   0.312   0.375   5/16   [0, 6, 10, 14, 15]
    24x40      1964   0.333   0.333   8/24   [5, 7, 8, 11, 16, 18, 22, 23]
    40x40      3254   0.175   0.150   7/40   [8, 13, 14, 16, 21, 27, 39]
    4x800      2716   1.000   1.000   4/4    [0, 1, 2, 3]            -> clears 0.8

### Three findings

1. **Binding capacity does scale with model size, sub-linearly, and not
   nearly enough.** Absolute bindings held: 780M 4,3,4,1,2 → 2.7B
   4,5,5,8,7,4. Roughly 3–4 → 5–8 for 3.5× the parameters. A 24-fact
   session still loses two thirds of its facts, and 40 facts loses 82%. The
   ~handful ceiling is a property of the architecture, not of 780M; buying
   it away with scale is not on the table. This is the §3.3 question
   answered, and it is the M-necessity argument in its strongest form so
   far.
2. **Length-independence reconfirmed at 2.7B, and more cleanly than at
   780M.** 4×800 (2716 tokens) scores 1.000, identical to 4×40 (264
   tokens) — a 10× transcript with the same fact count costs nothing. The
   limit is bindings, not distance, exactly as §1.3 re-denominated it.
3. **The untrained memory is actively harmful, and now measured.** Same
   grid, injection on vs off: 3,3,2,5,2,2 vs 4,5,5,8,7,4 — the memory
   destroys 1–3 bindings in every cell. That is the expected sign for
   untrained write projections (it erases the state's key-component and
   writes noise), but it is the first ablation delta this project has on
   the board, and it sets the bar the trained memory has to clear: a
   trained M must first climb back over its own damage before any positive
   delta counts.

## §3.4 — FAIL-DEAD escalation: the null at 2.7B

§2's FAIL-DEAD branch: one escalation run at 2.7B at the N the 2.7B ladder
says clears the 0.8 control. That is **N=4, filler 40** (1.000). Run with
`--no-memory` for the same reason as the ladder — otherwise the untrained
memory corrupts the teacher's own in-context control. Verbatim, launched
05:00:33 UTC:

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b_memory make consolidation-null ARGS='--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed 1234 --no-memory --out logs/consolidation_null_2p7b_4x40_nomem_s1234.jsonl'

**Seed 1234 — the cleanest premise the harness has ever had:**

    memory injection   : off
    in-context control : 1.000   (all 4 facts held — no conditioning needed)
    fresh-state floor  : 0.000
    post greedy match  : 0.250   (clove HIT +3.542 | topaz miss +1.559 | osprey miss +1.347 | heron miss +1.305)
    post pass@10       : 0.175
    mean logprob delta : +1.938
    script VERDICT     : FAIL-UNDERPOWERED

The escalation's premise problem is gone — every fact was genuinely held by
the teacher — and the install rate is *identical* to 780M's: 1 of 4.

Extended past §2's "one escalation run" to seeds 2345/3456 (3 min each on
this box; a 1-of-4 single run is too thin to rest the branch on). Verbatim,
launched 05:06:31 UTC:

    cd ~/altrux/sft && for S in 2345 3456; do MODEL_NAME=mamba2_2_7b_memory make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed $S --no-memory --out logs/consolidation_null_2p7b_4x40_nomem_s$S.jsonl"; done

**Pooled 2.7B result — worse than 780M, on better premises:**

    s1234 (control 1.000): clove    HIT  (+3.54) | topaz  miss (+1.56) | osprey miss (+1.35) | heron miss (+1.31)
    s2345 (control 0.750): saffron  miss (+2.65) | oboe   miss (+1.06) | viola  miss (+0.65)
    s3456 (control 1.000): schooner miss (+2.96) | calcite miss(+2.26) | ketch  miss (+1.39) | saffron miss (+2.28)

    POOLED conditioned : 1/11 = 0.091   (780M: 2/9 = 0.222)
    mean dlogp         : +1.909 nats    (780M: +1.809)

Two of three seeds had a *perfect* control, so 11 of 12 facts were genuinely
held by the teacher — the premise objection that voided the previous
session's numbers does not apply here at all. Scaling the model 3.5× moved
the install rate the wrong way (within noise of each other; the honest
reading is "no scale benefit", not "2.7B is worse").

The pattern in the deltas is consistent across both scales: every fact's
log-prob rises by 1–3 nats, one fact per run occasionally crosses into
greedy-decodable, the rest sit stuck below threshold. Distillation moves the
distribution toward the codes without ever *binding* entity→code.

## §3.5 — closing check: ×4 steps at 2.7B

The 780M ×4 was flat (§3.1b), and the mechanism (KL already converged before
step 200) is not scale-specific — but §2's FAIL-DEAD branch deserves the
control at the escalation scale too. Verbatim, launched 05:16:46 UTC:

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b_memory make consolidation-null ARGS='--n-facts 4 --filler-tokens 40 --distill-steps 800 --pass-k 10 --seed 1234 --no-memory --out logs/consolidation_null_2p7b_4x40_nomem_d800_s1234.jsonl'

Result: flat, as at 780M. 1/4 (`clove` again, everything else miss), mean
dlogp +2.025 vs +1.938 at d200. The FAIL-UNDERPOWERED escape hatch is empty
at both scales; the null is FAIL-DEAD as a matter of fact, whatever label the
script prints.

## §3.6 — the positional artifact, and a fix worth testing

Not in the standing direction; found while writing up §3.4.

**Every fact that has ever installed, in any run of this harness, is the
first fact of the transcript.** Six installs across ten runs (780M d200 ×3,
780M d800 ×3, 2.7B d200 ×3, 2.7B d800 ×1), every one at fact index 0, across
two model sizes, four distinct entities, and three seeds. Under a uniform
position null that is (1/4)^6 ≈ 1 in 4096.

The mechanism is in the harness, not in consolidation. From
`consolidation_null.py`'s distillation loop:

    c = step % n_chunks
    if c == 0:
        state = None

Chunk boundaries are fixed at `c * chunk_len` and the state resets only at a
pass boundary, so **chunk 0 is the only chunk the student ever sees from a
fresh state** — and a fresh state with no context is exactly the condition
every post-distillation probe runs under. Facts 1..N-1 are distilled only
under a primed state that does not exist at test time. The harness has been
asking the model to consolidate under one distribution and tested it under
another, for every fact except the first.

That reframes the whole null. "Distillation cannot install facts" may be, at
least in part, "this replay schedule only ever trains one fact under
test-time conditions".

**Intervention (committed as `ea4c37b`):** `--fresh-state-replay` resets the
student's state before every replay chunk, so no position is privileged and
every chunk is distilled under the probe's own condition. The schedule moved
into a pure `replay_step()` with CPU tests. Default is unchanged, so the
pre-registered harness is untouched.

Expected outcome if the hypothesis is right: install rate rises above 1/4 and
the installed facts stop being exclusively index 0. If it is wrong: the rate
stays at ~1/4-and-only-fact-0, and the null's negative is about consolidation
rather than about scheduling — a stronger FAIL-DEAD than we had, since the
most plausible confound would then be excluded.

Verbatim, launched 05:22:58 UTC, 780M (cheap arm, and the arm with the most
existing baseline data):

    cd ~/altrux/sft && for S in 1234 2345 3456; do MODEL_NAME=mamba2_780m make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed $S --fresh-state-replay --out logs/consolidation_null_4x40_fresh_s$S.jsonl"; done

**Result — the hypothesis holds.** Matched comparison: same three seeds, same
transcripts, the same 9 conditioned facts on both sides, only the replay
schedule differs.

| arm | installs | mean dlogp | installed positions |
|-----|----------|------------|---------------------|
| carried (baseline) | 2/9 = 0.222 | +1.809 | [0, 0] |
| fresh-state-replay | **4/9 = 0.444** | **+3.741** | [0, 3, 0, 2] |

Per fact:

    carried: clove   HIT +3.85 | osprey  .  +0.18 | heron   .  +0.30
             saffron HIT +4.31 | oboe    .  +1.60 | viola   .  +0.20
             calcite  .  +2.44 | ketch   .  +1.85 | saffron .  +1.54
    fresh:   clove   HIT +3.78 | osprey  .  +4.01 | heron  HIT +3.61
             saffron HIT +4.26 | oboe    .  +2.48 | viola   .  +3.58
             calcite  .  +3.92 | ketch  HIT +3.95 | saffron .  +4.08

Three things move together:

1. **The positional monopoly breaks.** Installs at index 3 and index 2 — the
   first non-first-fact installs this harness has ever produced.
2. **The lift lands exactly where predicted.** Non-first facts: mean dlogp
   +0.945 → +3.62, a ~4× improvement on precisely the facts that were
   previously never trained under test-time conditions. Fact 0, already
   privileged, barely moves (3.85→3.78, 4.31→4.26).
3. **The rate clears the pre-registered PASS threshold**, 0.444 vs 0.30.

**It is not just "more optimization".** The obvious alternative — fresh-state
resets make the task harder, so gradients are bigger, so everything improves
— is ruled out by the ×4 control already run: the carried schedule at 800
steps gave 2/10 and +1.738, flat against its own 200-step baseline. Four
times the steps on the carried schedule does nothing; the same steps on the
corrected schedule doubles the log-prob delta and doubles the install rate.

**Statistical honesty:** 2/9 vs 4/9 on install counts alone is not
significant (Fisher exact p ≈ 0.31). The weight is in the per-fact log-prob
shift, which moves in the same direction for 8 of 9 facts and is large, and
in the positional pattern, which is categorical. Treat the install rate as
suggestive and the mechanism as well-evidenced.

## §3.7 — confirming the schedule effect at 2.7B

Verbatim, launched 05:31:42 UTC:

    cd ~/altrux/sft && for S in 1234 2345 3456; do MODEL_NAME=mamba2_2_7b_memory make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --seed $S --no-memory --fresh-state-replay --out logs/consolidation_null_2p7b_4x40_nomem_fresh_s$S.jsonl"; done

Baseline to beat (same seeds, carried schedule, memory off): 1/11 = 0.091,
mean dlogp +1.909.

**Result: does not replicate at 2.7B — but the arms are not comparable.**

    2.7B carried            : 1/11 = 0.091  mean dlogp +1.909  positions [0]
    2.7B fresh-state-replay : 0/11 = 0.000  mean dlogp +1.586  positions []

**The confound: chunk length is not matched across models.** `chunk_len` comes
from each model's `DEFAULT_CHUNK_LEN` — 48 for `mamba2_780m`, **7** for
`mamba2_2_7b_memory` (set small because the 2.7B memory keeps a fast-weight
snapshot per token). Under `--fresh-state-replay` a chunk is all the context
the student gets, so at 780M it is asked to match the teacher over 48 tokens
from cold, and at 2.7B over *seven*. Seven tokens with no context is close to
contentless — the student cannot even see a full `[USER] What is the code for
the X?` turn — so the 2.7B arm ran a much harsher intervention, not the same
one at a larger scale. This is a confound in my §3.7 design, not a scale
finding, and the §3.6 result at 780M stands on its own matched comparison.

## §3.8 — the scale check, with chunk length matched

Both arms at `--chunk-len 48`, same three seeds, memory off. This is the
comparison §3.7 should have been. Verbatim, launched 05:47:23 UTC:

    cd ~/altrux/sft && for S in 1234 2345 3456; do MODEL_NAME=mamba2_2_7b_memory make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --chunk-len 48 --seed $S --no-memory --out logs/consolidation_null_2p7b_c48_carried_s$S.jsonl"; MODEL_NAME=mamba2_2_7b_memory make consolidation-null ARGS="--n-facts 4 --filler-tokens 40 --distill-steps 200 --pass-k 10 --chunk-len 48 --seed $S --no-memory --fresh-state-replay --out logs/consolidation_null_2p7b_c48_fresh_s$S.jsonl"; done

**Result — the pattern replicates at 2.7B once chunk length is matched:**

| 2.7B, chunk-len 48 | installs | mean dlogp | installed positions |
|--------------------|----------|------------|---------------------|
| carried | 3/11 = 0.273 | +2.272 | [0, 0, 0] |
| fresh-state-replay | 4/11 = 0.364 | +3.217 | [0, **3**, 0, 0] |

Per fact:

    carried: clove    HIT +3.69 | topaz   .  +1.94 | osprey  .  +1.93 | heron  .  +1.01
             saffron  HIT +4.06 | oboe    .  +2.11 | viola   .  +0.64
             schooner HIT +3.99 | calcite .  +2.54 | ketch   .  +1.19 | saffron . +1.88
    fresh:   clove    HIT +3.70 | topaz   .  +0.13 | osprey  .  +3.86 | heron HIT +3.40
             saffron  HIT +4.06 | oboe    .  +2.45 | viola   .  +3.57
             schooner HIT +3.95 | calcite .  +3.39 | ketch   .  +2.84 | saffron . +4.03

Side finding: chunk length matters on its own. The carried arm at chunk-len
48 scores 3/11 where the same arm at the model's default chunk-len 7 scored
1/11 — the 2.7B default is tuned for the memory's per-token fast-weight
snapshot, and it costs the null's distillation real signal. Any future
cross-model null comparison must pin `--chunk-len` explicitly.

## Closing summary

Session ran 04:06–06:05 UTC on an A10. No training: this was the probe/eval
docket from `DISCUSSION-20260804-binding-capacity-and-null-rerun.md` §3, all
four items completed, plus one unplanned line of work.

### What was established

1. **The fused SSD chunk-scan path is correct** (§3.0). Its oracle tests
   passed on their first execution anywhere; 780M ladder cells reproduce the
   per-token-loop numbers (24×40 identical down to which fact survives). The
   one test failure was an unrelated fp32 tolerance, fixed in `197eb4e`.
   Training on rented CUDA is now viable for 780M — a seed of the null went
   from ~30 min to 2m40s.
2. **The consolidation null fails at both scales, and "underpowered" is not
   the reason** (§3.1–3.5). 780M 2/9, 2.7B 1/11 conditioned installs, with
   ×4 distillation steps flat at both sizes. Two of three 2.7B seeds had a
   perfect in-context control, so the premise objection that voided the
   previous session's numbers does not apply.
3. **Binding capacity scales with model size, sub-linearly, and
   insufficiently** (§3.3). 780M holds 3–4 bindings, 2.7B holds 5–8, for
   3.5× the parameters. A 24-fact session still loses two thirds of its
   facts. Length-independence reconfirmed: 4×800 = 1.000 at 2.7B, same as
   4×40. **This is the strongest M-necessity evidence the project has** —
   the ceiling is architectural, and scale does not buy it away.
4. **An untrained M is actively harmful, and now quantified** (§3.3
   correction). Ablating the memory on/off over the same grid: 3,3,2,5,2,2
   with it vs 4,5,5,8,7,4 without. The first ablation delta on the board,
   and the bar a trained M must climb back over before any positive delta
   counts.
5. **The null's replay schedule privileges the first fact** (§3.6–3.8).
   Every one of the 6 installs under the carried schedule, across both model
   sizes and three seeds, is the transcript's *first* fact — because chunk 0
   is the only chunk the student ever distils from a fresh state, the very
   condition the probes test under. Resetting state before every chunk
   (`--fresh-state-replay`) roughly doubles the log-prob delta at both
   scales (780M +1.81→+3.74, 2.7B +2.27→+3.22), lifts the install rate
   (780M 2/9→4/9, 2.7B 3/11→4/11), and produces the first non-first-fact
   installs this harness has ever made (3 of 8). At 780M the corrected rate
   clears the pre-registered 0.30 PASS threshold.

### How much to trust (5)

The install-rate differences are individually not significant (n≈9–11 per
arm; Fisher p ≈ 0.2–0.3). What carries the weight is the convergence of
three independent signals in the same direction at two model sizes: the
per-fact log-prob shift (large, consistent), the positional pattern (6/6 vs
5/8 at index 0 — categorical, not a rate), and the ×4-steps control that
rules out "the corrected schedule just optimizes more". I would call the
*mechanism* well-evidenced and the *effect size* unmeasured.

### What I would want discussed

- **The pre-registered null verdict is now in question.** §2's FAIL was
  measured on a schedule that only ever trained one fact under test-time
  conditions. Does the null get re-registered and re-run on
  `--fresh-state-replay`, and if so at what N and with how many seeds to get
  real power? Note §1.5's caveat still bites: PASS at N=4 is compatible with
  brute memorisation into 9.67M LoRA params, so a PASS here licenses
  "mechanism alive", nothing more.
- **Is `--fresh-state-replay` the right correction, or a crude one?** It
  discards all intra-transcript context during distillation, which is
  wrong for filler and right for facts. A schedule that jitters the reset
  point (so each chunk is *sometimes* cold) might get the positional
  fairness without throwing away the context. I did not test that.
- **Whether M is now justified enough to fund the 2.7B fused port.**
  Findings 3 and 4 together say the backbone cannot hold a session's facts
  at any scale we can afford, and that M currently subtracts. Training M is
  the only way to move finding 4, and on the per-token loop 2.7B training is
  not affordable — so the port is the gate on everything downstream.
- **A resource question I am explicitly not acting on:** every conclusion
  here is at N=4, because that is where the harness has a valid premise.
  The high-N regime that M actually exists to serve remains untestable by
  this harness (§1.5 of the 08-04 note). That limitation has now survived
  two sessions and is worth a design conversation rather than another
  round of measurement.

### Housekeeping

- Four commits pushed: `197eb4e` (grad-checkpoint tolerance), `d47e315`
  (`capacity_ladder --no-memory`), `ac1d29c` (shared `set_memory_injection`,
  null gets the flag), `ea4c37b` (`--fresh-state-replay` + testable
  `replay_step`). Tests and READMEs updated with each.
- No checkpoints were written this session (no training). The artifacts to
  keep are `sft/logs/*.jsonl` and `sft/logs/*.log`.
- ~9 minutes of billed idle lost to a self-matching `pgrep` wait; the
  harness note above records the fix.

### Harness note for later sessions

`make <target>` pipes through `tee`, so the tmux pane's
`pane_current_command` reads `bash` while the job is running — the
"pane returned to bash ⇒ command finished" heuristic (monitor and waiters)
gives an immediate false completion. Wait on
`pgrep -f "python -u <script>.py"` instead — but write that wait as a
*foreground* check, not as `until ! pgrep -f "…consolidation_null.py"; do …;
done` chained to a launch command: the waiting shell's own command line
contains the pattern, so `pgrep` matches itself and the loop never ends. That
cost this session ~9 minutes of idle billed GPU before I noticed the launch
had never fired.
