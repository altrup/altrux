# Discussion notes — 2026-08-07: g2 results, erase geometry, and the warm-start run

Team debrief (altrup + Claude) of the 08-06/07 GH200 box session
(`EXPERIMENT_NOTES-20260806-234700.md`), which ran the corrected single-sleep
grid registered in `DISCUSSION-20260806-dream-distillation-ab-postmortem.md`
§5(a). Standing direction; supersedes that file's §5 run plan and amends its
§3 arm definitions and §6 rejected list as stated below. The pooled frontier
table was reproduced locally from the pulled jsonls during this debrief —
every number matches the run notes exactly. Two local probe rounds were run
during this debrief (§2); their raw jsonls are under
`sft/logs/erase_probe_ext_*` and `sft/logs/erase_probe_tail_*`.

**Standing frame, restated because it keeps getting dropped (altrup): the
objective is the installation-to-forgetting ratio — learning at minimal
damage — never installation speed or raw install count.** Every comparison
in this program is read as learned-vs-forgotten (iso-learning off the probe
curves where endpoints mislead); any decision rule that ranks arms or
operators by installs alone is wrong by construction.

## 1. What the g2 run established (agreed)

Floor-corrected (Δ vs the untrained no-sleep arm, per seed per fact), pooled
over 3 seeds at the final d800 probe:

| arm | Δmargin | Δ-installs | dPPL | EM | para | token-grads |
|---|---|---|---|---|---|---|
| sft-ref | +18.55 | 10/12 | +1.793 | 3/12 | 0.21 | 110,279 |
| B1 | +6.86 | 11/12 | +0.294 | 1/12 | 0.02 | 2,400 |
| A | +5.66 | 11/12 | +0.009 | 1/12 | 0.02 | 835,200 |
| CE-on-dream | +4.78 | 9/12 | +0.513 | 0/12 | 0.04 | 832,800 |
| B2 | +1.38 | 6/12 | −0.029 | 0/12 | 0.00 | 2,400 |
| A_bridge | +3.99 | 3/4 | +0.375 | 0/4 | 0.00 | 26,482 |

1. **The headline is real: dream distillation installs 10–11/12 facts at
   ~1/200th of fine-tuning's held-out damage**, on a metric with a floor,
   with the shared-dream invariant machine-asserted. Replicates 08-06 with
   both of that grid's fatal confounds removed.
2. **"B1 beat A" is NOT the registered claim** — B1's pooled Δ+6.86 > A's
   +5.66 comes entirely from seed 2345 (+10.18); A ties or beats B1 at the
   other two seeds. The per-seed-robust claim, and the one this file
   registers: **B1 matches A's learning at 348× fewer token-gradients.**
   First positive result for the erase mechanism.
3. **Carrying the ablation is worth ~5× the learning** (B1 vs B2, robust at
   every seed) — the exact contrast those arms were built to isolate. B1's
   damage is not free (+0.64 dPPL at its best-learning seed): learning and
   damage move together inside the erase family too.
4. **A's gentleness is objective + data**: CE-on-dream (same sequence, CE
   loss) is 56× A's damage but 3.5× gentler than sft-ref. Soft targets
   matter and dreaming on the dream matters.
5. **The registered margin metric had a 67% false-positive rate at the
   untrained floor** (a fixed random foil is not a calibrated control).
   Fixed in `1f1ab33`: everything is scored as Δ vs the per-(seed,fact)
   no-sleep floor. Process rule adopted (§7 housekeeping): **a metric is
   registered together with its floor measurement** — no metric is quoted
   until the do-nothing arm has been scored on it.
6. **The retrieval gap is the top open question**: dream arms move margins
   but greedy EM ≈ 0 and paraphrase ≈ 0 (sft-ref: 3/12, 0.21). The dream
   arms install a preference, not a retrievable answer. Working hypothesis:
   the dream *text* is the lever (cue-heavy, low-entropy, part-mojibake —
   base-model generation with untrained special-token embeddings), which is
   what the warm-start (§4) attacks. Re-read EM/paraphrase after it.
7. **Dream binding is the capacity bottleneck, not distillation**: per fact,
   installed ⟺ rehearsed-in-binding (11/11 conditioned). Teacher-dream
   binding quality is worth more than any arm change.
8. **Same-seed generation is nondeterministic on CUDA** (measured: same
   command, different dream, 2/4 vs 3/4 coverage). Cache everything shared;
   never rely on a seed to reproduce a dream. Seed 1234's in-use g2 dream
   was a favourable third draw — its absolute coverage must not be pooled
   naively (within-seed contrasts unaffected).
9. **On "A doesn't forget — doesn't that contradict the literature?"
   (discussed, resolved):** no. Arm A *is* the literature's cure —
   generative replay (DGR) + distillation-to-the-intact-model (LwF) — in
   the easy corner of the problem: one task, pristine teacher, LoRA-bounded
   drift, 4 facts. The literature's replay failures come from generator
   drift over long task sequences and replay-budget competition — exactly
   the pressures §4's multi-sleep phase introduces (and single-sleep
   cannot). CL is not solved; our grid simply hasn't applied the pressure
   yet. Corollary: the damage probe (battery PPL) is not the literature's
   "forgetting" (previous-task performance) — that measurement first exists
   in multi-sleep as the R-matrix.

Also measured on the GH200, standing hardware findings: chunked arms scale
near-linearly to 3 concurrent streams (~2.8× aggregate); per-token B arms
saturate at ~2.5 aggregate steps/s regardless of process count
(kernel-launch-latency bound, 12% real utilization). The A10 serial rule is
scoped to the A10 (`sft/CLAUDE.md`, `2fcbbc0`). The per-token bottleneck is
now largely mooted by the fused B arms (§3.5); B1 remains the one
deliberately-slow per-token arm and its registered budgets are affordable
solo (d800 ≈ 6.5 min, d3200 ≈ 25 min, d12800 ≈ 1.7 h if earned).

## 2. Erase geometry — measured this debrief (two local probe rounds)

`erase_probe.py` extended twice (working tree; commit in housekeeping):
round 1 (`erase_probe_ext_*`): 3 off-format bystanders + filler panel +
residual diagnostics, 3 seeds × {raw, state-svd}, γ=1. Round 2
(`erase_probe_tail_*`): 5 off-format + 2 near-cone-numeric bystanders,
worst-case reporting, bystander-query cos capture, a 202-question
address-space sweep, 6 seeds × {raw, state-svd} γ=1, plus a raw γ-sweep
{0.9, 0.95, 1.0} on 3 seeds. All on the local RX 7700S.

Round-2 pooled table (mean / worst single instance):

| panel | raw ĉ | deflated ĉ⊥ |
|---|---|---|
| target | −1.68 / −2.95 (18/19 greedy kills) | −1.53 / −2.81 (17/19) |
| sibling | −0.43 / −1.83 (32/57 flips) | −0.28 / −1.77 (19/57) |
| off-format bystander | −0.11 / −1.62 (14/104 flips) | −0.09 / −1.84 (8/100) |
| near-cone bystander | −0.13 / −0.92 (13/40 flips) | −0.04 / −0.46 (5/40) |
| filler span | −0.06 | −0.03 |

1. **Collateral concentrates in-cone, but the tail is ~10× the mean.**
   Off-format bystanders average −0.11 yet single instances lose 1.6–1.8
   nats — under *both* operators. Occasional single-memory kills are real;
   annihilation-scale events were not observed. Greedy flips are the honest
   count: deflation roughly halves them in every class.
2. **Deflation improves every row** (~9% target-kill cost for ~35% less
   sibling damage and 2.6× better near-cone worst case). The
   noisy-residual fear is disproven: residual norms ~0.99 (min 0.947),
   skip-cone guard never fired.
3. **Geometry reframe 1**: the fact-query cone (pairwise |cos| ≈ 0.92) is
   NOT the state's top singular direction — |cos(ĉ, v)| ≈ 0.08–0.10.
   Deflation works by removing a *small* component with *huge gain through
   S* (that ~10% of direction carries ~70% of the removed Frobenius mass).
4. **Geometry reframe 2 — the address-space sweep is a null instrument,
   and that is a finding**: ALL read queries share a ~0.75 common-mode
   |cos| (202 diverse questions — unrelated, numeric, and paraphrase
   classes all land at 0.745–0.753; random vectors would sit at ~0.09).
   The common mode lives in the queries, not in v (deflating v moves it
   ≤0.005). Consequently **|cos(query, erase direction)| does not predict
   damage** (near-cone bystanders sit at higher cos than off-format yet
   take less damage) — collateral must be measured as damage, not
   inferred from query geometry. The "sweep the address space to bound
   the assassination tail" plan is retracted with this.
5. **γ sweep, raw, 0.9/0.95/1.0 — the AGC prediction is REFUTED in this
   range**: target kill is slightly *sub*-linear toward γ=1 (0.867 / 0.935
   / 1.000 normalized; 9/9 greedy kills at every γ), and the
   target:sibling ratio is flat at ~3.5 across the sweep. Confirmed: γ
   buys no selectivity (the ratio is γ-invariant, as predicted); refuted:
   the gated-RMSNorm gain-control collapse near γ=1 (the 08-05 supra-
   linearity lives below ~0.9, not near 1). **γ stays 1.0** — best kill,
   no selectivity cost, and sub-1 γ reopens the amplification channel.
6. **Collateral physics, corrected for the record**: the erase removes a
   victim's readout energy ∝ cos²(k, ĉ⊥) *of its key against the cut
   direction, weighted through S's gain* — and per finding 4, realized
   query geometry is dominated by a common mode, so armchair cos²
   estimates mislead. Full clearing still requires an address collision
   (cos ≈ 1) at many layers simultaneously; multi-layer redundancy is the
   structural shield.
7. **Capacity is per-cone, not global**: the "~4 binding ceiling" was
   measured on 4 same-template code facts; these transcripts held 7–11
   items (4 codes + up to 7 bystanders) with bystanders binding
   near-certainty (baseline greedy: near-cone 10/12, off-format 26/30).
   Address crowding and answer difficulty limit codes; mixing relation
   types raises per-wake capacity.

**Operator decision: made empirically on the box, first cells** — rule in
§3.4. Single-shot evidence favors deflation on collateral; raw closes the
compensation harbor entirely (S(I−ĉĉᵀ)ĉ ≡ 0 — no erasure-resistant
direction exists, no v to co-opt, no blind spot). The remaining unknown is
cumulative (B1 erases hundreds of times per dream), which is an arm-level
measurement.

## 3. Decisions registered this debrief

1. **Warm-start SFT before everything, next run.** Purpose: train the
   `[USER]`/`[ASSISTANT]` special-token embeddings so dreams stop going
   off-distribution. Data: **general chat/instruction data rendered into
   the repo's marker format** (NOT filler-synthetic — rejected §5; NOT
   `train_chains.pt` — 2.7B-memory-model data full of `[SLEEP]`
   structure). Concretely (pinned at the dry-run's request): ~4000 turns
   of ultrachat rendered via `prepare_data.py`'s existing path into
   `USER_OPEN`/`ASST_OPEN` + literal-space format, ~400 LoRA steps at
   1e-4, LoRA rank/alpha matching `dream_sleep.py`'s defaults (else the
   checkpoint cannot load), one checkpoint. Every arm and the cache
   builder load it (`--init-adapter`; checkpoint SHA-256 in every result
   jsonl, asserted by the summarizer). The knowledge battery is
   **recalibrated after** the warm-start — mechanism: delete
   `data/knowledge_battery_mamba2_780m.json` and let the first
   warm-started invocation auto-rebuild it under `--init-adapter`
   (`load_or_build_battery`'s existing path). Acceptance check before any
   grid cell, pass/fail: decode one full dream; **zero mojibake tokens and
   zero bracket-mimicry lines in the free-running spans**, and the
   free-running text reads as English. Fallback if it fails: double the
   warm-start steps once (400 → 800) and re-check; a second failure is
   stop-and-report via the shutdown checklist, not a judgment call.
   **Scope: the whole session, ladder included** (altrup) — g2 stays the
   no-warm-start reference; the ladder carries its own d800 rungs.
   Committing the battery + checkpoint from the box is the one registered
   exception to the never-commit-data-from-the-instance habit (small
   files; ends the battery-regeneration drift).
2. **Wave-k dream generator/teacher (k ≥ 2): the student as of that
   sleep's start**, generating from its own carried state
   (self-distillation). Decisive argument: a frozen-base teacher
   structurally cannot rehearse consolidated facts (not in its weights, no
   longer in state) — it deletes the replay channel A's survival depends
   on, and cannot drift, deleting the literature's predicted failure mode.
   One **frozen-base-teacher control cell** (one seed, arm A) so the drift
   contribution is measurable. Wave-≥2 **cues cover only that wave's
   facts**; spontaneous rehearsal of earlier waves' facts is a measured
   observable, not a cue artifact.
3. **Sequential sft-ref joins multi-sleep** — fine-tune wave by wave, the
   standard CL baseline; where literature-style forgetting should appear
   (backward interference in its R-matrix).
4. **The erase, registered for all B arms:** γ = 1.0. Direction = the
   student's own current query. **The gradient runs through the ablation
   direction's query path (ĉ); v — when deflating — is always
   stop-gradiented** (in depth-1 arms this is automatic, since v derives
   from detached states; B2-fused-deep computes v under no-grad
   explicitly). Rationale (corrects the 08-06 §6 entry, whose reasoning
   was wrong for the query path — altrup's argument): a *detached*
   direction makes the per-step loss landscape treat the cut as fixed, so
   the read-path gradient myopically points at dodging via remnants; a
   *differentiable* direction correctly encodes "the cut follows the
   query" (for a raw own-query erase the ablated readout is identically
   zero under any rotation — the dodge gradient vanishes). What detaching
   v still guards: gradient through v's dependence on the writes would let
   the optimizer rotate the protected subspace onto the fact (co-opt the
   guard) and is numerically vicious through an SVD.
   **Operator (raw ĉ vs deflated ĉ⊥): decided by the box's first cells —
   B1-raw vs B1-deflated at d800, 3 seeds.** The comparison is paired
   within each seed (both operators share that seed's cached dream,
   warm-start checkpoint, and battery), so seed-level noise cancels.
   Decision rule, in order (per the standing frame — install-to-forget
   ratio, never install count):
   1. Dominance: one operator has ≥ Δ-installs AND ≤ dPPL at d800 → wins.
   2. Iso-learning: over the Δmargin range both operators reach on their
      probe curves, the one with less dPPL at matched Δmargin wins.
      Robustness guard (the g2 "B1 beat A" lesson): the pooled verdict
      must hold in ≥2 of 3 seeds, else it is "inconclusive".
   3. Inconclusive (curves cross, tiny overlap, or the guard fails) →
      **raw** (altrup's registered lean, on compensation-closure).
   The winner is the erase operator for the rest of the session. If raw
   wins, the harbor apparatus (deflation, skip-cone guard, v-energy
   fingerprint) retires with it.
5. **The fused B arms** (altrup's insight this debrief: B2's defining
   property — nothing counterfactual carries — makes it fusible; the
   per-token bottleneck was never intrinsic):
   - **B2-fused-detached** — per pass, all at that pass's fixed weights:
     (i) fused chunk-scan of the *intact* spine from a copy of the wake
     state under the **current student weights**; materialize per-token
     per-layer states, **detached**; (ii) all 512 tokens batched — each
     takes a copy of its states, ablates along the student's query for
     that token (per-layer interleaved), writes token t, reads → logit;
     KL at every position summed → **one optimizer step per pass**.
     Nothing counterfactual carries; the spine is recomputed each pass.
   - **B2-fused-deep** — identical machinery, spine NOT detached: full
     BPTT through the scan (v still no-grad). The depth variable,
     measured not assumed.
   - **B3-fused** — identical to B2-fused-detached except the spine is
     **the dream-generator's own state trajectory**, computed once per
     sleep from the generator snapshot (sleep 1: the warm-start
     checkpoint; sleep k: the student as of sleep k's start — the same
     weights that generated that sleep's dream) and cached. Consequences:
     the cached teacher logits and the host states are the same
     computation's outputs ("the state that emitted this logit, minus the
     fact"); **B3-fused ≡ B2-fused-detached at every sleep's pass 1**
     (machine-checked per sleep); they diverge within a sleep purely
     through B2's pass-level spine drift — B3 isolates within-sleep spine
     drift, refreshed each sleep. Depth is vacuous for B3 (spine states
     are constants w.r.t. the live weights).
   - Step currency: one fused-B step = one full-dream pass (A's
     currency). One **per-token-B2 bridge cell** (g2 form, optimizer step
     per token) for cross-run continuity. Expected economics: ~100× over
     in-process seed-batching; token-parity with A@d800 drops ~7.6 h →
     ~15 min. Equivalence tests: B3≡B2 at pass 1; per-token-B2 vs
     B2-fused-detached at matched token-gradients is a *comparison*, not
     an equivalence (different step currency — do not assert equality).
   - Depth decision rule: if the operator picker chooses raw AND the
     B2-fused pair's fresh-state installs track together (no compensation
     gap), **B2-fused-deep becomes the primary B2 going into multi-sleep**
     (under raw the erasure-resistant-writing channel has no readable
     harbor); under deflation, detached stays primary and deep keeps the
     v-energy fingerprint watch.
6. **B1 unchanged and retained** (per-token, ablate in place, carry
   detached between tokens — depth-1 stands; the deep-B1 rejection is
   re-affirmed and sharpened: its cross-token gradient passes through the
   product of every intervening ablation projector, which annihilates all
   directions except the protected subspace — the pathology is *projector
   composition on the differentiated spine*; B2's ablations are leaves of
   the graph, never composed, so its deep gradient is clean up to an
   incentive the fingerprint watches). B1 is the mechanism arm
   (erase-during-carry) and the erase family's best g2 learner; its
   slowness is affordable at every registered budget including multi-sleep
   (~6.5 min per sleep).
7. **Multi-sleep arm list** (registering the detail 08-06 deferred): A,
   B1, B2′, sequential sft-ref, no-sleep, frozen-base-teacher control
   (arm A, one seed); B2-fused-detached/deep and B3-fused ride along as
   budget allows (they are minutes-cheap). K=4 sleeps × 4 fresh facts,
   cues new-wave-only, per-arm per-wave dreams (generator = student at
   sleep start), probes on all facts so far + battery after every sleep,
   R-matrix / BWT / cumulative installation reported, fresh-state probes
   only (state continuity across sleeps is load-bearing for the B arms).

## 4. Next run plan — in order, on the box (GH200)

**LAUNCH GATE: the box does not launch until every item in this section's
local-harness-work block is implemented, tested, and PUSHED.** Steps (0),
(2), and half of (3) are unrunnable without it. An experimenter session
that arrives on a box and finds the harness work missing stops and reports
via the shutdown checklist — it does not build the harness on billed time.

**Budget guidance (not a hard cap — altrup):** plan the session around
~6 h GH200 (~$15); take multi-sleep in-session only if the ladder wraps
with roughly 4 h of headroom, else it is the next session's docket.

Regime unchanged unless stated: `--n-facts 4 --filler-tokens 40
--dream-tokens 512 --dream-temp 0.7 --cue-every 32 --cue-greedy 12 --lr
1e-4`, floor-corrected margin (installed iff Δ ≥ 1.0 nat vs the no-sleep
floor), binding-aware coverage gate ≥3/4 per seed cache, probes every 200
at d800 (every 1600 on the ladder).

**(0) Warm-start** (§3.1), which is `make warm-start` and nothing else —
`prepare_data.py --hf-dataset HuggingFaceH4/ultrachat_200k --max-examples
1000 --max-len 4096 --output data/warm_start.pt`, then `train.py --data
data/warm_start.pt --max-steps 400 --lr 1e-4 --chunk-len 512 --lora-rank
16 --lora-alpha 32 --lora-dropout 0 --keep-ckpts 1`, leaving the one
checkpoint dir `models/mamba2_780m/checkpoints/epoch-1/step-400` that
every cell then loads with `--init-adapter` (the 400→800 fallback is
`make warm-start ARGS="--max-steps 800"`). Read `make sanity-sample
ARGS="--data data/warm_start.pt"` before the training half starts.
Sequence: corpus → checkpoint → **delete
`data/knowledge_battery_mamba2_780m.json`** → first `--init-adapter`
invocation rebuilds it → dream-sample acceptance check. The battery
deletion is a written step, not a reminder: `load_or_build_battery`
loads any existing file unconditionally and the json carries no adapter
hash, so a forgotten deletion silently scores every cell against
cold-weight baselines. **If the 400→800 fallback fires, the new
checkpoint requires deleting the battery a second time.** Checkpoint
hash into the run notes; commit the battery and checkpoint from the box
(small files — ends the regeneration drift; this was the second
consecutive run to lose the battery). Both files are gitignored (the
battery under `sft/data/`, the checkpoint dir under
`models/*/checkpoints/`) — committing either needs `git add -f`.

**(1) Fresh dream caches**, seeds 1234/2345/3456, from the warm-started
weights, coverage-gated; generator per-token states cached for B3 (or
recomputed per sleep — implementer's choice, equivalence-tested). Read one
decoded dream + one cue joint. **Caches are built only after the session's
harness code is frozen** — generation is nondeterministic (§1.8), so a
cache built before a code change can never be regenerated; harness first,
caches last. If a seed misses the binding gate and rebuilds at
`--cue-every 24`, its arms run at 24 and the deviation is recorded:
within-seed contrasts stay valid, cross-seed pooling of absolute rates is
tainted (the g2 stance).

**(2) Operator picker: B1-raw vs B1-deflated, d800, 3 seeds**, paired
within seed, **plus a `--no-sleep` floor cell per seed** (probe-only,
cheap, filename containing `nosleep` — `summarize_grid.apply_floor`
keys on it). Without the floor cells Δ-installs and Δmargin are
uncomputable and the §3.4 rule cannot be applied; the g2 floor does not
transfer across the warm-start. Decision by the §3.4 rule (dominance →
iso-learning damage at matched Δmargin with the ≥2/3-seed robustness
guard → raw fallback). Winner is the session's erase operator. Run the
picker as direct `dream_sleep.py` invocations (B1-raw, B1-deflated,
no-sleep per seed, flags copied verbatim from `run_grid2.sh` into the
notes) — not the grid script twice, which duplicates every non-B1 arm
for zero information; step (3)'s grid run then skips the finished B1
cells via the done-record resume check.

**(3) d800 baselines** on the new caches: A, B1 (winning operator),
B2-fused-detached, B2-fused-deep, B3-fused, per-token-B2 bridge. The g2
d800 numbers do not transfer across the warm-start. **Open with a rate
probe: time one fused cell at `--distill-steps 20` and re-plan this
step from the measured per-pass rate** — a fused pass (spine
materialization + 512 batched counterfactuals) is heavier than an A
pass, `--distill-steps 800` counts passes, and §3.5's ~15 min figure
was token-parity with A@d800, a different budget than 800 fused passes.

**(4) Saturation ladder**, seed 1234, d3200 (`--probe-every 1600`): A, B1,
B2-fused-detached, B2-fused-deep. Stop rule: an arm flat between d800 and
d3200 does not buy d12800. B3-fused earns rungs only if its d800 Δmargin
is within 2× of B2-fused-detached's.

**(5) Multi-sleep** per §3.7 — this session if there is room; else it is
the next session's docket with (0)–(4)'s artifacts (checkpoint, caches)
pulled home and reused. At the ~4 h headroom boundary, finish the ladder
cleanly rather than start a multi-sleep that can't complete. **Multi-sleep needs its own cache build per seed**
(`--waves 4` consumes a different RNG stream, so the wave-1 facts and
transcript differ from the single-sleep cache's — verified locally; a
`--waves 4` cell against a single-sleep cache exits on the transcript
check). Use explicit `--dream-cache` paths (e.g. `dream_cache_w4_s<seed>.pt`)
and never pool step (2)–(4) numbers with step (5) numbers — different wake
transcripts.

**Local harness work before the box (TDD on the CPU fake backbone where
testable):**

- `--init-adapter <ckpt-dir>` in `dream_sleep.py` (before cache build AND
  training; trainable.pt hash in every jsonl; summarizer asserts
  equality).
- Warm start as `make warm-start` = `prepare_data.py` + `train.py`
  (`--max-steps`, the one new `train.py` flag).
- B2-fused-detached / B2-fused-deep / B3-fused per §3.5: fused spine,
  state materialization, batched counterfactual forward, detach flag,
  per-sleep B3≡B2 pass-1 equivalence test, generator-state caching.
  Per-token B2 path retained (bridge).
- Differentiable-ĉ erase (remove the direction detach; keep v no-grad);
  tests that v carries no grad in every arm including B2-fused-deep.
- Operator flag (raw/deflated) launchable per cell for the picker.
- Multi-sleep driver completion: sequential sft-ref, B2′ commit step,
  frozen-base control, per-wave dreams from the sleep-start snapshot
  (`62601d7` is the base — green on the CPU fake backbone only, unproven
  on hardware; treat as such).
- Box-side, early in the session (altrup: worth a shot, should be
  quick): `torch.compile(mode="reduce-overhead")` experiment for B1's
  per-token loop (power-iteration v instead of SVD if deflated survives),
  measured go/no-go — time one B1 cell compiled vs not before committing
  the picker cells to either path. Engineering only; results identical
  either way. In-process seed-batching (batch-S) is dropped (§5).
- Commit the erase-probe extensions; experimenter-command edits (§7).

**Shutdown rule (new, after the g2 cache loss): terminate only after
`lambda_pull.sh` has run and the artifact list (dream caches, sidecars,
battery, checkpoint) is verified present locally — sizes echoed into the
notes. A rushed shutdown records artifacts as UNRETRIEVED, never as
"carried by the pull."**

## 5. Explicitly considered and rejected

- **A bespoke warm-start trainer** (`warm_start.py`'s own corpus
  builder + training loop, plus `lora.save_adapter`/`load_adapter` and
  its single-file `.pt` adapter format) — built, then deleted unused. It
  was a second implementation of `prepare_data.py` + `train.py`: the
  same marker rendering, the same chunked loss, a parallel checkpoint
  format for the same trainable set. The chain does it with one new flag
  (`train.py --max-steps`) and inherits resume, mixing, the loss
  weights and `sanity_sample.py` for free. Reuse wins; the only thing
  lost is the bespoke corpus-invariant print, which the sanity-sample
  read covers.
- **Filler-synthetic warm-start corpus** (altrup): trains the model to
  emit the filler distribution the dreams already over-repeat; general
  data teaches the format without amplifying regurgitation.
- **Running the ladder without the warm-start for g2 comparability**
  (altrup): the warm-start was coming anyway; one preamble for the whole
  session beats a half-comparable ladder. Moot in any case — the g2 dream
  caches were lost unpulled (§7), so extending g2's curves was impossible.
- **Footprint subtraction** (exact per-fact ΔS via transcript
  counterfactual walks): exact targeted deletion, zero collateral by
  construction — but requires provenance, and the erase stays a pure
  function of (S, ĉ) (altrup). Parked as diagnostic-only.
- **Sub-1 γ as a selectivity dial** — measured dead (§2.5): the
  target:sibling ratio is flat in γ; γ=0.9 still kills 9/9 greedy. Lower
  γ buys nothing and reopens amplification.
- **"Gradient through the erase directions — rejected" (08-06 §6) is
  REVERSED for the query path** — see §3.4 for the corrected rationale
  and what remains detached (v, and B1's between-token carry, and
  B2-fused-detached's spine: *direction*-detach and *carry/spine*-detach
  are different knobs; only the first is reversed).
- **Deep-B1** — re-affirmed rejected (§3.6, projector-composition
  argument). **Deep-B3** — not rejected, *vacuous*: the spine is a
  snapshot's states, constants w.r.t. the live weights; removing the
  detach changes nothing.
- **Dropping B1 for speed** — it is the mechanism arm and the g2 erase
  family's best learner; its real budgets are minutes-to-~2 h.
- **In-process seed-batching (batch-S) for the B arms** — superseded by
  fusion (~100× vs ~3×); B1's registered budgets don't need it.
- **20–30 bystanders in one transcript** to hunt the collision tail: most
  would never bind (per-cone capacity); bound-bystander × seeds is the
  valid form. The address-space *sweep* variant was run and retracted as
  a null instrument (§2.4) — query-cos does not predict damage.
- **KV-cache (transformer) A/B now, before finishing the Mamba arc** —
  sequencing: exact slot deletion makes the transformer the clean control
  for interpreting Mamba results, which pays only after Mamba has an
  answer. Registered as the named follow-on program (§6).

## 6. Open questions

- **The retrieval gap** (§1.6) — re-read EM/paraphrase after the
  warm-start run.
- **The ~0.75 query common mode** (§2.4, new): all read queries share a
  large common-mode component that is not the state's top singular
  direction. What is it (a format/positional carrier? the conv's DC
  component?), and would deflating against the *query* common mode (not
  v) be the better surgical correction? Cheap local probe if the operator
  picker keeps deflation alive.
- **Address-collision ("assassination") tail**: single-memory kills of
  1.6–1.8 nats occur at low rate under both operators (worst-of-100-ish);
  annihilation was not observed. Query-cos does not predict the victims,
  so the remaining tool is adversarial search (optimize a prompt to
  maximize damage, not cos) — parked unless multi-sleep shows unexplained
  bystander loss.
- **KV-cache A/B program** (named this debrief, after the Mamba arc):
  attention's slot structure admits erases Mamba cannot express —
  attention-gated key ablation k_i ← k_i − a_i(q̂·k_i)q̂ and
  attention-gated value attenuation v_i ← (1 − γa_i)v_i; form (ii) dodges
  softmax renormalization and is the closest analogue of consume-on-read.
  Raw key-projection is the worst form there (zeroed scores → uniform
  attention → value smear). The overlap problem becomes a selection
  problem (softmax sharpening + multi-head redundancy) instead of an
  irreducible linear blend.
- **Warm-start's effect on dream binding** — multi-sleep measures
  dream-quality-across-sleeps for free; compare wave-1-after-warm-start
  against g2's base-model dreams for the direct effect.
- **Value-side erase, deflation-k under multi-cluster states, soft
  dreaming** — carried unchanged from 08-06 §7. **B2's saturation
  ceiling** — now measured by the fused ladder at trivial cost.

## 7. Housekeeping

- g2 run notes banked as `993843d` before discussion (flow rule).
- Working-tree rsync clobber of the 08-06 file's warm-start block:
  restored by re-applying the committed content (same failure shape as
  the 08-06 debrief's soft-dreaming clobber).
- **The g2 dream caches, sidecars, and battery are LOST** — the data pull
  was never executed before termination (only the logs sync ran); verified
  by subagent search including the pull script's manifest
  (`lambda_data_artifacts.sh` globs were correct; the pull just never
  ran). The run notes' claim that "the pull carries" them was an
  unchecked invariant — same lesson as the shared-dream bug; now §4's
  shutdown rule. Dream *texts* survive as decoded samples in the g2 logs
  plus two rejected sidecars.
- `sft/.env` sets `MODEL_NAME=mamba2_780m_memory_mix`; every 780m probe
  run needs `MODEL_NAME=mamba2_780m` set in the environment (dotenv does
  not override real env vars).
- `erase_probe.py` extensions (both rounds: bystander classes, filler
  panel, worst-case reporting, cos panels, γ sweep support, near-cone
  templates, new invariants) to be committed after review. Round-2 agent
  also fixed bystander grading (prefix match) and renamed a colliding
  near-cone noun (`parcel` → `shipment`), adding a
  question-matches->1-statement invariant.
- Experimenter-command edits agreed: (a) the shutdown pull-verify rule
  (§4); (b) the metric-floor rule (§1.5); plus three staleness fixes the
  dry-run audit surfaced: (c) the command's header names
  `mamba2_2_7b_memory` — this program runs `mamba2_780m`; (d) its baseline
  resume invocation points at the superseded 07-25 file; (e) its monitor
  snippet uses the `pane_current_command` completion poll that
  `sft/CLAUDE.md` explicitly bans (use the `EXIT=` file-marker scheme).
  Also (f) reword `sft/CLAUDE.md`'s serial-grid bullet so the A10-only
  scope is the headline, not an appended aside. [pending implementation]
- Dry-run audit (debrief step 5): a context-free Opus experimenter read
  the repo cold and narrated the session. Its plan matched intent;
  its 16 findings drove this file's launch gate, budget guidance,
  warm-start pinning (corpus/steps/rank), battery-recalibration
  mechanism, acceptance pass/fail line + fallback, cache-freeze rule,
  cue-24 pooling stance, the commit-from-box exception, the picker
  decision rule, and items (c)–(f) above. The step earned its place
  again.
- Chance-collision arithmetic for the record: for *random* directions at
  d_state=128, |cos| concentrates at ~0.09 and |cos|>0.5 is ~10⁻⁸/pair —
  but §2.4 shows realized query geometry is common-mode dominated, so
  this arithmetic must not be used to bound anything; damage is measured,
  not inferred.
- Next box session's docket is §4 verbatim; local TDD list is §4's
  harness-work block.
