# Discussion notes — 2026-08-06: single-sleep A/B postmortem, corrected arms, and the multi-sleep plan

Team debrief (altrup + Claude) of the 08-06 box session
(`EXPERIMENT_NOTES-20260806-010133.md`), which ran the single-sleep A/B grid
registered in `DISCUSSION-20260805-dream-distillation-cl-ab.md`. Standing
direction; supersedes the 08-05 file's §4–§5 run plan (its §3 mechanism
design and §6 rejected list remain in force except where amended here).

## 1. What the run established

Pooled 800-step cells, 3 seeds (`sft/summarize_grid.py` over
`sft/logs/dream_*.jsonl`; reproduced locally, exact):

| arm | learned (dlogp) | damage (dPPL) | battery lost | dream code cov |
|---|---|---|---|---|
| A (replay) | +1.83 | +0.14 | 1/72 | 3–4/4 |
| B2 (counterfactual) | +0.35 | −0.16 | 0/72 | 3–4/4 |
| B1 (drain, as-run) | +0.05 | −0.09 | 0/72 | 1/4 |
| B1-live (1 seed) | +0.03 | −0.04 | 0/24 | 0/4 |
| sft-ref | +0.61 | +2.18 | 2/72 | — |
| no-sleep | +0.000 | +0.0000 | 0/72 | — |

Trustworthy as stated:

- **A is far gentler than conventional fine-tuning.** dPPL +0.14 vs +2.18, no
  seed overlap. sft-ref never touches a dream, so the shared-dream bug (§2)
  cannot explain this.
- **B2 learns ~1/5 of A at literally zero measured damage** (dPPL negative at
  all 3 seeds, 0 battery items lost anywhere). Direction trustworthy despite
  §2: B2's dreams were as rehearsal-rich as A's at 2 of 3 seeds and it still
  learned 5× less. **Budget caveat (dry-run finding):** `--distill-steps` is
  not a common currency — one A step is a 48-token chunk, one B step is a
  single token, so B2@d800 received ~1/48 of A's token-gradients. Per unit
  of gradient signal B2 may be the *more* efficient arm; the rerun logs
  `token_gradients` per cell and reports the frontier on both axes.
- **As-run B1 and B1-live are at floor.** As-run B1's dream coverage collapse
  (1/4 every seed/budget) is mechanism-consistent: draining during generation
  destroys the rehearsal material. Retired (§6) — but note §3's corrected B1
  is a *different arm* and is not retired.
- **Greedy exact match is broken as a metric.** The install-feasibility probe
  showed facts at teacher-forced p≈0.68 on the full code and paraphrase 0.50
  scored as greedy misses — a digit-counting attractor (`' 1 2 3 4 5…'`)
  owns the trained phrasing. Every "0 installs" in the grid is partly this
  artifact.
- **The no-sleep floor is an exact zero**, so the harness itself doesn't leak.

Confounded (rerun required before quoting magnitudes):

- **The arms did not share dreams.** Registered: one cached teacher dream per
  seed, byte-identical across A/B1/B2. Actual: `run_grid.sh` runs each arm as
  a separate process and each regenerated its own dream — transcripts diverge
  at the fifth word (verified from the `dream` records), with wildly different
  fact compositions. Within-process caching was implemented correctly; the
  guarantee died in the harness→driver composition, and same-seed
  reproducibility silently failed across arm code paths (A and B2 diverge
  inside a *greedy* span — an unexplained RNG/numerics drift, root-cause on
  the box). Per-fact cross-arm comparisons from this grid are meaningless.
- **A's +1.83 dlogp is partly format prior.** The cues inject the answer
  format; the smoke measured ~+1 nat from format alone; and sft-ref — with
  *lower* dlogp — is the only arm with greedy installs and nonzero paraphrase
  rates in the registered grid. Composition of A's gain is unknown until the
  margin metric (§4) exists.
- **A ran a replay schedule the team had already superseded (dry-run
  finding).** `distill_replay` carries state across chunks within a pass —
  the schedule the 08-05 debrief found installs only fact 0 under
  fresh-state probes, and that `consolidation_null.py --fresh-state-replay`
  was added to fix (2/9 → 4/9 installs there). `dream_sleep.py` never got
  the flag; only ~1 of 11 chunks was ever practiced in the probed
  condition. Plausible partial explanation of A's high-dlogp / zero-install
  pattern.
- **Dream coverage counted misbound codes.** A's seed-1234 dream opens with
  "The code for the heron is 5 9 7 9 7" — osprey's code — and coverage
  counted it. Coverage must be binding-aware (§4).

Not run: two-wave. The registered ≥0.3 conditioned-install gate was not met
on greedy EM and the box correctly refused to reinterpret it mid-run. The
gate is re-registered on margin in §5.

altrup's pre-registered hypothesis (B slower but better learned/forgotten
ratio) is **not supported in single-sleep**: B2's ratio is technically best
(zero denominator) but A already achieves near-zero damage, so no budget in
this grid has B dominating. Neither does A dominate B2 — both sit on the
Pareto frontier (A better on learning, B2 on damage). The hypothesis moves to
the multi-sleep phase (§5c), where A's obligatory full-state clear creates a
forgetting channel (clearing-loss) that single-sleep cannot see.

## 2. The shared-dream bug — process lesson

The 08-05 file registered "byte-identical cached dream" in prose. The harness
honored it within one process; the grid driver voided it across processes;
nothing checked. **Process rule (adopted): a registered invariant ships with
its machine check** — hash the shared artifact into every result file and
make the summarizer assert equality, loudly. Prose invariants don't survive
refactors; asserted ones do. (Recorded in `.claude/commands/altrux-experimenter.md`
housekeeping, §8.)

## 3. Arm definitions — SIGNED OFF (altrup, this debrief)

These sequences supersede the 08-05 §3 sequences. The erase operator is
unchanged: γ=1.0, state-svd deflation k=1 (ĉ⊥ = normalize((I − vvᵀ)ĉ), v =
top right-singular direction of the heads-stacked state), skip-cone guard.

**Shared preamble, every arm and seed:**

1. Wake: model (adapters at init) reads the wake transcript (4 facts +
   filler); final state = **wake state**. Never cleared before sleep.
2. Teacher dream, generated **once per seed**, persisted to
   `sft/data/dream_cache_s{seed}.pt` containing: wake transcript token ids,
   wake state, dream token ids, per-position teacher logits, per-position
   per-layer read queries, and SHA-256 hashes of transcript and dream tokens.
   Generation: frozen base (adapters bypassed) from a copy of the wake state,
   intact (no erase), `[ASSISTANT]` marker as sole seed text, temp 0.7, EOS
   banned, cues per §4. **Every arm loads this file; no arm generates.**
   Every result jsonl records both hashes; `summarize_grid.py` asserts
   equality across all cells of a seed and refuses to pool on mismatch.
3. All installation probes are **fresh-state** (probe on a zeroed copy;
   the carried stream is never consumed by probing). Carried-state answers
   are a diagnostic column, never installation.

**Per-token micro-order for the erase arms** (matches `_mixer_step`'s fused
order; the read is from the post-write state). This is **per layer,
interleaved inside the forward** — implemented as an erase hook in
`_mixer_step`, not as a wrapper around the whole forward. A layer's query is
independent of *that layer's own* state, but it does depend on lower layers'
states through their outputs, so erasing layer i changes layers i+1…'s
queries for the same token; the interleaved form is the only one consistent
with "the state that generated this token", and it costs one forward per
token instead of two:

1. At layer ℓ, token t: x_t, B_t, ĉ_t via in_proj + conv (independent of
   layer ℓ's own state).
2. v from layer ℓ's current carried state; ĉ⊥ by deflation. **Both
   detached** — `erase_state` must call `.detach()` explicitly (today it is
   safe only by accident, via no-grad teacher walks); otherwise the
   optimizer can rotate queries to dodge the erase instead of installing
   facts.
3. Ablate the carried **past**: S ← S(I − ĉ⊥ĉ⊥ᵀ). B1: in place. B2: on a
   copy. The current token's write has not happened yet and always survives.
4. Decay + write token t, then read → layer output; after the last layer,
   the training logit.
5. KL vs the cached teacher logit at this position.

The queries used for erasure are the **student's own, from this forward**
(the code currently uses cached teacher queries — the §6-rejected variant —
so this is a real code change, as is moving B1's erase before the forward:
today's `distill_stateful` never ablates the state the student trains from
in the drain arm at all). B1's carry is advanced by the **student's own
forward, detached** — not by an extra teacher pass.

The logit trained on is always produced from the ablated version of the state
that *generated* that token; the post-token state is only carried (B1) or
discarded (B2), never trained on directly.

**Arm A — generative replay (field method):** student teacher-forced over the
cached dream, KL to cached teacher logits. **Registered form: full-sequence
chunking** — the chunk is the whole dream, one optimizer step per pass, from
a fresh (zero) state. This subsumes the fresh-vs-carried schedule question
(with one chunk per pass there is nothing to carry across) and eliminates
the chunk-boundary artifact (at chunk 48, boundaries fall mid-rehearsal, so
an answer could be practiced from a fresh state with its question in the
previous chunk — the 08-05 null found the carried variant of that schedule
installs only fact 0). Full-sequence BPTT needs ~19 GB of per-token state —
this is what moves the run to a GH200 (96 GB; also unlocks the fused SSD
kernel path, which must be used there per root CLAUDE.md — the per-token
loop is a ROCm-only workaround). One **bridge cell** at seed 1234 runs the
08-06 configuration (chunk 48, carried within pass) so the old numbers stay
interpretable. CE-on-dream uses A's registered form so the objective is the
only difference.

**Arm B1 — cumulative counterfactual (corrected; NOT the retired as-run
drain):** student teacher-forced over the **same cached dream** starting from
a copy of the **wake state**. Micro-order above with the ablation **in
place**; the ablated-and-advanced state carries. Each pass restarts from a
fresh copy of the wake state. The drain never touches generation — the dream
is fixed on disk.

Both B arms' gradients are **one token deep**: the carry is detached
between tokens (per-token optimizer steps make deeper graphs stale, and
gradient into past writes would let the optimizer learn erasure-resistant
rewriting — compensation via state instead of installation into weights).
Known asymmetry, accepted: A's full-sequence form backprops through its
whole recurrence; its state holds only dream content from a fresh start,
and in-context shortcuts don't score on fresh-state probes.

**Arm B2 — per-token counterfactual:** identical to B1 except step 3
operates on a **copy**, the training logit comes from the copy, the copy is
discarded, and the intact state carries. B1 ≡ B2 at token 1 (bit-identical
state, query, ablation, logit, gradient); they diverge from token 2 purely
through the carry. Any outcome difference is attributable to carrying the
ablation.

**CE-on-dream (new decomposition cell):** Arm A's sequence with
cross-entropy on the dream tokens instead of KL. Isolates whether A's
gentleness is the soft-target objective or the dream data. Caveat recorded:
cue splices are label noise for CE (mitigated by masking, §4), which biases
*against* CE-on-dream matching A — if it still matches, the "it's the
objective" verdict is safe.

**sft-ref:** CE on the raw wake transcript. **no-sleep:** nothing (floor).

**Retired:** as-run B1 (generate-while-draining) and B1-live — §6.

**Deferred, agreed for multi-sleep only (proposed this debrief, not part of
the single-sleep rerun): B2′ — install-then-erase.** Run B2; at end of
sleep, per fact: fresh-state margin check; only if it passes, apply one real
erase to the carried state along that fact's query. Erase as verified commit
step (memory policy), not as training signal. Its observable — freed
capacity, retained unconsolidated material vs A's clearing-loss — only
exists across multiple sleeps.

## 4. Metrics and dream-generation changes

- **Primary reliability: distractor-code margin.** Per fact:
  `margin = lp(correct code | question, fresh state) − lp(distractor code |
  same question, fresh state)`, where lp is the **sum** of token logprobs
  over the whole code (not the per-token mean — the threshold below is
  calibrated to the sum; log both). The distractor is a fixed random 5-digit
  code generated alongside the wake transcript **from its own RNG stream**
  (`Random(seed ^ const)`, so wake transcripts stay bit-identical to prior
  runs), stored in the dream cache, never equal to any real fact's code, one
  per fact per seed. Margin is immune to the format prior (both codes gain
  equally from "digits are likelier") and to the counting attractor. **A
  fact counts installed iff margin ≥ 1.0 nat** (correct ~2.7× the distractor
  on the full code). Chosen now, before data.
- **Paraphrase rate** secondary (it already caught what greedy missed).
  **Greedy EM** reported, never gating.
- **Binding-aware coverage:** a dream rehearsal counts only when the correct
  code appears adjacent to its own entity (window: same sentence). Misbound
  rehearsals reported separately.
- **Cues defer to sentence boundaries:** when the cue timer (`--cue-every`)
  fires, splice at the next `.`/newline within 20 tokens (hard-splice at 20
  if none). Removes mid-thought cuts (the prepare_chains splice lesson).
- **Cue tokens are loss-masked in every arm** — masked as **targets** only
  (positions whose next-token target is a cue token drop out of the KL/CE
  sum; the cue stays in context, and the first post-cue answer digit is NOT
  masked — it's the answer). They are state steering, not learning targets.
  A fully-masked chunk takes no optimizer step.
- **Cue accounting:** `--cue-every N` restarts its timer from the *end* of
  the spliced cue, so at `--cue-every 32` roughly 40% of a "512-token dream"
  is cue text and the trainable free-dream budget is correspondingly
  smaller. Accepted (installation feasibility beat purity, and cues are
  masked), but every cell logs its actual free-generation token count so the
  A-vs-sft-ref caveat is quantified.
- **Dream seed text carries the format separator**: the seed must be
  `[ASSISTANT]` + literal `" "` per the trained chat format — the current
  `.rstrip()` strips it, so every dream so far started one token
  off-format. Fix in the cache builder.
- **Periodic probes:** full probe battery (margin, paraphrase, battery ΔPPL)
  every 200 distill steps, streamed to the jsonl — every cell yields a
  learned-vs-forgotten *curve*, and iso-learning comparisons are read off
  curves, not engineered via hyperparameters.
- **LR fixed at 1e-4 for every arm and cell.** No LR matching between arms —
  LR changes damage physics (3e-4 sharpened the attractor and halved
  paraphrase), so matched-learning comparisons via LR are uninterpretable.
  A 3e-4 branch may be run and reported, separately, never pooled.
- The run log must include a decoded sample around one cue joint (sanity rule:
  read the splice, don't trust the counter).

## 5. Run plan — in order, on the box

**(a) Single-sleep rerun** — on a **GH200** (not the A10 — full-sequence
BPTT and the fused kernel path need it; distillation phases run several
times faster there, dream generation only modestly so, and execution stays
serial — the A10's 3-stream slowdown was context serialization, which speed
now makes moot). 6 arms × 3 seeds (1234/2345/3456), single budget d800:
A (full-sequence, fresh state), B1 (corrected), B2, CE-on-dream; sft-ref;
no-sleep — plus one bridge cell: A at the 08-06 configuration (chunk 48,
carried), seed 1234 only. Install-feasible regime:
`--n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7
--cue-every 32 --cue-greedy 12 --lr 1e-4 --distill-steps 800`, probes every
200 (the budget axis is replaced by the per-cell curve; d200 is dropped).
Every cell logs `token_gradients` — one A step is a 48-token chunk, one B
step is one token, so `--distill-steps` is not a cross-arm currency and the
frontier is reported on both axes. Result files use a fresh prefix
(`logs/g2_*.jsonl`) — the corrected B1 must not collide with the retired
drain arm's filenames in the summarizer's glob or the driver's resume check.
Serial execution (three-way concurrency measured 4× *worse* aggregate on
the A10). Before the grid: build the three dream caches, read one decoded
dream + one cue joint, verify binding-aware coverage ≥ 3/4 at every seed; a
seed below that gets its cache regenerated at `--cue-every 24` before any
arm runs (recorded per-cell; within-seed contrasts unaffected, cross-seed
pooling of absolute rates is what it taints).

**(b) Saturation ladder** — A-fresh and B2 at d3200 and d12800, seed 1234
only, same regime, `--probe-every 1600`. Plus one **deep-B2 cell** (no
detach on the carry: losses accumulate over the dream, one optimizer step
per pass, full BPTT through the intact spine) at seed 1234, matched token
budget — the gradient-depth variable measured directly rather than gated on
saturation. Score with the §7 fingerprint (dream re-write energy along v)
alongside the margin probes. Deep-B1 is not run (§6). Question: does B2's curve flatten
below A's level (per-token gap structurally too small) or keep climbing
(iso-learning comparison possible)? If B2 is still climbing at d12800,
extend it toward token-gradient parity with A@d800 (≈38,400 B-steps, ~7.6 h
— only the curve earns it); flat between d800 and d3200 ⇒ skip d12800 for
that arm, recording the deviation. Other seeds only if still climbing.

**(c) Multi-sleep A/B — the hypothesis's real test.** Gate: margin-install
fraction in (a), conditioned on binding-aware dream coverage, **pooled
across all three seeds, best dream arm, at the final d800 probe point,
≥ 0.3** (per-seed and per-probe-point values reported alongside so the gate
can be re-read without a rerun). If met: K=4 sleeps × 4 fresh facts per wake, arms A,
B2, B2′ (B2′ enters here). After every sleep: margin probes on **all facts
so far** + battery. Report the R-matrix (facts × sleeps), per-sleep curves,
backward transfer, and **cumulative installation** (a fact installing at
sleep 3 that A's clear destroyed at sleep 1 is the ratio advantage altrup
registered). Probes on zeroed copies only — state continuity across sleeps is
load-bearing for the B arms. Detailed cell list to be registered in its own
DISCUSSION section before launch; this entry fixes the shape and the gate.

Harness work items for (a), box or local: dream-cache build/load
(`--dream-cache`, refuse to run without it except the cache-builder itself);
`erase_hook` in `_mixer_step` (per-layer interleaved erase; hook-off must be
a byte-identical no-op, tested); corrected B1 (ablate before the student's
forward, in place, carry the student's detached state); student-query
erasure in both B arms (replacing the cached-teacher-query addressing);
explicit `.detach()` in `erase_state`; `--fresh-state-replay` for A;
boundary-deferred cues; cue target-masking; distractor codes in cache +
margin probes (sum-of-logprobs); binding-aware coverage (and delete the old
substring-coverage in `summarize_grid.py` — it counts misbindings by
construction); dream-seed separator fix; hash assertions in
`summarize_grid.py` **scoped to wave 1** (wave-≥2 dreams legitimately
differ per arm in (c) — record, don't assert); `token_gradients` logging;
periodic probes; add `sft/data/` (dream caches, knowledge battery) to the
box pull list in `scripts/lambda_data_artifacts.sh`. The A-vs-B2 same-seed
generation divergence gets one two-minute discriminating check — run the
*same* arm twice at one seed; if it diverges too, it's kernel
nondeterminism, not arm code paths — and no more (the cache makes it moot).
CPU-testable pieces (cache round-trip, masking, margin arithmetic, coverage
binding check, B1/B2 token-1 equivalence incl. gradients, detach, hook
no-op) get local TDD before the box session. The cache builder also writes
a decoded plain-text sidecar per seed (`sft/data/dream_s{seed}.txt`, cue
spans marked) so the literal training text lands on the local machine with
the pull.

## 6. Explicitly considered and rejected

- **As-run B1 (generate-while-draining).** Coverage 1/4 at every seed/budget;
  the drain starves the dream of rehearsal material and the drained teacher's
  logits stop expressing the facts. Structural, not tunable. The corrected B1
  (§3) trains on the shared intact dream instead.
- **B1-live.** One-seed exploratory, +0.03 dlogp — floor. The live feedback
  loop neither blew up nor learned; its control (as-run B1) shows the
  drain-during-generation was the fatal ingredient, not the live weights.
- **Matching arms by LR** (raise B's / lower A's) to compare forgetting at
  equal learning. LR is not a neutral speed dial (§4); curves from periodic
  probes answer the question without touching it.
- **Greedy EM as a gate.** Broken by the counting attractor; kept as a
  reported column only.
- **Erasing along cached teacher queries in B1/B2.** As the student drifts
  from the teacher, a cached direction loses aim against the state it cuts.
  Erase uses the student's own per-step query (free via `c_capture`);
  teacher queries stay in the cache for diagnostics.
- **B2′ in the single-sleep grid.** Indistinguishable from B2 there (probes
  zero the state anyway); deferred to (c).
- **Gradient through the erase directions.** Detached (§3) — a
  differentiable erase lets the optimizer dodge consumption instead of
  installing facts, the same compensation channel the γ=1 choice closed.
- **Deep-B1.** Cross-token gradient in B1 passes through every intervening
  ablation projector; the product annihilates all but the protected
  v-subspace, so its deep gradient is pre-aimed at the compensation harbor.
  Structurally confounded — the depth variable is measured on B2 instead.

## 7. Open questions

- **Cue-rate confound on A** (carried from the box's flags): cued dreams
  move A toward sft-ref; A won the frontier *with* cues. Is the A-vs-B
  contrast clean enough given all arms share the identical cued dream?
  Team's current answer: yes for A-vs-B (dream shared), unresolved for
  A-vs-sft-ref (CE-on-dream cell partly addresses it).
- **Value-side erase**: (I − γûûᵀ)S kills stored content along an output
  direction instead of the read address. Never measured; value-space cosine
  geometry unknown. Cheap local probe if the query-side cone keeps hurting.
- **Deflation k under multi-cluster states**: k=1 assumes one dominant mass
  direction. Several comparable spikes make v unstable and unprotected
  neighbors likely. Revisit when fact sets stop sharing one cone.
- **Soft dreaming** (08-05 §7): unchanged, parked behind the discrete
  baseline.
- **Warm-start before the first dream** (altrup, 08-06, agreed for the run
  after this one): the base model's first dream is generated with untrained
  `[USER]`/`[ASSISTANT]` embeddings, so the uncued stretches go
  off-distribution (mojibake, bracket-mimicry — observed live in the g2
  cache build). A short SFT pass first — enough to train the special-token
  embeddings — should keep free-running dream text coherent and raise
  rehearsal density per token. Not automatic today: `dream_sleep.py` builds
  fresh LoRA on the base every invocation with no checkpoint-load flag, so
  this needs a small harness addition (`--init-adapter <ckpt>`, loaded
  before both cache build and training) plus the warm-start becoming part
  of the shared preamble so every arm starts from the same weights. The
  multi-sleep phase partially measures the same effect for free: sleep 2+
  dreams come from trained embeddings, so dream-quality-across-sleeps is
  data.
- **B2's saturation ceiling** — measured by (b); if it flattens early, the
  per-token counterfactual gap is structurally too small and the erase
  mechanism's future is B2′'s policy role, not distillation signal.
- **B3 — teacher-spine counterfactual** (planned, altrup; named this
  debrief, not in the next run). Duplicate-GPU-work form, no state caching:
  the frozen teacher re-walks the cached dream per pass (no-grad, states
  only — its logits are already cached and identical). Per position t:
  1. Teacher's intact generating state S_{t−1} (the state before this
     token's logit).
  2. Student takes a copy, ablates it (same micro-order: deflated ĉ from
     the token, detached, cut before the write).
  3. Student's forward for token t from the ablated copy → logit → KL vs
     the cached teacher logit → optimizer step.
  4. Student's state is discarded entirely; the teacher continues from its
     own intact post-token state.
  Stationary counterfactual (the hosting spine never drifts with learning);
  costs 2× forward compute and accepts a train/probe state mismatch that
  grows as the student learns. Run after the registered grid establishes
  the student-hosted baseline, as the B2-vs-B3 contrast isolates what
  spine drift contributes.
- **Full-BPTT B2 variant** (promoted to a registered (b) cell, seed 1234):
  accumulate per-token losses over the dream, one optimizer step per pass,
  full graph (affordable on the GH200). Faster learning + fresh-state
  margins moving ⇒ depth was the bottleneck and the rewrite channel was
  benign; faster learning + flat fresh-state margins ⇒ compensation via
  erasure-resistant rewriting caught in the act. Either reading is
  informative; not part of the registered grid. The escape channel has a
  specific fingerprint: raw erase annihilates its own read identically
  (S(I−ĉĉᵀ)C = 0 — no resistant writing exists), so the only harbor is the
  deflation-protected v-subspace. Three bounds keep the channel narrow: the
  wake state is a cached constant (original fact storage cannot migrate —
  only the student's dream-time re-writes can be routed toward v); the
  harbor is rank-1 per layer (superposing facts there degrades retrieval,
  so gradient finds a partial equilibrium, not collapse); and the registered
  depth-1 form closes it entirely. If the deep variant runs, compensation
  shows up as dream re-write energy along v — measure that, expecting
  partial migration at benchmark scale (4 facts may fit the harbor) rather
  than collapse.

## 8. Housekeeping

- 08-06 notes banked as `db8a7bd` before discussion (flow rule followed).
- Working-tree rsync clobber of the 08-05 file's soft-dreaming block:
  restored from `2090b43`'s content (deletion stashed, then dropped).
- `sft/data/knowledge_battery_mamba2_780m.json` did **not** come back in the
  rsync pull — next box session must regenerate it (same self-calibration
  code, base PPL should reproduce ≈23.559) or push it, so batteries stay
  item-identical across runs.
- `.claude/commands/altrux-experimenter.md`: added the invariant rule (§2) —
  a registered invariant ships with its machine check.
- Local `main` carries unpushed debrief commits; altrup pushes.
- Dry-run audit (debrief step 5, first use): a context-free Opus
  experimenter read this file cold against the code and returned the
  findings folded in above — notably A's superseded replay schedule (M5),
  the non-comparable step currency (M6), the per-layer micro-order
  correction (M1), and three unenforced "properties" that are really code
  changes (student queries, detach, B1's erase placement). The step earned
  its place.
- Next box session's docket is §5 verbatim; local TDD list is in §5's
  harness work items.
