# Discussion notes — 2026-08-08: the headline collapse, the deep block, and the literature regime bridge

Team debrief (altrup + Claude) of the 08-08 GH200 session
(`EXPERIMENT_NOTES-20260808-135328.md`, banked `6abf671`), which ran the
08-07 docket: warm-start, operator picker, d800 baselines, fused arms,
partial ladder. Standing direction; supersedes
`DISCUSSION-20260807-g2-results-erase-geometry-and-warmstart-run.md` §4 as
the run plan and amends its §3 decisions as stated below.

**Standing frame, unchanged and re-affirmed by altrup this debrief: the
objective is the installation-to-forgetting ratio — learning at minimal
damage — never installation speed or raw install count.** A +1.88-margin arm
is acceptable if its forgetting cost is low; no arm is ranked by Δmargin
alone anywhere in this program.

## 1. What the 08-08 run established (agreed)

d800 frontier, pooled 3 seeds, floor-corrected, warm start `d4bf2e35…`
asserted on every cell (full tables in the run notes):

| arm | Δmargin | Δ-installs | dPPL | EM | para | token-grads |
|---|---|---|---|---|---|---|
| sft-ref | +22.25 | 10/12 | +1.828 | 3/12 | 0.23 | 110,279 |
| A | +7.72 | 11/12 | +1.055 | 2/12 | 0.12 | 836,800 |
| B1-deflated | +6.47 | 12/12 | +1.724 | 0/12 | 0.00 | 2,400 |
| B1-raw | +5.86 | 10/12 | +4.320 | 0/12 | 0.00 | 2,400 |
| B2tok-deflated | +2.67 | 3/4 (1 seed) | +0.461 | 0/4 | 0.00 | 800 |
| B3f-raw | +1.91 | 5/8 (2 seeds) | +2.572 | 0/8 | 0.00 | 564,800 |
| B2fd-raw | +1.88 | 2/4 (1 seed) | +1.448 | 0/4 | 0.00 | 276,800 |

1. **The g2 headline is dead: warm-started, A's damage ratio vs sft-ref is
   ~1.7×, not ~200×.** sft-ref +1.828 dPPL (essentially unchanged from g2)
   against A's +1.055 (up from +0.009), while sft-ref out-learns A 3× and
   out-retrieves it. Agreed reading: **g2's near-zero damage was
   substantially an artifact of degenerate (part-mojibake, off-distribution)
   dreams that barely moved the model**; make the dream real and the damage
   arrives with the learning. What dream distillation still wins outright is
   installation cost: B1-deflated's 12/12 at 2,400 token-gradients, 46×
   fewer than sft-ref for full coverage. The program's damage claim is
   **unproven in single-sleep and now lives or dies in multi-sleep's
   R-matrix** (single-sleep battery dPPL was never the literature's
   forgetting — 08-07 §1.9).
2. **Operator: DEFLATED, by §3.4 rule 1 dominance at 3/3 seeds** (12/12
   installs at +1.72 dPPL vs raw's 10/12 at +4.32; per-seed dominance on
   (installs, dPPL) all three seeds). The harbor apparatus (deflation,
   skip-cone guard, v-energy fingerprint) does not retire. The mid-session
   provisional raw call is overturned and its two supporting arguments are
   both dead: the 7–15× slowdown was a contention artifact (real cost
   ~1.75×), and single-shot readout removal does not predict cumulative arm
   behaviour.
3. **Single-shot erase geometry is not the operative quantity for arms that
   erase hundreds of times while carrying state** — raw zeroes the target
   readout exactly at every layer, deflation removes 0–20% of it, yet
   deflation wins cumulatively (plausibly: raw's total cut compounds
   destructively across the carry; see raw's +9.16 dPPL at seed 2345 and
   2/69 battery losses). Consequence, registered: **the local erase-probe
   program is demoted to diagnostics — no operator or arm decision may be
   made from single-shot geometry again.**
4. **Every fused-arm number this run is raw-only** (the fused block ran
   before the picker resolved). Given B1's raw-vs-deflated pairing shows raw
   ~2.5× the damage, all fused dPPL numbers are presumptively pessimistic
   and the fused family has never been measured under the winning operator.
   Fixed by next run's deep block (§3).
5. **The warm start works and moved the retrieval gap** — the first movement
   on 08-07 §1.6: A's paraphrase 0.02 → 0.12 (6×), EM 1/12 → 2/12, zero
   mojibake, 4/4 binding on all three seeds. A's damage rose with it
   (+0.009 → +1.055): learning and damage move together (the recurring
   pattern). Retrieval remains zero for every B arm.
6. **A saturates at d800; extra budget redistributes rather than
   accumulates** (flat on 2 seeds; strong facts decay while weak ones gain;
   `viola` is *anti-installed* and worsens monotonically, −8.23 → −11.05,
   while rehearsed 3× and correctly bound — a direct counterexample to
   08-07 §1.7's "installed ⟺ rehearsed-in-binding"). **B1-deflated is the
   one unsaturated arm** (+16% d800 → d3200, damage flat-to-falling), so it
   holds the d12800 rung claim. Competition-for-one-address-cone is the
   working hypothesis (§5).
7. **The fused machinery is sound**: spine fix `1e0f95c`, B3≡B2fd pass-1
   equivalence exact on two seeds on real hardware, ~100× wall-clock over
   per-token at matched token-gradients. B2-fused-deep has **never run**
   (OOM at 512 and 256 tokens; the deep graph is several × the state
   footprint; `--cf-batch`/`--spine-block` cannot fix it — checkpointing
   can, §3). B2tok-deflated (+2.67 at +0.461 dPPL) is the gentlest dream
   arm measured — 1 seed only.
8. Three real-hardware bugs found and fixed on billed time, none visible on
   the CPU fake backbone: batched `chunk_loss` (`10db48e`), the
   double-materialized spine (`1e0f95c`), B3's equivalence check holding two
   spines (`83ec296`). Standing lesson re-confirmed: green-on-fake-backbone
   is not evidence anything runs on hardware.

## 2. Decisions registered this debrief

1. **§3.1's acceptance clause is re-specified as a token-level threshold
   with a measured floor** (the run's request, honored): *plain-`]` tokens
   must be < 35% of marker-slot emissions in the free-running spans —
   counted at the token level (id 62 vs ids 50277/50278), never from the
   decoded string* (a decoded dream cannot distinguish the real special
   token from its plain-text imitation). Calibration: measured 77% at
   400 warm-start steps (fail), 26% at 800 (pass, with headroom under 35).
   Mojibake clause unchanged (zero non-ASCII). The experimenter's
   deliberate deviation past the second acceptance failure is **ratified**
   — all four recorded reasons endorsed, and the outcome (picker + frontier
   + ladder delivered) vindicated it.
2. **B3 is demoted from arm to tooling.** Its question is answered
   (within-sleep spine drift ≈ nothing at d800: +1.48 vs B2fd's +1.88, same
   installs). Retained: the pass-1 B3≡B2fd equivalence check as a harness
   invariant (it caught Bug 3), and B3f as B2fd's 2.6×-cheaper substitute
   **conditional on one deflated pairing confirming they still coincide**
   (§3 deep block). Reopen clause: B3 returns as an arm only if multi-sleep
   shows drift-linked anomalies in the B2 family.
3. **B-family consolidation is DELIBERATELY NOT DECIDED** (altrup): the
   next run's numbers come first. Recorded as a proposal only: B1-deflated
   survives unconditionally; the other slot is chosen on the
   install-to-forgetting ratio from {B4 variants, B2tok, B2fd, B2-deep if
   run} after the B4 block reports; target is two B arms into multi-sleep.
   (Amended same-day: B4 joined the candidate set with §2.7.)
4. **Multi-sleep is deferred a third time — explicitly deferred, not
   dropped** (altrup). It remains the decisive test of the
   no-catastrophic-forgetting claim (§1.1) and runs next-next session on
   whichever dream regime §3's bridge selects. If the next session runs
   long, cut the ladder and extras before cutting anything that feeds
   multi-sleep readiness.
5. **The dream regime gets a literature bridge** (altrup's direction): the
   program's one-dream/many-passes/cue-spliced regime is our invention —
   replay literature runs ~one pass over many fresh generated samples with
   coverage from volume, and nobody injects cues into a generation.
   Registered hypothesis: A's saturation/redistribution effects (§1.6) may
   be *repetition* pathologies rather than capacity limits. The bridge
   (§3 step 2) tests many-dreams/few-passes/no-splicing with weak steering
   only, paired against this run's numbers. Cue splicing is not deleted —
   the old regime remains the anchor until the bridge's adoption rule fires.
6. **All pre-checkpointing fused footprint/scheduling numbers are declared
   stale as of the checkpointing refactor**: the 82.7 GB steady footprint,
   the ≥65 GB-free requirement, "a fused cell is a solo tenant, fused cells
   run sequentially", and the `--cf-batch 128` ceiling were all measured on
   the materialize-everything spine. After the refactor the fused path
   should use far less VRAM. Next session **re-measures with a
   `--distill-steps 20` rate+footprint probe before scheduling anything**
   and re-derives the concurrency rule from the measurement — do not
   inherit the solo-tenant rule. (Amended same-day by decision 8: the
   refactor is now deferred, so until it actually lands the 08-08
   footprint rules — solo tenant, ≥65 GB free, `--cf-batch 128` — REMAIN
   VALID for any fused cell; the staleness declaration activates when the
   refactor does.)
7. **B4 — per-dream erase-then-replay — is registered as a new arm family
   and THE HEADLINE OF THE NEXT RUN** (altrup's direction, amending this
   file after its first dry-run pass). Mechanism, per dream: (i) during the
   teacher's generation pass, capture the raw per-layer read queries at
   every fact-rehearsal position (binding scan locates them; sidecar
   records per-fact contribution counts — a zero is loud); (ii) per layer,
   one SVD of the raw captured queries → orthonormal basis, rank r chosen
   by the frozen rank rule (below), capped at n-facts+1, every layer's
   spectrum + chosen r printed into the cache sidecar; (iii) erase the
   dream-start state once: S ← S(I − VVᵀ) — a projection, never a sum of
   per-query cuts (a summed cut over correlated queries over-subtracts
   into a sign-flipped anti-memory; worked example in the debrief
   transcript) and never σ-scaled (partial cuts compound as (1−γ)^N
   across N per-dream re-applications; projections are idempotent, so
   damage is independent of N); (iv) train the student on the dream as
   ORDINARY sequence training through the native SSD kernel against the
   teacher's cached logits — full BPTT, normal-training footprint, no
   spine, no cf-batch, no checkpointing. Variants, all from the same
   shared per-layer SVD: **B4-raw** (V as-is), **B4-deflated** (each
   column of V deflated against `state_top_dirs(S, k=1)` then
   re-orthonormalized — the faithful aggregate port of the operator that
   won the picker; harness asserts VᵀV = I for every final basis),
   **B4-qcm** (drop v₁, the query-consensus direction — budget-permitting;
   the direct test of 08-07 §6's query-common-mode question). Rank rule:
   two candidates — largest ratio-gap σᵢ/σᵢ₊₁ within the first n-facts+1
   slots, and σ > c·median(σ) (the median sits in the 128-direction noise
   tail; c ≈ 3–5) — the pilot runs both on every layer's real spectrum,
   prints disagreements, and the simpler rule that behaves is FROZEN
   before the box session; a spectrum with no clean structure is a
   stop-and-think finding, not a silent truncation. Framing, recorded:
   B1 = (erase every token, live student query), B4 = (erase once per
   dream, frozen teacher aggregate) — opposite corners of a 2×2. B4's
   single root projector means the projector-composition pathology that
   forbids deep-B1 does not exist for it: B4 is the one erase arm whose
   full BPTT is legitimate, and it gets it for free. Registered fallback
   variant if frozen-B4 badly underperforms B1: recompute the aggregate
   from the *student's current* queries each pass (the (once, live)
   corner) before concluding that carry is essential. Registered failure
   mode to watch: within-dream re-installation — after a fact's first
   rehearsal the state serves it again, so B4's pressure is per-dream,
   not per-read; the window is small iff dreams are non-repetitive, so
   the pilot measures per-fact within-dream repeat counts. Stakes,
   pre-registered: B4 ≈ B1-deflated's ratio at normal-training cost →
   carry was never the essential ingredient and the mechanism story
   simplifies enormously; B1 ≫ B4 → carry is the ingredient, B4 becomes
   the cheap screening arm, B1 keeps the mechanism crown. Either outcome
   is informative. Dreams veering off the facts is explicitly fine and
   plausibly protective (interleaved rehearsal of the general
   distribution); coverage stays aggregate-across-dreams, and no dream is
   penalized for wandering.
8. **The B2-deep 2×2 and the spine-checkpointing refactor are DEMOTED to
   conditional** (altrup, reversing this file's earlier deep-first
   ordering): checkpointing comes OFF the launch gate, and the deep block
   runs only if B4's numbers leave the depth question standing — B4
   delivers gradient-through-the-whole-trajectory for free, which was the
   question's substance. The 2×2 template in §3 is retained for that
   eventuality.

## 3. Next run plan — in order, on the box (GH200)

**LAUNCH GATE: no box until every item in §4's local-harness block is
implemented, tested, and PUSHED, and the §4 local pilot has reported.** The
gate's stop-and-report rule binds *box* sessions (an experimenter that
arrives on billed hardware and finds gate work missing stops); the gate
work itself is done by a local session on the teammate's machine — §4 is
local by definition. Push (commits and this file) before any launch:
`lambda_setup.sh` clones from GitHub, so unpushed fixes don't exist on a
box.

Budget guidance (not a cap): plan ~6 h. Priority on overrun: the B4 block
(which subsumes the regime bridge) > B2tok seeds > fused-deflated rerun >
ladder/competition extras. The conditional deep block (§2.8) is not in
this session's budget unless B4's results summon it AND the checkpointing
harness exists.

Regime unchanged unless stated: `--n-facts 4 --filler-tokens 40
--dream-tokens 512 --dream-temp 0.7 --lr 1e-4 --distill-steps 800
--probe-every 200`, floor-corrected margin, `--init-adapter` the step-800
warm start (`models/mamba2_780m/checkpoints/epoch-2/step-800`, trainable.pt
sha256 `d4bf2e3527befd5f78234a1baf3238624a956f78d3c24f1091d048be9ea08669` —
pulled home; re-upload with the launch, do NOT retrain a new warm start:
every registered pairing depends on this exact adapter). The three 512-token
seed caches and the battery likewise travel back with the launch
(`lambda_data_artifacts.sh` now carries `sft/data` by default — §6.3).

**(1) The B4 block — the session's headline (altrup), merged with the
regime bridge.** Per seed (all three): build the multi-dream cache — N
fresh un-spliced steered dreams (N and the steer prefix from the pilot),
teacher logits + captured fact queries + per-layer spectra in the cache;
aggregate binding gate ≥k dreams per fact. Arms on the shared dream set,
all at normal-training footprint (co-schedulable — these are NOT fused
cells; the 08-08 solo-tenant rules don't apply to them): **B4-raw,
B4-deflated** (B4-qcm budget-permitting), plus the bridge controls **A**
and **B1-deflated** on the same sets. Budget semantics per §3(2)'s pin:
`--distill-steps` stays the TOTAL step budget, spread p ≈ 800/N per
dream; B4's erase re-applied at each dream start (idempotent). Read
everything on the standing frame (ratio), paired against the 08-08
same-seed d800 numbers. Two registered verdicts come out of this block:
the regime adoption rule (§3(2)) and the B4-vs-B1 stakes (§2.7).

**(2) B2tok-deflated, seeds 2345/3456** — completes the gentlest-arm
measurement to 3 seeds; per-token cells, cheap, co-schedulable.

**(3) Budget-permitting, in order**: the fused family's deflated rerun —
B2fd-deflated + one B3f-deflated pairing (licenses the §2.2 substitute) —
under the STANDING 08-08 footprint rules (solo tenant, `--cf-batch 128`;
the checkpointing refactor is deferred, so these numbers are still
valid); then B1-deflated d12800; then the competition probe (both below).

**(4, CONDITIONAL — §2.8) The deep block** — runs only if B4's results
leave the depth question standing and the checkpointing harness has been
built. Retained spec: seed 1234, the existing 512-token cache, this
run's floor cells transfer (same adapter, battery, cache). A 2×2 on
{B2-fused-detached, B2-fused-deep} × {raw, deflated}; B2fd-raw exists
(+1.88 / 2-4 / +1.448), so three new fused cells. The block opens with its
own `--distill-steps 20` rate+footprint probe of the checkpointed arms
(§2.6's re-measurement) and takes `--cf-batch`/co-scheduling from that:

    MODEL_NAME=mamba2_780m HF_HOME=$PWD/../.cache/huggingface PYTHONPATH=$PWD/.. \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      uv run --no-sync python -u dream_sleep.py \
      --n-facts 4 --filler-tokens 40 --dream-tokens 512 --dream-temp 0.7 \
      --cue-every 32 --cue-greedy 12 --lr 1e-4 --distill-steps 800 --probe-every 200 \
      --erase-op $op --cf-batch <from (0)> --init-adapter $CK --seed 1234 \
      --arm $arm --out logs/g3_${arm}_${op}_s1234.jsonl

    # TEMPLATE, deliberately incomplete: --cf-batch (and any co-scheduling)
    #   comes from this block's opening re-probe, never from the 128 ceiling
    #   (sec 2.6 as amended). Arm spellings are pinned at gate close from
    #   `dream_sleep.py --help` and echoed into the run notes — whether deep
    #   is `--arm b2-fused-deep` or `--arm b2-fused-detached --deep`, and
    #   whether checkpointing is a flag or unconditional for deep, are the
    #   implementer's call, recorded.
    # $arm ∈ {B2-fused-detached (deflated only), B2-fused-deep (raw AND
    #   deflated)}

Read the deep-vs-detached contrast per operator on the standing frame
(ratio, not Δmargin). This block also delivers the fused family's first
deflated numbers (§1.4). **Fallback if checkpointed deep still OOMs at
512 tokens**: run the deep-vs-detached pair on the t256 cache
(`data/dream_cache_t256_s1234.pt`, pulled home, zero cells run against it,
binding 2/2/2/3) — an internally-valid paired contrast, not poolable with
the 512-token grid; record the deviation. The t128 cache is NOT a further
fallback: its dream never binds osprey or heron (sidecar: 2/2/0/0), so no
arm can install them — the "regenerate at tighter --cue-every" warning in
the 08-08 session logs belongs to it, not to the 512-token cache (verified
against both sidecars this debrief).

**The bridge-regime spec (the cache and rules step (1) builds on)** — same
seeds and wake transcripts, paired against
this run's d800 numbers. Per seed: build a **multi-dream cache** — N fresh
free-running dreams (N sized by the §4 pilot), **no cue splicing**
(`--cue-every`/`--cue-greedy` absent), each dream opened by the registered
steer prefix (exact text fixed by the pilot; prefix tokens are excluded from
scored/kept positions everywhere — they influence through state only);
teacher logits cached per dream; the whole set shared by every arm
(the shared-dream invariant, pluralized — the *set* hash asserted into
every result jsonl, layout and hash shape the implementer's call);
aggregate binding gate: every fact bound in ≥k dreams (k from the pilot).
**Budget semantics, pinned: `--distill-steps` stays the TOTAL optimizer-step
budget for the cell (800), spread p ≈ 800/N passes per dream — not
per-dream** — so the pairing against the 08-08 d800 numbers is at matched
total steps within each arm's own currency. Arms: per step (1).
**Adoption rule: the new regime
wins iff its install-to-forgetting ratio ≥ the old regime's same-seed pairs
AND retrieval (EM/para) is not worse; multi-sleep then runs on the winner.**
Old and new regime numbers are never pooled (different dreams).

**Detail for step (3)'s tail items**: B1-deflated d12800 (seed 1234,
`--probe-every 1600`; it alone holds a rung claim, §1.6); the competition
probe — one paired cell, seed 1234, `--n-facts 2` vs the existing 4-fact
result, same wake-transcript template: if per-fact margins at 2 facts
exceed the 4-fact per-fact margins materially, within-cone competition
gains direct support.

**(5) Multi-sleep: NOT this session** (§2.4). Next-next session, on the
bridge's winning regime, per 08-07 §3.7's arm list amended by whatever §2.3
consolidation decides.

## 4. Local work before the box (launch gate; TDD on the CPU fake backbone
where testable, but §1.8 stands — hardware-shaped code needs a hardware
smoke, so step (1) opens with one short B4 cell at a token budget before
committing the block)

- **The B4 harness** (the gate's centerpiece): (a) query capture at
  fact-rehearsal positions during cache build (reuse `Model.c_capture` —
  the capture plumbing `erase_probe.py` already uses), per-fact counts to
  the sidecar; (b) the per-layer SVD aggregate: shared basis, both rank
  rules computed and printed (§2.7), the three variant post-processings
  (raw / deflated-vs-state-top-dirs with re-orthonormalization / qcm),
  and a test asserting VᵀV = I for every final basis; (c) erase-at-
  dream-start plumbing + the B4 arm flag(s); (d) tests: projection
  idempotence (applying the eraser twice ≡ once), erased-readout-is-zero
  along every captured query for the raw variant on the fake backbone at
  h*p=4, and a repeat-application test across N≥3 mock dreams (damage
  independent of N).
- **Spine gradient-checkpointing — OFF THE GATE, conditional (§2.8)**:
  built only if the deep block is summoned. Retained spec: phase 2 of
  `spine_states` wrapped in `torch.utils.checkpoint` over a MixerState
  flatten/unflatten closure (run notes' option (a)); equality tests:
  checkpointed-deep gradients ≡ materialized-deep on the fake backbone at
  h*p=4 (per `1517985`), checkpointed-detached ≡ current-detached loss.
- **Multi-dream cache format + training loop**: N dreams + per-dream teacher
  logits per seed cache file; per-pass iteration over the dream set; per-arm
  step currencies unchanged within each dream.
- **Steer prefix: extend the existing `--dream-prompt`** — it is already
  plumbed through generation and into the jsonl (dry-run finding; do NOT
  add a parallel `--dream-prefix`). The actually-missing pieces: prefix
  tokens excluded from the keep/score mask in every arm, and the prefix
  text recorded in the sidecar.
- **Aggregate binding gate** for multi-dream caches (every fact bound in ≥k
  dreams; per-dream gate retired for these caches — today's `report_dream`
  only warns, it must gate).
- **Pin the verbatim arm spellings** for §3(1)'s template from
  `dream_sleep.py --help` and echo them into this file's margin or the run
  notes at gate close.
- **The local pilot (before any format is frozen): ~20 steered
  free-running dreams** against the seed-1234 wake transcript (extract from
  `dream_cache_s1234.pt` — do not regenerate), 512 tokens, temp 0.7,
  distinct generation seeds, on this machine (`HSA_OVERRIDE_GFX_VERSION`
  set; the manual mixer loop is slow here — overnight is acceptable, local
  time is free). The prefix/pilot circularity is resolved deliberately:
  draft 2–3 candidate prefixes (short, non-instruction — e.g. a bare
  marker-format opening), run the pilot over all candidates, select on
  per-fact rehearsal rate, record every candidate's numbers. Output:
  per-fact rehearsal rate → N = smallest count where every fact is bound in
  ≥k dreams with ~1.5× headroom, k = 2 unless the rates argue otherwise
  (record the reasoning). The pilot additionally reports, per §2.7:
  per-fact **within-dream repeat counts** (B4's re-installation window),
  and the per-layer **σ spectra of the captured queries with both rank
  rules applied** — where the rules disagree, print it; the surviving
  rule is frozen into the spec before the box. **Kill-condition: if
  covering all 4 facts needs
  N > ~16 dreams under the best candidate, the no-splicing regime is
  unaffordable as specced — stop, report, and the bridge is redesigned
  (stronger steering) rather than launched.**
- `summarize_grid.py`: extend to the multi-dream cache jsonls; investigate
  the `--fallback final_floor` no-rows behaviour the run notes flagged
  (ladder tables were computed by hand this run).

## 5. Explicitly considered and rejected

- **B1-deep and B3-deep as "deep" candidates** (asked this debrief):
  B1-deep stays rejected on the 08-07 §3.6 projector-composition argument
  (its cross-token gradient is annihilated except on the protected
  subspace — new math required to reopen, not budget). B3-deep is vacuous,
  not rejected: B3's spine is cached data, not a function of the live
  weights; there is nothing for BPTT to flow through, and a B3 whose spine
  responded to the weights *is* B2-fused-deep by definition. The deep
  program is B2-fused-deep only.
- **Restarting the program on a literature-standard setup** for external
  validity: the harness (warm start, floors, battery, caches, summarizer)
  is regime-agnostic; the literature difference is the dream regime alone,
  which §3(2) tests as paired cells at a fraction of a restart's cost and
  with cross-run continuity intact.
- **Prompting the model to generate the dream we want** (in place of cue
  splicing): a 780m base model with a light LoRA has no instruction-following
  to prompt at; a prompt's tokens shade the whole dream through state and
  cannot be masked out of what generation conditions on; and it changes what
  arm A means (distilling "the model following an instruction"). The *weak*
  version — a short non-instruction steer prefix plus coverage-from-volume —
  is exactly §3(2)'s bridge, kept.
- **Deciding the B consolidation this debrief** — deferred by altrup until
  the deep block reports (§2.3).
- **Running the fused ladder rungs for B3f/B2fd** — both far off the strong
  arms at d800 under raw; ladder budget goes to B1-deflated (and A is
  saturated, holding no rung).
- **Treating the 08-08 fused footprint numbers as standing hardware
  findings** — declared stale (§2.6); they describe deleted code once the
  checkpointing lands (which is now conditional — see §2.6's amendment).
- **B4 aggregate as a sum of per-query cuts** — over-subtracts along the
  shared cone into a sign-flipped anti-memory (at cos 0.92, a state
  component along the consensus direction lands at ≈ −0.9× its original
  value); the uniform-downscale correction (scale every cut by
  1/(1+Σ overlaps)) is exact only in the fully symmetric case and leaks
  ∝ asymmetry. The projection S(I − VVᵀ) IS the completed version of the
  scaled-sum idea (inverse-Gram cross-corrections), so the sum forms are
  strictly dominated.
- **σ-scaled partial erase** (cut each direction in proportion to its
  singular value): σ measures the query set, not the state's content;
  partial cuts leave re-amplifiable residue (γ-sweep precedent, 08-07
  §2.5); and a non-projection eraser compounds as (1−γ)^N under B4's
  per-dream re-application — damage would depend on N. All-or-nothing per
  direction, γ = 1.
- **Hard-coded σ thresholds** ("top half", mean-based midpoint
  (σ_min+σ_max)/2): top-half cuts ~64 of 128 address dims (~5 signal, ~59
  noise); the midpoint is dominated by the common-mode outlier σ₁ and
  keeps ONLY the contested direction while dropping every fact
  discriminant (fails even the cos-0.6 two-vector case). Threshold rules
  must be scale-invariant, σ₁-robust, and tail-aware — hence §2.7's two
  candidates and the pilot bake-off.
- **Erase at fixed token intervals ("chunk-boundary erase")** as the
  B1/B4 midpoint — proposed and withdrawn this debrief: an arbitrary
  hard-coded joint; dream boundaries are the semantic joints, and the
  multi-dream regime already provides them (altrup). Revisit only if B4's
  within-dream re-installation window measures large AND dream generation
  cannot be made less repetitive.

## 6. Open questions

1. **B4 vs B1: is carry the essential ingredient, or is the erase?** —
   §2.7's pre-registered stakes; the session's headline question
   (altrup's priority). Subsidiary: raw vs deflated at aggregate level,
   the qcm variant, and the within-dream re-installation window.
2. **Does BPTT through the spine change B2's ratio, under either
   operator?** — demoted with the deep block (§2.8); answered only if B4
   leaves it standing.
3. **Is A's saturation a repetition artifact?** — the B4 block's A-cell
   answers directly (fresh dreams accumulating where repeated dreams
   redistribute would dissolve much of the competition story).
4. **Within-cone competition** (`viola`: bound, rehearsed, anti-installed,
   monotonically worsening) — §3(3)'s 2-vs-4-fact probe; also watch for
   loser-facts in every multi-dream cell.
5. **The retrieval gap, B-family edition**: the warm start moved A's
   retrieval but no B arm has ever retrieved anything (EM/para 0 across the
   family). If B4 also retrieves nothing at d800-equivalent budgets,
   "installs a preference, never a retrievable answer" may be the erase
   mechanism's ceiling — worth a named measurement before multi-sleep
   bets on the family.
6. **The mechanism/outcome contradiction** (§1.3): why does a near-absent
   single-shot erase win cumulatively? The carry-compounding hypothesis is
   unmeasured; a cheap instrument would be battery dPPL vs token index
   within one B1 cell (raw's damage should grow super-linearly, deflated's
   ~linearly, if compounding is the mechanism).
7. **The ~0.75 query common mode** (08-07 §6) — now directly testable:
   B4-qcm is its first designed cell.
8. Carried unchanged: warm-start's effect on dream binding (multi-sleep
   measures it), value-side erase, deflation-k under multi-cluster states,
   soft dreaming.

## 7. Housekeeping

- Run notes banked as `6abf671` before discussion (flow rule).
- **Same-day amendment (B4)**: §2.7/2.8, the §3 reorder, and the §4/§5/§6
  changes above were added AFTER this file's first dry-run pass, in a
  second debrief sitting (altrup's direction). The B4 sections have NOT
  been dry-run — **re-run the cold-agent pass after the gate work lands
  and before any launch**; the first pass's findings on the pre-amendment
  file are recorded below and remain folded in.
- The banked notes end in a stale rsync-clobber fragment (the superseded
  "session operator: RAW" text glued after the DEFLATED verdict, lines
  ~780–785). Trimmed in a follow-up commit this debrief — the DEFLATED
  verdict paragraph is authoritative; third occurrence of the
  clobber-shape (08-06 soft-dreaming, 08-07 warm-start block).
- **`scripts/lambda_watchdog.sh` default `--pattern` extended** to the full
  alternation this session used
  (`train.py|probe_recall.py|consolidation_null.py|capacity_ladder.py|dream_sleep.py`)
  so a dream-distillation session no longer depends on 25-minute delay
  touches for its entire duration (the run notes' watchdog gotcha).
- **`scripts/lambda_data_artifacts.sh` empty-var footgun fixed**:
  `${LAMBDA_DATA_ARTIFACTS-…}` → `${LAMBDA_DATA_ARTIFACTS:-…}` (empty now
  means "defaults", not "pull nothing"), with `.env.example` documenting
  that disabling requires a non-matching pattern. Root cause of both the g2
  cache loss and this run's near-miss (an empty-but-set var survived in
  `scripts/.env` *and* in the tmux server environment — the second copy is
  why the first watchdog restart didn't fix it).
- This run's artifacts verified home byte-exact (receipt + local `ls`
  cross-check, 15:43 entry): 3× 512-token caches, t256/t128 caches (valid,
  zero cells run against them, not poolable with 512-token grids), all
  sidecars, battery, step-800 checkpoint.
- Per-arm budget currencies (pass vs token-step) make shared-d budgets
  apples-to-oranges; multi-sleep planning should register per-arm budgets
  at each arm's measured saturation instead of one shared `--distill-steps`.
  Noted for the multi-sleep docket, not actioned here.
- Dry-run audit (flow step 5): a context-free Opus experimenter read the
  repo cold, correctly refused to launch (gate fails: 5 of 6 §4 items
  missing, pilot never run), and returned 10 findings; the substantive ones
  are folded in above — the gate-semantics clarification and push-before-
  launch rule (§3 header), the §3(1) command demoted to a template with the
  stale `--cf-batch` excised and arm spellings deferred to gate close, the
  deep-OOM fallback to the t256 cache, the `--distill-steps` total-budget
  pin (§3(2)), the `--dream-prompt` reuse (a fully-plumbed steer flag
  already existed — §4 had specced duplicate work), the pilot's
  prefix-candidate resolution of the pilot/prefix circularity, and the N/k
  selection rule. Its most consequential catch — a "fact this dream never
  binds" warning in the session logs that §3 might be pairing the whole
  deep block against — was run down during this debrief: the warning
  belongs to the discarded t128 cache (sidecar 2/2/0/0), not the 512-token
  cache (4/4/4/5, healthy). Two experimenter-command staleness items for
  the next command edit: its data-sanity fallback names
  `data/warm_start.pt`, which no longer exists (the warm start survives
  only as the checkpoint); and the command assumes it is always on a box —
  it needs a "local invocation: gate work only, no launch authority"
  branch. [command edits pending implementation]
