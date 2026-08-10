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
9. **Extracted this debrief from the pulled jsonls (probe curves the run
   notes never surfaced): A was already saturated at its FIRST probe —
   step 200, i.e. its 200th pass over the dream — on all three seeds**
   (per-fact-sum Δmargin 200→800: +40.3→+38.7, +17.9→+15.8, +37.9→+38.2;
   flat-to-declining, redistribution visible from the first measurement —
   viola at −8.9 by step 200). B1-deflated at step 800 has read the dream
   1.6 times and is still climbing steeply on 2/3 seeds. **The shared
   d800 budget therefore compared A two hundred passes past saturation
   with B1 mid-climb on its second reading** — different regions of two
   unrelated exposure curves. Consequences: (a) **the A-vs-B1
   token-gradient efficiency ratio is UNMEASURABLE from existing data** —
   A's true requirement is only bounded in (0, 69k], so the ratio lies
   anywhere in ~0.1×–87×, direction unknown; the 349× figure (and any
   "corrected" variant) is retired until fine probes produce a real
   saturation point; (b) process lesson, standing: **probe floors must
   sit BELOW the region where saturation is plausible** — d800 with
   probe-every-200 hid a pass-≤200 saturation through two full runs and
   a debrief.

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
   every position that passes the **state-dependency gate** (§2.9.2 —
   self-supervised, no fact knowledge; sidecar records per-fact
   contribution counts via the validation overlay — a zero is loud);
   (ii) per layer,
   one SVD of the raw captured queries → orthonormal basis, rank r chosen
   by the frozen rank rule (below), capped only by the prod-side address
   budget (below — never by the known fact count), every layer's
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
   two candidates — largest ratio-gap σᵢ/σᵢ₊₁, and σ > c·median(σ) (the
   median sits in the 128-direction noise tail; c ≈ 3–5) — both searched
   within a **prod-side address budget**: an eraser may spend at most a
   fixed fraction of the state's address dimensions per sleep (~1/16 of
   d_state, a resource constraint statable without any fact knowledge —
   NEVER the injected fact count, which exists only in the harness;
   caught by altrup after two fact-count caps leaked into this rule in
   violation of §2.9.1) — the pilot runs both on every layer's real
   spectrum,
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
9. **Third-sitting amendments (B4 refinements, all altrup+Claude same
   day):**
   1. **Prod-validity is a standing design constraint on the eraser
      (altrup): no ground-truth fact knowledge anywhere in the mechanism
      path.** The known facts are evaluation instruments only. This
      rejects fact-aware capture (probe-forced fact questions, and the
      binding-scan position gate) as the *mechanism* — both survive as
      validation overlays.
   2. **Capture gate = state-dependency, self-supervised**: re-score the
      dream's tokens under a blank (fresh) state with the same weights;
      capture queries at positions where the with-state and blank-state
      predictions diverge sharply (threshold from the pilot, printed).
      A memory read is a position where the state changed the prediction —
      defined by the state, not by a fact list. Generic reads ("the")
      drop out automatically. Validation, printed per cache: agreement
      (precision/recall) between the gate and the binding scan's
      rehearsal positions. [Amended 2026-08-10 (altrup): the gate SELECTS
      B4's queries; it is NOT a dream-validity requirement. A dream with
      an empty gate gets an empty eraser — it read nothing from the
      state, so there is nothing to deny and it contributes no denial
      pressure — loudly counted, build continues; an empty variant basis
      (e.g. qcm over one query) is likewise a NOTE. The only gate-level
      stop-and-think is the pilot's separability kill-condition, which
      tests the gate CONCEPT, not any one dream.]
   3. **The warm start is RETRAINED next session, with a new special
      token `<|endofconversation|>`** appended to every rendered
      conversation (boundary-marker family, matching `<|endoftext|>`;
      NOT named "end of dream" — a token means what its data placement
      teaches, and the corpus teaches conversation-end; dream-end is a
      sampling-time interpretation). Same regime (800 steps, 2 epochs,
      pinned corpus). Consequence, priced and accepted (~2 box-hours):
      the old-regime anchor cells (A, B1-deflated, no-sleep floors × 3
      seeds), battery, and caches are rebuilt under the new adapter;
      the 08-08 numbers become the old-adapter record and are never
      paired cross-adapter. §3's "do NOT retrain" pin is superseded by
      this registration.
   4. **Dream termination**: eos (`<|endoftext|>` = end of assistant
      turn in this corpus) is UNBANNED in dream sampling; the dream ends
      when the model emits `<|endofconversation|>`, with `--dream-tokens`
      demoted to a hard max (and a turn-count backstop if `<|eoc|>`
      never fires). Dreams become variable-length and self-terminating.
   5. **`<|endofconversation|>` as the dream-start steer**: prefixing the
      dream with the boundary token conditions generation to open a
      fresh conversation while the state (untouched by tokens) still
      carries the wake memory — a single trained steer token, in-
      distribution by construction. Registered as a pilot steer
      candidate and the expected winner over hand-written prefixes; it
      doubles as the multi-dream separator.
   6. **The A-vs-B4 contrast is the targeting measurement — no new
      control cell needed** (corrected in-debrief by altrup; a proposed
      "reset-control" was struck when reading `distill_replay` confirmed
      arm A's student ALREADY trains from a fresh state — `state=None`,
      `dream_sleep.py:680` — an intact-state student would have zero KL
      and could never learn). The spectrum as it actually stands:
      **A = total denial** (blank student re-learns everything
      state-dependent; diffuse signal; measured +7.72 / +1.055 dPPL /
      836,800 token-grads), **B4 = targeted denial** (student keeps all
      non-fact context — content the captured queries never implicate
      cannot enter V — so the KL localizes on the facts),
      **B1 = continuous targeted denial**. Pre-registered prediction:
      B4 ≈ A on installs at materially lower damage and far fewer
      effective token-gradients — "surgical A". If B4 ≈ A on damage too,
      targeting buys nothing over blanket amnesia and the SVD apparatus
      is decoration.
   7. **Scrub-state is rejected; scrub-capture is parked** (§5 entry);
      **lucid/on-policy dreaming is a named future direction** (§6).
10. **Fourth-sitting decisions — the run's FINAL shape. This subsection
    supersedes any earlier line it contradicts; a fresh session should
    treat §2.10 + §3 + §4 as the complete spec.**
    1. **Arms next run: A, B4-raw, B4-deflated. Nothing else** (altrup).
       B1 leaves the block — the §2.7 carry-vs-once stakes are DEFERRED
       to a later dream-set run (pairing B4 against 08-08's B1 is
       invalid cross-adapter and cross-regime). B2tok seeds and the
       fused-deflated rerun are struck from the docket.
    2. **The hundreds-of-passes regime is retired program-wide**
       (altrup: "A was supposed to be the literature control, not
       something we made up"). Replay-literature norm (from model
       knowledge — `RESEARCH-20260805`'s prior-art note records no
       exposure norms): many distinct replay samples × ~1–3 exposures
       each (DGR: fresh generator samples per batch; LLM synthetic
       rehearsal: corpus upfront, 1–3 epochs); nobody trains hundreds
       of epochs on one tiny sample. Next run: **one pass per dream**;
       one optional multi-epoch variant cell (~3 epochs over the set)
       prices the repeat knob. Motivating data in §1.9.
    3. **Per-dream erasers** (altrup, overriding a pooled-eraser
       proposal): each dream's V is built from that dream's own gated
       captures and applied to that dream's fresh wake-state copy —
       "each dream denies what it reads; everything else stays."
       Recorded counter (Claude, once, accepted-and-overruled): the
       rank budget already prevented pooled-V growth with N, and
       per-dream bases are noisier estimates of the same fact subspace
       (~4–8 queries per SVD). The pilot prints per-dream basis sizes
       and cross-dream V-overlap so the noise cost is measured, not
       argued. (Per-SLEEP rebuild in multi-sleep is unchanged — the
       eraser tracks the state, which changes at sleep boundaries.)
    4. **Carry matrix, pinned.** Student weights carry across the whole
       dream set (one optimizer trajectory — resetting them per dream
       would leave only dream N's learning). Student state resets at
       every dream boundary: A from blank, B4 from a fresh erased copy
       of the wake state (never erase-once-then-carry: carried state
       would ferry dream k's re-written facts into dream k+1).
       Teacher weights and logits are the frozen sleep-start snapshot,
       cached. **All dreams are generated upfront at cache build**,
       each from a fresh copy of the intact wake state: pairing
       requires one shared set (generation is nondeterministic), and
       interleaving generation with training couldn't change the
       dreams anyway — the generator is the sleep-start snapshot by
       registration; generation that tracks the student is the parked
       lucid direction (§6.8).
    5. **Probes at every dream boundary**: the full round (4-fact
       install + battery dPPL), seconds-to-tens-of-seconds each,
       rate-checked in the opening smoke (degrade to every-other
       boundary if it measurably drags, recorded). The per-dream
       install/damage curve is a primary deliverable — the step-200
       blind spot (§1.9) is the lesson.
    6. **The eraser goal block — what every gate/weighting/rank choice
       optimizes (altrup: the goal matters as much as the details).**
       Prime directive: a starting state that behaves identically to
       the intact wake state EXCEPT where the state was supplying
       remembered content — the KL gap opens at memory reads and
       nowhere else. Proxies, scored on the actual wake state (oracle
       labels allowed — validation side): **target removal** (% readout
       removed along oracle fact-read queries; too low → no
       consolidation pressure, degenerates toward a do-nothing cell)
       vs **collateral removal** (% readout removed along context reads
       and battery-item queries; too high → damage + violates
       leave-context-in, degenerates toward A). Every scheme is a point
       on this plane; better = closer to (100%, 0%). Decision
       procedure, lexicographic: (i) discard schemes failing mechanical
       sanity (VᵀV=I, address budget, non-pathological cross-dream
       stability); (ii) a dominating scheme wins; (iii) else highest
       target-removal among schemes with collateral below the knee of
       the pilot's own collateral distribution; (iv) ties break toward
       simplicity (fewer constants, hard before weighted, plainer rank
       rule); (v) full table + chosen point into the pilot report, and
       **the freeze happens at a team check-in — experimenter proposes,
       team ratifies, then frozen for the box.** Named NON-goals:
       label accuracy (precision/recall vs the binding scan is
       diagnostic only — the state, not the labels, is what the eraser
       touches); and **downstream install numbers — the pilot must
       never pick the gate by mini-training outcomes** (tuning the
       mechanism on the experiment's own metric; it would also overfit
       per-dream noise). The eraser is chosen on state geometry alone;
       the box run measures whether it works, blind.
    7. **Gate-pilot design** (hard-vs-weighted + thresholds + rank
       rule, one instrumented run): capture EVERYTHING during the
       pilot generation (every position's query, both distributions,
       divergence D_t) so every scheme is evaluated OFFLINE from one
       run (harness-only instrumentation; prod runs one frozen
       scheme). Test 1 — separability: D_t distributions for
       binding-scan fact positions vs all else, AUC per dream and
       pooled; **kill-condition: if fact reads don't separate from
       context reads on divergence, the gate concept fails — stop and
       rethink before any harness is built on it.** Test 2 — bake-off
       over {hard@swept-τ, divergence-weighted+floor, weighted-capped
       (√D or clip)}: per scheme build each dream's per-layer V and
       score (a) subspace fidelity vs the oracle basis (principal
       angles), (b) THE DECISION METRIC — the target-vs-collateral
       erase tradeoff of §2.10.6 measured by applying V to the wake
       state, (c) cross-dream stability. Deliverable: one table,
       scheme × {AUC, P/R at operating point, oracle overlap, target
       removed, collateral removed, rank distribution}; frozen choice
       + constants per §2.10.6(v).
    8. **Steer prefix: candidates and criterion.** Candidates —
       **no-prefix** (the incumbent: every dream this program has ever
       generated starts bare, and 08-08's free spans rehearsed 4/4
       from state alone), `<|endofconversation|>`, `<|eoc|>[USER]`,
       `<|eoc|>[USER]␣` (literal-space variant), and the
       instruction-text fallback ("<|eoc|> You are now dreaming,
       generate dreams based on past events [USER]") — measured, not
       argued, though the 780m's instruction-following prior is ~nil.
       Token candidates are re-checked after the §3(0) retrain (the
       pilot's old adapter has never seen `<|eoc|>`). **Criterion:
       coverage FEASIBILITY, not rehearsal-rate maximization** (altrup)
       — a candidate qualifies if every fact appears somewhere across
       an affordable set (N ≤ ~16 with headroom); among qualifiers
       take the simplest, never the densest (density-optimizing would
       re-create the cue-stuffed dreams this regime retired). Unrelated
       and off-topic dreams are explicitly fine.
    9. **Packing is MIXED, with a small recap fraction** (altrup's
       hallucination concern, sharpened: packing exclusively unrelated
       conversations after `<|eoc|>` trains the model to IGNORE ITS
       STATE at the boundary — an anti-recall signal aimed at the
       exact position every dream starts from). Most packed boundaries
       are followed by an unrelated conversation (clean fresh-start);
       a small fraction (~⅓ or less — the filler-synthetic
       regurgitation rejection is the watch-item, and the acceptance
       check's repeat clause is the alarm) are followed by a
       MECHANICAL RECAP of the preceding conversation (quote/shuffle
       earlier turns into a short follow-up exchange). Post-boundary
       distribution becomes "maybe new, maybe recall" — dream
       semantics, trained. In prod the `<|eoc|>`-means-dreaming
       association is TRAINED, never prompted; when that vision
       matures, align with the 2.7B chain corpus's `[SLEEP]`-token
       convention instead of inventing a parallel one.
    10. **Sampling policy** — day/wake generation: `<|endoftext|>`
        normal (ends the agent's turn); `<|endofconversation|>`
        **masked from logits** (sleep is system-triggered; the model
        can never put itself to sleep). Dream generation: both freely
        sampled; `<|endoftext|>` is dream-internal turn structure and
        ends nothing; `<|endofconversation|>` ends the dream
        (`--dream-tokens` a hard max, turn-count backstop). Wake is
        teacher-forced in-experiment, so the day-side mask is a
        registered runtime/backend policy, not next-run code.
    11. **Rich wake transcripts**: wake = facts + heterogeneous
        distractor content (the existing bystander machinery —
        off-format facts, different relation templates — plus a slice
        of ordinary ultrachat dialogue), because the A-vs-B4 contrast
        is entirely about non-fact state content and 40 tokens of
        filler starves it, and real waking holds many things (08-07
        §2.7 pre-cleared capacity: 7–11 items bind fine).
        **Automated collision guard**: the transcript builder checks
        distractor text against fact names/codes AND battery answer
        strings and refuses to build on overlap (the parcel→shipment
        precedent). **Context-leakage probe class**: post-training
        fresh-state QA on distractor content — neither arm should
        install it; leakage = untargeted consolidation, measured at
        its origin. **Sequencing constraint: the pilot runs on the NEW
        wake shape** — N, prefix, and gate thresholds all inherit from
        rehearsal rates off the rich state; a crowded state may lower
        them, and "un-cued dreams can't cover facts from a realistic
        state" (kill-condition firing) would itself be a headline
        regime finding, learned locally for free.
    12. **A's mechanics, for the record after an in-debrief
        misstatement**: `distill_replay` trains the student from a
        FRESH state (`state=None`, `sft/dream_sleep.py:680`) — an
        intact-state student would have zero KL and could never learn.
        A = total denial; B4 = targeted denial; the A-vs-B4 pairing IS
        the targeting measurement (§2.9.6).
    13. **B4's sleep-exit carry, registered post-gate (altrup,
        2026-08-09)**: the state a B4 sleep carries into the next wake
        is the wake state ablated by ALL dreams' reads — per-dream
        ablation during training, the union at sleep exit ("the dreams
        collectively denied it, so it lives in weights now").
        Construction: stack every dream's final V rows, re-orthonormalize,
        project ONCE — never apply per-dream erasers sequentially (a
        product of complement projectors is order-dependent and not a
        projection; the union projection is idempotent, the §2.7
        argument again). Implementation deferred to the multi-sleep
        docket — set caches refuse `--waves > 1` until then, and the
        code's current `ARM_CARRY` intact-state line is a placeholder
        this supersedes. Watch-item, measured not argued: union rank vs
        the per-SLEEP address budget (per-dream d_state/16 caps can
        union past d_state/16 when cross-dream overlap is low — read
        the pilot's cross-dream V-overlap). A's multi-sleep carry
        (blank) is registered where it always was, 08-07 §3.7's docket.

    14. **SUBSTRATE SWITCH: the next run is plain Mamba2-2.7B, not 780M
        (altrup, 2026-08-10, post-pilot).** Motivation: the §4 pilot showed
        generator quality is the binding constraint on the un-spliced
        regime (coverage {1,1,1,4}/20 no-prefix, {3,0,0,0}/20
        instruction-text; drift, verbatim loops, one-shot rehearsal — see
        `EXPERIMENT_NOTES-20260810-024146.md`), scale attacks exactly
        that, and §2.9.3's retrain already severs every cross-run pairing
        — the rebuild is paid either way, so the switch is uniquely cheap
        NOW. The A-vs-B4 ranking-transfer risk (arms differ in
        sensitivity to dream quality) is also retired by measuring on the
        stronger generator directly. Consequences, priced: per-token cost
        ~3.5×; arms are A + B4-raw + B4-deflated only (ordinary sequence
        training, co-schedulable — the reduction is what makes this
        affordable); cut to 2 seeds before cutting cells on overrun; the
        steer/coverage pilot re-runs ON THE BOX post-retrain (the local
        card's 8 GB makes 2.7B local pilots marginal); the block opens
        with a binding-capacity smoke (4-fact wake, cue-free) because
        every capacity number on record was measured on 780M and is
        assumed, not known, at 2.7B. Mechanics: a new plain
        `models/mamba2_2_7b` folder (thin adaptation of `mamba2_780m` —
        `state-spaces/mamba2-2.7b`, same interface, same SPECIAL_TOKENS
        incl. `<|eoc|>`; `mamba2_2_7b_memory` is the memory-augmented
        variant and is NOT this); every §3 command reads
        `MODEL_NAME=mamba2_2_7b` and checkpoint paths under
        `models/mamba2_2_7b/`. Dream-quality expectation, registered
        honestly: coherence/looping improve with high confidence,
        rehearsal FREQUENCY is the unknown the on-box pilot measures.
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
--probe-every 200`, floor-corrected margin. [Gate-close margin notes:
(a) for SET-cache cells (`--dreams N`) the budget flags are `--dreams N
--dream-epochs 1 --probe-every-dream 1`; `--distill-steps`/`--probe-every`
do not apply to a set. (b) Rich-wake integers, pinned to what the §4 pilot
actually ran (its N/coverage numbers are conditioned on this transcript
shape): `--wake-bystanders 3 --wake-nearcone 2 --wake-dialogue 2`.
(c) Deep-arm spellings, resolved: both `--arm b2-fused-deep` and
`--arm b2-fused-detached` exist as their own arm names.] `--init-adapter` the step-800
warm start (`models/mamba2_780m/checkpoints/epoch-2/step-800`, trainable.pt
sha256 `d4bf2e3527befd5f78234a1baf3238624a956f78d3c24f1091d048be9ea08669` —
pulled home; SUPERSEDED by §2.9.3: the session opens by retraining the
warm start with `<|endofconversation|>` and rebuilding anchors, floors,
battery, and caches under the new adapter, whose hash replaces this pin in
the run notes. The 08-08 caches/battery still travel back for the
old-adapter record and the conditional deep block.

**(0) Warm-start retrain** (§2.9.3): rendered corpus with
`<|endofconversation|>` appended per conversation, 800 steps / 2 epochs
(`make warm-start` — the target's default IS this regime as of the gate-close
edit), acceptance check per §2.1's token-level threshold (now also counting
`<|endofconversation|>` emission sanity in a decoded dream); then battery
deletion + rebuild — verbatim, before any cell:
`rm sft/data/knowledge_battery_mamba2_780m.json` (the old-adapter battery
travels to the box via the artifact upload and `load_or_build_battery`
silently reuses it — there is no rebuild flag) — anchor cells (A,
B1-deflated, no-sleep × 3 seeds, old regime) rerun under the new adapter.

**(1) The B4 block — the session's headline (altrup), merged with the
regime bridge. Arms: A, B4-raw, B4-deflated — NOTHING ELSE (§2.10.1;
B4-qcm only on explicit leftover budget).** Per seed (all three): build
the multi-dream cache — N fresh un-spliced dreams from the RICH wake
transcript (§2.10.11), each self-terminating (§2.10.10), steer prefix
per the pilot's §2.10.8 verdict, teacher logits + gated captured queries
+ per-layer spectra + per-dream erasers in the cache; aggregate coverage
gate ≥k dreams per fact; sanity read per the standing rule (decoded
dream + termination-reason counts + repeat counts) before any cell.
Training per §2.10.4's carry matrix: **one pass per dream** (§2.10.2; the
optional ~3-epoch variant cell prices repetition), probes at every dream
boundary (§2.10.5). All three arms are ordinary sequence training —
co-schedulable; the 08-08 solo-tenant rules don't apply to them (A
doubles as the total-denial endpoint of §2.9.6's spectrum — no separate
reset cell exists). Read
everything on the standing frame (ratio), against this session's own
floors (08-08 numbers are old-adapter record only). The block's
registered verdicts: the A-vs-B4 targeting stakes (§2.9.6) and the
raw-vs-deflated aggregate operator question.

**(2) STRUCK (§2.10.1)**: B2tok seeds, the fused-deflated rerun, and
B1-deflated's d12800 rung are all off this docket. B1's rung claim and
the §2.7 carry-vs-once stakes wait for a later dream-set run that
invites B1 back.

**(3) Budget-permitting**: the multi-epoch A/B4 variant cell (§2.10.2),
then the competition probe (below).

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
**Budget semantics, superseded by §2.10.2: ONE pass per dream** — the
hundreds-of-passes regime is retired; the optional multi-epoch variant
cell is the only place repetition appears, priced separately. Arms: per
step (1). The old-regime adoption rule is moot — the retrain (§2.9.3)
severed cross-adapter pairing, so the old regime is simply retired and
this regime stands on its own floors; multi-sleep runs on it.
Old and new regime numbers are never pooled (different dreams, different
adapter).

**Detail for step (3)'s tail item**: the competition
probe — one paired cell, seed 1234, `--n-facts 2` vs the 4-fact
result, same wake-transcript template: if per-fact margins at 2 facts
exceed the 4-fact per-fact margins materially, within-cone competition
gains direct support. (B1-deflated's d12800 rung left the docket with B1
— §2.10.1.)

**(5) Multi-sleep: NOT this session** (§2.4). Next-next session, on the
bridge's winning regime, per 08-07 §3.7's arm list amended by whatever §2.3
consolidation decides.

## 4. Local work before the box (launch gate; TDD on the CPU fake backbone
where testable, but §1.8 stands — hardware-shaped code needs a hardware
smoke, so step (1) opens with one short B4 cell at a token budget before
committing the block)

- **The B4 harness** (the gate's centerpiece): (a) query capture behind
  the state-dependency gate (§2.9.2) during cache build (reuse
  `Model.c_capture` — the capture plumbing `erase_probe.py` already
  uses; the blank-state re-score pass, the divergence threshold plumbing,
  and the binding-scan validation overlay), per-fact counts to
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
  notes at gate close. [PINNED at gate close, from `--help`: `--arm replay`
  (A), `--arm b4-raw`, `--arm b4-deflated`, `--arm b4-qcm`; a dream-SET
  cache accepts exactly these four. Set cache: `--build-dream-cache
  --dreams N` → `data/dream_set_s<seed>.pt` (+ `.pilot.pt` under
  `--pilot-capture`); epochs knob `--dream-epochs` counts passes over the
  set; binding gate `--bind-min-dreams` (default 2); gate threshold
  `--gate-threshold` (nats, pilot-frozen); rank rule `--rank-rule
  {ratio-gap,median}`; rich wake `--wake-bystanders/--wake-nearcone/
  --wake-dialogue` (default 0 — the box docket passes them explicitly);
  probes `--probe-every-dream`.]
- **The local pilot (before any format is frozen): ~20 steered
  free-running dreams** against the seed-1234 wake transcript (extract from
  `dream_cache_s1234.pt` — do not regenerate), 512 tokens, temp 0.7,
  distinct generation seeds, on this machine (`HSA_OVERRIDE_GFX_VERSION`
  set; the manual mixer loop is slow here — overnight is acceptable, local
  time is free). Candidates and criterion per §2.10.8 — {no-prefix,
  `<|eoc|>`, `<|eoc|>[USER]`, `<|eoc|>[USER]␣`, instruction-text
  fallback}, selected on coverage FEASIBILITY, never rehearsal-rate
  maximization; token candidates re-checked after the §3(0) retrain
  (the pilot's old adapter has never seen `<|eoc|>`); the pilot runs on
  the RICH wake shape (§2.10.11 sequencing). Output: N = smallest count
  where every fact is bound in ≥k dreams with ~1.5× headroom, k = 2
  unless the rates argue otherwise (record the reasoning). The pilot additionally reports, per §2.7:
  per-fact **within-dream repeat counts** (B4's re-installation window),
  and the per-layer **σ spectra of the captured queries with both rank
  rules applied** — where the rules disagree, print it; the surviving
  rule is frozen into the spec before the box. **Kill-condition: if
  covering all 4 facts needs
  N > ~16 dreams under the best candidate, the no-splicing regime is
  unaffordable as specced — stop, report, and the bridge is redesigned
  (stronger steering) rather than launched.**
- **Tokenizer + termination plumbing**: `<|endofconversation|>` into the
  model interface's `SPECIAL_TOKENS`, `prepare_data.py` appending it per
  conversation, dream sampling unbanning eos and stopping on
  `<|endofconversation|>` with the `--dream-tokens` max and turn-count
  backstop (§2.9.4) — all TDD-able locally; the retrain itself is box
  work (§3(0); no local training).
- **Mixed packing** (§2.10.9): `prepare_data.py` packs 2–3 conversations
  per example with `<|endofconversation|>` between and at the end; a
  small fraction of boundaries followed by a mechanical same-conversation
  recap exchange instead of an unrelated conversation. Post-`<|eoc|>`
  acceptance clause: after the retrain, a decode conditioned on
  `<|eoc|>` alone must open a fresh well-formed `[USER]` conversation
  (floor set from its first measurement).
- **Rich wake transcript builder** (§2.10.11): facts + bystander items +
  ultrachat slice; the automated fact/battery collision guard (refuses
  to build on overlap); the context-leakage probe class.
- **Gate-pilot instrumentation** (§2.10.7): capture-everything cache
  path for pilot runs + the offline scheme-sweep/scoring tooling
  producing the §2.10.7 table.
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
- **Scrub-state initialization** (run the dream once with the B1-style
  erase hook, no training, use the final state to train from position 0)
  — rejected on two contaminations: (a) time travel — the student's
  start state contains all 512 dream writes including the tokens it is
  then trained to predict, breaking the teacher/student same-history
  pairing; (b) the facts return via writes — the hook ablates reads
  only, so the dream's rehearsal text re-writes the facts into the
  scrubbed state, and with no hook at training time nothing cuts them
  again. The salvage — **scrub-captured queries** (harvest the erased
  pass's read directions, state discarded) — is PARKED, not registered:
  try it only if the state-dependency-gated teacher capture measurably
  misses (altrup lukewarm).
- **Fact-aware eraser construction** (probe-forced fact questions;
  binding-scan position gating of the capture) — rejected as mechanism
  by §2.9.1's prod-validity constraint; retained as validation overlays
  only.
- **A separate "reset-control" cell** — struck (§2.9.6): the blank-state
  student is not a new arm, it is arm A as implemented
  (`distill_replay`, fresh state, `dream_sleep.py:680`). The targeting
  question is answered by the existing A-vs-B4 pairing.
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
8. **Lucid / on-policy dream distillation** (named this debrief): the
   student (erased start, live weights) samples the dream token by token;
   the intact frozen teacher runs in parallel on the same stream and
   supplies per-position logits; KL trains the student in the states it
   actually visits, including post-error states — the standard cure for
   the train-on-teacher/deploy-on-self mismatch, and the closest
   construction to "the dreamer steers the dream while the memory system
   annotates it". Registered constraints: full student authorship without
   rehearsal seeding reruns generate-while-draining (1/4 coverage, floor
   — 08-06), so the viable form is teacher-seeded cue islands with
   student-authored continuation; every pass produces a fresh dream, so
   the shared-dream pairing across arms does not exist for this arm (its
   comparisons are within-arm or vs-its-own-floor); per-token generation
   economics (two forwards + backward per token, no cached teacher
   logits). Future arm, not next-run work.
9. Carried unchanged: warm-start's effect on dream binding (multi-sleep
   measures it), value-side erase, deflation-k under multi-cluster states,
   soft dreaming.

## 7. Housekeeping

- Run notes banked as `6abf671` before discussion (flow rule).
- **Fourth sitting (§2.10, same day)**: probe-curve extraction from the
  pulled jsonls (§1.9), the arm reduction, the passes retirement, per-
  dream erasers, the goal block, the gate-pilot design, mixed packing,
  rich wake, and the A-mechanics correction (§2.10.12) — recorded after
  the third sitting's commit was deliberately reverted so the file
  lands as one unit. The next session (local gate work, then box) is
  expected to run from THIS FILE COLD — §2.10 + §3 + §4 are the spec,
  §2.7–2.9 the rationale; nothing load-bearing lives only in the
  conversation. The re-dry-run before launch is still required and now
  covers all of it.
- The `altrux-debrief` command gains a standing rule from this session
  (altrup): a DISCUSSION file must register the GOAL — what is being
  optimized for, what the mechanism is FOR — with the same rigor as the
  mechanism details. [implemented this debrief]
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
