# Discussion notes — 2026-07-24 (later session): the next-run plan

Team discussion (altrup + Claude), continuing
`DISCUSSION-20260724-data-structure-and-consolidation.md`. That note
deliberately contained no plan; this one commits to one. Owner's calibration,
recorded verbatim in spirit: not 100% sure this works, happy to try it. The
plan below is committed *unless a step-1 diagnostic falsifies its premise* —
two of them can veto pieces of it (see §6).

## 1. Decisions

### 1.1 Model and trainables

- `mamba2_780m_memory_mix` (read layer 16), **fresh LoRA from base** — no BX
  warm start. The embedding fix invalidates BX adapters (tuned against frozen
  random marker rows), the old data plausibly trained the §1.2 recency bias
  in, and a warm start would make any persisting bias uninterpretable
  (baked-in vs re-learned). Sunk cost is ~12M LoRA tokens ≈ hours.
- Trainable: rank-16/alpha-32 LoRA on `in_proj`/`out_proj`, memory subsystem
  slow weights, **plus the two `[USER]`/`[ASSISTANT]` embedding rows** — init
  from the mean of their BPE-spelling embeddings, exempted from the freeze.
  Tied `lm_head` rows follow, which is what lets the model learn to *emit*
  markers — a hard prerequisite for CL sleep mode (dreams generate `[USER]`).
  The rest of the embedding table stays frozen. Recurrence params
  (`A_log`/`dt_bias`/`conv1d`/`D`) stay frozen — standard for Mamba PEFT,
  and largely compensable through the adapted `in_proj`'s Δt/B/C.

### 1.2 Code fixes before data generation

1. One canonical special-token encoding of the markers across all corpora,
   plus a marker-consistency assertion in `prepare_chains.py`'s `validate()`.
2. `extend_embeddings`: meaningful init + trainable marker rows (§1.1).
3. The Makefile's incorrect "real long multi-turn conversations" LongAlign
   description.

### 1.3 Data — four components by token share

**35% Wikipedia IMR cram blocks** (MemTrain-style, shape corrected):

- Alternating turns. `[USER]` = a few fresh Wikipedia passages plus, at a
  *varied* position in the turn (not always last), one earlier passage's
  sentence re-shown with its NER entity blanked. `[ASSISTANT]` = that
  sentence **completed** — Wikipedia's own text, entity restored. Zero
  invented phrasing anywhere; the blank marker is the only non-dataset token.
  The recall items double as the assistant turns, so every user turn is
  legally answered with no filler.
  - Bare-entity answers were considered and rejected: ×16-weighted one-word
    assistant turns would train one-word assistant style. The completed
    sentence keeps answers natural; the copied span is visible in context so
    it earns no fake recall credit — weighting and filter key on the entity
    span only.
- **Entity substitution** (same-type swap, MemTrain's "shuffle/string
  replace" step): the answer must not exist in pretrained weights, only in
  the passage. Watch the locality control for factuality cost.
- **Gap curriculum**: per-item random gap under a **progressive ceiling** —
  ceiling starts inside the BPTT window and grows through training,
  randomized beneath it (how RMT/BABILong actually trained; supersedes both
  pure log-uniform-from-step-zero and per-batch-only randomization).
  Per-item randomization also defeats the skim shortcut: at read time the
  model cannot tell which passage will be probed soon, so no learnable
  signal supports a skim-the-passages policy, and within-window items punish
  it where credit reaches.
- **Two-sided solvability filter**, every item: (A) backbone given
  source+cue must answer (well-posed); (B) backbone given the item's *actual
  interference stream* + cue, source absent, must fail (memory-required).
  Teacher-forced scoring with the plain backbone (which *is* the M-ablated
  model), test B amortized by caching SSM state at each cloze position.
  ~one forward pass over the cram corpus ≈ order 10% of run compute, offline,
  cached. Log the discard rate. Plus an explicit string/alias check that
  retriever hard negatives don't contain the target entity.
- Held-out **articles** (not just items) reserved for eval — the Wikipedia
  analog of the existing vocab-slice discipline.

**15% babilong-style needle items** — the RMT-proven form, under the
identical curriculum, filter, and weighting. Minority slice: bAbI's tiny
template space trains format-matching as much as memory. Kept because its
failure modes are disjoint from IMR's (synthetic facts cannot leak from
pretraining; IMR is unreplicated). Graded probes attribute which slice moved
recall, and the next run doubles down on that one.

**35% repaired conversational chains** — deployment shape, not retention
pressure:

- ultrachat regenerated uncapped (the 1024 cap lives in the old `train.pt`
  artifact; current `--max-len` default is 32768 — pass it explicitly in the
  Makefile anyway). Conversations never cut.
- LongAlign/babilong split-QA: the dataset's own question moved **verbatim**
  to the tail — head = document with question mechanically removed, tail =
  `[USER] <question> [ASSISTANT] <answer>`. Fail closed when the question
  isn't extractable. This supersedes the synthesized "Going back to the
  document…" addressing turn from §3.5 of the prior note: templated phrasing
  is the saturated form 2606.27472 warns about. Open question carried: the
  synthesized turn was also a disambiguating retrieval key; a bare question
  after interference may be ambiguous. The filter drops ambiguous items —
  measure the drop rate before declaring this settled.
- No question preview before documents (2605.26099's ordering rejected):
  MemTrain's IMR also has no preview, and not knowing what will be asked is
  the deployment-shaped skill. A/B candidate someday, not the default.
- **~10% of chains keep wipe-sleeps purely as attribution probes** —
  including mid-conversation ones. The other 90% carry **no engineered
  mid-conversation retention signal**: a wipe-less version would need
  suspend-and-resume of conversations, and ultrachat turns don't survive
  that ("what about the second one?" resumed after three episodes is exactly
  the unaddressed-referent ambiguity that plausibly trained the recency
  bias). One component, one job — retention pressure lives in the cram
  slices.
- (For the record: mid-conversation sleeps were 0 in every dataset ever
  trained because the eligibility conditions were mutually exclusive —
  everything ≥4096 tokens was two-turn with its only boundary outside the
  middle third; everything with middle-third boundaries was capped at 1024.)

**15% plain long documents** — LM ballast; protects general quality; feeds
the locality control.

### 1.4 Loss

- Weight 1.0 on **all** tokens — we do train on carrier/cram text. The
  hard zero-on-carrier mask (2605.26099) stays rejected as untested on a
  pretrained backbone (§3.4 of the prior note).
- Recall targets (entity spans, split-QA answers — what `recall_masks`
  marks): **×16**, which at ~1% token share puts retention at roughly 15% of
  gradient mass. **Ramped 1→16 across the beta-anneal window** so the two
  pressures don't peak together (see §2.2).
- eos-weight 32 and other locked BX constants unchanged.

### 1.5 Training config

- Chains + ballast: the locked BX constants (batch 24, chunk-len 48,
  memory-window 8, accum-tokens 1536).
- **Cram blocks only: chunk-len 512 + gradient checkpointing + reduced
  batch.** Rationale in §2.1. VRAM/tok-s measured on the rented card before
  locking; the BPTT tax is paid only where the credit reach is needed.
- Beta anneal scheduled to open during the dense short-gap cram phase
  (§2.2).

## 2. Findings this session (new since the morning notes)

### 2.1 The trainer detaches state every chunk — recall loss cannot credit writes

Verified in code: `sft/train.py:744` detaches state at each chunk boundary;
the runs used chunk-len 48 (BX constants — not the default 16). So a recall
loss thousands of tokens after a write is **causally disconnected from the
write**: M's write path learns only from its self-supervised surprise rule
plus write→read pairs inside one 48-token window. Sparse recall supervision
isn't just sparse — under truncated BPTT it never reaches the weights that
write.

Context from precedent: MemTrain's 64× GRPO rollouts are *their* solution to
credit across gaps BPTT can't span; Titans/2605.26099 backprop through their
full (short) sequences; RMT fine-tuned with full BPTT across segments. Our
fix is the selective chunk-512 window plus a curriculum whose early gaps fit
inside it.

Flagged as hypothesis, not finding: "this is why `o_t` is empty" (§6.4 of
the prior note) is plausible but unverified. Cheap test: does M's retrieved
content correlate with write→read distance ≤ chunk length?

### 2.2 The loss-8 anneal window is where memory suppression gets decided

Owner observation: with the injection zero-initialized (`o_proj` weight and
bias — verified, exact no-op at init), loss still climbs to a consistent ~8
as the beta anneal opens, then recovers. Interpretation: unshaped reads
pollute the residual stream; the gradient's cheapest exit is suppressing the
memory path (beta/o_proj → 0). **Whatever data is in front of the model
during the anneal decides whether M becomes useful or becomes zeroed.**
Hence: anneal aligned with the dense short-gap cram phase, and the ×16
recall multiplier ramped so it doesn't amplify the spike.

Also for the record: loss ~8 baseline causes, mostly on the fix list —
marker tokens as frozen random rows cost ~10 nats each (fixed by §1.1);
malformed splice tails (fixed by repair); eos-weight 32 inflates the printed
number; post-wipe tokens are high-loss *by design* (the retention signal).

### 2.3 Precedent survey: bolt-on memories come in two families; we're in the empty quadrant

- Family 1, most published successes: **exact/retrieval stores on frozen
  backbones** (Memorizing Transformers, LongMem, CAMELoT, RETRO-fitting,
  Larimar). Storage is exact by construction; only *reading* is learned.
  Common trick: zero-init/gated insertion (Flamingo gates; our `o_proj`
  zero-init is the same move).
- Family 2: **learned fast-weight memories** (Titans, TTT, ATLAS,
  Hope/Nested-Learning) — essentially all trained from scratch, memory
  load-bearing from token one.
- Closest true precedent: **RMT/ARMT** — memory bolted onto pretrained
  GPT-2/BERT and fine-tuned, successful on BABILong. Its injection is *no
  injection*: memory is extra token embeddings (prepended read positions,
  appended write positions, outputs carried to the next segment), so the
  pretrained attention machinery already knows how to read/write it, and
  fine-tuning only teaches content. What it needed: segment curriculum +
  full BPTT across segments — independently the two load-bearing parts of
  this plan.
- Nobody trains a bolt-on fast-weight memory over a pretrained backbone on
  natural language. We are attempting the empty quadrant, knowingly. RMT's
  contrast also sharpens our biggest structural disadvantage: the backbone
  has never learned to interpret M's output, which RMT never had to solve —
  this elevates the §6.4 read-content diagnostic.

### 2.4 MemTrain provenance check

arXiv 2606.03197, Peking University + Samsung Research, submitted
2026-06-02. Seven weeks old at time of writing; no citations, follow-ups, or
reproductions found (uninformative at this age). Not an industry standard —
no industry standard exists for memory-training data. Trust posture: we
depend on a *property* (answer absent locally, present in memory) verified
per-item by our own filter, not on their results. Taken from the paper: the
task shape and the 1:29:120 starting ratio. Taken on faith, gated by our own
probes: that the task shape teaches transferable memory use.

### 2.5 Why the loss can't be minimized by zeroing M (the design's four mechanisms)

1. Filtered items are verified unanswerable without memory — suppression
   leaves ~15% of loss mass unreducible.
2. Short-gap items sit inside the BPTT window — recall loss credits the
   writes end-to-end; per-item gap randomization forces the write policy to
   be uniform (no skim shortcut).
3. The anneal opens exactly when M is profitable (§2.2).
4. Pre-flight diagnostics (§6) ensure the write path is alive before the
   budget is spent.

## 3. Continual-learning design (settled this session, runs after the training above)

Two modes on the trained model:

- **Wake**: normal assistant generation. Mode switching: fixed schedule (or
  sleep-on-suspension) for all testing; a surprise/critic threshold is a
  later refinement, deliberately unsolved now.
- **Sleep** (consolidation semantics — SSM *not* wiped, per §4.2 of the
  prior note): chunk-wise dreaming with no EOS stop, memory **writes
  disabled**; per chunk: generate with M intact storing logits → apply
  **decay-on-read erase** → one batched teacher-forced pass with the eroded
  M → KL(stored ‖ new) into the LoRA.
  - **Decay-on-read, not reverse gradient**: erase = descend ‖M(q)‖² at the
    read queries — the same single-input gradient-step operator as a write
    with the target swapped from v to 0, so exactly as targeted as writes
    are. Gradient *ascent* on the write loss was considered and rejected:
    it drives M(q) to anti-v (a confident wrong answer, not an absent one)
    and is unbounded, bleeding into other queries through the shared MLP.
  - **KL on stored logits, not CE on sampled tokens**: the sampled token
    still drives the dream forward; the full pre-erase distribution is the
    training target. One-hot CE on own samples is weak signal + drift risk.
  - Erase-on-read + writes-off means a long sleep empties M — consistent
    with the §4.3 buffer position (M holds wake→sleep; weights hold
    long-term). Unread content survives to the next wake; degrades
    gracefully.
  - **Log read coverage** during sleep (even crudely via read-gate
    activity): a sleep that consolidated nothing must be distinguishable
    from one that worked.
- **Transcript-consolidation control arm** (same distill loss, wake
  transcript as source): the prior note's §4.6 comparison, with its recorded
  prediction that transcript beats M-replay. M-replay is the arm under test,
  not the default.

## 4. Considered and rejected this session

- **Warm-starting from any BX checkpoint** (§1.1 — three independent
  reasons).
- **Unfreezing the whole embedding table** — only the two marker rows;
  pretrained rows aren't the problem.
- **Question preview before documents** — trains selective consolidation;
  deployment needs indiscriminate retention; MemTrain-IMR-consistent.
- **Synthesized addressing turns** ("Going back to the document…") —
  templated; superseded by moving the dataset's own question (open
  ambiguity/drop-rate question carried, §1.3).
- **Bare-entity assistant answers** — style contamination under ×16.
- **Raising chunk-len globally** — batch bought essentially all of the
  measured 2.3× throughput; BPTT tax paid only on cram blocks.
- **Reverse gradient descent as forgetting** (§3 — anti-content, unbounded).
- **CE on sampled dream tokens** (§3 — KL on stored logits instead).
- **Wipes as the primary retention mechanism** — kept only as the ~10%
  attribution fraction; cram is the mechanism (matches deployment, matches
  the §4.3 buffer requirement, and the wipe-trained result was already
  under suspicion).
- **A pure-repair two-step** (repair-only run for a clean §1.2 falsification,
  then study-grade data) — the combined run was chosen: the goal is a
  working memory to build CL on, §2.4 of the prior note already predicts a
  repair-only run under-pressures M, and attribution inside the run comes
  from probes (M-ablated contrasts, graded interference, the wipe fraction,
  per-slice comparison) rather than from data minimalism. Cost accepted: a
  persisting recency-only result on this run won't cleanly settle §1.2's
  hypothesis.
- **New corpora beyond Wikipedia** (FineWeb-Edu as carrier) — abundance
  argument only binds at ~20×+ our token budget; a scale decision, not a
  quality one. MSC stays eval-only per the prior note.

## 5. Efficiency notes

- The filter is forward-only ≈ 10% of run compute, once, offline (§1.3).
- MemTrain's *training method* (GRPO, 64× rollouts) remains non-transferable
  and is far less compute-efficient than our token-level CE; their rollout
  multiplier exists to solve the credit problem we solve with chunk-512.
- Raising `--memory-window` would *increase* throughput (fewer write ops)
  but coarsen writes — distinct knob from chunk-len; not part of this plan.
- Sparse recall supervision does not reduce FLOPs/token; what matters is
  signal-per-GPU-hour, which the ×16 weighting and credit fix raise while
  tok/s falls somewhat on the cram fraction.

## 6. Gates and open risks

1. **Keystone**: can SSM-defeating interference fit in 512 tokens? The §6.1
   capacity probe answers it; if not, within-window items can't be
   memory-required and the credit mechanism needs rework. Run first, this
   box, this week.
2. **§6.4 `o_t` check** — is the write path alive at all? Plus the new
   distance-correlation variant (§2.1).
3. Filter discard rate on Wikipedia (pretraining leakage) — entity
   substitution should hold it down; measure.
4. Split-QA ambiguity drop rate after removing addressing turns (§1.3).
5. Chunk-512 VRAM/tok-s on the rented card — measured before constants lock.
6. Held-out-article discipline for all Wikipedia-derived eval items.
7. Locality control (§6.5 of the prior note) — recall must not be bought
   with general-knowledge collapse, especially under entity substitution.
8. **NaN watch on the box run** — local diagnostics
   (`../research/RESEARCH-20260724-local-diagnostics.md` §3) found fresh-random-M under
   the trained write knobs goes non-finite at inference in ~1/3 of
   fresh-state passes on the BX1 checkpoint. Every training sequence starts
   from exactly that combination. First hours of the box run: watch
   `GRAD_NORM` and `[nonfinite-write]` in the logs, and treat any
   non-finite event as a stop-and-diagnose, not noise. If the new run
   changes eta/theta/alpha dynamics, re-check before committing the full
   budget.

## 7. Housekeeping

- This note. Code work (fixes, generators, filter, trainer changes) starts
  next session, in the order of §1.2 → diagnostics → generators.
