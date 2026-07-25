# Discussion notes — 2026-07-24: training-data structure and what consolidation actually requires

Team discussion (altrup + Claude) following `EXPERIMENT_NOTES-20260724-014851.md`.

**Deliberately contains no next-run plan.** The discussion reached conclusions
that change the framing more than they settle a recipe, and several of them
call the current program's premises into question. Writing a box plan on top of
that would be premature. This records where the thinking landed.

Evidence for everything factual below is in
`RESEARCH-20260724-training-data-structure.md` (our own data measured, and the
comparable literature read in full). Read that first; this note is the
interpretation, not the evidence.

## 1. Reinterpretation of the 780M screen

1. **The screen's data was structurally broken, and the verdict rests on it.**
   9.0% of role transitions in generated chains are malformed — `[USER]` turns
   with no answer, `[ASSISTANT]` turns answering a question with nothing
   addressing it. At full scale that is 5,498 splices, of which 1,580 cut
   *multi-turn conversations* mid-dialogue. Any comparison between BX0/BX1/BX2
   was a comparison between arms trained on that.
2. **Candidate mechanism for the recency result — a hypothesis, not a
   finding.** A spliced tail is unaddressed: nothing in the sequence says which
   question the trailing `[ASSISTANT]` answers. Under that ambiguity the
   loss-minimising policy is to bet on recent context, since in the
   non-spliced majority the recent turn *is* the correct referent. We may have
   trained the recency bias by making retention un-addressable, then measured
   that the model does not retain. Falsifying this requires re-running the
   screen on repaired data.
3. **The mix-vs-state conclusion should be held loosely.** BX1's gain
   decomposes as +0.0036 recency / +0.0001 long-range at 12.2M — i.e. M's
   entire measurable contribution was recency-weighted. BX0 (state@32) had the
   only rising long-range term (−0.0007 → +0.0055, ~2.6σ) and was stopped at
   6.1M while mix ran to 12.2M. We picked a winner on an unequal budget, on
   broken data, with the arm showing the desired signal cut early.
4. **The gist-delta ruler is weaker than we treated it.** A "significant"
   +0.004 mean-logprob delta is compatible with the model being unable to
   recall a single name. Independently, `2607.00368` documents NLL improving at
   every model scale while free-form recall stays at 0.0%.

## 2. Existing issues with the data

Measured; see the research note for method and decoded evidence.

1. **94% of training tokens are documents.** The conversational half is 6% and
   capped at 1024 tokens.
2. **There is no long multi-turn conversational data at all.** LongAlign and
   babilong are 100% two-turn (one large user turn, one answer). The only
   multi-turn source is ultrachat, capped. Long implies single-turn; multi-turn
   implies short. The `Makefile`'s description of LongAlign-10k as "real long
   multi-turn conversations" is incorrect.
3. **The splices are malformed** (§1.1). Both malformations — dangling user
   turn and unaddressed answer — come from the same cut at an `[ASSISTANT]`
   marker; cutting at a `[USER]` marker yields two well-formed halves.
4. **Retention-bearing tokens are ~1% of the corpus and compete with ~99%
   local-context loss.** The comparable paper (`2605.26099`) has a similar
   *fraction* but it is their entire objective — carrier tokens get zero
   gradient. Our ratio is inverted relative to the only prior art that runs
   this setup.
5. **Mid-conversation sleeps have been a no-op in every dataset ever trained
   on.** Two independent causes: the 1024 cap (below `--mid-sleep-min-len`),
   and locally the absent role markers.
6. **Role markers are encoded inconsistently across corpora**, and the special
   tokens' embedding rows are randomly initialised *and* frozen
   (`TARGET_LORA_MODULES` excludes embeddings; `tie_embeddings` makes `lm_head`
   share them). The model reads the markers as random vectors and cannot learn
   to emit them. The three-token BPE spelling carries better representations
   than the single special token.
7. **Counts cannot catch a structural bug.** Every count in every regen log
   was correct and reproducible while the data was malformed. Now a rule in the
   root `CLAUDE.md` and a validation pass in `prepare_chains.py`.

## 3. Potential new data methods

Recorded as candidates with their known problems, not as a chosen recipe.

### 3.1 Cram, don't wipe

The sleep-wipe is an **attribution device, not a mechanism**: it manufactures
the beyond-SSM regime cheaply, substituting a discontinuity for length. Its
cost is that the zeroed state is OOD — our own dream verdict is already flagged
as possibly "about the wipe, not about M."

Saturating the SSM with interfering content instead keeps it in a normal
trained regime, gives a graded rather than binary control, matches deployment
(where nothing is ever zeroed), and — the decisive practical point — **fits a
small token budget**, because interference defeats a fixed-size state at far
shorter lengths than distance does. Cost: attribution becomes graded, requiring
an ablation at each interference level rather than one binary contrast.

### 3.2 Masked-entity construction (MemTrain-style), with its shape corrected

MemTrain masks **all** occurrences of the target entity document-wide, so the
answer string never appears in the input. That makes its end-to-end task
**evidence aggregation, not recall.** Its *auxiliary* task — Intermediate
Memory Recall — is the one that tests carry-across-a-reset: the model builds
memory from an unmasked chunk, then is shown that chunk again with a hole
punched in it, so the surrounding sentence is a retrieval cue while the answer
can only come from memory. Their ablation puts IMR at **+7.03**, and their
decoupled variant (IMR not feeding back into the memory-writing objective)
*degrades* at long context — the recall signal has to shape the write.

The two tasks map onto our two north-star abilities: aggregation → **#1 gist
persistence**; IMR → **#2 specific-fact recall**, which has been at zero since
435. If adopted, their emphasis should be inverted (IMR primary).

Mechanically it is causal-safe because the placeholder sits in the *input* and
the answer is **generated as a later turn** — loss lands on generated answer
tokens, which is what `recall_masks` already marks. It is not next-token
prediction at the mask position.

Transferable: the data construction (spaCy NER, retriever, shuffle, string
replace) and the 1 : 29 : 120 pivot / hard-negative / random-padding ratio.
**Not** transferable: the training method. MemTrain is GRPO RL on a frozen
instruct model whose "memory" is a ≤1024-token natural-language string
re-pasted into the next prompt — no fast weights, no token-level
cross-entropy, and a 64× rollout multiplier that exists only to estimate GRPO
advantages.

### 3.3 A two-sided solvability filter

Generating recall items without filtering produces items answerable from local
context, which inflate every downstream number. MemTrain has no such filter
(spaCy NER plus random choice). Two cheap ablation checks define a well-posed
item:

- **Well-posed:** an M-ablated model *given* the source chunk can answer it. If
  not, the item is ambiguous.
- **Memory-requiring:** an M-ablated model *without* the source chunk cannot.
  If it can, the answer was inferable from elsewhere.

Only items passing both are worth training on.

### 3.4 Inverting the loss shape

`2605.26099` supervises only post-boundary tokens; consolidation-phase chunks
get N recurrent passes and **zero gradient**. Adopting that would put all loss
on recall targets and none on carrier text.

Risk, and the reason this stays a candidate: it was demonstrated training from
scratch on procedurally-generated tasks with tiny vocabularies. Withholding
gradient from ~99% of natural-language tokens while LoRA-tuning a *pretrained*
backbone is not the same experiment, and no paper read here addresses it.
Whether the right form is a hard 0/1 mask or a softer post-recall reweighting
is unresolved.

### 3.5 Repairing the splices rather than removing them

- Documents: synthesize an addressing user turn in front of the resumed answer
  (*"Going back to the document I sent earlier — <question>"*), which converts
  an illegal assistant-boundary cut into a legal user-boundary one **and**
  supplies M with a retrieval key it currently lacks. Question text is
  extractable mechanically — `prepare_babilong.py:52` puts it after the final
  newline. Fail closed when no question is locatable.
- Conversations: never cut. Their retention signal is the mid-conversation
  sleep in place, which needs no addressing and cannot malform.
- `2605.26099` places the **question before the context** so the model can
  consolidate selectively; our babilong prep does the reverse.

### 3.6 Sleep placement by suspension, not cadence

Sleeps belong where an episode is *suspended* incomplete; a natural EOS ending
is self-marking. The suspension sleep does double duty — it marks the
suspension and guarantees the suspended content is out of the SSM when
resumed — so suspension-marking is exactly sufficient for the retention signal
without a blanket policy. Sleeping at *every* join additionally removes an
SSM-leakage channel that would otherwise be misattributed to M; its cost is
losing the un-slept topic-change case, which is partly covered for free because
uncapped ultrachat conversations shift topic internally.

### 3.7 A distance curriculum

BABILong's RMT/ARMT baselines train a curriculum over segment count (1, 2, 4,
6, 8, 16, 32) with the count **randomised per batch** to avoid overfitting a
context size. We sample chain budget log-uniform from step zero with no
curriculum.

### 3.8 Plain language modelling, Titans-style, as a control

Training on FineWeb-Edu at 4K with no engineered structure is the only setup
with a published good outcome, and it removes every confound at once. The
objection is our own recorded finding (`prepare_interference.py:6`): the
backbone's SSM alone handles the recall load our data presents, so *"the
gradient has no reason to use the Titans neural memory and learns to suppress
it instead."* Titans trained **from scratch**, so M was load-bearing from token
one; a bolt-on M on a pretrained backbone at 4K has no pressure to be used.

The variant that keeps the simplicity while creating pressure is plain LM at
sequence lengths the SSM cannot hold (32K+), which LongAlign supplies.

### 3.9 Corpus roles

- **LongAlign: keep, as background carrier** — our PG19 — not as
  conversational data. It is 214M tokens, so a properly-sized run would exhaust
  it; FineWeb-Edu is the unbounded alternative and is what Titans used.
- **MSC: eval and a small seed, not the spine.** It has genuine human-written
  implicit multi-session structure, which per `2606.27472` cannot be
  synthesized without saturating. But it is short chitchat, its remembered
  content is thin persona facts, and **its own baselines do not train
  parametric memory** — prior sessions arrive as truncated context or a gold
  summary.
- **Procedural construction over abundant carrier text is the only thing that
  scales**, and abundance is the binding constraint: Titans' 760M saw 30B
  tokens; our best arm saw 12.2M.

## 4. Conclusions that are not about data

These changed the framing more than the recipe.

1. **M and the SSM state are the same size.** At 780M the SSM state is
   48 × 64 × 128 × 48 = **18.9M values**; M's fast weights are
   6144×1536 + 1536×6144 = **18.9M values**. So M is not a capacity upgrade
   over Mamba2's own state — which is itself a linear associative test-time
   memory via State Space Duality. M's only possible advantages are
   **nonlinearity** (2-layer MLP vs linear map) and **selectivity**
   (surprise-gated gradient write vs fixed per-token rule). Neither has ever
   been trained for, and the measured recency-only contribution suggests
   neither is currently delivered. This sharpens the concern that M is
   functioning as a redundant second SSM.
2. **Sleep means two different things and conflating them is a live risk.**
   Memory-training sleep wipes the backbone so M is the only surviving channel
   — an attribution device. Consolidation sleep pauses the stream to run
   offline passes and commit weights, and should **not** wipe the backbone:
   there is nothing to attribute, and generating over a zeroed state is the
   OOD condition that plausibly killed the dream result. Same word, opposite
   treatment of the SSM. The B0 dream verdict was measured under the wrong
   phase's semantics and should be reopened rather than carried forward as
   negative.
3. **For the continual-learning plan, M may only need to be a short-term
   buffer.** If consolidation moves content into weights at each sleep, M needs
   to hold from wake to sleep, not across sleeps — long-term durability is the
   weights' job. That makes the requirement *interference capacity within a
   session*, not cross-sleep retention, which is both easier and cheaper to
   measure. It assumes consolidation works (we have one confounded negative and
   no implementation), and it picks a side in the neuroscience split recorded
   in `RESEARCH-20260722` (standard systems consolidation, buffer) over
   Multiple Trace Theory (permanent episodic store). Worth recording as a
   choice rather than an assumption.
4. **Deferred consolidation cannot avoid examples.** New input→output behaviour
   cannot be installed in existing weights without a specification of which
   inputs map to which outputs, and that specification *is* data — text,
   activations, or (k,v) pairs. Specifically:
   - M is an MLP fitted to (k,v) pairs, **not a table of them**. The pairs are
     the training signal and are not recoverable. M can only be *queried*, and
     queries need keys, which come from residuals of content no longer held.
     Searching for keys does not rescue this: the key space is 1536-dimensional,
     and off-manifold keys are meaningless while the realizable manifold is only
     known from data.
   - Closed-form rank-one weight editing (`W' = W + (v − Wk)kᵀ/kᵀk`, the
     ROME/MEMIT mechanism) genuinely changes base weights with no optimizer —
     but it consumes k and v, so the point above blocks it.
   - Folding a frozen SSM state also fails: the term `x_tᵀ(W_Cᵀh*)` is linear in
     the block input, but Mamba2 gates between the scan output and `out_proj`,
     so it cannot be absorbed into an existing matrix and requires a new linear
     path.
   - Distilling M into a fresh low-rank expert is **not** consolidation into
     base parameters. It relocates the store from fast weights to slow weights.
5. **The escape from (4) is to not defer.** Consolidating while the content is
   still in context makes the live stream the examples, with nothing stored or
   generated. That is what `mamba2_780m_continuous_learning` is designed around
   — continuous gradient accumulation, applied when a critic signals it is
   warranted. It also makes M's role in that design unclear, since the thing
   carrying information between arrival and commit is the accumulated gradient.
6. **Whether M is on the critical path is untested.** `RESEARCH-20260722`
   already specced the deciding comparison: M-direct-read vs
   consolidate-from-transcript vs consolidate-from-M-replay, with its own
   prediction that transcript beats M-direct-read. If M-replay ≈
   transcript-replay, M's justification narrows to compression and
   interference-free replay under storage or privacy constraints — not fidelity.
7. **Our probe gate is bridge-tier.** Teacher-forced logprob measures whether
   probability mass moves toward an answer, not whether the model would produce
   it. `2607.00368` separates proxy / bridge / target-behavior and finds nothing
   in 24 audited papers reaching the behavioral tier. A genuine behavioral
   readout needs generation, which window-1 serving degenerates on. Until that
   is resolved, claims should be labelled bridge-tier explicitly.
8. **Titans is not the precedent we assumed.** It contains no data recipe (its
   packing and reset behaviour are absent from both it and Gated DeltaNet), it
   constructs no data requiring survival across a reset, and its 2M-context
   claim appears only in the abstract, introduction, and conclusion, attached
   to no experiment. Its longest documented evaluation is 16K.

## 5. Explicitly considered and rejected

- **A special boundary token at episode joins.** No such token exists at
  serving time, so training one teaches reliance on a crutch that will not be
  there, and it would still not distinguish a genuine topic change from a
  continuation.
- **`--max-wake 1` as the mechanism for boundary marking.** Superseded by the
  suspension reasoning in §3.6, which reaches every-join-sleeps for a better
  reason and leaves a deliberate un-slept fraction available.
- **Token-denominated log-uniform gaps.** Superseded by the segment-count
  curriculum (§3.7), which is what the precedent actually does.
- **Restricting cuts to `[USER]` markers only.** Syntactically legal but
  semantically insufficient: if the preceding assistant turn ended in a
  question, the tail opens with a user turn answering something invisible, and
  that is undetectable syntactically.
- **MSC as the conversational spine** (§3.9).
- **Templated fact/correction generation.** `2606.27472` built exactly this and
  found frontier models score 100% by last-mention scan — *"synthetic templated
  supersession is therefore saturated and cannot surface the failure."*
  `probe_correction.py`'s own phrasing is an instance of the saturated form.
- **Adopting Titans' data recipe.** There isn't one (§4.8).
- **A prefix-distance sweep as a priority.** Titans' longest documented eval is
  16K and `2605.26099`'s accuracy evals top out near 3.3K; our 6,144-token
  probe is mid-range for this literature. Distance is not the binding
  constraint.
- **Distillation into a fresh expert, described as consolidation.** It moves
  the store rather than changing base parameters (§4.4).
- **Reading (k,v) pairs out of M for closed-form editing.** M does not contain
  them (§4.4).
- **A one-sided solvability filter** (reject only items answerable without
  memory). It keeps ambiguous items; §3.3 needs both directions.

## 6. Measurement gaps

Stated as gaps in what we know, not as a run plan.

1. **The SSM's interference capacity** — how many competing items before the
   backbone alone fails. This number sizes every training example under §3.1,
   and below it every training token teaches that M is unnecessary. The
   instrument exists (`probe_recall.py --n-facts`, built for exactly this) and
   fell out of use when we moved to the gist eval.
2. **The SSM's forgetting horizon** — at what distance an early fact is gone
   from the backbone alone.
3. **Whether the correction probe's cross-scenario diagonal is positive.** The
   contrast is already computed (all four names scored under both scenarios,
   differing only in what was written to M) but is not surfaced as a contrast
   in the printout, and no trustworthy run of it exists.
4. **Whether `o_t` carries content at all.** `model.py:92-94` records a prior
   observation that `o_t` was *tiny* while beta sat open — "the gate was open
   but there was nothing substantial to inject." If that still holds, the
   binding problem is write fidelity, and read-path and data work are both
   premature.
5. **Whether M's contribution survives a locality control.** `2607.00368` saw
   recall bought at the cost of locality collapsing 141/144 → 14/144. We have
   never checked whether our memory damages unrelated knowledge.

## 7. Housekeeping done this session

- `c1142aa` — root `CLAUDE.md`: data generators must print structural
  invariants and decoded samples, not only counts.
- `5a631ee` — `prepare_chains.py` validation pass: malformed role adjacencies,
  silent-join counts, max concurrent suspended episodes, decoded samples per
  event kind.
- `c71b47d` — `probe_correction.py --no-filler`, separating "never written"
  from "written then overwritten".
- `addfd5f` — `RESEARCH-20260724-training-data-structure.md`.

Outstanding small items, unscheduled: the `Makefile`'s incorrect LongAlign
description, a marker-consistency assertion in `validate()`, and the
`extend_embeddings` initialization and trainability fix (§2.6) — the last of
which breaks checkpoint continuity with every BX checkpoint.
