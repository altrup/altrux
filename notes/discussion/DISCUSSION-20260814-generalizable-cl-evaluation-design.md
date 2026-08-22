# Discussion notes — 2026-08-14: generalizable continual-learning evaluation

Decision record after the broad continual-learning review in
[`RESEARCH-20260814-continual-learning-foundations.md`](../research/RESEARCH-20260814-continual-learning-foundations.md).
The research note records what the field establishes. This discussion records
our derived experiment direction.

The primary objective is evidence that the mechanism generalizes across
knowledge types, language forms, task orders, and model configurations. Code
binding alone is not the target claim.

This direction supersedes the next science launch in
[`DISCUSSION-20260811-multisleep-adaptive-wake-and-agent-migration.md`](DISCUSSION-20260811-multisleep-adaptive-wake-and-agent-migration.md).
The implemented six-wake harness remains useful as a pilot and calibration
tool. Do not discard it or launch it as the headline experiment. This note does
not authorize implementation or a paid run.

## 1. Current experiment verdict

The current experiment has a sound CL measurement skeleton:

- six sequential wakes;
- fresh-state evaluation after every sleep;
- a complete retention matrix;
- matched no-sleep floors;
- backward transfer, installation, paraphrase, knowledge-battery, and PPL
  measurements;
- three seeds with failures kept visible.

Its learning distribution is narrow: every target is an entity-to-five-digit-
code binding. The same fact set is counterbalanced across seeds. This can show
that a method learns and retains synthetic code bindings. It cannot show that
the method learns arbitrary knowledge or broad language tasks.

The one-nat margin threshold is an installation verdict for the code family.
It is not a general learning threshold and it is not the definition of
forgetting. Forgetting is the later continuous loss of performance on an item
that the model learned earlier.

## 2. Generalizability axes

A generalizable claim needs evidence across four independent axes:

1. **Content:** different kinds of novel knowledge, not only codes.
2. **Language:** unseen wording, indirect questions, and natural dialogue
   contexts.
3. **Sequence:** different fact sets, scenarios, and learning orders.
4. **Model:** confirmation outside one checkpoint and model scale.

More examples from the same code template improve precision on that template.
They do not establish these four forms of generalization.

## 3. Two-stage experiment program

### 3.1 Stage 1 — controlled heterogeneous knowledge

Keep the useful structure of the current run:

- six wakes;
- adaptive natural dialogue;
- four target items per wake;
- fresh-state probes after every sleep;
- full retention matrices;
- fixed, hashed artifacts and matched floors.

Replace the single code schema with a typed knowledge-item schema. The proposed
initial families are:

1. **Opaque bindings:** the current entity codes.
2. **Natural attributes:** locations, materials, categories, colors, or other
   short values.
3. **Dates and quantities:** values whose form and token length differ from
   codes.
4. **Relations or short rules:** relations between entities or a short
   conditional behavior.

Use one item from each family in every wake. Use invented entities and
counterfactual values so the target is demonstrably new. Reject items that the
no-sleep arm already answers correctly or whose floor leaves no measurable
learning headroom.

Every item needs preregistered probes for:

- the original question;
- paraphrased questions;
- an indirect question;
- a matched distractor or alternative answer;
- fresh-state evaluation after every later sleep.

Do not raise the four-item wake load merely to increase sample count. That load
is tied to measured binding capacity. Increase evidence through independent
streams instead.

The final experiment needs independently generated manifests with different
facts, scenarios, and task orders. Reusing one fact set across optimization
seeds measures stochastic variation, not content generalization. The proposed
minimum is three independent manifests crossed with training seeds, with a
separate final holdout manifest set. Freeze the exact count after a cost and
power calculation, before any outcome is seen.

A correction or revision family is scientifically important, but it changes
the task from retention to controlled replacement. Register it as a separate
condition or a later stage unless its scorer and old-versus-new behavior are
settled before Stage 1.

### 3.2 Stage 2 — external language-task confirmation

After Stage 1 selects a method without seeing the final external holdout, test
the winner on a preregistered subset of a standard continual-instruction suite,
such as CITB or TRACE.

Stage 2 asks whether the result extends from controlled novel facts to
heterogeneous language tasks. It must retain the benchmark's task-valid scores
and full sequential performance matrix. It must also measure damage to the
warm-start model's inherited capabilities.

Do not begin with the full external benchmark. Standard instruction datasets
are far from the wake/sleep mechanism and can hide whether a failure comes from
memory capture, dream coverage, training, or evaluation. Stage 1 isolates those
mechanisms first; Stage 2 supplies external validity.

## 4. Required comparison arms

The primary comparison set becomes:

1. **Generative replay:** the current 300-distinct-dream treatment.
2. **Bounded raw experience replay:** retain a fixed-size sample of prior wake
   material and interleave it with current-wake training.
3. **Sequential SFT:** train only on the current realized wake transcript.
4. **No update:** the matched floor and recurrent-state control.
5. **Offline joint training:** a reference for attainable learning and
   intransigence, not a deployable CL arm.

Raw experience replay is mandatory. The field repeatedly finds that simple
replay matches or beats specialized CL machinery. Without this arm, the run can
show that dreams beat current-wake SFT, but not that they improve on standard
continual learning.

Compare raw and generative replay at matched:

- gradient or training tokens;
- optimizer updates;
- persistent storage;
- trainable parameters.

Report generation compute separately. A generated treatment trades stored raw
data for generation cost and cached artifacts; it is not resource-free.

## 5. Learning, forgetting, and damage metrics

Keep one retention matrix per knowledge family:

\[
R_{i,j} = \text{performance on cohort }j\text{ after sleep }i.
\]

Each family uses a task-valid continuous score. Do not average raw code margin,
exact match, ROUGE, execution success, or other unlike scales. Report each
family separately. A macro-average is allowed only after a preregistered
normalization.

Primary sequential measurements:

- performance directly after the item's learning sleep;
- final average performance;
- backward transfer from own-sleep to final performance;
- max-to-current forgetting, which catches a later peak followed by collapse;
- performance against the offline joint reference.

Behavioral secondary measurements:

- exact or task-valid success;
- paraphrase retrieval;
- indirect retrieval;
- distractor rejection;
- installed-item counts under a family-specific, preregistered rule.

The current one-nat threshold remains only for the opaque-code family. Variable
answer lengths and task types need their own calibrated behavioral rules. Every
new metric ships with a no-update floor before its values are pooled or quoted.

Pre-existing capability damage needs:

- the expanded self-calibrated knowledge battery;
- answer-log-probability distributions and lower-tail change;
- held-out perplexity;
- a frozen external suite for knowledge, reasoning, instruction following, and
  generation format that the warm-start model can perform before training.

The external suite must be selected and frozen in a calibration phase that is
separate from final evaluation. A small hand-written battery cannot bound broad
model behavior by itself.

## 6. Independence and reporting

Content manifests, training randomness, and task order measure different
sources of variation. Record them separately. Do not pool every fact exposure
as if it were an independent run.

Report:

- every manifest, seed, arm, wake, and family value;
- means and uncertainty across independent streams;
- failures and outlier streams without pooling them away;
- task-order and token-distance effects;
- learning and forgetting curves, not only final endpoints;
- training tokens, generation tokens, storage, wall time, and peak VRAM.

The winning Stage 1 comparison must be repeated on at least one held-out model
scale. Cross-architecture confirmation is a stronger later claim; it is not
required to establish that the mechanism is robust within the current Mamba
lineage.

## 7. Success criteria and claim boundary

A method supports a generalizable continual-knowledge claim only if:

1. it learns above the no-update floor across all registered knowledge
   families;
2. retention survives later wakes without one family, especially codes,
   carrying the result;
3. the conclusion holds on unseen manifests and task orders;
4. it matches or dominates bounded raw replay on the learning–forgetting
   frontier, or provides a clear registered resource advantage;
5. external capability damage is not worse than the relevant baselines;
6. the result confirms on the held-out model scale;
7. Stage 2 preserves the conclusion on external language tasks.

Before Stage 2, the maximum claim is **generalization across controlled novel
knowledge types in adaptive conversations**. After a successful Stage 2, the
claim can expand to continual language-task learning. Neither stage alone
supports deployment-wide absence of catastrophic forgetting.

## 8. Required harness changes

The design implies these implementation slices:

1. Generalize the manifest from `(entity, five-digit code)` pairs to typed
   knowledge items with injection text, probes, targets, distractors, family,
   and scorer metadata.
2. Add family scorers and per-family retention matrices without breaking the
   existing code scorer.
3. Add bounded raw experience replay and an offline joint reference with
   explicit resource accounting.
4. Add max-to-current forgetting and normalized macro reporting.
5. Add independent content manifests, task-order identities, and final-holdout
   separation.
6. Add the frozen external capability suite and its warm-start calibration
   artifact.
7. Hash every shared manifest, probe set, replay buffer, and calibration
   artifact into every result; refuse to pool mismatches.
8. Extend fake-backbone and real-hardware smoke tests to every new arm and
   scorer before the science launch.

Do not implement these slices until the unresolved design choices below are
frozen and the user gives explicit implementation approval.

## 9. Open decisions before implementation

- Final Stage 1 knowledge families and their exact scorers.
- Whether correction/revision is part of Stage 1 or a separate condition.
- Number of independent manifests and optimization seeds after the cost and
  power calculation.
- Raw-replay storage unit, buffer size, sampling policy, and matched-budget
  rule.
- Offline joint-reference schedule.
- External capability suite and warm-start pass criteria.
- CITB or TRACE subset, task order, and task-valid metrics for Stage 2.
- Primary and held-out model scales.

These are scientific choices. Freeze them in a follow-up discussion before
harness implementation or paid execution.
