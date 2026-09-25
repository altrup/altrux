# Research 2026-09-25: evaluation methodology of "Language Models Need Sleep"

## Decision

The paper measures one end-point score per benchmark after exposure: accuracy, ChRF, perplexity, avg@16, no-context SQuAD accuracy, or ARC success rate.
It gives no retention matrix, no backward transfer, no forgetting metric, no seeds, no variance, and no matched-token accounting. The only forgetting signal is the gap between single-language and sequential training in one translation task.
Our LAMA-CKL report already measures more than the paper: a per-cycle curve on both sets, forgetting at peak and at the end, three seeds, and token and compute accounting.
Adopt nothing into the per-fact scorer or into the published LAMA-CKL aggregates. One optional addition goes on top of the report: a compute-to-target line from their efficiency appendix, with the target fixed before the run.

## Source and scope

- **Paper.** Behrouz, Hashemi, Javanmard, and Mirrokni, *Language Models Need
  Sleep: Learning to Self-Modify and Consolidate Memories*
  ([arXiv 2606.03979](https://arxiv.org/abs/2606.03979)). Google Research and
  Cornell. This note reads **v2 (10 July 2026)** in full through the
  [HTML version](https://arxiv.org/html/2606.03979v2), including the result
  figures as images. All section links below point to v2. The front matter
  says a version was on OpenReview from September 2025.
- **Scope.** This note reviews the evaluation only: data, metrics, baselines,
  controls, and reporting. It does not evaluate the mechanism. The
  [2026-07-22 consolidation landscape](RESEARCH-20260722-memory-consolidation-landscape.md)
  and the [2026-08-05 dream-distillation prior art](RESEARCH-20260805-dream-distillation-prior-art.md)
  summarize the mechanism.
- **Marker.** `[NOT IN PAPER]` means that we searched v2 for the item and it
  is absent, as in the
  [2026-07-24 data-structure note](RESEARCH-20260724-training-data-structure.md).
- **Correction to the 2026-07-22 note.** That note lists "stated limits". v2
  has no limitations section. The nearest text is the list of SEAL challenges
  in [Section 3.4](https://arxiv.org/html/2606.03979v2#S3.SS4) and the
  OPSD failure modes in
  [Appendix A.4](https://arxiv.org/html/2606.03979v2#A1.SS4.SSS0.Px4).

## Benchmarks and tasks

[Section 4](https://arxiv.org/html/2606.03979v2#S4) has seven experiments.
[Appendix B](https://arxiv.org/html/2606.03979v2#A2) says that every
experiment follows the settings of its original benchmark and refers to those
papers for detail. Thus many protocol details are absent from this paper.

### Class-incremental text classification

- **Data.** CLINC150 (150 intents, 23.7K queries), Banking77 (77 intents,
  13,083 examples), and DBpedia level-2 with 70 classes, subsampled to 10K
  train and 1K test ([B.1](https://arxiv.org/html/2606.03979v2#A2.SS1)).
- **New knowledge.** New intent or ontology classes. The protocol follows
  InCA (Momeni et al., 2025)
  ([Section 4.1](https://arxiv.org/html/2606.03979v2#S4.SS1)).
- **Held out.** The test split of each dataset.
- **Stream.** `[NOT IN PAPER]` number of class increments, class order, and
  items per increment.
- **Backbones.** Llama-3B and Llama3-8B.
- **Only this phase.** Memory consolidation without dreaming.

### Consolidation levels on long context

- **Data.** MK-NIAH from RULER, LongHealth (20 case documents of about 5.1K
  to 6.8K words, 200 sampled questions), and QASPER (full paper text as
  context) ([B.1](https://arxiv.org/html/2606.03979v2#A2.SS1)).
- **New knowledge.** The long context. The evidence is in the context at
  test time.
- **Sweep.** One to four memory levels, crossed with the lowest update
  frequency of 512, 2K, or 8K
  ([Figure 4](https://arxiv.org/html/2606.03979v2#S4.F4)).
- **Stream.** No sequential stream. `[NOT IN PAPER]` backbone and item
  counts for MK-NIAH and QASPER.

### Continual translation of a novel language (CTNL)

- **Data.** MTOB (Kalamang) and Manchu, two languages unseen in
  pretraining, as in Nested Learning
  ([Section 4.1](https://arxiv.org/html/2606.03979v2#S4.SS1)).
- **New knowledge.** Each language, given in context.
- **Stream.** Two conditions. (a) Learn and test each language alone.
  (b) Learn both in sequence, then test each.
- **Held out.** `[NOT IN PAPER]` test split, item count, language order, and
  backbone. `[NOT IN PAPER]` whether the language material is still in the
  context at test time.

### BABILong

- **Data.** BABILong at context lengths from 0 to 10M tokens
  ([B.2](https://arxiv.org/html/2606.03979v2#A2.SS2)).
- **Training.** The small models, Hope included, are fine-tuned with the
  official BABILong training protocol. This is an in-distribution test of a
  trained task, not a continual stream.
- **Held out.** `[NOT IN PAPER]` which BABILong sub-tasks, and how many items.

### Mathematical reasoning

- **Data.** AIME-24, AIME-25, and HMMT-25
  ([Table 2](https://arxiv.org/html/2606.03979v2#S4.SS1.fig1)).
- **Backbones.** Qwen3-1.7B and Qwen3-8B.
- **Training.** 100 steps for Sleep and SFT, 500 for GRPO; learning rate
  5e-6, batch 32, LoRA rank 64, alpha 128
  ([Table 5](https://arxiv.org/html/2606.03979v2#A2.T5)).
- **Stream.** None. This is one post-training run, not continual learning.
  `[NOT IN PAPER]` the training data.

### Knowledge incorporation (SQuAD)

- **Data.** SQuAD passages, with the setup of SEAL (Zweiger et al., 2025)
  "including the choice of models and parameters"
  ([Section 4.1](https://arxiv.org/html/2606.03979v2#S4.SS1)).
- **New knowledge.** Facts in a passage. The model trains on dreams made from
  the passage. The test has no passage in context.
- **Two settings.** Single passage (n = 1), and continued pretraining with
  200 passages in one run, tested on all 974 related questions. Five dreams
  per passage in the continued-pretraining setting.
- **Held out.** The questions. `[NOT IN PAPER]` the base model and the grader.
  The SEAL paper itself uses Qwen-2.5-7B and a gpt-4.1 judge
  ([SEAL](https://arxiv.org/html/2506.10943), Appendix B); this paper does not
  restate either.
- **Only this experiment and ARC** use the full Sleep (consolidation plus
  dreaming).

### Few-shot ARC

- **Data.** A filtered ARC subset: 11 training tasks and 8 held-out tasks.
  Backbone Llama-3.2-1B ([B.3](https://arxiv.org/html/2606.03979v2#A2.SS3)).
- **Procedure.** For each training task, sample 60 dreams and reject 45. At
  test time, generate 5 dreams per unseen task and apply each independently.
- **Held out.** The 8 evaluation tasks.

## Metrics

### Per-item scores

| Experiment | Score | Source |
|---|---|---|
| Class-incremental | Accuracy (%) | [Figure 3](https://arxiv.org/html/2606.03979v2#S3.F3) |
| MK-NIAH, LongHealth | Accuracy (%) | [Figure 4](https://arxiv.org/html/2606.03979v2#S4.F4) |
| QASPER | Perplexity, lower is better | [Figure 4](https://arxiv.org/html/2606.03979v2#S4.F4) |
| CTNL | ChRF, Manchu to English and Kalamang to English | [Figure 5](https://arxiv.org/html/2606.03979v2#S4.F6.fig1) |
| BABILong | Accuracy (%) per context length | [Figure 6](https://arxiv.org/html/2606.03979v2#S4.F6.fig2) |
| Math | avg@16 | [Tables 1 and 2](https://arxiv.org/html/2606.03979v2#S4.F6.fig3) |
| SQuAD | "Mean no-context SQuAD accuracy" | [Table 3](https://arxiv.org/html/2606.03979v2#S4.SS1.fig2) |
| ARC | Fraction of dreams that give a correct answer | [B.3](https://arxiv.org/html/2606.03979v2#A2.SS3) |

`[NOT IN PAPER]` the answer-matching rule for any accuracy: exact match,
normalized match, multiple-choice letter, or judge.

The ARC unit is a dream, not a task. With 8 tasks and 5 dreams each, the
80% figure is 32 of 40 dream trials. The paper does not report how many of
the 8 tasks are solved.

### Acquisition and retention

- **Acquisition.** Each experiment reports one post-exposure score. There is
  no separate acquisition curve.
- **Retention of new knowledge.** Only CTNL tests it. The comparison is
  sequential against single-language ChRF, read by eye from Figure 5. The
  text says ICL "largely revert[s] to pre-trained behavior" and Hope-3
  "nearly recovers its single-language performance".
- **Retention of prior knowledge.** `[NOT IN PAPER]` for every experiment.
  No experiment measures what the base model knew before the update.
  Continued pretraining on 200 SQuAD passages reports only the score on
  those passages.
- **Checkpoints.** One evaluation point per run. The memory-level axis in
  Figures 4 and 5 is a configuration sweep, not a time axis.
  `[NOT IN PAPER]` checkpoint-selection rule.
- **Summary metrics.** `[NOT IN PAPER]` a retention matrix R[i, j], ACC,
  BWT, FWT, max-to-current forgetting, or intransigence as defined in the
  [2026-08-14 foundations note](RESEARCH-20260814-continual-learning-foundations.md#5-standard-catastrophic-forgetting-metrics).
  The class-incremental result is one bar per method, which is at most the
  final-average ACC.

### Figure and text mismatches

These come from reading the v2 figures as images. They do not change the
conclusion, but they limit how much weight a number can carry.

- The Figure 5 legend says Sleep-1/2/3; the text says Hope-1/2/3. The caption
  says red is single-language and blue is continual. With that key, the blue
  Sleep-2 and Sleep-3 points lie above the red ones, so sequential training
  scores higher than single-language training. The text says Hope-3 "nearly
  recovers" single-language performance. One of the two is wrong.
- The text for Figure 6 (BABILong) names RMT, ARMT, Titans, and GPT-4o-mini.
  The plot shows only Llama3 + RAG, Llama3.1-8B-Instruct, GPT-4, and
  Llama + Sleep. The caption refers to red and blue points; the plot has
  lines in other colours.
- The text calls the ablation table "Figure 6" and cites tables as
  "Section 4.1" ([Section 4.2](https://arxiv.org/html/2606.03979v2#S4.SS2)).

## Baselines and controls

| Baseline | Where | What it controls for |
|---|---|---|
| ICL | Class-incremental, long context, CTNL | No parametric update. For class-incremental it is "same continual pre-training process but without sleep". |
| EWC | Class-incremental | Parameter regularization |
| InCA | Class-incremental | External learner |
| Hope without sleep | Class-incremental | The true no-sleep control: same architecture, no explicit distillation |
| DuoAttention, Cartridges | Long context; Cartridges also CTNL | Efficient context and KV compression |
| SFT | CTNL, math | Plain fine-tuning. For CTNL, no numbers: off the plot |
| Base, GRPO, OPSD | Math | Post-training alternatives |
| Base, fine-tune with no dreaming, SEAL | SQuAD | No update; passage-only update; RL self-edit |
| ICL, TTT, SEAL | ARC | No update; test-time training; RL self-edit |
| RAG, GPT-4, Llama3.1-8B-Instruct | BABILong | Retrieval and long-context models |

Sources: [Section 4.1](https://arxiv.org/html/2606.03979v2#S4.SS1),
[B.2](https://arxiv.org/html/2606.03979v2#A2.SS2),
[B.3](https://arxiv.org/html/2606.03979v2#A2.SS3).

**Ablations.** Table 1 removes imitation learning, the semantic reward, or
expansion on Qwen3-8B math, and adds expansion to OPSD. Table 3 removes
gradient-based selection, the random expert, or dreaming on SQuAD.

**Replay baseline.** `[NOT IN PAPER]`. No experience-replay or
rehearsal arm appears. The foundations note records replay as the strongest
recurring baseline in the field.

**Compute and token accounting.**

- Parameters: five extra MLP blocks of dimension 64; active parameter count
  equal to the base model
  ([Appendix B](https://arxiv.org/html/2606.03979v2#A2)).
- Steps: fixed per method for math only (Table 5). Sleep and SFT both take
  100 steps, but a Sleep step includes dream generation, RL, and distillation.
- Efficiency: at equal steps, SFT is 4x more efficient than Sleep. At equal
  target performance, SFT needs 4.3x, 3.6x, and 4.8x the wall-clock time on
  AIME-24, AIME-25, and HMMT-25
  ([B.5](https://arxiv.org/html/2606.03979v2#A2.SS5)).
- `[NOT IN PAPER]` source tokens, generated dream tokens, gradient tokens,
  hardware, GPU-hours, and baseline hyperparameters outside Table 5. There is
  no matched-token comparison.

## Seeds, variance, and absent items

- `[NOT IN PAPER]` number of seeds or runs for any result.
- `[NOT IN PAPER]` standard deviation, standard error, confidence interval, or
  significance test. No figure has error bars.
- `[NOT IN PAPER]` code, data, or split release.
- `[NOT IN PAPER]` checkpoint-selection rule and any validation split for it.
- `[NOT IN PAPER]` any measure of damage to general capability outside the
  benchmark under test.
- `[NOT IN PAPER]` exact baseline numbers for SFT and Cartridges on CTNL.
- The SQuAD and ARC numbers for SEAL and TTT are not checked here against the
  SEAL paper.

Several gains are small against this missing variance. Examples: SQuAD
continued pretraining, 44.3 (two-level) against 43.2 (SEAL); math, Qwen3-8B
AIME-25, 69.0 (Sleep) against 68.1 (base).

## Mapping onto Altrux

**Constraint.** The LAMA-CKL reproduction gate in the
[2026-08-21 selection note](RESEARCH-20260821-continual-knowledge-benchmark-selection.md#faithful-reproduction-gate)
stays intact. The per-fact scorer in
[`evaluation.py`](../../sft/experiments/lama_ckl/evaluation.py) stays
paper-faithful: teacher-forced argmax accuracy over the object tokens of the
last object occurrence, from fresh recurrent state, with no evidence in the
probe. The four published metrics stay. Anything taken from this paper is a
labelled addition on top of the report.

| Their metric | LAMA-CKL per-fact score | `report.py` aggregate | Change if we model on theirs | Case against the change |
|---|---|---|---|---|
| One end-point accuracy per run (Figure 3, Table 3) | Mean object-token accuracy per set | `final_accuracy`, `top_accuracy` | Report only the final point. | Drops the per-cycle curve and `top_accuracy`, which is the published reproduction metric. Reject. |
| No-context generated-answer accuracy, judge-graded by inheritance from SEAL (Table 3) | Teacher-forced object-token argmax, no generation | None | Add generated-answer exact match on the same probes as a secondary line. | A judge adds a non-deterministic outside model. Generation adds a decode policy. The object is a short token span, so argmax is already strict. Not needed for the gate. |
| Sequential against isolated training gap (Figure 5) | The not-to-forget set tests prior facts directly | `forgetting_at_peak`, `final_forgetting` | Add an isolated-cohort control arm. | LAMA-CKL has one to-learn cohort, so there is no sequence to isolate. Our forgetting lines already measure prior-fact loss directly, which the paper never does. Revisit only with a multi-cohort stream. |
| Consolidation-stage sweep (Figures 4 and 5) | Not applicable | Not applicable | Sweep dream depth (dream count or sleep steps) as a treatment axis. | Multiplies arms by seeds. It is a mechanism study, not an evaluation change. Do it after the gate, if at all. |
| avg@16 sampled score (Tables 1 and 2) | Single deterministic pass | None | Sample the probe several times. | Adds sampling variance to a deterministic published metric. Reject. |
| Wall-clock to matched performance (B.5) | Not applicable | `wall_seconds`, `gpu_hours`, `token_gradients`, `optimizer_steps` | Add cycles-to-target and GPU-hours-to-target, from existing `cycle_seconds` per cycle. | The target sits on the test curve. An arm that never reaches it gives no number. Fix the target before the run from the reproduction result, and keep the line secondary. This is the one optional adoption. |
| Fraction of dreams correct (ARC) | Not applicable | Not applicable | None. | Counts dream trials, not items, so it inflates n. Our score is per fact. Reject. |

Our report already covers what the paper lacks: both sets after every cycle,
`forgetting_at_peak` and `final_forgetting` against the cycle-0 baseline, three
registered seeds with standard error, a hashed split, and source, wake, review,
generated, and gradient token counts per arm. It does not yet report a
retention matrix, because LAMA-CKL has one cohort; the per-cycle curve is the
matrix for that case.

The paper also does not isolate parametric knowledge in most experiments. In
the long-context tasks and BABILong the evidence is in the context at test
time, and CTNL does not say otherwise. Only the SQuAD experiment tests with
no evidence in context. By the selection rule of the 2026-08-21 note, only
that experiment is comparable to the Altrux claim.

## Deferred: knowledge seeding

This note does not evaluate knowledge seeding. It is an upward distillation
from the model before a base-weight update (teacher) into a new low-rank
expert in the next slower memory block (student). The objective mixes a
GKD-style on-policy divergence with an RL "learning to imitate" reward, which
combines a frozen semantic reward model and a thresholded Levenshtein
similarity ([Section 3.3](https://arxiv.org/html/2606.03979v2#S3.SS3),
Equations 3 and 4). It is the nearest published relative of the Altrux
frozen-teacher dream distillation, per the
[2026-08-05 prior-art note](RESEARCH-20260805-dream-distillation-prior-art.md).
Table 1 gives its only component ablation, on math, without variance. Review
it only if a later Altrux arm adds an imitation or sequence-level reward.

## Deferred: parameter-expansion experts

This note does not evaluate parameter expansion. Each sleep adds a low-rank
expert to a sparse mixture-of-experts MLP block. Consolidation then trains
only that expert and resets the experts of the faster block
([Section 3.2](https://arxiv.org/html/2606.03979v2#S3.SS2)). Dreaming also
routes to a random extra expert
([Section 3.4](https://arxiv.org/html/2606.03979v2#S3.SS4)). The mechanism
requires a mixture-of-experts backbone. Altrux does not plan one: after the
current tests, the planned final backbone is a Nemotron hybrid of about 30B
parameters. The only transferable idea is "train new parameters, freeze old
ones", which LoRA arms already are. The paper's own evidence for expansion is
one row in Table 1 ("w/o Expansion") and one OPSD row, with no seeds.

## Decision risks and stopping condition

- **Risk: under-reading the paper.** Appendix B defers protocol details to the
  cited benchmark papers. A detail marked `[NOT IN PAPER]` can exist in
  InCA, Nested Learning, SEAL, or BABILong. That does not change the
  decision: the paper reports no retention metric, no variance, and no token
  accounting of its own.
- **Risk: a later version.** A v3 can add seeds or a limitations section.
  Re-check the version before a writeup cites this note.
- **Stopping condition.** Do not change the scorer or the four published
  aggregates on the basis of this paper. Add the compute-to-target line only
  after the reproduction gate passes and with the target fixed first.
