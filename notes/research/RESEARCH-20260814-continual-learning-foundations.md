# Research — 2026-08-14: continual-learning foundations, methods, and evaluation

Broad literature review, not standing experiment direction. This note answers
two questions: which papers define the mainstream continual-learning (CL)
field, and how does that field measure catastrophic forgetting? It does not
review sleep, recurrent state, or Altrux-specific mechanisms in depth; the
existing consolidation and dream-distillation notes cover those topics.

Paper titles, publication venues, core mechanisms, and metric definitions were
checked against primary papers or official proceedings. Method limitations are
either stated by the source or direct consequences of the method's declared
assumptions. This is a literature map, not a reproduction study.

## 1. Main findings

1. **The repository's prior review is specialized, not representative of CL as
   a field.** It starts from sleep, replay, context distillation, and recurrent
   memory. Mainstream CL starts from sequential interference, then divides
   methods into replay, regularization, and parameter-isolation families.
2. **Simple replay is the strongest recurring baseline.** Multiple papers find
   that ordinary experience replay or even memory-only retraining can match or
   beat more elaborate CL mechanisms. Any new method that does not compare
   against a fair replay baseline has not established much.
3. **The evaluation setting is part of the result.** Task-incremental results,
   where a task identifier selects a head or route at inference, are not
   comparable with class-incremental or task-free results.
4. **Catastrophic forgetting is normally a change in held-out performance over
   a sequence, not a threshold crossing.** The standard object is a performance
   matrix evaluated after every task. Backward transfer and max-to-current
   forgetting summarize that matrix.
5. **Language models need a wider retention surface.** Past-task scores remain
   necessary, but an LLM can also lose general knowledge, instruction following,
   output format, safety behavior, or zero-shot transfer without failing the
   narrow tasks in its continual stream.

## 2. Problem and evaluation settings

McCloskey and Cohen's 1989 study of
[catastrophic interference](https://doi.org/10.1016/S0079-7421(08)60536-8)
showed that sequential training can overwrite distributed representations of
earlier material. Robins' 1995 review of
[rehearsal and pseudorehearsal](https://doi.org/10.1080/09540099550039318)
established the oldest durable remedy: interleave old examples, or generated
proxies for them, with new learning. The complementary-learning-systems account
of [McClelland, McNaughton, and O'Reilly
(1995)](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf)
explains why a fast store plus slow, interleaved learning avoids destructive
interference. These three works define the problem and the replay intuition
before modern deep CL.

Modern papers use several settings. The labels below follow the explicit
taxonomy in [van de Ven, Tuytelaars, and Tolias
(2022)](https://www.nature.com/articles/s42256-022-00568-3):

- **Task-incremental learning (Task-IL):** task identity is available at
  inference. Separate output heads, masks, or routes are permitted. This is
  usually the easiest setting.
- **Domain-incremental learning (Domain-IL):** the input distribution changes,
  the output semantics stay fixed, and task identity is unavailable at
  inference.
- **Class-incremental learning (Class-IL):** new classes arrive over time and
  inference must choose among all classes seen so far without a task identifier.
- **Online, streaming, or task-free CL:** examples arrive once or in small
  batches, and clean task boundaries may be absent. Papers differ on whether
  delayed labels, replay, and boundary detection are allowed, so those details
  must be stated.

The dataset name does not identify the setting. Split CIFAR with one head per
task is Task-IL; the same data with one shared head and no task identifier is
Class-IL. [Hsu et al. (2018)](https://arxiv.org/abs/1810.12488) show that such
scenario differences can be larger than the differences between algorithms.

## 3. Canonical method families

### 3.1 Functional and parameter regularization

Regularization methods retain one fixed-capacity model and constrain later
updates. They need no raw replay buffer, but protection of old behavior competes
with plasticity for new learning.

| Paper | Mechanism | Main assumption or ceiling |
|---|---|---|
| [Learning without Forgetting (LwF), Li and Hoiem 2016](https://arxiv.org/abs/1606.09282) | Distill the old model's outputs while training on new-task data. | Old outputs on new data must constrain the relevant old behavior; the original setup uses task-specific outputs. |
| [Elastic Weight Consolidation (EWC), Kirkpatrick et al. 2017](https://doi.org/10.1073/PNAS.1611835114) | Penalize movement of parameters with high estimated Fisher importance for earlier tasks. | The common diagonal Fisher is a local approximation; accumulated protection reduces available plasticity. |
| [Synaptic Intelligence (SI), Zenke, Poole, and Ganguli 2017](https://proceedings.mlr.press/v70/zenke17a.html) | Accumulate each parameter's contribution to loss reduction during training, then protect important parameters. | Importance penalties cannot recover past examples and still consume fixed model capacity. |
| [Memory Aware Synapses (MAS), Aljundi et al. 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.html) | Estimate parameter importance from output sensitivity, without labels. | It has the same fixed-capacity stability–plasticity limit as other importance penalties. |

### 3.2 Rehearsal and replay

Replay retains old examples, generated examples, or old outputs and interleaves
them with current learning. It is the most persistent method family and the
most important baseline.

| Paper | Mechanism | Main assumption or ceiling |
|---|---|---|
| [iCaRL, Rebuffi et al. 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html) | Keep a bounded exemplar set, distill old outputs, and classify by nearest exemplar mean. | Specialized to Class-IL; a fixed total buffer gives fewer exemplars to each class as classes accumulate. |
| [Gradient Episodic Memory (GEM), Lopez-Paz and Ranzato 2017](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) | Project updates so loss does not rise on episodic samples from previous tasks. | Stores raw examples and solves a constrained update whose cost grows with task constraints. |
| [Deep Generative Replay (DGR), Shin et al. 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html) | Train a generator and solver, then mix generated old-task samples with current data. | Adds a generator and depends on the fidelity of generated samples and labels over repeated cycles. |
| [A-GEM, Chaudhry et al. 2019](https://arxiv.org/abs/1812.00420) | Replace GEM's per-task constraints with one averaged replay-gradient constraint. | Cheaper than GEM, but the relaxation does not protect each previous task separately. |
| [On Tiny Episodic Memories, Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486) | Jointly train on each current batch and a tiny episodic buffer. | Stores raw data, but demonstrates that a simple replay baseline can beat specialized methods even with one example per class. |
| [Dark Experience Replay (DER/DER++), Buzzega et al. 2020](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | Replay buffered inputs while matching stored historical logits; DER++ also uses labels. | Stores examples and outputs, including errors present when the examples entered the buffer. |

Replay comparisons are meaningful only at matched storage and update budgets.
A method that stores a fixed number of examples per task has memory growth that
a fixed-total-buffer method does not. Generated replay trades raw-data storage
for generator parameters, generation compute, and possible distribution drift;
it is not memory-free in the broader resource sense.

### 3.3 Parameter isolation and expansion

Isolation methods prevent direct overwrite by assigning different parameters or
masks to different tasks. Near-zero forgetting can therefore conceal growing
capacity or reliance on an inference-time task identifier.

| Paper | Mechanism | Main assumption or ceiling |
|---|---|---|
| [Progressive Neural Networks, Rusu et al. 2016](https://arxiv.org/abs/1606.04671) | Freeze one network column per task and connect each new column laterally to old features. | Parameters and inference cost grow with tasks; task identity selects a column. |
| [PackNet, Mallya and Lazebnik 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.html) | Prune and freeze weights for each task, then train later tasks in freed capacity. | Capacity eventually fills; inference needs the task mask. |
| [Hard Attention to the Task (HAT), Serra et al. 2018](https://proceedings.mlr.press/v80/serra18a.html) | Learn near-binary task masks and protect units selected by earlier masks. | Requires task identity and has a finite reusable-capacity budget. |

These methods answer a different question from a fixed-model, task-free learner.
They are useful upper bounds on interference, but they are not fair direct
comparators unless model growth and task routing are allowed in the target
setting.

## 4. Strong baselines changed the field's conclusions

Three papers are especially important because they test whether complexity is
buying real continual-learning ability:

- [On Tiny Episodic Memories](https://arxiv.org/abs/1902.10486) finds that
  ordinary experience replay with very small buffers outperforms several
  purpose-built CL methods in a one-pass protocol.
- [GDumb, Prabhu, Torr, and Dokania
  2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3587_ECCV_2020_paper.php)
  keeps a balanced memory and trains from scratch on it at evaluation time. It
  beats many continual learners despite not updating a deployed model online.
  The result exposes benchmarks that reward memory selection more than
  continual optimization.
- [DER](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)
  combines ordinary replay with historical-logit matching and remains a strong,
  simple general-CL baseline without explicit task boundaries.

The minimum serious comparison set is therefore:

1. sequential fine-tuning;
2. no update, where meaningful;
3. simple experience replay at the same persistent-memory and update budget;
4. joint or offline training as an upper reference, not a deployable CL arm;
5. the proposed method.

For generative replay, add raw replay at matched training tokens and a
generator-free fine-tuning reference. Otherwise the experiment cannot separate
the value of generated data from the value of replay itself.

## 5. Standard catastrophic-forgetting metrics

Let there be `T` learning stages. Define

\[
R_{i,j} = \text{held-out performance on task or cohort }j
          \text{ after learning stage }i.
\]

The performance can be accuracy, exact match, F1, ROUGE, execution success, or
another task-valid score. A paper should not average unlike raw metrics without
a declared normalization. GEM formalized the common matrix summaries:

\[
\mathrm{ACC} = \frac{1}{T}\sum_{j=1}^{T} R_{T,j}
\]

is final average performance, and

\[
\mathrm{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}(R_{T,j} - R_{j,j})
\]

is backward transfer. Negative BWT means later learning reduced earlier-task
performance. Positive BWT means later learning improved it.

GEM also defines forward transfer relative to an untrained baseline `b_j`:

\[
\mathrm{FWT} = \frac{1}{T-1}\sum_{j=2}^{T}(R_{j-1,j} - b_j).
\]

FWT measures performance on a future task before training it. It is not the
same as how well the model learns that task after training.

[RWalk, Chaudhry et al.
2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Arslan_Chaudhry__Riemann_Walk_ECCV_2018_paper.pdf)
defines max-to-current forgetting for an earlier task `j` after stage `k`:

\[
f_j^k = \max_{\ell \in \{j,\ldots,k-1\}} R_{\ell,j} - R_{k,j},
\qquad
F_k = \frac{1}{k-1}\sum_{j=1}^{k-1} f_j^k.
\]

This differs from `-BWT` when a task improves after its own learning stage and
then loses that improvement. Report both when backward improvement is possible.

RWalk also defines **intransigence** by comparing current-task performance with
a reference trained jointly on all data available through that stage. It
detects the false success mode where strong protection retains old tasks only
because the learner refuses to acquire new ones. When a joint reference is not
feasible, report the diagonal `R[j,j]`, within-stage learning curves, or another
declared plasticity measure. There is no single standard scalar called
"plasticity."

A binary threshold can be a useful application verdict, but it is not the
field's definition of forgetting. A threshold discards the magnitude and path
of performance changes. It should be reported beside, not instead of, the
continuous matrix.

## 6. Evaluation protocol and common failure modes

### 6.1 What a comparable result must state

- CL setting and all train- and test-time task information;
- task order, number of stages, boundary availability, and number of passes;
- model initialization and pretraining;
- raw-example, generated-example, logit, optimizer, and model-growth storage;
- replay ratio, total updates or training tokens, and compute;
- hyperparameter-selection data and whether it overlaps the test stream;
- evaluation cadence, full per-stage results, task orders or seeds, and
  uncertainty.

[A-GEM](https://arxiv.org/abs/1812.00420) introduced a one-pass protocol and
selected hyperparameters on disjoint tasks. [GDumb](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3587_ECCV_2020_paper.php)
showed that several established comparisons encoded weak or inconsistent
assumptions. [BudgetCL, Prabhu et al.
2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Prabhu_Computationally_Budgeted_Continual_Learning_What_Does_Matter_CVPR_2023_paper.pdf)
showed that rankings can change under explicit compute limits.

### 6.2 Benchmark families

- **Permuted MNIST and Split MNIST:** cheap historical sanity checks. They are
  insufficient evidence for a general CL claim.
- **Split CIFAR-10/100, miniImageNet, and TinyImageNet:** common constructed
  task and class streams. Results require the exact head, class order, split,
  augmentation, and pretraining details.
- **CORe50:** a purpose-built object-recognition stream with new-instance,
  new-class, and mixed streams. See [Lomonaco and Maltoni
  2017](https://proceedings.mlr.press/v78/lomonaco17a.html).
- **CLEAR:** real temporal change across image data from 2004–2014 and a
  near-future streaming protocol. Its authors show that conventional IID
  evaluation can inflate estimated deployment performance. See [Lin et al.
  2022](https://arxiv.org/abs/2201.06289).

Final ACC alone hides collapse followed by recovery. The full matrix or a
per-stage curve is the primary artifact. Multiple task orders matter because
some sequences share representations and labels more than others. Small
per-task test sets require uncertainty estimates; a one-point accuracy change
is not automatically a mechanism effect.

## 7. Language-model continual learning

Language CL inherits the classic matrix but changes the meaning of a task and
the surface on which damage appears.

| Paper | Contribution to the bridge from classic CL |
|---|---|
| [Episodic Memory in Lifelong Language Learning, d'Autume et al. 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/f8d2e80c1458ea2501f98a2cafadb397-Abstract.html) | One-pass text-classification and QA streams without dataset identity or boundaries at train or test time; retrieval and local adaptation replace a simple task oracle. |
| [LAMOL, Sun, Ho, and Lee 2020](https://openreview.net/pdf?id=Skgxcn4YDS) | One language model learns heterogeneous text-to-text tasks and generates pseudo-examples of earlier tasks. This is the direct NLP form of generative replay. |
| [ConTinTin, Yin, Li, and Xiong 2022](https://aclanthology.org/2022.acl-long.218/) | Sequential Natural-Instructions tasks with explicit FWT and BWT, making natural-language task descriptions part of continual transfer. |
| [Continual-T0, Scialom, Chakrabarty, and Muresan 2022](https://aclanthology.org/2022.emnlp-main.410/) | Adds eight tasks to an instruction-tuned model while evaluating old tasks and zero-shot abilities across a much larger inherited task set. |
| [CITB, Zhang et al. 2023](https://aclanthology.org/2023.findings-emnlp.633/) | Long continual-instruction streams with task matrices, BWT, FWT, initial-task retention, and held-out unseen-task performance. |
| [TRACE, Wang et al. 2024](https://openreview.net/forum?id=3qa4YLkcEw) | Eight heterogeneous instruction datasets plus changes in general ability, instruction following, and safety after continual training. |

Classic image CL normally uses one metric across clean classification splits.
Language CL may mix accuracy, F1, ROUGE, code execution, and exact match. It
also starts with broad pretrained behavior that is not represented in the
continual stream. A complete LLM evaluation therefore needs three surfaces:

1. **stream learning and retention:** the full matrix for the tasks or knowledge
   introduced during the sequence;
2. **transfer and generalization:** unseen tasks, paraphrases, and alternative
   instructions;
3. **pre-existing capability retention:** broad knowledge, language modeling,
   instruction following, output format, safety, and generation reliability.

Perplexity is useful for diffuse language-model damage, but it cannot show
which usable capabilities were lost. Behavioral probes are useful for specific
losses, but a small hand-written battery cannot bound broad general ability.
Both are needed, with claim scope matched to their coverage.

## 8. Mapping the current Altrux protocol to the field

This mapping refers to the
[governing six-wake design](../discussion/DISCUSSION-20260811-multisleep-adaptive-wake-and-agent-migration.md)
and its [retention analysis](../../sft/experiments/adaptive/analysis.py).

The current six-wake adaptive experiment already has the standard evaluation
skeleton:

- every prior fact is evaluated after every wake;
- the run records a full retention matrix;
- backward transfer is final performance minus performance after the fact's
  learning wake;
- probes run from fresh recurrent state, so the matrix measures weight
  retention rather than carried working state;
- matched no-sleep results remove each fact's untrained floor;
- seed values and uncertainty remain visible.

These choices are stronger than a final-only result. The matched no-sleep
correction is custom, but it is appropriate because raw code margins have a
nonzero and fact-dependent floor.

The task score is also custom: five-digit entity codes are synthetic facts,
the correct-code-versus-foil log-probability margin is continuous performance,
and a floor-corrected margin of at least one nat counts as installation. The
threshold is an installation verdict. **It is not catastrophic forgetting.**
Forgetting is the later change in margin, exact match, or paraphrase performance
for facts learned in earlier wakes.

The self-calibrated knowledge battery and held-out perplexity are locality
controls for pre-existing capability damage. Correct-to-incorrect battery flips
are behavioral losses; answer-log-probability changes add sensitivity; PPL
catches diffuse distribution damage. These are useful, but their claim is
limited to the battery and held-out text they cover.

For a field-compatible report of the current run, use:

1. the complete floor-corrected margin matrix;
2. final average margin or cumulative margin as the ACC analogue;
3. BWT from own-wake to final margin;
4. max-to-current average forgetting, added as a report calculation;
5. installation, exact match, and paraphrase rates as behavioral secondary
   metrics;
6. knowledge-battery losses and log-probability tails plus held-out PPL;
7. training tokens, replay/generated tokens, persistent artifacts, wall time,
   and model-growth budget per arm.

The narrow defensible claim is **continual installation and retention of novel
synthetic factual bindings across wake/sleep cycles**. The present protocol
does not by itself establish general continual language learning, broad
knowledge editing, or deployment-wide absence of catastrophic forgetting.

## 9. Minimum reading path

For the shortest route through the field:

1. McCloskey and Cohen 1989 — the failure;
2. Robins 1995 — replay and pseudoreplay;
3. LwF, EWC, SI, iCaRL, GEM, and DGR — the 2016–17 method families;
4. On Tiny Episodic Memories, GDumb, and DER — the strong-baseline correction;
5. GEM and RWalk — matrix metrics, forgetting, transfer, and intransigence;
6. van de Ven et al. — Task-IL, Domain-IL, and Class-IL distinctions;
7. LAMOL, Continual-T0, CITB, and TRACE — the language-model bridge.

The repository's sleep and dream-distillation notes should be read after this
path. They are specialized continuations of replay and consolidation, not a
substitute for the field baseline.
