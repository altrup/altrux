# Research — 2026-08-21: continual-knowledge benchmark selection

## Decision

First reproduce the **QLoRA fine-tuning baseline on LAMA-CKL** from Seo, Lee,
and Yeo, *Train-Attention: Meta-Learning Where to Focus in Continual Knowledge
Learning* ([NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b111780a4a1c3beecb43b708ad7415e-Paper-Conference.pdf),
[official code and data pipeline](https://github.com/ybseo-ac/TAALM)). This is
the closest published test of the present Altrux claim: new text changes model
weights, then probes measure acquisition of those facts and retention of prior
facts without supplying the evidence at inference.

After reproducing that result, follow the paper's published sampling procedure
to create and freeze a Mamba-specific 500/500 split. The authors explicitly
recommend resampling under the same zero/one accuracy constraints for models
outside the LLaMA family. Run backbone-native LoRA fine-tuning adaptations and
Altrux on that one shared artifact. This is a benchmark-compliant evaluation on
a new backbone, not a numerical reproduction of the Llama-2 result.

Do not use OAKS for this claim. It is a strong benchmark for online
inference-time adaptation with accumulated context, RAG, and agent memory, not
for parametric learning. Do not start with CITB, Continual-T0, or TRACE: they
measure sequential task and instruction ability, not whether new factual text
becomes retained parametric knowledge.

This is a benchmark choice, not a claim that LAMA-CKL represents general
continual learning. The first reproducible claim boundary is: **on the published
LAMA-CKL protocol, the treatment changes the acquisition/retention trade-off of
factual bindings in the tested model.** It does not establish live world-model
updates, broad instruction retention, agent memory quality, or general safety.

## Target and selection rule

Altrux has a wake phase that receives text, a dream phase, and a later weight
consolidation phase. The decisive state is the model parameter state after
consolidation. Evaluation must begin from fresh recurrent state and must not
provide learned evidence in the prompt. This is **continual knowledge learning
(CKL)**: parametric acquisition, update, and retention from text. It differs
from:

- **Continual instruction/task learning:** sequential supervised NLP tasks; the
  output contract, task distribution, and often input format change.
- **Context, RAG, or agent-memory adaptation:** base weights can stay fixed
  while the system retrieves, summarizes, or retains external state at
  inference. This cannot show that consolidation installed knowledge in weights.

## Candidate comparison

### LAMA-CKL — selected

- **Venue and target.** LAMA-CKL is a NeurIPS 2024 published parametric CKL
  experiment. It tests whether a causal LM learns factual object tokens from
  evidence documents while retaining facts that it answered before the update.
  It is based on LAMA T-REx triples, not a retrieval-prompt benchmark.
  [Section 4.1](https://arxiv.org/html/2407.16920#S4.SS1) defines 500
  *to-learn* items from time-variant relations with zero initial accuracy and
  500 *not-to-forget* items from time-invariant relations with initial accuracy
  one.
- **Stream, adaptation, and metrics.** Each of 30 epochs trains on the same 500
  to-learn evidence documents, then tests object-label accuracy on descriptive
  sentences for both sets. The published metrics are Top Accuracy, epoch of Top
  Accuracy, Not-to-Forget Accuracy at that checkpoint, and Total Knowledge
  (their sum). The [protocol](https://arxiv.org/html/2407.16920#S4.SS1.SS1)
  directly measures a parametric update: no evidence is in the probe. For the
  seed-42 Llama-2-7B QLoRA fine-tuning run, the published checkpoint target is
  0.1150 Top Accuracy at epoch 16, 0.8174 Not-to-Forget Accuracy, and 0.9324
  Total Knowledge.
- **Models, compute, and reproduction.** The main result uses Llama-2-7B with
  QLoRA or K-Adapter; it also reports TinyLlama-1.1B. Update evaluation uses
  eight 24-GB RTX 3090 GPUs, batch 64, 30 epochs, and reports 25 minutes for the
  time. QLoRA uses rank 64, alpha 16, NF4/BF16 and expands 160M parameters on
  Llama-2-7B. [Hardware](https://arxiv.org/html/2407.16920#A1.SS1) and
  [adapter accounting](https://arxiv.org/html/2407.16920#A1.SS3) are published.
  This fits a rented multi-GPU calibration. Full TAALM meta-training needs a
  separate six-hour run on one A100 82-GB GPU, but is not needed for the
  selected QLoRA baseline reproduction.
- **Release and access.** The official repository supplies scripts, data
  pipeline, and benchmark material, but declares no repository license. The
  pipeline downloads [LAMA](https://github.com/facebookresearch/LAMA); the
  backbone uses the Llama 2 community license. Review both before reuse or
  redistribution.
- **Validity limits.** The paper directs users to resample both sets outside
  the LLaMA family. The zero/one filters make the test model-conditioned. The
  same 500 facts repeat for 30 epochs, so this is controlled repeated update,
  not a natural high-volume stream. Selecting Top Accuracy on the test curve is
  optimistic unless a checkpoint rule is fixed without that curve. Relation
  labels are only a proxy for whether each fact changed in the world. These are
  limits of the published protocol, not a reason to build a local replacement.
  It also repeats one to-learn cohort for 30 epochs instead of presenting a
  sequence of new cohorts. It measures the stability--plasticity trade-off, but
  does not by itself establish retention across multiple knowledge updates.

### TemporalWiki — second benchmark, not the first reproduction

- **Venue and target.** TemporalWiki is an EMNLP 2022 published parametric
  temporal-language-modeling benchmark. It continually pretrains on Wikipedia
  changes and probes factual knowledge from Wikidata without retrieved evidence.
  The [paper](https://aclanthology.org/2022.emnlp-main.418.pdf) defines
  TWiki-Diffsets from consecutive English Wikipedia snapshots and TWiki-Probes
  from matching Wikidata snapshots, split into Changed and Unchanged facts.
- **Stream, protocol, and metrics.** The paper starts from an August 2021
  updated GPT-2 Large, then makes four monthly updates through December. FULL
  trains each full snapshot once; DIFF trains each diff once (about 4.6B versus
  347M token updates per month). It compares DIFF, RecAdam, Mix-Review, LoRA,
  and K-Adapter. Evaluation is proper-noun perplexity on Changed and Unchanged
  probes after every update, plus non-diff perplexity.
  [Baseline details](https://aclanthology.org/2022.emnlp-main.418.pdf#page=5)
  are published.
- **Models, release, and feasibility.** The base is GPT-2 Large (774M). Official
  [code](https://github.com/joeljang/temporalwiki),
  [dataset generator](https://github.com/joeljang/TemporalWikiDatasets), and
  five August--December snapshot artifacts are public. The repository specifies
  an old Python 3.8 / PyTorch 1.9 CUDA environment and has no declared license.
  A faithful run is much larger than LAMA-CKL: the initial August update is
  about 546k steps; each FULL update is 140k steps. DIFF is cheaper, but still
  is not the first available-hardware calibration.
- **Validity limits.** The authors state that Wikipedia and Wikidata edits need
  not be world-fact changes, and knowledge deletion is absent. Probe phrases are
  synthetic subject--relation--object strings; high perplexity led to light
  tuning. A later NeurIPS analysis found Diffsets contain evidence for both
  Changed and Unchanged probes, so both scores can move together instead of
  showing a clean plasticity/stability trade-off. See the original
  [limitations](https://aclanthology.org/2022.emnlp-main.418.pdf#page=8) and
  [later analysis](https://arxiv.org/html/2407.16920#S4.SS3).

### OCKL — correct target, weak reproduction package

- **Venue and target.** *Online Continual Knowledge Learning for Language
  Models* is an arXiv preprint, not a verified archival venue at this review.
  It is explicitly parametric: at each time step, text updates model weights and
  a QA stream evaluates the updated model. Its
  [formulation](https://arxiv.org/html/2311.09632#S3.SS1) requires one pass
  under a real-time constraint.
- **Stream, metrics, and models.** It derives dated Wikidata knowledge and QA
  streams for 2019--2023. The knowledge stream has 94,568 examples; the
  non-redundant stream has 1,929,045 texts. It reports Exact Match, BWT, FWT,
  Knowledge Acquisition Rate, and Knowledge Gap. Its baselines are vanilla
  continual pretraining, RecAdam, Mix-Review, LoRA, K-Adapter, Modular, and
  distillation, on T5-base and T5-large (about 222M and 737M parameters).
  [Data](https://arxiv.org/html/2311.09632#S3.SS3),
  [metrics](https://arxiv.org/html/2311.09632#S3.SS4), and
  [models](https://arxiv.org/html/2311.09632#S4.SS1) are specified.
- **Release, feasibility, and validity limits.** The paper links no official
  code or data repository. Thus no released artifact or license permits a
  faithful run; reconstruction from raw Wikidata would be a new local
  benchmark. Knowledge Gap is distance between selected internal
  representations, not an externally validated fact score. The large stream is
  also costly. Exclude OCKL for reproducibility, not because its target is
  wrong.

### OAKS — exclude from parametric-consolidation evaluation

- **Venue, target, and data.** OAKS is ACL 2026 Main, with official
  [code](https://github.com/kaistAI/OAKS) and
  [datasets](https://github.com/adobe-research/OAKS). It tests online
  adaptation over sequential **context chunks**. OAKS-BABI has 1.2k questions,
  128k-token contexts, 65 chunks, and 4.7 answer changes per question;
  OAKS-Novel has curated novel contexts and 870 multiple-choice questions.
  [Dataset details](https://arxiv.org/html/2603.07392#S3.SS1) are published.
- **Protocol, metric, and scale.** At each interval, the model receives all
  prior chunks and answers the same questions. The score is interval-level
  accuracy. The paper evaluates 14 models with concatenated context, RAG, and
  agent-memory systems, mostly on eight A100 80GB GPUs. The base run takes up
  to 128k context tokens and has a rolling-window switch.
  [Evaluation](https://arxiv.org/html/2603.07392#S3.SS2) and
  [inference methods](https://arxiv.org/html/2603.07392#S4) show no
  weight-update training protocol.
- **Access and validity limits.** Code and data repositories declare no
  license. The 128k context run is expensive, and its result measures
  long-context state tracking and retrieval quality. It cannot support a claim
  about Altrux weight consolidation. OAKS-BABI is synthetic; OAKS-Novel is
  narrative multiple choice. There is also a release-version conflict: the
  paper reports 870 OAKS-Novel questions, while the current official code
  repository lists 435. Pin a data revision before any future inference-memory
  replication. Use it only later for inference-time memory.

## Why instruction benchmarks are not the first target

[CITB](https://aclanthology.org/2023.findings-emnlp.633/) is a published
continual-instruction benchmark with InstrDialog and InstrDialog++ task streams.
[Continual-T0](https://aclanthology.org/2022.emnlp-main.410/) adds eight
language-generation tasks while retaining performance on 70 datasets.
[TRACE](https://arxiv.org/abs/2310.06762) tests aligned models across
heterogeneous instruction datasets plus general ability, instruction following,
and safety. These are useful later checks for aligned-behavior damage. Their
training signal is supervised task performance, not a new-text corpus that must
become parametric factual knowledge. They answer another question. CITB has
official [code and data](https://github.com/hyintell/CITB); Continual-T0 has
official [code](https://github.com/ThomasScialom/T0_continual_learning). TRACE
is a preprint at the cited version; do not use it as a substitute without a
published release.

## Faithful reproduction gate

Reproduce the published **Llama-2-7B QLoRA fine-tuning LAMA-CKL baseline**
before an Altrux treatment. Faithful means all of the following stay fixed:

- official LAMA source, pipeline, 500/500 Llama-2-specific splits, evidence
  documents, descriptive probe form, and object-token accuracy;
- Llama-2-7B, QLoRA configuration, quantization, optimizer, learning rate,
  batch size, maximum length, 30 epochs, and seed policy;
- both-set evaluation after every epoch and the four published metrics;
- the published Top Accuracy checkpoint rule, while explicitly identifying it
  as test-curve selection. Report full curves and five-seed dispersion, not
  only the selected peak.

For the later Mamba comparison, change the backbone once before defining the
arms. Then vary only the **update treatment** across those arms. Keep the fixed
benchmark source, sampling rules, probe form, scorer, and no-evidence-at-test
rule. The present Altrux treatment is implemented for Mamba recurrent state,
so construct the Mamba split through the official model-conditioned pipeline,
as the paper requires for a non-LLaMA model. Freeze and hash that artifact
before any arm runs. Every Mamba arm uses the same base checkpoint and split.
Do not add local facts, templates, probes, thresholds, or scorers.

Keep the two claims separate: the Llama-2 run is the faithful reproduction;
the Mamba run is a benchmark-compliant cross-backbone evaluation. Do not use
the published Llama-2 numbers as the Mamba control.

Minimum later arms: frozen no-update; a backbone-native LoRA fine-tuning
adaptation; a backbone-native LoRA Mix-Review adaptation; and Altrux. These are
cross-backbone adaptations, not published QLoRA configurations. Mix-Review
review tokens count as training work and persistent data. The frozen arm checks
the construction floor; it is expected to score 0/1 by selection, so it is not
a competitive learner. The published oracle is an optional ceiling, not a fair
method arm because it has token labels unavailable to normal methods.

For every arm, report source tokens, replay/generated tokens, update epochs and
optimizer steps, trainable and total parameters, adapter/replay/dream artifact
bytes, peak VRAM, GPU model/count, elapsed GPU-hours, and checkpoint-selection
data. Match source-token and update budgets, or label the result an efficiency
comparison rather than a method comparison. Disable context and retrieval during
the factual probe.

## Decision risks and stopping condition

The main risk is model-conditioned data selection. A successful Llama-2 result
does not automatically transfer to Altrux's architecture. The first calibration
is the faithful published baseline on its published backbone. The second is the
frozen Mamba split: it must contain 500 eligible to-learn and 500 eligible
not-to-forget items under the published rules before treatment comparison. If
the base model cannot supply those sets, stop rather than relax the benchmark.
A second risk is narrow fact-template coverage; report no claim beyond these
relations unless TemporalWiki or another released parametric benchmark is added
unchanged.

No bespoke benchmark is proposed. Before the run, derive and freeze a
reproduction tolerance from the published five-run statistic. If the selected
published baseline misses that tolerance using the official artifact, stop
before comparing Altrux. That is an environment or artifact calibration failure,
not evidence for or against the method.
