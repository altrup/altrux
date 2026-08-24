# Research — 2026-08-23: modern LLM continual-learning evaluations

## Decision

Use published LAMA-CKL as the first controlled Altrux test. It tests
parametric acquisition and retention of factual object labels. It is not a
general continual-learning benchmark.

Later Altrux claims need public, non-bespoke evaluations on separate learning
surfaces: temporal or domain pre-training, continual instruction or task
learning, and aligned conversational behavior. An inference-time memory test is
also useful, but it is not evidence that knowledge entered model weights.

## 1. What is common practice

There is no single standard LLM continual-learning suite. The shared practice
is a fixed ordered stream, an update after each stage, and evaluation of the
new stage and earlier stages after every update. The primary artifact is a
per-stage curve or performance matrix, not only a final score.

For a task matrix, report final average performance, backward transfer (BWT),
and, where defined, forward transfer (FWT). Also report max-to-current
forgetting when a task can improve and later collapse. This is the common
continual-learning measurement described in the repository's
[foundations review](RESEARCH-20260814-continual-learning-foundations.md).

LLM studies add a second surface. They test inherited behavior outside the
stream, such as general knowledge, instruction following, or safety. This is
important, but the selected external suite bounds only that suite.

The details below are benchmark-family rules, not universal rules. In
particular, LAMA-CKL uses a selected zero/one fact cohort and a peak checkpoint
score. Those are useful controls for its experiment. They are not the normal
definition of continual-learning forgetting.

## 2. Taxonomy and representative evaluations

| Learning surface | Representative published study or benchmark | Data and stream | Learning target and output | Retention measurement | Release and feasibility | Main validity limit |
| --- | --- | --- | --- | --- | --- | --- |
| Continual factual knowledge | [LAMA-CKL, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b111780a4a1c3beecb43b708ad7415e-Paper-Conference.pdf) | 500 *to-learn* evidence documents repeat for 30 epochs. A separate 500-item *not-to-forget* cohort stays fixed. | A causal LM learns object-label tokens from evidence. It predicts the object tokens in a descriptive factual sentence, with no evidence in the probe. | To-learn accuracy measures acquisition. Not-to-forget accuracy at the to-learn peak measures retention. The paper also reports top accuracy, its epoch, and their sum. | The [official TAALM repository](https://github.com/ybseo-ac/TAALM) supplies the conversion pipeline and scripts. It has no declared repository license. The published Llama-2-7B QLoRA run used eight 24-GB GPUs, but the controlled update is small. | One repeated cohort is not a stream of new cohorts. The split is model-conditioned. Peak selection on the test curve is optimistic unless fixed before the run. It covers a narrow factual form. |
| Continual factual and temporal knowledge | [TemporalWiki, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.418/) | Consecutive 2021 English Wikipedia snapshots provide full corpora and monthly Diffsets. Matching Wikidata snapshots provide Changed and Unchanged probes. | GPT-2 Large continues causal-LM pre-training on full snapshots or Diffsets. The probe scores proper-noun perplexity. | Changed probes measure update. Unchanged probes and non-diff text measure retention after each monthly update. | [Code](https://github.com/joeljang/temporalwiki) and [dataset construction code](https://github.com/joeljang/TemporalWikiDatasets) are public. The released stream is large: each Diffset is about 347M tokens, and the project uses an old CUDA stack. | Wikipedia and Wikidata edits are proxies for world change. Evidence can overlap Changed and Unchanged probes. Deletion is not tested. Proper-noun perplexity is not direct answer accuracy. |
| Continual instruction and task learning | [CITB, Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.633/) and [Continual-T0, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.410/) | CITB processes Super-NaturalInstructions into a 19-task dialogue stream and a 38-task mixed stream. Continual-T0 starts from T0, then adds eight text-generation tasks while retaining inherited task suites. | A sequence-to-sequence instruction model learns supervised input-to-output tasks. Outputs are task-valid text, not factual-label tokens. | Per-task sequential scores support task matrices, BWT, FWT, initial-task retention, and unseen-task transfer. Continual-T0 also checks old and zero-shot task sets. | CITB has [official code and processed data](https://github.com/hyintell/CITB). Continual-T0 has [official code](https://github.com/ThomasScialom/T0_continual_learning). The official CITB 19-task stream is practical after an interface port. | Task metrics differ across generation tasks. A task can be contaminated by pre-training. Task identity, instruction form, replay access, and selection protocol can change the result. CITB's dialogue tasks are supervised text-to-text tasks, not live adaptive conversations. These tests do not show that text evidence became new parametric facts. |
| Continual pre-training, domain adaptation, and temporal adaptation | [TiC-LM, ACL 2025](https://aclanthology.org/2025.acl-long.1551/) | TiC-CommonCrawl spans 114 monthly dumps from May 2013 through July 2024, with up to 2.9T available tokens. Dynamic evaluations cover Wikipedia, Stack Exchange, and code documentation; static tasks measure inherited ability. | A causal LM first trains on an initial month and then continues pre-training under fixed per-period token budgets, with or without replay. | Per-period scores support backward transfer, forward transfer, in-distribution performance, and static/dynamic capability curves. | The [official code](https://github.com/apple/ml-tic-lm) is public. The paper reports more than 150 experiments, 220B–440B-token main training settings, and a 2.9T-token available stream. A faithful main run is not a near-term local experiment. | Compute scale can dominate method choice. Common Crawl time is a distribution proxy, not a clean sequence of explicit facts or tasks. Dynamic domains age at different rates, so retaining older data is not always beneficial. |
| Aligned-model capability retention | [TRACE, arXiv preprint 2023](https://arxiv.org/abs/2310.06762), submitted to NeurIPS 2024, not an archival proceedings paper at this review | Eight sequential instruction datasets cover domains, Chinese and German tasks, code, and mathematics. The authors also test general ability, instruction following, and safety after training. | An aligned chat model learns each supervised task. Output is task text or code. The external probes cover five general-ability dimensions plus judge-scored instruction following and safety. | Target-task performance and BWT measure the stream. General Ability Delta, Instruction-Following Delta, and Safety Delta measure change from the starting aligned model. | The [official repository](https://github.com/BeyonderXX/TRACE) releases code under Apache-2.0 and processed-data instructions. It supports LoRA and full tuning, but uses an old CUDA stack and a GPT-4 judge for part of evaluation. | This is the best released direct evaluation found for post-update aligned behavior, but it is a preprint. A GPT judge and one safety data source do not certify safety. No archival, dedicated continual-alignment benchmark was located. |
| Online context and inference-time memory — separate from weight learning | [OAKS, ACL 2026](https://aclanthology.org/2026.acl-long.1956/) | Fine-grained context chunks update facts over time. OAKS-BABI is synthetic. OAKS-Novel uses curated narratives. | The model keeps, retrieves, or reads context at inference. It answers questions at each interval; no weight-update protocol is required. | Interval accuracy shows whether the system tracks changed facts and resists distraction. | [Code](https://github.com/kaistAI/OAKS) and [data](https://github.com/adobe-research/OAKS) are public. Long-context runs need substantial memory and compute. | This measures inference memory, retrieval, and long-context control. It cannot show parametric consolidation. The paper and current repository report different OAKS-Novel counts; pin a revision before use. |

### 2.1 Facts and evidence in LAMA-CKL

LAMA-CKL facts do not come from local Altrux prompts or generated dreams.
They start in the public [LAMA](https://github.com/facebookresearch/LAMA)
package. Its T-REx portion is a subset of Wikidata triples; the underlying
[T-REx source](https://aclanthology.org/L18-1544/) aligns Wikidata triples with
Wikipedia abstracts. LAMA supplies the triple and its descriptive factual
sentence.

The TAALM authors convert that package through their published
[`LAMA_ckl_pipeline.ipynb`](https://github.com/ybseo-ac/TAALM/blob/master/LAMA_ckl_pipeline.ipynb).
The paper and released notebook differ on one source filter. Appendix A.1 says
to select one evidence document longer than 70 tokens, require both subject and
object, and truncate it to 512 tokens. In reviewed official revision
[`b12f344`](https://github.com/ybseo-ac/TAALM/blob/b12f344a9dbae555c239635b1c192c555bed001b/LAMA_ckl_pipeline.ipynb),
the notebook instead selects the longest `evidences[].masked_sentence` by
Python character count, replaces `[MASK]` with the object, and requires the
result to exceed 200 characters and contain the subject and object. The
training and evaluation tokenizers apply the 512-token truncation later.

The 70-token and 200-character rules are not equivalent. Treat this as a
paper/code implementation mismatch, not as two descriptions of the same test.
A reproduction must pin the source revision and released artifact or exact code
path it follows, report the mismatch, and use the same choice for every
comparable arm. Do not silently rewrite the released pipeline to match the
paper prose.

It then selects to-learn units from relations marked time-variant when the
pre-update scorer has zero object-label accuracy. It selects not-to-forget
units from time-invariant relations when it has accuracy one. Each final unit
therefore contains this evidence, a triple, and a descriptive probe sentence.
The [NeurIPS paper specifies the selection and the 500/500, 30-epoch
protocol](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b111780a4a1c3beecb43b708ad7415e-Paper-Conference.pdf).

Therefore, the exact Altrux artifact must be the output of that official
pipeline, pinned by source revision, model used for zero/one selection, seed,
and file hash. The paper, repository, and inspected notebook establish the
source chain above. Do not claim more than the official artifact records.

## 3. What LAMA-CKL does and does not establish

LAMA-CKL is a strong first control because it separates training evidence from
the factual probe and fixes an acquisition-versus-retention conflict. A positive
Altrux result can support this narrow statement:

> On the published LAMA-CKL protocol, the treatment changes the factual
> acquisition and retention trade-off for the tested model and frozen split.

It does not establish continual task learning, broad domain adaptation,
conversation quality, safety retention, or inference-time memory. It also does
not establish retention over a sequence of distinct fact cohorts.

The current governing protocol in
[the LAMA-CKL wake/dream discussion](../discussion/DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md)
is compatible with this boundary: first reproduce the published Llama-2-7B
QLoRA result; then make one Mamba-conditioned official 500/500 split; then
compare only update treatments on that split.

## 4. Staged external-benchmark roadmap

1. **LAMA-CKL reproduction gate.** Reproduce the published Llama-2-7B QLoRA
   baseline with the official pipeline, evidence, probes, scorer, 30 epochs,
   and per-epoch curves. Freeze a reproduction tolerance from the published
   multi-seed result. Then use a separately frozen Mamba-conditioned official
   split for frozen, native LoRA, Mix-Review, and Altrux arms. This is the first
   controlled test, not the final claim.
2. **CITB.** Run one published CITB stream unchanged after the factual method
   and budget are frozen. Report the benchmark task matrix, task-valid scores,
   BWT, FWT where defined, and all replay or generated-data storage. This tests
   whether the method preserves heterogeneous supervised abilities. It is not
   a factual-consolidation reproduction.
3. **TemporalWiki.** Run the released Diffset and probe artifacts unchanged.
   This adds repeated temporal corpus updates, Changed and Unchanged facts, and
   a larger source-token budget. Keep retrieval off at probe time. Report the
   full update curve, source and replay tokens, compute, and the published
   perplexity measures. Do not replace it with a local fact stream.
4. **Continual pre-training.** TiC-LM is the strongest modern public target
   found, but its main configurations require hundreds of billions of training
   tokens. Select a published configuration only when that scale is feasible;
   do not create an unvalidated local subset. Earlier archival tests such as
   [Lifelong Pretraining](https://aclanthology.org/2022.naacl-main.351/) and
   [ELLE](https://aclanthology.org/2022.findings-acl.220/) remain useful context,
   but their old stacks and weaker release packages make them poor default
   reproductions. This stage tests unlabeled corpus adaptation separately from
   instruction tuning.
5. **Aligned-behavior check.** TRACE is a useful public, non-bespoke
   supplementary check, but label it as a preprint and do not make it the sole
   safety claim. Pin its official data and judge configuration. Report target
   tasks, BWT, general-ability change, instruction-following change, and safety
   change separately. Prefer an archival public continual-alignment benchmark
   if one becomes available before this stage.
6. **Optional inference-memory branch.** Use OAKS only if Altrux exposes a
   context, retrieval, or persistent-state system at inference. Report it as
   online state tracking, not as weight consolidation, and do not combine its
   scores with the parametric roadmap.

Every stage must pin the benchmark release and data hash, preserve the official
scorer and task order, evaluate after each stage, and report seeds, checkpoint
selection data, source tokens, replay or generated tokens, persistent bytes,
trainable parameters, peak memory, and GPU-hours. A new local data source,
probe, threshold, or scorer would make a new benchmark and must not be used to
extend these external claims.

## 5. Answers future sessions can rely on

- LAMA-CKL is an archival NeurIPS 2024 parametric factual-update test. It is
  the right first Altrux control, not evidence for general LLM continual
  learning.
- LAMA-CKL facts originate in LAMA T-REx: Wikidata triples aligned with
  Wikipedia material. Its official TAALM pipeline selects model-conditioned
  500/500 cohorts and supplies the evidence/probe artifact.
- CITB is the next intended surface after the factual gate. It tests supervised
  continual instruction tuning, including dialogue tasks. TemporalWiki tests
  parametric temporal knowledge. TiC-LM tests web-scale continual pre-training.
  They answer different questions and their raw scores cannot be combined.
- TRACE has the clearest released aligned-behavior evaluation surface, but it
  is an arXiv preprint, not an archival benchmark. It cannot certify safety.
- OAKS is an archival online knowledge-stream evaluation, but it tests
  inference-time context and memory. It is not evidence that weights retained
  knowledge.
- A result needs per-stage acquisition and retention curves, matched budgets,
  and external capability checks. A final score or a custom threshold is not
  sufficient.

## 6. Open uncertainties

- No archival, dedicated continual-alignment benchmark was located. This review
  found TRACE as the public direct evaluation; its publication status must stay
  visible.
- No archival benchmark was located for learning from open-ended, adaptive
  user/assistant conversations. CITB includes dialogue tasks, but it is a
  supervised text-to-text task stream and must not be described as equivalent.
- The TAALM repository is public but has no declared repository license. Check
  the LAMA and backbone licenses before download, redistribution, or release.
- The OAKS paper reports 870 OAKS-Novel questions, while the official repository
  previously listed 435. Pin a commit and inspect the files before a run.
- A faithful port of older released code to the current ROCm environment is an
  engineering reproduction issue. Do not change a benchmark protocol to avoid
  that work.

## Sources checked

Primary and official sources checked for this note: the NeurIPS 2024
[LAMA-CKL paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b111780a4a1c3beecb43b708ad7415e-Paper-Conference.pdf)
and [TAALM release](https://github.com/ybseo-ac/TAALM); the EMNLP 2022
[TemporalWiki paper](https://aclanthology.org/2022.emnlp-main.418/) and its
official repositories; the EMNLP 2022 [Continual-T0 paper](https://aclanthology.org/2022.emnlp-main.410/);
the Findings of EMNLP 2023 [CITB paper](https://aclanthology.org/2023.findings-emnlp.633/)
and release; the NAACL 2022 [Lifelong Pretraining paper](https://aclanthology.org/2022.naacl-main.351/);
the Findings of ACL 2022 [ELLE paper](https://aclanthology.org/2022.findings-acl.220/)
and release; the ACL 2025 [TiC-LM paper](https://aclanthology.org/2025.acl-long.1551/)
and release; the TRACE [preprint](https://arxiv.org/abs/2310.06762) and
release; the ACL 2026 [OAKS paper](https://aclanthology.org/2026.acl-long.1956/)
and releases; and the original [T-REx paper](https://aclanthology.org/L18-1544/).

The repository-local sources were the notes index, the 2026-08-14 foundations
review, the 2026-08-21 benchmark-selection review, the 2026-08-23 LAMA-CKL
protocol, and their directly relevant linked discussion and prior-art notes.
