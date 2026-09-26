# Research 2026-09-26: CKL as the second external benchmark

## Decision

Keep LAMA-CKL as stage one. Register CKL (Jang et al., ICLR 2022) as stage
two of the external-benchmark roadmap, ahead of CITB. Do not build it until
the LAMA-CKL reproduction gate has run. Do not change the LAMA-CKL scorer or
split for it.

CKL answers the deployment question directly: how many old facts are lost per
new fact learned, on a real news stream, with published baselines to compare
against. LAMA-CKL is a single-paper benchmark that shares CKL's LAMA lineage
but is cheaper and has a published number to calibrate the harness on.

## 1. What CKL consists of

Source: [Towards Continual Knowledge Learning of Language Models](https://arxiv.org/abs/2110.03215),
sections 3.1, 3.2, Table 1, Table 2. Code and data:
[joeljang/continual-knowledge-learning](https://github.com/joeljang/continual-knowledge-learning).

**Training stream.** CC-RecentNews: 221,779 news articles, about 168M tokens,
collected after the base model's pretraining cutoff. The main run continues
pretraining for 4 epochs, 25k global steps, about 673M token updates. A
two-phase variant splits the corpus into two time slices.

**Probes.** Four cloze datasets, scored zero-shot by exact match on the
generated answer. F1 and precision at k are in the paper's appendix.

| Set | Size | Measures | Construction |
| --- | --- | --- | --- |
| InvariantLAMA | 17,474 | Retention of time-invariant knowledge | 28 time-invariant T-REx relations, same LAMA source as LAMA-CKL |
| UpdatedLAMA | 924 | Updating of conflicting knowledge | Crowd-sourced cloze statements whose answer differs between old and new corpora |
| NewLAMA | 797 | Acquisition of new knowledge | Crowd-sourced, expert-verified absent from the old corpus |
| NewLAMA-Easy | 11,177 | Acquisition at scale, looser definition | Paraphrased sentences from new articles, masked |

**Metric.** FUAR, Forgotten over Updated plus Acquired. It divides the drop
in InvariantLAMA exact match by the gain in UpdatedLAMA plus NewLAMA. A value
of one means one old fact lost per new fact learned. Zero means no forgetting.

**Baselines.** T5-large and GPT-2 with Vanilla, RecAdam, Mix-Review, LoRA,
K-Adapters (k=2, 3), and Modular. Per-epoch curves are reported.

## 2. How it differs from LAMA-CKL

- The stream is a real corpus, not 500 selected evidence documents.
  Acquisition is harder and the published numbers are low.
- Updating is a separate axis. LAMA-CKL has no conflicting-fact set.
- The probe split is not conditioned on the model. Every model sees the same
  items.
- Scoring is exact match on generated text, not teacher-forced object-token
  argmax.
- The main run is about 100 times the LAMA-CKL token budget.

## 3. Risks, ranked

1. **Age of the new-knowledge window.** CC-RecentNews covers 2020 to 2021.
   Any modern base model has seen that news, so NewLAMA and UpdatedLAMA stop
   measuring acquisition. Options: a base model with a cutoff before 2020, or
   a regenerated stream in the TemporalWiki style. This must be settled
   before any CKL run is scheduled. [NOT CHECKED] whether the Mamba backbones
   in `models/` have pretraining data from after 2020.
2. **Cost.** 673M token updates per arm, times seeds, times arms, plus dream
   generation for the treatment arm. Budget it against the rented-box rate
   before committing.
3. **Cycle definition for a large stream.** The wake/dream protocol defines
   a cycle over a small evidence set. Dreaming over 168M tokens of news once
   per cycle is not viable. The stream needs a chunking rule, which the
   [2026-08-23 review](RESEARCH-20260823-modern-llm-continual-learning-evaluations.md)
   already lists as unbuilt.
4. **License.** [NOT CHECKED] The repository's data license.

## 4. What transfers from the LAMA-CKL work

- InvariantLAMA is the same T-REx cloze form as the LAMA-CKL not-to-forget
  set, so the retention side needs no new data pipeline. It could also serve
  as a larger second retention probe inside LAMA-CKL runs, reported
  separately and never mixed into the published aggregates.
- The per-cycle curve, seeds, and token accounting in `report.py` carry over
  unchanged. FUAR is one extra derived column.
- New work is the exact-match generation scorer and the stream chunking rule.
