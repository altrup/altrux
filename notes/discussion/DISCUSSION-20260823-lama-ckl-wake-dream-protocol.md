# Discussion notes — 2026-08-23: LAMA-CKL wake/dream protocol

This note freezes the first Altrux treatment to implement after the benchmark
selection in
[`RESEARCH-20260821-continual-knowledge-benchmark-selection.md`](../research/RESEARCH-20260821-continual-knowledge-benchmark-selection.md).
It replaces the bespoke six-wake experiment as the next science target. The old
harness and results remain useful engineering evidence, but they do not define
the benchmark or its score.

## 1. Decision

Use the published LAMA-CKL experiment for the primary evaluation:

1. Replicate the published Llama-2-7B QLoRA baseline on one GH200 first.
2. Use the official model-conditioned sampling procedure to freeze one
   Mamba-specific 500-to-learn/500-not-to-forget artifact.
3. Compare frozen Mamba, backbone-native LoRA, Mix-Review, and Altrux on that
   shared artifact.

Use the official evidence documents as the waking input. Do not add local
facts, generated dialogues, Luna-authored text, local probes, or a custom
success threshold. Evaluation starts from fresh recurrent state and uses the
published object-token accuracy and checkpoint metrics.

The upstream paper used eight GPUs with microbatch 8, for an effective batch
of 64. The available experiment hardware is one GH200. Run the pinned authors'
trainer with one visible GPU, microbatch 8, and gradient accumulation 8. This
keeps 64 examples per optimizer update and the published number of updates per
epoch. Keep every other released argument, all 30 epochs, the official data,
metric, seed, and frozen acceptance tolerances unchanged. Call this the
**single-GH200 batch-equivalent replication**, not an exact hardware
reproduction: DDP, dropout streams, and floating-point reduction order differ.
Run this gate once with the released seed-42 sampler. Five-seed upstream
dispersion is a later strengthening, not part of this registered gate. Do not
tune the adaptation toward the released score. A failed frozen gate stops the
Mamba comparison.

The Mamba comparison starts from the recap-0.5 warm start:

```text
models/mamba2_2_7b/checkpoints/recap050/epoch-2/step-800/trainable.pt
SHA-256 226e95765f9e2c0a9fa335d5f70af8fb1d63bbf0f30c4427097b116375a11f3c
```

The root `checkpoints/epoch-2/step-800` file is the retired recap-0.8 adapter
and must not be selected by path convention.

## 2. Wake and sleep schedule

Map one published training epoch to one wake/sleep cycle. Run 30 cycles and
evaluate both official fact sets after every cycle, as the published baseline
does after every epoch.

In each wake, present the same official 500 evidence documents once in the
official frozen order. Render each document as one user turn and let the arm's
current model generate its assistant turn greedily, with a 64-token backstop.
The reply must emit the tokenizer EOS before that backstop and must not emit
`<|endofconversation|>` inside the wake. Preserve the recurrent state across
all 500 exchanges. The wake itself ends open, after the last assistant EOS.
Record every prompt and generated token exactly;
the official evidence is the controlled source exposure, while the reply is a
model process measurement and can differ after the arms' weights diverge.

Dream generation runs from copies of the open post-wake state. It does not
consume or replace the state carried to the next wake. Each dream starts from
the same open state and a one-token prompt:

```text
<|endofconversation|>
```

The model writes everything after it, the `[USER]` turn included. Generation
stops when the model emits a second `<|endofconversation|>` or when it reaches
the fixed token limit.

The state carried to the next wake is the open state plus one
`<|endofconversation|>`, fed with the pre-treatment weights for every arm. So
the carried state of the frozen, LoRA, and Mix-Review arms is the same as with
a wake that closes itself.

**Amended 2026-10-05 (altrup).** The first version of this section closed the
wake with `<|endofconversation|>` and started each dream with a fixed
instruction turn, `[USER] Dream about the preceding experience. Rehearse what
matters without copying it verbatim.[ASSISTANT] `. The instruction is removed
before any box run used it. Reasons:

- The recap-0.5 warm start was trained so that about half of the
  `<|endofconversation|>` boundaries lead to a recap of the conversation
  before them. The boundary token alone is thus an in-distribution way to
  start a rehearsal. The instruction sentence was never in the training
  corpus.
- The earlier `<|eoc|>`-prompt failures
  ([2026-08-10 23:15](../experiments/EXPERIMENT_NOTES-20260810-231500.md),
  23:45 UTC) were on the recap-⅓ adapter, with 20 dreams per candidate, which
  the same note calls noise. They do not apply to this adapter.

Known cost: about half the dreams are expected to be unrelated new
conversations. The dream diagnostics show the split. If coverage is poor, the
instruction turn is the registered fallback arm, with 100 or more dreams per
candidate.

**Amended 2026-10-10 (altrup).** The instruction turn is withdrawn as the
fallback arm: generated from a real post-wake state it was echoed, not
followed (293/300 unique dreams, 1 correct binding). The wake rendering,
wake size, dream count per cycle, backstop, and duplicate rule are
re-registered in
[`DISCUSSION-20261010`](DISCUSSION-20261010-lama-ckl-first-box-debrief.md),
which supersedes this section where they differ.

Use the existing 300-distinct-dream treatment as the initial Altrux arm: no cue
splicing, fact-aware prompt, content filter, coverage target, or regeneration
after inspection. Distil one pass over every realized dream. Cache and hash the
complete set before training. A retry is allowed only for a technical failure
that produces no valid artifact; it cannot depend on dream content.

Use the existing successful treatment settings: 512 tokens per dream,
temperature 0.7, KL temperature 1.0, AdamW at `1e-4`, and one optimizer step per
dream. Native LoRA uses one fixed seed-42 shuffled pass over the 500 evidence
documents per cycle. Mix-Review pairs those batches with the 500 retention
documents in the released fixed seed-0 review order. The Mamba replication
seeds are 42, 43, and 44; all use the same seed-42 split artifact.

Full-vocabulary teacher logits are a rolling, per-cycle training artifact, not
a permanent result. After successful distillation, retain the exact dream token
IDs and text, generation seeds, stop reasons, set hash, teacher-checkpoint hash,
batch topology, and metrics. These values and the saved teacher checkpoint can
reconstruct the logits. This keeps a 30-cycle seed from retaining about 660 GiB
of redundant full-logit caches.

Do not add a `[DREAM]` token in this experiment. An untrained token has no
meaning. Training it would require supervised dream examples, add a bespoke
dream policy, change the starting checkpoint, and require a new benchmark
split. Reconsider it only if both the boundary token and the fallback
instruction fail as invocation mechanisms.

## 3. Dream quality as a continual-learning observable

Dream quality is a secondary process measurement, not a replacement benchmark.
The official LAMA-CKL acquisition and retention scores remain primary.

Record these diagnostics for every sleep without using them to select,
regenerate, stop, or tune dreams:

- correct subject/object bindings from the official to-learn set;
- subject/object misbindings and contradictions within that set;
- overlap with the official evidence documents, including the longest copied
  span and the existing copy fraction;
- unique dream hashes and duplicate count;
- generated `<|endofconversation|>` termination rate and length distribution;
- decoded samples as the dreams are produced.

Report these values by cycle. This shows whether continued consolidation
changes the model's ability to produce useful replay. The decisive utility test
is still downstream: whether distilling the realized dreams improves the next
official LAMA-CKL acquisition/retention measurement.

Do not change the dream prompt after viewing a benchmark score. Do not choose
a checkpoint from dream diagnostics. Any later comparison with the instruction
turn or with `[DREAM]` is a preregistered ablation, not an adaptive repair to
this run.

## 4. Comparison and claim boundary

The Llama-2 run is a single-GH200 batch-equivalent replication gate. The Mamba
run is a benchmark-compliant cross-backbone comparison. Neither is an exact
numerical reproduction of the released eight-GPU result.

All Mamba arms use the same warm-start weights, split, evidence order, source
exposure, evaluation schedule, and random-seed policy. Report generated and
training tokens, optimizer steps, artifact bytes, wall time, GPU-hours, and
peak VRAM for each arm. If update-token budgets are not matched, label the
result as a method-and-resource comparison.

A positive result supports only this claim: on the published LAMA-CKL
protocol, Altrux changes the acquisition/retention trade-off for factual
bindings on the tested Mamba model. It does not establish general continual
learning or broad dream quality.

## 5. Implementation docket

Implement and commit these slices in order:

1. Pin the official TAALM commit and verify its released LAMA-CKL artifacts,
   then run the published Llama-2-7B QLoRA baseline without modifying its
   trainer or evaluator. Adapt only the pinned launcher's GPU visibility and
   accumulation count as registered above. The released notebook, not the
   paper's conflicting prose, defines artifact construction: choose the longest
   `masked_sentence` by character count, replace `[MASK]` with the object,
   require more than 200 characters plus subject and object presence, and
   apply the 512-token limit during training tokenization.
2. Add official metric reporting, per-cycle curves, artifact hashes, and the
   reproduction tolerance gate.
3. Build and freeze the Mamba-conditioned 500/500 split from the pinned
   recap-0.5 checkpoint. Preserve the published object-token metric's meaning,
   but align Mamba object tokens from the last object character span because
   GPT-NeoX tokenizes a standalone object differently from the same text inside
   the descriptive sentence. Record this cross-tokenizer adaptation in the
   artifact manifest; do not apply it to the upstream Llama reproduction.
4. Add the evidence-document wake path and verify its exact decoded structure
   and recurrent-state continuity.
5. Add the `<|endofconversation|>` dream prompt and the separate wake-closing
   step, generated-token masking, stopping, caching, and technical-failure
   rules.
6. Generalize existing dream diagnostics from synthetic code bindings to the
   official LAMA subject/object records. Keep them read-only with respect to
   generation and training.
7. Add the frozen, native-LoRA, Mix-Review, and Altrux Mamba arms with complete
   resource accounting.
8. Run local fake-model tests and a short real-model smoke before any paid
   30-cycle run.

The first paid Altrux comparison remains blocked until the batch-equivalent
baseline passes its frozen tolerance and the Mamba split satisfies the official
zero/one selection rules.
