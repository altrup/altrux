# Discussion notes — 2026-08-23: LAMA-CKL wake/dream protocol

This note freezes the first Altrux treatment to implement after the benchmark
selection in
[`RESEARCH-20260821-continual-knowledge-benchmark-selection.md`](../research/RESEARCH-20260821-continual-knowledge-benchmark-selection.md).
It replaces the bespoke six-wake experiment as the next science target. The old
harness and results remain useful engineering evidence, but they do not define
the benchmark or its score.

## 1. Decision

Use the published LAMA-CKL experiment for the primary evaluation:

1. Reproduce the published Llama-2-7B QLoRA baseline first.
2. Use the official model-conditioned sampling procedure to freeze one
   Mamba-specific 500-to-learn/500-not-to-forget artifact.
3. Compare frozen Mamba, backbone-native LoRA, Mix-Review, and Altrux on that
   shared artifact.

Use the official evidence documents as the waking input. Do not add local
facts, generated dialogues, Luna-authored text, local probes, or a custom
success threshold. Evaluation starts from fresh recurrent state and uses the
published object-token accuracy and checkpoint metrics.

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
all 500 exchanges. Append one `<|endofconversation|>` token only after the
complete wake to close it. Record every prompt and generated token exactly;
the official evidence is the controlled source exposure, while the reply is a
model process measurement and can differ after the arms' weights diverge.

Dream generation runs from copies of the intact post-wake state. It does not
consume or replace the state carried to the next wake. Each dream starts from
the same post-wake state and this fixed, content-free transition:

```text
<|endofconversation|>
[USER] Dream about the preceding experience. Rehearse what matters without copying it verbatim.
[ASSISTANT]
```

The first `<|endofconversation|>` in this display is the wake-closing token.
The role markers and their literal spaces use the model's registered chat
format. Generation stops when the model emits `<|endofconversation|>` or when
it reaches the fixed token limit.

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

`<|endofconversation|>` and the instruction have separate purposes. The token
closes waking. The instruction selects dream generation. The warm start also
learned that `<|endofconversation|>` can lead to an unrelated conversation, so
the boundary token alone does not specify the required mode.

Do not add a `[DREAM]` token in this experiment. An untrained token has no
meaning. Training it would require supervised dream examples, add a bespoke
dream policy, change the starting checkpoint, and require a new benchmark
split. Reconsider it only if the fixed instruction fails as an invocation
mechanism.

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

Do not change the instruction after viewing a benchmark score. Do not choose a
checkpoint from dream diagnostics. Any later comparison with no instruction or
with `[DREAM]` is a preregistered ablation, not an adaptive repair to this run.

## 4. Comparison and claim boundary

The Llama-2 run is a faithful reproduction gate. The Mamba run is a
benchmark-compliant cross-backbone comparison. It is not a numerical
reproduction of the Llama-2 result.

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
   then reproduce the published Llama-2-7B QLoRA baseline without locally
   rewriting its trainer or evaluator. The released notebook, not the paper's
   conflicting prose, defines artifact construction: choose the longest
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
5. Add the wake-closing `<|endofconversation|>` plus fixed dream instruction,
   generated-token masking, stopping, caching, and technical-failure rules.
6. Generalize existing dream diagnostics from synthetic code bindings to the
   official LAMA subject/object records. Keep them read-only with respect to
   generation and training.
7. Add the frozen, native-LoRA, Mix-Review, and Altrux Mamba arms with complete
   resource accounting.
8. Run local fake-model tests and a short real-model smoke before any paid
   30-cycle run.

The first paid Altrux comparison remains blocked until the faithful baseline
passes its frozen reproduction tolerance and the Mamba split satisfies the
official zero/one selection rules.
