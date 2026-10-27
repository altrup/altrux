# Research notes — 2026-10-24: learned dreamers (ReST-EM, gradient matching, forgetting-aware rewards)

Background for the dreamer track registered in
[the 2026-10-24 fifth box debrief](../discussion/DISCUSSION-20261024-lama-ckl-fifth-box-debrief.md)
§3.2. Mechanics and literature only; decisions live in the debrief. SEAL was
read this session from the paper's HTML (`arxiv.org/html/2506.10943v2`), and
its numbers below are quoted from it. Every other citation is from memory
with its arXiv id; verify the id before quoting a number from it.

## 1. The problem a learned dreamer solves

Self-distillation from the model's own dreams installs whatever the dreams
consistently contain. Five boxes of LAMA-CKL showed the dreams contain the
wake's shape far more consistently than its facts (frame, then reply;
debriefs of 2026-10-13 and 2026-10-24). A learned dreamer is a training
stage that changes which dreams the model produces, chosen by what the
dreams do when learned from, so that the loop has something worth
distilling. The loop itself (wake, dream from the boundary, empty-state
student, one step) is unchanged.

## 2. SEAL — Self-Adapting Language Models (Zweiger et al., MIT, arXiv 2506.10943, 2025)

One model, two nested loops, a single prompted format.

**Inner loop (the learning step).** With passage C in context, the model
writes a "self-edit" SE: free text under the prompt `Let's read the
following passage and produce a list of implications derived directly or
indirectly from the content. / Passage: {passage} / Implications:`. Plain
text, no special token, no chat template stated. The output is split on
newlines into training documents; a LoRA (rank 64, alpha 128, all MLP and
attention projections, lr 3e-4, 2 epochs, batch 10 in the RL stage) is
trained on them with ordinary next-token cross-entropy, **no passage in
context**. The adapted model θ' answers C's SQuAD questions with no passage;
GPT-4.1 grades yes/no. That accuracy is the reward. θ' is discarded.

**Outer loop (ReST-EM, §3 below).** Per round: 50 passages × 5 self-edits
at temperature 1 × 3 seeds; keep the self-edits whose reward beat the
no-edit baseline; SFT the model (LoRA) on (passage → kept self-edit).
Two rounds. PPO and GRPO "were unstable" in their hands.

**Results, Qwen2.5-7B, no-context SQuAD accuracy (%).**

| setting | base | passage only | untrained self-edits | SEAL r1 | SEAL r2 | GPT-4.1 implications |
|---|---|---|---|---|---|---|
| one passage, fresh LoRA each | 32.7 | 33.5 | 39.7 | 43.7 | 47.0 | 46.3 |
| 200 passages, one LoRA | 32.7 | 36.0 | — | — | 58.2 | 59.4 |

Qwen2.5-3B, one passage: 25.1 base, 31.9 untrained self-edits, 37.0 SEAL.
Prompt-format ablation (Table 10, before RL → after round 2): implications
39.7 → 47.0; implications-very-long 45.0 → 52.1; **no-prompt (the raw
continuation after the passage) 13.8 → 18.9**. The instruction is worth
about 25 points, the RL about 7.

Few-shot ARC, Llama-3.2-1B-Instruct, 8 tasks filtered to be solvable
under test-time training: in-context 0 %, untrained self-edits 20 %, SEAL
72.5 %, oracle 100 %.

**Sequential edits (their §5, Figure 6).** About eight passages, each
triggering a self-edit and an update on the same model, re-scoring earlier
passages after each: accuracy on earlier passages "gradually declines";
"without complete collapse". Numbers only in the figure. Whether the
updated model writes the next self-edit is implied by "we sequentially
update the model" and never stated; whether LoRAs are merged or stacked is
not stated.

**What they did not measure.** Generation quality after edits, repetition
or degenerate output, general-capability benchmarks before or after RL.
"We do not explicitly optimize for retention in our current training
setup." No KL penalty, no length cap, no fluency check, no early stopping.
Appendix B.11: RL "appears to have dramatically increased the length" of
self-edits. The protection they had was structural: ReST-EM fine-tunes only
on the model's own fluent samples, in two small rounds, behind a fixed
instruction that production conversation never uses.

**Their stated limitations.** 30–45 s per reward (fine-tune + eval +
grader); every training passage needs paired questions, which "prevents RL
training of SEAL from scaling to unlabeled corpora" (proposed fix: the
model writes its own questions); catastrophic forgetting.

**Deploy.** No questions. Passage in, self-edit out, LoRA kept. The headline
numbers are deploy numbers on a held-out split.

## 3. ReST-EM (Singh et al., DeepMind, arXiv 2312.06585, 2023)

Reinforced Self-Training with Expectation-Maximisation: sample several
outputs per input from the current policy; keep those whose reward clears
a threshold (E-step); supervised fine-tune the policy on the kept pairs
(M-step); repeat. Also called rejection-sampling fine-tuning or filtered
behaviour cloning; STaR (Zelikman et al., arXiv 2203.14465) is the same
loop with a correctness check. No value network, no ratio clipping, no
reward model. Properties: amplifies only what the current policy sometimes
produces (needs a nonzero hit rate to start); converges fast to an output
style with diminishing returns after two or three rounds; cannot leave the
language manifold because every target is the model's own sample.

The threshold is not essential. Best-of-K per input with no threshold is
the same loop (the kept sample is on average better than the policy's
mean whenever the reward is alive, so the M-step moves the policy the
right way), and a dead reward then shows as a vanishing gap between the
best sample's score and the mean, a reading rather than a rule. The soft
form is filtered behaviour cloning with weights: fine-tune on every sample
with its loss weighted by its standing in the group (for example
max(0, 1 + d/σ) with d the sample's advantage over a reference and σ the
group's spread), which drops nothing and has no constant. GRPO is the
policy-gradient relative of that (advantage relative to the group mean,
normalised by its spread), and is what SEAL found unstable.

## 4. First-order reward: gradient alignment

For one SGD step of size α on a dream d, the change in any loss L is, to
first order,

```
ΔL ≈ −α · ⟨ ∇_θ L_in(d) , ∇_θ L ⟩
```

With L the loss of the conversation's continuation (its answers given its
questions, or its later turns, rendered alone from an empty state; loss on
the assistant tokens), the step's predicted gain is the inner product
above. This is the quantity SEAL's reward measures by actually fine-tuning
and evaluating, computed instead from one backward per dream. Under Adam
the realised first step is the per-coordinate sign of the gradient, not
the gradient, so the inner product describes an SGD step; see the debrief
§3.4 for the optimizer decision.

**Cosine or projection.** The cosine cos(∇L_in, g) scores direction only;
the inner product α⟨∇L_in, g⟩ scores direction times size, and is the
first-order predicted drop in L, in nats, after an SGD step of size α
(dividing by ‖g‖, as the 10-24 draft did, puts the term in a unit no
other term shares; the debrief §3.2 keeps every term in nats). Under SGD the step scales with
‖∇L_in‖, so two dreams of equal direction and different size make
different progress and only the projection sees it. The cosine's virtue
is that magnitude cannot be gamed (longer dreams, higher-surprise tokens
have larger gradients); that matters when the dreamer's logits are
optimised directly (§6) and not when selecting among the policy's own
samples, where the only magnitude lever is length. The debrief §3.2 uses
the projection for selection and keeps the cosine for the dense term.

**Second-order correction.** The expansion to second order is

```
ΔL ≈ −α · ⟨G, g⟩ + ½ α² · Gᵀ H G
```

with H the Hessian of L. A step that is too large stops paying off. With
α²H approximated by ακ·I, maximising the predicted drop means maximising
α⟨G, g⟩ − ½ακ‖G‖². Setting κ = α makes this −½‖αG − g‖² up to a
constant: the squared distance between the step and the target gradient,
i.e. gradient matching in the literal sense. The distance form has an
arbitrary target size (g is a gradient, not the displacement that would
learn the continuation); the fitted-κ form has a measured one (least
squares of the residual, measured drop minus α⟨G, g⟩, on −½α‖G‖² over a
set of actual steps). Both score a zero gradient at zero, so neither
rewards the empty dream.

**Contrast target.** When the continuation's replies share a shape (SQuAD
answers: a short span, then a stop), g carries that shape as a component
every reply has, and any dream that teaches the shape earns part of the
inner product. The fix is a difference of gradients: g = ∇L(C) − ∇L(C⁻),
with C⁻ the same question answered with another row's reply. The shared
component cancels and the content component remains ("Rollo, not
Lamarck"). One extra backward per row. The same construction on an unanswerable
question, ∇L(abstention) − ∇L(guess), points from guessing toward
abstaining on every such row alike: a consistent reply-shape direction,
which is why the debrief keeps abstention on the hinge side (protected,
never paid for) rather than as a learning target.

Gradients add, so for a batch B whose loss is the mean over its n dreams
the batch gradient is G_B = (1/n) Σ_{i∈B} ∇ᵢ and the first-order term is
(α/n) Σ_{i∈B} ⟨∇ᵢ, g⟩, a sum of per-dream scalars; the quadratic term
needs the Gram matrix ⟨∇ᵢ, ∇ⱼ⟩ over the pool. Choosing the best batch of n from a
pool of K·n is then arithmetic on those numbers: greedy forward selection
is exact for the additive part and a swap pass handles the one bend the
hinge (§5) adds.

Lineage: dataset condensation by gradient matching (Zhao, Mopuri, Bilen,
arXiv 2006.05929, 2021) synthesises data whose training gradient matches
the real data's; LESS (Xia et al., arXiv 2402.04333, 2024) selects training
examples by gradient similarity to a target task. The dreamer reward is
LESS's score with the model as the data source and one document's answers
as the target.

## 5. Forgetting term: A-GEM as a selection criterion

GEM (Lopez-Paz and Ranzato, arXiv 1706.08840, 2017) and A-GEM (Chaudhry
et al., arXiv 1812.00420, 2019) constrain each update to the half-space
⟨∇, g_old⟩ ≥ 0, where g_old is the gradient of the loss on a reference
batch of old knowledge; A-GEM uses one random reference batch and projects
the step when the constraint fails. As a reward term,

```
r(B) = α⟨G_B, g_new⟩ − max(0, −α⟨G_B, g_old⟩)
```

penalises only batches whose step would raise the old loss; with both
terms in the same unit (nats of predicted loss change after the step)
the penalty's weight is 1 and not a knob. The 10-24 draft divided each
term by its own ‖g‖, which hid a weight of ‖g_new‖/‖g_old‖ in the hinge.
A reference row with no slope contributes nothing to g_old, and the
quadratic term of §4 does not protect it either: its κ is fitted on the
new row's loss and measures how fast learning overshoots, not the
curvature of old rows. Flat old facts have no first-order guard; the
retention measurement is what sees the damage. g_old is the gradient of a mean over rows of
per-row per-token means, so a four-token fact and a 500-token
conversation weigh the same; a token-pooled mean would let the long rows
hide the facts the hinge exists for. A-GEM draws its one reference batch at random over all old tasks, so
"old" can be several things at once: old facts, abstention replies, the
model's conversational behaviour, and its own good dreams (the previous
round's kept dreams in their wake context). The debrief holds one hinge
per source rather than one on the pooled batch: pooled, a step that
raises one source's loss and lowers another's by more passes unpenalised,
and the kept-dream source exists to guard dreaming itself. The last
makes the hinge a first-order guard against the step that damages
dreaming itself, the failure the fifth box measured, at no extra cost.
Known limitations carry over: a single reference batch is a noisy direction
in a high-dimensional space (most dream gradients have near-zero cosine
with it, so the hinge fires on blatant interference only), and the
first-order per-step view does not see cumulative drift. The hinge is
one-sided: it rewards nothing for lowering the old loss, so a dream that
only rehearses earns about zero and enters a batch only as a repair beside
an interfering dream. A two-sided term would reward rehearsal directly, at
the cost of a dreamer that can earn reward by ignoring the new row. The degenerate optimum of any such pair is a zero-gradient dream (nothing
learned, nothing forgotten); the learning term blocks it only on rows
where some dream helps. On a row where none does, the best batch *is* the
smallest-gradient one, so ReST-EM's "keep the best" would train toward
document-free dreams at the share of such rows; the debrief weights each
kept batch's SFT by its positive score over the round's mean positive
score, so dead rows teach nothing.

Batch-level scoring is what lets rehearsal emerge: a dream that installs
the new fact and interferes, beside a dream that rehearses the interfered
fact, passes as a pair and fails one at a time. This is pseudorehearsal
(Robins 1995) found by the reward rather than imposed.

## 6. Dense term: gradient matching through the dreamer's logits

Let π_t be the dreamer's distribution at dream position t (the teacher
logits the cycle already records) and let the student's inner loss use π
as soft targets (the cycle's KL loss up to a constant):

```
L_in(θ; π) = Σ_t Σ_v π_t(v) · [ −log p_θ(v | d_<t, ∅) ]
```

Then with g = ∇_θ L_QA and one SGD step of size α on L_in, the change in
L_QA to first order is ΔL_QA ≈ −α⟨∇_θ L_in, g⟩, and

```
∂(ΔL_QA) / ∂π_t(v) = −α · gᵀ ∇_θ[−log p_θ(v | d_<t, ∅)] = α · gᵀ ∇_θ log p_θ(v | d_<t, ∅)
```

**Sign convention** (corrected 2026-10-25; the 10-24 draft had it
inverted): a token is helpful when raising its probability lowers L_QA,
i.e. when ⟨∇_θ log p_θ(v | d_<t, ∅), g⟩ < 0. Define the per-token score
with that sign, r_t(v) = −⟨∇_θ log p_θ(v | d_<t, ∅), g⟩, positive for
helpful tokens. The dreamer loss is the negative expected score under its
own distribution, and the vocabulary sum collapses:

```
L_gm = −Σ_t Σ_v π_t(v) r_t(v) = −⟨ ∇_θ L_in(θ; π_θ) , g ⟩
```

i.e. −cos(∇_θ L_in(θ; π_θ), g) in the normalised form, the same quantity
§4 scores for selection, now differentiated.

**Why a second derivative.** The loss is a function of a gradient,
ℓ = −cos(G(π), g) with G(π) = ∇_θ L_in(θ; π). Training the dreamer needs
dℓ/dπ = (∂ℓ/∂G)(∂G/∂π), and ∂G/∂π = ∂²L_in/∂θ∂π is a mixed second
derivative. There is no route from ℓ to π except through G; if G is
computed and detached, ℓ is a constant and the dreamer gets nothing. The
same fact read the other way: computing r_t(v) for every token at every
position would take vocabulary × positions backward passes (about 25
million per dream); because ∇_θ L_in is linear in π, one backward through
the kept graph of the first backward yields the whole table at once.
Computation: student forward on the dream, `grad(L_in, θ,
create_graph=True)`, cosine with the detached g, backward into π_θ and
through the dreamer's forward (document in state) into θ. One double
backward; no vocabulary loop. It is a policy gradient with the expectation
over the vocabulary taken exactly, hence low variance, and hence confined
to the positions the dream sampled: it reshapes the distribution at each
visited position given its prefix and credits neither the prefix choice
nor the length (the EOS position is the one length-relevant token it can
move). Its degenerate optimum is content tokens everywhere ("keyword
soup"), which a batch cosine also scores well, so a KL penalty to the
untrained policy is load-bearing whenever it is on.

**Chains.** Unroll the step: θ₁ = θ₀ − α∇L_in(θ₀; π₁), dream row 2 from
θ₁, score it, and backpropagate through θ₁ into π₁. Dream 1 then receives
a direct gradient saying how to change so that the model it produces
dreams row 2 better: the lookahead as a training signal, with no branching
and no filter (MAML's multi-step meta-gradient, Finn et al.). Every step's
graph is kept for the backward, so memory grows with chain length; L = 2
is the practical limit for a 2.7B model on one card. Truncated
backpropagation through time (the RNN recipe) keeps graphs for a window of
W steps and detaches earlier states, so chain length is free and credit
reaches W rows back. Discrete sampling limits the signal as before: the
backward reaches π only at the sampled positions. The dreamer's own
weights receive gradient on two paths, through its logits at each row and
through the stepped states it dreamed from, all landing on the one
trainable set.

Lineage: MAML's one-step meta-gradient (Finn et al., arXiv 1703.03400);
Generative Teaching Networks (Such et al., arXiv 1912.07768, 2020) trained a
data generator by meta-gradients through a learner's training on small
vision tasks; MEND (Mitchell et al., arXiv 2110.11309) trains an editor on
gradients with an outer edit loss. No published instance at LLM scale with
the model as its own generator was found this session.

Engineering: the double backward must pass through the manual Mamba2
mixer path and gradient checkpointing (`use_reentrant=False`); the fused CUDA kernels have no double backward. **Checked 2026-10-26 (gate 2b entry point, `alignment_gate.py`):** the manual path is not twice differentiable as it stands. Every norm it calls (`layer_norm_fn`, the block and final `RMSNorm`, the mixer's gated `RMSNorm`) is a mamba_ssm Triton `autograd.Function` whose backward returns tensors with no graph, so `create_graph=True` loses the second-order path at each norm and raises no error: on a tiny model a finite-difference check of d⟨G, g⟩/dπ had relative error 0.997 with the Triton norms and 0.002 with plain-torch norms (`sft/tests/experiments/lama_ckl/test_alignment_gate.py`). Arm (b) needs a plain-torch norm path in `models/` before gate 2b can pass (added the same day: `model.torch_norms = True` routes every norm through `models/norms.py`'s plain-torch reference norms, parity 4.9e-7 relative in fp32 on a tiny model, finite-difference error 0.003 on against 0.90 off, `models/tests/test_mamba2_780m_torch_norms.py`); the 10-24 reading "pure PyTorch ops, so supported in principle" was wrong for the norms.

## 7. Why self-training loops collapse, and what stabilises them

Training on text sampled from a state that is mostly the model's own text
is a positive-feedback system: whatever the model does most consistently, it
does more of next cycle, with gain above one while the self-generated share
of the signal outweighs the outside share (our loop: ≈10,000 dream tokens
against ≈1,500 document tokens per cycle). Shumailov et al. (Nature 2024,
arXiv 2305.17493) and Alemohammad et al. ("self-consuming generative models
go MAD", arXiv 2307.01850) show recursive training on own samples loses
the tails and converges on a mode; Gerstgrasser et al. (arXiv 2404.01413)
show that accumulating real data alongside generated data avoids it. The
stabilisers in the literature are all an outside anchor: fresh real data
each round, a frozen reference the policy is penalised for leaving, or
selection by an external signal (STaR, SEAL, ReST-EM). Complementary
learning systems (McClelland, McNaughton, O'Reilly 1995) is the brain's
version: a fast store of the day's episodes, replayed into a slow store,
the world returning each morning.

The repetition mechanism measured in the fifth box (a habit held in balance
by the warm start, say it once then stop, tipped when distillation raised
one side everywhere; chains then distilled as normal text) is the
in-context repetition bias of language models (Holtzman et al., arXiv
1904.09751) made context-free by learning from one's own samples.

## 8. Weights versus a separate store

Gradient descent on text installs whatever the text consistently contains;
it does not distinguish a fact from a habit. Knowledge editing tried to
install facts without touching anything else by localised edits rather
than descent on text: ROME (Meng et al., arXiv 2202.05262), MEMIT (arXiv
2210.07229), MEND (arXiv 2110.11309). Edits generalise poorly and degrade
the model as they accumulate past a few thousand. The field's standing
answer for facts in deployed LLMs is retrieval; for weights it is continual
pretraining with real old data mixed in (our mix-review baseline). Our arm
is the storage-free version on purpose (2026-10-13 debrief §2). The
condition under which this note's question reopens the retired fast-store
architecture is in the 2026-10-24 debrief §2.
