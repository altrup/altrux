# Continual Learning Model

Goal is to train a Neural Network that has access to the internal state of an LLM to train that LLM based on user-provided reward signals in the range [-1, 1].

## Base Model

[state-spaces/mamba2-370m](https://huggingface.co/state-spaces/mamba2-370m)

## Architecture

The model is a modified version of the base model with an added **critic branch** that taps into the base model's intermediate representations to produce a per-token reward signal used for online weight updates. The reward is sparse: in the vast majority of token positions it is 0 (nothing to learn), and becomes nonzero only at "teachable moments," such as when the user provides corrective or novel information.

Because user feedback arrives as text, it is tokenized and flows through the base model into its internal state. The critic reads that state, so it doubles as the **feedback interpreter**: no separate text-to-reward module is needed. The base model's own representations do the language understanding; the critic detects when the state reflects feedback and emits a nonzero reward.

### Forward Pass

```
Input Tokens
     │
     ▼
[Embedding Layer]
     │
     ▼
[Mamba Layers]  ← first 2/3 of layers
     │
     ├──────────────────────────────────┐
     │                                  │
     ▼                                  ▼
[Remaining Mamba Layers]         [Critic Branch]
     │                           (new mamba layers,
     ▼                            1/3 depth of original model)
[LM Head → Logits]                      │
                                        ▼
                          [Linear → tanh → per-token reward r(t) ∈ [-1, 1]]
                          (≈0 most positions; nonzero on teachable moments)
```

### Critic Branch

- Branches off from the hidden state at the **2/3 depth mark** of the base model.
- Has **1/3 the depth** of the original model.
- **Runs on every token**, reading the per-token hidden states. It must see input tokens (not just output positions) because the user's verbal feedback enters the model as input, and that is precisely the signal the critic needs to detect.
- Emits a **per-token reward r(t)** passed through `tanh`, normalizing it to [-1, 1]. This preserves the per-token structure the Mamba layers already produce and gives a dense signal for credit assignment.
- The reward is **sparse by design**: ≈0 at most positions, nonzero only when the internal state reflects something worth learning from.

### Weight Update Mechanism

The critic's per-token reward is used to compute a loss whose gradient is backpropagated into the **main model's weights**, enabling online learning directly from human feedback without a separate offline reward-model training loop. Updates are **event-driven**: while the critic runs every token, a meaningful weight update only occurs where the reward is nonzero.

## Catastrophic Forgetting

Mitigation strategy TBD.

---

## Open Questions / TODO

- **Sparse-reward data weighting:** Because rewards are ≈0 in the vast majority of cases, a critic trained naively can minimize loss by always predicting 0 and ignoring the rare important cases. We will likely need to **weight the data** (upweight nonzero/teachable examples) or split the critic into a gate ("is this a teachable moment?") and a magnitude/sign head, so it does not collapse to always predicting 0.
- **Update timing:** Whether a nonzero reward triggers an immediate hard weight update at deployment, or a softer mechanism (adapter/memory update, in-context conditioning). This choice drives the rest of the design and the forgetting strategy.
- **Credit assignment:** How per-token rewards attribute credit to the relevant earlier output tokens that the feedback refers to.
- **Catastrophic forgetting:** Select a mitigation (e.g. KL-regularization to the frozen base, replay, EWC, or constrained/LoRA updates).
- **Critic capacity:** Validate that a 1/3-depth critic is necessary; a shallower probe may suffice and would free VRAM.