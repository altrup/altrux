# Continual Learning Model

Goal is to train a Neural Network that has access to the internal state of an LLM to train that LLM based on user-provided reward signals in the range [-1, 1].

## Base Model

[ibm-granite/granite-4.0-h-micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro) — a hybrid attention + Mamba architecture, chosen for its small size and accessible internal state.

## Architecture

The model is a modified version of the base model with an added **critic branch** that taps into the base model's intermediate representations to produce a scalar reward signal used for online weight updates.

### Forward Pass

```
Input Tokens
     │
     ▼
[Embedding Layer]
     │
     ▼
[Attention + Mamba Layers]  ← first 2/3 of layers
     │
     ├──────────────────────────────────┐
     │                                  │
     ▼                                  ▼
[Remaining Attention +           [Critic Branch]
 Mamba Layers]                   (new attention + mamba layers,
     │                            2/3 depth of original model)
     ▼                                  │
[LM Head → Logits]                      ▼
                                 [Linear → tanh → scalar ∈ [-1, 1]]
```

### Critic Branch

- Branches off from the hidden state at the **2/3 depth mark** of the base model
- Has **2/3 the depth** of the original model (deeper than the remaining main branch, but not full depth)
- Outputs a single scalar value passed through `tanh`, normalizing it to [-1, 1]
- This scalar serves as an **intrinsic reward signal** used to adjust the weights of the main model

### Weight Update Mechanism

The critic output is used to compute a loss against the user-provided reward signal. The gradient from this loss is backpropagated into the **main model's weights**, enabling online learning directly from human feedback without a separate reward model training loop.

## Catastrophic Forgetting

Mitigation strategy TBD.