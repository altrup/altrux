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

### Training Phases

Training is split into two distinct phases:

**Phase 1 — Train the critic (base model frozen)**

The user rates each generated response with a reward signal in [-1, 1]. The critic is trained via MSE loss to predict those ratings. The base model's weights are not touched.

**Phase 2 — Train the base model via the critic**

The critic is frozen. For each forward pass, the critic's reward output is used as a policy signal: `loss = -reward`. The gradient flows backward through the critic's computation graph into the base model's trunk (layers 0–26), nudging the base model to produce representations the critic scores highly. No user reward input is needed in this phase.

## Catastrophic Forgetting

Mitigation strategy TBD.

---

## Setup

```bash
# Create the virtual environment and install all dependencies (including dev tools)
make install

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> **Memory note:** The critic branch deep-copies 27 of the 40 base model layers, roughly doubling GPU memory. Use `--4bit` or `--8bit` for the base model to reduce peak usage (critic branch stays in bfloat16).

## Usage

### Launch the UI

```bash
python -m continual_learning
```

Or, after `pip install -e .`:

```bash
continual-learning
```

#### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `ibm-granite/granite-4.0-h-micro` | HuggingFace model ID |
| `--4bit` | off | Load base model in 4-bit quantization |
| `--8bit` | off | Load base model in 8-bit quantization |
| `--phase` | `1` | Starting phase: `1` = train critic, `2` = train base model |
| `--critic-lr` | `1e-4` | Critic branch learning rate |
| `--base-lr` | `1e-6` | Base model learning rate |
| `--port` | `7860` | Local port for the Gradio UI |
| `--share` | off | Create a public Gradio link |

### Workflow

**Phase 1 — collect ratings, train the critic:**

1. Run with `--phase 1` (the default).
2. Enter a prompt and click **Generate**.
3. Read the response and the critic's current reward estimate.
4. Drag the **Reward Signal** slider to your rating (−1 = bad, +1 = good) and click **Train**.
5. Repeat until the critic's estimates converge toward your ratings.

**Phase 2 — let the critic improve the base model:**

1. Run with `--phase 2` (or switch the radio button in the UI).
2. Enter a prompt and click **Generate**, then click **Train** — no rating needed.
3. The critic scores the response and its reward signal is backpropagated into the base model.

## Testing

Tests use a tiny in-process mock model so they run without a GPU or network connection.

```bash
make test       # run all 22 tests
make test-cov   # same with a coverage report
```

| Test file | What it covers |
|---|---|
| `tests/test_model.py` | Architecture: split index, critic depth, hook capture, reward bounds |
| `tests/test_trainer.py` | Phase isolation: `critic_step` never modifies the base model; `policy_step` never modifies the critic |
| `tests/test_ui.py` | Smoke tests: `create_ui` constructs without errors for both phases |

### Programmatic use

```python
from continual_learning import ContinualLearningModel, Trainer, TrainingConfig

model = ContinualLearningModel()
model.eval()
trainer = Trainer(model, TrainingConfig())

# Phase 1: train the critic on human feedback
response, estimated_reward = model.generate("Explain transformers in one sentence.")
loss = trainer.critic_step(
    prompt="Explain transformers in one sentence.",
    response=response,
    user_reward=0.8,
)
print(f"Critic loss: {loss:.4f}")

# Phase 2: use the trained critic to update the base model
reward, loss = trainer.policy_step(
    prompt="Explain transformers in one sentence.",
    response=response,
)
print(f"Policy loss: {loss:.4f}  Critic reward: {reward:.3f}")
```