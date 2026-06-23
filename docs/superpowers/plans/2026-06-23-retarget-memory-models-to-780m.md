# Retarget memory & continuous-learning models to mamba2-780m Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `mamba2_2_7b_memory` and `mamba2_2_7b_continuous_learning` on the `state-spaces/mamba2-780m` backbone instead of `state-spaces/mamba2-2.7b`, renaming both folders to match, and delete the 2.7B versions entirely.

**Architecture:** Mechanical retarget, not a redesign — same Titans front-end + gated-delta-into-SSM-state architecture, recalculated for the 780m backbone's actual dims (`d_model=1536, n_layer=48, nheads=48` vs. the 2.7b's `2560`/`64`/`80`), with `QUANTIZE_LORA_BASE` dropped (plain LoRA) since the smaller backbone fits this project's 8GB dev GPU without 4-bit quantization.

**Tech Stack:** Python, PyTorch, `mamba_ssm` (`MambaLMHeadModel`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-23-retarget-memory-models-to-780m-design.md`.
- 2.7B versions are deleted, not kept alongside (per spec's "Scope").
- `mamba2_780m_continuous_learning` has no `model.py` (pending a separate critic-gated redesign) — only metadata (`MODEL_ID` references in `README.md`/`train_hooks.py`/`__init__.py`) is retargeted; no architecture work happens there.
- `mamba2_780m_memory`'s backbone dims: `D_MODEL=1536, N_LAYER=48, NHEADS=48, HEADDIM=64, D_STATE=128` (confirmed via the cached `state-spaces/mamba2-780m` `config.json`).
- `mamba2_780m_memory`'s injection depth/coverage, rescaled proportionally: `READ_LAYER=32`, `INJECTED_LAYERS=range(16, 48, 2)` (16 layers).
- `mamba2_780m_memory` drops `QUANTIZE_LORA_BASE` entirely — plain LoRA, no `bitsandbytes` quantization.
- `mamba2_780m_memory`'s `DEFAULT_CHUNK_LEN` stays at `2` (not re-extrapolated) — flagged in comments as needing re-measurement via `make preflight`, not guessed upward.
- No change to the gated-delta merge math, the Titans front-end's write/read mechanics, or `Model._mixer_step`'s per-token loop structure.

---

### Task 1: Rename and retarget `mamba2_780m_continuous_learning`

**Files:**
- Rename (git mv): `models/mamba2_2_7b_continuous_learning/` → `models/mamba2_780m_continuous_learning/`
- Modify: `models/mamba2_780m_continuous_learning/README.md`
- Modify: `models/mamba2_780m_continuous_learning/train_hooks.py`
- Modify: `models/mamba2_780m_continuous_learning/__init__.py`

**Interfaces:**
- Produces: no code interface (this model has no `model.py`) — only doc/comment content changes. Later tasks don't depend on anything from this task.

- [ ] **Step 1: Rename the folder**

```bash
cd /media/storage/Altrup/Code/Github/altrux
git mv models/mamba2_2_7b_continuous_learning models/mamba2_780m_continuous_learning
```

- [ ] **Step 2: Verify the rename**

Run: `ls models/mamba2_780m_continuous_learning/`
Expected: `README.md  __init__.py  train_hooks.py` (no `__pycache__` needed; ignore if present)

- [ ] **Step 3: Rewrite `README.md`**

Replace the full content of `models/mamba2_780m_continuous_learning/README.md` with:

```markdown
# mamba2_780m_continuous_learning

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) — a 780M-parameter Mamba2 state-space language model.

## Tokenizer

`EleutherAI/gpt-neox-20b`. Mamba2 checkpoints from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer is the standard pairing used in the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block. These are the linear layers that dominate parameter count in each Mamba2 block and where adapting them gives the most leverage for fine-tuning, analogous to targeting `q_proj`/`v_proj` in a transformer. Plain full-precision LoRA, no 4-bit quantization — the 780M backbone fits this project's dev GPU (8GB) comfortably without it, unlike the 2.7B backbone this model used previously (see git history), which needed QLoRA.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) are registered as tokenizer special tokens (`SPECIAL_TOKENS` in `model.py`), so each role marker is a single atomic token id instead of several ordinary BPE pieces. They're bare — no trailing space — since the marker is a vocab-level concept distinct from prompt formatting; callers append a literal `" "` separator explicitly when building text (see `sft/prepare_data.py`, `backend/app/model/registry.py`). `load_base` resizes `backbone.embedding`/`lm_head` to fit the grown vocabulary, re-tying them (`tie_embeddings: true` in this model's config). Because `TARGET_LORA_MODULES` doesn't include the embedding or `lm_head`, the new tokens' embedding rows stay frozen at their (random) initial values during LoRA SFT — the model can only learn to use the markers via the LoRA-adapted `in_proj`, not by adjusting the embeddings themselves.

## Planned: critic-gated gradient accumulation

`model.py` is removed for now, pending a redesign around a dedicated, integrated critic path: a small head (likely outputting a scalar) that judges the model's own output as it goes. Gradients accumulate continuously rather than being applied per-example; once the critic signals that its output warrants it, the accumulated gradient is applied in a training step. The previous `Model` wrapper (manual `_mixer_step` token loop, `MixerState` threading, chunked training) and `train_hooks.py`'s training loop are gone along with it — see git history for the prior implementation if useful as a starting point. This README section will be replaced with real `Model wrapper quirks`/`Critic` documentation once the redesign lands.

This model has no `<revise>`-tag behavior — that's specific to [`mamba2_780m`](../mamba2_780m/README.md). The two models target different problems (revision-on-feedback vs. continuous learning) on the same backbone.
```

- [ ] **Step 4: Update `train_hooks.py`**

Replace the full content of `models/mamba2_780m_continuous_learning/train_hooks.py` with:

```python
# model.py is removed pending a redesign around a dedicated critic path --
# see README.md's "Planned: critic-gated gradient accumulation" section.
# This file's training hooks depended directly on model.py and are removed
# along with it; they'll come back once the critic path is implemented.
```

(This is the same content as before — no `MODEL_ID`/dims were referenced here, so nothing else changes.)

- [ ] **Step 5: Update `__init__.py`**

Replace the full content of `models/mamba2_780m_continuous_learning/__init__.py` with:

```python
# model.py is removed pending a redesign around a dedicated critic path --
# see README.md's "Planned: critic-gated gradient accumulation" section.
```

(Same content as before — no model-name references existed here either.)

- [ ] **Step 6: Verify nothing imports the old name**

Run: `grep -rn "mamba2_2_7b_continuous_learning" --include="*.py" --include="*.md" . | grep -v node_modules | grep -v docs/superpowers/specs | grep -v docs/superpowers/plans`
Expected: no output (the only remaining hits should be in historical spec/plan docs, which are intentionally excluded by this grep and left alone).

- [ ] **Step 7: Commit**

```bash
git add -A models/mamba2_780m_continuous_learning models/mamba2_2_7b_continuous_learning
git commit -m "$(cat <<'EOF'
Retarget mamba2_780m_continuous_learning to the 780m backbone

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Rename and retarget `mamba2_780m_memory`'s `model.py`, `train_hooks.py`, `__init__.py`

**Files:**
- Rename (git mv): `models/mamba2_2_7b_memory/` → `models/mamba2_780m_memory/`
- Modify: `models/mamba2_780m_memory/model.py`
- Modify: `models/mamba2_780m_memory/train_hooks.py`
- Modify: `models/mamba2_780m_memory/__init__.py`

**Interfaces:**
- Produces: `models.mamba2_780m_memory` package exporting the same names as before (`MODEL_ID, TOKENIZER_ID, TARGET_LORA_MODULES, USER_OPEN, ASST_OPEN, SPECIAL_TOKENS, Model, load_base, load_inference`) — **`QUANTIZE_LORA_BASE` is no longer exported** (removed). Task 3's renamed test imports `models.mamba2_780m_memory.model as M` and `models.mamba2_780m_memory.train_hooks as hooks`. Task 4's cross-reference updates assume this package path.

- [ ] **Step 1: Rename the folder**

```bash
cd /media/storage/Altrup/Code/Github/altrux
git mv models/mamba2_2_7b_memory models/mamba2_780m_memory
```

- [ ] **Step 2: Edit `model.py` — module docstring and `MODEL_ID`/QLoRA comment**

In `models/mamba2_780m_memory/model.py`, change:

```python
"""Mamba2-2.7B backbone (frozen) + a trainable long-term memory subsystem.
```

to:

```python
"""Mamba2-780M backbone (frozen) + a trainable long-term memory subsystem.
```

Then change:

```python
MODEL_ID = "state-spaces/mamba2-2.7b"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
# The backbone is QLoRA-adapted (frozen, 4-bit-quantized in_proj/out_proj +
# trainable low-rank adapters) rather than left fully frozen -- see
# QUANTIZE_LORA_BASE below. The memory subsystem (front-end, gate
# projections) is separate from LoRA: it has no pretrained weights to adapt,
# so it trains with ordinary full-parameter gradients from a random init.
TARGET_LORA_MODULES: list[str] = ["in_proj", "out_proj"]
# Opt-in, per-model flag: load_base() below checks this and, if set, quantizes
# TARGET_LORA_MODULES to 4-bit (NF4) via models.common.quantize_lora_targets,
# for true QLoRA rather than plain full-precision LoRA. Other models in this
# repo don't define this constant and so stay on plain LoRA.
QUANTIZE_LORA_BASE = True
```

to:

```python
MODEL_ID = "state-spaces/mamba2-780m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
# The backbone is LoRA-adapted (frozen, full-precision in_proj/out_proj +
# trainable low-rank adapters) rather than left fully frozen, on the theory
# that a fully frozen backbone is unlikely to integrate a memory signal
# injected straight into its SSM state well. This 780M backbone is small
# enough to fit this project's dev GPU (8GB) without 4-bit quantization,
# unlike the 2.7B backbone this model used previously (see git history),
# which needed QLoRA -- so no QUANTIZE_LORA_BASE flag here, same as
# mamba2_780m. The memory subsystem (front-end, gate projections) is
# separate from LoRA: it has no pretrained weights to adapt, so it trains
# with ordinary full-parameter gradients from a random init.
TARGET_LORA_MODULES: list[str] = ["in_proj", "out_proj"]
```

- [ ] **Step 3: Edit `model.py` — backbone shape and injection constants**

Change:

```python
# Mamba2-2.7B backbone shape (state-spaces/mamba2-2.7b).
D_MODEL = 2560
N_LAYER = 64
NHEADS = 80
HEADDIM = 64
D_STATE = 128

# Memory subsystem hyperparameters (see README.md for the rationale).
READ_LAYER = 42
INJECTED_LAYERS: tuple[int, ...] = tuple(range(20, N_LAYER, 2))
BOTTLENECK_R = 128
MEM_DIM = D_MODEL
MEM_HIDDEN = 4 * D_MODEL
# Caps the per-token test-time gradient step in _NeuralMemory.write() (see
# its comment there) -- bounds the update size regardless of how large the
# raw prediction error/gradient is, which an untrained memory MLP can
# produce on its very first few tokens. Tried raising this to 3000 on the
# theory that w1/w2's ~26M elements each make a *healthy* gradient norm
# naturally land in the thousands -- wrong in practice: it only delayed the
# explosion by a few tokens rather than preventing it, since momentum `s` in
# write() has no decay of its own (only `p` decays, via alpha) -- eta near 1
# lets bounded-but-nonzero per-step contributions accumulate across tokens
# regardless of the single-step clip. 1.0 is the only value confirmed (with
# q/k/v normalized -- see _rms_normalize) to run a full example through
# preflight without going non-finite. Revisit alongside decaying/clamping
# momentum itself, not just the per-step gradient, before raising this again.
MAX_WRITE_GRAD_NORM = 1.0
```

to:

```python
# Mamba2-780M backbone shape (state-spaces/mamba2-780m).
D_MODEL = 1536
N_LAYER = 48
NHEADS = 48
HEADDIM = 64
D_STATE = 128

# Memory subsystem hyperparameters (see README.md for the rationale).
# READ_LAYER/INJECTED_LAYERS are rescaled proportionally from this model's
# previous 2.7B backbone (READ_LAYER=42/64, INJECTED_LAYERS=range(20,64,2))
# to preserve the same relative depth (~2/3) and coverage (~1/3) on this
# backbone's 48 layers, not re-derived from scratch -- see git history for
# the original rationale (weak keys too early, collapsed-to-next-token too
# late).
READ_LAYER = 32
INJECTED_LAYERS: tuple[int, ...] = tuple(range(16, N_LAYER, 2))
BOTTLENECK_R = 128
MEM_DIM = D_MODEL
MEM_HIDDEN = 4 * D_MODEL
# Caps the per-token test-time gradient step in _NeuralMemory.write() (see
# its comment there) -- bounds the update size regardless of how large the
# raw prediction error/gradient is, which an untrained memory MLP can
# produce on its very first few tokens. Tried raising this to 3000 on the
# theory that w1/w2's ~9.4M elements each (on this backbone's MEM_DIM/
# MEM_HIDDEN) make a *healthy* gradient norm naturally land in the
# thousands -- wrong in practice: it only delayed the explosion by a few
# tokens rather than preventing it, since momentum `s` in write() has no
# decay of its own (only `p` decays, via alpha) -- eta near 1 lets
# bounded-but-nonzero per-step contributions accumulate across tokens
# regardless of the single-step clip. 1.0 is the only value confirmed (with
# q/k/v normalized -- see _rms_normalize) to run a full example through
# preflight without going non-finite. Revisit alongside decaying/clamping
# momentum itself, not just the per-step gradient, before raising this again.
MAX_WRITE_GRAD_NORM = 1.0
```

- [ ] **Step 4: Edit `model.py` — `Model` class docstring**

Change:

```python
class Model(nn.Module):
    """QLoRA-adapted Mamba2-2.7B backbone with the memory subsystem spliced in.

    in_proj/out_proj are frozen and 4-bit-quantized (see QUANTIZE_LORA_BASE,
    load_base); everything else in the backbone is frozen at bf16 (see
    load_base for why not fp32); LoRA adapters on in_proj/out_proj and the
    memory subsystem itself are the only trainable parameters.
```

to:

```python
class Model(nn.Module):
    """LoRA-adapted Mamba2-780M backbone with the memory subsystem spliced in.

    in_proj/out_proj and the rest of the backbone are frozen at bf16 (see
    load_base for why not fp32); LoRA adapters on in_proj/out_proj and the
    memory subsystem itself are the only trainable parameters.
```

- [ ] **Step 5: Edit `model.py` — `load_base`**

Change:

```python
def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model, with TARGET_LORA_MODULES quantized to
    4-bit per QUANTIZE_LORA_BASE. Used by sft/train.py and the backend
    registry; LoRA adapters themselves are attached separately by the
    caller (see sft/lora.py / backend/app/model/lora.py), after this.

    Loads in bf16, not fp32: quantize_lora_targets only shrinks the model
    *after* from_pretrained has already materialized it on `device`, so the
    transient peak during loading is the full unquantized model's size --
    at fp32 that's ~11GB for this 2.7B-parameter backbone, more than this
    project's dev GPU (8GB) has, so loading itself would OOM before
    quantization ever got a chance to run. bf16 halves that peak to ~5.4GB.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings, quantize_lora_targets

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer))
    if QUANTIZE_LORA_BASE:
        quantize_lora_targets(model, TARGET_LORA_MODULES)
    return model
```

to:

```python
def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/train.py and the backend
    registry; LoRA adapters themselves are attached separately by the
    caller (see sft/lora.py / backend/app/model/lora.py), after this.

    Loads in bf16, not fp32, matching the other Mamba2 models in this repo
    (see models/mamba2_780m/model.py's load_base) -- this 780M-parameter
    backbone is small enough that plain LoRA (no 4-bit quantization) fits
    this project's dev GPU (8GB) comfortably, unlike the 2.7B backbone this
    model used previously (see git history), which needed QLoRA to fit.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer))
    return model
```

- [ ] **Step 6: Verify `model.py` has no remaining `QUANTIZE_LORA_BASE`/`2.7` references**

Run: `grep -n "QUANTIZE_LORA_BASE\|2\.7\|2_7b" models/mamba2_780m_memory/model.py`
Expected: no output

- [ ] **Step 7: Edit `__init__.py`**

Read the current file first (`models/mamba2_780m_memory/__init__.py`), then change:

```python
from .model import (
    ASST_OPEN,
    MODEL_ID,
    QUANTIZE_LORA_BASE,
    SPECIAL_TOKENS,
    TARGET_LORA_MODULES,
    TOKENIZER_ID,
    USER_OPEN,
    Model,
    load_base,
    load_inference,
)

__all__ = [
    "MODEL_ID",
    "TOKENIZER_ID",
    "TARGET_LORA_MODULES",
    "QUANTIZE_LORA_BASE",
    "USER_OPEN",
    "ASST_OPEN",
    "SPECIAL_TOKENS",
    "Model",
    "load_base",
    "load_inference",
]
```

to:

```python
from .model import (
    ASST_OPEN,
    MODEL_ID,
    SPECIAL_TOKENS,
    TARGET_LORA_MODULES,
    TOKENIZER_ID,
    USER_OPEN,
    Model,
    load_base,
    load_inference,
)

__all__ = [
    "MODEL_ID",
    "TOKENIZER_ID",
    "TARGET_LORA_MODULES",
    "USER_OPEN",
    "ASST_OPEN",
    "SPECIAL_TOKENS",
    "Model",
    "load_base",
    "load_inference",
]
```

- [ ] **Step 8: Edit `train_hooks.py` — module docstring and `DEFAULT_CHUNK_LEN` comment**

In `models/mamba2_780m_memory/train_hooks.py`, change:

```python
"""Training hooks for mamba2_2_7b_memory, called by sft/train.py's generic
training loop. Contrast with models/mamba2_780m/train_hooks.py: this model's
Model.forward(input_ids, state) is stateful, so sft/train.py processes
examples in --chunk-len chunks with `state` carried (and detached) across
chunks of the SAME example -- never across different examples -- bounding
training RAM by chunk length rather than example length. See
models/mamba2_2_7b_memory/README.md for why long examples aren't truncated at
data-prep time instead.
```

to:

```python
"""Training hooks for mamba2_780m_memory, called by sft/train.py's generic
training loop. Contrast with models/mamba2_780m/train_hooks.py: this model's
Model.forward(input_ids, state) is stateful, so sft/train.py processes
examples in --chunk-len chunks with `state` carried (and detached) across
chunks of the SAME example -- never across different examples -- bounding
training RAM by chunk length rather than example length. See
models/mamba2_780m_memory/README.md for why long examples aren't truncated at
data-prep time instead.
```

Then change:

```python
EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b
# This model's manual, unfused, per-token mixer step costs ~1GB of VRAM per
# token while a backward graph is live -- on this project's dev GPU (8GB),
# chunk_len 512 (or even 32) OOMs before finishing a single chunk's forward
# pass; chunk_len 4 gets through forward but OOMs in .backward(). chunk_len 3
# was confirmed to OOM in .backward() too once the non-finite-loss bug (see
# model.py's MAX_WRITE_GRAD_NORM/_rms_normalize/eta cap) was fixed and
# backward could actually be reached -- 2 is the largest value confirmed to
# get all the way through both forward and backward without OOMing. Gradient
# checkpointing on the per-token mixer step would be the way to raise this
# again within the same 8GB budget (at the cost of ~2x forward compute);
# override with --chunk-len if running on a GPU with more VRAM.
DEFAULT_CHUNK_LEN = 2
```

to:

```python
EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b
# Measured against this model's *previous* 2.7B QLoRA backbone: this model's
# manual, unfused, per-token mixer step cost ~1GB of VRAM per token while a
# backward graph was live, and on this project's dev GPU (8GB) chunk_len 2
# was the largest value confirmed to get all the way through both forward
# and backward without OOMing (chunk_len 3+ OOM'd in .backward()). The
# backbone is now the smaller 780M model with plain LoRA (no 4-bit
# quantization) instead -- both changes shrink the real per-token VRAM cost,
# but by how much hasn't been re-measured on this backbone, so this value is
# left unchanged (conservative) rather than guessed upward. Re-run `make
# preflight` with a range of --chunk-len values on this backbone to find the
# new ceiling; gradient checkpointing on the per-token mixer step is the way
# to raise it further within a fixed VRAM budget if needed.
DEFAULT_CHUNK_LEN = 2
```

- [ ] **Step 9: Edit `train_hooks.py` — `setup_training` docstring**

Change:

```python
def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the backbone (quantizing TARGET_LORA_MODULES if
    QUANTIZE_LORA_BASE), attaches LoRA, then wraps in Model -- training
    operates on the full memory-augmented wrapper, not the raw backbone, since
    the memory subsystem (front_end, injections) only exists on Model.
    Model.__init__ already freezes everything except lora_A/lora_B and the
    memory subsystem -- nothing extra to freeze here. Returns (model,
    trainable_params)."""
```

to:

```python
def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the backbone, attaches LoRA, then wraps in Model -- training
    operates on the full memory-augmented wrapper, not the raw backbone, since
    the memory subsystem (front_end, injections) only exists on Model.
    Model.__init__ already freezes everything except lora_A/lora_B and the
    memory subsystem -- nothing extra to freeze here. Returns (model,
    trainable_params)."""
```

- [ ] **Step 10: Verify no remaining old references in this package**

Run: `grep -n "mamba2_2_7b_memory\|QUANTIZE_LORA_BASE\|2\.7b" models/mamba2_780m_memory/model.py models/mamba2_780m_memory/train_hooks.py models/mamba2_780m_memory/__init__.py`
Expected: no output

- [ ] **Step 11: Commit**

```bash
git add -A models/mamba2_780m_memory models/mamba2_2_7b_memory
git commit -m "$(cat <<'EOF'
Retarget mamba2_780m_memory's model.py/train_hooks.py to the 780m backbone

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rename and update the synthetic regression test, run it to confirm the wiring still works

**Files:**
- Rename (git mv): `models/tests/test_mamba2_2_7b_memory_train_hooks.py` → `models/tests/test_mamba2_780m_memory_train_hooks.py`
- Modify: `models/tests/test_mamba2_780m_memory_train_hooks.py`

**Interfaces:**
- Consumes: `models.mamba2_780m_memory.model` (`M`) and `models.mamba2_780m_memory.train_hooks` (`hooks`) from Task 2 — same attribute names as before (`M.D_MODEL`, `M.N_LAYER`, `M.NHEADS`, `M.HEADDIM`, `M.D_STATE`, `M.READ_LAYER`, `M.INJECTED_LAYERS`, `M.MEM_DIM`, `M.MEM_HIDDEN`, `M._TitansFrontEnd`, `M._GatedDeltaInjection`, `M.causal_conv1d_update`, `hooks.setup_training`, `hooks.chunk_loss`). This test is fully synthetic (monkeypatches every dimension to tiny values), so it is unaffected by the real 780m dims — only the module path and docstring text change.

- [ ] **Step 1: Rename the test file**

```bash
cd /media/storage/Altrup/Code/Github/altrux
git mv models/tests/test_mamba2_2_7b_memory_train_hooks.py models/tests/test_mamba2_780m_memory_train_hooks.py
```

- [ ] **Step 2: Update the module docstring**

In `models/tests/test_mamba2_780m_memory_train_hooks.py`, change:

```python
"""Fast (seconds, no download) regression test for mamba2_2_7b_memory's
training path.

Built at a tiny synthetic size instead of the real 2.7B checkpoint -- this is
what catches a model whose forward silently fails to connect some part of
itself to the loss (exactly the class of bug found in this model's history:
an int/string key mismatch that made the memory merge never run at all, and
a wrong autograd flag that disconnected k_proj/v_proj from the outer loss --
neither crashed, both produced grad=None for whole branches that should have
been nonzero). Run this before trusting a real training run, not after.

Requires a real GPU (skipped otherwise): Mamba2's mixer uses a Triton kernel
internally (the fused RMSNorm path) that doesn't run on CPU tensors.
"""
```

to:

```python
"""Fast (seconds, no download) regression test for mamba2_780m_memory's
training path.

Built at a tiny synthetic size instead of the real checkpoint -- this is
what catches a model whose forward silently fails to connect some part of
itself to the loss (exactly the class of bug found in this model's history:
an int/string key mismatch that made the memory merge never run at all, and
a wrong autograd flag that disconnected k_proj/v_proj from the outer loss --
neither crashed, both produced grad=None for whole branches that should have
been nonzero). Run this before trusting a real training run, not after.

Requires a real GPU (skipped otherwise): Mamba2's mixer uses a Triton kernel
internally (the fused RMSNorm path) that doesn't run on CPU tensors.
"""
```

- [ ] **Step 3: Update the import paths**

Change:

```python
import models.mamba2_2_7b_memory.model as M
import models.mamba2_2_7b_memory.train_hooks as hooks
```

to:

```python
import models.mamba2_780m_memory.model as M
import models.mamba2_780m_memory.train_hooks as hooks
```

- [ ] **Step 4: Run the test**

Run: `cd /media/storage/Altrup/Code/Github/altrux/sft && PYTHONPATH=.. uv run --no-sync pytest ../models/tests/test_mamba2_780m_memory_train_hooks.py -v`
Expected: all tests pass, or all are skipped with "Requires a real GPU" if run on a machine without one — either is fine; a hard import/collection error is not (it would indicate the rename broke the module path).

- [ ] **Step 5: Verify no remaining old references in this file**

Run: `grep -n "mamba2_2_7b_memory" models/tests/test_mamba2_780m_memory_train_hooks.py`
Expected: no output

- [ ] **Step 6: Commit**

```bash
git add -A models/tests/test_mamba2_780m_memory_train_hooks.py models/tests/test_mamba2_2_7b_memory_train_hooks.py
git commit -m "$(cat <<'EOF'
Rename mamba2_780m_memory's regression test to match the retargeted package

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Rewrite `mamba2_780m_memory/README.md`

**Files:**
- Modify: `models/mamba2_780m_memory/README.md`

**Interfaces:**
- Consumes: the constants from Task 2 (`D_MODEL=1536, N_LAYER=48, NHEADS=48, HEADDIM=64, D_STATE=128, READ_LAYER=32, INJECTED_LAYERS=range(16,48,2)`).
- Produces: nothing consumed by later tasks — this is a leaf doc-only task.

- [ ] **Step 1: Replace the full content of `models/mamba2_780m_memory/README.md`**

```markdown
# mamba2_780m_memory

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) — the same 780M-parameter Mamba2 backbone as [`mamba2_780m_continuous_learning`](../mamba2_780m_continuous_learning/README.md), wrapped with a trainable long-term memory subsystem: an addressable associative store updated by the gated delta rule, fed by a Titans-style nonlinear front-end that decides what's worth writing. The memory subsystem (front-end, gate projections) is new and trained from scratch with full gradients. The backbone itself is **not** frozen — it's adapted via LoRA, on the theory that a fully frozen backbone is unlikely to integrate a memory signal injected straight into its SSM state well, since none of its own weights ever get a chance to adjust to that new input. Letting LoRA nudge the backbone's own projections should help it learn to actually use what the memory injects, rather than just tolerate it.

Backbone shape: `d_model=1536`, `n_layer=48`, `d_inner=3072` (expand 2), `nheads=48`, `headdim=64`, `d_state=128`, `ngroups=1` (B/C shared across heads). Per head, Mamba2's own SSM state is a 64×128 matrix, written with a rank-1 outer product and read via the C projection. At injected layers (see below), the memory subsystem's gated-delta rule is merged directly into this same state tensor — not a side accumulator added on top — so memory shares both the *form* (rank-1/low-rank associative writes, projection-based reads) and the actual storage with Mamba2's own state.

## Tokenizer

`EleutherAI/gpt-neox-20b`, same as the other Mamba2 models in this repo — the checkpoint ships without its own tokenizer, so the GPT-NeoX-20B tokenizer (the standard pairing from the original Mamba training recipe) is used here too.

## LoRA target modules

`["in_proj", "out_proj"]` — the same choice as the other Mamba2 models in this repo (Mamba2's analogue of a transformer's q/k/v/o projections). Plain full-precision LoRA, no 4-bit quantization — this 780M backbone fits this project's dev GPU (8GB) comfortably without it, unlike the 2.7B backbone this model used previously (see git history), which needed QLoRA to fit.

The memory subsystem itself (front-end, gate projections) stays outside LoRA either way — it's trained with full gradients from a random init, since it has no pretrained weights to adapt.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`), registered as tokenizer special tokens the same way as the other models in this repo, so each role marker is a single atomic token id. Bare, no trailing space — callers append a literal `" "` separator explicitly.

## `Model` wrapper quirks

The wrapper runs the frozen Mamba2 stack with one shared memory block spliced in, rather than per-layer memory blocks. Per-token data flow:

```
token t
  │
  ▼
embedding
  │
  ▼
layers 0..15 ───────────────────────────────────────────────► (unmodified Mamba2 mixer step)
  │
  ▼
layer 16: ssm_state = ssm_state·dA + dBx                  (Mamba2's own decay+write)
          ssm_state = clear·(ssm_state - β·(ssm_state·key)key) + β·p·key   (gated-delta merge, signals from token t-1's o_t)
          y = C·ssm_state ; persisted ssm_state IS the merged one
  │
  ▼
layer 17 ─────────────────────────────────────────────────────► (unmodified, odd layers have no injection)
  │
  ▼
  ⋮ (even layers 18, 20, ..., 30 each merge their own signals the same way as layer 16)
  │
  ▼
layer 32: same merge as above, using signals from token t-1's o_t
  │
  ├──► residual (pre-layer-32 residual stream, NOT affected by layer 32's own merge above)
  │       │
  │       ▼
  │     Stage 1: Titans front-end
  │       write: M_t = (1-α)M_{t-1} + (η·S_{t-1} - θ·∇L),  L = ‖M(k)-v‖²
  │       read:  o_t = M_t(q_t)              ──┐
  │       surprise = ‖M(k)-v‖² (detached)  ────┤ broadcast to all 16 injected layers (incl. layer 32 itself)
  ▼                                            │
layer 33 (no injection) ◄───────────────────────┘
  │
  ▼
layer 34: gated-delta merge using THIS token's o_t (already computed at layer 32) ──► ⋮ ──► layer 46
  │
  ▼
layer 47 (no injection) ──► norm_f ──► lm_head ──► logits_t
```

Note the subtlety at layer 32: the merge at layer 32 (using signals derived from `o_t` computed on the *previous* token) happens first, then the residual entering layer 32 — unaffected by that merge — is what Stage 1 reads. So this token's `o_t` isn't visible to layer 32's own merge until token *t+1*; there's no same-token cycle. Layers 34–46 get this token's `o_t` immediately (same token, later in the layer stack); layers 16–30 (and 32 itself) always lag by one token.

### Sequential, per-token execution (not chunked)

Because the gated-delta accumulator at layer *i* depends on the memory read at `READ_LAYER`, and that read for token *t* must be visible to *later* layers of the *same* token while only being visible to *earlier* layers on the *next* token, the backbone can't run through Mamba2's fused/chunked parallel-scan kernels — those process a whole sequence in one kernel call and don't expose a per-token, pre-readout hook. `Model.forward` therefore loops over time explicitly, replicating Mamba2's own incremental-decode arithmetic (`_mixer_step`) for every layer, every token — the same asymptotic cost as autoregressive decoding, just paid during training too. This is materially slower than the library's native chunked training path; revisit if it becomes a bottleneck (e.g. a custom chunked kernel that exposes the pre-readout state).

### Read location

The memory reads the residual stream once, at roughly 2/3 depth (layer ~32 of 48, `READ_LAYER`). Earlier layers produce weak keys (too little semantic content yet); later layers are already collapsed toward next-token prediction. This depth is exposed as a hyperparameter and is the first thing to sweep if recall is weak.

### Stage 1 — Titans-style neural front-end (shared, single instance)

Three dedicated projections turn the layer-32 residual into query/key/value (`q`, `k`, `v`); a small deep MLP `M` (~2 layers, wide hidden) *is* the memory — its weights are the memory content, not an activation. Writing is a test-time gradient step on the associative loss `L = ‖M(k) - v‖²`:

```
S_t = η_t·S_{t-1} - θ_t·∇L
M_t = (1 - α_t)·M_{t-1} + S_t
```

with `η`/`θ`/`α` produced by small data-dependent sigmoid projections. Reading is a pure forward pass, `o_t = M_t(q_t)` (a 1536-dim vector), with no weight update. The gradient magnitude from the write step ("surprise") is exported downstream to modulate Stage 2's write strength.

### Stage 2 — gated-delta merge directly into Mamba2's own SSM state (one signal-generator per injected layer)

Each injected layer derives its own write signals from `o_t` through a dedicated `1536 → 128 → expand` bottleneck: a per-head value `p_t` (64-dim × 48 heads = 3072), a shared 128-dim key/address `B_t`, a write strength `beta_t = sigmoid(W_beta·o_t + surprise_t)`, and a clear gate `clear_t = sigmoid(W_clear·o_t)` initialized near 1. Unlike an earlier version of this design, there is **no separate accumulator** — the gated-delta rule is applied directly to Mamba2's own per-layer `ssm_state`, immediately after Mamba2's own decay+write and before the `C` readout, in `Model._mixer_step`:

```
ssm_state_t = ssm_state_t · dA + dBx                                      (Mamba2's own update, unmodified)
ssm_state_t = clear_t·(ssm_state_t - beta_t·(ssm_state_t·B_t)·B_t^T) + beta_t·p_t·B_t^T   (gated-delta merge)
y_t = C_t · ssm_state_t                                                   (Mamba2's own readout, unmodified)
```

and the *merged* `ssm_state_t` — not a separate copy — is what gets persisted to the next token. The delta term overwrites only the address being written and leaves other content intact (content-addressed forgetting); `clear_t` is a data-dependent global wipe for topic/segment boundaries, initialized near 1 so it never forces decay on its own. `beta_t` is initialized near 0 (`beta_proj.bias = -4`) so the merge is a no-op at init, matching `clear_t`'s near-1 init — together they mean the wrapped backbone behaves exactly like the unmodified pretrained model until these gates learn otherwise. This stage is essentially a near-reference implementation of the accumulator in Yang, Kautz & Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025, arXiv:2412.06464) — applied to the backbone's own state rather than a side accumulator.

Because memory now lives in the same tensor Mamba2 itself decays via `A` each step, memory content is subject to the backbone's own (frozen, pretrained-for-its-own-purposes) per-head decay rate in addition to the gated-delta dynamics — there's no longer an independent persistence mechanism insulated from the backbone's recurrence. That's a deliberate tradeoff versus the earlier separate-accumulator design (see git history): simpler (one state per layer, not two), but memory's effective retention is now coupled to whatever decay rate `A` already encodes for that head, which was learned for the backbone's own purposes, not for long-term memory retention.

### Injection

Mamba2's existing `C` projection reads the merged state directly, so no separate reader module is needed — "injection" here means *which* layers run the gated-delta merge above, not an additive side-channel.

Only a subset of layers carry a merge point — every even layer from 16 to 46 inclusive (`INJECTED_LAYERS = range(16, 48, 2)` in `model.py`), 16 of 48 layers total:

```
16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46
```

`READ_LAYER` (32) falls inside this range, so layer 32 both merges memory into its own state *and* is read by Stage 1 — see the diagram note above for why that doesn't create a same-token cycle. The injected-layer set is itself a hyperparameter, not architecturally required to be this exact stride/range; it (and `READ_LAYER`) is rescaled proportionally from this model's previous 2.7B/64-layer backbone, not re-derived from scratch — see git history for the original rationale.

### Parameter budget

~17M trainable against the 780M frozen backbone (~2.2%), excluding LoRA adapter params (which scale with `--lora-rank` and aren't fixed by the architecture) — Titans front-end (QKV + knob projections) ~7.1M, per-layer injection bottlenecks ~9.8M (16 layers × ~0.61M each, including that layer's gate projections).

### Open architectural question

The Titans front-end and the gated-delta accumulator are both gradient-based associative memories (the delta rule is one step of gradient descent on the same associative objective), so they may be partially redundant. Before committing to the full two-stage stack, the plan is to ablate a gated-delta-only variant (plain nonlinear projection straight to `p`/`B`, no Stage 1) against the full stack, and keep Stage 1 only if it earns its cost in long-context recall.

This model has no `<revise>`-tag behavior — that's specific to [`mamba2_780m_continuous_learning`](../mamba2_780m_continuous_learning/README.md). The two models target different problems (revision-on-feedback vs. long-context memory) on the same backbone.

## Training data

The whole point of this model is long-range recall, so it needs long-session training data, not the short multi-turn chats the other models in this repo use. See [`sft/README.md`](../../sft/README.md#long-context-data-mamba2_780m_memory) — `MODEL_NAME=mamba2_780m_memory make data-memory` builds a mix of real long conversations ([`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k)) and synthetic needle-in-haystack recall QA ([`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong)), since long text alone doesn't force the memory gates to actually do anything — only tasks that depend on far-back information do.
```

- [ ] **Step 2: Verify no remaining old references**

Run: `grep -n "2\.7\|2_7b\|QUANTIZE_LORA_BASE\|bitsandbytes\|HSA_OVERRIDE" models/mamba2_780m_memory/README.md`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add models/mamba2_780m_memory/README.md
git commit -m "$(cat <<'EOF'
Rewrite mamba2_780m_memory's README for the 780m backbone

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update cross-references in the rest of the repo

**Files:**
- Modify: `CLAUDE.md`
- Modify: `models/CLAUDE.md`
- Modify: `sft/Makefile`
- Modify: `sft/README.md`
- Modify: `sft/train.py`
- Modify: `sft/smoke_test.py`
- Modify: `backend/app/model/lora.py`
- Modify: `backend/app/model/registry.py`
- Modify: `models/mamba2_780m/model.py`
- Modify: `models/mamba2_780m/train_hooks.py`
- Modify: `models/mamba2_780m/README.md`

**Interfaces:**
- Consumes: the new package name `mamba2_780m_memory` from Task 2 (these are all comment/doc references, not imports — confirmed by repo-wide grep that no code imports either old package by name outside its own files).
- Produces: nothing — this is the final cleanup task; Task 6 verifies its completeness.

- [ ] **Step 1: `CLAUDE.md`**

Change:

```
Every model in `models/` therefore drives Mamba2's mixer manually in plain PyTorch (see `models/mamba2_2_7b_memory/model.py`'s `_mixer_step` and its README for the investigation) instead of calling `Mamba2.forward()`/`.step()`.
```

to:

```
Every model in `models/` therefore drives Mamba2's mixer manually in plain PyTorch (see `models/mamba2_780m_memory/model.py`'s `_mixer_step` and its README for the investigation) instead of calling `Mamba2.forward()`/`.step()`.
```

- [ ] **Step 2: `models/CLAUDE.md`**

Change all three occurrences of `mamba2_2_7b_memory` to `mamba2_780m_memory` (lines referencing `chunk_extra_log`'s example, the `mask_slice`/checkpoint-save explanation, and the final "See ... for the one that also defines `chunk_extra_log`" pointer). The surrounding text is unchanged — only the model name.

- [ ] **Step 3: `sft/Makefile`**

Change:

```makefile
# Long-context training data for mamba2_2_7b_memory: THUDM/LongAlign-10k
# (real long multi-turn conversations) mixed with RMT-team/babilong (synthetic
# needle-in-haystack recall QA, so the memory gates get genuine long-range
# recall pressure, not just long text -- see models/mamba2_2_7b_memory/README.md).
#
# --max-len here is deliberately way above any real example (LongAlign-10k's
# longest is ~65k tokens) -- this --max-len only controls what gets written to
# disk, which is nearly free (a few hundred KB per extra 100k tokens), so there
# is no reason to truncate at prep time. RAM is a *training*-time problem and
# belongs in chunked/truncated-BPTT training (Model.forward's state param is
# built for this -- see models/mamba2_2_7b_memory/README.md), not here.
```

to:

```makefile
# Long-context training data for mamba2_780m_memory: THUDM/LongAlign-10k
# (real long multi-turn conversations) mixed with RMT-team/babilong (synthetic
# needle-in-haystack recall QA, so the memory gates get genuine long-range
# recall pressure, not just long text -- see models/mamba2_780m_memory/README.md).
#
# --max-len here is deliberately way above any real example (LongAlign-10k's
# longest is ~65k tokens) -- this --max-len only controls what gets written to
# disk, which is nearly free (a few hundred KB per extra 100k tokens), so there
# is no reason to truncate at prep time. RAM is a *training*-time problem and
# belongs in chunked/truncated-BPTT training (Model.forward's state param is
# built for this -- see models/mamba2_780m_memory/README.md), not here.
```

Then change:

```makefile
# Same gradient-wiring check as `preflight`, but on a short synthetic
# sequence instead of the real dataset's first valid example -- useful for
# mamba2_2_7b_memory, whose real examples are thousands of tokens and whose
# --chunk-len is capped at 2 on this hardware (see smoke_test.py). Still
# pays for loading the real model. Pass --length to change the synthetic
# sequence length (default 10).
```

to:

```makefile
# Same gradient-wiring check as `preflight`, but on a short synthetic
# sequence instead of the real dataset's first valid example -- useful for
# mamba2_780m_memory, whose real examples are thousands of tokens and whose
# --chunk-len is conservatively left at 2 (see smoke_test.py). Still pays
# for loading the real model. Pass --length to change the synthetic sequence
# length (default 10).
```

Then change:

```makefile
# train.py is generic across every model in models/ -- it dispatches to
# models/{MODEL_NAME}/train_hooks.py for the model-specific parts (how to
# load the model, how to run forward+backward for one example). For
# mamba2_2_7b_memory specifically:
#   MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt"
#
# On this machine's GPU (unsupported gfx1102 arch) bitsandbytes segfaults
# unless HSA_OVERRIDE_GFX_VERSION=11.0.0 is set when training that model --
# see the root CLAUDE.md. Not baked in here: it's a property of this specific
# machine's GPU, not of any model in general (a supported-arch ROCm GPU, or
# CUDA, needs neither).
```

to:

```makefile
# train.py is generic across every model in models/ -- it dispatches to
# models/{MODEL_NAME}/train_hooks.py for the model-specific parts (how to
# load the model, how to run forward+backward for one example). For
# mamba2_780m_memory specifically:
#   MODEL_NAME=mamba2_780m_memory make train ARGS="--data data/train_memory.pt"
```

(The `bitsandbytes`/`HSA_OVERRIDE_GFX_VERSION` paragraph is removed entirely — it no longer applies to any model in this Makefile's scope now that `mamba2_780m_memory` uses plain LoRA.)

- [ ] **Step 4: `sft/README.md`**

Change every occurrence of `mamba2_2_7b_memory` to `mamba2_780m_memory` in the "Long-context data" heading/anchor (`### Long-context data (\`mamba2_2_7b_memory\`)`), the `make data-memory` example, the `Model.forward`'s `state` param cross-reference, the "Training `mamba2_2_7b_memory`" heading, and the `make train`/`make resume` examples.

Then remove this line entirely (it no longer applies — plain LoRA needs no `bitsandbytes` override):

```markdown
On this machine's GPU (unsupported `gfx1102` arch), set `HSA_OVERRIDE_GFX_VERSION=11.0.0` in your shell before training this model — see the root `CLAUDE.md`.
```

The "Using the adapter" section's `mamba2_2_7b_memory`'s `front_end`/`injections` example also becomes `mamba2_780m_memory`'s `front_end`/`injections`.

- [ ] **Step 5: `sft/train.py`**

Change all four occurrences of `mamba2_2_7b_memory` (module docstring's "See ... and models/mamba2_2_7b_memory/train_hooks.py", the checkpointing docstring's "mamba2_780m) and a model with an additional full-gradient subsystem (mamba2_2_7b_memory's front_end/", the checkpoint-save docstring's "a model like mamba2_2_7b_memory has an additional full-gradient subsystem", the preflight docstring's "models/mamba2_2_7b_memory/train_hooks.py's history", and `_show_chunk_progress`'s "mamba2_2_7b_memory's manual per-token mixer step") to `mamba2_780m_memory`.

- [ ] **Step 6: `sft/smoke_test.py`**

Change:

```python
"""Fast real-model sanity check: loads the actual model (paying the real
load/quantize cost, same as `make preflight`) but runs the gradient-wiring
check on a short synthetic sequence instead of a real dataset example.

`make preflight` picks the first valid example in the real dataset, which
for mamba2_2_7b_memory can be tens of thousands of tokens -- at this model's
--chunk-len (2, capped that low by this hardware's 8GB VRAM, see
models/mamba2_2_7b_memory/train_hooks.py), that's thousands of slow chunks
before the check tells you anything. This script exists for the case where
you just want "does the wiring still work" fast, without waiting on dataset
example length -- it does NOT replace `make preflight` for confirming the
real dataset is actually usable end-to-end.
"""
```

to:

```python
"""Fast real-model sanity check: loads the actual model (paying the real
load/quantize cost, same as `make preflight`) but runs the gradient-wiring
check on a short synthetic sequence instead of a real dataset example.

`make preflight` picks the first valid example in the real dataset, which
for mamba2_780m_memory can be tens of thousands of tokens -- at this model's
--chunk-len (2, inherited conservatively from this model's previous 2.7B
QLoRA backbone -- see models/mamba2_780m_memory/train_hooks.py), that's
thousands of slow chunks before the check tells you anything. This script
exists for the case where you just want "does the wiring still work" fast,
without waiting on dataset example length -- it does NOT replace `make
preflight` for confirming the real dataset is actually usable end-to-end.
"""
```

- [ ] **Step 7: `backend/app/model/lora.py`**

Change:

```python
    """Loads sft/train.py's checkpoint format: every trainable parameter
    (trainable.pt), not just LoRA adapters -- a model like
    mamba2_2_7b_memory has an additional full-gradient subsystem (front_end,
    injections) that a LoRA-only load would silently miss. For a LoRA-only
    model, trainable.pt only ever contained lora_A/lora_B anyway, so this is
    a strict superset of the old adapter.pt-based load_lora, not a behavior
    change for those models."""
```

to:

```python
    """Loads sft/train.py's checkpoint format: every trainable parameter
    (trainable.pt), not just LoRA adapters -- a model like
    mamba2_780m_memory has an additional full-gradient subsystem (front_end,
    injections) that a LoRA-only load would silently miss. For a LoRA-only
    model, trainable.pt only ever contained lora_A/lora_B anyway, so this is
    a strict superset of the old adapter.pt-based load_lora, not a behavior
    change for those models."""
```

- [ ] **Step 8: `backend/app/model/registry.py`**

Change:

```python
        # Per-layer SSM/conv (and, for mamba2_2_7b_memory, memory) state --
```

to:

```python
        # Per-layer SSM/conv (and, for mamba2_780m_memory, memory) state --
```

Then change:

```python
            # Loaded *after* wrapping in Model, not before: a model like
            # mamba2_2_7b_memory has trainable state (front_end, injections)
            # that only exists on the Model wrapper, not on the raw backbone
            # -- loading into `base` would silently miss those keys.
```

to:

```python
            # Loaded *after* wrapping in Model, not before: a model like
            # mamba2_780m_memory has trainable state (front_end, injections)
            # that only exists on the Model wrapper, not on the raw backbone
            # -- loading into `base` would silently miss those keys.
```

- [ ] **Step 9: `models/mamba2_780m/model.py`**

Change all three occurrences of `mamba2_2_7b_memory` (the `MixerState` docstring's "Mirrors mamba2_2_7b_memory's MemoryState", the manual-mixer-step comment's "already used in mamba2_2_7b_memory's _mixer_step", and the forward-replacement comment's "pattern as mamba2_2_7b_memory's MemoryState") to `mamba2_780m_memory`.

- [ ] **Step 10: `models/mamba2_780m/train_hooks.py`**

Change both occurrences of `mamba2_2_7b_memory` (the module docstring's "models/mamba2_2_7b_memory/train_hooks.py does" and the mask-handling comment's "unlike mamba2_2_7b_memory") to `mamba2_780m_memory`.

- [ ] **Step 11: `models/mamba2_780m/README.md`**

Change:

```markdown
- Reassembles the backbone (`embedding`, `layers`, `norm_f`) and `lm_head` from `MambaLMHeadModel` directly rather than calling it as a black box. `forward` loops over tokens manually and threads a `MixerState` (per-layer SSM/conv state) across calls, instead of calling `Mamba2.forward`/`Block.forward` with an `inference_params` cache — `mamba_ssm`'s own fused kernels (the `causal_conv1d` compiled extension *and* its Triton SSD chunk-scan kernel) are broken on this project's dev hardware (an unsupported ROCm GPU architecture): the former segfaults, the latter hangs, both independent of model size or package version. `_mixer_step` manually replicates `Mamba2.step()`'s arithmetic in plain PyTorch instead (verified to match the library's own reference fallback to float32-epsilon precision). This is slower per-token than the fused path would be, but it's the only thing proven to actually run on this hardware — see `models/mamba2_2_7b_memory/README.md`, which already used this approach for an unrelated reason (splicing in its memory subsystem) before this was known to be necessary here too.
```

to:

```markdown
- Reassembles the backbone (`embedding`, `layers`, `norm_f`) and `lm_head` from `MambaLMHeadModel` directly rather than calling it as a black box. `forward` loops over tokens manually and threads a `MixerState` (per-layer SSM/conv state) across calls, instead of calling `Mamba2.forward`/`Block.forward` with an `inference_params` cache — `mamba_ssm`'s own fused kernels (the `causal_conv1d` compiled extension *and* its Triton SSD chunk-scan kernel) are broken on this project's dev hardware (an unsupported ROCm GPU architecture): the former segfaults, the latter hangs, both independent of model size or package version. `_mixer_step` manually replicates `Mamba2.step()`'s arithmetic in plain PyTorch instead (verified to match the library's own reference fallback to float32-epsilon precision). This is slower per-token than the fused path would be, but it's the only thing proven to actually run on this hardware — see `models/mamba2_780m_memory/README.md`, which already used this approach for an unrelated reason (splicing in its memory subsystem) before this was known to be necessary here too.
```

Then change:

```markdown
- `_apply_norm_f`/`_prenorm` branch on `fused_add_norm`: when fused, they call the Triton `layer_norm_fn` kernel directly instead of `norm`/`norm_f`, matching how the upstream model applies normalization. (This Triton kernel is *not* one of the broken ones above — it's exercised extensively by `mamba2_2_7b_memory` already.)
```

to:

```markdown
- `_apply_norm_f`/`_prenorm` branch on `fused_add_norm`: when fused, they call the Triton `layer_norm_fn` kernel directly instead of `norm`/`norm_f`, matching how the upstream model applies normalization. (This Triton kernel is *not* one of the broken ones above — it's exercised extensively by `mamba2_780m_memory` already.)
```

- [ ] **Step 12: Commit**

```bash
cd /media/storage/Altrup/Code/Github/altrux
git add CLAUDE.md models/CLAUDE.md sft/Makefile sft/README.md sft/train.py sft/smoke_test.py \
  backend/app/model/lora.py backend/app/model/registry.py \
  models/mamba2_780m/model.py models/mamba2_780m/train_hooks.py models/mamba2_780m/README.md
git commit -m "$(cat <<'EOF'
Update cross-references to mamba2_780m_memory across the repo

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Final repo-wide verification

**Files:** none modified — read-only verification task.

**Interfaces:**
- Consumes: the completed state of Tasks 1–5.

- [ ] **Step 1: Confirm no remaining references to either old package name outside historical docs**

Run:

```bash
cd /media/storage/Altrup/Code/Github/altrux
grep -rln "mamba2_2_7b_memory\|mamba2_2_7b_continuous_learning\|mamba2-2\.7b" \
  --include="*.py" --include="*.md" --include="Makefile" --include="*.example" . \
  | grep -v node_modules | grep -v docs/superpowers/specs | grep -v docs/superpowers/plans
```

Expected: no output. (Historical spec/plan docs in `docs/superpowers/specs/` and `docs/superpowers/plans/` are intentionally excluded — they're dated records of past decisions, not live documentation, and the new design/plan docs for *this* change correctly reference the new names already.)

- [ ] **Step 2: Confirm the renamed packages import cleanly**

Run:

```bash
cd /media/storage/Altrup/Code/Github/altrux
PYTHONPATH=. python3 -c "import models.mamba2_780m_continuous_learning; print('continuous_learning OK')"
PYTHONPATH=. python3 -c "
import models  # runs the selective_scan_cuda stub first, same as any real entry point
import models.mamba2_780m_memory as m
print('memory OK:', m.MODEL_ID, m.D_MODEL if hasattr(m, 'D_MODEL') else 'D_MODEL not exported (expected -- only model.py has it)')
"
```

Expected: both print their `OK` lines without raising. (`mamba2_780m_memory`'s `__init__.py` doesn't export `D_MODEL` — only `model.py` does — so the `hasattr` check is just confirming the package imports, not asserting that constant is part of the public `__all__`.)

- [ ] **Step 3: Re-run the full models test suite**

Run: `cd /media/storage/Altrup/Code/Github/altrux/sft && PYTHONPATH=.. uv run --no-sync pytest ../models/tests -v`
Expected: all tests pass (or skip on a GPU-only test, same as Task 3) — no collection errors from any other test file that might have referenced the old names.

- [ ] **Step 4: Confirm the old folders are actually gone, not just renamed-and-duplicated**

Run: `ls models/ | grep "2_7b"`
Expected: no output.

No commit for this task — it's verification-only. If any step fails, return to the relevant earlier task and fix it there (with its own commit) rather than papering over it here.
