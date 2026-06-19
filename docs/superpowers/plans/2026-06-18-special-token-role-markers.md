# Special Tokens for Role Markers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register `[USER]`/`[ASSISTANT]` role markers as real tokenizer special tokens (one atomic id each) instead of plain BPE-split strings, resizing each model's embedding/lm_head to match.

**Architecture:** A new shared `models/common.py` module exposes `build_tokenizer(model_mod)` (loads the tokenizer, ensures an eos token, registers `model_mod.SPECIAL_TOKENS`) and `extend_embeddings(model, new_vocab_size)` (grows `backbone.embedding`/`lm_head` in place, re-tying them if the model config ties embeddings). Each model module's `USER_OPEN`/`ASST_OPEN` drop their trailing space and become the single source of truth for `SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN]`. `load_base`/`load_inference` call both helpers. The separator between marker and content becomes explicit wherever text is built: `sft/prepare_data.py` inserts `" "` directly, and `backend/app/model/registry.py` computes the combined display/strip prefix once (`self.asst_open = model_mod.ASST_OPEN + " "`) — every other consumer of `registry.asst_open`/`user_open` (`inference.py`, `config.py`, the frontend) only ever reads that already-combined value, so none of them change.

**Tech Stack:** Python, PyTorch, HuggingFace `transformers` (`AutoTokenizer`), `mamba-ssm` (`MambaLMHeadModel`), `pytest` (new dev dependency).

## Global Constraints

- `USER_OPEN`/`ASST_OPEN` become bare markers (`"[USER]"`, `"[ASSISTANT]"`, no trailing space) and are exactly the list passed to `tokenizer.add_special_tokens` — no separate "bare" list duplicating them.
- The separator between marker and content is always a literal `" "` (not `"\n"` or anything else), so the final byte sequence fed to the tokenizer — and therefore `content`'s tokenization — is identical to before this change; only the marker collapses from several BPE pieces into one atomic id.
- Only two places need to know about the bare marker directly: the model module (defines it) and `backend/app/model/registry.py` (computes `model_mod.ASST_OPEN + " "` once, on assignment). `backend/app/routers/inference.py`, `backend/app/routers/config.py`, and `frontend/app/lib/api.ts` are unaffected and must not be edited — they only ever read `registry.asst_open`/`registry.user_open`.
- `extend_embeddings` must not assume any existing `pad_vocab_size_multiple` headroom covers the new tokens — always compute required size from the tokenizer and resize only if needed.
- No GPU and no real multi-GB checkpoint download required to run the new tests — use a tiny fake `nn.Module` for `extend_embeddings` tests and the already-locally-cached `EleutherAI/gpt-neox-20b` tokenizer (`local_files_only=True`) for `build_tokenizer` tests.
- Current checkpoints are disposable test runs — no migration path needed.

---

### Task 1: `models/common.py` + tests + pytest setup

**Files:**
- Create: `models/common.py`
- Create: `models/tests/__init__.py` (empty)
- Create: `models/tests/test_common.py`
- Modify: `sft/pyproject.toml`
- Modify: `sft/Makefile`

**Interfaces:**
- Produces: `build_tokenizer(model_mod) -> PreTrainedTokenizer` — `model_mod` is any module/namespace exposing `TOKENIZER_ID: str` and `SPECIAL_TOKENS: list[str]`.
- Produces: `extend_embeddings(model: nn.Module, new_vocab_size: int) -> None` — `model` exposes `model.backbone.embedding` (`nn.Embedding`), `model.lm_head` (`nn.Linear`), and `model.config.tie_embeddings` (`bool`). Mutates `model.backbone.embedding` and `model.lm_head` in place (reassigns the attributes); no return value.

- [ ] **Step 1: Add `pytest` as a dev dependency and a `make test` target**

In `sft/pyproject.toml`, add a `[dependency-groups]` table:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

In `sft/Makefile`, add (after the `sync` target):

```makefile
test:
	PYTHONPATH=$(PYTHONPATH) uv run --no-sync pytest ../models/tests
```

Also add `.PHONY: test` to the existing `.PHONY` line at the top of the Makefile (it currently reads `.PHONY: sync data prepare train resume` — change to `.PHONY: sync data prepare train resume test`).

- [ ] **Step 2: Sync the new dev dependency**

Run: `cd sft && uv sync --inexact`
Expected: completes without error; `pytest` importable via `uv run --no-sync python -c "import pytest"`.

- [ ] **Step 3: Write the failing tests**

Create `models/tests/__init__.py` (empty file).

Create `models/tests/test_common.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.common import build_tokenizer, extend_embeddings


class FakeConfig:
    def __init__(self, tie_embeddings: bool):
        self.tie_embeddings = tie_embeddings


class FakeBackbone(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding


class FakeModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, tie_embeddings: bool):
        super().__init__()
        self.config = FakeConfig(tie_embeddings)
        embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = FakeBackbone(embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = embedding.weight


def test_extend_embeddings_tied_grows_and_keeps_tie():
    model = FakeModel(vocab_size=10, d_model=4, tie_embeddings=True)
    old_weight = model.backbone.embedding.weight.detach().clone()

    extend_embeddings(model, new_vocab_size=12)

    assert model.backbone.embedding.weight.shape == (12, 4)
    assert model.lm_head.weight.shape == (12, 4)
    assert torch.allclose(model.backbone.embedding.weight[:10], old_weight)
    assert model.lm_head.weight is model.backbone.embedding.weight


def test_extend_embeddings_untied_grows_independently():
    model = FakeModel(vocab_size=10, d_model=4, tie_embeddings=False)
    old_embed = model.backbone.embedding.weight.detach().clone()
    old_lm_head = model.lm_head.weight.detach().clone()

    extend_embeddings(model, new_vocab_size=12)

    assert model.backbone.embedding.weight.shape == (12, 4)
    assert model.lm_head.weight.shape == (12, 4)
    assert torch.allclose(model.backbone.embedding.weight[:10], old_embed)
    assert torch.allclose(model.lm_head.weight[:10], old_lm_head)
    assert model.lm_head.weight is not model.backbone.embedding.weight


def test_extend_embeddings_noop_when_already_large_enough():
    model = FakeModel(vocab_size=10, d_model=4, tie_embeddings=True)
    embedding_before = model.backbone.embedding

    extend_embeddings(model, new_vocab_size=10)

    assert model.backbone.embedding is embedding_before


def test_build_tokenizer_registers_special_tokens_as_atomic():
    model_mod = SimpleNamespace(
        TOKENIZER_ID="EleutherAI/gpt-neox-20b",
        SPECIAL_TOKENS=["[USER]", "[ASSISTANT]"],
    )

    tokenizer = build_tokenizer(model_mod)

    assert tokenizer.eos_token_id is not None
    user_id = tokenizer.convert_tokens_to_ids("[USER]")
    assert user_id != tokenizer.unk_token_id

    ids = tokenizer.encode("[USER] hello\n", add_special_tokens=False)
    assert ids[0] == user_id
    tail_ids = tokenizer.encode(" hello\n", add_special_tokens=False)
    assert ids[1:] == tail_ids
```

- [ ] **Step 4: Run tests to verify they fail with ModuleNotFoundError**

Run: `cd sft && make test`
Expected: `ModuleNotFoundError: No module named 'models.common'`

- [ ] **Step 5: Implement `models/common.py`**

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_mod) -> PreTrainedTokenizerBase:
    """Load model_mod.TOKENIZER_ID and register model_mod.SPECIAL_TOKENS.

    Deterministic given the same TOKENIZER_ID + SPECIAL_TOKENS list: any two
    callers (sft and backend) building a tokenizer this way get matching token
    ids, even though they never share a tokenizer instance.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    tokenizer.add_special_tokens({"additional_special_tokens": model_mod.SPECIAL_TOKENS})
    return tokenizer


def extend_embeddings(model: nn.Module, new_vocab_size: int) -> None:
    """Grow model.backbone.embedding and model.lm_head to new_vocab_size in place.

    MambaLMHeadModel is a plain nn.Module, not a HF PreTrainedModel, so there is
    no built-in resize_token_embeddings — this reassigns the embedding/lm_head
    attributes with larger tensors, copying existing rows and initializing new
    ones from the existing embedding's std. Re-ties lm_head.weight to the new
    embedding.weight when model.config.tie_embeddings is set.
    """
    embedding = model.backbone.embedding
    old_vocab_size, d_model = embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        return

    std = embedding.weight.std().item()
    device = embedding.weight.device
    dtype = embedding.weight.dtype

    new_embedding = nn.Embedding(new_vocab_size, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = embedding.weight
        new_embedding.weight[old_vocab_size:].normal_(mean=0.0, std=std)
    model.backbone.embedding = new_embedding

    if getattr(model.config, "tie_embeddings", False):
        model.lm_head.weight = new_embedding.weight
        return

    old_lm_head = model.lm_head
    new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
        new_lm_head.weight[old_vocab_size:].normal_(mean=0.0, std=std)
    model.lm_head = new_lm_head
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd sft && make test`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add models/common.py models/tests/__init__.py models/tests/test_common.py sft/pyproject.toml sft/Makefile sft/uv.lock
git commit -m "Add models/common.py for special-token registration and embedding resize"
```

---

### Task 2: Wire up `models/mamba2_780m`

**Files:**
- Modify: `models/mamba2_780m/model.py`
- Modify: `models/mamba2_780m/__init__.py`
- Modify: `models/mamba2_780m/README.md`
- Modify: `models/CLAUDE.md`
- Modify: `CLAUDE.md` (root)
- Modify: `README.md` (root)

**Interfaces:**
- Consumes: `models.common.build_tokenizer(model_mod)`, `models.common.extend_embeddings(model, new_vocab_size)` from Task 1.
- Produces: `models.mamba2_780m.SPECIAL_TOKENS: list[str]`; `load_base`/`load_inference` now return a model whose `backbone.embedding`/`lm_head` are sized to match a tokenizer built via `build_tokenizer`.

- [ ] **Step 1: Add `SPECIAL_TOKENS` and wire resizing into `load_base`/`load_inference`**

In `models/mamba2_780m/model.py`, change:

```python
# Chat format tokens — must match the SFT training format exactly.
# The trailing space is significant; keep it.
USER_OPEN = "[USER] "
ASST_OPEN = "[ASSISTANT] "
```

to:

```python
# Chat format role markers — registered as tokenizer special tokens (see
# SPECIAL_TOKENS), so each is a single atomic token id. Callers append the
# separator between marker and content explicitly (e.g. USER_OPEN + " " + content)
# — keep that separator a literal space to match the SFT training format exactly.
USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN]
```

And change the bottom of the file from:

```python
def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/train.py."""
    return MambaLMHeadModel.from_pretrained(MODEL_ID, device=device)


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device))
```

to:

```python
def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/train.py."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer))
    return model


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device))
```

- [ ] **Step 2: Update `__init__.py`**

In `models/mamba2_780m/__init__.py`, add `SPECIAL_TOKENS` to both the `from .model import (...)` block and `__all__`:

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

- [ ] **Step 3: Sanity-check the real model loads and resizes correctly**

Run:
```bash
cd sft && HF_HOME=$(pwd)/../.cache/huggingface PYTHONPATH=$(pwd)/.. uv run --no-sync python -c "
from models.mamba2_780m import load_base
from models.common import build_tokenizer
import models.mamba2_780m as mod

tok = build_tokenizer(mod)
model = load_base('cpu')
print('tokenizer len:', len(tok))
print('embedding rows:', model.backbone.embedding.weight.shape[0])
print('tied:', model.lm_head.weight is model.backbone.embedding.weight)
assert model.backbone.embedding.weight.shape[0] >= len(tok)
"
```
Expected: `tokenizer len: 50279`, `tied: True`, no assertion error. `embedding rows` is
expected to be `>= 50279`, not necessarily exactly equal — `state-spaces/mamba2-780m`'s
config sets `pad_vocab_size_multiple: 16`, so the raw checkpoint's embedding is already
padded to 50288 rows before `extend_embeddings` runs, and since 50279 <= 50288,
`extend_embeddings` correctly no-ops rather than shrinking it back down. Expect
`embedding rows: 50288` against the currently cached checkpoint. (Note: this loads the
real ~780M checkpoint and may take a minute / require the cached weights already
present in `.cache/huggingface`.)

- [ ] **Step 4: Update docs**

In `models/CLAUDE.md`, add a bullet to the README checklist:

```markdown
- **Source** — the HuggingFace model ID and what it is
- **Tokenizer** — which tokenizer is paired with it and why
- **LoRA target modules** — why those specific modules were chosen as adapter targets
- **`Model` wrapper quirks** — anything non-obvious about the inference wrapper (e.g. deviations from the upstream model's forward pass)
- **Special tokens** — which role markers are registered as tokenizer special tokens and why
```

In root `CLAUDE.md`, the model export table currently reads:

```markdown
| `USER_OPEN` | `str` | User turn prefix (trailing space significant) |
| `ASST_OPEN` | `str` | Assistant turn prefix (trailing space significant) |
```

Change it to:

```markdown
| `USER_OPEN` | `str` | Bare user-turn role marker, registered as a tokenizer special token. Callers append a literal `" "` separator before content. |
| `ASST_OPEN` | `str` | Bare assistant-turn role marker, registered as a tokenizer special token. Callers append a literal `" "` separator before content. |
| `SPECIAL_TOKENS` | `list[str]` | `[USER_OPEN, ASST_OPEN]` — the list passed to `tokenizer.add_special_tokens` |
```

In root `README.md`, update the "Adding a new model" list of required exports to include `SPECIAL_TOKENS`:

```markdown
1. Create a `models/my_model/` package — see `models/mamba2_780m/` for the required interface (`MODEL_ID`, `TOKENIZER_ID`, `TARGET_LORA_MODULES`, `USER_OPEN`, `ASST_OPEN`, `SPECIAL_TOKENS`, `Model`, `load_base`, `load_inference`) and add a `README.md` following `models/CLAUDE.md`
```

In `models/mamba2_780m/README.md`, add a new section after "LoRA target modules":

```markdown
## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) are registered as tokenizer special tokens (`SPECIAL_TOKENS` in `model.py`), so each role marker is a single atomic token id instead of several ordinary BPE pieces. They're bare — no trailing space — since the marker is a vocab-level concept distinct from prompt formatting; callers append a literal `" "` separator explicitly when building text (see `sft/prepare_data.py`, `backend/app/model/registry.py`). `load_base` resizes `backbone.embedding`/`lm_head` to fit the grown vocabulary, re-tying them (`tie_embeddings: true` in this model's config). Because `TARGET_LORA_MODULES` doesn't include the embedding or `lm_head`, the new tokens' embedding rows stay frozen at their (random) initial values during LoRA SFT — the model can only learn to use the markers via the LoRA-adapted `in_proj`, not by adjusting the embeddings themselves.
```

- [ ] **Step 5: Commit**

```bash
git add models/mamba2_780m/model.py models/mamba2_780m/__init__.py models/mamba2_780m/README.md models/CLAUDE.md CLAUDE.md README.md
git commit -m "Register [USER]/[ASSISTANT] as special tokens for mamba2_780m"
```

---

### Task 3: Wire up `models/mamba2_2_7b_continuous_learning`

**Files:**
- Modify: `models/mamba2_2_7b_continuous_learning/model.py`
- Modify: `models/mamba2_2_7b_continuous_learning/__init__.py`
- Modify: `models/mamba2_2_7b_continuous_learning/README.md` (create the "Special tokens" section if the README doesn't already have one — check the existing file first)

**Interfaces:**
- Consumes: same as Task 2.
- Produces: same shape as Task 2, for this model.

- [ ] **Step 1: Apply the identical `model.py` change as Task 2 Step 1**

Same edits as Task 2 Step 1, applied to `models/mamba2_2_7b_continuous_learning/model.py` (the file is currently byte-identical to `mamba2_780m/model.py` except for `MODEL_ID`).

- [ ] **Step 2: Apply the identical `__init__.py` change as Task 2 Step 2**

Same edits, applied to `models/mamba2_2_7b_continuous_learning/__init__.py`.

- [ ] **Step 3: Verify the model's config actually has `tie_embeddings` set**

Run:
```bash
cat .cache/huggingface/hub/models--state-spaces--mamba2-2.7b/snapshots/*/config.json
```
If the cache doesn't have this checkpoint yet, skip the load-and-assert sanity check from Task 2 Step 3 for this model (it requires downloading the ~2.7B checkpoint) — note in the README instead that resizing was verified via `mamba2_780m` and `extend_embeddings`'s unit tests, and applies generically based on `model.config.tie_embeddings` for any Mamba2 checkpoint.

- [ ] **Step 4: Add the "Special tokens" README section**

Same content as Task 2 Step 4's README addition, adapted: if this model's config has `tie_embeddings: true`, use identical wording to `mamba2_780m/README.md`; if `false`, change "re-tying them (`tie_embeddings: true` in this model's config)" to "growing them independently (`tie_embeddings: false` in this model's config)".

- [ ] **Step 5: Commit**

```bash
git add models/mamba2_2_7b_continuous_learning/model.py models/mamba2_2_7b_continuous_learning/__init__.py models/mamba2_2_7b_continuous_learning/README.md
git commit -m "Register [USER]/[ASSISTANT] as special tokens for mamba2_2_7b_continuous_learning"
```

---

### Task 4: Switch `sft/prepare_data.py` to `build_tokenizer`

**Files:**
- Modify: `sft/prepare_data.py:1-91` (imports and tokenizer construction)

**Interfaces:**
- Consumes: `models.common.build_tokenizer(model_mod)` from Task 1.

- [ ] **Step 1: Replace ad hoc tokenizer construction**

In `sft/prepare_data.py`, change:

```python
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

_model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
TOKENIZER_ID = _model_mod.TOKENIZER_ID
USER_OPEN = _model_mod.USER_OPEN
ASST_OPEN = _model_mod.ASST_OPEN
```

to:

```python
import torch
from dotenv import load_dotenv

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.common import build_tokenizer

_model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
USER_OPEN = _model_mod.USER_OPEN
ASST_OPEN = _model_mod.ASST_OPEN
```

(The `from models.common import build_tokenizer` line must come after the `sys.path.insert` call, since `models` isn't importable before that.)

Then in `main()`, change:

```python
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
```

to:

```python
    tokenizer = build_tokenizer(_model_mod)
```

- [ ] **Step 2: Insert the separator explicitly in `format_conversation`**

`USER_OPEN`/`ASST_OPEN` no longer carry a trailing space (Task 2/3), so the two
lines that build turn text must add it. Change:

```python
        if role == "user":
            text = USER_OPEN + content + "\n"
            turn_ids = tokenizer.encode(text, add_special_tokens=False)
            turn_mask = [False] * len(turn_ids)
        elif role == "assistant":
            text = ASST_OPEN + content
```

to:

```python
        if role == "user":
            text = USER_OPEN + " " + content + "\n"
            turn_ids = tokenizer.encode(text, add_special_tokens=False)
            turn_mask = [False] * len(turn_ids)
        elif role == "assistant":
            text = ASST_OPEN + " " + content
```

- [ ] **Step 3: Verify behavior unchanged for non-special-token content**

Run:
```bash
cd sft && HF_HOME=$(pwd)/../.cache/huggingface PYTHONPATH=$(pwd)/.. uv run --no-sync python -c "
import json, tempfile, subprocess, sys
record = {'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]}
with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    f.write(json.dumps(record) + chr(10))
    path = f.name
subprocess.run([sys.executable, 'prepare_data.py', '--input', path, '--output', '/tmp/test_out.pt', '--max-len', '64'], check=True)
import torch
data = torch.load('/tmp/test_out.pt', weights_only=False)
print('examples:', len(data['ids']))
print('any trainable:', bool(data['masks'][0].any()))
"
```
Expected: `examples: 1`, `any trainable: True`, no errors.

- [ ] **Step 4: Commit**

```bash
git add sft/prepare_data.py
git commit -m "sft: use models.common.build_tokenizer and explicit role-marker separator"
```

---

### Task 5: Switch `backend/app/model/registry.py` to `build_tokenizer`

**Files:**
- Modify: `backend/app/model/registry.py:1-70`

**Interfaces:**
- Consumes: `models.common.build_tokenizer(model_mod)` from Task 1.

- [ ] **Step 1: Replace ad hoc tokenizer construction in `_load_blocking`**

In `backend/app/model/registry.py`, change:

```python
import asyncio
import importlib

import torch
from transformers import AutoTokenizer
from mamba_ssm.utils.generation import InferenceParams
```

to:

```python
import asyncio
import importlib

import torch
from mamba_ssm.utils.generation import InferenceParams

from models.common import build_tokenizer
```

Then change:

```python
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID)
```

to:

```python
        self.tokenizer = build_tokenizer(model_mod)
```

Note: `from models.common import build_tokenizer` works at module-import time here because the backend is always started with `PYTHONPATH` pointing at the repo root (per `models/CLAUDE.md`/root `CLAUDE.md`), unlike `sft/prepare_data.py` which manually inserts the repo root into `sys.path` at runtime — no `sys.path` change needed in this file.

- [ ] **Step 2: Compute the combined display/strip prefix once, on assignment**

`model_mod.USER_OPEN`/`ASST_OPEN` no longer carry a trailing space (Task 2/3).
`registry.user_open`/`asst_open` are read by `append_message` (string concatenation),
`inference.py` (prefix strip), `config.py` (`/config` payload), and ultimately the
frontend — all of which expect the trailing space, so it's added here, once. Change:

```python
        self.user_open = model_mod.USER_OPEN
        self.asst_open = model_mod.ASST_OPEN
```

to:

```python
        self.user_open = model_mod.USER_OPEN + " "
        self.asst_open = model_mod.ASST_OPEN + " "
```

No other line in this file changes — `append_message` already just concatenates
`self.user_open`/`self.asst_open` with `content`, so it's unaffected by this edit.

- [ ] **Step 3: Verify the backend still starts and loads the model**

Run: `cd backend && make dev` (start it, watch the startup log, then stop it)
Expected: log shows `loading model: mamba2_780m` (or whatever `MODEL_NAME` is set to) followed by no traceback, and the server reaches a state where it's listening (no `ModuleNotFoundError` or `AttributeError` about `AutoTokenizer`/`build_tokenizer`).

- [ ] **Step 4: Commit**

```bash
git add backend/app/model/registry.py
git commit -m "backend: use models.common.build_tokenizer and explicit role-marker separator"
```
