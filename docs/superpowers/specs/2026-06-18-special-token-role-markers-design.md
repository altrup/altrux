# Special tokens for USER_OPEN / ASST_OPEN role markers

## Problem

`USER_OPEN` ("[USER] ") and `ASST_OPEN` ("[ASSISTANT] ") are plain strings spliced
into prompt text and tokenized as ordinary BPE pieces (3+ tokens each). They are not
members of the tokenizer's vocabulary, so the model has no atomic, unambiguous signal
for "a new turn started here" — and the trailing space exists only to control how the
following word's BPE piece is chosen.

## Goal

Register the role markers as real tokenizer special tokens (one atomic token id each),
and resize the model's embedding/lm_head to accommodate them, while keeping the actual
tokenized prompt text identical to today's so SFT data shape doesn't change beyond the
marker itself.

## Decisions

### 1. `USER_OPEN`/`ASST_OPEN` become bare markers; the separator is explicit wherever text is built

`USER_OPEN = "[USER]"` and `ASST_OPEN = "[ASSISTANT]"` — no trailing space. These are
the single source of truth: they're exactly what gets registered via
`tokenizer.add_special_tokens({"additional_special_tokens": [USER_OPEN, ASST_OPEN]})`,
so there's no second, separately-maintained "bare" list shadowing them. A registered
special token is matched as a literal atomic substring wherever it appears in input
text during `encode()`, regardless of the `add_special_tokens=False` flag (that flag
only controls auto-prepended BOS/EOS) — so this doesn't require the marker to be
adjacent to anything in particular; it's recognized atomically no matter what follows.

Baking a trailing space into the marker's own string value conflates two different
concerns: the marker is a vocab-level concept (must be unambiguous and content-free),
while the whitespace separating it from the turn's content is a prompt-formatting
concern. Keeping them separate (as e.g. ChatML/Llama-style chat templates do) means
every place that builds display- or training-ready text appends the separator
explicitly: `USER_OPEN + " " + content + "\n"`. The separator stays a literal `" "`
(not switched to `"\n"` or anything else) so the resulting byte sequence — and
therefore `content`'s tokenization — is identical to today's; only the marker itself
collapses from several BPE pieces into one atomic id.

In practice only two call sites need to know about the bare marker at all:
`models/<model>/model.py` (defines it, includes it in `SPECIAL_TOKENS`) and
`backend/app/model/registry.py`, which computes the combined display/strip prefix
once — `self.asst_open = model_mod.ASST_OPEN + " "` — and stores *that*. Every other
consumer of `registry.asst_open`/`registry.user_open` (`backend/app/routers/inference.py`'s
strip, `backend/app/routers/config.py`'s `/config` payload, and the frontend's
`asstOpen` prefix-buffering in `frontend/app/lib/api.ts`) already only ever reads
`registry.asst_open`/`registry.user_open`, never the model module's constant
directly — so none of them change. Only `sft/prepare_data.py`'s two text-building
lines need the explicit `+ " " +`, since it reads `_model_mod.USER_OPEN`/`ASST_OPEN`
directly rather than through the registry.

### 2. Shared helper: `models/common.py`

Both model dirs currently duplicate identical `TOKENIZER_ID`/`USER_OPEN`/`ASST_OPEN`.
Special-token registration and embedding resize must use byte-identical logic across
models to keep vocab/id assignment deterministic, so it lives in one place:

```python
# models/common.py
def build_tokenizer(model_mod) -> PreTrainedTokenizer:
    """Load TOKENIZER_ID, ensure eos_token exists, register SPECIAL_TOKENS."""

def extend_embeddings(model, new_vocab_size: int) -> None:
    """Grow backbone.embedding and lm_head to new_vocab_size in place.
    Copies existing rows, initializes new rows (normal init matching the
    existing embedding's std), re-ties lm_head.weight to embedding.weight
    if model.config.tie_embeddings."""
```

`MambaLMHeadModel` is a plain `nn.Module`, not a HF `PreTrainedModel` — there is no
built-in `resize_token_embeddings()`, so `extend_embeddings` is custom. It must not
assume any existing padding headroom (e.g. `pad_vocab_size_multiple` slack) covers the
new tokens; it always computes the required size from the tokenizer and resizes if
needed, growing both `backbone.embedding` and `lm_head` and re-tying them when
`tie_embeddings` is set in config (true for `mamba2-780m`).

### 3. Model module interface change

Each model module's existing `USER_OPEN`/`ASST_OPEN` change shape (decision 1), and
one new required export is added, defined directly from them:

```python
USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
SPECIAL_TOKENS: list[str] = [USER_OPEN, ASST_OPEN]
```

`load_base(device)` and `load_inference(device)` call `models.common.build_tokenizer`
and `models.common.extend_embeddings` internally so the returned model already has the
grown embedding/lm_head — callers (`sft/prepare_data.py`, `backend/app/model/registry.py`)
build their own tokenizer via the same `build_tokenizer` helper and get matching ids
because token registration order is deterministic given the same base tokenizer + same
`SPECIAL_TOKENS` list.

### 4. Call-site changes

- `sft/prepare_data.py`: replace ad hoc `AutoTokenizer.from_pretrained` + manual eos
  fallback with `models.common.build_tokenizer(_model_mod)`. `format_conversation`
  inserts the separator explicitly: `USER_OPEN + " " + content + "\n"` and
  `ASST_OPEN + " " + content` (decision 1).
- `backend/app/model/registry.py`: replace ad hoc tokenizer loading with
  `build_tokenizer`. `_load_blocking` calls `load_base`, which now returns a model
  whose embeddings already match the tokenizer built the same way. The two
  assignment lines become `self.user_open = model_mod.USER_OPEN + " "` /
  `self.asst_open = model_mod.ASST_OPEN + " "` — `append_message`'s body is
  otherwise unchanged, since it already just concatenates `self.user_open`/
  `self.asst_open` with content.
- `backend/app/routers/inference.py`, `backend/app/routers/config.py`,
  `frontend/app/lib/api.ts`: unaffected — all three only ever read
  `registry.asst_open`/`registry.user_open`, which still carry the trailing space,
  so none of their string logic changes.

### 5. Known characteristic, not a blocker

`TARGET_LORA_MODULES = ["in_proj", "out_proj"]` doesn't include the embedding or
`lm_head`, so the new tokens' embedding rows are frozen at their initial (randomly
initialized, untrained) values during LoRA SFT. The model can only learn to use the
new markers by having the LoRA-adapted `in_proj` learn to respond to those specific
(fixed) embedding vectors — not by adjusting the embeddings themselves. This is the
same pattern used elsewhere for adding vocab to frozen-embedding fine-tunes and is
expected to work, just worth noting in the model READMEs.

### 6. Checkpoint impact

No migration needed — current checkpoints are disposable test runs (confirmed with
user). Vocab size changes, so existing checkpoints become incompatible and would need
retraining.

## Files touched

- `models/common.py` (new)
- `models/mamba2_780m/model.py`, `models/mamba2_780m/__init__.py`, `models/mamba2_780m/README.md`
- `models/mamba2_2_7b_continuous_learning/model.py`, `__init__.py`, `README.md`
- `models/CLAUDE.md` (interface table: add `SPECIAL_TOKENS`)
- root `CLAUDE.md` (model export table: add `SPECIAL_TOKENS`)
- root `README.md` ("Adding a new model" required-exports list: add `SPECIAL_TOKENS`)
- `sft/prepare_data.py`
- `backend/app/model/registry.py`

## Testing

No existing test framework in this repo (verification is via manual "spike" scripts).
Add `pytest` as a dev dependency in `sft/pyproject.toml` (the project that already
pulls in `transformers`/`torch` and imports the `models` package) and write real unit
tests against `models/common.py` — these must not require downloading the real
multi-GB checkpoints or a GPU:

- `extend_embeddings`: build a tiny fake `nn.Module` with `backbone.embedding`
  (`nn.Embedding(10, 4)`), `lm_head` (`nn.Linear(4, 10, bias=False)`), and a
  `config.tie_embeddings` flag. Assert post-resize shapes, that old rows are
  preserved, and that `lm_head.weight is backbone.embedding.weight` when tied.
- `build_tokenizer`: use a small real tokenizer (e.g. `gpt2`, already tiny and
  commonly cached) with a fake `model_mod` namespace exposing `TOKENIZER_ID` and
  `SPECIAL_TOKENS`. Assert `len(tokenizer)` grew by exactly `len(SPECIAL_TOKENS)`,
  that `tokenizer.encode("[USER] hello")` contains the new token id exactly once
  as a single id (not split into multiple ids), and that an `eos_token` exists
  afterward.
- Round-trip check: encoding `USER_OPEN + " hello\n"` (i.e. `"[USER] hello\n"`,
  matching what `format_conversation`/`append_message` actually build) produces
  `[user_token_id] + <ids for " hello\n">`, where the tail matches
  `tokenizer.encode(" hello\n", add_special_tokens=False)` exactly — proving
  `content` tokenization is unchanged by the marker becoming atomic.
