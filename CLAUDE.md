# Claude Guidelines — altrux

## Cache policy

All HuggingFace model/tokenizer caches live in **`.cache/huggingface/` at the repo root** (gitignored), shared across all subprojects. Never write to `~/.cache`.

Each subproject's `Makefile` must point `HF_HOME` at the shared root cache using `$(CURDIR)` so the path is always correct regardless of where make is invoked:

```makefile
# one level deep (e.g. sft/, backend/)
HF_CACHE := $(CURDIR)/../.cache/huggingface
```

uv uses its default system cache (`~/.cache/uv`) — no override needed. The root `.gitignore` covers the shared `.cache/`.

## Models

Each model is a folder in `models/` at the repo root containing `model.py` (implementation), a thin `__init__.py` that re-exports the interface below, and a `README.md` documenting the model (see `models/CLAUDE.md` for the README checklist). A model must export:

| Name | Type | Description |
|------|------|-------------|
| `MODEL_ID` | `str` | HuggingFace model identifier |
| `TOKENIZER_ID` | `str` | HuggingFace tokenizer identifier |
| `TARGET_LORA_MODULES` | `list[str]` | Module name suffixes to attach LoRA adapters to |
| `USER_OPEN` | `str` | Bare user-turn role marker, registered as a tokenizer special token. Callers append a literal `" "` separator before content. |
| `ASST_OPEN` | `str` | Bare assistant-turn role marker, registered as a tokenizer special token. Callers append a literal `" "` separator before content. |
| `SPECIAL_TOKENS` | `list[str]` | `[USER_OPEN, ASST_OPEN]` — the list passed to `tokenizer.add_special_tokens` |
| `Model` | `nn.Module` | Inference wrapper class |
| `load_base(device)` | `fn` | Load raw HF model (used by sft) |
| `load_inference(device)` | `fn` | Load and wrap for inference (used by backend) |

Both `backend` and `sft` read `MODEL_NAME` from their `.env` and import `models.{MODEL_NAME}` at startup. The backend must be started with `PYTHONPATH` pointing at the repo root (the Makefile handles this). `sft` scripts also add the repo root to `sys.path` automatically.

## Always update READMEs and .env.example

Whenever you change user-facing behaviour — a new endpoint, a new CLI flag, changed defaults, a removed feature — update the relevant README(s) in the same change. The READMEs to keep in sync are:

- `README.md` — architecture and high-level design
- `backend/README.md` — API reference, setup, running
- `frontend/README.md` — frontend setup, dev server, environment
- `sft/README.md` — training guide, CLI flags

For the backend README, keep the API table up to date with any new or changed endpoints.

Whenever you add, rename, or remove a backend environment variable, also update `backend/.env.example`.
Whenever you add, rename, or remove a frontend environment variable, also update `frontend/.env.example`.
Whenever you add, rename, or remove an sft environment variable, also update `sft/.env.example`.

## Frontend

See `frontend/CLAUDE.md` for frontend-specific rules (quality checks, Tailwind colour policy).
