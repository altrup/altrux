# Backend

FastAPI server that loads the continual learning model and exposes API endpoints for token generation and session management. It exists for testing the model interactively; the model generates tokens until EOS.

## Requirements

- Python 3.14+
- ROCm 6.4 / HIP 7.2 (AMD GPU) — see note below for CUDA
- `uv`
- ROCm development headers: `rocm-devel`, `rocm-hip-devel`, `rocthrust-devel`

## Setup

```bash
cd backend
make sync
```

`make sync` does three things in order:
1. `uv sync` — installs pure-Python dependencies from `uv.lock`
2. `UV_TORCH_BACKEND=auto uv pip install torch` — installs the correct torch build for your hardware (ROCm or CUDA)
3. `MAX_JOBS=12 uv pip install causal-conv1d mamba-ssm --no-build-isolation` — compiles the Mamba2 CUDA/ROCm kernels against the torch you just installed

> **First run**: kernel compilation takes several minutes. Compiled kernels are cached in `~/.triton/cache/` — subsequent runs are fast.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `auto` uses CUDA/ROCm if available |
| `MODEL_NAME` | `mamba2_780m` | Filename (without `.py`) of the model in `models/` at the repo root. The model file defines the architecture, tokenizer, format tokens, and LoRA targets. |
| `SFT_CHECKPOINT` | _(unset)_ | Path to an SFT checkpoint directory (e.g. `../sft/checkpoints/step-1200`). When set, LoRA adapter weights are applied on top of the base model at startup. Rank and alpha are read automatically from `lora_config.json` inside the checkpoint. Leave unset to run the plain base model. |
| `REVISE_DATA_PATH` | `../sft/data/revise_collected.jsonl` | Path to the JSONL file where revision suggestions are appended. Relative paths are resolved from the backend directory. |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated list of allowed CORS origins. |

Chat format tokens (`USER_OPEN`, `ASST_OPEN`) are defined in the model file (`models/{MODEL_NAME}.py`) and are **not** configured here. The frontend fetches them automatically via `GET /config` at startup.

## Running

```bash
make dev
```

Starts the server on `http://localhost:8000` with hot reload. The model loads at startup and runs a warmup forward pass before accepting requests — startup takes ~30s after kernels are cached.

For production (no reload, all interfaces):

```bash
make run
```

> The Makefile sets `PYTHONPATH` to the repo root so `models/` is importable. If running manually (outside make), set `PYTHONPATH=/path/to/repo` before starting uvicorn.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, whether the model is loaded |
| GET | `/config` | Model chat format tokens (`user_open`, `asst_open`) |
| GET | `/device` | Which GPU the model is on |
| GET | `/session` | Full token history (source of truth) |
| DELETE | `/session` | Clear the session |
| PUT | `/session` | Append raw text — `{"text": "tell me more"}` |
| PUT | `/session/message` | Append a chat turn, wrapped with the model's openers — `{"role": "user", "content": "hi"}` |
| POST | `/session/revise` | Save a revision suggestion — `{"n": 1, "revision": "better answer", "weight": 0.5, "at_turn": 3}`. `n` is how many assistant turns back to target; `at_turn` is the message index of the anchor turn (defaults to the last assistant turn). |
| DELETE | `/session/revise` | Remove a single revision suggestion — `{"at_turn": 3, "n": 1}`. Returns 404 if the revision does not exist. |
| POST | `/generate` | Generate a response (batch) — `{"temperature": 0.8, "top_p": 0.95}` |
| POST | `/generate/stream` | Same, but stream tokens as NDJSON as they're generated |

**Generation:** both endpoints run until the model emits EOS (`<|endoftext|>`) or `max_tokens` is reached (default 512). `/generate` returns the whole response at once; `/generate/stream` emits one JSON object per line (`{"token", "token_id", "is_eos"}`) as each token is produced.
