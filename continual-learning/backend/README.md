# CL Backend

FastAPI server that loads the continual learning model and exposes API endpoints for token generation and session management. It exists for testing the model interactively; the model generates tokens until EOS.

## Requirements

- Python 3.14+
- ROCm 6.4 / HIP 7.2 (AMD GPU) — see note below for CUDA
- `uv`
- ROCm development headers: `rocm-devel`, `rocm-hip-devel`, `rocthrust-devel`

## Setup

```bash
cd continual-learning/backend
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
| `SFT_CHECKPOINT` | _(unset)_ | Path to an SFT checkpoint directory (e.g. `../../sft/checkpoints/step-1200`). When set, LoRA adapter weights are applied on top of the base model at startup. Rank and alpha are read automatically from `lora_config.json` inside the checkpoint. Leave unset to run the plain base model. |

## Running

```bash
make dev
```

Starts the server on `http://localhost:8000` with hot reload. The model loads at startup and runs a warmup forward pass before accepting requests — startup takes ~30s after kernels are cached.

For production (no reload, all interfaces):

```bash
make run
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status, whether the model is loaded |
| GET | `/device` | Which GPU the model is on |
| GET | `/session` | Full token history (source of truth) |
| DELETE | `/session` | Clear the session |
| PUT | `/session` | Append user text — `{"text": "tell me more"}` |
| POST | `/generate` | Generate a response (batch) — `{"temperature": 0.8, "top_p": 0.95}` |
| POST | `/generate/stream` | Same, but stream tokens as NDJSON as they're generated |

**Generation:** both endpoints run until the model emits EOS (`<|endoftext|>`) or `max_tokens` is reached (default 512). `/generate` returns the whole response at once: `generated_text` is the clean readable output with the trailing EOS token stripped, while the per-token `tokens` list still includes it flagged `is_eos: true`, so you can tell a natural stop from a `max_tokens` cutoff. `/generate/stream` emits one JSON object per line (`{"token", "token_id", "is_eos"}`) as each token is produced — used by the CLI for live-updating output. The EOS token is kept in the session history either way, so it's fed back into the model on continued generation.

The model is being fine-tuned to emit a `<revise>` tag after its response. For now that tag is generated as ordinary text — detecting it and acting on it is a TODO (see the top-level `README.md`).
