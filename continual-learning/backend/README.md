# CL Backend

FastAPI server that loads the continual learning model and exposes API endpoints for token generation, per-token reward collection, and session management.

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
| GET | `/health` | Status, mode, records collected |
| GET | `/device` | Which GPU the model is on |
| GET | `/mode` | Current mode (`frozen` / `unfrozen`) |
| POST | `/mode` | Switch mode — `{"mode": "frozen"}` |
| GET | `/session` | Full token history (source of truth) |
| DELETE | `/session` | Clear the session |
| PUT | `/session` | Append user text — `{"text": "tell me more"}` |
| POST | `/generate` | Generate one token — `{"temperature": 0.8, "top_p": 0.95}` |
| POST | `/reward` | Submit reward — `{"reward": 0.8}` |

**Modes:**
- `frozen` — no parameter updates; data is collected for critic training
- `unfrozen` — base model parameters trainable, critic frozen (no train step in v1)

## Data

Collected training data is written to `data/collected/` (gitignored):
- `trunk_hiddens.dat` — raw float32 bytes, one `d_model`-length vector per record
- `records.jsonl` — `{"index", "token_id", "reward"}` per line

Both files are O(1)-append and reconciled on startup to recover from crashes.
