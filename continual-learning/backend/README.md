# Backend

FastAPI server that runs the `ContinualLearningModel` and exposes a REST API.

## Setup

```bash
make install-backend   # from repo root
```

Or manually:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -e "backend[dev]"
```

### PyTorch: CUDA vs ROCm

`pip install torch` installs the CUDA (NVIDIA) build by default. If you have an AMD GPU you must explicitly install the ROCm build, otherwise `torch.cuda.is_available()` returns `False` and the model runs on CPU.

**NVIDIA GPU:**
```bash
backend/.venv/bin/pip install torch torchvision torchaudio
```

**AMD GPU (ROCm 6.4):**
```bash
backend/.venv/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.4
```

Check which ROCm version you have with `rocminfo --version`, then find the matching wheel at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

Verify GPU is visible after installing:
```bash
backend/.venv/bin/python -c "import torch; print(torch.__version__); print('GPU count:', torch.cuda.device_count())"
```

## Configuration (environment variables)

All config is read from environment variables (with `CL_` prefix). Copy `.env.example` to `.env` at the repo root and edit as needed. The dev Makefile target sources it automatically.

| Variable | Default | Description |
|----------|---------|-------------|
| `CL_MODEL` | `ibm-granite/granite-4.0-h-micro` | HuggingFace model ID |
| `CL_QUANTIZE` | _(none)_ | `4bit` or `8bit` quantization |
| `CL_CHECKPOINT_DIR` | `../checkpoints` | Path to checkpoint directory |
| `CL_NO_RESUME` | `0` | Set to `1` to start fresh, ignoring any checkpoint |
| `CL_CRITIC_LR` | `1e-4` | Critic branch learning rate |
| `CL_BASE_LR` | `1e-6` | Base model learning rate |
| `CL_SAVE_EVERY` | `10` | Auto-save every N steps; `0` to disable |
| `CL_KEEP_CHECKPOINTS` | `1` | Numbered snapshots to keep; `0` = keep all |
| `CL_DEVICES` | _(all)_ | Comma-separated CUDA indices, e.g. `0,1` |
| `CL_PORT` | `8000` | Server port |

## Dev server (with live reload)

```bash
make dev-backend    # from repo root
```

API docs available at `http://localhost:8000/docs`.

## API reference

### Generate

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Full generate (waits for completion) |
| `GET` | `/generate/stream` | SSE streaming generate |

**POST /generate**
```json
// request
{ "prompt": "Hello", "max_new_tokens": 256 }

// response
{ "response": "...", "estimated_reward": 0.42 }
```

**GET /generate/stream** — query params: `prompt`, `max_new_tokens`

SSE events:
```
data: {"chunk": "partial text"}
data: {"chunk": "full text", "estimated_reward": 0.42}
data: [DONE]
```

### Train

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/train/critic` | Phase 1 step |
| `POST` | `/train/policy` | Phase 2 step |
| `GET` | `/train/history` | Recent history (query: `n=10`) |

**POST /train/critic**
```json
// request
{ "prompt": "...", "response": "...", "user_reward": 0.5 }

// response
{ "loss": 0.04, "step": 1 }
```

**POST /train/policy**
```json
// request
{ "prompt": "...", "response": "..." }

// response
{ "reward": 0.6, "loss": -0.6, "step": 2 }
```

### Checkpoint

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/checkpoint/save` | Trigger manual save |
| `GET` | `/checkpoint/status` | Current step and save info |

## Testing

```bash
make test-backend    # from repo root
# or directly:
backend/.venv/bin/pytest backend/tests
```
