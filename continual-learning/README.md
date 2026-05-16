# Continual Learning Model

A system for training a hybrid attention+Mamba LLM with an attached critic branch, using human reward signals to guide online weight updates.

## Architecture

The model wraps [ibm-granite/granite-4.0-h-micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro) and adds a **critic branch** that taps into intermediate representations to produce a scalar reward in [-1, 1].

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
 Mamba Layers]                   (fresh attention + mamba layers,
     │                            2/3 depth of original model)
     ▼                                  │
[LM Head → Logits]                      ▼
                                 [Linear → tanh → scalar ∈ [-1, 1]]
```

The critic branches off at the **2/3 depth mark**, has **2/3 the depth** of the base model, and outputs a reward through `tanh` so it stays in [-1, 1].

Training runs in two phases:
- **Phase 1** — human rates responses; critic trained via MSE to predict those ratings (base model frozen)
- **Phase 2** — critic frozen; its reward signal backpropagates into the base model

## Structure

| Section | Path | Description |
|---------|------|-------------|
| Backend | [backend/](backend/) | FastAPI server: model, trainer, REST API with SSE streaming |
| Frontend | [frontend/](frontend/) | React + Vite web UI |
| CLI | [cli/](cli/) | Click-based CLI that talks to the backend over HTTP |
| Checkpoints | [checkpoints/](checkpoints/) | Shared checkpoint directory (not owned by any section) |

## Quick start

```bash
cp .env.example .env          # edit CL_MODEL, CL_QUANTIZE, CL_DEVICES, etc.
make install                  # create venvs, install all dependencies

# terminal 1
make dev-backend              # FastAPI on http://localhost:8000

# terminal 2
make dev-frontend             # Vite on http://localhost:5173
```

See each section's README for full details:
- [backend/README.md](backend/README.md) — env vars, API reference
- [frontend/README.md](frontend/README.md) — dev server, component overview
- [cli/README.md](cli/README.md) — subcommands, env vars

## Testing

```bash
make test            # run all tests (backend + cli + frontend)
make test-backend    # backend unit + API tests only
make test-cli        # CLI tests only
make test-frontend   # Vitest frontend tests only
```
