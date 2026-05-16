# CLI

Lightweight command-line client for the continual-learning backend. No model loading — all commands talk to a running backend over HTTP.

## Setup

```bash
make install-cli   # from repo root
# or:
python3 -m venv cli/.venv
cli/.venv/bin/pip install -e "cli[dev]"
```

## Usage

```
continual-learning [--server URL] <command>
```

The `--server` option defaults to `http://localhost:8000`. You can also set it via the `CL_SERVER_URL` environment variable.

### Commands

**generate** — generate a response (streams to stdout by default)
```bash
continual-learning generate "Explain transformers in one sentence."
continual-learning generate "Hello" --max-tokens 64
continual-learning generate "Hello" --no-stream   # non-streaming endpoint
```

**train critic** — Phase 1: train the critic on your reward signal
```bash
continual-learning train critic \
  --prompt "Explain transformers." \
  --response "Transformers use attention mechanisms." \
  --reward 0.8
```

**train policy** — Phase 2: update the base model using the critic's reward
```bash
continual-learning train policy \
  --prompt "Explain transformers." \
  --response "Transformers use attention mechanisms."
```

**save** — manually save a checkpoint
```bash
continual-learning save
```

**status** — show current step and checkpoint info
```bash
continual-learning status
```

**history** — show recent training history
```bash
continual-learning history         # last 10 entries
continual-learning history --n 25
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `CL_SERVER_URL` | Backend URL (default: `http://localhost:8000`) |

## Testing

```bash
make test-cli   # from repo root
# or:
cli/.venv/bin/pytest cli/tests
```
