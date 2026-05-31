# CL CLI

Interactive command-line client for the continual learning backend. Generates tokens one at a time and lets you rate each one, collecting reward data for critic training.

## Setup

Install once as a system-wide tool via uv:

```bash
uv tool install --editable continual-learning/cli
```

The `cl` command is then available in your PATH. Because it's an editable install, changes to `main.py` take effect immediately without reinstalling.

## Usage

### Chat (main loop)

```bash
cl chat
```

Continues the existing session if one is in progress, or prompts for initial text if the session is empty. Generates one token at a time — after each token the full conversation is shown with the newest token highlighted.

```
Continuing session (12 tokens)

─────────────────────────────────────────
Once upon a time there was a [king]
─────────────────────────────────────────
critic: 0.0023  (random — critic not yet trained)
Reward [-1..1, Enter=skip, q=quit]: 0.8
  ✓ saved (total: 1)
```

Pass `--reset` to discard the current session and start fresh:

```bash
cl chat --reset
```

- **Enter** — skip, no reward saved
- **float** — save reward in [-1, 1]
- **`history`** — print full token list with IDs
- **`q`** — quit

### Other commands

```bash
cl health                        # server status
cl mode get                      # current mode (frozen/unfrozen)
cl mode set frozen               # switch mode
cl session get                   # print current session token history
cl session reset --text "Hello"  # reset conversation to given text
```

All commands accept `--url` to point at a non-default server:

```bash
cl chat --url http://myserver:8000
```

## Default server

`http://localhost:8000` — start the backend with `make dev` from `continual-learning/backend/`.
