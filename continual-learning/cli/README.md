# CL CLI

Interactive command-line client for the continual learning backend. Generates a full response (until EOS) each turn, then lets you inject more text to continue the conversation.

## Setup

From this directory (`continual-learning/cli/`), install once as a system-wide tool via uv:

```bash
uv tool install --editable .
```

The `cl` command is then available in your PATH. Because it's an editable install, changes to `main.py` take effect immediately without reinstalling.

## Usage

### Chat (main loop)

```bash
cl chat
```

Continues the existing session if one is in progress, or prompts for initial text if the session is empty. Each turn the backend streams a response (until EOS) and the display updates live as tokens arrive, with the newly generated tokens highlighted. Then you're prompted to inject more text — submit empty to quit.

```
Continuing session (12 tokens)

─────────────────────────────────────────
Once upon a time there was a king who ruled a small kingdom.
─────────────────────────────────────────
Inject text into session (Enter on empty = quit):
```

Run `cl chat --help` for all options (e.g. `--temperature`, `--top-p`, `--reset`).

### Other commands

```bash
cl health                        # server status
cl session get                   # print current session token history
cl session reset --text "Hello"  # reset conversation to given text
```

All commands accept `--url` to point at a non-default server:

```bash
cl chat --url http://myserver:8000
```

## Default server

`http://localhost:8000` — start the backend with `make dev` from `continual-learning/backend/`.
