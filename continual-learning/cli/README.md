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

The conversation flows inline like a transcript. You type after the `[USER] ` prompt; the backend wraps it with the configured `[USER] ` opener, so you don't type that by hand. The assistant response then streams in place below it (in green) — the model emits its own `[ASSISTANT] ` opener, so it appears at the start of the response. Then the next `[USER] ` prompt appears. Submit an empty message to quit.

```
Enter = new line  |  Meta+Enter (or Esc then Enter) = submit  |  empty = quit
[USER] Hello
[ASSISTANT] How can I help you today?
[USER]
```

Run `cl chat --help` for all options (e.g. `--temperature`, `--top-p`, `--reset`).

### Other commands

```bash
cl health                        # server status
cl session get                   # print current session token history
cl session message "Hello"       # append a chat turn (wrapped with role openers)
cl session reset --text "Hello"  # reset conversation to given raw text
```

All commands accept `--url` to point at a non-default server:

```bash
cl chat --url http://myserver:8000
```

## Default server

`http://localhost:8000` — start the backend with `make dev` from `continual-learning/backend/`.
