# Claude Guidelines — altrux

## Cache policy

All caches must live **inside the project subfolder**, not in system or home directories. Never let HuggingFace, uv, or pip write to `~/.cache`.

Each project's `Makefile` must set these env vars on every command that touches the network or model weights:

```makefile
HF_HOME=.cache/huggingface
UV_CACHE_DIR=.cache/uv
```

Each project's `.gitignore` must ignore `.cache/`.

This keeps environments reproducible and self-contained — a `rm -rf sft/` or `rm -rf continual-learning/backend/` cleanly removes everything.
