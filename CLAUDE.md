# Claude Guidelines — altrux

## Cache policy

All HuggingFace model/tokenizer caches live in **`.cache/huggingface/` at the repo root** (gitignored), shared across all subprojects. Never write to `~/.cache`.

Each subproject's `Makefile` must point `HF_HOME` at the shared root cache using `$(CURDIR)` so the path is always correct regardless of where make is invoked:

```makefile
# one level deep (e.g. sft/)
HF_CACHE := $(CURDIR)/../.cache/huggingface

# two levels deep (e.g. continual-learning/backend/)
HF_CACHE := $(CURDIR)/../../.cache/huggingface
```

uv uses its default system cache (`~/.cache/uv`) — no override needed. The root `.gitignore` covers the shared `.cache/`.
