HF_CACHE := $(CURDIR)/../.cache/huggingface
PYTHONPATH := $(CURDIR)/..

.PHONY: sync install dev run spike

sync:
	uv sync
	UV_TORCH_BACKEND=auto uv pip install torch
	MAX_JOBS=12 uv pip install causal-conv1d mamba-ssm --no-build-isolation

install: sync

spike:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python spike/check_load.py

dev:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m uvicorn app.main:app --reload

run:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
