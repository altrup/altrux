HF_CACHE := $(CURDIR)/../.cache/huggingface
PYTHONPATH := $(CURDIR)/..

.PHONY: sync install dev run spike

sync:
	uv sync
	UV_TORCH_BACKEND=auto uv pip install torch --reinstall
	MAX_JOBS=12 uv pip install mamba-ssm --no-build-isolation
	@if uv run --no-sync python -c "import torch, sys; sys.exit(0 if (torch.cuda.is_available() and torch.version.hip is None) else 1)"; then \
		echo "CUDA (non-ROCm) GPU detected -- installing causal-conv1d (models/mamba2_2_7b_memory/model.py's _mixer_step already uses causal_conv1d_update when importable)"; \
		MAX_JOBS=12 uv pip install causal-conv1d --no-build-isolation; \
	else \
		echo "No CUDA (non-ROCm) GPU detected -- skipping causal-conv1d (broken on this project's ROCm gfx1102 dev box, see root CLAUDE.md); models fall back to the manual per-token conv"; \
	fi

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
