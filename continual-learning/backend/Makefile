.PHONY: sync install dev run spike

sync:
	uv sync
	UV_TORCH_BACKEND=auto uv pip install torch

install: sync

spike:
	uv run --no-sync python spike/check_load.py

dev:
	HSA_ENABLE_INTERRUPT=1 \
	uv run --no-sync uvicorn app.main:app --reload

run:
	HSA_ENABLE_INTERRUPT=1 \
	uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000