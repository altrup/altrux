.PHONY: sync install dev run

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=auto uv pip install torch

install: sync

dev:
	uv run --no-sync uvicorn app.main:app --reload

run:
	uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000