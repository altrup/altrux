.PHONY: install test test-cov

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest

test-cov:
	.venv/bin/pytest --cov=continual_learning --cov-report=term-missing
