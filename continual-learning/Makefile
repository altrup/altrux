.PHONY: dev-backend dev-frontend install install-backend install-frontend install-cli \
        test test-backend test-cli test-frontend

# ── install ───────────────────────────────────────────────────────────────────

install: install-backend install-frontend install-cli

install-backend:
	python3 -m venv backend/.venv
	backend/.venv/bin/pip install -e "backend[dev]"

install-frontend:
	cd frontend && npm install

install-cli:
	python3 -m venv cli/.venv
	cli/.venv/bin/pip install -e "cli[dev]"

# ── dev (run each in a separate terminal) ────────────────────────────────────

# Loads all CL_* config from .env (copy from .env.example and edit).
dev-backend:
	set -a && . ./.env && set +a && \
	  cd backend && .venv/bin/uvicorn cl_backend.main:app --reload --port $${CL_PORT:-8000}

dev-frontend:
	cd frontend && npm run dev

# ── test ──────────────────────────────────────────────────────────────────────

test: test-backend test-cli test-frontend

test-backend:
	backend/.venv/bin/pytest backend/tests

test-cli:
	cli/.venv/bin/pytest cli/tests

test-frontend:
	cd frontend && npm run test
