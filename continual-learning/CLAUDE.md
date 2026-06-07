# Claude Guidelines — continual-learning

## Always update the README

Whenever you change user-facing behaviour — a new endpoint, a new CLI flag, changed defaults, a removed feature — update the relevant README(s) in the same change. The READMEs to keep in sync are:

- `continual-learning/README.md` — architecture and high-level design
- `continual-learning/backend/README.md` — API reference, setup, running
- `continual-learning/cli/README.md` — CLI commands and flags
- `continual-learning/frontend/README.md` — frontend setup, dev server, environment

For the CLI README, don't enumerate every flag — just keep the usage example and key behaviour accurate, and point users to `--help` for the full option list. For the backend README, keep the API table up to date with any new or changed endpoints. For the frontend README, keep the setup steps and any env vars accurate.

## Frontend

See `frontend/CLAUDE.md` for frontend-specific rules (quality checks, Tailwind colour policy).

## Cache policy

HuggingFace cache is shared at the repo root (`../../.cache/huggingface` from `backend/`). The backend `Makefile` sets `HF_HOME=$(CURDIR)/../../.cache/huggingface` on every relevant target. See the root `CLAUDE.md` for the full policy.
