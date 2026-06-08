# Claude Guidelines — continual-learning

## Always update the README and .env.example

Whenever you change user-facing behaviour — a new endpoint, a new CLI flag, changed defaults, a removed feature — update the relevant README(s) in the same change. The READMEs to keep in sync are:

- `continual-learning/README.md` — architecture and high-level design
- `continual-learning/backend/README.md` — API reference, setup, running
- `continual-learning/frontend/README.md` — frontend setup, dev server, environment

For the backend README, keep the API table up to date with any new or changed endpoints. For the frontend README, keep the setup steps and any env vars accurate.

Whenever you add, rename, or remove a backend environment variable, also update `continual-learning/backend/.env.example` with the matching entry and a short comment.

Whenever you add, rename, or remove a frontend environment variable, also update `continual-learning/frontend/.env.example` with the matching entry and a short comment.

## Frontend

See `frontend/CLAUDE.md` for frontend-specific rules (quality checks, Tailwind colour policy).

## Cache policy

HuggingFace cache is shared at the repo root (`../../.cache/huggingface` from `backend/`). The backend `Makefile` sets `HF_HOME=$(CURDIR)/../../.cache/huggingface` on every relevant target. See the root `CLAUDE.md` for the full policy.
