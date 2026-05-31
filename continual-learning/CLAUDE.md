# Claude Guidelines — continual-learning

## Always update the README

Whenever you change user-facing behaviour — a new endpoint, a new CLI flag, changed defaults, a removed feature — update the relevant README(s) in the same change. The READMEs to keep in sync are:

- `continual-learning/README.md` — architecture and high-level design
- `continual-learning/backend/README.md` — API reference, setup, running
- `continual-learning/cli/README.md` — CLI commands and flags

For the CLI README, don't enumerate every flag — just keep the usage example and key behaviour accurate, and point users to `--help` for the full option list. For the backend README, keep the API table up to date with any new or changed endpoints.
