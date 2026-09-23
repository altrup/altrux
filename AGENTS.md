# Altrux repository instructions

## Cache policy

All HuggingFace model/tokenizer caches live in **`.cache/huggingface/` at the repo root** (gitignored), shared across all subprojects. Never write to `~/.cache`.

Each subproject's `Makefile` must point `HF_HOME` at the shared root cache using `$(CURDIR)` so the path is always correct regardless of where make is invoked:

```makefile
# one level deep (e.g. sft/, backend/)
HF_CACHE := $(CURDIR)/../.cache/huggingface
```

uv uses its default system cache (`~/.cache/uv`) — no override needed. The root `.gitignore` covers the shared `.cache/`.

## Running things — per-project uv, prefer the Makefiles

Each subproject (`sft/`, `backend/`) has its own `pyproject.toml` and its own `uv`-managed venv (`sft/.venv`, `backend/.venv`) — there is no root-level venv. `torch` and `mamba-ssm` are installed into each via `make sync`, not via `uv sync` alone, because they need hardware-specific builds (this hardware is ROCm): **any** bare `uv sync` or `uv pip install <other-package>` (even with `--no-deps`, confirmed) can silently re-resolve and install plain CUDA `torch` instead, clobbering the working ROCm build — this has happened twice in one session, from two different commands, neither of which was the obvious "installing torch" one. After *any* `uv pip install`/`uv sync` invocation, check `uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` — if you see a `+cu1xx` suffix or `nvidia-*` packages in the install log, fix it with `UV_TORCH_BACKEND=rocm6.4 uv pip install torch --reinstall`. Don't assume a narrowly-scoped install command is safe just because it doesn't mention torch.

Rebuilding this box's venv from scratch (2026-08-05 findings):

- `UV_TORCH_BACKEND=auto` (what `make sync` uses) resolves **CPU** torch here, not ROCm — the local rebuild needs `TORCH_BACKEND=rocm6.4 make sync` or the explicit reinstall above. On CUDA boxes `auto` behaves.
- This box has **no `hipcc`** (no ROCm SDK installed, only the runtime), so mamba-ssm's compiled-extension build fails. Install it with `MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install mamba-ssm --no-build-isolation` — the extension is exactly the thing `models/__init__.py` stubs out anyway.
- mamba-ssm ≥ 2.3 imports Mamba3 at package import, whose tilelang→tvm dependency dlopens `libtorch_cuda.so` — an `OSError` on ROCm torch that upstream's `except ImportError` guard misses. `models/__init__.py` pre-seeds a `mamba_ssm.modules.mamba3` stub on ROCm (same trick as `selective_scan_cuda`); nothing in this repo uses Mamba3.

Always prefer running through the subproject's `Makefile` targets (`make train`, `make test`, etc.) over ad-hoc `uv run` — the Makefiles set required env vars (`HF_HOME`, `PYTHONPATH`, and for GPU work `HSA_ENABLE_INTERRUPT=1` / `PYTORCH_CUDA_ALLOC_CONF`) that ad-hoc invocations easily miss. For one-off checks with no matching target, use `uv run --project <dir> --no-sync ...` and carry over those same env vars by hand.

This machine's GPU (AMD Radeon RX 7700S, `gfx1102`) isn't an officially-supported ROCm compute target for several libraries:

- `causal-conv1d` and `mamba_ssm`'s own Triton SSD-scan kernel are both broken here too — `causal-conv1d`'s compiled kernel segfaults (both the multi-token and single-token code paths) and the Triton scan kernel hangs. Confirmed independent of model size, package version, and a from-source rebuild — not something the `HSA_OVERRIDE_GFX_VERSION` spoof fixes. Re-confirmed 2026-07-02 via a direct `mamba_ssm.modules.mamba2.Mamba2` forward call (not just synthetic `causal_conv1d_fn` calls) — still segfaults (exit 139) with `causal-conv1d==1.6.2.post1`. Every model in `models/` therefore drives Mamba2's mixer manually in plain PyTorch (see `models/mamba2_780m/model.py`'s `_mixer_step` and `models/mamba2_2_7b_memory/model.py`'s README for the investigation) instead of calling `Mamba2.forward()`/`.step()`. `causal-conv1d` is consequently not installed at all on this box (removed from both Makefiles' `sync` targets) — it would just be unused dead weight (and a multi-minute compile) if it were. **This is specific to this box's unsupported ROCm gfx arch, not Mamba2/causal-conv1d in general** — `causal-conv1d` and the native fused/chunked kernel path work fine on a proper CUDA target (e.g. a rented H100), so training run there should use the real fused path instead of the manual loop; don't assume the ROCm workaround needs to travel with the code to every environment.
- `import mamba_ssm` needs a *working* GPU even for CPU-only work: its Triton layer-norm module queries CUDA device properties at import time, unguarded, and `HIP_VISIBLE_DEVICES=""` doesn't help (the query still raises). So when this box's GPU falls off the bus (`amdgpu ... device lost from bus`, GPU recovery failed — has happened mid-session; reboot to recover), the whole `make test` suite fails at *collection* through the model import chain, looking like a code breakage when it's the GPU. Check `journalctl -k` for amdgpu resets before debugging test errors that trace into `mamba_ssm/ops/triton/layer_norm.py`. Tests that don't import the models package (e.g. `sft/tests/preparation/test_prepare_chains.py`) still run.
- Uninstalling `causal-conv1d` exposed a separate, unrelated breakage: `mamba_ssm/__init__.py` unconditionally imports legacy Mamba-1 ops (`selective_scan_fn`/`mamba_inner_fn`, which nothing in this repo calls — every model here only uses Mamba2), and the compiled `selective_scan_cuda` extension those need fails to import on this machine (a ROCm runtime library naming/ABI mismatch unrelated to anything in this repo). `models/__init__.py` stubs `selective_scan_cuda` into `sys.modules` before `mamba_ssm` is ever imported, so Python's import system skips the real (broken, unused) import entirely. Don't remove that stub thinking it's dead code — it's load-bearing for every model's `import mamba_ssm` to succeed at all on this machine.

## Models

Each model is a folder in `models/` at the repo root containing `model.py` (implementation), a thin `__init__.py` that re-exports the interface below, a `train_hooks.py` (training-specific interface — see `models/AGENTS.md`), and a `README.md` documenting the model (see `models/AGENTS.md` for the README checklist). A model must export:

| Name | Type | Description |
|------|------|-------------|
| `MODEL_ID` | `str` | HuggingFace model identifier |
| `TOKENIZER_ID` | `str` | HuggingFace tokenizer identifier |
| `TARGET_LORA_MODULES` | `list[str]` | Module name suffixes to attach LoRA adapters to |
| `USER_OPEN` | `str` | Bare user-turn role marker, registered as a tokenizer special token. Callers append a literal `" "` separator before content. |
| `ASST_OPEN` | `str` | Bare assistant-turn role marker, registered as a tokenizer special token. Callers append a literal `" "` separator before content. |
| `EOC` | `str` (optional) | Conversation-boundary marker (`<|endofconversation|>`), registered as a tokenizer special token. Read with `getattr(model_mod, "EOC", None)` — a model that omits it renders no boundary and its dreams end only on the token budget or the turn backstop. |
| `SPECIAL_TOKENS` | `list[str]` | The list passed to `tokenizer.add_special_tokens` — the role markers plus `EOC` where the model defines one |
| `Model` | `nn.Module` | Inference wrapper class |
| `load_base(device)` | `fn` | Load raw HF model (used by sft) |
| `load_inference(device)` | `fn` | Load and wrap for inference (used by backend) |

Both `backend` and `sft` read `MODEL_NAME` from their `.env` and import `models.{MODEL_NAME}` at startup. The backend must be started with `PYTHONPATH` pointing at the repo root (the Makefile handles this). `sft` scripts also add the repo root to `sys.path` automatically.

## Protected paths — hand-written code

`PROTECTED_PATHS` at the repo root lists the files where a wrong line gives a
wrong scientific result without a crash: the memory mechanism, the losses and
erase operators, the benchmark scorer and split, the probes, and the
data-splice invariants. The person owns these files. Agents read them, review
them, write tests for them, and explain them, but do not edit them. A
PreToolUse hook (`.claude/hooks/protect_paths.py`) enforces this for Edit and
Write in local sessions; do not route around it with shell edits. Propose the
change as a diff in the conversation instead.

The rented-box experimenter is the one exception, with its own rules in
`.agents/skills/altrux-experimenter/SKILL.md`: it may patch a protected file
only to unblock a crash, never scoring or split logic, and the patch comes
home for review rather than being committed.

## Always update READMEs and .env.example

Whenever you change user-facing behaviour — a new endpoint, a new CLI flag, changed defaults, a removed feature — update the relevant README(s) in the same change. The READMEs to keep in sync are:

- `README.md` — architecture and high-level design
- `backend/README.md` — API reference, setup, running
- `frontend/README.md` — frontend setup, dev server, environment
- `sft/README.md` — training guide, CLI flags

For the backend README, keep the API table up to date with any new or changed endpoints.

Whenever you add, rename, or remove a backend environment variable, also update `backend/.env.example`.
Whenever you add, rename, or remove a frontend environment variable, also update `frontend/.env.example`.
Whenever you add, rename, or remove an sft environment variable, also update `sft/.env.example`.

## Research from existing notes first

Before external research, read `notes/README.md` and the relevant notes it
links. Use the existing research and decision record as the starting point so
work is not repeated and superseded direction is not restored by accident.
Search external primary sources only for information that the notes do not
cover, claims that are disputed or time-sensitive, or source verification that
the task requires. Update the relevant note and index when new research changes
or extends the repository's standing knowledge.

## Keep AGENTS.md files current

There's an `AGENTS.md` at the root and in some subdirectories (e.g. `models/AGENTS.md`, `sft/AGENTS.md`). Update the relevant one in the same change whenever you introduce or discover something a future session would otherwise have to rediscover the hard way — a non-obvious gotcha, a workaround for broken tooling, a convention that isn't visible just from reading the code, or a rule you had to be told twice. Don't record anything derivable by reading the code itself (that belongs in comments or a README, not here). Each `CLAUDE.md` imports its sibling `AGENTS.md`; don't duplicate instructions there.

When editing an `AGENTS.md`, also check whether the entry you're touching (or a neighboring one) has gone stale — e.g. describes a workaround for a bug that's since been fixed elsewhere — and trim or update it rather than only appending. Keeps the file a live reference instead of an append-only log.

## Live progress logs — and live *results*

Any operation that takes more than a few seconds must print live progress so it's clear something is happening. This includes data preparation, evaluation, model loading, and any other blocking step. A `\r`-based counter or periodic print is fine — silence is not. Never leave a long operation running with no output.

Progress counters alone are not enough: **results must stream as they are produced, never be held for an end-of-run report.** A long run has to be informative if read — or killed — at any point: decoded samples print the moment they're collected, running composition/verdict counters ride the progress line, and anything wall-clock-sensitive reports its rate and ETA. If the only way to learn what a script found is to let it finish, that's a bug. (A filter run once sat at "0 kept" for 100 blocks over an hour while the explanation — the per-test fail composition — existed internally but printed only at exit; the fix was ~5 lines.)

**Every log line carries a timestamp** (`[HH:MM:SS]` prefix, `sft/training/loop.py`'s existing idiom — including the periodic/`\r` status line). Rates, stalls, and durations must be reconstructable from the log alone; inferring a run's speed from file mtimes because the lines are undated is the failure mode this prevents.

## Sanity-check the artifact, not just the counts

Every data generator must print **structural invariants and a decoded sample**, not only quantities, and no dataset goes to a training run until someone has read that sample.

This is not a style preference. `sft/preparation/chains.py` shipped episode splices that left `[USER]` turns with no answer and `[ASSISTANT]` turns answering a question 32k tokens back — 9% of all role transitions malformed — through several runs. Every count in every regen log was correct and reproducible the whole time (`2753 chains, 230.9M tokens, 15857 sleeps, 5498 split-tails`), because counts confirm the generator did what it was told, never that what it was told was right. Only decoding the tokens at a join exposed it.

So a generator's log needs:

- **Invariant counts that can fail** — malformed role adjacencies, unaddressed turns, silent boundaries. Numbers whose correct value is zero, so a regression is visible.
- **Decoded text** — a few hundred characters either side of one instance of every structural event the generator creates (each join type, each splice, each boundary). Three samples read in ten seconds beats any aggregate.

The same applies to probes: a probe that reports a number without ever showing the tokens it scored can be measuring something other than what its name says.
