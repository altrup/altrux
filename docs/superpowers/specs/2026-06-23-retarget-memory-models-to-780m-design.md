# Retarget memory & continuous-learning models to mamba2-780m

## Summary

`mamba2_2_7b_memory` and `mamba2_2_7b_continuous_learning` are rebuilt on the
`state-spaces/mamba2-780m` backbone instead of `state-spaces/mamba2-2.7b`.
The 2.7B versions are deleted, not kept alongside. Folders are renamed to
match (`mamba2_780m_memory`, `mamba2_780m_continuous_learning`), continuing
this repo's convention of naming a model folder after its backbone size.

## Scope

### `mamba2_780m_continuous_learning`

This model's `model.py` is already removed pending a critic-gated-gradient-
accumulation redesign (see its current README). There is no `Model` wrapper
to port. Scope here is limited to retargeting the metadata that exists today
(`MODEL_ID`, any dims referenced in `train_hooks.py`/`README.md`) to the 780m
backbone, so the eventual redesign builds on top of 780m instead of 2.7b.
Folder renamed from `mamba2_2_7b_continuous_learning`.

### `mamba2_780m_memory`

Folder renamed from `mamba2_2_7b_memory`. Backbone and dependent constants
change; the architecture itself (Titans front-end + gated-delta merge into
Mamba2's own SSM state) is unchanged.

**Backbone dims** (confirmed via the cached `state-spaces/mamba2-780m`
`config.json` plus Mamba2's layer defaults):

| Constant | Old (2.7b) | New (780m) |
|---|---|---|
| `MODEL_ID` | `state-spaces/mamba2-2.7b` | `state-spaces/mamba2-780m` |
| `D_MODEL` | 2560 | 1536 |
| `N_LAYER` | 64 | 48 |
| `NHEADS` | 80 | 48 |
| `HEADDIM` | 64 | 64 |
| `D_STATE` | 128 | 128 |

**Injection depth/coverage**, rescaled proportionally to the new `N_LAYER`
so the relative depth and coverage fraction match the original design intent
(read at ~2/3 depth, inject across the latter ~third of the stack):

| Constant | Old (2.7b, /64) | New (780m, /48) |
|---|---|---|
| `READ_LAYER` | 42 (~66%) | 32 (~67%) |
| `INJECTED_LAYERS` | `range(20, 64, 2)` — 22 layers (~34%) | `range(16, 48, 2)` — 16 layers (~33%) |

`MEM_DIM`, `MEM_HIDDEN`, `BOTTLENECK_R` stay derived from `D_MODEL` as
before — no separate change needed, they fall out of the rename.

**`QUANTIZE_LORA_BASE` removed entirely** (omitted, like the plain
`mamba2_780m` model already does) — plain full-precision LoRA, no
`bitsandbytes` 4-bit quantization. `load_base` no longer calls
`quantize_lora_targets`. Rationale: the 2.7B backbone needed QLoRA to fit
this project's 8GB dev GPU; the 780M backbone (~1.5GB in bf16) plus the
memory subsystem fits comfortably without it, and dropping quantization
removes the `bitsandbytes`/ROCm segfault risk (the
`HSA_OVERRIDE_GFX_VERSION` workaround) for this model entirely.

**`DEFAULT_CHUNK_LEN`** stays at its current conservative value (`2`) rather
than being extrapolated to a new guessed number. Both the smaller backbone
and the removed quantization overhead shrink the real per-token VRAM cost,
but by how much isn't something to guess — the comment explaining the
existing value is updated to say it needs re-measuring via `make preflight`
once this change lands, rather than asserting a new number outright.

**`README.md`** is rewritten to match: backbone shape section, the
architecture diagram's literal layer numbers (`READ_LAYER`/`INJECTED_LAYERS`
values and the `0..N_LAYER-1` range they sit in), parameter budget estimate,
and the "LoRA target modules" section (replacing the QLoRA explanation with
the plain-LoRA rationale `mamba2_780m`'s README already uses).

### Cross-reference cleanup

The following files reference `mamba2_2_7b_memory`/`mamba2_2_7b_continuous_learning`
in comments or docs only — confirmed via repo-wide search that nothing
imports either model by its old package name except its own files — and get
those mentions updated to the new names:

- `CLAUDE.md`
- `models/CLAUDE.md`
- `sft/Makefile`
- `sft/README.md`
- `sft/train.py`
- `sft/smoke_test.py`
- `backend/app/model/lora.py`
- `backend/app/model/registry.py`
- `models/mamba2_780m/model.py`
- `models/mamba2_780m/train_hooks.py`
- `models/mamba2_780m/README.md`

### Tests

`models/tests/test_mamba2_2_7b_memory_train_hooks.py` is renamed to
`test_mamba2_780m_memory_train_hooks.py`. Its test logic is unaffected — it
already monkeypatches every dimension (`D_MODEL`, `N_LAYER`, `NHEADS`,
`HEADDIM`, `D_STATE`, `READ_LAYER`, `INJECTED_LAYERS`, `MEM_DIM`,
`MEM_HIDDEN`) to tiny synthetic values for speed, independent of the real
780m or 2.7b dims. Only the module path and docstring text referencing the
old name change.

## Out of scope

- No change to the gated-delta merge math, the Titans front-end's
  write/read mechanics, or `Model._mixer_step`'s per-token loop structure.
- No change to `mamba2_780m_continuous_learning`'s eventual critic-gated
  redesign — that remains a separate, future piece of work.
- No re-measurement of `DEFAULT_CHUNK_LEN` as part of this change; that's
  flagged as a follow-up to do empirically via `make preflight` after the
  rename/retarget lands.
- No data/checkpoint migration — there are no existing 780m-memory
  checkpoints to migrate, and 2.7b checkpoints are abandoned along with the
  2.7b code.
