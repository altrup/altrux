# SFT source organization

2026-08-12. Applies to `sft/`.

## Problem

`sft/` has grown as a sequence of experiments. Its active Python source is now
more than 20,000 lines in one flat directory. The size is concentrated in a
few files:

- `dream_sleep.py` is about 3,000 lines and owns cache schemas, generation,
  distillation, erasure integration, probes, reporting, argument parsing, and
  orchestration.
- `train.py` is about 1,450 lines. `run_training` is more than 600 lines and
  contains another 400-line segment runner.
- `adaptive_multisleep.py` is about 900 lines. Its backend owns model setup,
  wake sampling, sleep treatments, probes, caching, aggregation metadata, and
  CLI execution.
- The remaining preparation, diagnostic, and reporting programs are mixed at
  the same directory level even when they belong to distinct workflows.

The problem is not widespread copy-and-paste. It is unclear ownership and
large orchestration units. Dream, erasure, adaptive-wake, and consolidation
code share several real primitives; training is mostly independent; data
preparation has a few small dependency clusters.

## Goals

1. Organize all maintained `sft` Python code by domain.
2. Give each implementation module one clear responsibility.
3. Split large orchestration functions at existing behavioral seams.
4. Preserve every current top-level command, Make target, documented option,
   and repository-used import.
5. Keep every B-family and erasure experiment runnable and tested.
6. Permit migration in small, independently tested commits.

## Non-goals

- No experiment, arm, option, artifact, or test is retired.
- No scientific behavior, default, output schema, random stream, or execution
  order changes as part of this work.
- `sft` does not become an installable library.
- The refactor does not introduce a plugin system, dependency-injection
  framework, registry, or new third-party dependency.
- Files are not split only to satisfy a line limit.

## Target layout

The project directory is already the useful namespace. Do not add a redundant
`altrux_sft/` or `src/` level.

```text
sft/
├── <existing top-level commands>.py
├── progress.py
├── adapters/
│   └── lora.py
├── training/
│   ├── datasets.py
│   ├── checkpoints.py
│   ├── loop.py
│   └── cli.py
├── preparation/
│   ├── conversations.py
│   ├── babilong.py
│   ├── chains.py
│   ├── cram.py
│   ├── needles.py
│   ├── interference.py
│   ├── filtering.py
│   ├── merge.py
│   ├── eligibility.py
│   └── inspection.py
├── experiments/
│   ├── facts.py
│   ├── inference.py
│   ├── locality.py
│   ├── consolidation/
│   │   ├── null.py
│   │   └── capacity.py
│   ├── erasure/
│   │   ├── operators.py
│   │   ├── gating.py
│   │   ├── wake_items.py
│   │   ├── pilot.py
│   │   └── probe.py
│   ├── dreams/
│   │   ├── types.py
│   │   ├── cache.py
│   │   ├── generation.py
│   │   ├── distillation.py
│   │   ├── probes.py
│   │   ├── runner.py
│   │   └── cli.py
│   └── adaptive/
│       ├── manifest.py
│       ├── wake.py
│       ├── coordinator.py
│       ├── backend.py
│       ├── analysis.py
│       ├── runner.py
│       └── cli.py
├── diagnostics/
│   ├── acceptance.py
│   ├── correction.py
│   ├── dream_fidelity.py
│   ├── knobs.py
│   ├── reads.py
│   ├── recall.py
│   ├── topic_choice.py
│   └── training_smoke.py
├── providers/
│   └── user_generator.py
├── reporting/
│   └── grid.py
├── tests/
│   ├── training/
│   ├── preparation/
│   ├── experiments/
│   ├── diagnostics/
│   ├── providers/
│   ├── reporting/
│   └── test_compatibility.py
└── data/                       # Existing generated artifacts, not Python code
```

Every new code directory is a normal Python package. `sft/` itself remains the
project root and does not need an `__init__.py`.

The listed modules are the intended final set, not scaffolding. A module is
created only when its current code moves. Empty packages and forwarding
modules inside the implementation tree are not allowed.

## Compatibility entry points

Every current top-level Python file remains at its current path. After its
implementation moves, the file becomes a small compatibility entry point that:

1. explicitly imports and re-exports its supported names;
2. imports the domain `main` function; and
3. calls that function only under `if __name__ == "__main__"`.

Do not use wildcard imports. Each wrapper gets an explicit `__all__` so the
compatibility surface is visible. The initial surface includes every
non-private name imported by another repository file or test, plus every name
documented for direct use. Existing cross-module use of a private name, such
as `filter_items` importing `prepare_cram._find`, must move to a named domain
function; the old private name remains re-exported while the compatibility
wrapper exists.

These forms must continue to work:

```bash
python train.py --help
python dream_sleep.py --help
python adaptive_multisleep.py --help
```

```python
from train import load_checkpoint
from dream_sleep import DreamCache
from adaptive_wake import WakePlan
```

The wrappers are permanent compatibility surfaces. Removing one is a separate
user-visible change and is outside this refactor.

## Current-file mapping

| Current file | Implementation destination |
|---|---|
| `acceptance_check.py` | `diagnostics/acceptance.py` |
| `adaptive_multisleep.py` | `experiments/adaptive/{backend,analysis,runner,cli}.py` |
| `adaptive_wake.py` | `experiments/adaptive/{manifest,wake,coordinator,analysis}.py` |
| `adaptive_wake_smoke.py` | `experiments/adaptive/runner.py` with a thin smoke entry point |
| `b4.py` | `experiments/erasure/{operators,gating}.py` |
| `capacity_ladder.py` | `experiments/consolidation/capacity.py` |
| `consolidation_null.py` | `progress.py`, `experiments/{facts,inference}.py`, and `experiments/consolidation/null.py` |
| `dream_fidelity.py` | `diagnostics/dream_fidelity.py` |
| `dream_sleep.py` | `experiments/dreams/` modules |
| `eligibility_check.py` | `preparation/eligibility.py` |
| `erase_probe.py` | `experiments/erasure/{operators,wake_items,probe}.py` |
| `filter_items.py` | `preparation/filtering.py` |
| `gate_pilot.py` | `experiments/erasure/pilot.py` |
| `lora.py` | `adapters/lora.py` |
| `measure_knobs.py` | `diagnostics/knobs.py` |
| `merge_data.py` | `preparation/merge.py` |
| `prepare_babilong.py` | `preparation/babilong.py` |
| `prepare_chains.py` | `preparation/chains.py` |
| `prepare_cram.py` | `preparation/cram.py` |
| `prepare_data.py` | `preparation/conversations.py` |
| `prepare_interference.py` | `preparation/interference.py` |
| `prepare_needles.py` | `preparation/needles.py` |
| `probe_correction.py` | `diagnostics/correction.py` |
| `probe_recall.py` | `diagnostics/recall.py` |
| `probes_common.py` | `experiments/locality.py` |
| `read_diagnostic.py` | `diagnostics/reads.py` |
| `sanity_sample.py` | `preparation/inspection.py` |
| `smoke_test.py` | `diagnostics/training_smoke.py` |
| `summarize_grid.py` | `reporting/grid.py` |
| `topic_choice.py` | `diagnostics/topic_choice.py` |
| `train.py` | `training/{datasets,checkpoints,loop,cli}.py` |
| `user_generator_cli.py` | `providers/user_generator.py` |

## Module responsibilities

### Training

- `datasets.py` owns `DataSpec`, data-spec parsing, share validation, dataset
  loading, grouping, fingerprints, and deterministic training order.
- `checkpoints.py` owns checkpoint discovery, rotation, save, and load. It does
  not import the CLI or training loop.
- `loop.py` owns slots, segment execution, schedules, progress state, and
  `run_training`.
- `cli.py` owns argument parsing, model hook loading, configuration assembly,
  and the one call into `run_training`.

`run_segment` becomes a top-level implementation function. `run_training`
keeps run-level orchestration but does not parse arguments, load datasets,
implement checkpoint serialization, or contain a second training loop.

`progress.py` owns the timestamp and duration-formatting functions used across
training, preparation, and experiments. It has no domain imports and is the
only deliberately cross-domain leaf module.

### Preparation

Each preparation module owns one artifact transformation. Existing real
sharing remains explicit:

- `needles.py` can depend on public block-building primitives in `cram.py`.
- `chains.py` can depend on fact schemas in `interference.py`.
- `filtering.py` can depend on a public passage-search function in `cram.py`.
- `inspection.py` reads artifacts but never participates in generation.

Preparation modules do not import experiment CLIs. Shared code is promoted
only when at least two current transformations use the same behavior.

### Experiment primitives

- `facts.py` owns `Fact`, synthetic fact generation, transcript construction,
  cue construction, and fact-specific parsing and matching.
- `inference.py` owns model-agnostic chunk execution, generation, token scoring,
  and KL helpers.
- `locality.py` owns the calibrated knowledge battery, held-out perplexity,
  margin scoring, and battery summaries.

These modules replace the accidental use of `consolidation_null.py` as a
shared library. They contain named experiment concepts, not miscellaneous
helpers.

### Consolidation

`consolidation/null.py` owns only the transcript-distillation null experiment.
`consolidation/capacity.py` owns the capacity ladder. Both consume the shared
fact and inference modules.

### Erasure

- `operators.py` owns rank-one, deflated, subspace, and scaled erasure tensor
  operations plus state-direction calculations.
- `gating.py` owns divergence gates, basis aggregation, rank selection, and
  gate-family weighting.
- `wake_items.py` owns bystander construction, collision checks, mixed wake
  turns, and distractor reports.
- `pilot.py` owns pilot capture schemas and offline gate comparison.
- `probe.py` owns erase-efficacy experiment setup and its CLI behavior.

This breaks the current `b4` → `erase_probe` dependency. Tensor operators do
not import probe orchestration. B-family implementations remain first-class
consumers of these modules.

### Dreams

- `types.py` owns `Dream`, `DreamCache`, `CachedDream`, and `DreamSetCache`.
- `cache.py` owns cache identity, validation, load/save, merge, rebase, and
  sidecars.
- `generation.py` owns single and batched frozen-teacher generation.
- `distillation.py` owns replay, drain, counterfactual, fused, live, B4, and
  sequential-SFT training operations.
- `probes.py` owns dream-copy, rehearsal, binding, leakage, and dream-set
  diagnostics.
- `runner.py` owns cache construction, per-sleep treatment dispatch, and the
  wake/sleep experiment loop.
- `cli.py` owns parsing, validation, model setup, and delegation to `runner`.

The B-family stays in `distillation.py` because it is a dream treatment. Its
linear algebra stays in `experiments/erasure/` so probes and treatments use
one implementation.

The `dream_sleep.main` body is replaced by CLI setup plus one runner call.
Probe closures become methods or top-level functions with explicit inputs.
`run_sleep` becomes treatment dispatch; each treatment implementation remains
independently callable and testable.

### Adaptive multi-sleep

- `manifest.py` owns manifest dataclasses, validation, canonical identity, and
  counterbalancing.
- `wake.py` owns command-driven user generation, partial wake persistence,
  exact-token replay, and immutable wake artifacts.
- `coordinator.py` owns arm ordering, Wake 1 forking, state flow, and the
  backend protocol.
- `backend.py` adapts model, dream, probe, and artifact operations to the
  coordinator. It does not aggregate results or parse CLI arguments.
- `analysis.py` owns floor correction, retention matrices, rehearsal analysis,
  and cross-seed uncertainty.
- `runner.py` owns seed resumption, stream files, final result storage, and
  aggregate execution.
- `cli.py` validates command selection, builds backends, and delegates.

`DreamSleepBackend` remains a concrete backend, not a hierarchy. Its model
construction, wake, sleep, and probe paths use the dream and locality modules
instead of duplicating them.

### Diagnostics, providers, and reporting

Diagnostic programs remain separate because each answers a different
question. They may consume training, preparation, or experiment APIs, but
domain implementation modules do not import diagnostics.

`providers/user_generator.py` remains independent of model training and owns
only the authenticated Claude/Codex command adaptation.

`reporting/grid.py` consumes result artifacts. It must not import model-loading
or training code.

## Dependency direction

The permitted dependency flow is:

```text
top-level wrappers
        ↓
domain cli / runner / diagnostic
        ↓
domain implementation
        ↓
named shared primitives / progress
```

More specifically:

- `training` depends on model `train_hooks` and `adapters`, not experiments.
- any domain can depend on `progress`, which depends only on the standard
  library.
- `preparation` depends only on preparation siblings and tokenizer/model
  interfaces needed to build artifacts.
- consolidation and erasure depend on experiment primitives.
- dreams depend on experiment primitives and erasure operations.
- adaptive depends on dreams and experiment locality primitives.
- diagnostics can depend on any stable domain API.
- reporting and providers remain leaf-independent from model execution.

No implementation module imports a top-level compatibility wrapper. This rule
is what prevents circular imports during and after migration.

Do not add `common.py`, `core.py`, `utils.py`, or a catch-all context object.
If code has no clear domain owner, leave it with its current caller until a
second real consumer establishes a boundary.

## Function boundaries

Line counts are review signals, not targets. A file above roughly 600 lines or
a function above roughly 100 lines requires a check for mixed responsibility,
but a sequential algorithm is not split into meaningless one-use helpers only
to pass the threshold.

A function must not combine more than one of these phases:

1. parse or validate user configuration;
2. construct models, tokenizers, optimizers, or datasets;
3. execute a training or generation loop;
4. score results;
5. persist artifacts; or
6. render reports.

Helpers are extracted when they name a real phase, invariant, or reusable
operation. Helpers that only rename two lines or forward all arguments are not
an improvement.

## Artifact compatibility

The refactor must preserve:

- checkpoint fields and resume behavior;
- prepared-data schemas;
- dream-cache fields, identity hashes, and the ability to load existing cache
  files whose pickle globals point at `dream_sleep`;
- wake, manifest, stream, seed-result, and aggregate schemas;
- deterministic seed derivation and random draw order;
- decoded sidecar structure and invariant checks; and
- existing default paths.

Compatibility wrappers must expose old pickle globals before any cache classes
move. A fixture made by the current code must load after the extraction. New
caches must round-trip through both the domain loader and the top-level
compatibility loader. A serialized-format migration is separate work.

## Test organization

Tests move with their implementation domain, but movement alone does not
justify rewriting assertions. The final test tree mirrors the source domains.

`test_compatibility.py` covers the stable boundary:

- every former top-level module imports;
- documented and repository-used symbols remain available;
- each top-level CLI delegates to the same parser and `main` path;
- existing dream caches load through the compatibility module; and
- Make targets still resolve the same commands.

Behavior tests stay focused on calls, state, artifacts, seeds, and invariants.
They do not assert complete help text or log wording.

Before moving a large orchestration path, add a characterization test only for
behavior not already protected. Do not create duplicate old-path and new-path
test suites. The compatibility test proves the old path; domain tests prove
the behavior once.

## Migration sequence

Every phase leaves the repository green and keeps top-level commands usable.
Each bullet is a commit-sized sub-feature unless its tests show it must be
smaller.

1. **Compatibility inventory.** Add the import/CLI/cache compatibility test
   before moving definitions.
2. **Training.** Extract datasets and checkpoints, then the segment loop, then
   the training CLI.
3. **Preparation.** Move the conversation/cram/needle/interference/chain
   clusters, then the independent preparation and inspection commands.
4. **Experiment primitives.** Extract facts, inference, and locality from
   `consolidation_null.py` and `probes_common.py`; leave their wrappers green.
5. **Erasure.** Extract tensor operators, gating, wake items, pilot, and probe.
   Confirm all B-family tests before proceeding.
6. **Dreams.** Move types and cache compatibility first, then generation and
   probes, then distillation and runner orchestration, and finally the CLI.
7. **Adaptive.** Move manifest/wake/coordinator, analysis, backend, runner, and
   CLI in that order.
8. **Diagnostics, providers, and reporting.** Move the remaining small command
   implementations behind their existing wrappers.
9. **Test layout and cleanup.** Move tests into matching domain directories,
   remove temporary forwarding imports inside implementation packages, update
   `sft/README.md`, and run the complete available suite.

Do not combine file movement with behavior changes. If extraction exposes a
bug, first finish or revert the pure move; fix the bug in its own TDD slice.

## Completion criteria

The reorganization is complete when:

- every maintained top-level script is a compatibility wrapper or a deliberate
  small command whose full implementation is already one responsibility;
- `dream_sleep.py`, `train.py`, and `adaptive_multisleep.py` contain no runtime
  orchestration beyond compatibility and CLI delegation;
- no implementation imports a top-level wrapper;
- the dependency directions above have no cycle;
- all current Make targets, CLI paths, and supported imports work;
- B-family tests and commands remain present;
- existing checkpoints, prepared data, dream caches, and adaptive artifacts
  remain readable;
- deterministic tests prove unchanged seeds and artifact identities;
- the focused CPU-safe suite passes locally; and
- the full suite and real command smokes pass on the required GPU environment.

## Rejected alternatives

### Keep all modules flat

Files such as `dream_cache.py` and `train_checkpoints.py` would reduce the
largest files but make the already crowded project root larger. It does not
express domain ownership.

### Add `altrux_sft/` or `src/`

This adds a redundant namespace to a project that is not installed as a
library. `sft/` already supplies the project boundary.

### Use `lib/`, `core/`, or `common/`

These names organize by reuse rather than responsibility. The current sharing
is domain-specific, so a generic bucket would hide the same coupling under a
new directory.

### Retire B-family code during the move

B-family experiments remain required. Deletion would change experimental
capability and make the refactor impossible to assess as behavior-preserving.
