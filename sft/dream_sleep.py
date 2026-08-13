"""Dream-distillation sleep: the continual-learning A/B between consolidating a
session by *dreaming it back out of the state* and consolidating it by
conventional fine-tuning.

The protocol and the arm sequences are registered in
notes/DISCUSSION-20260806-dream-distillation-ab-postmortem.md sec 3, which
supersedes the 08-05 file's sequences -- implemented here verbatim. Wake is
exactly stock: no gate, no erase, no new parameters. The erase fires only
inside a sleep, at gamma = 1.0 (sec 3a's probe result -- sub-1 gamma is both a
no-op under the mixer's gated RMSNorm and an invitation to compensate), along
the student's own current read query; `--erase-op` picks the operator (raw
query, or the query with the state's top singular direction deflated out --
DISCUSSION-20260807 sec 3.4's picker decides which). The direction is
differentiable -- the cut follows the query -- while the deflation basis v is
always stop-gradiented. The erase never touches the wake path.

  wake          -- the consolidation-null generator's transcript: --n-facts
                   entity->code facts separated by --filler-tokens of
                   digit-free filler, primed into the state.
  cache         -- --build-dream-cache generates this seed's ONE teacher dream
                   (the student as of the sleep's start -- the --init-adapter
                   warm start, else the base -- from a copy of the wake state,
                   intact) and
                   persists it with the wake state, the teacher logits and
                   queries, distractor codes and sha-256 hashes of both token
                   sequences. Every arm loads it; no arm generates.
                   --dreams N builds the multi-dream set instead (sec 2.10.4):
                   N dreams generated upfront, each from a fresh copy of the
                   intact wake state, each carrying its own gated queries,
                   per-layer spectra and eraser. The SET hash rides every
                   result record.
  sleep --arm   -- replay        (A):  student teacher-forced over the cached
                                       dream from a FRESH state, KL to the
                                       cached logits; the chunk is the whole
                                       dream (--chunk-len is the bridge cell);
                                       carries nothing.
                   drain         (B1): teacher-forced over the same dream from
                                       a copy of the wake state; per token, per
                                       layer, the carried past is ablated along
                                       the student's own read query before the
                                       write; the ablated state carries.
                   counterfactual(B2): B1 with the ablation on a copy that is
                                       trained on and discarded; the intact
                                       state carries. B1 == B2 at token 1.
                   counterfactual-commit
                                 (B2'): B2, then at sleep end one real erase per
                                       fact that passes a fresh-state margin
                                       check -- the erase as verified memory
                                       policy, not as training signal.
                   --ce-on-dream      : A's sequence, cross-entropy on the
                                       dream tokens instead of KL.
                   --deep             : B2 with the pass's losses accumulated
                                       and one optimizer step per pass (full
                                       BPTT through the spine).
                   b2-fused-detached  : B2's whole dream in one step (sec 3.5)
                                       -- the intact spine materialized once
                                       per pass and detached, then every
                                       position's counterfactual batched.
                   b2-fused-deep      : the same, spine not detached (BPTT
                                       through the scan; v still no-grad).
                   b3-fused           : the same, on the dream generator's own
                                       state trajectory, constant across
                                       passes. Equals b2-fused-detached at
                                       pass 1, machine-checked every sleep.
                   b4-raw / b4-deflated / b4-qcm
                                 (B4): erase ONCE per dream (a projection along
                                       the frozen teacher's aggregate of the
                                       state-dependency-gated read queries),
                                       then ordinary sequence training against
                                       the cached logits -- full BPTT, no
                                       spine. Needs a multi-dream cache
                                       (--dreams N); the variant names the
                                       post-processing of the shared SVD.
                   drain-live         : one online adapters-on pass. Retired
                                       (sec 6), kept for reference.
                   --sft-ref          : the CE-on-raw-text convention, on the
                                       stored wake transcript.
                   --no-sleep         : the floor -- no training at all.
  probes        -- the distractor-code margin (primary), greedy exact match
                   (reported, never gating), ~4 paraphrases per fact
                   (generality), the self-calibrated knowledge battery and
                   held-out ppl (locality/forgetting), then the carried-state
                   column as a DIAGNOSTIC -- never scored as installation. The
                   full battery streams every --probe-every steps, so a cell
                   yields a learned-vs-forgotten curve rather than a point.

The dream arms train the same objective (KL to the frozen teacher's cached
logits) on the same cached dream; the student's state deprivation is the only
manipulated variable. `--waves K` runs K wake/sleep rounds on one carried
state (sec 3.7's registered shape is 4 x 4 fresh facts): wave 1 distils the
seed's shared cache, every later sleep generates its own dream from the state
it carried in, cued on that wave's facts only, and every fact so far is
probed after every sleep -- which is the only form that can price consumption
(the in-context control) and backward transfer (the R-matrix, BWT and
cumulative installation the run reports).

Box tool: this trains a LoRA and holds a full-vocab logit cache for the dream
-- it runs on rented CUDA hardware, never the local ROCm box. Only the pure
pieces are CPU-testable (tests/test_dream_sleep.py, tests/test_dream_cache.py),
which is why the torch and models.* imports live inside the functions that need
them.

Usage (from sft/, env vars as in the Makefile):
    make dream-sleep ARGS="--build-dream-cache --seed 1234 --cue-every 32"
    make dream-sleep ARGS="--arm counterfactual --distill-steps 800 --seed 1234"
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.facts import (
    CODE_DIGITS,
    Fact,
    build_facts,
    build_turns,
    cue_rungs,
    exact_match,
    extract_answer,
    normalize,
    render_turns,
)
from experiments.inference import generate, kl_loss, replay_step, run_chunks, target_logprob
from experiments.dreams.types import (
    CachedDream,
    Dream,
    DreamCache,
    DreamSetCache,
    dream_set_sha,
    token_sha,
)
from experiments.dreams.cache import (
    aggregate_binding,
    assert_aggregate_binding,
    binding_coverage,
    copy_fraction,
    default_cache_path,
    dream_bases,
    dream_sidecar_text,
    fact_read_positions,
    gate_agreement,
    load_dream_cache,
    merge_dream_sets,
    pilot_path,
    rebase_dream_set,
    save_dream_cache,
    sidecar_path,
    write_dream_set_sidecar,
    write_dream_sidecar,
)
from experiments.dreams.generation import (
    _repeat_state,
    _teacher_dream_batch,
    copy_state,
    dream_generation_seed,
    dream_seed_text,
    frozen_teacher,
    generate_replay_dreams,
    rehearsal_fraction,
    sample_next,
    state_to,
    teacher_dream as _teacher_dream,
)
from experiments.dreams.probes import (
    basis_overlap,
    battery_read_queries,
    blank_state_logits,
    dream_is_degenerate,
    longest_verbatim_run,
    probe_leakage,
    report_dream,
    report_dream_set,
)
from consolidation_null import (
    GEN_TOKENS,
    report_transcript,
)
from progress import fmt_duration, ts
from b4 import erase_state_subspace
from experiments.erasure.gating import (
    RANK_RULES,
    VARIANTS,
    gated_positions,
    state_divergence,
)
from gate_pilot import PilotCapture, PilotDream
from experiments.erasure.probe import (
    group_by_layer,
)
from experiments.erasure.operators import (
    deflate,
    rank1_erase,
    state_top_dirs,
)
from experiments.erasure.wake_items import (
    build_mixed_turns,
    build_wake_items,
    report_distractors,
)
from lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from experiments.locality import (
    BATTERY_CANDIDATES,
    HELDOUT_TEXT,
    battery_summary,
    code_margin,
    load_or_build_battery,
    logprob_sum,
    perplexity,
    score_battery_batched,
    validate_battery_candidates,
)

# Registered erase parameters (sec 3a). Gamma and k are deliberately not
# flags: do not re-tune gamma downward without new evidence of a kind the
# erase probe could not see. The operator is a flag only because sec 3.4
# registers the raw-vs-deflated comparison as a cell to be run.
GAMMA = 1.0
DEFLATE_K = 1
ERASE_OPS = ("raw", "deflated")
ERASE_OP = "deflated"
# A query with almost nothing left after deflation is all shared cone and no
# discriminative sliver: skip it rather than erase noise (erase_probe.py).
CONE_SKIP = 0.05

DREAM_TOKENS = 512
PRINT_EVERY = 16
# The wake transcript's ordinary-dialogue slice (sec 2.10.11), drawn from a
# split the warm start never trains on. Most candidates are discarded by the
# collision guard, so the pool is far larger than any slice.
WAKE_DIALOGUE_SOURCE = "HuggingFaceH4/ultrachat_200k"
WAKE_DIALOGUE_SPLIT = "test_sft"
WAKE_DIALOGUE_POOL = 400
# A self-terminating dream ends on <|eoc|>; this bounds the pathological case
# where it never comes and the model just keeps opening turns (sec 2.9.4).
DREAM_MAX_TURNS = 32
STOP_REASONS = ("eoc", "max-tokens", "turn-backstop")
# Regeneration attempts for a dream that fails the per-dream acceptance check
# (sec 2.1's mojibake clause). Content-free, so prod-valid (sec 2.9.1).
DREAM_RETRIES = 2

# Fused-B knobs (sec 3.5). `SPINE_BLOCK` trades the spine's sequential depth
# (T/block whole-block forwards, then `block` batched token steps) against the
# batch width of the second phase; ~sqrt(dream length) is the minimum. Neither
# changes the trajectory, only how it is computed.
SPINE_BLOCK = 32
CF_BATCH = 128

# The cue timer defers its splice to the next sentence end, so a cue never cuts
# a thought in half (the prepare_chains splice lesson); this is how far it will
# wait before splicing anyway.
CUE_DEFER_MAX = 20
CUE_STOPS = (".", "\n")

# The distractor codes' RNG stream is the wake seed xor this, so adding them
# leaves every prior run's wake transcript bit-identical.
DISTRACTOR_SALT = 0x5EED

# B4 (sec 2.7): one erase per dream, from the frozen teacher's aggregate. The
# arm name carries the variant, since the variant IS the operator being
# compared -- there is no --erase-op axis crossing it.
B4_ARMS = {f"b4-{variant}": variant for variant in VARIANTS}
# The sigma arm reuses the raw basis but removes each direction in proportion
# to its own singular value instead of all-or-nothing (sec 5's rejected
# sigma-scaling, tested empirically because both objections concern REPEATED
# application and single-sleep erases once per dream).
SIGMA_ARM = "b4-sigma"
B4_ARMS[SIGMA_ARM] = "raw"

# What each arm hands to the next wake, per the sec 3 sequences.
ARM_CARRY = {
    **dict.fromkeys(B4_ARMS, "intact"),
    "replay": "none",
    "ce-on-dream": "none",
    "drain": "drained",
    "counterfactual": "intact",
    "counterfactual-commit": "committed",
    "drain-live": "drained",
    "sft-ref": "none",
    "no-sleep": "intact",
    "b2-fused-detached": "intact",
    "b2-fused-deep": "intact",
    "b3-fused": "intact",
}
FUSED_ARMS = ("b2-fused-detached", "b2-fused-deep", "b3-fused")
ARMS = ("replay", "drain", "counterfactual", "counterfactual-commit", "drain-live",
        *FUSED_ARMS, *B4_ARMS)
# Arms a multi-dream cache runs (sec 2.10.1): A and the B4 family, nothing else.
DREAM_SET_ARMS = ("replay", *B4_ARMS)
# Gate threshold in nats of KL(with-state || blank-state). A placeholder until
# the sec 4 pilot freezes it on real spectra -- always pass it explicitly.
GATE_THRESHOLD = 1.0
RANK_RULE = "ratio-gap"
# Facts must bind in at least this many dreams of a set (sec 3's aggregate
# coverage gate); the pilot may raise it.
BIND_MIN_DREAMS = 2
# Below this a dream carries no content to double-count, and two builds can
# produce it identically without having shared a seed offset.
MIN_DISTINCT_DREAM_TOKENS = 8

# The wake session's own question phrasing, reused verbatim as a rehearsal cue.
USER_CUE = "{user} What is the code for the {entity}?"

# Generality probes: the question is reworded, the assistant stem is not, so a
# miss is a failure to retrieve rather than a failure to match a format.
PARAPHRASE_TEMPLATES = [
    "{u} Remind me, which code was assigned to the {entity}?",
    "{u} I need the {entity}'s code.",
    "{u} Which digits belong to the {entity}?",
    "{u} Could you tell me the code that goes with the {entity}?",
]


from experiments.dreams.distillation import (
    distill_counterfactual,
    distill_dream_set,
    distill_fused,
    distill_live,
    distill_replay,
    distill_sft,
    dream_from_cached,
    erase_state,
    erase_ssm,
    erased_start,
    erased_start_scaled,
    fused_pass,
    make_erase_hook,
    sft_steps,
    spine_states,
)
from experiments.dreams.runner import (
    build_cache,
    build_cues,
    build_dream_set,
    cl_summary,
    commit_erase,
    generate_wave_dream,
    load_live_wake_scenarios,
    r_matrix_row,
    r_matrix_rows,
    render_live_wake_transcript,
    run_dream_set_sleep,
    run_sleep,
    validate_live_wake_args,
    validate_wave_args,
)

# Keep this module as the historical import and pickle namespace.  The CLI
# implementation lives in the domain module and delegates back to this
# namespace at runtime where old callers still monkeypatch helpers.
from experiments.dreams import cli as _cli

build_parser = _cli.build_parser
main = _cli.main
teacher_dream = _cli.teacher_dream
paraphrase_prompts = _cli.paraphrase_prompts
make_emit = _cli.make_emit
file_sha = _cli.file_sha
load_init_adapter = _cli.load_init_adapter
build_distractors = _cli.build_distractors
target_keep_mask = _cli.target_keep_mask
scored_keep = _cli.scored_keep
load_dialogue_records = _cli.load_dialogue_records
run_rebase = _cli.run_rebase
run_merge = _cli.run_merge
dream_from_cache = _cli.dream_from_cache

__all__ = [
    "CachedDream", "Dream", "DreamCache", "DreamSetCache", "Fact",
    "build_parser", "main", "teacher_dream", "paraphrase_prompts",
    "make_emit", "file_sha", "load_init_adapter", "build_distractors",
    "target_keep_mask", "scored_keep", "load_dialogue_records",
    "run_rebase", "run_merge", "dream_from_cache", "load_dream_cache",
    "save_dream_cache", "merge_dream_sets", "rebase_dream_set",
    "copy_fraction", "dream_bases", "longest_verbatim_run",
    "dream_is_degenerate", "battery_read_queries", "probe_leakage",
    "generate_replay_dreams", "copy_state", "dream_generation_seed",
    "dream_seed_text", "frozen_teacher", "state_to", "distill_replay",
    "distill_counterfactual", "distill_dream_set", "distill_fused",
    "distill_live", "distill_sft", "dream_from_cached", "erase_state",
    "erase_ssm", "erased_start", "erased_start_scaled", "fused_pass",
    "make_erase_hook", "sft_steps", "spine_states", "build_cache",
    "build_cues", "build_dream_set", "cl_summary", "commit_erase",
    "generate_wave_dream", "load_live_wake_scenarios", "r_matrix_row",
    "r_matrix_rows", "render_live_wake_transcript", "run_dream_set_sleep",
    "run_sleep", "validate_live_wake_args", "validate_wave_args",
]


if __name__ == "__main__":
    main()
