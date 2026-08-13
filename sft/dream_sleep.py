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


def paraphrase_prompts(fact: Fact, user_open: str, asst_open: str) -> list[str]:
    """One prompt per paraphrase template, each ending in the same answer stem
    the trained phrasing uses."""
    stem = f"{asst_open} The code for the {fact.entity} is"
    return [t.format(u=user_open, entity=fact.entity) + stem for t in PARAPHRASE_TEMPLATES]


def teacher_dream(
    model,
    wake_state,
    seed_ids: torch.Tensor,
    n_tokens: int,
    temperature: float,
    drain: bool,
    decode_token,
    needles: Sequence[str],
    cues: Sequence[Sequence[int]] = (),
    cue_every: int = 0,
    cue_greedy: int = 0,
    frozen: bool = True,
    erase_op: str = ERASE_OP,
    stop_id: int | None = None,
    turn_id: int | None = None,
    max_turns: int = DREAM_MAX_TURNS,
) -> Dream:
    """Compatibility entry point for the generation module's teacher."""
    return _teacher_dream(
        model, wake_state, seed_ids, n_tokens, temperature, drain, decode_token, needles,
        cues, cue_every, cue_greedy, frozen, erase_op, stop_id, turn_id, max_turns,
        erase_state_fn=erase_state,
    )


def make_emit(out_file, **stamped: object):
    """Result-jsonl writer. Every record carries `stamped` -- the run
    parameters the summarizer needs on each line to know which cell it is
    pooling."""
    def emit(record: dict[str, object]) -> None:
        out_file.write(json.dumps({**stamped, **record}) + "\n")
        out_file.flush()

    return emit


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_init_adapter(model, ckpt: str | Path, rank: int, alpha: float) -> str:
    """Load a train.py checkpoint directory into an already-LoRA-attached
    model and return its trainable.pt SHA-256 (the hash stamped on every
    record). A rank/alpha mismatch is fatal: a warm start that silently
    half-applied would be indistinguishable in the results from one that
    worked."""
    from train import load_checkpoint

    ckpt = Path(ckpt)
    config = json.loads((ckpt / "lora_config.json").read_text())
    if (config["rank"], float(config["alpha"])) != (rank, float(alpha)):
        raise ValueError(
            f"{ckpt} was trained at LoRA rank {config['rank']} alpha {config['alpha']}, "
            f"this run is rank {rank} alpha {alpha} -- the adapters do not correspond. "
            f"Retrain the warm start at this run's config, or run at the checkpoint's."
        )
    load_checkpoint(model, ckpt)
    return file_sha(ckpt / "trainable.pt")


def build_distractors(facts: Sequence[Fact], seed: int, taken: Sequence[str] = ()) -> dict[str, str]:
    """One fixed foil code per fact, drawn from its own RNG stream so the wake
    transcript's draws are unchanged. The margin metric (sec 4) scores the
    correct code against these, which is immune to the format prior and to the
    digit-counting attractor that broke greedy exact match.

    `taken` is every code already spoken for by an earlier wave (its facts and
    its foils): multi-sleep draws a wave's foils from the same stream, and a
    foil that is another wave's real code would score that fact as forgotten."""
    rng = random.Random(seed ^ DISTRACTOR_SALT)
    taken = {f.code for f in facts} | set(taken)
    distractors: dict[str, str] = {}
    for fact in facts:
        code = " ".join(str(rng.randrange(10)) for _ in range(CODE_DIGITS))
        while code in taken:
            code = " ".join(str(rng.randrange(10)) for _ in range(CODE_DIGITS))
        taken.add(code)
        distractors[fact.entity] = code
    return distractors


def target_keep_mask(cue_flags: Sequence[bool]) -> list[bool]:
    """Which positions contribute to the KL/CE sum: cue tokens are masked as
    TARGETS only (sec 4). The cue stays in context and the last cue position is
    kept -- it predicts the first answer digit, which is the thing being
    learned."""
    n = len(cue_flags)
    return [t + 1 >= n or not cue_flags[t + 1] for t in range(n)]


def scored_keep(cue_flags: Sequence[bool], prefix_len: int) -> list[bool]:
    """`target_keep_mask` over the steer prefix as well as the cue spans
    (sec 4): prefix tokens condition the dream through state only, in every
    arm, so they are masked as TARGETS exactly the way cue text is -- and
    position prefix_len-1 is kept, because it predicts the first free token."""
    return target_keep_mask([cue or t < prefix_len for t, cue in enumerate(cue_flags)])


def load_dialogue_records(n: int = WAKE_DIALOGUE_POOL) -> list[dict]:
    """Candidate conversations for the wake transcript's dialogue slice."""
    from datasets import load_dataset

    print(f"[{ts()}] loading {n} {WAKE_DIALOGUE_SOURCE} candidates for the wake dialogue slice")
    ds = load_dataset(WAKE_DIALOGUE_SOURCE, split=f"{WAKE_DIALOGUE_SPLIT}[:{n}]")
    return [{"messages": r["messages"]} for r in ds]


def run_rebase(args, cache_path: Path) -> None:
    """The --rebase-gate-family mode: recomputes erasers from cached queries,
    no model and no generation, so it runs and returns before the run path."""
    rebased = rebase_dream_set(load_dream_cache(cache_path), args.rebase_gate_family,
                               args.rank_rule)
    save_dream_cache(rebased, cache_path)
    write_dream_set_sidecar(rebased, sidecar_path(cache_path))
    print(f"[{ts()}] re-based {cache_path}: {len(rebased.dreams)} dreams now carry "
          f"{args.rebase_gate_family} erasers, set_sha {rebased.set_sha[:12]} unchanged")


def run_merge(args, cache_path: Path) -> None:
    """The --merge-dream-sets mode: pure data, no model, so it runs before any
    adapter is loaded and returns before the run path begins."""
    merged = merge_dream_sets([load_dream_cache(p) for p in args.merge_dream_sets])
    save_dream_cache(merged, cache_path)
    write_dream_set_sidecar(merged, sidecar_path(cache_path))
    print(f"[{ts()}] merged {len(args.merge_dream_sets)} caches -> {cache_path}: "
          f"{len(merged.dreams)} dreams, set_sha {merged.set_sha[:12]}")
    report_dream_set(merged, args.bind_min_dreams, args.rank_rule)


def dream_from_cache(cache: DreamCache, device) -> Dream:
    import torch

    return Dream(
        tokens=torch.tensor([cache.dream_ids], dtype=torch.long, device=device),
        logits=cache.teacher_logits,
        queries=cache.queries,
        final_state=None,
        token_texts=cache.token_texts,
        skipped_cone=0,
        cue_flags=cache.cue_flags,
        prefix_len=cache.prefix_len,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", choices=ARMS, default="replay", help="Sleep protocol, per DISCUSSION sec 3 (default: %(default)s)")
    parser.add_argument("--sft-ref", action="store_true", help="Reference arm: CE on the wake transcript instead of a dream")
    parser.add_argument("--no-sleep", action="store_true", help="Floor arm: no training at all, wake state carried")
    parser.add_argument("--waves", type=int, default=1, help="Wake/sleep waves, each on the state the last one carried; the registered multi-sleep shape is 4 (default: %(default)s)")
    parser.add_argument("--wave-teacher", choices=("base", "current"), default=None,
                        help="Who generates the dream for waves after the first: the frozen base, or the model this run has trained. Required when --waves > 1.")
    parser.add_argument("--n-facts", type=int, default=4, help="Facts per wave -- the measured binding ceiling (default: %(default)s)")
    parser.add_argument("--filler-tokens", type=int, default=40, help="Filler tokens between consecutive facts (default: %(default)s)")
    parser.add_argument("--wake-bystanders", type=int, default=0,
                        help="Off-format bystander items in the wake transcript -- non-fact state content the "
                             "A-vs-B4 targeting contrast is about (default: %(default)s)")
    parser.add_argument("--wake-nearcone", type=int, default=0,
                        help="Numeric-but-off-relation bystander items in the wake transcript (default: %(default)s)")
    parser.add_argument("--wake-dialogue", type=int, default=0,
                        help=f"Ordinary {WAKE_DIALOGUE_SOURCE} exchanges mixed into the wake transcript "
                             "(default: %(default)s)")
    parser.add_argument("--live-wake", action="store_true",
                        help="Use the required adaptive wake plan and user-generator command")
    parser.add_argument("--wake-plan", default=None,
                        help="Frozen JSON turn count and four injection positions; required by --live-wake")
    parser.add_argument("--user-generator-command", nargs="+", default=None,
                        help="Provider adapter command; required by --live-wake")
    parser.add_argument("--wake-scenarios", default=None,
                        help="Frozen JSON list of one held-out scenario spine per wake")
    parser.add_argument("--user-generator-provider", default=None,
                        help="Pinned user-generator provider, recorded in each wake artifact")
    parser.add_argument("--user-generator-model", default=None,
                        help="Pinned user-generator model, recorded in each wake artifact")
    parser.add_argument("--user-generator-version", default=None,
                        help="Pinned user-generator command or model version")
    parser.add_argument("--wake-artifacts", default="data/live_wakes",
                        help="Immutable realized-wake artifacts")
    parser.add_argument("--dream-tokens", type=int, default=DREAM_TOKENS, help="Dream length per sleep (default: %(default)s)")
    parser.add_argument("--dream-temp", type=float, default=1.0, help="Dream sampling temperature (default: %(default)s)")
    parser.add_argument("--dream-prompt", default="", help="Text seeding the dream after the assistant marker (sec 4's category-cue fallback)")
    parser.add_argument("--cue-greedy", type=int, default=12, help="Tokens after each cue decoded greedily -- the recalled code, which temperature sampling almost never gets right (default: %(default)s)")
    parser.add_argument("--cue-every", type=int, default=0, help="Force a fact's question stem into the dream every N tokens, cycling the wave's facts; 0 leaves generation free (default: %(default)s)")
    parser.add_argument("--gate-family", default="hard",
                        choices=("hard", "weighted", "sqrt", "clip", "power2", "power3", "expmed"),
                        help="How gated queries are weighted into the SVD. hard treats every "
                             "kept position alike; the rest weight by state-dependency "
                             "divergence (sec 2.10.7's bake-off axis)")
    parser.add_argument("--rebase-gate-family", default=None,
                        choices=("hard", "weighted", "sqrt", "clip", "power2", "power3", "expmed"),
                        help="Recompute an existing --dream-cache's erasers under this family "
                             "and rewrite it. The dreams, queries and divergences are unchanged, "
                             "so no generation is repeated.")
    parser.add_argument("--merge-dream-sets", nargs="+", default=None, metavar="CACHE",
                        help="Merge these set caches (built concurrently with disjoint "
                             "--dream-seed-offset) into one at --dream-cache, report it, and "
                             "stop. Refuses caches from a different wake transcript or "
                             "generator, or any repeated dream.")
    parser.add_argument("--dream-seed-offset", type=int, default=0,
                        help="Shift this build's generation seeds, so several processes can "
                             "extend one wake state's dream set concurrently instead of "
                             "regenerating identical dreams")
    parser.add_argument("--dreams", type=int, default=0,
                        help="Build/expect a multi-dream cache of N dreams instead of the single-dream one "
                             "(sec 2.10.4's literature-shaped regime); 0 is the single-dream cache "
                             "(default: %(default)s)")
    parser.add_argument("--dream-epochs", type=int, default=1,
                        help="Passes per dream in a dream set. The registered form is one (sec 2.10.2); the "
                             "multi-epoch variant cell prices repetition (default: %(default)s)")
    parser.add_argument("--gate-threshold", type=float, default=GATE_THRESHOLD,
                        help="State-dependency gate (sec 2.9.2): capture a position's read queries when the "
                             "with-state and blank-state next-token distributions diverge by at least this "
                             "many nats. Frozen by the pilot (default: %(default)s)")
    parser.add_argument("--rank-rule", choices=RANK_RULES, default=RANK_RULE,
                        help="Which rank rule truncates each layer's SVD; both are computed and printed either "
                             "way (sec 2.7) (default: %(default)s)")
    parser.add_argument("--bind-min-dreams", type=int, default=BIND_MIN_DREAMS,
                        help="Aggregate binding gate: every fact must bind in at least this many dreams of the "
                             "set, or the cache build fails (default: %(default)s)")
    parser.add_argument("--probe-every-dream", type=int, default=1,
                        help="Run the full probe round at every Nth dream boundary; 2 is the registered "
                             "degradation if probes measurably drag (default: %(default)s)")
    parser.add_argument("--pilot-capture", action="store_true",
                        help="Multi-dream cache builds only: also write the sec 2.10.7 gate-pilot capture "
                             "beside the cache -- every position's read queries and D_t, plus the battery "
                             "items' read queries, so gate_pilot.py can score every gating scheme offline. "
                             "Harness instrumentation; the cache itself is unchanged")
    parser.add_argument("--build-dream-cache", action="store_true",
                        help="Generate this seed's teacher dream, write the cache and its decoded sidecar, and stop. "
                             "Every arm then loads that one dream; no arm generates.")
    parser.add_argument("--dream-cache", default=None,
                        help="Shared dream cache for this seed (default: data/dream_cache_s<seed>.pt)")
    parser.add_argument("--ce-on-dream", action="store_true",
                        help="Decomposition cell: arm A's sequence with cross-entropy on the dream tokens instead of KL")
    parser.add_argument("--fresh-state-replay", action="store_true",
                        help="Arm A: reset the student's state before every chunk, not just at pass boundaries "
                             "(a no-op in the registered full-sequence form, where the chunk is the whole dream)")
    parser.add_argument("--deep", action="store_true",
                        help="B2 only: accumulate the pass's losses and take one optimizer step with the graph "
                             "intact (full BPTT through the spine)")
    parser.add_argument("--spine-block", type=int, default=SPINE_BLOCK,
                        help="Fused B arms: tokens per whole-block forward when materializing the dream spine; "
                             "the blocks then step through their own tokens as one batch (default: %(default)s)")
    parser.add_argument("--cf-batch", type=int, default=CF_BATCH,
                        help="Fused B arms: dream positions whose counterfactuals are forwarded together, "
                             "accumulating into the one optimizer step per pass (default: %(default)s)")
    parser.add_argument("--probe-every", type=int, default=200,
                        help="Stream the full probe battery every N distillation steps, so every cell yields a "
                             "learned-vs-forgotten curve; 0 probes only at the end (default: %(default)s)")
    parser.add_argument("--probe-batch-size", type=int, default=1,
                        help="Independent fact and paraphrase probes per batch; freeze after hardware smoke")
    parser.add_argument("--battery-batch-size", type=int, default=1,
                        help="Independent battery prompts per batch; freeze after hardware smoke")
    parser.add_argument("--dream-batch-size", type=int, default=1,
                        help="Independent dream generations per batch; freeze after hardware smoke")
    parser.add_argument("--distill-steps", type=int, default=200, help="Optimizer steps per sleep (default: %(default)s)")
    parser.add_argument("--accum-window", type=int, default=1, help="Positions accumulated per optimizer step in the per-token arms (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--kl-temp", type=float, default=1.0, help="Distillation temperature (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward chunk (default: the model's DEFAULT_CHUNK_LEN)")
    parser.add_argument("--init-adapter", default=None,
                        help="Warm-start checkpoint directory (a train.py step-N/ dir) loaded into the model before "
                             "the dream cache, the battery or any training; its trainable.pt SHA-256 is stamped "
                             "on every record and the summarizer refuses to pool cells that disagree")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS, help="Tokens generated per probe (default: %(default)s)")
    parser.add_argument("--battery", default=None, help="Knowledge-battery artifact (default: data/knowledge_battery_<model>.json)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="logs/dream_sleep.jsonl", help="Per-fact results jsonl (default: %(default)s)")
    parser.add_argument("--erase-op", choices=ERASE_OPS, default=ERASE_OP,
                        help="Ablation operator for the B arms (sec 3.4's picker): cut along the query itself "
                             "(raw), or along what survives deflating it against the state's top singular "
                             "direction (deflated) (default: %(default)s)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.sft_ref and args.no_sleep:
        raise SystemExit("--sft-ref and --no-sleep are different arms; pass one")
    if args.ce_on_dream and (args.sft_ref or args.no_sleep):
        raise SystemExit("--ce-on-dream is arm A's sequence with a different objective; it is not a reference arm")
    if args.deep and args.arm != "counterfactual":
        raise SystemExit("--deep is a B2 cell (sec 6: deep-B1 is structurally confounded)")
    if args.dream_epochs < 1:
        raise SystemExit("--dream-epochs is passes per dream; it cannot be below 1")
    if min(args.probe_batch_size, args.battery_batch_size, args.dream_batch_size) < 1:
        raise SystemExit("batch sizes must be at least one")
    if args.dreams and args.waves > 1:
        raise SystemExit("a dream set is single-sleep this run (sec 2.4 defers multi-sleep); --waves 1")
    validate_wave_args(args)
    validate_live_wake_args(args)
    mode = "sft-ref" if args.sft_ref else ("no-sleep" if args.no_sleep else
                                           ("ce-on-dream" if args.ce_on_dream else args.arm))

    cache_path = Path(args.dream_cache or default_cache_path(args.seed, args.dreams))
    if args.merge_dream_sets:
        run_merge(args, cache_path)
        return
    if args.rebase_gate_family:
        run_rebase(args, cache_path)
        return
    if not args.build_dream_cache and not cache_path.exists():
        raise SystemExit(
            f"no dream cache at {cache_path}. Every arm distils the one dream this seed's cache holds -- "
            f"build it first:\n  make dream-sleep ARGS=\"--build-dream-cache --seed {args.seed} ...\""
        )

    import importlib
    import os

    import torch
    from dotenv import load_dotenv

    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "mamba2_780m")
    model_mod = importlib.import_module(f"models.{model_name}")
    train_hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    from models.common import build_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_len = args.chunk_len or getattr(train_hooks, "DEFAULT_CHUNK_LEN", 48)
    print(f"[{ts()}] model {model_name} on {device}, arm {mode}, carries {ARM_CARRY[mode]}, "
          f"chunk_len {chunk_len}, seed {args.seed}, gamma {GAMMA}, erase {args.erase_op}"
          f"{f' (state-svd k={DEFLATE_K})' if args.erase_op == 'deflated' else ''}")

    model, trainable = train_hooks.setup_training(device, args.lora_rank, args.lora_alpha, DEFAULT_DROPOUT)
    if getattr(model, "c_capture", "missing") == "missing":
        raise SystemExit(f"model {model_name} has no c_capture hook -- this harness is for mamba2_780m")
    adapter_sha = None
    if args.init_adapter:
        adapter_sha = load_init_adapter(model, args.init_adapter, args.lora_rank, args.lora_alpha)
        print(f"[{ts()}] warm start {args.init_adapter}: sha256 {adapter_sha[:12]}")
    model.eval()
    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    # None for a model that registers no conversation-end marker: such a dream
    # has only the token budget and the turn backstop to end it.
    eoc = getattr(model_mod, "EOC", None)
    stop_id = tokenizer.convert_tokens_to_ids(eoc) if eoc else None
    stops = (".", "\n", user_open, asst_open)
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device)

    def decode(ids) -> str:
        return tokenizer.decode(ids)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    all_facts = build_facts(args.n_facts * args.waves, rng)
    validate_battery_candidates(
        BATTERY_CANDIDATES,
        lambda answer: len(tokenizer(" " + answer, add_special_tokens=False)["input_ids"]),
        [part for fact in all_facts for part in (fact.entity, fact.code)],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w")

    cache = None if args.build_dream_cache else load_dream_cache(cache_path)
    is_set = isinstance(cache, DreamSetCache)
    if is_set and args.arm not in DREAM_SET_ARMS and not (args.sft_ref or args.no_sleep):
        raise SystemExit(f"{cache_path} is a dream set, which runs arms {DREAM_SET_ARMS} only (sec 2.10.1)")
    if is_set and args.waves > 1:
        raise SystemExit(f"{cache_path} is a dream set and multi-sleep is deferred (sec 2.4); --waves 1")
    if args.arm in B4_ARMS and not is_set:
        raise SystemExit(f"arm {args.arm} erases once per dream and needs a dream set; {cache_path} holds one dream")

    # The dream-SET hash rides every record (sec 3), the way the single-dream
    # hashes already ride the cache record.
    emit = make_emit(out_file, erase_op=args.erase_op, init_adapter_sha256=adapter_sha,
                     init_adapter=Path(args.init_adapter).resolve().name if args.init_adapter else None,
                     dream_set_sha=cache.set_sha if is_set else None,
                     probe_batch_size=args.probe_batch_size, battery_batch_size=args.battery_batch_size,
                     dream_batch_size=args.dream_batch_size)

    distractors = dict(cache.distractors) if cache else {}
    if is_set:
        print(f"[{ts()}] dream set {cache_path}: {len(cache.dreams)} dreams, set_sha {cache.set_sha[:12]}, "
              f"transcript_sha {cache.transcript_sha[:12]}, prefix {cache.dream_prompt!r}, "
              f"gate {cache.gate_threshold} nats, rank rule {cache.rank_rule}, "
              f"generated by {cache.generator[:12]}")
    elif cache is not None:
        print(f"[{ts()}] dream cache {cache_path}: {len(cache.dream_ids)} tokens "
              f"({cache.free_tokens} freely generated), transcript_sha {cache.transcript_sha[:12]} "
              f"dream_sha {cache.dream_sha[:12]}, generated by {cache.generator[:12]}")
    if cache is not None:
        if cache.generator != (adapter_sha or "base"):
            raise SystemExit(
                f"dream cache {cache_path} was generated by {cache.generator[:12]} but this cell runs at "
                f"{(adapter_sha or 'base')[:12]}: sec 4(1) registers the cache as an artifact OF the "
                f"warm-started weights. Rebuild the cache under the same --init-adapter."
            )

    # ---- probes -----------------------------------------------------------
    def answer_probe(prompt: str, answer: str, state=None) -> tuple[str, float]:
        prompt_ids, target_ids = encode(prompt), encode(" " + answer)
        generation = decode(generate(model, prompt_ids, copy.deepcopy(state), args.gen_tokens, 0.0)[0].cpu())
        return generation, target_logprob(model, prompt_ids, target_ids, copy.deepcopy(state))

    def margin_probe(fact: Fact, state) -> tuple[float, bool, float, float]:
        """The primary reliability metric (sec 4): the correct code's summed
        log-prob against a fixed distractor code's, same question, same state.
        Both codes gain equally from the format prior, so the gap is what the
        weights actually learned."""
        prompt_ids = encode(cue_rungs(fact, user_open, asst_open)[0][0])
        correct = logprob_sum(model, prompt_ids, encode(" " + fact.code), copy.deepcopy(state))
        foil = logprob_sum(model, prompt_ids, encode(" " + distractors[fact.entity]), copy.deepcopy(state))
        margin, installed = code_margin(correct, foil)
        return margin, installed, correct, foil

    def probe_facts(facts: Sequence[tuple[int, Fact]], state, phase: str, wave: int, paraphrases: bool,
                    baseline: dict[str, float] | None, step: int | None = None,
                    collect: dict[str, dict[str, object]] | None = None) -> dict[str, float]:
        """Greedy exact match, teacher-forced code log-prob and the distractor
        margin per fact, plus the paraphrase battery where asked. Streams one
        record per fact."""
        logprobs: dict[str, float] = {}
        hits = para_hits = installs = 0
        for i, (fact_wave, fact) in enumerate(facts):
            generation, logprob = answer_probe(cue_rungs(fact, user_open, asst_open)[0][0], fact.code, state)
            matched = exact_match(generation, fact.code, stops)
            logprobs[fact.entity] = logprob
            margin = correct_sum = foil_sum = None
            installed = False
            if fact.entity in distractors:
                margin, installed, correct_sum, foil_sum = margin_probe(fact, state)
                installs += installed
            para: list[dict[str, object]] = []
            if paraphrases:
                for prompt in paraphrase_prompts(fact, user_open, asst_open):
                    gen, _ = answer_probe(prompt, fact.code, state)
                    para.append({"prompt": prompt, "greedy": gen, "match": exact_match(gen, fact.code, stops)})
            para_rate = sum(bool(p["match"]) for p in para) / len(para) if para else 0.0
            if collect is not None:
                collect[fact.entity] = {"fact_wave": fact_wave, "margin": margin, "install": installed}
            hits += matched
            para_hits += para_rate
            delta = logprob - baseline[fact.entity] if baseline and fact.entity in baseline else None
            emit({
                "phase": phase, "wave": wave, "arm": mode, "step": step, "fact_wave": fact_wave, "fact": fact.entity,
                "category": fact.category, "code": fact.code, "greedy": generation, "match": matched,
                "logprob": logprob, "logprob_delta": delta, "paraphrases": para, "paraphrase_rate": para_rate,
                "margin": margin, "lp_sum_correct": correct_sum, "lp_sum_distractor": foil_sum,
                "margin_install": installed, "distractor": distractors.get(fact.entity),
            })
            print(f"[{ts()}]  {phase} w{wave}{'' if step is None else f' s{step}'} {fact.entity:<11} "
                  f"{'HIT ' if matched else 'miss'} lp {logprob:+.3f}"
                  f"{'' if delta is None else f' (d {delta:+.3f})'} "
                  f"{'margin   n/a' if margin is None else f'margin {margin:+7.3f}'}"
                  f"{' INSTALL' if installed else '        '} "
                  f"para {para_rate:.2f} | running match {hits / (i + 1):.2f} install {installs / (i + 1):.2f} "
                  f"para {para_hits / (i + 1):.2f} | {generation[:40]!r}", flush=True)
        return logprobs

    battery_path = Path(args.battery or f"data/knowledge_battery_{model_name}.json")
    heldout = encode(HELDOUT_TEXT)

    def battery_probe(prompt: str) -> tuple[str, float]:
        item = next((c for c in BATTERY_CANDIDATES if c[0] == prompt), None)
        return answer_probe(prompt, item[1] if item else "", None)

    def scored_battery_probe(prompt: str) -> tuple[str, float]:
        answer = next(str(i["answer"]) for i in battery if i["prompt"] == prompt)
        return answer_probe(prompt, answer, None)

    print(f"\n[{ts()}] === pre-training locality baseline (base model, fresh state) ===")
    battery = load_or_build_battery(battery_path, BATTERY_CANDIDATES, battery_probe,
                                    checkpoint_sha=adapter_sha or "base")
    if len(battery) < 100:
        raise SystemExit(f"the self-calibrated knowledge battery kept {len(battery)} items; need at least 100")
    base_ppl = perplexity(model, heldout, chunk_len, "heldout ppl")
    print(f"[{ts()}] held-out ppl {base_ppl:.3f} over {heldout.shape[1]} tokens; battery {len(battery)} items")
    emit({"phase": "baseline", "arm": mode, "battery_items": len(battery), "ppl": base_ppl,
          "batch_sizes": {"probe": args.probe_batch_size, "battery": args.battery_batch_size,
                          "dream": args.dream_batch_size}})

    def locality(wave: int, step: int | None = None) -> None:
        scored = score_battery_batched(
            battery, lambda prompts: [scored_battery_probe(prompt) for prompt in prompts], args.battery_batch_size)
        summary = battery_summary(scored)
        for record in scored:
            emit({"phase": "battery", "wave": wave, "arm": mode, "step": step, **record})
            if not record["correct"]:
                print(f"[{ts()}]  battery LOST {record['prompt']!r} -> {str(record['greedy_post'])[:40]!r} "
                      f"(dlp {float(record['logprob_delta']):+.3f})", flush=True)
        ppl = perplexity(model, heldout, chunk_len, "heldout ppl")
        emit({"phase": "locality", "wave": wave, "arm": mode, "step": step, "ppl": ppl,
              "ppl_delta": ppl - base_ppl, **summary})
        print(f"[{ts()}]  battery retained {summary['retained_rate']:.3f} ({summary['lost']} lost of "
              f"{summary['items']}), mean dlogp {summary['mean_logprob_delta']:+.4f}")
        print(f"[{ts()}]  held-out ppl {ppl:.3f}  (dPPL {ppl - base_ppl:+.4f})")

    # ---- waves ------------------------------------------------------------
    rich_wake = bool(args.wake_bystanders or args.wake_nearcone or args.wake_dialogue)
    dialogue_records = load_dialogue_records() if args.wake_dialogue else []
    carried = None
    seen: list[tuple[int, Fact]] = []
    fresh_baseline: dict[str, float] = {}
    leak_baseline: dict[str, float] = {}
    committed: set[str] = set()
    r_matrix: dict[tuple[int, int], dict[str, float]] = {}
    for wave in range(1, args.waves + 1):
        facts = all_facts[(wave - 1) * args.n_facts : wave * args.n_facts]
        if wave > 1:
            # Wave 1's foils come from the cache, so every arm shares them; later
            # waves derive theirs, avoiding every code already in play.
            distractors |= build_distractors(
                facts, args.seed, taken=set(distractors.values()) | {f.code for _, f in seen})
        token_len = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
        if rich_wake:
            items, wake_distractors = build_wake_items(
                facts, args.wake_bystanders, args.wake_nearcone, args.wake_dialogue,
                dialogue_records, rng, user_open, asst_open,
                [str(item["answer"]) for item in battery])
            turns = build_mixed_turns(items, args.filler_tokens, token_len, rng)
        else:
            wake_distractors = []
            turns = build_turns(facts, args.filler_tokens, token_len, rng)
        text = render_turns(turns, user_open, asst_open)
        transcript = encode(text)
        print(f"\n[{ts()}] === wave {wave} wake ===")
        report_transcript(text, decode(transcript[0].cpu()), facts, turns, transcript.shape[1])
        report_distractors(decode(transcript[0].cpu()), wake_distractors)
        emit({"phase": "transcript", "wave": wave, "arm": mode, "tokens": transcript.shape[1],
              "facts": [f.entity for f in facts],
              "distractors": [d.label for d in wake_distractors]})

        if args.build_dream_cache:
            builder = build_dream_set if args.dreams else build_cache
            builder(model, args, cache_path, transcript, facts, chunk_len,
                    encode, decode, tokenizer, user_open, asst_open, stop_id,
                    adapter_sha=adapter_sha,
                    **({"battery": battery} if args.dreams else {}))
            out_file.close()
            print(f"\n[{ts()}] cache built; every arm of seed {args.seed} now distils this dream")
            return

        if wave == 1:
            if token_sha(transcript[0].tolist()) != cache.transcript_sha:
                raise SystemExit(
                    f"wave-1 transcript does not match {cache_path}'s: this cell would distil a dream generated "
                    f"from a different wake session. Rebuild the cache for seed {args.seed}."
                )
            record = {"phase": "cache", "wave": 1, "arm": mode, "seed": args.seed, "path": str(cache_path),
                      "transcript_sha": cache.transcript_sha, "dream_generator": cache.generator}
            if is_set:
                record |= {"dreams": len(cache.dreams), "set_sha": cache.set_sha,
                           "dream_shas": [d.dream_sha for d in cache.dreams],
                           "dream_prompt": cache.dream_prompt, "gate_threshold": cache.gate_threshold,
                           "rank_rule": cache.rank_rule,
                           "stop_reasons": [d.stop_reason for d in cache.dreams],
                           "dream_tokens": [len(d.dream_ids) for d in cache.dreams]}
            else:
                record |= {"stop_reason": cache.stop_reason, "dream_sha": cache.dream_sha,
                           "dream_tokens": len(cache.dream_ids), "free_tokens": cache.free_tokens,
                           "cue_tokens": len(cache.dream_ids) - cache.free_tokens}
            emit(record)

        # The fresh-state floor for this wave's facts, before they are anywhere
        # but the transcript -- what every later log-prob delta is measured against.
        print(f"[{ts()}] fresh-state floor, wave {wave} facts")
        fresh_baseline |= probe_facts([(wave, f) for f in facts], None, "floor", wave, False, None)
        if wake_distractors:
            print(f"[{ts()}] fresh-state floor, wave {wave} distractor content")
            leak_baseline |= probe_leakage(wake_distractors, answer_probe, emit, mode, wave,
                                           "leak_floor", stops)

        if wave == 1:
            # The cached wake state, not a fresh forward: it is the state the
            # cached dream was generated from, so the arms are not comparing
            # their own re-derivations of it.
            carried = state_to(copy.deepcopy(cache.wake_state), device)
        else:
            carried = run_chunks(model, transcript, carried, chunk_len, f"wake {wave}", keep_logits=False)[1]
        seen += [(wave, f) for f in facts]

        # In-context control. On wave 2 this is the consumption price: B1 enters
        # selectively vacated, B2 full, A empty.
        print(f"[{ts()}] in-context control, wave {wave} facts (on the carried state)")
        probe_facts([(wave, f) for f in facts], carried, "in_context", wave, False, None)

        def periodic_probe(step: int, wave: int = wave) -> None:
            """The full battery mid-sleep, so every cell yields a curve and
            iso-learning comparisons are read off it rather than engineered
            with hyperparameters (sec 4)."""
            model.eval()
            print(f"\n[{ts()}] --- wave {wave} probe battery at step {step} ---")
            probe_facts(seen, None, "probe", wave, True, fresh_baseline, step=step)
            locality(wave, step=step)
            model.train()

        if mode == "no-sleep":
            print(f"\n[{ts()}] === wave {wave} sleep: none (--no-sleep floor) ===")
        else:
            print(f"\n[{ts()}] === wave {wave} sleep: {mode} ===")
            carried = run_sleep(mode, model, opt, args, wave, carried, transcript, seen, chunk_len, cache,
                                encode, decode, tokenizer, user_open, asst_open, emit, periodic_probe,
                                stop_id, verify=lambda f: margin_probe(f, None), committed=committed)

        print(f"\n[{ts()}] === wave {wave} probes: fresh state, no context ===")
        scored: dict[str, dict[str, object]] = {}
        probe_facts(seen, None, "probe", wave, True, fresh_baseline, step=args.distill_steps,
                    collect=scored)
        if wake_distractors:
            probe_leakage(wake_distractors, answer_probe, emit, mode, wave, "leakage", stops,
                          baseline=leak_baseline, step=args.distill_steps)
        locality(wave, step=args.distill_steps)

        # Column `wave` of the R-matrix, streamed the moment the sleep produces
        # it: read at any point, the log already says what each earlier wave's
        # facts are worth now.
        for fact_wave, stats in sorted(r_matrix_row(scored).items()):
            r_matrix[(fact_wave, wave)] = stats
            emit({"phase": "r_matrix", "arm": mode, "sleep": wave, "fact_wave": fact_wave, **stats})
            print(f"[{ts()}]  R[wave {fact_wave}][sleep {wave}] mean margin {stats['mean_margin']:+7.3f} "
                  f"installs {stats['installs']}/{stats['facts']}")
        print(f"\n[{ts()}] === wave {wave} carried-state diagnostic ({ARM_CARRY[mode]}) -- NOT installation ===")
        if carried is None:
            print(f"[{ts()}]  arm {mode} carries nothing; column empty by construction")
        else:
            probe_facts(seen, carried, "carried", wave, False, None)

    if args.waves > 1:
        summary = cl_summary(r_matrix, args.waves)
        print(f"\n[{ts()}] === R-matrix (rows: the wave that taught the facts; columns: after sleep j) ===")
        for i, row in enumerate(r_matrix_rows(r_matrix, args.waves), start=1):
            print(f"[{ts()}]  wave {i}: " + "  ".join("     ." if m is None else f"{m:+8.3f}" for m in row))
        print(f"[{ts()}] BWT {'n/a' if summary['bwt'] is None else f'{summary['bwt']:+.3f}'}   "
              f"installs {summary['installs_final']} of a peak {summary['installs_peak']}")
        emit({"phase": "cl_summary", "arm": mode, "seed": args.seed,
              "r_matrix": r_matrix_rows(r_matrix, args.waves), **summary})

    # The cell's completion marker. Periodic probes write a locality record
    # every --probe-every steps, so "has a locality record" says a cell started,
    # not that it finished; the driver's resume check and the summarizer both
    # key on this record instead.
    emit({"phase": "done", "arm": mode, "seed": args.seed, "waves": args.waves,
          "steps": args.distill_steps})
    out_file.close()
    print(f"\n[{ts()}] done -> {out_path}")




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


if __name__ == "__main__":
    main()
