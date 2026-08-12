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
import contextlib
import copy
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation_null import (
    CODE_DIGITS,
    GEN_TOKENS,
    Fact,
    build_facts,
    build_turns,
    cue_rungs,
    exact_match,
    extract_answer,
    fmt_duration,
    generate,
    kl_loss,
    normalize,
    render_turns,
    replay_step,
    report_transcript,
    run_chunks,
    target_logprob,
    ts,
)
from b4 import (
    RANK_RULES,
    VARIANTS,
    address_budget,
    aggregate_basis,
    erase_state_subspace,
    gated_positions,
    rank_median,
    rank_ratio_gap,
    state_divergence,
    variant_basis,
)
from gate_pilot import PilotCapture, PilotDream
from erase_probe import (
    build_mixed_turns,
    build_wake_items,
    deflate,
    group_by_layer,
    rank1_erase,
    report_distractors,
    state_top_dirs,
)
from lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from probes_common import (
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


@contextlib.contextmanager
def frozen_teacher(model) -> Iterator[object]:
    """Run the enclosed forwards as the frozen base model: LoRA adapters
    bypassed (scale 0 zeroes both the forward's adapter term and the merged
    `.weight` the fused kernel reads) and the marker delta -- zero-initialised,
    so zeroing it is the base model -- temporarily zeroed too."""
    from lora import LoRALinear

    adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
    scales = [m.scale for m in adapters]
    delta = getattr(model, "marker_delta", None)
    saved = delta.delta.detach().clone() if delta is not None else None
    for m in adapters:
        m.scale = 0.0
    if delta is not None:
        delta.delta.data.zero_()
    try:
        yield model
    finally:
        for m, s in zip(adapters, scales, strict=True):
            m.scale = s
        if delta is not None:
            delta.delta.data.copy_(saved)


def sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Next token from (B, V) logits. Nothing is masked: eos is dream-internal
    turn structure and <|eoc|> is how a dream ends (DISCUSSION-20260808
    sec 2.10.10)."""
    import torch

    last = logits.float().clone()
    if temperature <= 0:
        return last.argmax(dim=-1, keepdim=True)
    return torch.multinomial(torch.softmax(last / temperature, dim=-1), num_samples=1)


def erase_ssm(ssm_state: torch.Tensor, c: torch.Tensor, gamma: float = GAMMA, k: int = DEFLATE_K,
              op: str = ERASE_OP):
    """The registered erase for one layer, in the operator the cell was
    launched with (sec 3.4's B1-raw vs B1-deflated picker): `raw` attenuates
    the state's read along the query itself, `deflated` first removes the
    query's component along the state's own top singular direction. Returns
    (state, skipped) -- the number of near-cone directions skipped, which only
    `deflated` can produce, when nothing discriminative survives the deflation.

    The direction is differentiable (the cut follows the query); the protected
    subspace it is deflated against is stop-gradiented at its source.

    Batched over the leading dimension throughout: the fused arms ablate every
    dream position at once, each with its own basis and its own skip decision.

    This is what Model.erase_hook is fed: the ablation lands on the carried
    past, before this token's decay+write (sec 3's micro-order)."""
    import torch

    if op not in ERASE_OPS:
        raise ValueError(f"unknown erase op {op!r}; expected one of {ERASE_OPS}")
    c = c.to(ssm_state.device)
    if op == "raw":
        return rank1_erase(ssm_state, c, gamma), 0
    direction = deflate(c, state_top_dirs(ssm_state, k))
    skip = direction.float().norm(dim=-1) < CONE_SKIP * c.float().norm(dim=-1)
    if bool(skip.all()):
        return ssm_state, int(skip.sum())
    erased = rank1_erase(ssm_state, direction, gamma)
    if bool(skip.any()):
        erased = torch.where(skip.view(-1, *([1] * (ssm_state.dim() - 1))), ssm_state, erased)
    return erased, int(skip.sum())


def erase_state(state, queries: Sequence[torch.Tensor], gamma: float = GAMMA, k: int = DEFLATE_K,
                op: str = ERASE_OP) -> int:
    """Apply the registered erase to every layer of `state` in place, each with
    that layer's own read query. Returns the number of near-cone directions
    skipped."""
    skipped = 0
    for i, c in enumerate(queries):
        state.ssm_states[i], hit = erase_ssm(state.ssm_states[i], c, gamma, k, op)
        skipped += hit
    return skipped


def make_erase_hook(erase_op: str):
    """The `Model.erase_hook` the counterfactual arms run their forward under,
    paired with a read of how many near-cone directions it has skipped so far."""
    skipped = 0

    def hook(layer_idx: int, ssm_state, c):
        nonlocal skipped
        erased, was_skipped = erase_ssm(ssm_state, c, op=erase_op)
        skipped += was_skipped
        return erased

    return hook, lambda: skipped


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


def token_sha(ids: Sequence[int]) -> str:
    """SHA-256 over a token sequence. A registered invariant ships with its
    machine check (sec 2): every result file records these and the summarizer
    refuses to pool cells whose dream or transcript disagree."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


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


def fact_read_positions(token_texts: Sequence[str], facts: Sequence[Fact]) -> dict[str, list[int]]:
    """Token positions inside a BOUND rehearsal of each fact's code -- the
    binding scan, as positions rather than counts.

    Sec 2.9.1: this is a VALIDATION overlay. Nothing in the eraser's path may
    call it; it exists to score the state-dependency gate after the fact.
    """
    text = "".join(token_texts)
    spans, pos = [], 0
    for piece in token_texts:
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    out: dict[str, list[int]] = {}
    for fact in facts:
        hits: set[int] = set()
        at = text.find(fact.code)
        while at != -1:
            start = max(text.rfind(stop, 0, at) for stop in CUE_STOPS) + 1
            ends = [text.find(stop, at) for stop in CUE_STOPS]
            end = min([e for e in ends if e != -1], default=len(text))
            if fact.entity.lower() in text[start:end].lower():
                hits |= {i for i, (lo, hi) in enumerate(spans) if lo < at + len(fact.code) and hi > at}
            at = text.find(fact.code, at + 1)
        out[fact.entity] = sorted(hits)
    return out


def gate_agreement(gate: Sequence[int], fact_positions: dict[str, list[int]]) -> dict[str, object]:
    """Precision/recall of the state-dependency gate against the binding scan,
    plus each fact's contribution count (sec 2.9.2).

    Diagnostic only -- sec 2.10.6 names label accuracy a NON-goal, since the
    state and not the labels is what the eraser touches. A fact contributing
    zero gated positions is still loud: the eraser cannot address what it never
    captured.
    """
    reads = {t for positions in fact_positions.values() for t in positions}
    gated = set(gate)
    hit = len(gated & reads)
    return {"precision": hit / len(gated) if gated else 0.0,
            "recall": hit / len(reads) if reads else 0.0,
            "gated": len(gated), "read_positions": len(reads),
            "per_fact": {e: len(gated & set(p)) for e, p in fact_positions.items()}}


def load_dialogue_records(n: int = WAKE_DIALOGUE_POOL) -> list[dict]:
    """Candidate conversations for the wake transcript's dialogue slice."""
    from datasets import load_dataset

    print(f"[{ts()}] loading {n} {WAKE_DIALOGUE_SOURCE} candidates for the wake dialogue slice")
    ds = load_dataset(WAKE_DIALOGUE_SOURCE, split=f"{WAKE_DIALOGUE_SPLIT}[:{n}]")
    return [{"messages": r["messages"]} for r in ds]


def probe_leakage(items, answer_probe, emit, arm: str, wave: int, phase: str,
                  stops: Sequence[str], baseline: dict[str, float] | None = None,
                  step: int | None = None) -> dict[str, float]:
    """Fresh-state QA on the wake transcript's distractor content (sec 2.10.11).

    Neither arm should install any of it: A denies the student the whole state
    and B4 denies only what the dream read, so distractor content is what
    "targeted" is supposed to leave alone. A rising log-prob here is untargeted
    consolidation, measured at its origin. Returns each item's log-prob, which
    is the floor a later call is read against."""
    logprobs: dict[str, float] = {}
    hits = 0
    for i, item in enumerate(items):
        generation, logprob = answer_probe(item.prompt, item.answer.strip())
        answer, truth = extract_answer(generation, stops), normalize(item.answer)
        matched = answer == truth or answer.startswith(truth + " ")
        logprobs[item.label] = logprob
        hits += matched
        delta = logprob - baseline[item.label] if baseline and item.label in baseline else None
        emit({"phase": phase, "wave": wave, "arm": arm, "step": step, "item": item.label,
              "kind": item.cls, "answer": item.answer.strip(), "greedy": generation,
              "match": matched, "logprob": logprob, "logprob_delta": delta})
        print(f"[{ts()}]  {phase} w{wave}{'' if step is None else f' s{step}'} {item.label:<11} "
              f"{'HIT ' if matched else 'miss'} lp {logprob:+.3f}"
              f"{'' if delta is None else f' (d {delta:+.3f})'} "
              f"| running leak {hits / (i + 1):.2f} | {generation[:40]!r}", flush=True)
    return logprobs


def binding_coverage(text: str, facts: Sequence[Fact]) -> tuple[dict[str, int], dict[str, int]]:
    """Rehearsals that actually bind: a code counts only where it appears in
    the same sentence as its own entity. A code sitting next to a *different*
    fact's entity is a misbinding and is reported separately -- the 08-06 grid
    counted "The code for the heron is <osprey's code>" as coverage."""
    bound = {f.entity: 0 for f in facts}
    misbound = {f.entity: 0 for f in facts}
    for sentence in re.split(r"[.\n]", text):
        low = sentence.lower()
        for fact in facts:
            if fact.code not in sentence:
                continue
            if fact.entity.lower() in low:
                bound[fact.entity] += 1
            elif any(other.entity.lower() in low for other in facts if other.entity != fact.entity):
                misbound[fact.entity] += 1
    return bound, misbound


def copy_state(state):
    """A copy of a MixerState whose tensors keep their autograd history --
    `copy.deepcopy` refuses non-leaf tensors, which is what the deep-BPTT
    variant carries."""
    new = copy.copy(state)
    for attr in ("conv_states", "ssm_states"):
        if hasattr(state, attr):
            setattr(new, attr, [t.clone() for t in getattr(state, attr)])
    return new


def dream_seed_text(asst_open: str, prompt: str) -> str:
    """The dream's seed text. The trained chat format is the marker plus a
    literal space, so the separator survives an empty --dream-prompt."""
    return f"{asst_open} {prompt}" if prompt else f"{asst_open} "


def longest_verbatim_run(dream_tokens: Sequence[str], transcript_tokens: Sequence[str],
                         n: int = 12) -> int:
    """Longest run of consecutive dream tokens appearing verbatim in the wake
    transcript.

    `copy_fraction` answers "how much of this dream is reused phrasing"; this
    answers "did the dream REPLAY the transcript". They differ sharply: a dream
    repeating wake sentences one at a time scores a high fraction with a run of
    ~15, while a wholesale replay shows a run of hundreds.
    """
    if not dream_tokens or not transcript_tokens or n <= 0:
        return 0
    grams = {tuple(transcript_tokens[i : i + n]) for i in range(len(transcript_tokens) - n + 1)}
    best = current = 0
    covered = [False] * len(dream_tokens)
    for i in range(len(dream_tokens) - n + 1):
        if tuple(dream_tokens[i : i + n]) in grams:
            for j in range(i, i + n):
                covered[j] = True
    for flag in covered:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def copy_fraction(dream_tokens: Sequence[str], transcript_tokens: Sequence[str],
                  n: int = 12, cue_flags: Sequence[bool] | None = None) -> float:
    """Fraction of a dream's tokens that sit inside a run of at least `n`
    consecutive tokens appearing verbatim in the wake transcript.

    A recall-trained corpus can push generation from dreaming ABOUT the wake
    into REPLAYING it. Rehearsal counting cannot see that -- a verbatim copy
    scores perfect fact coverage -- so copying is measured separately, and a
    dream set reports both. `n` is well above ordinary language reuse: short
    shared phrases ("the code for the") are not regurgitation.
    """
    if cue_flags is not None:
        # Spliced cue text is the wake session's own phrasing, so it matches
        # the transcript by construction and says nothing about the model.
        keep = [i for i, token in enumerate(dream_tokens)
                if not (i < len(cue_flags) and cue_flags[i])]
        dream_tokens = [dream_tokens[i] for i in keep]
    if not dream_tokens or not transcript_tokens or n <= 0:
        return 0.0
    grams: set[tuple[str, ...]] = {
        tuple(transcript_tokens[i : i + n]) for i in range(len(transcript_tokens) - n + 1)
    }
    copied = [False] * len(dream_tokens)
    for i in range(len(dream_tokens) - n + 1):
        if tuple(dream_tokens[i : i + n]) in grams:
            for j in range(i, i + n):
                copied[j] = True
    return sum(copied) / len(copied)


def rehearsal_fraction(token_texts: Sequence[str], needles: Sequence[str]) -> tuple[float, dict[str, int]]:
    """Fraction of dream tokens whose characters fall inside an occurrence of
    a fact's entity or code, and the occurrence count per needle. A dream that
    never rehearses the facts never touches the bindings, so nothing can
    distil -- this is the first number to read in a smoke run (sec 4)."""
    text = "".join(token_texts).lower()
    starts, pos = [], 0
    for piece in token_texts:
        starts.append((pos, pos + len(piece)))
        pos += len(piece)
    covered = [False] * len(token_texts)
    counts: dict[str, int] = {}
    for needle in needles:
        low = needle.lower()
        counts[needle] = 0
        at = text.find(low)
        while at != -1:
            counts[needle] += 1
            for i, (lo, hi) in enumerate(starts):
                if lo < at + len(low) and hi > at:
                    covered[i] = True
            at = text.find(low, at + 1)
    return (sum(covered) / len(token_texts) if token_texts else 0.0), counts


@dataclass
class Dream:
    """A cached teacher pass: the dream tokens, the teacher's logits at every
    position, and the read query every layer issued there (what the erase and
    the counterfactual ablation are addressed with)."""

    tokens: torch.Tensor  # (1, T)
    logits: torch.Tensor  # (T, V), cpu float32
    queries: list[list[torch.Tensor]]  # T x n_layers, cpu
    final_state: object
    token_texts: list[str]
    skipped_cone: int
    cue_flags: list[bool] = field(default_factory=list)  # True where the token was spliced in as a cue
    stop_reason: str = "max-tokens"  # one of STOP_REASONS
    prefix_len: int = 0  # steer-prefix tokens, never scored (sec 2.10.8)


@dataclass
class DreamCache:
    """The one teacher dream per seed that every arm distils (sec 3's shared
    preamble). Generated once by `--build-dream-cache`, loaded by every arm --
    no arm generates, so a cross-arm comparison is a comparison of arms.

    `transcript_sha`/`dream_sha` are recomputed on load and on every result
    file, which is what makes "byte-identical across arms" an assertion rather
    than a sentence in a design doc."""

    seed: int
    transcript_ids: list[int]
    dream_ids: list[int]
    wake_state: object
    teacher_logits: torch.Tensor  # (T, V), cpu float32
    queries: list[list[torch.Tensor]]
    token_texts: list[str]
    cue_flags: list[bool]
    distractors: dict[str, str]
    facts: list[tuple[str, str, str]]  # entity, category, code
    stop_reason: str = "max-tokens"
    # The steer prefix (sec 2.10.8), as text and as its token count: prefix
    # tokens condition the dream through state only and are excluded from every
    # arm's scored positions.
    dream_prompt: str = ""
    prefix_len: int = 0
    transcript_sha: str = ""
    dream_sha: str = ""
    # Which weights emitted this dream: "base" or the --init-adapter SHA-256.
    generator: str = "base"

    def __post_init__(self) -> None:
        self.transcript_sha = self.transcript_sha or token_sha(self.transcript_ids)
        self.dream_sha = self.dream_sha or token_sha(self.dream_ids)

    @property
    def fact_list(self) -> list[Fact]:
        return [Fact(*f) for f in self.facts]

    @property
    def free_tokens(self) -> int:
        """Tokens the model actually generated, as opposed to spliced cue text
        -- the trainable free-dream budget (sec 4's cue accounting)."""
        return sum(not flag for flag in self.cue_flags)


@dataclass
class CachedDream:
    """One dream of a multi-dream cache (sec 2.10.4).

    Carries everything a B4 cell needs and nothing it has to recompute: the
    tokens, the frozen teacher's logits, the state-dependency gate's decision
    (its per-position divergence and the positions it kept), the raw queries at
    those positions, each layer's spectrum with both rank rules' answers, and
    the eraser itself -- one orthonormal basis per layer per variant, all three
    from the same shared SVD.
    """

    dream_ids: list[int]
    token_texts: list[str]
    teacher_logits: torch.Tensor  # (T, V), cpu float32
    cue_flags: list[bool]
    prefix_len: int
    stop_reason: str
    divergence: list[float]  # D_t at every position
    gate_positions: list[int]
    queries: list[list[torch.Tensor]]  # gated positions x n_layers
    spectra: list[list[float]]  # per layer
    ranks: dict[str, list[int]]  # rank rule -> per-layer r
    bases: dict[str, list[torch.Tensor]]  # variant -> per-layer (r, n), rows orthonormal
    dream_sha: str = ""

    def __post_init__(self) -> None:
        self.dream_sha = self.dream_sha or token_sha(self.dream_ids)

    @property
    def free_tokens(self) -> int:
        return sum(not flag for flag in self.cue_flags)


def dream_set_sha(dreams: Sequence[CachedDream]) -> str:
    """The set hash asserted into every result jsonl (sec 3): over the dreams'
    own hashes in order, so a reordered or substituted set is a different set."""
    return hashlib.sha256("\n".join(d.dream_sha for d in dreams).encode()).hexdigest()


@dataclass
class DreamSetCache:
    """The N-dream cache the literature-shaped regime distils (sec 2.10.4).

    Every dream is generated upfront, each from a fresh copy of the INTACT wake
    state, by the sleep-start snapshot; the whole set is shared by every arm,
    which is what makes an A-vs-B4 pairing a comparison of arms.
    """

    seed: int
    transcript_ids: list[int]
    wake_state: object
    dreams: list[CachedDream]
    distractors: dict[str, str]
    facts: list[tuple[str, str, str]]  # entity, category, code
    dream_seed_offset: int = 0
    gate_family: str = "hard"
    dream_prompt: str = ""
    gate_threshold: float = GATE_THRESHOLD
    rank_rule: str = RANK_RULE
    generator: str = "base"
    transcript_sha: str = ""
    set_sha: str = ""

    def __post_init__(self) -> None:
        self.transcript_sha = self.transcript_sha or token_sha(self.transcript_ids)
        self.set_sha = self.set_sha or dream_set_sha(self.dreams)

    @property
    def fact_list(self) -> list[Fact]:
        return [Fact(*f) for f in self.facts]


def aggregate_binding(dreams: Sequence[CachedDream], facts: Sequence[Fact]) -> dict[str, int]:
    """How many dreams of the set bind each fact (sec 3). Coverage is aggregate
    across dreams -- no dream is penalized for wandering off the facts, which
    sec 2.7 calls explicitly fine and plausibly protective."""
    counts = {f.entity: 0 for f in facts}
    for dream in dreams:
        bound, _ = binding_coverage("".join(dream.token_texts), facts)
        for entity, n in bound.items():
            counts[entity] += n > 0
    return counts


def assert_aggregate_binding(dreams: Sequence[CachedDream], facts: Sequence[Fact],
                             min_dreams: int) -> dict[str, int]:
    """The aggregate binding gate. Unlike the single-dream report, which only
    warns, this REFUSES the cache (sec 4): a fact bound in too few dreams is a
    fact no arm can install, and the whole set's numbers would be conditioned
    on it."""
    counts = aggregate_binding(dreams, facts)
    short = {e: n for e, n in counts.items() if n < min_dreams}
    if short:
        raise SystemExit(
            f"aggregate binding gate FAILED: {short} bound in fewer than {min_dreams} of "
            f"{len(dreams)} dreams. A fact the set never binds is one no arm can install -- "
            f"raise --dreams, or revisit the steer prefix, before running any cell."
        )
    return counts


def rebase_dream_set(cache: "DreamSetCache", family: str, rank_rule: str) -> "DreamSetCache":
    """Recompute every dream's erasers under a different gating family.

    The family changes only how the cached queries are WEIGHTED into each
    layer's SVD; the dreams, their queries and their divergences are already
    stored, so nothing is regenerated and every dream hash — and the set hash —
    is unchanged. A re-based cache is the same experiment carrying a different
    eraser.
    """
    from gate_pilot import scheme_weights

    for dream in cache.dreams:
        gate = dream.gate_positions
        if not gate:
            continue
        # The cache stores queries for the GATED positions only, in gate order,
        # while `gate` holds absolute token positions -- so the divergence is
        # read by absolute position and the queries by their own index.
        weights = (None if family == "hard"
                   else scheme_weights([dream.divergence[t] for t in gate], family))
        spectra, ranks, bases = dream_bases(dream.queries, range(len(gate)), cache.wake_state,
                                            rank_rule, weights)
        dream.spectra, dream.ranks, dream.bases = spectra, ranks, bases
    cache.gate_family = family
    cache.rank_rule = rank_rule
    return cache


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


def merge_dream_sets(caches: Sequence["DreamSetCache"]) -> "DreamSetCache":
    """One dream set from several built concurrently with disjoint
    `--dream-seed-offset`.

    The arms share a dream set by registration and the summarizer checks that
    sharing against the set hash, so the pieces have to become one artifact
    rather than a convention. Everything that would make pooling meaningless is
    refused rather than warned about: a different wake transcript is a
    different state, a different generator is a different teacher, and a
    repeated dream (two processes given the same offset) would count once as
    coverage and twice as training.
    """
    if not caches:
        raise SystemExit("merge_dream_sets: nothing to merge")
    first = caches[0]
    seen: dict[str, int] = {}
    dreams: list[CachedDream] = []
    for i, cache in enumerate(caches):
        if cache.transcript_ids != first.transcript_ids:
            raise SystemExit(
                f"merge_dream_sets: cache {i} has a different wake transcript "
                f"({token_sha(cache.transcript_ids)[:12]} vs {first.transcript_sha[:12]}) -- "
                f"its dreams came from a different state and cannot pool.")
        if cache.generator != first.generator:
            raise SystemExit(
                f"merge_dream_sets: cache {i} has generator {cache.generator[:12]}, "
                f"first has {first.generator[:12]} -- a different teacher wrote those dreams.")
        for dream in cache.dreams:
            # A dream that terminated immediately ("[ASSISTANT] <eoc>") is
            # byte-identical across seeds for an innocent reason and carries no
            # content to double-count. A repeated REAL dream means two builds
            # shared a --dream-seed-offset.
            if len(dream.dream_ids) > MIN_DISTINCT_DREAM_TOKENS:
                if dream.dream_sha in seen:
                    raise SystemExit(
                        f"merge_dream_sets: cache {i} repeats a {len(dream.dream_ids)}-token "
                        f"dream already in cache {seen[dream.dream_sha]} "
                        f"(sha {dream.dream_sha[:12]}) -- two builds shared a "
                        f"--dream-seed-offset.")
                seen[dream.dream_sha] = i
            dreams.append(dream)
    merged = copy.copy(first)
    merged.dreams = dreams
    merged.set_sha = dream_set_sha(dreams)
    return merged


def save_dream_cache(cache: DreamCache, path: str | Path) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_dream_cache(path: str | Path) -> DreamCache | DreamSetCache:
    """Load and re-verify, either shape. A cache whose tokens no longer hash to
    what it was saved with is refused outright: every downstream number is
    conditioned on the arms having seen the same tokens."""
    import torch

    # The cache holds a MixerState, not just tensors, so weights_only is off --
    # it is this repo's own artifact, written by the cache builder.
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(cache, DreamSetCache):
        for i, dream in enumerate(cache.dreams):
            if token_sha(dream.dream_ids) != dream.dream_sha:
                raise SystemExit(f"dream set {path} is corrupt: dream {i}'s tokens do not match its sha-256")
        if token_sha(cache.transcript_ids) != cache.transcript_sha or dream_set_sha(cache.dreams) != cache.set_sha:
            raise SystemExit(f"dream set {path} is corrupt: the set no longer matches its recorded sha-256")
        return cache
    # Unpickling restores __dict__ without __init__, so a cache written before
    # the field existed has no attribute to read.
    cache.generator = getattr(cache, "generator", "base")
    cache.stop_reason = getattr(cache, "stop_reason", "max-tokens")
    cache.dream_prompt = getattr(cache, "dream_prompt", "")
    cache.prefix_len = getattr(cache, "prefix_len", 0)
    for name, ids, recorded in (("transcript", cache.transcript_ids, cache.transcript_sha),
                                ("dream", cache.dream_ids, cache.dream_sha)):
        if token_sha(ids) != recorded:
            raise SystemExit(f"dream cache {path} is corrupt: {name} tokens do not match their recorded sha-256")
    return cache


def dream_sidecar_text(cache: DreamCache) -> str:
    """The dream as plain text with the spliced cue spans bracketed, so the
    literal training text is readable without a GPU."""
    parts: list[str] = []
    in_cue = False
    for text, cue in zip(cache.token_texts, cache.cue_flags, strict=True):
        if cue and not in_cue:
            parts.append("[CUE]")
        elif in_cue and not cue:
            parts.append("[/CUE]")
        in_cue = cue
        parts.append(text)
    if in_cue:
        parts.append("[/CUE]")
    return "".join(parts)


def sidecar_path(cache_path: Path) -> Path:
    """Where a cache's decoded sidecar lives: beside it, named after it. The
    default cache (dream_cache_s<seed>.pt) keeps its historical dream_s<seed>.txt,
    so a multi-sleep cache in the same directory gets its own file instead of
    overwriting the single-sleep one."""
    return cache_path.with_name(cache_path.stem.replace("dream_cache", "dream", 1) + ".txt")


def pilot_path(cache_path: Path) -> Path:
    """Where a pilot capture lives: beside its cache (sec 2.10.7)."""
    return cache_path.with_suffix(".pilot.pt")


def write_dream_sidecar(cache: DreamCache, path: str | Path) -> None:
    facts = cache.fact_list
    bound, misbound = binding_coverage("".join(cache.token_texts), facts)
    header = [
        f"seed {cache.seed}   dream_sha {cache.dream_sha}   transcript_sha {cache.transcript_sha}",
        f"generated by: {cache.generator}   ended: {cache.stop_reason}",
        f"steer prefix: {cache.dream_prompt!r} ({cache.prefix_len} tokens, excluded from every "
        f"arm's scored positions)",
        f"{len(cache.dream_ids)} dream tokens, {cache.free_tokens} freely generated, "
        f"{len(cache.dream_ids) - cache.free_tokens} spliced cue text",
        "facts: " + ", ".join(f"{f.entity}={f.code} (foil {cache.distractors[f.entity]})" for f in facts),
        "bound rehearsals: " + ", ".join(f"{e}={n}" for e, n in bound.items()),
        "misbound rehearsals: " + ", ".join(f"{e}={n}" for e, n in misbound.items()),
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(header) + dream_sidecar_text(cache) + "\n")


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
    """Sequences 1 of arms A/B2 (`drain=False`) and B1 (`drain=True`): the
    frozen teacher generates a dream from the wake state, one token at a time
    (the read-query capture is per-token only), caching logits and queries;
    under `drain` it erases along each token's own query as it goes, so the
    next token is produced from the drained state.

    The decoded dream prints live, with the running fact-rehearsal fraction.

    `cues` (with `cue_every`) forces question stems into the dream in rotation,
    every `cue_every` tokens. Free generation rehearses the wake facts only by
    luck -- measured coverage swings from 4/4 codes to 0/4 across seeds at
    every fixed temperature and length -- and a fact the dream never mentions
    is one no arm can install. Each cue ends mid-answer, so the code itself is
    still sampled from the state rather than forced. A fired cue timer waits
    for the next sentence end (up to CUE_DEFER_MAX tokens) before splicing, so
    the cue never cuts a thought in half.

    Termination (sec 2.9.4): the dream ends when the model emits `stop_id`
    (<|eoc|>), with `n_tokens` a hard max. `turn_id` (eos, which ends an
    assistant turn) is counted only for the backstop that bounds a dream where
    `stop_id` never comes. The reason is recorded on the Dream.
    """
    import torch

    state = copy.deepcopy(wake_state)
    ids = [int(i) for i in seed_ids[0].tolist()]
    cue_flags = [False] * len(ids)
    next_cue, cue_at, greedy_left = 0, len(ids) + cue_every, 0
    deferred = -1  # >= 0 once the timer has fired and the splice is waiting
    logits_cache: list[torch.Tensor] = []
    queries: list[list[torch.Tensor]] = []
    texts: list[str] = []
    skipped = 0
    turns = 0
    stop_reason = "max-tokens"
    started = time.time()
    teacher_ctx = frozen_teacher(model) if frozen else contextlib.nullcontext()
    with teacher_ctx, torch.no_grad():
        for t in range(n_tokens):
            token = torch.tensor([[ids[t]]], dtype=torch.long, device=seed_ids.device)
            model.c_capture = []
            logits, state = model(token, state=state)
            per_layer = group_by_layer(model.c_capture, len(model.layers))[0]
            model.c_capture = None

            logits_cache.append(logits[0, -1].float().cpu())
            queries.append([c.cpu() for c in per_layer])
            texts.append(decode_token(ids[t]))
            if drain:
                skipped += erase_state(state, per_layer, op=erase_op)
            if t + 1 >= len(ids):
                if cues and cue_every and deferred < 0 and len(ids) >= cue_at:
                    deferred = 0
                at_boundary = any(stop in texts[-1] for stop in CUE_STOPS)
                if deferred >= 0 and (at_boundary or deferred >= CUE_DEFER_MAX):
                    cue = [int(i) for i in cues[next_cue % len(cues)]]
                    ids.extend(cue)
                    cue_flags.extend([True] * len(cue))
                    next_cue += 1
                    cue_at = len(ids) + cue_every
                    greedy_left = cue_greedy
                    deferred = -1
                else:
                    deferred += deferred >= 0
                    temp = 0.0 if greedy_left > 0 else temperature
                    greedy_left = max(0, greedy_left - 1)
                    sampled = int(sample_next(logits[:, -1], temp).item())
                    ids.append(sampled)
                    cue_flags.append(False)
                    turns += sampled == turn_id
                    if sampled == stop_id:
                        stop_reason = "eoc"
                    elif turn_id is not None and turns >= max_turns:
                        stop_reason = "turn-backstop"

            if (t + 1) % PRINT_EVERY == 0 or t + 1 == n_tokens or stop_reason != "max-tokens":
                frac, _ = rehearsal_fraction(texts, needles)
                rate = (t + 1) / (time.time() - started)
                print(f"[{ts()}]  dream {t + 1}/{n_tokens} rehearsal {frac:.2f} {rate:.1f} tok/s "
                      f"ETA {fmt_duration((n_tokens - t - 1) / rate)} | "
                      f"{''.join(texts[-PRINT_EVERY:])!r}", flush=True)
            if stop_reason != "max-tokens":
                print(f"[{ts()}]  dream ended after {len(texts)} tokens: {stop_reason}", flush=True)
                break
    # The stop token itself is not kept: every consumer relies on
    # len(tokens) == len(logits) == len(queries), and the teacher's prediction
    # of it is already the final logit row.
    kept = len(logits_cache)
    return Dream(
        tokens=torch.tensor([ids[:kept]], dtype=torch.long, device=seed_ids.device),
        logits=torch.stack(logits_cache),
        queries=queries,
        final_state=state,
        token_texts=texts,
        skipped_cone=skipped,
        cue_flags=cue_flags[:kept],
        stop_reason=stop_reason,
        prefix_len=len(seed_ids[0]),
    )


def distill_replay(model, opt, dream: Dream, steps: int, chunk_len: int, kl_temp: float, on_step,
                   fresh_state: bool = False, keep: Sequence[bool] | None = None, ce: bool = False,
                   init_state=None) -> int:
    """Arm A sequence 3: the student is teacher-forced over the dream from a
    FRESH state, in chunks, KL to the cached logits (`ce` swaps that for plain
    cross-entropy on the dream tokens -- the CE-on-dream decomposition cell).
    One optimizer step is one chunk. Returns the number of token positions that
    contributed a gradient, which is the cross-arm budget currency
    `--distill-steps` is not.

    The registered form passes the whole dream as one chunk, so there is
    nothing to carry and every position is practised from a fresh state. At a
    shorter `chunk_len` (the bridge cell) `fresh_state` resets before every
    chunk instead of only at pass boundaries, so no chunk is privileged --
    under the carried schedule only chunk 0 ever follows a reset, and that is
    the only condition the fresh-state probes measure.

    `init_state` is what a reset resets TO: `None` is arm A -- total denial,
    the student re-learns everything state-dependent from blank -- and B4
    passes a fresh copy of the erased wake state instead (sec 2.9.6). Every
    reset takes its own copy, so a multi-epoch pass starts where pass 1 did.
    """
    import torch.nn.functional as F

    length = dream.tokens.shape[1]
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
    tokens = 0
    for step in range(steps):
        c, reset = replay_step(step, n_chunks, fresh_state)
        if reset:
            state = None if init_state is None else copy_state(init_state)
        lo, hi = c * chunk_len, min((c + 1) * chunk_len, length)
        scored = [t for t in range(lo, hi) if (keep is None or keep[t]) and not (ce and t + 1 >= length)]
        if not scored:
            on_step(step, 0.0)  # a fully-masked chunk takes no optimizer step
            continue
        logits, state = model(dream.tokens[:, lo:hi], state=state)
        state = state.detach()
        sel = [t - lo for t in scored]
        if ce:
            targets = dream.tokens[0, [t + 1 for t in scored]]
            loss = F.cross_entropy(logits[0, sel].float(), targets)
        else:
            loss = kl_loss(dream.logits[scored].unsqueeze(0).to(logits.device), logits[:, sel], kl_temp)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += len(scored)
        on_step(step, loss.item())
    return tokens


def erased_start_scaled(wake_state, bases: Sequence[torch.Tensor],
                        spectra: Sequence[Sequence[float]]):
    """`erased_start` with per-direction partial cuts from each layer's own
    spectrum -- the sigma arm."""
    from b4 import erase_subspace_scaled, sigma_gammas

    state = copy_state(wake_state)
    for layer, basis in enumerate(bases):
        if basis.numel() == 0:
            continue
        gammas = sigma_gammas(spectra[layer], basis.shape[0])
        state.ssm_states[layer] = erase_subspace_scaled(
            state.ssm_states[layer], basis, gammas)
    return state


def erased_start(wake_state, bases: Sequence[torch.Tensor]):
    """A fresh copy of the wake state with one dream's eraser applied ONCE.

    Sec 2.10.4 pins this: never erase-once-then-carry. A carried state would
    ferry dream k's re-written facts into dream k+1, and the projection is
    idempotent precisely so that re-deriving it per dream costs nothing.
    """
    state = copy_state(wake_state)
    erase_state_subspace(state, bases)
    return state


def dream_from_cached(cached: CachedDream, device) -> Dream:
    import torch

    return Dream(
        tokens=torch.tensor([cached.dream_ids], dtype=torch.long, device=device),
        logits=cached.teacher_logits,
        queries=cached.queries,
        final_state=None,
        token_texts=cached.token_texts,
        skipped_cone=0,
        cue_flags=cached.cue_flags,
        stop_reason=cached.stop_reason,
    )


def distill_dream_set(model, opt, dreams: Sequence[CachedDream], wake_state, variant: str | None,
                      epochs: int, kl_temp: float, on_step, on_boundary,
                      sigma_scaled: bool = False) -> int:
    """Sec 2.10.4's carry matrix over the whole dream set.

    Student WEIGHTS carry across the set -- one optimizer trajectory, since
    resetting them per dream would leave only dream N's learning. Student STATE
    resets at every dream boundary: `variant=None` is arm A, from blank;
    a B4 variant starts each dream from a fresh erased copy of the wake state.

    Each dream is ORDINARY sequence training against its cached teacher logits
    -- the chunk is the whole dream, full BPTT through the model's own sequence
    path, no spine and no per-token loop.

    `epochs` counts passes over the SET, not over each dream: dream 1..N, then
    1..N again. That is the replay-literature shape sec 2.10.2's variant cell
    exists to price; massing a dream's repeats back to back would rebuild the
    hundreds-of-passes-on-one-dream regime that same section retires. One
    epoch is the registered form.
    """
    tokens = step = 0
    for epoch in range(epochs):
        for i, cached in enumerate(dreams):
            dream = dream_from_cached(cached, wake_state.ssm_states[0].device)
            if variant is None:
                start = None
            elif sigma_scaled:
                start = erased_start_scaled(wake_state, cached.bases[variant], cached.spectra)
            else:
                start = erased_start(wake_state, cached.bases[variant])
            tokens += distill_replay(
                model, opt, dream, steps=1, chunk_len=dream.tokens.shape[1], kl_temp=kl_temp,
                on_step=lambda s, loss, base=step: on_step(base + s, loss),
                keep=scored_keep(cached.cue_flags, cached.prefix_len), init_state=start)
            step += 1
            on_boundary(i, epoch, step)
    return tokens


def distill_counterfactual(model, opt, dream: Dream, wake_state, steps: int, kl_temp: float, accum: int,
                           in_place: bool, deep: bool, on_step, keep: Sequence[bool] | None = None,
                           erase_op: str = ERASE_OP):
    """The corrected B arms (sec 3), both teacher-forced over the cached dream
    from a copy of the cached wake state. Per token, per layer, inside the
    forward: compute C, deflate it against that layer's state, ablate the
    carried past, then decay+write+read -- so the logit trained on is always
    produced from the ablated version of the state that generated that token.

    B1 (`in_place`) ablates the carried state and carries the student's own
    detached forward; B2 ablates a copy, trains on it, discards it, and carries
    the intact state (advanced by the same student forward with the hook off).
    They are identical at token 1 -- same state, query, ablation, logit and
    gradient -- and diverge from token 2 purely through the carry.

    `deep` accumulates the pass's losses and takes one optimizer step with the
    graph intact (full BPTT through the spine). B2 only: B1's cross-token
    gradient runs through every intervening ablation projector, which aims it
    at the deflation-protected subspace (sec 6).

    Returns (token gradients, the state this arm carries).
    """
    import torch

    if deep and in_place:
        raise ValueError("deep gradients are measured on B2 only -- deep-B1 is structurally confounded (sec 6)")

    hook, skipped = make_erase_hook(erase_op)

    length = dream.tokens.shape[1]
    step = tokens = 0
    state = None
    while step < steps:
        state = copy_state(wake_state)
        losses: list[torch.Tensor] = []
        for t in range(length):
            if step >= steps:
                break
            token = dream.tokens[:, t : t + 1]
            masked = keep is not None and not keep[t]
            target_state = state if in_place else copy_state(state)

            model.erase_hook = hook
            try:
                if masked:
                    # Still consumed and still carried; just not learned from.
                    with torch.no_grad():
                        _, advanced = model(token, state=target_state)
                else:
                    logits, advanced = model(token, state=target_state)
            finally:
                model.erase_hook = None

            if not masked:
                loss = kl_loss(dream.logits[t].view(1, 1, -1).to(logits.device), logits, kl_temp)
                if deep:
                    losses.append(loss)
                else:
                    (loss / accum).backward()
                    if (step + 1) % accum == 0:
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    on_step(step, loss.item())
                tokens += 1
                step += 1

            if in_place:
                state = advanced if deep else advanced.detach()
            elif deep:
                _, state = model(token, state=state)
            else:
                with torch.no_grad():
                    _, state = model(token, state=state)
                state = state.detach()
        if deep and losses:
            total = torch.stack(losses).sum()
            total.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            on_step(step - 1, total.item() / len(losses))
            state = state.detach()
    print(f"\n[{ts()}]  near-cone erases skipped: {skipped()} of "
          f"{tokens * max(1, len(getattr(model, 'layers', [1])))}")
    return tokens, state


def _state_layers(state, attr: str) -> list:
    return list(getattr(state, attr))


def _state_attrs(state) -> tuple[str, ...]:
    return tuple(a for a in ("conv_states", "ssm_states") if hasattr(state, a))


def spine_states(model, tokens: torch.Tensor, wake_state, block: int) -> dict[str, list[torch.Tensor]]:
    """The intact dream spine, materialized: per layer, the state carried INTO
    each of the T tokens, stacked as a (T, ...) tensor.

    Two phases, so the sequential depth is ~T/block + block rather than T: the
    block boundaries come from whole-block forwards -- the fused SSD chunk-scan
    wherever that kernel is usable, the per-token loop otherwise -- and then
    every block steps through its own tokens simultaneously as one batch. The
    trajectory is the plain serial one either way (tests/test_dream_sleep.py).

    Gradient follows the caller's grad mode: B2-fused-detached materializes the
    spine under no_grad, B2-fused-deep back-propagates through both phases.
    """
    import torch

    attrs = _state_attrs(wake_state)
    length = tokens.shape[1]
    block = max(1, min(block, length))
    n_blocks = (length + block - 1) // block
    padded = tokens
    if n_blocks * block > length:
        padded = torch.cat([tokens, tokens.new_zeros(1, n_blocks * block - length)], dim=1)

    # Cloned, unlike the phase-2 snapshots below: a fused chunk-scan kernel is
    # free to write its output over the states it was handed.
    starts = []
    state = copy_state(wake_state)
    for b in range(n_blocks):
        starts.append({a: [t.clone() for t in _state_layers(state, a)] for a in attrs})
        _, state = model(padded[:, b * block : (b + 1) * block], state=state)

    batched = copy.copy(wake_state)
    for attr in attrs:
        setattr(batched, attr, [torch.cat([s[attr][i] for s in starts], dim=0)
                                for i in range(len(starts[0][attr]))])

    block_tokens = padded.view(n_blocks, block)
    # Written in place, one snapshot at a time: collecting the T snapshots and
    # then stacking them holds the whole spine twice at once, and one copy is
    # already n_layers * T * state_bytes -- ~37 GB for a 512-token dream on the
    # 780m's 48 layers, so the transient second copy alone overflows a 94 GB
    # card before a single counterfactual is forwarded.
    spine: dict[str, list[torch.Tensor]] = {
        attr: [t.new_empty((n_blocks, block, *t.shape[1:])) for t in _state_layers(batched, attr)]
        for attr in attrs
    }
    for j in range(block):
        for attr in attrs:
            for i, t in enumerate(_state_layers(batched, attr)):
                spine[attr][i][:, j] = t
        _, batched = model(block_tokens[:, j : j + 1], state=batched)

    for attr in attrs:
        spine[attr] = [t.reshape(-1, *t.shape[2:])[:length] for t in spine[attr]]
    return spine


def fused_pass(model, dream: Dream, wake_state, spine: dict[str, list[torch.Tensor]], scored: Sequence[int],
               kl_temp: float, erase_op: str, cf_batch: int, backward: bool, retain: bool = False):
    """One counterfactual pass over the whole dream at this pass's weights:
    every scored position takes its own copy of the spine's per-layer states,
    ablates them along the student's own query for that token (per layer,
    interleaved inside the forward), writes the token, reads, and is scored
    against the cached teacher logits. Nothing carries between positions --
    B2's defining property, and what makes the pass batchable.

    Micro-batches of `cf_batch` positions accumulate into one gradient; the
    returned loss is the sum over positions, so the micro-batch size is a
    memory knob and not a hyperparameter. Returns (summed KL, skipped)."""
    import torch

    hook, skipped = make_erase_hook(erase_op)
    total = 0.0
    device = dream.tokens.device
    for lo in range(0, len(scored), cf_batch):
        sel = list(scored[lo : lo + cf_batch])
        idx = torch.tensor(sel, dtype=torch.long, device=device)
        cf = copy.copy(wake_state)
        for attr, layers in spine.items():
            setattr(cf, attr, [t[idx] for t in layers])

        model.erase_hook = hook
        try:
            logits, _ = model(dream.tokens[0, idx].view(-1, 1), state=cf)
        finally:
            model.erase_hook = None

        teacher = dream.logits[sel].unsqueeze(1).to(logits.device)
        loss = kl_loss(teacher, logits, kl_temp) * len(sel)
        if backward:
            loss.backward(retain_graph=retain)
        total += float(loss.detach())
    return total, skipped()


def distill_fused(model, opt, dream: Dream, wake_state, steps: int, kl_temp: float, on_step,
                  keep: Sequence[bool] | None = None, erase_op: str = ERASE_OP, deep: bool = False,
                  block: int = SPINE_BLOCK, cf_batch: int = CF_BATCH,
                  frozen_spine: dict[str, list[torch.Tensor]] | None = None, check=None) -> int:
    """The fused B arms (sec 3.5). One optimizer step is one full-dream pass --
    arm A's currency.

    B2-fused-detached recomputes the intact spine each pass under the current
    weights and detaches it; B2-fused-deep (`deep`) keeps its graph and
    back-propagates through the scan; B3-fused passes `frozen_spine`, the dream
    generator's own trajectory, constant across passes.

    At pass 1 the student's weights are still the generator snapshot, so a
    B3-fused pass must reproduce B2-fused-detached's exactly; the arm checks
    that itself, every sleep, and reports it through `check`.
    """
    import torch

    length = dream.tokens.shape[1]
    scored = [t for t in range(length) if keep is None or keep[t]]
    skipped = tokens = 0
    for step in range(steps):
        if frozen_spine is not None:
            spine = frozen_spine
        elif deep:
            spine = spine_states(model, dream.tokens, wake_state, block)
        else:
            with torch.no_grad():
                spine = spine_states(model, dream.tokens, wake_state, block)

        loss, cut = fused_pass(model, dream, wake_state, spine, scored, kl_temp, erase_op, cf_batch,
                               backward=True, retain=deep)
        skipped += cut

        # Still at the generator snapshot's weights: this is the only moment
        # the two spines are comparable, so the check runs before the step.
        if step == 0 and frozen_spine is not None and check is not None:
            # The live spine is a second full copy -- ~37 GB at 512 tokens on
            # the 780m's 48 layers -- so the generator's spine parks on the
            # host for the length of the comparison instead of the two
            # coexisting on-card. B3's own loss is already computed above.
            device = frozen_spine["ssm_states"][0].device
            parked = {a: [t.to("cpu") for t in layers] for a, layers in frozen_spine.items()}
            for layers in frozen_spine.values():
                layers.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                live = spine_states(model, dream.tokens, wake_state, block)
                reference, _ = fused_pass(model, dream, wake_state, live, scored, kl_temp, erase_op,
                                          cf_batch, backward=False)
            del live
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for attr, layers in parked.items():
                frozen_spine[attr].extend(t.to(device) for t in layers)
            equivalent = abs(loss - reference) <= 1e-4 * max(1.0, abs(reference))
            check({"pass": 1, "b3_loss": loss, "b2_loss": reference,
                   "abs_diff": abs(loss - reference), "equivalent": equivalent})
            print(f"[{ts()}]  pass-1 equivalence B3-fused vs B2-fused-detached: "
                  f"{'OK' if equivalent else 'FAILED'} ({loss:.6f} vs {reference:.6f})", flush=True)

        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += len(scored)
        on_step(step, loss / max(1, len(scored)))
    print(f"\n[{ts()}]  near-cone erases skipped: {skipped} of "
          f"{tokens * max(1, len(getattr(model, 'layers', [1])))}")
    return tokens


def distill_live(
    model, opt, wake_state, seed_ids: torch.Tensor, n_tokens: int, temperature: float,
    kl_temp: float, accum: int, decode_token, needles: Sequence[str], on_step,
    erase_op: str = ERASE_OP,
) -> tuple[object, Dream]:
    """B1-live sequence 1: one online adapters-on pass. Per token -- forward
    from the current state and store the logits, erase that state along the
    token's own read query, take a gradient step on the post-erase forward
    against the stored logits, sample the next token from the *stored* logits,
    and continue from the drained state the post-erase forward produced.

    Exploratory (1 seed, not part of the frontier): the observable is whether
    the decoded dream stays coherent as the handoff to the weights proceeds.
    """
    import torch

    state = copy.deepcopy(wake_state)
    ids = [int(i) for i in seed_ids[0].tolist()]
    texts: list[str] = []
    logits_cache: list[torch.Tensor] = []
    skipped = 0
    started = time.time()
    for t in range(n_tokens):
        token = torch.tensor([[ids[t]]], dtype=torch.long, device=seed_ids.device)

        model.c_capture = []
        with torch.no_grad():
            stored, _ = model(token, state=copy.deepcopy(state))
        per_layer = group_by_layer(model.c_capture, len(model.layers))[0]
        model.c_capture = None

        skipped += erase_state(state, per_layer, op=erase_op)
        logits, state = model(token, state=state)
        loss = kl_loss(stored.detach(), logits, kl_temp) / accum
        loss.backward()
        if (t + 1) % accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        state = state.detach()
        on_step(t, loss.item() * accum)

        logits_cache.append(stored[0, -1].float().cpu())
        texts.append(decode_token(ids[t]))
        if t + 1 >= len(ids):
            ids.append(int(sample_next(stored[:, -1], temperature).item()))
        if (t + 1) % PRINT_EVERY == 0 or t + 1 == n_tokens:
            frac, _ = rehearsal_fraction(texts, needles)
            rate = (t + 1) / (time.time() - started)
            print(f"[{ts()}]  live dream {t + 1}/{n_tokens} rehearsal {frac:.2f} {rate:.1f} tok/s "
                  f"ETA {fmt_duration((n_tokens - t - 1) / rate)} | "
                  f"{''.join(texts[-PRINT_EVERY:])!r}", flush=True)
    dream = Dream(
        tokens=torch.tensor([ids[:n_tokens]], dtype=torch.long, device=seed_ids.device),
        logits=torch.stack(logits_cache),
        queries=[],
        final_state=state,
        token_texts=texts,
        skipped_cone=skipped,
    )
    return state, dream


def distill_sft(model, opt, ids: torch.Tensor, steps: int, chunk_len: int, on_step) -> int:
    """--sft-ref: plain next-token cross-entropy on the stored wake transcript
    verbatim, from a fresh state. The CE-on-raw-text convention the dream arms
    are being compared against -- the objective-type confound lives here and
    only here."""
    import torch.nn.functional as F

    length = ids.shape[1] - 1
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
    tokens = 0
    for step in range(steps):
        c = step % n_chunks
        if c == 0:
            state = None
        lo, hi = c * chunk_len, min((c + 1) * chunk_len, length)
        logits, state = model(ids[:, lo:hi], state=state)
        state = state.detach()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), ids[0, lo + 1 : hi + 1])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += hi - lo
        on_step(step, loss.item())
    return tokens


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


def default_cache_path(seed: int, dreams: int) -> Path:
    """Where this seed's cache lives. A multi-dream build writes the set path,
    and an arm cell picks a set up automatically once one exists for the seed
    -- which cache was chosen is printed, so the preference is never silent."""
    dream_set = Path(f"data/dream_set_s{seed}.pt")
    return dream_set if dreams or dream_set.exists() else Path(f"data/dream_cache_s{seed}.pt")


def state_to(state, device):
    """Move a MixerState's tensors onto `device` (the cache is written and read
    on cpu, so it survives a change of box)."""
    for attr in ("conv_states", "ssm_states"):
        if hasattr(state, attr):
            setattr(state, attr, [t.to(device) for t in getattr(state, attr)])
    return state


def build_cues(seen, encode, user_open: str, asst_open: str) -> list[list[int]]:
    """Each cue is the wake session's question plus the answer stem, stopping
    before the code -- so the cue names which fact to recall and the state still
    has to supply the digits."""
    return [encode(f"{USER_CUE.format(user=user_open, entity=f.entity)}"
                   f"{asst_open} The code for the {f.entity} is")[0].tolist()
            for _, f in seen]


def report_dream(cache: DreamCache) -> None:
    """The artifact, not just the counts (root CLAUDE.md): binding-aware
    coverage, and the decoded text either side of one cue joint."""
    facts = cache.fact_list
    text = "".join(cache.token_texts)
    bound, misbound = binding_coverage(text, facts)
    print(f"[{ts()}] dream ended: {cache.stop_reason}  ({len(cache.dream_ids)} tokens)")
    print(f"[{ts()}] bound rehearsals {bound}  (misbound: {misbound})")
    print(f"[{ts()}] bound code coverage {sum(v > 0 for v in bound.values())}/{len(facts)}, "
          f"{cache.free_tokens}/{len(cache.dream_ids)} tokens freely generated")
    joint = next((i for i in range(1, len(cache.cue_flags)) if cache.cue_flags[i] and not cache.cue_flags[i - 1]), None)
    if joint is None:
        print(f"[{ts()}] no cue joint in this dream (--cue-every 0?)")
    else:
        lo, hi = max(0, joint - 24), min(len(cache.token_texts), joint + 40)
        print(f"[{ts()}] decoded around cue joint at token {joint}:\n"
              f"  ...{''.join(cache.token_texts[lo:joint])!r} >>CUE>> {''.join(cache.token_texts[joint:hi])!r}...")
    print(f"[{ts()}] decoded dream:\n{text!r}")
    if sum(v > 0 for v in bound.values()) < len(facts):
        print(f"[{ts()}] WARNING: a fact this dream never binds is one no arm can install. "
              f"Regenerate this seed at a tighter --cue-every before running the grid.")


def validate_wave_args(args) -> None:
    """Multi-sleep needs a teacher for waves after the first, and which one is a
    protocol choice: `current` is the registered protocol (sec 3.2 -- the
    student as of that sleep's start, generating from its own carried state),
    `base` is the drift-contribution control, a teacher pinned at the base
    across every sleep. The two measure different things, so the harness
    refuses to pick."""
    if args.waves > 1 and not getattr(args, "wave_teacher", None):
        raise SystemExit(
            "--waves > 1 needs --wave-teacher base|current: the registered protocol is `current` "
            "(sec 3.2), `base` is the one-seed drift-contribution control. Register the choice "
            "before running multi-sleep."
        )


def generate_wave_dream(model, args, carried, facts, encode, decode, tokenizer,
                        user_open, asst_open, teacher: str, stop_id: int | None) -> Dream:
    """The dream a wave after the first distils: generated from that wave's own
    carried state, cued on that wave's facts.

    Wave 1 loads the seed's cached dream, shared byte-identically across arms.
    A later wave cannot: each arm reaches it with a different carried state (A
    cleared, B1 selectively vacated, B2 intact), which is the whole object of
    the multi-sleep contrast, so these dreams legitimately differ per arm and
    are recorded rather than asserted equal.
    """
    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    needles = [f.entity for f in facts] + [f.code for f in facts]
    cues = build_cues([(0, f) for f in facts], encode, user_open, asst_open) if args.cue_every else []

    model.eval()
    return teacher_dream(
        model, carried, seed_ids, args.dream_tokens, args.dream_temp,
        drain=False, decode_token=lambda i: decode([i]), needles=needles,
        cues=cues, cue_every=args.cue_every, cue_greedy=args.cue_greedy,
        frozen=(teacher == "base"), stop_id=stop_id, turn_id=tokenizer.eos_token_id)


def r_matrix_row(scored: dict[str, dict[str, object]]) -> dict[int, dict[str, float]]:
    """One probe sweep, split by the wave that taught each fact -- column j of
    the R-matrix (sec 3.7). Facts with no foil carry no margin and are left
    out rather than averaged in as zero."""
    rows: dict[int, dict[str, float]] = {}
    for record in scored.values():
        if record.get("margin") is None:
            continue
        row = rows.setdefault(int(record["fact_wave"]), {"mean_margin": 0.0, "installs": 0, "facts": 0})
        row["mean_margin"] += float(record["margin"])
        row["installs"] += int(bool(record["install"]))
        row["facts"] += 1
    for row in rows.values():
        row["mean_margin"] /= row["facts"]
    return rows


def r_matrix_rows(r: dict[tuple[int, int], dict[str, float]], waves: int) -> list[list[float | None]]:
    """The R-matrix as `waves` rows of `waves` mean margins, `None` where a
    wave's facts did not exist yet (the empty upper triangle)."""
    return [[r[(i, j)]["mean_margin"] if (i, j) in r else None for j in range(1, waves + 1)]
            for i in range(1, waves + 1)]


def cl_summary(r: dict[tuple[int, int], dict[str, float]], waves: int) -> dict[str, object]:
    """The continual-learning readout over a finished multi-sleep run (sec 3.7).

    Backward transfer is the mean, over the waves taught before the last sleep,
    of how far their margin moved between the sleep that taught them and the
    final one -- negative is forgetting. Cumulative installation is reported as
    what survives the final sleep beside the peak any sleep reached: a fact
    installed at sleep 1 and destroyed at sleep 2 shows up as the gap.
    """
    deltas = [r[(i, waves)]["mean_margin"] - r[(i, i)]["mean_margin"]
              for i in range(1, waves) if (i, waves) in r and (i, i) in r]
    peak = sum(max(v["installs"] for (wave, _), v in r.items() if wave == i)
               for i in sorted({i for i, _ in r}))
    return {"bwt": sum(deltas) / len(deltas) if deltas else None,
            "installs_final": sum(v["installs"] for (_, j), v in r.items() if j == waves),
            "installs_peak": peak, "waves": waves}


def commit_erase(model, carried, seen, committed: set[str], encode, user_open: str, asst_open: str,
                 verify, erase_op: str, emit, wave: int) -> list[str]:
    """B2's commit step -- arm B2' (08-06 sec 3, "install-then-erase"). At the
    end of a sleep, per fact: a fresh-state margin check, and only where it
    passes does one real erase along that fact's own query land on the carried
    state. The erase is a verified memory policy here, never a training signal;
    its observable (freed capacity vs A's clearing-loss) only exists across
    sleeps.

    A fact commits once. Re-cutting along the same query at every later sleep
    would charge the bystanders again to delete what is already gone.
    """
    import torch

    fired: list[str] = []
    for _, fact in seen:
        if fact.entity in committed:
            continue
        margin, installed, *_ = verify(fact)
        record = {"phase": "commit", "wave": wave, "fact": fact.entity, "margin": margin,
                  "committed": bool(installed)}
        if installed:
            prompt = encode(f"{USER_CUE.format(user=user_open, entity=fact.entity)}"
                            f"{asst_open} The code for the {fact.entity} is")
            model.c_capture = []
            with torch.no_grad():
                model(prompt, state=copy_state(carried))
            queries = group_by_layer(model.c_capture, len(model.layers))[-1]
            model.c_capture = None
            record["skipped_cone"] = erase_state(carried, queries, op=erase_op)
            committed.add(fact.entity)
            fired.append(fact.entity)
        emit(record)
        print(f"[{ts()}]  commit {fact.entity:<11} margin {margin:+7.3f} "
              f"{'ERASED' if installed else 'held (not installed)'}", flush=True)
    return fired


def build_cache(model, args, cache_path: Path, transcript, facts, chunk_len,
                encode, decode, tokenizer, user_open, asst_open, stop_id: int | None,
                adapter_sha: str | None = None) -> DreamCache:
    """Generate this seed's one teacher dream and persist it with its hashes,
    its distractor codes and a decoded sidecar. From a copy of the wake state,
    intact -- no arm ever regenerates (sec 3's shared preamble).

    The generator is the student as of this sleep's start (sec 3.2): the
    warm-start checkpoint when `--init-adapter` loaded one, the base otherwise
    (a fresh student *is* the base -- zero-init lora_B, zero marker delta). The
    zeroing path is for arm semantics that pin a teacher during training, not
    for un-warm-starting the generator, so it fires only without an adapter."""
    import torch

    print(f"\n[{ts()}] === building dream cache for seed {args.seed} -> {cache_path} ===")
    wake_state = run_chunks(model, transcript, None, chunk_len, "wake", keep_logits=False)[1]

    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    needles = [f.entity for f in facts] + [f.code for f in facts]
    seen = [(1, f) for f in facts]
    cues = build_cues(seen, encode, user_open, asst_open) if args.cue_every else []

    model.eval()
    dream = teacher_dream(model, wake_state, seed_ids, args.dream_tokens, args.dream_temp,
                          drain=False, decode_token=lambda i: decode([i]), needles=needles,
                          cues=cues, cue_every=args.cue_every, cue_greedy=args.cue_greedy,
                          frozen=adapter_sha is None, stop_id=stop_id,
                          turn_id=tokenizer.eos_token_id)

    cache = DreamCache(
        seed=args.seed,
        transcript_ids=[int(i) for i in transcript[0].tolist()],
        dream_ids=[int(i) for i in dream.tokens[0].tolist()],
        wake_state=state_to(wake_state, torch.device("cpu")),
        teacher_logits=dream.logits.cpu(),
        queries=[[c.cpu() for c in per_layer] for per_layer in dream.queries],
        token_texts=dream.token_texts,
        cue_flags=dream.cue_flags,
        distractors=build_distractors(facts, args.seed),
        facts=[(f.entity, f.category, f.code) for f in facts],
        stop_reason=dream.stop_reason,
        dream_prompt=args.dream_prompt,
        prefix_len=seed_ids.shape[1],
        generator=adapter_sha or "base",
    )
    save_dream_cache(cache, cache_path)
    sidecar = sidecar_path(cache_path)
    write_dream_sidecar(cache, sidecar)
    print(f"[{ts()}] wrote {cache_path} and {sidecar}")
    print(f"[{ts()}] transcript_sha {cache.transcript_sha}\n[{ts()}] dream_sha      {cache.dream_sha}")
    report_dream(cache)
    return cache


def dream_is_degenerate(text: str) -> bool:
    """Sec 2.1's mojibake clause, applied per dream: any replacement character
    or a non-ASCII flood. Ordinary unicode punctuation (curly quotes, dashes)
    stays acceptable. Content-free -- no fact knowledge (sec 2.9.1)."""
    if not text or "�" in text:
        return True
    return sum(ord(c) > 127 for c in text) / len(text) > 0.2


def blank_state_logits(model, tokens, chunk_len: int, frozen: bool):
    """Re-score a dream's own tokens under a BLANK state at the same weights --
    the gate's denominator (sec 2.9.2). A memory read is a position where the
    state changed the prediction, which is defined by the state and not by any
    fact list, so generic reads drop out on their own."""
    with (frozen_teacher(model) if frozen else contextlib.nullcontext()):
        logits, _ = run_chunks(model, tokens, None, chunk_len, "blank re-score", keep_logits=True)
    return logits[0].float()


def dream_bases(queries: Sequence[Sequence[torch.Tensor]], gate: Sequence[int], wake_state,
                rank_rule: str, weights: Sequence[float] | None = None):
    """Per layer: ONE SVD over this dream's gated queries, both rank rules
    computed inside the prod-side address budget, and all three variants
    post-processed from that one shared basis (sec 2.7).

    Returns (spectra, ranks, bases): every layer's spectrum and both rules'
    answers go to the sidecar, so a spectrum with no clean structure is visible
    as a stop-and-think finding rather than a silent truncation.
    """
    import torch

    n_layers = len(wake_state.ssm_states)
    spectra: list[list[float]] = []
    ranks: dict[str, list[int]] = {rule: [] for rule in RANK_RULES}
    bases: dict[str, list[torch.Tensor]] = {variant: [] for variant in VARIANTS}
    for layer in range(n_layers):
        # The gate selects B4's queries; it is not a dream-validity
        # requirement. No gated queries -> an empty eraser: the dream read
        # nothing from the state, so there is nothing to deny.
        if not gate:
            d_state = wake_state.ssm_states[layer].shape[-1]
            spectra.append([])
            for rule in RANK_RULES:
                ranks[rule].append(0)
            for variant in VARIANTS:
                bases[variant].append(torch.zeros(0, d_state))
            continue
        v_full, sigma = aggregate_basis([queries[t][layer] for t in gate], weights)
        budget = address_budget(v_full.shape[1])
        chosen = {"ratio-gap": rank_ratio_gap(sigma, budget), "median": rank_median(sigma, budget)}
        for rule, r in chosen.items():
            ranks[rule].append(r)
        spectra.append([float(x) for x in sigma])
        for variant in VARIANTS:
            basis = variant_basis(v_full, chosen[rank_rule], variant, wake_state.ssm_states[layer])
            if basis.shape[0] == 0:
                print(f"[{ts()}]  NOTE: layer {layer}'s {variant} basis is empty at rank "
                      f"{chosen[rank_rule]} over {len(gate)} gated queries -- this dream's "
                      f"{variant} eraser removes nothing at this layer.")
            identity = basis @ basis.T
            if basis.shape[0] and not bool((identity - torch.eye(basis.shape[0])).abs().max() < 1e-4):
                raise SystemExit(f"layer {layer}'s {variant} basis is not orthonormal (sec 2.7 asserts V^T V = I)")
            bases[variant].append(basis.cpu())
    return spectra, ranks, bases


def basis_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of the smaller subspace two bases share -- the mean squared
    principal cosine. Sec 2.10.3 prices per-dream erasers' noise with this
    rather than arguing about it."""
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.float() @ b.float().T).pow(2).sum() / min(a.shape[0], b.shape[0]))


def report_dream_set(cache: DreamSetCache, min_dreams: int, rank_rule: str) -> dict[str, object]:
    """The artifact, not just the counts (root CLAUDE.md), for a dream set:
    termination reasons, per-dream basis sizes, cross-dream V-overlap, the
    gate's agreement with the binding scan, per-fact within-dream repeat counts
    (B4's re-installation window, sec 2.7), and a decoded sample of one dream's
    start. The aggregate binding gate fires at the end -- it REFUSES the cache
    rather than warning (sec 4)."""
    facts = cache.fact_list
    print(f"\n[{ts()}] === dream set: {len(cache.dreams)} dreams, set_sha {cache.set_sha[:12]} ===")
    reasons = {reason: sum(d.stop_reason == reason for d in cache.dreams) for reason in STOP_REASONS}
    print(f"[{ts()}] termination reasons: {reasons}")
    gateless = sum(not d.gate_positions for d in cache.dreams)
    if gateless:
        print(f"[{ts()}] dreams with an empty gate (empty eraser, no denial pressure): "
              f"{gateless} of {len(cache.dreams)}")
    for i, dream in enumerate(cache.dreams):
        bound, misbound = binding_coverage("".join(dream.token_texts), facts)
        agreement = gate_agreement(dream.gate_positions, fact_read_positions(dream.token_texts, facts))
        sizes = [len(b) for b in dream.bases[VARIANTS[0]]]
        print(f"[{ts()}]  dream {i}: {len(dream.dream_ids)} tokens, ended {dream.stop_reason}, "
              f"{len(dream.gate_positions)} gated positions, basis rank "
              f"{min(sizes)}-{max(sizes)} over {len(sizes)} layers ({rank_rule})")
        copied = copy_fraction(dream.dream_ids[dream.prefix_len :], cache.transcript_ids,
                               cue_flags=dream.cue_flags[dream.prefix_len :])
        print(f"[{ts()}]    within-dream repeats {bound}  (misbound {misbound})  "
              f"verbatim-copied from the wake transcript: {copied:.1%}")
        print(f"[{ts()}]    gate vs binding scan: precision {agreement['precision']:.2f} "
              f"recall {agreement['recall']:.2f}, per-fact contribution {agreement['per_fact']}")
        for entity, n in agreement["per_fact"].items():
            if n == 0:
                print(f"[{ts()}]    NOTE: the gate captured no read of {entity} in this dream -- "
                      f"the eraser cannot address what it never captured.")
    copies = [copy_fraction(d.dream_ids[d.prefix_len :], cache.transcript_ids,
                            cue_flags=d.cue_flags[d.prefix_len :]) for d in cache.dreams]
    longest = [longest_verbatim_run(d.dream_ids[d.prefix_len :], cache.transcript_ids)
               for d in cache.dreams]
    print(f"[{ts()}] longest verbatim run per dream: max {max(longest)} tokens "
          f"(a run of hundreds is the transcript being REPLAYED; ~15 is a reused sentence)")
    print(f"[{ts()}] verbatim copying of the wake transcript: mean {sum(copies) / len(copies):.1%}, "
          f"max {max(copies):.1%}  (a dream that replays the wake is not a dream -- watch this "
          f"when the warm start is recall-heavy)")
    for variant in VARIANTS:
        overlaps = [basis_overlap(a.bases[variant][i], b.bases[variant][i])
                    for a, b in zip(cache.dreams, cache.dreams[1:], strict=False)
                    for i in range(len(a.bases[variant]))]
        if overlaps:
            print(f"[{ts()}] cross-dream V-overlap ({variant}): mean {sum(overlaps) / len(overlaps):.3f}, "
                  f"max {max(overlaps):.3f}")
    first = cache.dreams[0]
    print(f"[{ts()}] decoded start of dream 0 (prefix + first free tokens):\n"
          f"  >>PREFIX>> {''.join(first.token_texts[:first.prefix_len])!r} "
          f">>FREE>> {''.join(first.token_texts[first.prefix_len:first.prefix_len + 48])!r}")
    counts = assert_aggregate_binding(cache.dreams, facts, min_dreams)
    print(f"[{ts()}] aggregate binding gate PASSED (>= {min_dreams} dreams per fact): {counts}")
    return {"stop_reasons": reasons, "binding": counts}


def write_dream_set_sidecar(cache: DreamSetCache, path: str | Path) -> None:
    facts = cache.fact_list
    lines = [
        f"seed {cache.seed}   set_sha {cache.set_sha}   transcript_sha {cache.transcript_sha}",
        f"generated by: {cache.generator}   {len(cache.dreams)} dreams",
        f"steer prefix: {cache.dream_prompt!r}   gate threshold {cache.gate_threshold} nats "
        f"   rank rule {cache.rank_rule}",
        "facts: " + ", ".join(f"{f.entity}={f.code} (foil {cache.distractors[f.entity]})" for f in facts),
        "dreams bound per fact: " + ", ".join(f"{e}={n}" for e, n in aggregate_binding(cache.dreams, facts).items()),
        "",
    ]
    for i, dream in enumerate(cache.dreams):
        bound, misbound = binding_coverage("".join(dream.token_texts), facts)
        agreement = gate_agreement(dream.gate_positions, fact_read_positions(dream.token_texts, facts))
        lines += [
            f"--- dream {i}  sha {dream.dream_sha[:12]}  {len(dream.dream_ids)} tokens  "
            f"ended {dream.stop_reason} ---",
            f"within-dream repeats: {bound}   misbound: {misbound}   "
            f"verbatim-copied: {copy_fraction(dream.dream_ids[dream.prefix_len :], cache.transcript_ids, cue_flags=dream.cue_flags[dream.prefix_len :]):.1%}",
            f"gate: {len(dream.gate_positions)} positions, precision {agreement['precision']:.3f} "
            f"recall {agreement['recall']:.3f}, per-fact contribution {agreement['per_fact']}",
            "per-layer rank: " + "  ".join(f"{rule}={dream.ranks[rule]}" for rule in RANK_RULES),
            "per-layer spectra: " + "; ".join(
                " ".join(f"{s:.3g}" for s in spectrum) for spectrum in dream.spectra),
            "".join(dream.token_texts),
            "",
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def battery_read_queries(model, items, wake_state, encode, n_layers: int):
    """Each battery prompt's per-position, per-layer read queries, run from a
    copy of the wake state (sec 2.10.7).

    This is the on-GPU half of sec 2.10.6's collateral pool -- the offline
    scorer reads these, it never runs a model. One token at a time, because the
    chunked path issues no per-token capture.
    """
    import torch

    out: dict[str, list[list[torch.Tensor]]] = {}
    # The caller may hand over a state state_to already moved to CPU for the
    # cache; the forward runs wherever the model is.
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, item in enumerate(items):
            prompt = str(item["prompt"])
            ids = encode(prompt).to(device)
            state = state_to(copy_state(wake_state), device)
            positions: list[list[torch.Tensor]] = []
            for t in range(ids.shape[1]):
                model.c_capture = []
                _, state = model(ids[:, t : t + 1], state=state)
                positions.append([c.half().cpu() for c in group_by_layer(model.c_capture, n_layers)[0]])
                model.c_capture = None
            out[prompt] = positions
            print(f"[{ts()}]  battery read queries {i + 1}/{len(items)}: {len(positions)} positions "
                  f"from {prompt[:40]!r}", flush=True)
    return out


def dream_generation_seed(seed: int, index: int, attempt: int, offset: int = 0) -> int:
    """The generation seed for dream `index` of a set.

    `offset` shifts the whole block so that several processes can extend ONE
    wake state's set concurrently: without it every process derives the same
    seeds from --seed and produces byte-identical dreams, duplicating work
    instead of covering more of the distribution. Retry attempts live in a
    band far above any offset, so a regeneration cannot collide with another
    process's slot.
    """
    return seed * 1000 + offset + index + 1_000_000 * attempt


def build_dream_set(model, args, cache_path: Path, transcript, facts, chunk_len,
                    encode, decode, tokenizer, user_open, asst_open, stop_id: int | None,
                    adapter_sha: str | None = None, battery=None) -> DreamSetCache:
    """The multi-dream cache (sec 2.10.4): N dreams generated upfront, EACH
    from a fresh copy of the intact wake state by the sleep-start snapshot,
    each gated and reduced to its own per-layer eraser.

    Generation is nondeterministic, so pairing requires one shared set; and
    interleaving generation with training could not change the dreams anyway --
    the generator is the sleep-start snapshot by registration.
    """
    import torch

    print(f"\n[{ts()}] === building a {args.dreams}-dream set for seed {args.seed} -> {cache_path} ===")
    wake_state = run_chunks(model, transcript, None, chunk_len, "wake", keep_logits=False)[1]

    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    prefix_len = seed_ids.shape[1]
    needles = [f.entity for f in facts] + [f.code for f in facts]
    cues = build_cues([(1, f) for f in facts], encode, user_open, asst_open) if args.cue_every else []
    print(f"[{ts()}] steer prefix {args.dream_prompt!r} -> {prefix_len} tokens, excluded from every "
          f"arm's scored positions; gate threshold {args.gate_threshold} nats, rank rule {args.rank_rule}")
    print(f"[{ts()}] cue splicing: " + (f"every {args.cue_every} tokens, {args.cue_greedy} greedy "
                                        f"({len(cues)} cues)" if cues else "off (free-running dreams)"))

    model.eval()
    dreams: list[CachedDream] = []
    pilot: list[PilotDream] = []
    started = time.time()
    regenerated = 0
    for i in range(args.dreams):
        for attempt in range(DREAM_RETRIES + 1):
            gen_seed = dream_generation_seed(args.seed, i, attempt,
                                             getattr(args, 'dream_seed_offset', 0))
            torch.manual_seed(gen_seed)
            print(f"\n[{ts()}] --- dream {i + 1}/{args.dreams} (generation seed {gen_seed}) ---")
            dream = teacher_dream(model, wake_state, seed_ids, args.dream_tokens, args.dream_temp,
                                  drain=False, decode_token=lambda i: decode([i]), needles=needles,
                                  cues=cues, cue_every=args.cue_every, cue_greedy=args.cue_greedy,
                                  frozen=adapter_sha is None, stop_id=stop_id,
                                  turn_id=tokenizer.eos_token_id)
            if not dream_is_degenerate("".join(dream.token_texts)):
                break
            regenerated += 1
            print(f"[{ts()}]  dream {i + 1} came out degenerate (attempt {attempt + 1} of "
                  f"{DREAM_RETRIES + 1}); regenerating under a bumped seed")
        else:
            raise SystemExit(
                f"dream slot {i} stayed degenerate through {DREAM_RETRIES + 1} attempts -- "
                f"generation is off the rails at this temperature/adapter, not unlucky. "
                f"Stop and rethink (sec 2.1's mojibake clause), don't truncate."
            )
        blank = blank_state_logits(model, dream.tokens, chunk_len, frozen=adapter_sha is None)
        divergence = state_divergence(dream.logits, blank)
        gate = gated_positions(divergence, args.gate_threshold, prefix_len, dream.cue_flags)
        print(f"[{ts()}]  state-dependency gate: {len(gate)} of {len(dream.token_texts)} positions "
              f"(D_t median {float(divergence.median()):.3f}, max {float(divergence.max()):.3f})")
        if not gate:
            # Not a failure: the gate selects B4's queries, it does not
            # validate dreams. This dream read nothing from the state, so its
            # eraser is empty and it contributes no denial pressure.
            print(f"[{ts()}]  NOTE: no position passed the gate at {args.gate_threshold} nats -- "
                  f"this dream's eraser is empty (B4 starts it from the intact wake state).")
        from gate_pilot import scheme_weights

        family = getattr(args, "gate_family", "hard")
        weights = (None if family == "hard"
                   else scheme_weights([dream.divergence[t] for t in gate], family))
        spectra, ranks, bases = dream_bases(dream.queries, gate, wake_state, args.rank_rule, weights)
        dreams.append(CachedDream(
            dream_ids=[int(t) for t in dream.tokens[0].tolist()],
            token_texts=dream.token_texts,
            teacher_logits=dream.logits.cpu(),
            cue_flags=dream.cue_flags,
            prefix_len=prefix_len,
            stop_reason=dream.stop_reason,
            divergence=[float(d) for d in divergence],
            gate_positions=gate,
            queries=[[c.cpu() for c in dream.queries[t]] for t in gate],
            spectra=spectra,
            ranks=ranks,
            bases=bases,
        ))
        if args.pilot_capture:
            pilot.append(PilotDream(
                dream_sha=dreams[-1].dream_sha,
                token_texts=dream.token_texts,
                divergence=[float(d) for d in divergence],
                queries=[[c.half().cpu() for c in per_layer] for per_layer in dream.queries],
                cue_flags=dream.cue_flags,
                prefix_len=prefix_len,
                stop_reason=dream.stop_reason,
            ))
        elapsed = time.time() - started
        print(f"[{ts()}]  dream {i + 1} cached; {fmt_duration(elapsed)} elapsed, "
              f"ETA {fmt_duration(elapsed / (i + 1) * (args.dreams - i - 1))}", flush=True)

    print(f"\n[{ts()}] degenerate dreams regenerated: {regenerated}  "
          f"(a rising count is an adapter/temperature finding, not noise)")
    cache = DreamSetCache(
        seed=args.seed,
        transcript_ids=[int(t) for t in transcript[0].tolist()],
        wake_state=state_to(wake_state, torch.device("cpu")),
        dreams=dreams,
        distractors=build_distractors(facts, args.seed),
        facts=[(f.entity, f.category, f.code) for f in facts],
        dream_seed_offset=getattr(args, "dream_seed_offset", 0),
        gate_family=getattr(args, "gate_family", "hard"),
        dream_prompt=args.dream_prompt,
        gate_threshold=args.gate_threshold,
        rank_rule=args.rank_rule,
        generator=adapter_sha or "base",
    )
    # The report gates (sec 4), so it runs BEFORE anything is written: an arm
    # cell prefers a set cache whenever one exists for the seed, so a refused
    # build that left its file behind would poison every later cell silently.
    report_dream_set(cache, args.bind_min_dreams, args.rank_rule)
    save_dream_cache(cache, cache_path)
    sidecar = sidecar_path(cache_path)
    write_dream_set_sidecar(cache, sidecar)
    print(f"\n[{ts()}] wrote {cache_path} and {sidecar}")
    if args.pilot_capture:
        capture = PilotCapture(
            seed=args.seed, facts=cache.facts,
            wake_state=cache.wake_state, dreams=pilot,
            battery_queries=battery_read_queries(model, battery, wake_state, encode,
                                                 len(model.layers)) if battery else {},
            gate_threshold=args.gate_threshold, rank_rule=args.rank_rule, set_sha=cache.set_sha,
        )
        torch.save(capture, pilot_path(cache_path))
        print(f"[{ts()}] pilot capture -> {pilot_path(cache_path)}: {len(pilot)} dreams, every "
              f"position's queries and D_t, {len(capture.battery_queries)} battery prompts. "
              f"Score it offline with gate_pilot.py.")
    return cache


def run_dream_set_sleep(mode, model, opt, args, wave, wake_state, seen, cache: DreamSetCache,
                        emit, periodic_probe, on_step, started: float) -> object:
    """One sleep over a dream set (sec 2.10.4/2.10.5). Arms A and B4 only --
    both are ordinary sequence training, which is what makes them
    co-schedulable and what makes B4's full BPTT legitimate (sec 2.7)."""
    variant = B4_ARMS.get(mode)
    facts = [f for _, f in seen]
    probe_seconds = 0.0
    train_started = time.time()

    def on_boundary(index: int, epoch: int, step: int) -> None:
        nonlocal probe_seconds
        dream = cache.dreams[index]
        bound, misbound = binding_coverage("".join(dream.token_texts), facts)
        emit({"phase": "dream", "wave": wave, "arm": mode, "dream": index, "epoch": epoch,
              "step": step,
              "dream_sha": dream.dream_sha, "tokens": len(dream.dream_ids),
              "stop_reason": dream.stop_reason, "gated_positions": len(dream.gate_positions),
              "basis_rank": [len(b) for b in dream.bases[variant]] if variant else None,
              "bound_by_fact": bound, "misbound_by_fact": misbound,
              "bound_cov": sum(v > 0 for v in bound.values())})
        if index % args.probe_every_dream:
            return
        print()
        at = time.time()
        periodic_probe(step)
        probe_seconds += time.time() - at

    tokens = distill_dream_set(model, opt, cache.dreams, wake_state, variant, args.dream_epochs,
                               args.kl_temp, on_step, on_boundary,
                               sigma_scaled=(mode == SIGMA_ARM))
    train_seconds = time.time() - train_started - probe_seconds
    # Sec 2.10.5: the boundary probe round is a primary deliverable, but it is
    # rate-checked rather than assumed cheap.
    print(f"\n[{ts()}] boundary probes {fmt_duration(probe_seconds)} against "
          f"{fmt_duration(train_seconds)} of training")
    if probe_seconds > train_seconds:
        print(f"[{ts()}] WARNING: boundary probes cost more than the training they measure -- "
              f"rerun with --probe-every-dream 2 and record the deviation.")
    emit({"phase": "sleep", "wave": wave, "arm": mode, "dreams": len(cache.dreams),
          "epochs": args.dream_epochs, "steps": len(cache.dreams) * args.dream_epochs,
          "token_gradients": tokens, "probe_seconds": probe_seconds,
          "train_seconds": train_seconds, "seconds": time.time() - started})
    print(f"[{ts()}] {tokens} token-gradients over {len(cache.dreams)} dreams "
          f"x {args.dream_epochs} epoch(s) in {fmt_duration(time.time() - started)}")
    return wake_state


def run_sleep(mode, model, opt, args, wave, wake_state, transcript, seen, chunk_len, cache,
              encode, decode, tokenizer, user_open, asst_open, emit, periodic_probe, stop_id,
              verify=None, committed=None):
    """One sleep. Wave 1 distils the seed's cached dream, shared byte-identically
    across arms; a later wave generates its own from the state it carried in
    (sec 3.7). Returns the state carried into the next wake (None where the arm
    clears it)."""
    import torch

    started = time.time()
    wave_facts = [f for w, f in seen if w == wave]
    # A dream set's budget is one pass per dream (sec 2.10.2), and B1-live's is
    # the dream itself -- neither is --distill-steps.
    dream_set = isinstance(cache, DreamSetCache) and mode not in ("sft-ref", "no-sleep")
    total = (len(cache.dreams) * args.dream_epochs if dream_set else
             args.dream_tokens if mode == "drain-live" else args.distill_steps)

    def on_step(step: int, loss: float) -> None:
        emit({"phase": "kl", "wave": wave, "arm": mode, "step": step, "loss": loss,
              "accum_window": args.accum_window})
        rate = (step + 1) / (time.time() - started)
        print(f"\r[{ts()}]  {mode} step {step + 1}/{total} loss {loss:.4f} "
              f"{rate:.2f} step/s ETA {fmt_duration((total - step - 1) / rate)}", end="", flush=True)
        if args.probe_every and not dream_set and (step + 1) % args.probe_every == 0 and step + 1 < total:
            print()
            periodic_probe(step + 1)

    if dream_set:
        # Probes ride the dream boundaries instead of a step counter (sec 2.10.5).
        model.train()
        carried = run_dream_set_sleep(mode, model, opt, args, wave, wake_state, seen, cache,
                                      emit, periodic_probe, on_step, started)
        model.eval()
        torch.cuda.empty_cache()
        return carried

    needles = [f.entity for _, f in seen] + [f.code for _, f in seen]

    # Before model.train(): generation is at fixed weights, in eval mode.
    dream = None
    if mode not in ("sft-ref", "drain-live"):
        if wave == 1:
            dream = dream_from_cache(cache, wake_state.ssm_states[0].device)
        else:
            dream = generate_wave_dream(model, args, wake_state, wave_facts, encode, decode,
                                        tokenizer, user_open, asst_open, teacher=args.wave_teacher,
                                        stop_id=stop_id)
            emit({"phase": "cache", "wave": wave, "arm": mode, "seed": args.seed,
                  "dream_sha": token_sha(dream.tokens[0].tolist()),
                  "dream_tokens": dream.tokens.shape[1],
                  "free_tokens": sum(not f for f in dream.cue_flags),
                  "cues_cover": [f.entity for f in wave_facts],
                  "dream_generator": "base" if args.wave_teacher == "base" else "student"})

    model.train()
    if mode == "sft-ref":
        token_gradients = distill_sft(model, opt, transcript, args.distill_steps, chunk_len, on_step)
        print()
        emit({"phase": "sleep", "wave": wave, "arm": mode, "steps": args.distill_steps,
              "token_gradients": token_gradients, "seconds": time.time() - started})
        model.eval()
        return None

    if mode == "drain-live":
        seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
        carried, dream = distill_live(model, opt, wake_state, seed_ids, args.dream_tokens, args.dream_temp,
                                      args.kl_temp, args.accum_window,
                                      lambda i: decode([i]), needles, on_step, erase_op=args.erase_op)
        token_gradients = args.dream_tokens
        print()
    else:
        keep = scored_keep(dream.cue_flags, dream.prefix_len) if dream.cue_flags else None
        if mode in ("replay", "ce-on-dream"):
            # Registered form: the chunk is the whole dream, one optimizer step
            # per pass, from a fresh state. --chunk-len is the bridge cell.
            replay_chunk = args.chunk_len or dream.tokens.shape[1]
            print(f"[{ts()}] replay chunk {replay_chunk} tokens over {dream.tokens.shape[1]}, "
                  f"fresh-state-replay {args.fresh_state_replay}, objective "
                  f"{'CE on the dream tokens' if mode == 'ce-on-dream' else 'KL to the cached teacher'}")
            token_gradients = distill_replay(model, opt, dream, args.distill_steps, replay_chunk, args.kl_temp,
                                             on_step, fresh_state=args.fresh_state_replay, keep=keep,
                                             ce=(mode == "ce-on-dream"))
            carried = None
        elif mode in FUSED_ARMS:
            generator_spine = None
            if mode == "b3-fused":
                # The generator snapshot: the weights that produced this
                # sleep's dream, i.e. the student before this sleep trains it.
                print(f"[{ts()}] b3-fused: materializing the generator's own spine over "
                      f"{dream.tokens.shape[1]} tokens (block {args.spine_block})", flush=True)
                with torch.no_grad():
                    generator_spine = spine_states(model, dream.tokens, wake_state, args.spine_block)
            token_gradients = distill_fused(
                model, opt, dream, wake_state, args.distill_steps, args.kl_temp, on_step, keep=keep,
                erase_op=args.erase_op, deep=(mode == "b2-fused-deep"), block=args.spine_block,
                cf_batch=args.cf_batch, frozen_spine=generator_spine,
                check=lambda record: emit({"phase": "equivalence", "wave": wave, "arm": mode, **record}))
            carried = wake_state
        else:
            token_gradients, drained = distill_counterfactual(
                model, opt, dream, wake_state, args.distill_steps, args.kl_temp, args.accum_window,
                in_place=(mode == "drain"), deep=args.deep, on_step=on_step, keep=keep,
                erase_op=args.erase_op)
            carried = drained if mode == "drain" else wake_state
        print()

    fraction, counts = rehearsal_fraction(dream.token_texts, needles)
    facts = [f for _, f in seen]
    bound, misbound = binding_coverage("".join(dream.token_texts), facts)
    emit({"phase": "dream", "wave": wave, "arm": mode, "tokens": len(dream.token_texts),
          "rehearsal_fraction": fraction, "needle_counts": counts, "skipped_cone": dream.skipped_cone,
          "bound_cov": sum(v > 0 for v in bound.values()), "misbound": sum(misbound.values()),
          "bound_by_fact": bound, "misbound_by_fact": misbound,
          "temperature": args.dream_temp, "prompt": args.dream_prompt,
          "decoded": "".join(dream.token_texts)})
    emit({"phase": "sleep", "wave": wave, "arm": mode, "steps": args.distill_steps,
          "token_gradients": token_gradients, "deep": args.deep, "seconds": time.time() - started})
    print(f"[{ts()}] dream rehearsal fraction {fraction:.3f}  bound {bound}  misbound {misbound}")
    print(f"[{ts()}] {token_gradients} token-gradients in {fmt_duration(time.time() - started)}")
    if fraction == 0.0:
        print(f"[{ts()}] WARNING: the dream never rehearsed a fact -- reads never touched the bindings, so "
              f"nothing could distil. Seed the dream with --dream-prompt before reading anything into this arm.")
    print(f"[{ts()}] decoded dream:\n{''.join(dream.token_texts)!r}")

    if mode == "counterfactual-commit":
        if verify is None:
            raise SystemExit("arm counterfactual-commit needs a fresh-state margin check to commit against")
        print(f"\n[{ts()}] === wave {wave} commit step (fresh-state margin, then one real erase) ===")
        model.eval()
        commit_erase(model, carried, seen, set() if committed is None else committed,
                     encode, user_open, asst_open, verify, args.erase_op, emit, wave)

    model.eval()
    torch.cuda.empty_cache()
    return carried


if __name__ == "__main__":
    main()
