"""Dream-distillation sleep: the continual-learning A/B between consolidating a
session by *dreaming it back out of the state* and consolidating it by
conventional fine-tuning.

The protocol and the arm sequences are registered in
notes/DISCUSSION-20260806-dream-distillation-ab-postmortem.md sec 3, which
supersedes the 08-05 file's sequences -- implemented here verbatim. Wake is
exactly stock: no gate, no erase, no new parameters. The erase fires only
inside a sleep, is hard-coded at gamma = 1.0 with the state's top singular
direction deflated out (sec 3a's probe result -- sub-1 gamma is both a no-op
under the mixer's gated RMSNorm and an invitation to compensate), and never
touches the wake path.

  wake          -- the consolidation-null generator's transcript: --n-facts
                   entity->code facts separated by --filler-tokens of
                   digit-free filler, primed into the state.
  cache         -- --build-dream-cache generates this seed's ONE teacher dream
                   (frozen base, from a copy of the wake state, intact) and
                   persists it with the wake state, the teacher logits and
                   queries, distractor codes and sha-256 hashes of both token
                   sequences. Every arm loads it; no arm generates.
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
                   --ce-on-dream      : A's sequence, cross-entropy on the
                                       dream tokens instead of KL.
                   --deep             : B2 with the pass's losses accumulated
                                       and one optimizer step per pass (full
                                       BPTT through the spine).
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
manipulated variable. `--waves 2` runs wake -> sleep -> wake(new facts, on the
carried state) -> sleep -> probes, which is the only form that can price
consumption (wave-2 in-context control) and backward transfer (wave-1 recall
after sleep 2).

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
    fmt_duration,
    generate,
    kl_loss,
    render_turns,
    replay_step,
    report_transcript,
    run_chunks,
    target_logprob,
    ts,
)
from erase_probe import deflate, group_by_layer, rank1_erase, state_top_dirs
from lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK
from probes_common import (
    BATTERY_CANDIDATES,
    HELDOUT_TEXT,
    battery_summary,
    code_margin,
    load_or_build_battery,
    logprob_sum,
    perplexity,
    score_battery,
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

# The cue timer defers its splice to the next sentence end, so a cue never cuts
# a thought in half (the prepare_chains splice lesson); this is how far it will
# wait before splicing anyway.
CUE_DEFER_MAX = 20
CUE_STOPS = (".", "\n")

# The distractor codes' RNG stream is the wake seed xor this, so adding them
# leaves every prior run's wake transcript bit-identical.
DISTRACTOR_SALT = 0x5EED

# What each arm hands to the next wake, per the sec 3 sequences.
ARM_CARRY = {
    "replay": "none",
    "ce-on-dream": "none",
    "drain": "drained",
    "counterfactual": "intact",
    "drain-live": "drained",
    "sft-ref": "none",
    "no-sleep": "intact",
}
ARMS = ("replay", "drain", "counterfactual", "drain-live")

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


def sample_next(logits: torch.Tensor, temperature: float, banned: Sequence[int]) -> torch.Tensor:
    """Next token from (B, V) logits, with `banned` ids made unreachable --
    EOS is banned during a dream so the model keeps rehearsing to budget."""
    import torch

    last = logits.float().clone()
    for token_id in banned:
        last[:, token_id] = -float("inf")
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

    Batched over the leading dimension throughout: an arm that ablates many
    dream positions at once gets a basis and a skip decision per element.

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


def token_sha(ids: Sequence[int]) -> str:
    """SHA-256 over a token sequence. A registered invariant ships with its
    machine check (sec 2): every result file records these and the summarizer
    refuses to pool cells whose dream or transcript disagree."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def build_distractors(facts: Sequence[Fact], seed: int) -> dict[str, str]:
    """One fixed foil code per fact, drawn from its own RNG stream so the wake
    transcript's draws are unchanged. The margin metric (sec 4) scores the
    correct code against these, which is immune to the format prior and to the
    digit-counting attractor that broke greedy exact match."""
    rng = random.Random(seed ^ DISTRACTOR_SALT)
    taken = {f.code for f in facts}
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
    transcript_sha: str = ""
    dream_sha: str = ""

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


def save_dream_cache(cache: DreamCache, path: str | Path) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_dream_cache(path: str | Path) -> DreamCache:
    """Load and re-verify. A cache whose tokens no longer hash to what it was
    saved with is refused outright: every downstream number is conditioned on
    the arms having seen the same tokens."""
    import torch

    # The cache holds a MixerState, not just tensors, so weights_only is off --
    # it is this repo's own artifact, written by the cache builder.
    cache: DreamCache = torch.load(path, map_location="cpu", weights_only=False)
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


def write_dream_sidecar(cache: DreamCache, path: str | Path) -> None:
    facts = cache.fact_list
    bound, misbound = binding_coverage("".join(cache.token_texts), facts)
    header = [
        f"seed {cache.seed}   dream_sha {cache.dream_sha}   transcript_sha {cache.transcript_sha}",
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
    )


def teacher_dream(
    model,
    wake_state,
    seed_ids: torch.Tensor,
    n_tokens: int,
    temperature: float,
    banned: Sequence[int],
    drain: bool,
    decode_token,
    needles: Sequence[str],
    cues: Sequence[Sequence[int]] = (),
    cue_every: int = 0,
    cue_greedy: int = 0,
    frozen: bool = True,
    erase_op: str = ERASE_OP,
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
                    ids.append(int(sample_next(logits[:, -1], temp, banned).item()))
                    cue_flags.append(False)

            if (t + 1) % PRINT_EVERY == 0 or t + 1 == n_tokens:
                frac, _ = rehearsal_fraction(texts, needles)
                rate = (t + 1) / (time.time() - started)
                print(f"[{ts()}]  dream {t + 1}/{n_tokens} rehearsal {frac:.2f} {rate:.1f} tok/s "
                      f"ETA {fmt_duration((n_tokens - t - 1) / rate)} | "
                      f"{''.join(texts[-PRINT_EVERY:])!r}", flush=True)
    return Dream(
        tokens=torch.tensor([ids[:n_tokens]], dtype=torch.long, device=seed_ids.device),
        logits=torch.stack(logits_cache),
        queries=queries,
        final_state=state,
        token_texts=texts,
        skipped_cone=skipped,
        cue_flags=cue_flags[:n_tokens],
    )


def distill_replay(model, opt, dream: Dream, steps: int, chunk_len: int, kl_temp: float, on_step,
                   fresh_state: bool = False, keep: Sequence[bool] | None = None, ce: bool = False) -> int:
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
    """
    import torch.nn.functional as F

    length = dream.tokens.shape[1]
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
    tokens = 0
    for step in range(steps):
        c, reset = replay_step(step, n_chunks, fresh_state)
        if reset:
            state = None
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

    skipped = 0

    def hook(layer_idx: int, ssm_state, c):
        nonlocal skipped
        erased, was_skipped = erase_ssm(ssm_state, c, op=erase_op)
        skipped += was_skipped
        return erased

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
    print(f"\n[{ts()}]  near-cone erases skipped: {skipped} of {tokens * max(1, len(getattr(model, 'layers', [1])))}")
    return tokens, state


def distill_live(
    model, opt, wake_state, seed_ids: torch.Tensor, n_tokens: int, temperature: float,
    banned: Sequence[int], kl_temp: float, accum: int, decode_token, needles: Sequence[str], on_step,
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
            ids.append(int(sample_next(stored[:, -1], temperature, banned).item()))
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
    parser.add_argument("--waves", type=int, default=1, help="Wake/sleep waves; 2 adds a second wake on the carried state (default: %(default)s)")
    parser.add_argument("--wave-teacher", choices=("base", "current"), default=None,
                        help="Who generates the dream for waves after the first: the frozen base, or the model this run has trained. Required when --waves > 1.")
    parser.add_argument("--n-facts", type=int, default=4, help="Facts per wave -- the measured binding ceiling (default: %(default)s)")
    parser.add_argument("--filler-tokens", type=int, default=40, help="Filler tokens between consecutive facts (default: %(default)s)")
    parser.add_argument("--dream-tokens", type=int, default=DREAM_TOKENS, help="Dream length per sleep (default: %(default)s)")
    parser.add_argument("--dream-temp", type=float, default=1.0, help="Dream sampling temperature (default: %(default)s)")
    parser.add_argument("--dream-prompt", default="", help="Text seeding the dream after the assistant marker (sec 4's category-cue fallback)")
    parser.add_argument("--cue-greedy", type=int, default=12, help="Tokens after each cue decoded greedily -- the recalled code, which temperature sampling almost never gets right (default: %(default)s)")
    parser.add_argument("--cue-every", type=int, default=0, help="Force a fact's question stem into the dream every N tokens, cycling the wave's facts; 0 leaves generation free (default: %(default)s)")
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
    parser.add_argument("--probe-every", type=int, default=200,
                        help="Stream the full probe battery every N distillation steps, so every cell yields a "
                             "learned-vs-forgotten curve; 0 probes only at the end (default: %(default)s)")
    parser.add_argument("--distill-steps", type=int, default=200, help="Optimizer steps per sleep (default: %(default)s)")
    parser.add_argument("--accum-window", type=int, default=1, help="Positions accumulated per optimizer step in the per-token arms (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--kl-temp", type=float, default=1.0, help="Distillation temperature (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward chunk (default: the model's DEFAULT_CHUNK_LEN)")
    parser.add_argument("--init-adapter", default=None,
                        help="Warm-start adapter checkpoint (warm_start trainer's save_adapter output) loaded into "
                             "the model before the dream cache, the battery or any training; its SHA-256 is stamped "
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
    mode = "sft-ref" if args.sft_ref else ("no-sleep" if args.no_sleep else
                                           ("ce-on-dream" if args.ce_on_dream else args.arm))

    cache_path = Path(args.dream_cache or f"data/dream_cache_s{args.seed}.pt")
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
        from lora import load_adapter
        loaded = load_adapter(model, args.init_adapter, args.lora_rank, args.lora_alpha)
        adapter_sha = file_sha(args.init_adapter)
        print(f"[{ts()}] warm start {args.init_adapter}: {loaded} tensors, sha256 {adapter_sha[:12]}")
    model.eval()
    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    stops = (".", "\n", user_open, asst_open)
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device)

    def decode(ids) -> str:
        return tokenizer.decode(ids)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    all_facts = build_facts(args.n_facts * args.waves, rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w")

    emit = make_emit(out_file, erase_op=args.erase_op, init_adapter_sha256=adapter_sha,
                     init_adapter=os.path.basename(args.init_adapter) if args.init_adapter else None)

    cache = None if args.build_dream_cache else load_dream_cache(cache_path)
    distractors = cache.distractors if cache else {}
    if cache is not None:
        print(f"[{ts()}] dream cache {cache_path}: {len(cache.dream_ids)} tokens "
              f"({cache.free_tokens} freely generated), transcript_sha {cache.transcript_sha[:12]} "
              f"dream_sha {cache.dream_sha[:12]}")

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
                    baseline: dict[str, float] | None, step: int | None = None) -> dict[str, float]:
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
    battery = load_or_build_battery(battery_path, BATTERY_CANDIDATES, battery_probe)
    if not battery:
        raise SystemExit("the self-calibrated knowledge battery is empty -- no locality baseline to measure against")
    base_ppl = perplexity(model, heldout, chunk_len, "heldout ppl")
    print(f"[{ts()}] held-out ppl {base_ppl:.3f} over {heldout.shape[1]} tokens; battery {len(battery)} items")
    emit({"phase": "baseline", "arm": mode, "battery_items": len(battery), "ppl": base_ppl})

    def locality(wave: int, step: int | None = None) -> None:
        scored = score_battery(battery, scored_battery_probe)
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
    carried = None
    seen: list[tuple[int, Fact]] = []
    fresh_baseline: dict[str, float] = {}
    for wave in range(1, args.waves + 1):
        facts = all_facts[(wave - 1) * args.n_facts : wave * args.n_facts]
        turns = build_turns(facts, args.filler_tokens, lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"]), rng)
        text = render_turns(turns, user_open, asst_open)
        transcript = encode(text)
        print(f"\n[{ts()}] === wave {wave} wake ===")
        report_transcript(text, decode(transcript[0].cpu()), facts, turns, transcript.shape[1])
        emit({"phase": "transcript", "wave": wave, "arm": mode, "tokens": transcript.shape[1],
              "facts": [f.entity for f in facts]})

        if args.build_dream_cache:
            build_cache(model, args, cache_path, transcript, facts, chunk_len,
                        encode, decode, tokenizer, user_open, asst_open)
            out_file.close()
            print(f"\n[{ts()}] cache built; every arm of seed {args.seed} now distils this dream")
            return

        if wave == 1:
            if token_sha(transcript[0].tolist()) != cache.transcript_sha:
                raise SystemExit(
                    f"wave-1 transcript does not match {cache_path}'s: this cell would distil a dream generated "
                    f"from a different wake session. Rebuild the cache for seed {args.seed}."
                )
            emit({"phase": "cache", "wave": 1, "arm": mode, "seed": args.seed, "path": str(cache_path),
                  "transcript_sha": cache.transcript_sha, "dream_sha": cache.dream_sha,
                  "dream_tokens": len(cache.dream_ids), "free_tokens": cache.free_tokens,
                  "cue_tokens": len(cache.dream_ids) - cache.free_tokens})

        # The fresh-state floor for this wave's facts, before they are anywhere
        # but the transcript -- what every later log-prob delta is measured against.
        print(f"[{ts()}] fresh-state floor, wave {wave} facts")
        fresh_baseline |= probe_facts([(wave, f) for f in facts], None, "floor", wave, False, None)

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
                                encode, decode, tokenizer, user_open, asst_open, emit, periodic_probe)

        print(f"\n[{ts()}] === wave {wave} probes: fresh state, no context ===")
        probe_facts(seen, None, "probe", wave, True, fresh_baseline, step=args.distill_steps)
        locality(wave, step=args.distill_steps)
        print(f"\n[{ts()}] === wave {wave} carried-state diagnostic ({ARM_CARRY[mode]}) -- NOT installation ===")
        if carried is None:
            print(f"[{ts()}]  arm {mode} carries nothing; column empty by construction")
        else:
            probe_facts(seen, carried, "carried", wave, False, None)

    # The cell's completion marker. Periodic probes write a locality record
    # every --probe-every steps, so "has a locality record" says a cell started,
    # not that it finished; the driver's resume check and the summarizer both
    # key on this record instead.
    emit({"phase": "done", "arm": mode, "seed": args.seed, "waves": args.waves,
          "steps": args.distill_steps})
    out_file.close()
    print(f"\n[{ts()}] done -> {out_path}")


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
    protocol choice: the frozen base keeps wave 1's contract (a stationary
    teacher that never drifts), while the current model distils a student the
    run has already consolidated into. The two measure different things, so the
    harness refuses to pick."""
    if args.waves > 1 and not getattr(args, "wave_teacher", None):
        raise SystemExit(
            "--waves > 1 needs --wave-teacher base|current: wave 1's dream comes from the frozen base, "
            "and after that 'the teacher' is ambiguous (frozen base vs the student trained by earlier "
            "sleeps). Register the choice before running multi-sleep."
        )


def generate_wave_dream(model, args, carried, facts, encode, decode, tokenizer,
                        user_open, asst_open, teacher: str) -> Dream:
    """The dream a wave after the first distils: generated from that wave's own
    carried state, cued on that wave's facts.

    Wave 1 loads the seed's cached dream, shared byte-identically across arms.
    A later wave cannot: each arm reaches it with a different carried state (A
    cleared, B1 selectively vacated, B2 intact), which is the whole object of
    the multi-sleep contrast, so these dreams legitimately differ per arm and
    are recorded rather than asserted equal.
    """
    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    banned = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    needles = [f.entity for f in facts] + [f.code for f in facts]
    cues = build_cues([(0, f) for f in facts], encode, user_open, asst_open) if args.cue_every else []

    model.eval()
    return teacher_dream(
        model, carried, seed_ids, args.dream_tokens, args.dream_temp, banned,
        drain=False, decode_token=lambda i: decode([i]), needles=needles,
        cues=cues, cue_every=args.cue_every, cue_greedy=args.cue_greedy,
        frozen=(teacher == "base"))


def build_cache(model, args, cache_path: Path, transcript, facts, chunk_len,
                encode, decode, tokenizer, user_open, asst_open) -> DreamCache:
    """Generate this seed's one teacher dream and persist it with its hashes,
    its distractor codes and a decoded sidecar. Frozen base, from a copy of the
    wake state, intact -- no arm ever regenerates (sec 3's shared preamble)."""
    import torch

    print(f"\n[{ts()}] === building dream cache for seed {args.seed} -> {cache_path} ===")
    wake_state = run_chunks(model, transcript, None, chunk_len, "wake", keep_logits=False)[1]

    seed_ids = encode(dream_seed_text(asst_open, args.dream_prompt))
    banned = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    needles = [f.entity for f in facts] + [f.code for f in facts]
    seen = [(1, f) for f in facts]
    cues = build_cues(seen, encode, user_open, asst_open) if args.cue_every else []

    model.eval()
    dream = teacher_dream(model, wake_state, seed_ids, args.dream_tokens, args.dream_temp, banned,
                          drain=False, decode_token=lambda i: decode([i]), needles=needles,
                          cues=cues, cue_every=args.cue_every, cue_greedy=args.cue_greedy)

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
    )
    save_dream_cache(cache, cache_path)
    sidecar = cache_path.parent / f"dream_s{args.seed}.txt"
    write_dream_sidecar(cache, sidecar)
    print(f"[{ts()}] wrote {cache_path} and {sidecar}")
    print(f"[{ts()}] transcript_sha {cache.transcript_sha}\n[{ts()}] dream_sha      {cache.dream_sha}")
    report_dream(cache)
    return cache


def run_sleep(mode, model, opt, args, wave, wake_state, transcript, seen, chunk_len, cache,
              encode, decode, tokenizer, user_open, asst_open, emit, periodic_probe):
    """One sleep, on the cached dream. Returns the state carried into the next
    wake (None where the arm clears it)."""
    import torch

    started = time.time()
    # B1-live's budget is the dream itself (one online pass), not --distill-steps.
    total = args.dream_tokens if mode == "drain-live" else args.distill_steps

    def on_step(step: int, loss: float) -> None:
        emit({"phase": "kl", "wave": wave, "arm": mode, "step": step, "loss": loss,
              "accum_window": args.accum_window})
        rate = (step + 1) / (time.time() - started)
        print(f"\r[{ts()}]  {mode} step {step + 1}/{total} loss {loss:.4f} "
              f"{rate:.2f} step/s ETA {fmt_duration((total - step - 1) / rate)}", end="", flush=True)
        if args.probe_every and (step + 1) % args.probe_every == 0 and step + 1 < total:
            print()
            periodic_probe(step + 1)

    needles = [f.entity for _, f in seen] + [f.code for _, f in seen]
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
        banned = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        carried, dream = distill_live(model, opt, wake_state, seed_ids, args.dream_tokens, args.dream_temp,
                                      banned, args.kl_temp, args.accum_window,
                                      lambda i: decode([i]), needles, on_step, erase_op=args.erase_op)
        token_gradients = args.dream_tokens
        print()
    else:
        dream = dream_from_cache(cache, wake_state.ssm_states[0].device)
        keep = target_keep_mask(dream.cue_flags) if dream.cue_flags else None
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

    model.eval()
    torch.cuda.empty_cache()
    return carried


if __name__ == "__main__":
    main()
