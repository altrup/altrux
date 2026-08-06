"""Dream-distillation sleep: the continual-learning A/B between consolidating a
session by *dreaming it back out of the state* and consolidating it by
conventional fine-tuning.

The protocol and the four arm sequences are registered in
notes/DISCUSSION-20260805-dream-distillation-cl-ab.md sec 3 -- implemented here
verbatim, numbered as they are numbered there. Wake is exactly stock: no gate,
no erase, no new parameters. The erase fires only inside a sleep, is
hard-coded at gamma = 1.0 with the state's top singular direction deflated out
(sec 3a's probe result -- sub-1 gamma is both a no-op under the mixer's gated
RMSNorm and an invitation to compensate), and never touches the wake path.

  wake          -- the consolidation-null generator's transcript: --n-facts
                   entity->code facts separated by --filler-tokens of
                   digit-free filler, primed into the state.
  sleep --arm   -- replay        (A):  frozen teacher + wake state generates a
                                       dream; student distils it from a FRESH
                                       state; carries nothing.
                   drain         (B1): teacher generates while erasing along
                                       each token's own read query; student
                                       forwards each position from that
                                       position's drained state; carries the
                                       final drained state.
                   counterfactual(B2): shares A's dream; student forwards each
                                       position from a copy of the wake state
                                       advanced to t and ablated along c_t;
                                       carries the intact wake state.
                   drain-live         : one online adapters-on pass -- forward,
                                       erase, gradient step against the stored
                                       pre-erase logits, sample, continue from
                                       the drained state. Exploratory.
                   --sft-ref          : the CE-on-raw-text convention, on the
                                       stored wake transcript.
                   --no-sleep         : the floor -- no training at all.
  probes        -- fresh-state reliability, ~4 paraphrases per fact
                   (generality), the self-calibrated knowledge battery and
                   held-out ppl (locality/forgetting), then the carried-state
                   column as a DIAGNOSTIC -- never scored as installation.

All three registered arms train the same objective (KL to the frozen teacher's
cached logits); the student's state deprivation is the only manipulated
variable. `--waves 2` runs wake -> sleep -> wake(new facts, on the carried
state) -> sleep -> probes, which is the only form that can price consumption
(wave-2 in-context control) and backward transfer (wave-1 recall after sleep 2).

Box tool: this trains a LoRA and holds a full-vocab logit cache for the dream
-- it runs on rented CUDA hardware, never the local ROCm box. Only the pure
pieces are CPU-testable (tests/test_dream_sleep.py), which is why the torch and
models.* imports live inside the functions that need them.

Usage (from sft/, env vars as in the Makefile):
    make dream-sleep ARGS="--arm counterfactual --dream-tokens 512 --distill-steps 200"
    make dream-sleep ARGS="--arm drain --waves 2"
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import random
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation_null import (
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
    report_transcript,
    run_chunks,
    target_logprob,
    ts,
)
from erase_probe import deflate, group_by_layer, rank1_erase, state_top_dirs
from probes_common import (
    BATTERY_CANDIDATES,
    HELDOUT_TEXT,
    battery_summary,
    load_or_build_battery,
    perplexity,
    score_battery,
)

# Registered erase parameters (sec 3a) -- deliberately not flags. Do not
# re-tune gamma downward without new evidence of a kind the erase probe
# could not see.
GAMMA = 1.0
DEFLATE_K = 1
# A query with almost nothing left after deflation is all shared cone and no
# discriminative sliver: skip it rather than erase noise (erase_probe.py).
CONE_SKIP = 0.05

DREAM_TOKENS = 512
PRINT_EVERY = 16

# What each arm hands to the next wake, per the sec 3 sequences.
ARM_CARRY = {
    "replay": "none",
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


def erase_state(state, queries: Sequence[torch.Tensor], gamma: float = GAMMA, k: int = DEFLATE_K) -> int:
    """Apply the registered erase to every layer of `state` in place, each with
    that layer's own read query. Returns the number of near-cone directions
    skipped."""
    skipped = 0
    for i, c in enumerate(queries):
        direction = deflate(c.to(state.ssm_states[i].device), state_top_dirs(state.ssm_states[i], k))
        if direction.float().norm() < CONE_SKIP * c.float().norm():
            skipped += 1
            continue
        state.ssm_states[i] = rank1_erase(state.ssm_states[i], direction, gamma)
    return skipped


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
    still sampled from the state rather than forced.
    """
    import torch

    state = copy.deepcopy(wake_state)
    ids = [int(i) for i in seed_ids[0].tolist()]
    next_cue, cue_at, greedy_left = 0, len(ids) + cue_every, 0
    logits_cache: list[torch.Tensor] = []
    queries: list[list[torch.Tensor]] = []
    texts: list[str] = []
    skipped = 0
    started = time.time()
    with frozen_teacher(model), torch.no_grad():
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
                skipped += erase_state(state, per_layer)
            if t + 1 >= len(ids):
                if cues and cue_every and len(ids) >= cue_at:
                    ids.extend(int(i) for i in cues[next_cue % len(cues)])
                    next_cue += 1
                    cue_at = len(ids) + cue_every
                    greedy_left = cue_greedy
                else:
                    temp = 0.0 if greedy_left > 0 else temperature
                    greedy_left = max(0, greedy_left - 1)
                    ids.append(int(sample_next(logits[:, -1], temp, banned).item()))

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
    )


def distill_replay(model, opt, dream: Dream, steps: int, chunk_len: int, kl_temp: float, on_step) -> None:
    """Arm A sequence 3: the student is teacher-forced over the dream from a
    FRESH state, in chunks, KL to the cached logits. One optimizer step is one
    chunk; state is carried and detached within a pass and reset at pass
    boundaries -- the null's truncated-BPTT shape."""
    length = dream.tokens.shape[1]
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
    for step in range(steps):
        c = step % n_chunks
        if c == 0:
            state = None
        lo, hi = c * chunk_len, min((c + 1) * chunk_len, length)
        logits, state = model(dream.tokens[:, lo:hi], state=state)
        state = state.detach()
        loss = kl_loss(dream.logits[lo:hi].unsqueeze(0).to(logits.device), logits, kl_temp)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        on_step(step, loss.item())


def distill_stateful(
    model, opt, dream: Dream, wake_state, steps: int, kl_temp: float, accum: int, drain: bool, on_step
) -> None:
    """Arm B1 sequence 2 (`drain=True`) and B2 sequence 2 (`drain=False`).

    Per dream position t the student forwards that one token from a copy of the
    position's state and matches the cached teacher logits: B1's state is the
    teacher's own drained state at t, B2's is the intact wake state advanced to
    t and then ablated along c_t (the copy is discarded either way).

    The per-position states are re-walked with the frozen teacher each pass
    rather than cached: at this model's shape a snapshot is ~38 MB, so a
    512-token dream would be ~19 GB of cache. The teacher is frozen and the
    dream tokens are fixed, so the walk is deterministic and the snapshots are
    the same ones the cached teacher pass produced.
    """
    import torch

    length = dream.tokens.shape[1]
    step = 0
    while step < steps:
        state = copy.deepcopy(wake_state)
        for t in range(length):
            if step >= steps:
                break
            token = dream.tokens[:, t : t + 1]
            per_layer = [c.to(token.device) for c in dream.queries[t]]

            student_state = copy.deepcopy(state)
            if not drain:
                erase_state(student_state, per_layer)
            logits, _ = model(token, state=student_state)
            loss = kl_loss(dream.logits[t].view(1, 1, -1).to(logits.device), logits, kl_temp) / accum
            loss.backward()
            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            on_step(step, loss.item() * accum)
            step += 1

            with frozen_teacher(model), torch.no_grad():
                _, state = model(token, state=state)
            if drain:
                erase_state(state, per_layer)


def distill_live(
    model, opt, wake_state, seed_ids: torch.Tensor, n_tokens: int, temperature: float,
    banned: Sequence[int], kl_temp: float, accum: int, decode_token, needles: Sequence[str], on_step,
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

        skipped += erase_state(state, per_layer)
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


def distill_sft(model, opt, ids: torch.Tensor, steps: int, chunk_len: int, on_step) -> None:
    """--sft-ref: plain next-token cross-entropy on the stored wake transcript
    verbatim, from a fresh state. The CE-on-raw-text convention the dream arms
    are being compared against -- the objective-type confound lives here and
    only here."""
    import torch.nn.functional as F

    length = ids.shape[1] - 1
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
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
        on_step(step, loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", choices=ARMS, default="replay", help="Sleep protocol, per DISCUSSION sec 3 (default: %(default)s)")
    parser.add_argument("--sft-ref", action="store_true", help="Reference arm: CE on the wake transcript instead of a dream")
    parser.add_argument("--no-sleep", action="store_true", help="Floor arm: no training at all, wake state carried")
    parser.add_argument("--waves", type=int, default=1, help="Wake/sleep waves; 2 adds a second wake on the carried state (default: %(default)s)")
    parser.add_argument("--n-facts", type=int, default=4, help="Facts per wave -- the measured binding ceiling (default: %(default)s)")
    parser.add_argument("--filler-tokens", type=int, default=40, help="Filler tokens between consecutive facts (default: %(default)s)")
    parser.add_argument("--dream-tokens", type=int, default=DREAM_TOKENS, help="Dream length per sleep (default: %(default)s)")
    parser.add_argument("--dream-temp", type=float, default=1.0, help="Dream sampling temperature (default: %(default)s)")
    parser.add_argument("--dream-prompt", default="", help="Text seeding the dream after the assistant marker (sec 4's category-cue fallback)")
    parser.add_argument("--cue-greedy", type=int, default=12, help="Tokens after each cue decoded greedily -- the recalled code, which temperature sampling almost never gets right (default: %(default)s)")
    parser.add_argument("--cue-every", type=int, default=0, help="Force a fact's question stem into the dream every N tokens, cycling the wave's facts; 0 leaves generation free (default: %(default)s)")
    parser.add_argument("--distill-steps", type=int, default=200, help="Optimizer steps per sleep (default: %(default)s)")
    parser.add_argument("--accum-window", type=int, default=1, help="Positions accumulated per optimizer step in the per-token arms (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--kl-temp", type=float, default=1.0, help="Distillation temperature (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=None, help="Tokens per forward chunk (default: the model's DEFAULT_CHUNK_LEN)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS, help="Tokens generated per probe (default: %(default)s)")
    parser.add_argument("--battery", default=None, help="Knowledge-battery artifact (default: data/knowledge_battery_<model>.json)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="logs/dream_sleep.jsonl", help="Per-fact results jsonl (default: %(default)s)")
    args = parser.parse_args()

    if args.sft_ref and args.no_sleep:
        raise SystemExit("--sft-ref and --no-sleep are different arms; pass one")
    mode = "sft-ref" if args.sft_ref else ("no-sleep" if args.no_sleep else args.arm)

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
          f"chunk_len {chunk_len}, seed {args.seed}, gamma {GAMMA} (state-svd k={DEFLATE_K})")

    model, trainable = train_hooks.setup_training(device, args.lora_rank, args.lora_alpha, 0.0)
    if getattr(model, "c_capture", "missing") == "missing":
        raise SystemExit(f"model {model_name} has no c_capture hook -- this harness is for mamba2_780m")
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

    def emit(record: dict[str, object]) -> None:
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()

    # ---- probes -----------------------------------------------------------
    def answer_probe(prompt: str, answer: str, state=None) -> tuple[str, float]:
        prompt_ids, target_ids = encode(prompt), encode(" " + answer)
        generation = decode(generate(model, prompt_ids, copy.deepcopy(state), args.gen_tokens, 0.0)[0].cpu())
        return generation, target_logprob(model, prompt_ids, target_ids, copy.deepcopy(state))

    def probe_facts(facts: Sequence[tuple[int, Fact]], state, phase: str, wave: int, paraphrases: bool,
                    baseline: dict[str, float] | None) -> dict[str, float]:
        """Greedy exact match + teacher-forced code log-prob per fact, plus the
        paraphrase battery where asked. Streams one record per fact."""
        logprobs: dict[str, float] = {}
        hits = para_hits = 0
        for i, (fact_wave, fact) in enumerate(facts):
            generation, logprob = answer_probe(cue_rungs(fact, user_open, asst_open)[0][0], fact.code, state)
            matched = exact_match(generation, fact.code, stops)
            logprobs[fact.entity] = logprob
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
                "phase": phase, "wave": wave, "arm": mode, "fact_wave": fact_wave, "fact": fact.entity,
                "category": fact.category, "code": fact.code, "greedy": generation, "match": matched,
                "logprob": logprob, "logprob_delta": delta, "paraphrases": para, "paraphrase_rate": para_rate,
            })
            print(f"[{ts()}]  {phase} w{wave} {fact.entity:<11} {'HIT ' if matched else 'miss'} "
                  f"lp {logprob:+.3f}{'' if delta is None else f' (d {delta:+.3f})'} "
                  f"para {para_rate:.2f} | running match {hits / (i + 1):.2f} para {para_hits / (i + 1):.2f} "
                  f"| {generation[:50]!r}", flush=True)
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

    def locality(wave: int) -> None:
        scored = score_battery(battery, scored_battery_probe)
        summary = battery_summary(scored)
        for record in scored:
            emit({"phase": "battery", "wave": wave, "arm": mode, **record})
            if not record["correct"]:
                print(f"[{ts()}]  battery LOST {record['prompt']!r} -> {str(record['greedy_post'])[:40]!r} "
                      f"(dlp {float(record['logprob_delta']):+.3f})", flush=True)
        ppl = perplexity(model, heldout, chunk_len, "heldout ppl")
        emit({"phase": "locality", "wave": wave, "arm": mode, "ppl": ppl, "ppl_delta": ppl - base_ppl, **summary})
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

        # The fresh-state floor for this wave's facts, before they are anywhere
        # but the transcript -- what every later log-prob delta is measured against.
        print(f"[{ts()}] fresh-state floor, wave {wave} facts")
        fresh_baseline |= probe_facts([(wave, f) for f in facts], None, "floor", wave, False, None)

        carried = run_chunks(model, transcript, carried, chunk_len, f"wake {wave}", keep_logits=False)[1]
        seen += [(wave, f) for f in facts]

        # In-context control. On wave 2 this is the consumption price: B1 enters
        # selectively vacated, B2 full, A empty.
        print(f"[{ts()}] in-context control, wave {wave} facts (on the carried state)")
        probe_facts([(wave, f) for f in facts], carried, "in_context", wave, False, None)

        if mode == "no-sleep":
            print(f"\n[{ts()}] === wave {wave} sleep: none (--no-sleep floor) ===")
        else:
            print(f"\n[{ts()}] === wave {wave} sleep: {mode} ===")
            carried = run_sleep(mode, model, opt, args, wave, carried, transcript, seen, chunk_len,
                                encode, decode, tokenizer, user_open, asst_open, emit)

        print(f"\n[{ts()}] === wave {wave} probes: fresh state, no context ===")
        probe_facts(seen, None, "probe", wave, True, fresh_baseline)
        locality(wave)
        print(f"\n[{ts()}] === wave {wave} carried-state diagnostic ({ARM_CARRY[mode]}) -- NOT installation ===")
        if carried is None:
            print(f"[{ts()}]  arm {mode} carries nothing; column empty by construction")
        else:
            probe_facts(seen, carried, "carried", wave, False, None)

    out_file.close()
    print(f"\n[{ts()}] done -> {out_path}")


def run_sleep(mode, model, opt, args, wave, wake_state, transcript, seen, chunk_len,
              encode, decode, tokenizer, user_open, asst_open, emit):
    """One sleep. Returns the state carried into the next wake (None where the
    arm clears it)."""
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

    model.train()
    if mode == "sft-ref":
        distill_sft(model, opt, transcript, args.distill_steps, chunk_len, on_step)
        print()
        model.eval()
        return None

    seed = encode(f"{asst_open} {args.dream_prompt}".rstrip())
    banned = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    needles = [f.entity for _, f in seen] + [f.code for _, f in seen]

    def decode_token(token_id: int) -> str:
        return decode([token_id])

    # Each cue is the wake session's question plus the answer stem, stopping
    # before the code -- so the cue names which fact to recall and the state
    # still has to supply the digits.
    cues = [encode(f"{USER_CUE.format(user=user_open, entity=f.entity)}"
                   f"{asst_open} The code for the {f.entity} is")[0].tolist()
            for _, f in seen] if args.cue_every else []

    if mode == "drain-live":
        carried, dream = distill_live(model, opt, wake_state, seed, args.dream_tokens, args.dream_temp,
                                      banned, args.kl_temp, args.accum_window, decode_token, needles, on_step)
        print()
    else:
        model.eval()
        dream = teacher_dream(model, wake_state, seed, args.dream_tokens, args.dream_temp, banned,
                              drain=(mode == "drain"), decode_token=decode_token, needles=needles,
                              cues=cues, cue_every=args.cue_every, cue_greedy=args.cue_greedy)
        model.train()
        if mode == "replay":
            distill_replay(model, opt, dream, args.distill_steps, chunk_len, args.kl_temp, on_step)
            carried = None
        else:
            distill_stateful(model, opt, dream, wake_state, args.distill_steps, args.kl_temp,
                             args.accum_window, drain=(mode == "drain"), on_step=on_step)
            carried = dream.final_state if mode == "drain" else wake_state
        print()

    fraction, counts = rehearsal_fraction(dream.token_texts, needles)
    emit({"phase": "dream", "wave": wave, "arm": mode, "tokens": args.dream_tokens,
          "rehearsal_fraction": fraction, "needle_counts": counts, "skipped_cone": dream.skipped_cone,
          "temperature": args.dream_temp, "prompt": args.dream_prompt,
          "decoded": "".join(dream.token_texts)})
    print(f"[{ts()}] dream rehearsal fraction {fraction:.3f}  needle hits {counts}  "
          f"(near-cone erases skipped: {dream.skipped_cone})")
    if fraction == 0.0:
        print(f"[{ts()}] WARNING: the dream never rehearsed a fact -- reads never touched the bindings, so "
              f"nothing could distil. Seed the dream with --dream-prompt before reading anything into this arm.")
    print(f"[{ts()}] decoded dream:\n{''.join(dream.token_texts)!r}")

    model.eval()
    torch.cuda.empty_cache()
    return carried


if __name__ == "__main__":
    main()
