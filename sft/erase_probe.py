"""Erase-efficacy probe: does the rank-1 state erase `S <- S(I - g c c^T)`
actually make a primed Mamba2 SSM forget the targeted fact -- and only it?

This is the physics check for the dream-distillation sleep protocol
(notes/DISCUSSION-20260805-dream-distillation-cl-ab.md sec 3, step 2): the
protocol assumes that erasing the state along a fact's own read queries
degrades that binding while leaving the others intact. The algebra says so
for cleanly separable keys; a real state after a real transcript is where
the key geometry gets measured.

Shape of the run (pure inference, no training):

  prime    -- the consolidation-null transcript (N facts, digit-free filler)
              is run into the state once.
  baseline -- greedy recall + teacher-forced code log-prob for every fact
              from the primed state.
  capture  -- for each fact, teacher-force its probe question + answer one
              token at a time and record every layer's read query C at the
              answer-span positions (the reads that retrieve the code).
  erase    -- for each target fact and each --gammas value: copy the primed
              state, apply the rank-1 erase per layer at every captured
              query, then re-probe ALL facts from the edited state.

Expected if the protocol is sound: the target's recall/log-prob degrades
monotonically with gamma; off-target facts stay at baseline. Also reported:
the pairwise cosine overlap of the facts' mean read queries -- the direct
measure of whether the key geometry makes the erase surgical or blunt.

780M-specific: uses Model.c_capture (models/mamba2_780m/model.py). Priming
uses the model's normal dispatch (fused SSD kernel where available, the
per-token loop otherwise); capture drives tokens one at a time by design.
Runs on the local 8 GB box.

Usage (from sft/, env vars as in the Makefile):
    make erase-probe ARGS="--n-facts 4 --filler-tokens 40"
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation_null import (
    FILLER_SENTENCES,
    GEN_TOKENS,
    Fact,
    build_facts,
    build_turns,
    cue_rungs,
    digits,
    exact_match,
    extract_answer,
    generate,
    normalize,
    render_turns,
    report_transcript,
    run_chunks,
    target_logprob,
    ts,
)

T = TypeVar("T")

# A bystander whose baseline greedy answer misses and whose baseline log-prob
# sits below this never bound in the first place -- its post-erase delta is
# noise around a floor, so it is reported but kept out of the collateral means.
UNBOUND_LOGPROB = -2.0
FILLER_SPAN = 15

# Off-format bystanders: same transcript, different relation templates, no
# digits (the transcript's digit-free-filler invariant is what makes the code
# probes meaningful). Disjoint from consolidation_null's ENTITY_POOL.
PEOPLE = ["Alice", "Bertram", "Clara", "Dmitri", "Elena", "Farid", "Greta", "Hugo"]
CITIES = ["Paris", "Lisbon", "Oslo", "Nagoya", "Cairo", "Toronto", "Perth", "Dublin"]
EVENTS = ["meeting", "rehearsal", "inspection", "briefing", "handover", "audit"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OBJECTS = ["parcel", "crate", "canister", "satchel", "toolbox", "kettle"]
WEIGHTS = ["nine kilograms", "four kilograms", "seven kilograms", "twelve kilograms", "three kilograms", "six kilograms"]

# Near-cone bystanders: numeric answers in the same spaced-digit shape as a
# code, but a different relation -- the hard case for a rank-1 erase aimed at
# "the code for X is". Pools disjoint from PEOPLE above.
SENDERS = ["Ingrid", "Joaquin", "Kenji", "Lucia", "Marek", "Noor", "Ottoline", "Priya"]
FIRMS = ["Halverson", "Redmond", "Castellane", "Ophir", "Vantage", "Kirkwall", "Solvay", "Norbury"]


class Bystander:
    def __init__(self, label: str, question: str, statement: str, prompt: str, answer: str,
                 cls: str = "offformat") -> None:
        self.label, self.question, self.statement = label, question, statement
        self.prompt, self.answer, self.cls = prompt, answer, cls


def build_nearcone(n: int, rng: random.Random, user_open: str, asst_open: str) -> list[Bystander]:
    """n numeric-but-off-relation facts: spaced digits like a code, different
    relation. Digit counts (3, 4) differ from CODE_DIGITS so no answer can
    collide with a code as a full string."""
    senders, firms = rng.sample(SENDERS, len(SENDERS)), rng.sample(FIRMS, len(FIRMS))
    out = []
    for i in range(n):
        if i % 2 == 0:
            who, val = senders.pop(), " ".join(str(rng.randrange(10)) for _ in range(3))
            label, q = who, f"How much does the shipment from {who} weigh?"
            stmt, stem, ans = f"The shipment from {who} weighs {val} kilograms.", f"The shipment from {who} weighs", f" {val}"
        else:
            org, val = firms.pop(), " ".join(str(rng.randrange(10)) for _ in range(4))
            label, q = org, f"How much does the invoice from {org} total?"
            stmt, stem, ans = f"The invoice from {org} totals {val} dollars.", f"The invoice from {org} totals", f" {val}"
        out.append(Bystander(label, q, stmt, f"{user_open} {q}{asst_open} {stem}", ans, "nearcone"))
    return out


def build_sweep(entities: list[str]) -> list[tuple[str, str]]:
    """Fixed address-space sweep battery: (class, question). `unrelated` and
    `numeric` should sit far from any code-erase direction; `paraphrase` asks
    for the erased facts by other words and is *expected* to score high."""
    who = ["Alice", "Bertram", "Clara", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ingrid", "Joaquin"]
    where = ["Paris", "Lisbon", "Oslo", "Nagoya", "Cairo", "Toronto", "Perth", "Dublin", "Bergen", "Recife"]
    thing = ["kettle", "satchel", "lantern", "ledger", "trellis", "compass", "awning", "cistern", "mandolin", "spatula"]
    when = ["harvest", "regatta", "recital", "inspection", "handover", "vigil", "parade", "audit", "rehearsal", "banquet"]
    org = ["Halverson", "Redmond", "Castellane", "Ophir", "Vantage", "Kirkwall", "Solvay", "Norbury", "Perreault", "Aldington"]
    out: list[tuple[str, str]] = []
    for i in range(10):
        out += [("unrelated", q) for q in (
            f"Where does {who[i]} live?",
            f"What did {who[i]} say about the proposal?",
            f"Is {where[i]} warmer than the coast in autumn?",
            f"Who is responsible for the {when[i]}?",
            f"What colour is the {thing[i]}?",
            f"Why was the {when[i]} postponed?",
            f"Which company took over {org[i]}?",
            f"How would you describe the {thing[i]}?",
            f"What language is spoken in {where[i]}?",
            f"Did {who[i]} attend the {when[i]}?",
            f"What is the {thing[i]} made of?",
            f"Where is the nearest station to {where[i]}?",
            f"Who owns the {thing[i]}?",
            f"What season is best for visiting {where[i]}?",
        )]
    out += [("numeric", q) for i in range(10) for q in (
        f"How many people were at the {when[i]}?",
        f"What is the phone number for {org[i]}?",
        f"How much does the {thing[i]} weigh?",
    )]
    para = ["What is the access number for the {e}?", "What's the {e}'s code?", "Tell me the code for the {e}.",
            "Can you recall the {e} code?", "What code was assigned to the {e}?", "Remind me of the code for the {e}.",
            "The code for the {e} is what?", "Do you remember the {e}'s access number?"]
    out += [("paraphrase", t.format(e=e)) for e in entities for t in para]
    return out


def build_bystanders(n: int, rng: random.Random, user_open: str, asst_open: str) -> list[Bystander]:
    """n off-format facts, cycling three relation templates with fillers drawn
    without replacement. `stem` is the assistant prefix the probe leaks, so the
    answer log-prob is scored on the value alone."""
    pools = {k: rng.sample(v, len(v)) for k, v in
             (("people", PEOPLE), ("cities", CITIES), ("events", EVENTS),
              ("weekdays", WEEKDAYS), ("objects", OBJECTS), ("weights", WEIGHTS))}
    out = []
    for i in range(n):
        if i % 3 == 0:
            who, where = pools["people"].pop(), pools["cities"].pop()
            label, q, stmt, stem, ans = who, f"Where does {who} live?", f"{who} lives in {where}.", f"{who} lives in", f" {where}"
        elif i % 3 == 1:
            ev, day = pools["events"].pop(), pools["weekdays"].pop()
            label, q, stmt, stem, ans = ev, f"When is the {ev}?", f"The {ev} is on {day}.", f"The {ev} is on", f" {day}"
        else:
            obj, w = pools["objects"].pop(), pools["weights"].pop()
            label, q, stmt, stem, ans = obj, f"How much does the {obj} weigh?", f"The {obj} weighs {w}.", f"The {obj} weighs", f" {w}"
        out.append(Bystander(label, q, stmt, f"{user_open} {q}{asst_open} {stem}", ans))
    return out


def build_mixed_turns(items, filler_tokens, token_len, rng):
    """build_turns, but the item sequence may interleave off-format bystanders
    with the code facts. Identical rng consumption when there are none."""
    turns: list[tuple[str, str]] = []
    for i, item in enumerate(items):
        if i:
            used = 0
            while used < filler_tokens:
                for role in ("user", "assistant"):
                    sentence = rng.choice(FILLER_SENTENCES)
                    turns.append((role, sentence))
                    used += token_len(sentence)
        if isinstance(item, Fact):
            turns.extend([("user", f"What is the code for the {item.entity}?"),
                          ("assistant", f"The code for the {item.entity} is {item.code}.")])
        else:
            turns.extend([("user", item.question), ("assistant", item.statement)])
    return turns


def rank1_erase(ssm_state: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """S(I - g chat chat^T): attenuate the state's read along `c` by `gamma`,
    leave orthogonal reads untouched. A (near-)zero query is a no-op rather
    than a divide-by-zero. Computed in fp32, returned in the state's dtype.

    The direction is detached: gradient flows through the state's contents,
    never through the address the erase is aimed at -- a differentiable erase
    lets the optimizer rotate its queries to dodge consumption instead of
    installing facts (DISCUSSION-20260806 sec 3)."""
    import torch

    s32, c32 = ssm_state.float(), c.detach().float()
    norm = c32.norm(dim=-1, keepdim=True)
    if norm.max().item() < 1e-8:
        return ssm_state
    chat = c32 / norm.clamp_min(1e-8)
    read = torch.einsum("bhpn,bn->bhp", s32, chat)
    return (s32 - gamma * torch.einsum("bhp,bn->bhpn", read, chat)).to(ssm_state.dtype)


def group_by_layer(flat: list[T], n_layers: int) -> list[list[T]]:
    """Model.c_capture is token-major (layer 0..L-1 for token 0, then token
    1, ...); regroup into one list of per-layer queries per token."""
    if len(flat) % n_layers:
        raise ValueError(f"capture length {len(flat)} not divisible by n_layers {n_layers}")
    return [flat[i : i + n_layers] for i in range(0, len(flat), n_layers)]


def deflate(c: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove `c`'s components along the rows of `basis` (assumed orthonormal):
    (I - V^T V) c. Shapes: c (b, n), basis (k, n)."""
    import torch

    c, basis = c.detach(), basis.detach()
    coeffs = torch.einsum("bn,kn->bk", c.float(), basis.float())
    return c - torch.einsum("bk,kn->bn", coeffs, basis.float()).to(c.dtype)


def state_top_dirs(ssm_state: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right-singular directions of the state's address space -- the
    directions the stored keys share (the interference cone), computed from
    the state alone. Heads stacked so the result is per layer. Returns (k, n)."""
    import torch

    m = ssm_state[0].detach().float().reshape(-1, ssm_state.shape[-1])  # (h*p, n)
    _, _, vh = torch.linalg.svd(m, full_matrices=False)
    return vh[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-facts", type=int, default=4)
    parser.add_argument("--filler-tokens", type=int, default=40)
    parser.add_argument("--bystanders", type=int, default=5,
                        help="Off-format facts (different relation templates) added to the transcript and probed, "
                             "but never captured and never erased -- collateral outside the query cone (default: %(default)s)")
    parser.add_argument("--nearcone", type=int, default=2,
                        help="Numeric-but-off-relation bystanders (spaced digits, different relation) -- collateral "
                             "just outside the code cone (default: %(default)s)")
    parser.add_argument("--sweep", action="store_true", default=True, help="Run the address-space sweep battery")
    parser.add_argument("--no-sweep", dest="sweep", action="store_false")
    parser.add_argument("--filler-spans", type=int, default=3,
                        help="Filler continuation spans scored before/after each erase (default: %(default)s)")
    parser.add_argument("--gammas", default="0.25,0.5,1.0", help="Comma-separated erase strengths (default: %(default)s)")
    parser.add_argument("--span", choices=["answer", "question"], default="answer",
                        help="Which positions' read queries address the erase: the answer span alone, or question+answer (default: %(default)s)")
    parser.add_argument("--deflate", choices=["none", "state-svd", "others"], default="none",
                        help="Orthogonalize each erase direction before applying it: against the state's own top "
                             "singular directions (state-svd -- what a dream loop can compute), or against the other "
                             "facts' mean queries (others -- the oracle upper bound). Default: %(default)s")
    parser.add_argument("--deflate-k", type=int, default=1, help="Singular directions to deflate against for state-svd (default: %(default)s)")
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    parser.add_argument("--chunk-len", type=int, default=None, help="Priming chunk length (default: the model's DEFAULT_CHUNK_LEN)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="logs/erase_probe.jsonl")
    args = parser.parse_args()

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
    gammas = [float(g) for g in args.gammas.split(",")]
    print(f"[{ts()}] model {model_name} on {device}, chunk_len {chunk_len}, seed {args.seed}, span {args.span}, gammas {gammas}")

    model = model_mod.load_inference(device)
    if getattr(model, "c_capture", "missing") == "missing":
        raise SystemExit(f"model {model_name} has no c_capture hook -- this probe is for mamba2_780m")
    model.eval()
    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    stops = (".", "\n", user_open, asst_open)

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    facts = build_facts(args.n_facts, rng)
    bystanders = build_bystanders(args.bystanders, rng, user_open, asst_open)
    bystanders += build_nearcone(args.nearcone, rng, user_open, asst_open)
    def token_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    if bystanders:
        items = [*facts, *bystanders]
        rng.shuffle(items)
        turns = build_mixed_turns(items, args.filler_tokens, token_len, rng)
    else:
        turns = build_turns(facts, args.filler_tokens, token_len, rng)
    text = render_turns(turns, user_open, asst_open)
    transcript = encode(text)
    decoded = tokenizer.decode(transcript[0].cpu())
    report_transcript(text, decoded, facts, turns, transcript.shape[1])
    if bystanders:
        bad = sum(1 for b in bystanders if decoded.count(b.statement) != 1)
        collide = sum(1 for b in bystanders for f in facts if b.label == f.entity)
        digit_collide = sum(1 for b in bystanders for f in facts if digits(b.answer) == digits(f.code))
        print(f"[{ts()}]   bystanders: {len(bystanders)} "
              f"({sum(b.cls == 'offformat' for b in bystanders)} offformat, {sum(b.cls == 'nearcone' for b in bystanders)} nearcone)"
              f", statements not stated exactly once: {bad}  (must be 0)")
        print(f"[{ts()}]   bystander/fact label collisions: {collide}  (must be 0)")
        print(f"[{ts()}]   duplicate bystander labels: {len(bystanders) - len({b.label for b in bystanders})}  (must be 0)")
        print(f"[{ts()}]   bystander questions matching >1 statement: "
              f"{sum(1 for b in bystanders if sum(o.question == b.question for o in bystanders) > 1)}  (must be 0)")
        print(f"[{ts()}]   bystander/fact answer-digit collisions: {digit_collide}  (must be 0)")
        for cls in ("offformat", "nearcone"):
            first = next((b for b in bystanders if b.cls == cls), None)
            if first is None:
                continue
            at = decoded.find(first.statement)
            print(f"[{ts()}]   sample around first {cls} bystander ({first.label}):\n"
                  f"    ...{decoded[max(0, at - 200) : at + 200]!r}...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w")

    def emit(record: dict[str, object]) -> None:
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()

    unbound: set[str] = set()
    spans: list[int] = []

    def probe_all(state, label: str) -> list[dict[str, object]]:
        """Greedy recall + teacher-forced code log-prob for every fact from
        (copies of) `state`."""
        results = []
        for fact in facts:
            prompt = encode(cue_rungs(fact, user_open, asst_open)[0][0])
            target = encode(" " + fact.code)
            gen = tokenizer.decode(generate(model, prompt, copy.deepcopy(state), args.gen_tokens, 0.0)[0].cpu())
            lp = target_logprob(model, prompt, target, copy.deepcopy(state))
            results.append({"fact": fact.entity, "kind": "fact", "bclass": "fact", "code": fact.code, "greedy": gen,
                            "match": exact_match(gen, fact.code, stops), "logprob": lp})
            print(f"[{ts()}]   {label} {fact.entity:<11} {'HIT ' if results[-1]['match'] else 'miss'} "
                  f"lp {lp:+.3f}  {gen[:60]!r}", flush=True)
        for b in bystanders:
            prompt, target = encode(b.prompt), encode(b.answer)
            gen = tokenizer.decode(generate(model, prompt, copy.deepcopy(state), args.gen_tokens, 0.0)[0].cpu())
            lp = target_logprob(model, prompt, target, copy.deepcopy(state))
            ans, truth = extract_answer(gen, stops), normalize(b.answer)
            # The near-cone answer is the digits; greedy continues with the unit.
            match = ans == truth or ans.startswith(truth + " ")
            results.append({"fact": b.label, "kind": "bystander", "bclass": b.cls, "code": b.answer.strip(), "greedy": gen,
                            "match": match, "logprob": lp, "unbound": b.label in unbound})
            print(f"[{ts()}]   {label} {b.label:<11} [{b.cls[:4]}] {'HIT ' if match else 'miss'} "
                  f"lp {lp:+.3f}  {'[UNBOUND] ' if b.label in unbound else ''}{gen[:60]!r}", flush=True)
        return results

    def filler_logprobs(state) -> list[float]:
        return [target_logprob(model, transcript[:, p - FILLER_SPAN : p], transcript[:, p : p + FILLER_SPAN],
                               copy.deepcopy(state)) for p in spans]

    def capture_tokens(seq: torch.Tensor, state) -> list[list[torch.Tensor]]:
        """Drive `seq` one token at a time from a copy of `state`; return
        per-token per-layer read queries."""
        st = copy.deepcopy(state)
        model.c_capture = []
        with torch.no_grad():
            for t in range(seq.shape[1]):
                _, st = model(seq[:, t : t + 1], state=st)
        per_token = group_by_layer(model.c_capture, len(model.layers))
        model.c_capture = None
        return per_token

    def capture_span(prompt: torch.Tensor, target: torch.Tensor, state) -> list[list[torch.Tensor]]:
        seq = torch.cat([prompt, target], dim=1)
        per_token = capture_tokens(seq, state)
        # Positions whose *output* produces answer tokens: last prompt token
        # through second-to-last of the sequence.
        start = prompt.shape[1] - 1 if args.span == "answer" else 0
        return per_token[start : seq.shape[1] - 1]

    def capture_queries(fact) -> list[list[torch.Tensor]]:
        return capture_span(encode(cue_rungs(fact, user_open, asst_open)[0][0]), encode(" " + fact.code), primed)

    def mean_layers(per_token: list[list[torch.Tensor]]) -> torch.Tensor:
        """(L, d_state): each layer's mean read query over the captured span."""
        return torch.stack([torch.stack(layers).float() for layers in per_token]).mean(dim=0).squeeze(1)

    print(f"\n[{ts()}] === prime ===")
    _, primed = run_chunks(model, transcript, None, chunk_len, "prime", keep_logits=False)

    print(f"\n[{ts()}] === baseline (primed state, no erase) ===")
    baseline = probe_all(primed, "base")
    base_by_fact = {r["fact"]: r for r in baseline}
    unbound |= {r["fact"] for r in baseline
                if r["kind"] == "bystander" and not r["match"] and r["logprob"] < UNBOUND_LOGPROB}
    for r in baseline:
        emit({"phase": "baseline", **r, "unbound": r["fact"] in unbound})
    if bystanders:
        for cls in ("offformat", "nearcone"):
            members = [b for b in bystanders if b.cls == cls]
            if not members:
                continue
            bound = [b.label for b in members if b.label not in unbound]
            print(f"[{ts()}] {cls} bystanders bound at baseline: {len(bound)}/{len(members)}"
                  + (f"  unbound: {sorted(b.label for b in members if b.label in unbound)}" if len(bound) < len(members) else ""))
            emit({"phase": "bound_counts", "bclass": cls, "bound": len(bound), "total": len(members),
                  "unbound_labels": sorted(b.label for b in members if b.label in unbound)})
        nc = next((b for b in bystanders if b.cls == "nearcone"), None)
        if nc is not None:
            r = next(r for r in baseline if r["fact"] == nc.label)
            print(f"[{ts()}] nearcone sample: prompt {nc.prompt!r}\n"
                  f"[{ts()}]   truth {nc.answer!r}  baseline greedy {r['greedy']!r}  match {r['match']}  lp {r['logprob']:+.3f}")

    # Filler-continuation panel: teacher-forced continuation of transcript
    # spans that are pure filler (no digits, no role marker in the window), so
    # the erase's damage to non-fact state content is visible.
    ids = transcript[0].tolist()
    for _ in range(2000):
        if len(spans) >= args.filler_spans:
            break
        p = rng.randrange(FILLER_SPAN * 2, len(ids) - FILLER_SPAN)
        window = tokenizer.decode(ids[p - FILLER_SPAN : p + FILLER_SPAN])
        if any(ch.isdigit() for ch in window) or "code for" in window:
            continue
        if any(b.label in window for b in bystanders):
            continue
        if any(abs(p - q) < FILLER_SPAN * 2 for q in spans):
            continue
        spans.append(p)
    print(f"\n[{ts()}] === filler-continuation spans ({len(spans)} x {FILLER_SPAN} tokens) ===")
    base_filler = filler_logprobs(primed)
    for p, lp in zip(spans, base_filler, strict=True):
        print(f"[{ts()}]   @{p:<6} lp {lp:+.3f}  prompt {tokenizer.decode(ids[p - FILLER_SPAN : p])!r}"
              f" -> {tokenizer.decode(ids[p : p + FILLER_SPAN])!r}", flush=True)
        emit({"phase": "filler_baseline", "pos": p, "logprob": lp,
              "prompt": tokenizer.decode(ids[p - FILLER_SPAN : p]), "span": tokenizer.decode(ids[p : p + FILLER_SPAN])})

    print(f"\n[{ts()}] === capture read queries ({args.span} span) ===")
    queries = {}
    for fact in facts:
        queries[fact.entity] = capture_queries(fact)
        print(f"[{ts()}]   {fact.entity:<11} {len(queries[fact.entity])} positions x {len(model.layers)} layers")

    # Key geometry: pairwise cosine of each fact's mean read query, averaged
    # over layers -- the direct measure of erase bluntness.
    print(f"\n[{ts()}] === read-query overlap (mean |cos|, averaged over layers) ===")
    mean_q = {e: mean_layers(q) for e, q in queries.items()}
    names = [f.entity for f in facts]
    print(f"[{ts()}]   {'':<11} " + " ".join(f"{n:>9}" for n in names))
    overlaps = {}
    for a in names:
        row = []
        for b in names:
            cos = torch.nn.functional.cosine_similarity(mean_q[a], mean_q[b], dim=-1).abs().mean().item()
            row.append(cos)
            overlaps[f"{a}|{b}"] = cos
        print(f"[{ts()}]   {a:<11} " + " ".join(f"{v:9.3f}" for v in row))
    emit({"phase": "overlap", "span": args.span, "mean_abs_cos": overlaps})

    # Raw read queries share a large common-mode component (every pair sits near
    # |cos| 0.77), so the panel is reported both raw and with the state's top
    # singular direction removed from both sides -- the geometry `--deflate
    # state-svd` actually erases along.
    svd_basis = [state_top_dirs(primed.ssm_states[i], args.deflate_k) for i in range(len(model.layers))]

    def defl(v: torch.Tensor, i: int) -> torch.Tensor:
        return deflate(v.unsqueeze(0), svd_basis[i])[0]

    mean_qd = {e: torch.stack([defl(mean_q[e][i], i) for i in range(len(model.layers))]) for e in names}
    variants = (("raw", mean_q, lambda v, i: v), ("deflated", mean_qd, defl))

    def cos_vs_facts(q: torch.Tensor, mq, tf) -> list[tuple[float, str, int]]:
        """|cos| of a (L, n) query bundle against every fact's erase direction,
        per layer -- (cos, fact, layer) for all pairs."""
        return [(torch.nn.functional.cosine_similarity(tf(q[i], i), mq[e][i], dim=-1).abs().item(), e, i)
                for e in names for i in range(len(model.layers))]

    def quantile(xs: list[float], p: float) -> float:
        s = sorted(xs)
        return s[min(len(s) - 1, int(p * len(s)))]

    if bystanders:
        print(f"\n[{ts()}] === bystander query vs erase direction (|cos| over bystander x fact x layer) ===")
        bqs = {b.label: mean_layers(capture_span(encode(b.prompt), encode(b.answer), primed)) for b in bystanders}
        for vname, mq, tf in variants:
            pairs = [(c, b.label, b.cls, e, i) for b in bystanders for c, e, i in cos_vs_facts(bqs[b.label], mq, tf)]
            for cls in ("offformat", "nearcone"):
                sub = [p for p in pairs if p[2] == cls]
                if not sub:
                    continue
                cs = [p[0] for p in sub]
                top = max(sub)
                print(f"[{ts()}]   {vname:<9} {cls:<10} n={len(cs)}  mean {sum(cs) / len(cs):.3f}  p95 {quantile(cs, 0.95):.3f}  "
                      f"max {top[0]:.3f} ({top[1]} vs {top[3]}, layer {top[4]})", flush=True)
                emit({"phase": "cos_bystander", "variant": vname, "bclass": cls, "n": len(cs), "mean": sum(cs) / len(cs),
                      "p95": quantile(cs, 0.95), "max": top[0], "max_label": top[1], "max_fact": top[3], "max_layer": top[4]})

    if args.sweep:
        battery = build_sweep(names)
        print(f"\n[{ts()}] === address-space sweep battery ({len(battery)} questions, fresh state) ===")
        hits: dict[str, list[tuple[float, str, str, str, int]]] = {v[0]: [] for v in variants}
        for n_done, (cls, q) in enumerate(battery, 1):
            last = capture_tokens(encode(f"{user_open} {q}{asst_open}"), None)[-1]
            qv = torch.stack(last).float().squeeze(1)  # (L, n)
            for vname, mq, tf in variants:
                c, e, i = max(cos_vs_facts(qv, mq, tf))
                hits[vname].append((c, cls, q, e, i))
            if n_done % 20 == 0 or n_done == len(battery):
                print(f"\r[{ts()}]   swept {n_done}/{len(battery)}  running max |cos| raw {max(h[0] for h in hits['raw']):.3f}"
                      f" deflated {max(h[0] for h in hits['deflated']):.3f}",
                      end="" if n_done < len(battery) else "\n", flush=True)
        for vname, _, _ in variants:
            for cls in ("unrelated", "numeric", "paraphrase"):
                sub = sorted((h for h in hits[vname] if h[1] == cls), reverse=True)
                cs = [h[0] for h in sub]
                fracs = {t: sum(c > t for c in cs) / len(cs) for t in (0.3, 0.5, 0.7, 0.9)}
                print(f"[{ts()}]   {vname:<9} {cls:<10} n={len(cs)}  mean {sum(cs) / len(cs):.3f}  max {cs[0]:.3f}  "
                      + " ".join(f">{t}: {f:.3f}" for t, f in fracs.items()))
                for c, _, q, e, i in sub[:10]:
                    print(f"[{ts()}]       {c:.3f}  L{i:<3} vs {e:<10} {q!r}")
                emit({"phase": "cos_sweep", "variant": vname, "class": cls, "n": len(cs), "mean": sum(cs) / len(cs),
                      "max": cs[0], "fracs": {str(t): f for t, f in fracs.items()},
                      "top10": [{"cos": c, "question": q, "fact": e, "layer": i} for c, _, q, e, i in sub[:10]]})

    # "others" deflation basis: per layer, the other facts' mean queries,
    # orthonormalized. The oracle -- a dream loop cannot know these.
    others_basis: dict[str, list[torch.Tensor]] = {}
    if args.deflate == "others":
        for fact in facts:
            per_layer_basis = []
            for i in range(len(model.layers)):
                stack = torch.stack([mean_q[o.entity][i] for o in facts if o.entity != fact.entity])
                q_mat, _ = torch.linalg.qr(stack.float().T)
                per_layer_basis.append(q_mat.T)  # (k, n) orthonormal rows
            others_basis[fact.entity] = per_layer_basis

    print(f"\n[{ts()}] === erase sweep (deflate={args.deflate}) ===")
    skipped = 0
    for fact in facts:
        for gamma in gammas:
            edited = copy.deepcopy(primed)
            before_norms = [s.float().norm().item() for s in edited.ssm_states]
            resid, cosv = [], []
            for per_layer in queries[fact.entity]:
                for i, c in enumerate(per_layer):
                    raw = c
                    if args.deflate == "state-svd":
                        basis = state_top_dirs(edited.ssm_states[i], args.deflate_k)
                        c = deflate(c, basis)
                    elif args.deflate == "others":
                        c = deflate(c, others_basis[fact.entity][i])
                    if args.deflate != "none":
                        raw_n = raw.float().norm().item()
                        resid.append(c.float().norm().item() / max(raw_n, 1e-8))
                        cosv.append((raw.float() - c.float()).norm().item() / max(raw_n, 1e-8))
                    # A query living almost entirely in the deflated cone has no
                    # reliable discriminative direction left -- skip, don't erase noise.
                    if args.deflate != "none" and c.float().norm() < 0.05 * per_layer[i].float().norm():
                        skipped += 1
                        continue
                    edited.ssm_states[i] = rank1_erase(edited.ssm_states[i], c, gamma)
            after_norms = [s.float().norm().item() for s in edited.ssm_states]
            frob_frac = sum(1 - a / max(b, 1e-8) for a, b in zip(after_norms, before_norms, strict=True)) / len(before_norms)
            print(f"\n[{ts()}] erase target={fact.entity} gamma={gamma}  "
                  f"Frobenius removed {frob_frac:.4f} (mean over layers)"
                  + (f"  resid |c_def|/|c| mean {sum(resid) / len(resid):.4f} min {min(resid):.4f}  "
                     f"|cos(c,v)| mean {sum(cosv) / len(cosv):.4f} max {max(cosv):.4f}" if resid else ""))
            emit({"phase": "diagnostics", "erased": fact.entity, "gamma": gamma, "deflate": args.deflate,
                  "frob_fraction_removed": frob_frac, "n_directions": len(resid),
                  "resid_mean": sum(resid) / len(resid) if resid else None,
                  "resid_min": min(resid) if resid else None,
                  "cos_mean": sum(cosv) / len(cosv) if cosv else None,
                  "cos_max": max(cosv) if cosv else None})
            results = probe_all(edited, f"g={gamma}")
            for r in results:
                base = base_by_fact[r["fact"]]
                emit({
                    "phase": "erase", "erased": fact.entity, "gamma": gamma, "span": args.span,
                    "deflate": args.deflate, **r,
                    "baseline_match": base["match"], "baseline_logprob": base["logprob"],
                    "logprob_delta": r["logprob"] - base["logprob"],
                    "is_target": r["fact"] == fact.entity,
                })
            edited_filler = filler_logprobs(edited)
            for p, lp, b in zip(spans, edited_filler, base_filler, strict=True):
                emit({"phase": "filler", "erased": fact.entity, "gamma": gamma, "deflate": args.deflate,
                      "pos": p, "logprob": lp, "baseline_logprob": b, "logprob_delta": lp - b})
            if spans:
                print(f"[{ts()}]   filler dlogp " + " ".join(f"{lp - b:+.3f}" for lp, b in zip(edited_filler, base_filler, strict=True))
                      + f"  mean {sum(lp - b for lp, b in zip(edited_filler, base_filler, strict=True)) / len(spans):+.3f}", flush=True)
    if args.deflate != "none":
        print(f"[{ts()}] deflation skipped {skipped} near-cone erase directions")

    print(f"\n[{ts()}] === summary (logprob delta vs baseline, mean over runs) ===")
    out_file.flush()
    all_records = [json.loads(line) for line in out_path.open()]
    records = [r for r in all_records if r["phase"] == "erase"]
    fillers = [r for r in all_records if r["phase"] == "filler"]
    diags = [r for r in all_records if r["phase"] == "diagnostics"]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def worst(rs: list[dict], ident) -> str:
        if not rs:
            return "n/a"
        r = min(rs, key=lambda r: r["logprob_delta"])
        return f"{r['logprob_delta']:+7.3f} [{ident(r)}]"

    for gamma in gammas:
        at = [r for r in records if r["gamma"] == gamma]
        tgt = [r for r in at if r["is_target"]]
        sib = [r for r in at if not r["is_target"] and r["kind"] == "fact"]
        groups = {cls: [r for r in at if r.get("bclass") == cls and not r["unbound"]] for cls in ("offformat", "nearcone")}
        byu = [r["logprob_delta"] for r in at if r["kind"] == "bystander" and r["unbound"]]
        tgt_flips = sum(1 for r in at if r["is_target"] and r["baseline_match"] and not r["match"])
        off_flips = sum(1 for r in at if not r["is_target"] and r["baseline_match"] and not r["match"])
        fil = [r for r in fillers if r["gamma"] == gamma]
        dg = [r for r in diags if r["gamma"] == gamma]
        ident = lambda r: f"{r['fact']} | erased {r['erased']}"
        print(f"[{ts()}]   gamma {gamma:<5} target dlogp {mean([r['logprob_delta'] for r in tgt]):+7.3f} ({tgt_flips} flips)"
              f"   worst {worst(tgt, ident)}")
        print(f"[{ts()}]            sibling  dlogp {mean([r['logprob_delta'] for r in sib]):+7.3f} ({off_flips} off-target flips)"
              f"   worst {worst(sib, ident)}")
        for cls, rs in groups.items():
            print(f"[{ts()}]            {cls:<10} (bound, n={len(rs)}) {mean([r['logprob_delta'] for r in rs]):+7.3f}"
                  f"   worst {worst(rs, ident)}")
        print(f"[{ts()}]            bystander(unbound, n={len(byu)}) {mean(byu):+7.3f}")
        print(f"[{ts()}]            filler   dlogp {mean([r['logprob_delta'] for r in fil]):+7.3f}"
              f"   worst {worst(fil, lambda r: f'@{r['pos']} | erased {r['erased']}')}")
        print(f"[{ts()}]            Frobenius removed {mean([d['frob_fraction_removed'] for d in dg]):.4f}"
              + (f"   resid mean {mean([d['resid_mean'] for d in dg]):.4f} min {min(d['resid_min'] for d in dg):.4f}"
                 f"   |cos| mean {mean([d['cos_mean'] for d in dg]):.4f}" if args.deflate != "none" else ""))
    out_file.close()
    print(f"[{ts()}] -> {out_path}")


if __name__ == "__main__":
    main()
