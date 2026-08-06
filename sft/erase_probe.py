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
    GEN_TOKENS,
    build_facts,
    build_turns,
    cue_rungs,
    exact_match,
    generate,
    render_turns,
    report_transcript,
    run_chunks,
    target_logprob,
    ts,
)

T = TypeVar("T")


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
    turns = build_turns(facts, args.filler_tokens, lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"]), rng)
    text = render_turns(turns, user_open, asst_open)
    transcript = encode(text)
    report_transcript(text, tokenizer.decode(transcript[0].cpu()), facts, turns, transcript.shape[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w")

    def emit(record: dict[str, object]) -> None:
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()

    def probe_all(state, label: str) -> list[dict[str, object]]:
        """Greedy recall + teacher-forced code log-prob for every fact from
        (copies of) `state`."""
        results = []
        for fact in facts:
            prompt = encode(cue_rungs(fact, user_open, asst_open)[0][0])
            target = encode(" " + fact.code)
            gen = tokenizer.decode(generate(model, prompt, copy.deepcopy(state), args.gen_tokens, 0.0)[0].cpu())
            lp = target_logprob(model, prompt, target, copy.deepcopy(state))
            results.append({"fact": fact.entity, "code": fact.code, "greedy": gen,
                            "match": exact_match(gen, fact.code, stops), "logprob": lp})
            print(f"[{ts()}]   {label} {fact.entity:<11} {'HIT ' if results[-1]['match'] else 'miss'} "
                  f"lp {lp:+.3f}  {gen[:60]!r}", flush=True)
        return results

    def capture_queries(fact) -> list[list[torch.Tensor]]:
        """Teacher-force the fact's probe prompt + true answer one token at a
        time from a copy of the primed state; return per-token per-layer read
        queries over the chosen span."""
        prompt = encode(cue_rungs(fact, user_open, asst_open)[0][0])
        target = encode(" " + fact.code)
        seq = torch.cat([prompt, target], dim=1)
        state = copy.deepcopy(primed)
        model.c_capture = []
        with torch.no_grad():
            for t in range(seq.shape[1]):
                _, state = model(seq[:, t : t + 1], state=state)
        per_token = group_by_layer(model.c_capture, len(model.layers))
        model.c_capture = None
        # Positions whose *output* produces answer tokens: last prompt token
        # through second-to-last of the sequence.
        start = prompt.shape[1] - 1 if args.span == "answer" else 0
        return per_token[start : seq.shape[1] - 1]

    print(f"\n[{ts()}] === prime ===")
    _, primed = run_chunks(model, transcript, None, chunk_len, "prime", keep_logits=False)

    print(f"\n[{ts()}] === baseline (primed state, no erase) ===")
    baseline = probe_all(primed, "base")
    base_by_fact = {r["fact"]: r for r in baseline}
    for r in baseline:
        emit({"phase": "baseline", **r})

    print(f"\n[{ts()}] === capture read queries ({args.span} span) ===")
    queries = {}
    for fact in facts:
        queries[fact.entity] = capture_queries(fact)
        print(f"[{ts()}]   {fact.entity:<11} {len(queries[fact.entity])} positions x {len(model.layers)} layers")

    # Key geometry: pairwise cosine of each fact's mean read query, averaged
    # over layers -- the direct measure of erase bluntness.
    print(f"\n[{ts()}] === read-query overlap (mean |cos|, averaged over layers) ===")
    mean_q = {
        e: torch.stack([torch.stack(layers).float() for layers in q]).mean(dim=0)  # (L, 1, d_state)
        for e, q in queries.items()
    }
    names = [f.entity for f in facts]
    print(f"[{ts()}]   {'':<11} " + " ".join(f"{n:>9}" for n in names))
    overlaps = {}
    for a in names:
        row = []
        for b in names:
            qa, qb = mean_q[a].squeeze(1), mean_q[b].squeeze(1)
            cos = torch.nn.functional.cosine_similarity(qa, qb, dim=-1).abs().mean().item()
            row.append(cos)
            overlaps[f"{a}|{b}"] = cos
        print(f"[{ts()}]   {a:<11} " + " ".join(f"{v:9.3f}" for v in row))
    emit({"phase": "overlap", "span": args.span, "mean_abs_cos": overlaps})

    # "others" deflation basis: per layer, the other facts' mean queries,
    # orthonormalized. The oracle -- a dream loop cannot know these.
    others_basis: dict[str, list[torch.Tensor]] = {}
    if args.deflate == "others":
        for fact in facts:
            per_layer_basis = []
            for i in range(len(model.layers)):
                stack = torch.stack([mean_q[o.entity][i, 0] for o in facts if o.entity != fact.entity])
                q_mat, _ = torch.linalg.qr(stack.float().T)
                per_layer_basis.append(q_mat.T)  # (k, n) orthonormal rows
            others_basis[fact.entity] = per_layer_basis

    print(f"\n[{ts()}] === erase sweep (deflate={args.deflate}) ===")
    skipped = 0
    for fact in facts:
        for gamma in gammas:
            edited = copy.deepcopy(primed)
            for per_layer in queries[fact.entity]:
                for i, c in enumerate(per_layer):
                    if args.deflate == "state-svd":
                        basis = state_top_dirs(edited.ssm_states[i], args.deflate_k)
                        c = deflate(c, basis)
                    elif args.deflate == "others":
                        c = deflate(c, others_basis[fact.entity][i])
                    # A query living almost entirely in the deflated cone has no
                    # reliable discriminative direction left -- skip, don't erase noise.
                    if args.deflate != "none" and c.float().norm() < 0.05 * per_layer[i].float().norm():
                        skipped += 1
                        continue
                    edited.ssm_states[i] = rank1_erase(edited.ssm_states[i], c, gamma)
            print(f"\n[{ts()}] erase target={fact.entity} gamma={gamma}")
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
    if args.deflate != "none":
        print(f"[{ts()}] deflation skipped {skipped} near-cone erase directions")

    print(f"\n[{ts()}] === summary (logprob delta vs baseline, mean over runs) ===")
    records = [json.loads(line) for line in out_path.open() if '"phase": "erase"' in line]
    for gamma in gammas:
        tgt = [r["logprob_delta"] for r in records if r["gamma"] == gamma and r["is_target"]]
        off = [r["logprob_delta"] for r in records if r["gamma"] == gamma and not r["is_target"]]
        tgt_flips = sum(1 for r in records if r["gamma"] == gamma and r["is_target"] and r["baseline_match"] and not r["match"])
        off_flips = sum(1 for r in records if r["gamma"] == gamma and not r["is_target"] and r["baseline_match"] and not r["match"])
        print(f"[{ts()}]   gamma {gamma:<5} target dlogp {sum(tgt) / len(tgt):+7.3f} ({tgt_flips} flips)   "
              f"off-target dlogp {sum(off) / len(off):+7.3f} ({off_flips} flips)")
    out_file.close()
    print(f"[{ts()}] -> {out_path}")


if __name__ == "__main__":
    main()
