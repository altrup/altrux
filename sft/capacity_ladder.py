"""In-context capacity ladder: how many competing facts can the SSM state
actually hold, and is the limit fact *count* or transcript *distance*?

Motivation. `consolidation_null.py`'s in-context positive control -- the check
that the transcript is held losslessly before asking whether distillation can
move it into weights -- measured 1.000 at 4 facts and 0.03 at 40. That gap
invalidates the null at 40 facts (the teacher does not know the facts, so
distillation has nothing to install) and leaves two variables confounded: the
two configurations differ in both fact count and transcript length.

This script primes on a transcript and runs only the in-context probe, over a
grid of (n_facts, filler_tokens). It answers two things:

  1. The largest n_facts whose control still clears CONTROL_THRESHOLD -- the
     N at which the consolidation null must be run to have a valid premise.
  2. Whether the limit is capacity or distance: holding n_facts fixed while
     stretching filler separates state *erosion over tokens* from
     *interference between competing bindings*.

No distillation, no LoRA training -- prime plus one greedy probe per fact, so
it costs a single forward pass over each transcript.

Usage (from sft/, env as in the Makefile):
    make capacity-ladder
    make capacity-ladder ARGS="--grid 4x200,8x200,16x200 --seed 1234"
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.facts import (  # noqa: E402
    build_facts,
    build_turns,
    cue_rungs,
    exact_match,
    render_turns,
    role_adjacency_violations,
)
from experiments.inference import generate, run_chunks  # noqa: E402
from consolidation_null import ts  # noqa: E402

# The consolidation null's own gate: below this the transcript is not held
# losslessly and any consolidation verdict measures the wrong thing.
CONTROL_THRESHOLD = 0.8

# n_facts x filler_tokens. The fixed-filler sweep gives the capacity curve;
# 40x40 (many facts, short transcript) and 4x800 (few facts, long transcript)
# are the two cells that separate count from distance.
DEFAULT_GRID = "4x200,8x200,16x200,24x200,40x40,4x800"


def parse_grid(spec: str) -> list[tuple[int, int]]:
    cells = []
    for part in spec.split(","):
        n, _, filler = part.strip().partition("x")
        cells.append((int(n), int(filler)))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", default=DEFAULT_GRID, help="n_facts x filler_tokens cells (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=None)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="logs/capacity_ladder.jsonl")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="disable the neural-memory injection, measuring the backbone alone",
    )
    args = parser.parse_args()

    import importlib
    import os
    import random

    import torch
    from dotenv import load_dotenv

    load_dotenv()
    model_name = os.getenv("MODEL_NAME", "mamba2_780m")
    model_mod = importlib.import_module(f"models.{model_name}")
    train_hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    from models.common import build_tokenizer, set_memory_injection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk_len = args.chunk_len or getattr(train_hooks, "DEFAULT_CHUNK_LEN", 48)
    grid = parse_grid(args.grid)
    print(f"[{ts()}] model {model_name} on {device}, chunk_len {chunk_len}, seed {args.seed}")
    print(f"[{ts()}] grid: {', '.join(f'{n}x{f}' for n, f in grid)}")

    # Rank/alpha are irrelevant -- lora_B is zero-init, so the adapters are
    # identity and this measures the base model. setup_training is just the
    # shared loader.
    model, _ = train_hooks.setup_training(device, 16, 32.0, 0.0)
    model.eval()
    has_memory = set_memory_injection(model, not args.no_memory)
    print(f"[{ts()}] memory injection: {'absent' if not has_memory else ('off' if args.no_memory else 'on')}")
    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    stops = (".", "\n", user_open, asst_open)

    def encode(text: str) -> torch.Tensor:
        return torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]], dtype=torch.long, device=device)

    def token_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w")

    results: list[dict[str, object]] = []
    for n_facts, filler in grid:
        rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        facts = build_facts(n_facts, rng)
        turns = build_turns(facts, filler, token_len, rng)
        transcript = encode(render_turns(turns, user_open, asst_open))
        n_tokens = transcript.shape[1]
        violations = role_adjacency_violations(turns)
        print(f"\n[{ts()}] === {n_facts} facts x {filler} filler -> {n_tokens} tokens "
              f"(role violations {violations}, must be 0) ===")

        _, primed = run_chunks(model, transcript, None, chunk_len, "prime", keep_logits=False)

        n_hit = 0
        # Position within the transcript matters for the distance question:
        # a recency-limited state should hit late facts and miss early ones.
        hit_positions: list[int] = []
        for i, fact in enumerate(facts):
            prompt = encode(cue_rungs(fact, user_open, asst_open)[0][0])
            gen = tokenizer.decode(generate(model, prompt, copy.deepcopy(primed), args.gen_tokens, 0.0)[0].cpu())
            hit = exact_match(gen, fact.code, stops)
            n_hit += hit
            if hit:
                hit_positions.append(i)
            out_file.write(json.dumps({
                "n_facts": n_facts, "filler_tokens": filler, "transcript_tokens": n_tokens,
                "index": i, "fact": fact.entity, "code": fact.code, "gen": gen, "match": hit,
            }) + "\n")
            out_file.flush()
            print(f"\r[{ts()}]  probe {i + 1}/{n_facts} {fact.entity:<11} "
                  f"{'HIT ' if hit else 'miss'}  running {n_hit / (i + 1):.2f}", end="", flush=True)
        print()

        rate = n_hit / n_facts
        # Late-half hit rate: if the state is recency-limited rather than
        # capacity-limited, this stays high while the overall rate falls.
        late = [p for p in hit_positions if p >= n_facts / 2]
        late_rate = len(late) / max(1, n_facts - n_facts // 2)
        verdict = "OK" if rate >= CONTROL_THRESHOLD else "BELOW CONTROL THRESHOLD"
        print(f"[{ts()}] {n_facts}x{filler}: in-context {rate:.3f} ({n_hit}/{n_facts})  "
              f"late-half {late_rate:.3f}  hit indices {hit_positions}  -> {verdict}")
        record = {
            "n_facts": n_facts, "filler_tokens": filler, "transcript_tokens": n_tokens,
            "in_context_rate": rate, "late_half_rate": late_rate,
            "hit_indices": hit_positions, "threshold": CONTROL_THRESHOLD, "phase": "cell",
            "memory_injection": has_memory and not args.no_memory,
        }
        results.append(record)
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        del primed
        torch.cuda.empty_cache()

    out_file.close()
    print(f"\n[{ts()}] === ladder summary ===")
    print(f"[{ts()}]   {'cell':<12} {'tokens':>7}  {'rate':>6}  {'late':>6}")
    for r in results:
        cell = f"{r['n_facts']}x{r['filler_tokens']}"
        print(f"[{ts()}]   {cell:<12} {r['transcript_tokens']:>7}  "
              f"{float(r['in_context_rate']):>6.3f}  {float(r['late_half_rate']):>6.3f}")
    passing = [r for r in results if float(r["in_context_rate"]) >= CONTROL_THRESHOLD]
    if passing:
        best = max(passing, key=lambda r: int(r["n_facts"]))
        print(f"[{ts()}] largest n_facts clearing {CONTROL_THRESHOLD}: "
              f"{best['n_facts']} (filler {best['filler_tokens']}) -- run the null there")
    else:
        print(f"[{ts()}] NO cell cleared {CONTROL_THRESHOLD} -- the null has no valid operating point on this grid")
    print(f"[{ts()}] -> {out_path}")


if __name__ == "__main__":
    main()
