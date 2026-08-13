"""Natural-language fact-correction probe: does the neural memory carry a
conversational correction ("no, the president is X now") across a sleep?

A short conversation asserts a stale fact, the user corrects it, a filler
exchange pushes the correction's window past a write boundary, then the
backbone is wiped (sleep) and the question is re-asked. Candidate answers
are scored teacher-forced (mean logprob per answer token) -- free generation
is deliberately not used (window-1 serving degenerates on question-shaped
prompts; see EXPERIMENT_NOTES-20260723). Conditions:

  no-sleep      -- SSM intact, memory on: the in-context ceiling.
  sleep-intact  -- SSM wiped, memory kept: the memory test.
  sleep-none    -- SSM wiped, injections disabled: the plain-backbone floor.

plus a no-context parametric baseline. Scenario A corrects to the name the
parametric prior already favors (a degenerate anchor -- Pile-era backbones
answer "Donald Trump"); scenario B corrects to a counterfactual name, so
the signal is tracking-the-correction, not matching-the-prior. The verdict
is sleep-intact vs sleep-none on the corrected name.

Turn formatting goes through prepare_data.format_conversation so the probe
cannot drift from the training format (user turns end with "\\n", assistant
turns with EOS).

  make probe-correction ARGS="--checkpoint ../models/<name>/checkpoints/<dir>/step-N"
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path

import torch

FILLER_ANSWER = (
    "Golden leaves drift down, carried on a cooling wind, and autumn settles in."
    " The trees let go of their summer colors one by one, and the paths fill with"
    " soft red and yellow drifts that rustle underfoot. The days grow shorter, the"
    " evenings turn crisp, and the whole town seems to slow down a little. Markets"
    " fill with pumpkins and apples, kitchens smell of cinnamon, and everyone digs"
    " their sweaters out of storage. It is a season of winding down and gathering"
    " in, of long shadows in the afternoon and early lamplight, a quiet pause"
    " before winter arrives with its snow and silence."
)

QUESTION = "Who is the current president?"
ANSWER_STEM = "The current president is"
NAMES = ["Donald Trump", "Joe Biden", "Kamala Harris", "Barack Obama"]
SCENARIOS = [
    ("A: corrected->Trump", "Donald Trump", "Joe Biden"),
    ("B: corrected->Harris", "Kamala Harris", "Joe Biden"),
]


def build_prefix(tokenizer, markers: tuple[str, str], corrected: str, wrong: str, device, filler: bool = True) -> torch.Tensor:
    import probe_recall as pr
    from preparation.conversations import format_conversation

    messages = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": f"{ANSWER_STEM} {wrong}."},
        {"role": "user", "content": f"No, that is outdated. {corrected} won the most"
         f" recent election. The current president is {corrected}, not {wrong}."},
        {"role": "assistant", "content": f"Thanks for the correction. I will remember that"
         f" the current president is {corrected} now, and that {wrong} is no longer the president."},
    ]
    # The filler both pushes the correction past a write boundary and is the
    # likeliest thing to evict it under the delta rule -- run both arms to
    # tell "never written" apart from "written then overwritten".
    if filler:
        messages += [
            {"role": "user", "content": "While I have you, could you also write me a short piece about autumn?"},
            {"role": "assistant", "content": FILLER_ANSWER},
        ]
    ids, _, _ = format_conversation(messages, tokenizer, 1 << 30, *markers)
    keep = len(ids) - len(ids) % pr.CHUNK_LEN  # truncate, don't pad: pad ids would pollute the state
    return torch.tensor(ids[:keep], device=device).unsqueeze(0)


def build_query(tokenizer, markers: tuple[str, str], device) -> torch.Tensor:
    from preparation.conversations import format_conversation

    ids, _, _ = format_conversation([{"role": "user", "content": QUESTION}], tokenizer, 1 << 30, *markers)
    ids += tokenizer.encode(f"{markers[1]} {ANSWER_STEM}", add_special_tokens=False)
    return torch.tensor(ids, device=device).unsqueeze(0)


def main() -> None:
    import probe_recall as pr
    from models.common import build_tokenizer
    from training.checkpoints import latest_checkpoint, load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=None, help="Checkpoint step dir to probe (default: latest)")
    parser.add_argument("--chunk-len", type=int, default=24, help="Forward chunk length (8 fits 2.7B on an 8 GB card)")
    parser.add_argument("--memory-window", type=int, default=8)
    parser.add_argument("--no-filler", action="store_true",
                        help="Sleep immediately after the correction, with no intervening filler turn")
    args = parser.parse_args()

    pr.CHUNK_LEN = args.chunk_len

    model_name = os.getenv("MODEL_NAME", "mamba2_2_7b_memory")
    model_mod = importlib.import_module(f"models.{model_name}")
    mmod = importlib.import_module(f"models.{model_name}.model")
    train_hooks = importlib.import_module(f"models.{model_name}.train_hooks")

    # Inference-only: write()'s create_graph=True second-order graph is pure
    # overhead here (probe_recall does the same). skip_writes additionally
    # freezes M during scoring passes, so every candidate is scored against
    # the memory exactly as the prefix left it.
    orig_write = mmod._NeuralMemory.write
    skip_writes = {"on": False}

    def patched_write(self, ks, vs, etas, thetas, alphas, create_graph=True):
        if skip_writes["on"]:
            return (torch.zeros(ks.shape[0], ks.shape[1], device=ks.device, dtype=ks.dtype),
                    torch.zeros(ks.shape[1], device=ks.device, dtype=ks.dtype))
        return orig_write(self, ks, vs, etas, thetas, alphas, create_graph=False)

    mmod._NeuralMemory.write = patched_write

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(__file__).parents[2] / "models" / model_name / "checkpoints"
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(ckpt_dir)
    if ckpt is None or not ckpt.is_dir():
        raise SystemExit(f"not a checkpoint dir: {ckpt}")
    print(f"probing checkpoint: {ckpt}")

    lora_cfg = json.loads((ckpt / "lora_config.json").read_text())
    model, _ = train_hooks.setup_training(device, lora_cfg["rank"], lora_cfg["alpha"], 0.0)
    load_checkpoint(model, ckpt)
    model.eval()
    model.set_memory_window(args.memory_window)
    tokenizer = build_tokenizer(model_mod)
    markers = (model_mod.USER_OPEN, model_mod.ASST_OPEN)
    query = build_query(tokenizer, markers, device)

    # One resident MemoryState at a time (an 8 GB card fits exactly one at
    # 2.7B): each scoring pass recomputes the short prefix instead of cloning.
    def fresh_state(prefix, condition: str):
        skip_writes["on"] = False
        if prefix is None:
            return None
        _, state = pr.run_chunks(model, prefix, None, "prefix", keep_logits=False)
        if condition.startswith("sleep"):
            model.sleep_slot(state, 0)
        return state

    def score_names(prefix, condition: str) -> dict[str, float]:
        out = {}
        for name in NAMES:
            # Retry non-finite passes: unreliable-GPU insurance (the local
            # gfx1102 card corrupts individual passes), free elsewhere.
            for attempt in range(3):
                state = fresh_state(prefix, condition)
                skip_writes["on"] = True
                model.injection_enabled = condition not in ("sleep-none", "parametric")
                s = pr.score_targets(model, query, torch.tensor(
                    tokenizer.encode(f" {name}.", add_special_tokens=False), device=device).unsqueeze(0),
                    state, f"{condition}:{name}")
                model.injection_enabled = True
                del state
                torch.cuda.empty_cache()
                if math.isfinite(s.item()):
                    break
                print(f"    [{condition}] {name}: non-finite, retrying", flush=True)
            out[name] = round(s.item(), 3)
            print(f"    [{condition}] {name}: {out[name]:+.3f}", flush=True)
        return out

    def topk_after_query(prefix, condition: str, k: int = 8) -> list[tuple[str, float]]:
        state = fresh_state(prefix, condition)
        skip_writes["on"] = True
        model.injection_enabled = condition not in ("sleep-none", "parametric")
        logits, _ = pr.run_chunks(model, query, state, f"{condition}:topk")
        model.injection_enabled = True
        last = logits[0, query.shape[1] - 1].float()
        del state
        torch.cuda.empty_cache()
        if not torch.isfinite(last).all():
            return [("<non-finite logits>", float("nan"))]
        probs = torch.softmax(last, -1)
        top = probs.topk(k)
        return [(tokenizer.decode([int(i)]), round(p.item(), 3)) for p, i in zip(top.values, top.indices)]

    results: dict[str, dict] = {}
    with torch.no_grad():
        results["parametric (no context)"] = {"scores": score_names(None, "parametric"),
                                              "top": topk_after_query(None, "parametric")}
        for scen, corrected, wrong in SCENARIOS:
            prefix = build_prefix(tokenizer, markers, corrected, wrong, device, filler=not args.no_filler)
            print(f"\n=== {scen} (prefix {prefix.shape[1]} tokens) ===", flush=True)
            rows: dict[str, dict] = {}
            rows["no-sleep"] = {"scores": score_names(prefix, "no-sleep")}
            rows["sleep-intact"] = {"scores": score_names(prefix, "sleep-intact"),
                                    "top": topk_after_query(prefix, "sleep-intact")}
            rows["sleep-none"] = {"scores": score_names(prefix, "sleep-none")}
            results[scen] = rows

    print("\n\n========== RESULTS (mean logprob per answer token) ==========")
    for scen, rows in results.items():
        print(f"\n--- {scen} ---")
        if "scores" in rows:
            rows = {"": rows}
        for cond, data in rows.items():
            print(f"  {cond or 'baseline':>14}: " + "  ".join(f"{n}: {v:+.3f}" for n, v in data["scores"].items()))
            if "top" in data:
                print(f"  {'top-8 next':>14}: " + ", ".join(f"{t!r} {p}" for t, p in data["top"]))


if __name__ == "__main__":
    main()
