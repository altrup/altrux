"""Locality probes shared by the CL A/B arms: a self-calibrated knowledge
battery and a fixed held-out perplexity slice.

Both measure catastrophic forgetting rather than installation (see
notes/DISCUSSION-20260805-dream-distillation-cl-ab.md sec 4). The battery is
built by self-calibration -- candidate completions are run through the base
model greedy and only its own consistent hits are kept, so a lost item is the
model forgetting something it demonstrably knew, not a question it never could
answer. The battery is built once per model and cached to a json artifact; the
same items are then re-scored after every sleep, with per-item log-prob drops
as the sensitive measure underneath the correct->incorrect flips.

ppl over HELDOUT_TEXT is the backstop for diffuse degradation that no item
battery catches.

Everything here is CPU-testable except `perplexity`, which needs a model; the
torch imports live inside the functions that need them for that reason.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from consolidation_null import normalize, run_chunks, ts

# probe(prompt) -> (greedy continuation, mean log-prob of the expected answer)
Probe = Callable[[str], tuple[str, float]]

# A fact counts as installed at this much margin over its distractor code --
# the correct code roughly 2.7x likelier over the whole code
# (notes/DISCUSSION-20260806 sec 4). Chosen before the data, not after.
MARGIN_INSTALL = 1.0

BatteryItem = dict[str, str | float | bool]

# Candidate completions for the battery. Deliberately simple and diverse --
# what matters is that they are *the model's own* knowledge (self-calibration
# drops the rest), not that they are hard.
BATTERY_CANDIDATES: list[tuple[str, str]] = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Egypt is", "Cairo"),
    ("The largest planet in the solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for iron is", "Fe"),
    ("Water freezes at a temperature of zero degrees", "Celsius"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("The author of Pride and Prejudice is Jane", "Austen"),
    ("The theory of general relativity was developed by Albert", "Einstein"),
    ("The first president of the United States was George", "Washington"),
    ("The longest river in Africa is the", "Nile"),
    ("The tallest mountain in the world is Mount", "Everest"),
    ("The ocean between Europe and North America is the", "Atlantic"),
    ("A triangle has three", "sides"),
    ("A year on Earth has twelve", "months"),
    ("The opposite of hot is", "cold"),
    ("The opposite of ancient is", "modern"),
    ("Cows produce a drink called", "milk"),
    ("Bees produce a sweet substance called", "honey"),
    ("The study of living organisms is called", "biology"),
    ("The language spoken in Brazil is", "Portuguese"),
    ("The language most widely spoken in Mexico is", "Spanish"),
    ("The currency of the United Kingdom is the", "pound"),
    ("The currency of Japan is the", "yen"),
    ("The planet closest to the sun is", "Mercury"),
    ("The red planet is", "Mars"),
    ("Photosynthesis is the process by which plants convert sunlight into", "energy"),
    ("The human body pumps blood using the", "heart"),
    ("The organ that filters blood is the", "kidney"),
    ("A baby dog is called a", "puppy"),
    ("A baby cat is called a", "kitten"),
    ("The season that follows winter is", "spring"),
    ("The day that follows Monday is", "Tuesday"),
    ("The month that follows June is", "July"),
    ("The Great Wall is located in", "China"),
    ("The Eiffel Tower is located in", "Paris"),
    ("Mount Fuji is located in", "Japan"),
    ("The programming language named after a snake is", "Python"),
]

# Fixed held-out general text for the ppl backstop. Never trained on, never
# generated from -- the same slice at every measurement, so a ppl delta is
# attributable to the sleep and nothing else.
HELDOUT_TEXT = (
    "The harbour town had grown around a single stone pier, and for most of its history the pier "
    "decided everything: where the market stood, which streets were paved first, and how far the "
    "houses could climb the hill before the wind made them impractical. Fishermen worked in the "
    "early hours and slept through the afternoon, so the town kept two schedules that rarely "
    "overlapped. Visitors found this confusing. A shop might be shuttered at noon and busy at "
    "four in the morning, and nobody thought to explain it.\n\n"
    "Weather arrived from the west with little warning. The older buildings faced away from it, "
    "presenting blank walls to the sea and their windows to the sheltered courtyards behind. "
    "Newer construction ignored the convention and paid for it in repairs. Every few winters a "
    "storm would take a roof, and the owners would rebuild in the same direction, convinced that "
    "the previous failure had been unlucky rather than predictable.\n\n"
    "Trade moved inland along a road that followed the river valley. Carts carried salt fish and "
    "returned with grain, timber, and news. The road was not maintained by anyone in particular, "
    "which meant it was maintained by everyone occasionally: a farmer would fill a rut, a carter "
    "would clear a fallen branch, and in this way the route stayed passable without ever being "
    "repaired properly. It took two days in good conditions and as long as a week after rain.\n\n"
    "The town's records begin late and are incomplete. What survives is mostly commercial - "
    "quantities, prices, the names of ships - and it is from these ledgers that historians have "
    "reconstructed the rest. The reconstruction is necessarily partial. A ledger records that a "
    "cargo was sold; it does not record who argued about the price, or whether the buyer walked "
    "home along the pier afterwards, thinking about the weather."
)


def _words(text: str) -> list[str]:
    return [w.strip(".,;:!?\"'()").lower() for w in normalize(text).split()]


def battery_hit(generation: str, answer: str) -> bool:
    """The greedy continuation counts as correct when it *opens* with the
    expected answer -- whole words, so "Parisian" is not a hit for "Paris"."""
    gen, ans = _words(generation), _words(answer)
    return bool(ans) and gen[: len(ans)] == ans


def build_battery(candidates: Sequence[tuple[str, str]], probe: Probe) -> list[BatteryItem]:
    """Self-calibration: keep only the candidates the model already answers
    correctly, with the baseline answer log-prob each was kept at."""
    kept: list[BatteryItem] = []
    for i, (prompt, answer) in enumerate(candidates):
        generation, logprob = probe(prompt)
        hit = battery_hit(generation, answer)
        if hit:
            kept.append({"prompt": prompt, "answer": answer, "logprob": logprob, "greedy": generation})
        print(f"[{ts()}]  battery calibrate {i + 1}/{len(candidates)} {'KEEP' if hit else 'drop'} "
              f"{prompt!r} -> {generation[:40]!r}  (kept {len(kept)})", flush=True)
    return kept


def load_or_build_battery(path: str | Path, candidates: Sequence[tuple[str, str]], probe: Probe) -> list[BatteryItem]:
    """The battery is a property of the model, not of a run: build it once and
    reuse the artifact, so every arm and seed is scored on identical items."""
    path = Path(path)
    if path.exists():
        items: list[BatteryItem] = json.loads(path.read_text())
        print(f"[{ts()}] knowledge battery: {len(items)} cached items from {path}")
        return items
    items = build_battery(candidates, probe)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1))
    print(f"[{ts()}] knowledge battery: kept {len(items)}/{len(candidates)} candidates -> {path}")
    return items


def score_battery(items: Sequence[BatteryItem], probe: Probe) -> list[BatteryItem]:
    """Re-probe a built battery. Every item was correct when built, so
    `correct=False` is a forgetting flip."""
    scored: list[BatteryItem] = []
    for item in items:
        generation, logprob = probe(str(item["prompt"]))
        scored.append({
            **item,
            "greedy_post": generation,
            "correct": battery_hit(generation, str(item["answer"])),
            "logprob_post": logprob,
            "logprob_delta": logprob - float(item["logprob"]),
        })
    return scored


def battery_summary(scored: Sequence[BatteryItem]) -> dict[str, float | int]:
    if not scored:
        return {"items": 0, "lost": 0, "retained_rate": 0.0, "mean_logprob_delta": 0.0}
    lost = sum(1 for r in scored if not r["correct"])
    return {
        "items": len(scored),
        "lost": lost,
        "retained_rate": (len(scored) - lost) / len(scored),
        "mean_logprob_delta": sum(float(r["logprob_delta"]) for r in scored) / len(scored),
    }


def code_margin(correct_logprob: float, distractor_logprob: float) -> tuple[float, bool]:
    """(margin, installed) from the two summed code log-probs. The margin is
    immune to the format prior the cues inject -- both codes gain equally from
    "digits are likelier here" -- and to the digit-counting attractor that
    makes greedy exact match unusable as a gate."""
    margin = correct_logprob - distractor_logprob
    return margin, margin >= MARGIN_INSTALL


def logprob_sum(model, prompt: torch.Tensor, target: torch.Tensor, state) -> float:
    """Teacher-forced log-prob of `target`, SUMMED over its tokens -- the
    reduction MARGIN_INSTALL is calibrated to (consolidation_null's
    `target_logprob` averages instead, which is the reported per-token column)."""
    import torch

    seq = torch.cat([prompt, target], dim=1)
    with torch.no_grad():
        logits, _ = model(seq, state=state)
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    start = prompt.shape[1] - 1
    idx = torch.arange(start, seq.shape[1] - 1, device=seq.device)
    return logprobs[idx, target[0]].sum().item()


def nll_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """Mean next-token NLL of `ids` (T,) under `logits` (T, V)."""
    import torch.nn.functional as F

    return F.cross_entropy(logits[:-1].float(), ids[1:]).item()


def perplexity(model, ids: torch.Tensor, chunk_len: int, label: str = "ppl") -> float:
    """Teacher-forced perplexity of `ids` (1, T) from a fresh state."""
    import math

    logits, _ = run_chunks(model, ids, None, chunk_len, label, keep_logits=True)
    return math.exp(nll_from_logits(logits[0], ids[0].cpu()))
