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

import hashlib
import json
import math
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


class CalibrationError(RuntimeError):
    pass

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
    ("The capital of Spain is", "Madrid"), ("The capital of Canada is", "Ottawa"),
    ("The capital of Australia is", "Canberra"), ("The capital of India is", "Delhi"),
    ("The capital of Germany is", "Berlin"), ("The capital of Greece is", "Athens"),
    ("The capital of Norway is", "Oslo"), ("The capital of Sweden is", "Stockholm"),
    ("The capital of Finland is", "Helsinki"), ("The capital of Portugal is", "Lisbon"),
    ("The capital of Ireland is", "Dublin"), ("The capital of Poland is", "Warsaw"),
    ("The capital of Austria is", "Vienna"), ("The capital of Hungary is", "Budapest"),
    ("The capital of Thailand is", "Bangkok"), ("The capital of Kenya is", "Nairobi"),
    ("The capital of Peru is", "Lima"), ("The capital of Chile is", "Santiago"),
    ("The capital of Argentina is", "Buenos Aires"), ("The capital of Turkey is", "Ankara"),
    ("The capital of South Korea is", "Seoul"), ("The capital of Indonesia is", "Jakarta"),
    ("The capital of Vietnam is", "Hanoi"), ("The capital of Morocco is", "Rabat"),
    ("The capital of Nigeria is", "Abuja"), ("The largest ocean is the", "Pacific"),
    ("The smallest continent is", "Australia"), ("The Sahara is a", "desert"),
    ("The Amazon is a", "river"), ("The Nile flows through", "Egypt"),
    ("The Earth orbits the", "Sun"), ("The Moon orbits the", "Earth"),
    ("The gas humans breathe is", "oxygen"), ("Plants absorb", "carbon dioxide"),
    ("The center of an atom is the", "nucleus"), ("The chemical symbol for sodium is", "Na"),
    ("The chemical symbol for silver is", "Ag"), ("The chemical symbol for carbon is", "C"),
    ("The chemical symbol for oxygen is", "O"), ("The chemical symbol for hydrogen is", "H"),
    ("The chemical symbol for helium is", "He"), ("The chemical symbol for copper is", "Cu"),
    ("The chemical symbol for lead is", "Pb"), ("The chemical symbol for tin is", "Sn"),
    ("The chemical symbol for zinc is", "Zn"), ("The chemical symbol for calcium is", "Ca"),
    ("A week has seven", "days"), ("A day has twenty four", "hours"),
    ("An hour has sixty", "minutes"), ("A minute has sixty", "seconds"),
    ("A rectangle has four", "sides"), ("A square has four equal", "sides"),
    ("Five plus five equals", "ten"), ("Twelve divided by three is", "four"),
    ("Seven times eight is", "fifty six"), ("The square root of nine is", "three"),
    ("The Roman numeral for five is", "V"), ("The first month is", "January"),
    ("The last month is", "December"), ("The day before Friday is", "Thursday"),
    ("The season before summer is", "spring"), ("The opposite of up is", "down"),
    ("The opposite of light is", "dark"), ("The opposite of fast is", "slow"),
    ("The opposite of early is", "late"), ("The color of grass is", "green"),
    ("The color of the sky on a clear day is", "blue"), ("A banana is usually", "yellow"),
    ("Snow is", "white"), ("Coal is", "black"), ("A rose is a", "flower"),
    ("An oak is a", "tree"), ("A salmon is a", "fish"), ("An eagle is a", "bird"),
    ("A whale is a", "mammal"), ("A frog is an", "amphibian"),
    ("A spider has eight", "legs"), ("A dog says", "woof"), ("A cat says", "meow"),
    ("A horse says", "neigh"), ("A sheep says", "baa"), ("A book is read with the", "eyes"),
    ("Sound is heard with the", "ears"), ("Food is tasted with the", "tongue"),
    ("A clock tells the", "time"), ("A thermometer measures", "temperature"),
    ("A ruler measures", "length"), ("A scale measures", "weight"),
    ("The author of Hamlet is William", "Shakespeare"),
    ("The author of The Odyssey is", "Homer"), ("The author of 1984 is George", "Orwell"),
    ("The author of The Hobbit is J. R. R.", "Tolkien"),
    ("The Mona Lisa was painted by Leonardo", "da Vinci"),
    ("The Statue of Liberty is in", "New York"), ("The pyramids are in", "Egypt"),
    ("The Colosseum is in", "Rome"), ("Big Ben is in", "London"),
    ("The Taj Mahal is in", "India"), ("The Great Barrier Reef is near", "Australia"),
    ("The first man on the Moon was Neil", "Armstrong"),
    ("The first person to circumnavigate Earth was Ferdinand", "Magellan"),
    ("The United Nations was founded after World War", "Two"),
    ("The United States Declaration of Independence was signed in", "1776"),
    ("The Titanic sank in", "1912"), ("The Renaissance began in", "Italy"),
    ("The ancient Olympics began in", "Greece"), ("The French Revolution began in", "1789"),
    ("The inventor associated with the telephone is Alexander Graham", "Bell"),
    ("The inventor associated with the light bulb is Thomas", "Edison"),
    ("The first computer programmer was Ada", "Lovelace"),
    ("The language of ancient Rome was", "Latin"), ("The language of Iran is", "Persian"),
    ("The language of Egypt is", "Arabic"), ("The language of China is", "Mandarin"),
    ("The language of Japan is", "Japanese"), ("The language of France is", "French"),
    ("The language of Germany is", "German"), ("The language of Russia is", "Russian"),
    ("The language of Korea is", "Korean"), ("The language of Greece is", "Greek"),
    ("The currency of the United States is the", "dollar"),
    ("The currency of Europe is", "euro"), ("The currency of India is", "rupee"),
    ("The currency of China is", "yuan"), ("The currency of Mexico is", "peso"),
    ("The currency of South Korea is", "won"), ("The currency of Russia is", "ruble"),
    ("The first letter of the alphabet is", "A"), ("The last letter of the alphabet is", "Z"),
    ("A sentence ends with a", "period"), ("A question ends with a", "question mark"),
    ("A synonym for happy is", "glad"), ("A synonym for begin is", "start"),
    ("A synonym for small is", "tiny"), ("A synonym for large is", "big"),
    ("A synonym for smart is", "clever"), ("A synonym for quick is", "fast"),
    ("A loaf is made from", "bread"), ("Cheese comes from", "milk"),
    ("Butter comes from", "milk"), ("Bread is baked in an", "oven"),
    ("Ice melts into", "water"), ("Steam is water", "vapor"),
    ("A bicycle has two", "wheels"), ("A car has four", "wheels"),
    ("A keyboard has", "keys"), ("A computer has a", "screen"),
    ("The Sun rises in the", "east"), ("The Sun sets in the", "west"),
    ("North is opposite", "south"), ("East is opposite", "west"),
    ("The hottest season is", "summer"), ("The coldest season is", "winter"),
    ("A doctor works in a", "hospital"), ("A teacher works at a", "school"),
    ("A chef works in a", "kitchen"), ("A farmer grows", "crops"),
    ("A librarian works in a", "library"), ("A pilot flies an", "airplane"),
    ("A firefighter puts out", "fires"), ("A painter uses a", "brush"),
    ("A musician plays", "music"), ("A photographer uses a", "camera"),
    ("The main character in Peter Pan is", "Peter Pan"),
    ("Cinderella loses a glass", "slipper"), ("Pinocchio has a wooden", "nose"),
    ("Sherlock Holmes lives in", "London"), ("Batman lives in", "Gotham"),
    ("Superman is from planet", "Krypton"), ("Mickey Mouse is a", "mouse"),
    ("The Beatles were a", "band"), ("Jazz is a type of", "music"),
    ("Soccer is played with a", "ball"), ("Tennis is played with a", "racket"),
    ("A chess board has sixty four", "squares"), ("A baseball team has nine", "players"),
    ("An Olympic marathon is", "running"), ("A piano has", "keys"),
]


def candidate_bank_hash(candidates: Sequence[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(list(candidates), separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def validate_battery_candidates(candidates: Sequence[tuple[str, str]], token_count: Callable[[str], int],
                                forbidden: Sequence[str]) -> None:
    """Reject answers outside the registered length and any wake collision."""
    needles = [needle.casefold() for needle in forbidden if needle]
    for prompt, answer in candidates:
        if not 1 <= token_count(answer) <= 3:
            raise CalibrationError(f"battery answer must be one to three tokenizer tokens: {answer!r}")
        text = f"{prompt} {answer}".casefold()
        if any(needle in text for needle in needles):
            raise CalibrationError(f"battery candidate collides with a wake target: {prompt!r}")


def calibrate_battery(path: str | Path, candidates: Sequence[tuple[str, str]], probe: Probe,
                      *, checkpoint_sha: str) -> dict[str, object]:
    """Build once for one warm start and reject changed calibration inputs."""
    path = Path(path)
    bank_sha = candidate_bank_hash(candidates)
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict) or "items" not in loaded:
            raise CalibrationError(f"battery calibration at {path} is not immutable metadata")
        if loaded.get("checkpoint_sha256") != checkpoint_sha:
            raise CalibrationError("battery calibration checkpoint does not match this warm start")
        if loaded.get("candidate_bank_sha256") != bank_sha:
            raise CalibrationError("battery calibration candidate bank does not match")
        return loaded
    artifact: dict[str, object] = {"version": 1, "checkpoint_sha256": checkpoint_sha,
                                   "candidate_bank_sha256": bank_sha,
                                   "items": build_battery(candidates, probe)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1, sort_keys=True))
    return artifact

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


def load_or_build_battery(path: str | Path, candidates: Sequence[tuple[str, str]], probe: Probe,
                          *, checkpoint_sha: str | None = None) -> list[BatteryItem]:
    """The battery is a property of the model, not of a run: build it once and
    reuse the artifact, so every arm and seed is scored on identical items."""
    if checkpoint_sha is not None:
        artifact = calibrate_battery(path, candidates, probe, checkpoint_sha=checkpoint_sha)
        items = artifact["items"]
        if not isinstance(items, list):
            raise CalibrationError("battery calibration items are invalid")
        return items  # type: ignore[return-value]  # JSON boundary validated above.
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


def load_or_build_battery_batched(
    path: str | Path,
    candidates: Sequence[tuple[str, str]],
    probe: Callable[[Sequence[tuple[str, str]]], Sequence[tuple[str, float]]],
    *,
    batch_size: int,
    checkpoint_sha: str,
) -> list[BatteryItem]:
    """Build the immutable calibration artifact with real prompt batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    path = Path(path)
    bank_sha = candidate_bank_hash(candidates)
    if path.exists():
        artifact = json.loads(path.read_text())
        if (not isinstance(artifact, dict) or artifact.get("checkpoint_sha256") != checkpoint_sha
                or artifact.get("candidate_bank_sha256") != bank_sha or not isinstance(artifact.get("items"), list)):
            raise CalibrationError("battery calibration metadata does not match")
        return artifact["items"]  # type: ignore[return-value]  # JSON boundary checked above.

    kept: list[BatteryItem] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        results = probe(batch)
        if len(results) != len(batch):
            raise CalibrationError("batched calibration returned a different number of results")
        for (prompt, answer), (generation, logprob) in zip(batch, results, strict=True):
            if battery_hit(generation, answer):
                kept.append({"prompt": prompt, "answer": answer, "logprob": logprob, "greedy": generation})
        print(f"[{ts()}]  battery calibrate {min(start + batch_size, len(candidates))}/{len(candidates)} "
              f"(kept {len(kept)}, batch {len(batch)})", flush=True)
    artifact: dict[str, object] = {"version": 1, "checkpoint_sha256": checkpoint_sha,
                                   "candidate_bank_sha256": bank_sha, "items": kept}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1, sort_keys=True))
    return kept


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


def score_battery_batched(
    items: Sequence[BatteryItem], probe: Callable[[Sequence[str]], Sequence[tuple[str, float]]], batch_size: int,
) -> list[BatteryItem]:
    """Score independent prompts in fixed input order and reject bad batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    scored: list[BatteryItem] = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        results = probe([str(item["prompt"]) for item in batch])
        if len(results) != len(batch):
            raise ValueError("batched probe returned a different number of results")
        for item, (generation, logprob) in zip(batch, results, strict=True):
            scored.append({**item, "greedy_post": generation,
                           "correct": battery_hit(generation, str(item["answer"])),
                           "logprob_post": logprob,
                           "logprob_delta": logprob - float(item["logprob"])})
    return scored


def battery_summary(scored: Sequence[BatteryItem]) -> dict[str, float | int]:
    if not scored:
        return {"items": 0, "lost": 0, "retained_rate": 0.0, "mean_logprob_delta": 0.0,
                "median_logprob_delta": 0.0, "p10_logprob_delta": 0.0}
    lost = sum(1 for r in scored if not r["correct"])
    deltas = sorted(float(r["logprob_delta"]) for r in scored)
    p10_index = (len(deltas) - 1) * 0.1
    lo, hi = math.floor(p10_index), math.ceil(p10_index)
    return {
        "items": len(scored),
        "lost": lost,
        "retained_rate": (len(scored) - lost) / len(scored),
        "mean_logprob_delta": sum(float(r["logprob_delta"]) for r in scored) / len(scored),
        "median_logprob_delta": (deltas[len(deltas) // 2] if len(deltas) % 2 else
                                  (deltas[len(deltas) // 2 - 1] + deltas[len(deltas) // 2]) / 2),
        "p10_logprob_delta": deltas[lo] + (deltas[hi] - deltas[lo]) * (p10_index - lo),
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
