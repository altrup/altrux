"""Per-cycle census of a LAMA-CKL dream run: all dreams of every cycle, plus adapter step readings.

With --frozen, also the acquisition census (DISCUSSION-20261024 sec 1.1, sec 6.2b)."""

import argparse
import json
import re
from collections import Counter, deque
from collections.abc import Sequence
from pathlib import Path

import torch

from experiments.lama_ckl.dreamer_corpus import is_acknowledgement
from experiments.lama_ckl.protocol import WAKE_FRAMES
from experiments.lama_ckl.split import DEFAULT_WARMSTART
from progress import heartbeat, ts

PROMPT = "<|endofconversation|>"
FRAMES = tuple(frame.strip() for frame in WAKE_FRAMES)
LAGS = (1, 5, 20)
WORDS = re.compile(r"\w+")
TURN_END = re.compile(r"\n|<\|endoftext\|>|<\|endofconversation\|>|\[USER\]|\[ASSISTANT\]")
Trigram = tuple[str, str, str]


def _trigrams(text: str) -> list[Trigram]:
    words = WORDS.findall(text.lower())
    return list(zip(words, words[1:], words[2:], strict=False))


def document_trigrams(documents: Sequence[str]) -> list[set[Trigram]]:
    """Each document's trigrams that occur in no other distinct document of the set."""
    sets = {document: set(_trigrams(document)) for document in documents}
    frequency = Counter(trigram for trigrams in sets.values() for trigram in trigrams)
    return [{t for t in sets[document] if frequency[t] == 1} for document in documents]


def revisits(dream: str, document: set[Trigram]) -> bool:
    """The pinned revisit rule: two distinct shared trigrams, at their first
    occurrence in the dream, at non-overlapping positions."""
    first: dict[Trigram, int] = {}
    for position, trigram in enumerate(_trigrams(dream)):
        if trigram in document:
            first.setdefault(trigram, position)
    return bool(first) and max(first.values()) - min(first.values()) >= 3


def short_turns(text: str) -> list[str]:
    """`[ASSISTANT]` turns, read to the next newline, EOS or role marker, of one to eight words."""
    turns = (TURN_END.split(part, 1)[0].strip() for part in text.split("[ASSISTANT]")[1:])
    return [turn for turn in turns if 1 <= len(turn.split()) <= 8]


def _flat(path: Path) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return torch.cat(
        [state[k].double().flatten() for k in sorted(state) if state[k].is_floating_point()]
    )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _counts(payload: dict) -> dict[str, int]:
    texts = [d["text"].removeprefix(PROMPT) for d in payload["dreams"]]
    return {
        "frame_anywhere": sum(any(f in t for f in FRAMES) for t in texts),
        "frame_early": sum(any(0 <= t.find(f) < 60 for f in FRAMES) for t in texts),
        "frame_strict": sum(t.lstrip().startswith(FRAMES) for t in texts),
        "no_user": sum("[USER]" not in t for t in texts),
        "assistant_present": sum("[ASSISTANT]" in t for t in texts),
        "eoc_stop": sum(d["stop_reason"] == "eoc" for d in payload["dreams"]),
    }


def _text_counts(texts: Sequence[str], documents: Sequence[set[Trigram]]) -> dict[str, object]:
    per_turn = [sum(revisits(text, document) for text in texts) for document in documents]
    shorts = [short_turns(text) for text in texts]
    literals = Counter(turn for turns in shorts for turn in set(turns)).most_common(1)
    return {
        "revisits_per_turn": per_turn,
        "revisiting_dreams": sum(any(revisits(t, d) for d in documents) for t in texts),
        "short_turn_dreams": sum(bool(turns) for turns in shorts),
        "acknowledgement_dreams": sum(any(map(is_acknowledgement, turns)) for turns in shorts),
        "top_short_turn": literals[0][0] if literals else None,
        "top_short_turn_dreams": literals[0][1] if literals else 0,
    }


def _cycle_dirs(directory: Path) -> list[Path]:
    numbered = [path for path in directory.glob("cycle-*") if path.name[6:].isdigit()]
    return sorted(numbered, key=lambda path: int(path.name[6:]))


def run(directory: Path, warm_start: Path) -> list[dict]:
    warm = _flat(warm_start)
    previous = warm
    steps: deque[torch.Tensor] = deque(maxlen=max(LAGS) + 1)
    rows = []
    complete = []
    for cycle_dir in _cycle_dirs(directory):
        if (cycle_dir / "dreams.json").exists() and (cycle_dir / "trainable.pt").exists():
            complete.append(cycle_dir)
        else:
            print(f"[{ts()}] skip {cycle_dir.name}: incomplete", flush=True)
    wakes = [json.loads((cycle_dir / "wake.json").read_text())["turns"] for cycle_dir in complete]
    unique = iter(document_trigrams([turn["document"] for turns in wakes for turn in turns]))
    for cycle_dir, turns in zip(complete, wakes, strict=True):
        documents = [next(unique) for _ in turns]
        payload = json.loads((cycle_dir / "dreams.json").read_text())
        diagnostics = payload["diagnostics"]
        distil_path = cycle_dir / "distil.json"
        distil = json.loads(distil_path.read_text()) if distil_path.exists() else {}
        adapter = _flat(cycle_dir / "trainable.pt")
        step = adapter - previous
        steps.append(step)
        row = {
            "cycle": int(cycle_dir.name.split("-")[1]),
            **_counts(payload),
            "duplicate_dreams": diagnostics["duplicate_dreams"],
            "mean_copy_fraction": diagnostics["mean_copy_fraction"],
            "subjects_mentioned": diagnostics["subjects_mentioned"],
            **_text_counts([d["text"].removeprefix(PROMPT) for d in payload["dreams"]], documents),
            "max_dream_loss": distil.get("max_dream_loss"),
            "loss": distil.get("loss"),
            "clipped": distil.get("clipped"),
            "grad_norm": distil.get("grad_norm"),
            "step_norm": float(step.norm()),
            "dist_from_warm_start": float((adapter - warm).norm()),
            **{
                f"cos_lag{lag}": _cos(step, steps[-1 - lag]) if len(steps) > lag else None
                for lag in LAGS
            },
        }
        previous = adapter
        rows.append(row)
        cosines = " ".join(
            "-" if row[f"cos_lag{lag}"] is None else f"{row[f'cos_lag{lag}']:.2f}" for lag in LAGS
        )
        loss_max = "-" if row["max_dream_loss"] is None else f"{row['max_dream_loss']:.3f}"
        grad = "n/a" if row["grad_norm"] is None else f"{row['grad_norm']:.3f}"
        clip = "n/a" if row["clipped"] is None else int(row["clipped"])
        per_turn = " ".join(map(str, row["revisits_per_turn"]))
        print(
            f"[{ts()}] cycle {row['cycle']:3d} | anywhere {row['frame_anywhere']:2d} early {row['frame_early']:2d}"
            f" strict {row['frame_strict']:2d} no_user {row['no_user']:2d} asst {row['assistant_present']:2d}"
            f" eoc {row['eoc_stop']:2d} dup {row['duplicate_dreams']:2d} copy {row['mean_copy_fraction']:.2f}"
            f" | revisit {per_turn} ({row['revisiting_dreams']}) subj {row['subjects_mentioned']}"
            f" short {row['short_turn_dreams']:2d} ack {row['acknowledgement_dreams']:2d}"
            f" top {row['top_short_turn']!r} x{row['top_short_turn_dreams']}"
            f" | loss_max {loss_max} | clip {clip} grad {grad}"
            f" | step {row['step_norm']:.3f} cos(1,5,20) {cosines}"
            f" | dist {row['dist_from_warm_start']:.3f}",
            flush=True,
        )
        heartbeat()
    (directory / "census.json").write_text(
        json.dumps({"warm_start": str(warm_start), "cycles": rows}, indent=2)
    )
    print(f"[{ts()}] wrote {directory / 'census.json'} ({len(rows)} cycles)", flush=True)
    return rows


def _setting(settings: dict[str, object], key: str, flag: int | None) -> int:
    value = settings.get(key)
    if value is None and flag is None:
        raise SystemExit(f"run.json has no {key}; pass --{key.replace('_', '-')}")
    if value is not None and flag is not None and int(value) != flag:
        raise SystemExit(f"--{key.replace('_', '-')} {flag} disagrees with run.json's {value}")
    return int(value if value is not None else flag)


def _scores(path: Path) -> list[float]:
    return json.loads(path.read_text())["to_learn_scores"]


def acquisition(
    directory: Path,
    frozen: Sequence[Path],
    docs_per_wake: int | None = None,
    eval_every: int | None = None,
) -> dict[str, list[int]]:
    """Rows by outcome under the one definition of DISCUSSION-20261024 sec 1.1 / 3.1."""
    run_dir = next(path for path in directory.resolve().parents if (path / "run.json").exists())
    settings = json.loads((run_dir / "run.json").read_text())
    per_wake = _setting(settings, "docs_per_wake", docs_per_wake)
    every = _setting(settings, "eval_every", eval_every)
    cycles = {
        int(json.loads(path.read_text())["cycle"]): _scores(path)
        for path in directory.glob("cycle-*/result.json")
    }
    last = max(cycles)
    expected = {
        c for c in range(last + 1) if (c + 1) % every == 0 or c == settings["cycles_per_epoch"] - 1
    }
    if set(cycles) != expected:
        raise SystemExit(f"evaluated cycles {sorted(cycles)} != eval-every rule {sorted(expected)}")
    evaluations = [(-1, _scores(run_dir / "epoch-00" / "result.json")), *sorted(cycles.items())]
    frozen_scores = []
    for path in frozen:
        found = [_scores(result) for result in path.rglob("result.json")]
        if not found:
            raise SystemExit(f"no result.json under {path}")
        frozen_scores += found
    names = ("acquired", "kept", "late", "denominator", "not_evaluated")
    rows: dict[str, list[int]] = {name: [] for name in names}
    rows |= {"excluded_frozen": [], "excluded_before_wake": [], "excluded": []}
    for row in range(len(evaluations[0][1])):
        wake = row // per_wake
        after = [scores[row] for cycle, scores in evaluations if cycle >= wake]
        if not after:
            rows["not_evaluated"].append(row)
            continue
        reasons = {
            "excluded_frozen": any(scores[row] > 0 for scores in frozen_scores),
            "excluded_before_wake": any(s[row] > 0 for c, s in evaluations if c < wake),
        }
        for reason in (reason for reason, hit in reasons.items() if hit):
            rows[reason].append(row)
        if any(reasons.values()):
            rows["excluded"].append(row)
            continue
        rows["denominator"].append(row)
        if after[0] == 1.0:
            rows["acquired"].append(row)
            if after[-1] == 1.0:
                rows["kept"].append(row)
        elif any(score > 0 for score in after):
            rows["late"].append(row)
    print(
        f"[{ts()}] acquisition over cycles {min(cycles)}-{last} (docs_per_wake {per_wake},"
        f" eval_every {every}, {len(frozen_scores)} frozen evaluations): "
        + " ".join(f"{name} {len(ids)}" for name, ids in rows.items()),
        flush=True,
    )
    for name in ("acquired", "kept", "late", "excluded_frozen", "excluded_before_wake"):
        print(f"[{ts()}]   {name} rows: {rows[name]}", flush=True)
    (directory / "acquisition.json").write_text(json.dumps(rows, indent=1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="an epoch-NN/ or work/epoch-NN.*/ dir holding cycle-NN/ dirs"
    )
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARMSTART / "trainable.pt")
    parser.add_argument(
        "--frozen", type=Path, nargs="+", help="frozen run dirs; runs the acquisition census"
    )
    parser.add_argument("--docs-per-wake", type=int, help="only when run.json lacks it")
    parser.add_argument("--eval-every", type=int, help="only when run.json lacks it")
    args = parser.parse_args()
    run(args.directory, args.warm_start)
    if args.frozen:
        acquisition(args.directory, args.frozen, args.docs_per_wake, args.eval_every)


if __name__ == "__main__":
    main()
