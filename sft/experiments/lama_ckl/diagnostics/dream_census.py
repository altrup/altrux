"""Per-cycle census of a LAMA-CKL dream run: all dreams of every cycle, plus adapter step readings."""

import argparse
import json
from collections import deque
from pathlib import Path

import torch

from experiments.lama_ckl.split import DEFAULT_WARMSTART
from progress import heartbeat, ts

PROMPT = "<|endofconversation|>"
FRAME = "Remember this for later:"
LAGS = (1, 5, 20)


def _flat(path: Path) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return torch.cat(
        [state[k].float().flatten() for k in sorted(state) if state[k].is_floating_point()]
    )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _counts(payload: dict) -> dict[str, int]:
    texts = [d["text"].removeprefix(PROMPT) for d in payload["dreams"]]
    return {
        "frame_anywhere": sum(FRAME in t for t in texts),
        "frame_early": sum(0 <= t.find(FRAME) < 60 for t in texts),
        "frame_strict": sum(t.lstrip().startswith(FRAME) for t in texts),
        "no_user": sum("[USER]" not in t for t in texts),
        "assistant_present": sum("[ASSISTANT]" in t for t in texts),
        "eoc_stop": sum(d["stop_reason"] == "eoc" for d in payload["dreams"]),
    }


def run(directory: Path, warm_start: Path) -> list[dict]:
    warm = _flat(warm_start)
    previous = warm
    steps: deque[torch.Tensor] = deque(maxlen=max(LAGS) + 1)
    rows = []
    for cycle_dir in sorted(directory.glob("cycle-*")):
        if not ((cycle_dir / "dreams.json").exists() and (cycle_dir / "trainable.pt").exists()):
            print(f"[{ts()}] skip {cycle_dir.name}: incomplete", flush=True)
            continue
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
            "max_dream_loss": distil.get("max_dream_loss"),
            "loss": distil.get("loss"),
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
        print(
            f"[{ts()}] cycle {row['cycle']:3d} | anywhere {row['frame_anywhere']:2d} early {row['frame_early']:2d}"
            f" strict {row['frame_strict']:2d} no_user {row['no_user']:2d} asst {row['assistant_present']:2d}"
            f" eoc {row['eoc_stop']:2d} dup {row['duplicate_dreams']:2d} copy {row['mean_copy_fraction']:.2f}"
            f" | loss_max {loss_max} | step {row['step_norm']:.3f} cos(1,5,20) {cosines}"
            f" | dist {row['dist_from_warm_start']:.3f}",
            flush=True,
        )
        heartbeat()
    (directory / "census.json").write_text(
        json.dumps({"warm_start": str(warm_start), "cycles": rows}, indent=2)
    )
    print(f"[{ts()}] wrote {directory / 'census.json'} ({len(rows)} cycles)", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="an epoch-NN/ or work/epoch-NN.*/ dir holding cycle-NN/ dirs"
    )
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARMSTART / "trainable.pt")
    args = parser.parse_args()
    run(args.directory, args.warm_start)


if __name__ == "__main__":
    main()
