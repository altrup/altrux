"""Experiment: lama-ckl

Pin, verify, and summarize the official TAALM LAMA-CKL release."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from progress import ts


TAALM_REPOSITORY = "https://github.com/ybseo-ac/TAALM.git"
TAALM_COMMIT = "b12f344a9dbae555c239635b1c192c555bed001b"
RELEASE_FILES: dict[str, dict[str, int | str]] = {
    "variant.jsonl": {
        "rows": 500,
        "sha256": "03b6c60d4642ec3ee2af3e7e0a2c4ed24bbb1fae7d63c5b205e6b12da1056248",
    },
    "invariant_descriptive.jsonl": {
        "rows": 500,
        "sha256": "c0b1925b92ff4050fac269e78ae11d806b9b3400e8770bad476c9be599b2a82d",
    },
    "invariant_schematic.jsonl": {
        "rows": 500,
        "sha256": "236ed1d27269a0d9359a89c8ca4f9a8c8401362ecaee8f60eb5590ec958d4cd7",
    },
    "train_attention_traindata.jsonl": {
        "rows": 4166,
        "sha256": "755e7cca18a341951958fb8ebc491a2d5005ba445892ab6d20f3e6b75a76283e",
    },
}
REQUIRED_KEYS = {
    "uuid",
    "relation_code",
    "subject",
    "object",
    "evidence",
    "task_descriptive",
    "invariant",
}
PUBLISHED_FINETUNE = {
    "top_accuracy": 0.115,
    "epoch": 16,
    "not_to_forget_accuracy": 0.8174,
}
ACCURACY_TOLERANCE = 0.02
EPOCH_TOLERANCE = 2
SINGLE_GPU_SMOKE_SIZE = 64

_SINGLE_GPU_REPLACEMENTS = (
    ("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", "CUDA_VISIBLE_DEVICES=0"),
    ("--train_grad_accum_step=1", "--train_grad_accum_step=8"),
)
_SMOKE_REPLACEMENTS = (
    ("--max_epochs=30", "--max_epochs=1"),
    (
        '--train_data="./data/LAMA_ckl/variant.jsonl"',
        '--train_data="./data/LAMA_ckl/gh200_smoke/variant.jsonl"',
    ),
    (
        '--eval_data_changed="./data/LAMA_ckl/variant.jsonl"',
        '--eval_data_changed="./data/LAMA_ckl/gh200_smoke/variant.jsonl"',
    ),
    (
        '--eval_data_unchanged="./data/LAMA_ckl/invariant_descriptive.jsonl"',
        '--eval_data_unchanged="./data/LAMA_ckl/gh200_smoke/invariant_descriptive.jsonl"',
    ),
    ('--add_to_title=""', '--add_to_title="gh200_smoke"'),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one upstream marker: {old}")
    return text.replace(old, new)


def adapt_single_gpu_script(text: str, *, smoke: bool = False) -> str:
    """Make the pinned eight-GPU launcher batch-equivalent on one GPU."""
    for old, new in _SINGLE_GPU_REPLACEMENTS:
        text = _replace_once(text, old, new)
    if smoke:
        for old, new in _SMOKE_REPLACEMENTS:
            text = _replace_once(text, old, new)
    return text


def prepare_single_gpu_smoke(root: str | Path) -> dict[str, int]:
    """Build the deterministic 64-row inputs for one accumulated update."""
    source = Path(root) / "data" / "LAMA_ckl"
    output = source / "gh200_smoke"
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, int] = {}
    for name in ("variant.jsonl", "invariant_descriptive.jsonl"):
        rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
        if len(rows) < SINGLE_GPU_SMOKE_SIZE:
            raise ValueError(f"{source / name} has fewer than {SINGLE_GPU_SMOKE_SIZE} rows")
        selected = rows[:SINGLE_GPU_SMOKE_SIZE]
        malformed = sum(not REQUIRED_KEYS <= row.keys() for row in selected)
        duplicates = len(selected) - len({row.get("uuid") for row in selected})
        missing_bindings = sum(
            row.get("subject") not in row.get("evidence", "")
            or row.get("object") not in row.get("evidence", "")
            for row in selected
        )
        if malformed or duplicates or missing_bindings:
            raise ValueError(
                f"{name} smoke invariants failed: malformed={malformed}, duplicates={duplicates}, "
                f"missing_bindings={missing_bindings}"
            )
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in selected))
        report[name] = len(selected)
    return report


def verify_release(
    root: str | Path,
    files: Mapping[str, Mapping[str, int | str]] = RELEASE_FILES,
) -> dict[str, dict[str, object]]:
    """Verify exact release bytes and structural invariants."""
    root = Path(root)
    report: dict[str, dict[str, object]] = {}
    for name, expected in files.items():
        path = root / name
        digest = _sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"{path} sha256 {digest} != {expected['sha256']}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != expected["rows"]:
            raise ValueError(f"{path} has {len(rows)} rows, expected {expected['rows']}")
        malformed = sum(
            not REQUIRED_KEYS <= row.keys()
            or not all(
                isinstance(row.get(key), str) and row[key]
                for key in (
                    "uuid",
                    "relation_code",
                    "subject",
                    "object",
                    "evidence",
                    "task_descriptive",
                )
            )
            for row in rows
        )
        duplicates = len(rows) - len({row.get("uuid") for row in rows})
        missing_bindings = sum(
            isinstance(row.get("evidence"), str)
            and (
                row.get("subject") not in row["evidence"]
                or row.get("object") not in row["evidence"]
            )
            for row in rows
        )
        if malformed or duplicates or missing_bindings:
            raise ValueError(
                f"{path} invariants failed: malformed={malformed}, duplicates={duplicates}, "
                f"missing_bindings={missing_bindings}"
            )
        report[name] = {
            "sha256": digest,
            "rows": len(rows),
            "invariants": {
                "malformed": malformed,
                "duplicate_uuid": duplicates,
                "missing_subject_or_object": missing_bindings,
            },
            "sample": rows[0],
        }
    return report


def reproduction_summary(
    curve: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int | bool]:
    """Return the official first-peak checkpoint metrics and gate verdict."""
    if not curve:
        raise ValueError("the reproduction curve is empty")
    peak = max(curve, key=lambda row: (float(row["to_learn_accuracy"]), -int(row["epoch"])))
    top = float(peak["to_learn_accuracy"])
    epoch = int(peak["epoch"])
    retained = float(peak["not_to_forget_accuracy"])
    passed = (
        abs(top - PUBLISHED_FINETUNE["top_accuracy"]) <= ACCURACY_TOLERANCE
        and abs(epoch - PUBLISHED_FINETUNE["epoch"]) <= EPOCH_TOLERANCE
        and abs(retained - PUBLISHED_FINETUNE["not_to_forget_accuracy"]) <= ACCURACY_TOLERANCE
    )
    return {
        "top_accuracy": top,
        "epoch": epoch,
        "not_to_forget_accuracy": retained,
        "total_knowledge": round(top + retained, 4),
        "passes_gate": passed,
    }


def load_official_result(path: str | Path) -> list[dict[str, float | int]]:
    """Normalize the pickle written by the pinned evaluation_run.py."""
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    raw = value.get("eval_result") if isinstance(value, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(f"{path} has no eval_result mapping")
    return [
        {
            "epoch": int(epoch) + 1,
            "to_learn_accuracy": float(metrics["acc_changed"]),
            "not_to_forget_accuracy": float(metrics["acc_unchanged"]),
        }
        for epoch, metrics in sorted(raw.items(), key=lambda item: int(item[0]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify a pinned TAALM checkout")
    verify.add_argument("root", type=Path, help="TAALM repository root")
    summarize = subparsers.add_parser("summarize", help="gate an official result pickle")
    summarize.add_argument("result", type=Path)
    adapt = subparsers.add_parser("adapt-single-gpu", help="emit the pinned launcher for one GPU")
    adapt.add_argument("root", type=Path, help="TAALM repository root")
    adapt.add_argument("--smoke", action="store_true")
    prepare = subparsers.add_parser(
        "prepare-single-gpu-smoke", help="build one-update smoke inputs"
    )
    prepare.add_argument("root", type=Path, help="TAALM repository root")
    args = parser.parse_args()
    if args.command == "verify":
        report = verify_release(args.root / "data" / "LAMA_ckl")
        for name, item in report.items():
            sample = item["sample"]
            print(f"[{ts()}] {name}: rows={item['rows']} sha256={item['sha256']}")
            print(f"[{ts()}] invariants={json.dumps(item['invariants'], sort_keys=True)}")
            print(
                f"[{ts()}] sample={sample['task_descriptive']!r} evidence={sample['evidence'][:300]!r}"
            )
    elif args.command == "summarize":
        curve = load_official_result(args.result)
        for row in curve:
            print(
                f"[{ts()}] epoch={row['epoch']} to_learn={row['to_learn_accuracy']:.6f} "
                f"not_to_forget={row['not_to_forget_accuracy']:.6f}"
            )
        summary = reproduction_summary(curve)
        print(f"[{ts()}] summary={json.dumps(summary, sort_keys=True)}")
        if not summary["passes_gate"]:
            raise SystemExit("official Llama-2-7B QLoRA reproduction gate failed")
    elif args.command == "adapt-single-gpu":
        script = (args.root / "scripts" / "eval" / "lamackl" / "finetune.sh").read_text()
        print(adapt_single_gpu_script(script, smoke=args.smoke), end="")
    else:
        report = prepare_single_gpu_smoke(args.root)
        for name, count in report.items():
            sample = json.loads(
                (args.root / "data" / "LAMA_ckl" / "gh200_smoke" / name).read_text().splitlines()[0]
            )
            print(f"[{ts()}] {name}: rows={count} malformed=0 duplicates=0 missing_bindings=0")
            print(
                f"[{ts()}] sample={sample['task_descriptive']!r} evidence={sample['evidence'][:300]!r}"
            )


if __name__ == "__main__":
    main()
