"""Build the frozen Mamba-conditioned LAMA-CKL 500/500 artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

from experiments.lama_ckl.data import build_candidates, select_split
from experiments.lama_ckl.evaluation import score_records
from progress import ts

WARMSTART_SHA256 = "226e95765f9e2c0a9fa335d5f70af8fb1d63bbf0f30c4427097b116375a11f3c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    paths = [root / "relations.jsonl", *sorted((root / "TREx").glob("P*.jsonl"))]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _serialized(rows: Sequence[dict[str, object]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"artifact already exists with different content: {path}")
    if not path.exists():
        path.write_text(text)


def freeze_split(
    output: str | Path,
    learned: Sequence[dict[str, object]],
    retained: Sequence[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Write immutable split files and their manifest."""
    output = Path(output)
    artifacts: dict[str, dict[str, object]] = {}
    for name, rows in (("variant.jsonl", learned), ("invariant_descriptive.jsonl", retained)):
        text = _serialized(rows)
        digest = hashlib.sha256(text.encode()).hexdigest()
        _write_once(output / name, text)
        artifacts[name] = {
            "rows": len(rows),
            "sha256": digest,
            "invariants": {
                "duplicate_uuid": len(rows) - len({row["uuid"] for row in rows}),
                "missing_subject_or_object": sum(
                    str(row["subject"]) not in str(row["evidence"])
                    or str(row["object"]) not in str(row["evidence"])
                    for row in rows
                ),
            },
            "sample": rows[0] if rows else None,
        }
    manifest = {**metadata, "artifacts": artifacts}
    _write_once(output / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lama-root",
        type=Path,
        required=True,
        help="unpacked official LAMA data directory containing relations.jsonl and TREx/",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("../.cache/lama_ckl/mamba2_2_7b_recap050")
    )
    parser.add_argument("--model-name", default="mamba2_2_7b")
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=Path("../models/mamba2_2_7b/checkpoints/recap050/epoch-2/step-800"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Mamba-conditioned split scoring requires a CUDA GPU")
    adapter_path = args.init_adapter / "trainable.pt"
    adapter_sha = _sha256(adapter_path)
    if adapter_sha != WARMSTART_SHA256:
        raise SystemExit(f"warm-start sha256 {adapter_sha} != {WARMSTART_SHA256}")
    source_sha = source_tree_sha(args.lama_root)
    candidates: list[dict[str, object]] = []
    for row in build_candidates(args.lama_root):
        candidates.append(row)
        if len(candidates) % 1000 == 0:
            print(f"[{ts()}] source candidates {len(candidates)}", flush=True)
    if not candidates:
        raise SystemExit("the official-notebook source filter produced no candidates")
    print(f"[{ts()}] source candidates {len(candidates)}, sha256 {source_sha}")
    print(
        f"[{ts()}] candidate sample task={candidates[0]['task_descriptive']!r} "
        f"evidence={str(candidates[0]['evidence'])[:500]!r}"
    )

    device = torch.device("cuda")
    model_mod = importlib.import_module(f"models.{args.model_name}")
    hooks = importlib.import_module(f"models.{args.model_name}.train_hooks")
    config = json.loads((args.init_adapter / "lora_config.json").read_text())
    model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
    from models.common import build_tokenizer

    from training.checkpoints import load_checkpoint

    load_checkpoint(model, args.init_adapter)
    tokenizer = build_tokenizer(model_mod)
    print(f"[{ts()}] warm start {args.init_adapter}, sha256 {adapter_sha}")
    descriptive = score_records(
        model, tokenizer, candidates, "task_descriptive", args.batch_size, args.max_length, device
    )
    schematic = score_records(
        model, tokenizer, candidates, "task_schematic", args.batch_size, args.max_length, device
    )
    for row, desc, schem in zip(candidates, descriptive, schematic, strict=True):
        row["scores"] = {"descriptive": desc, "schematic": schem}
    learned, retained = select_split(candidates, args.size, random.Random(args.seed))
    manifest = freeze_split(
        args.output,
        learned,
        retained,
        {
            "source": "facebookresearch/LAMA T-REx",
            "source_sha256": source_sha,
            "pipeline": "TAALM LAMA_ckl_pipeline.ipynb",
            "taalm_commit": "b12f344a9dbae555c239635b1c192c555bed001b",
            "model_name": args.model_name,
            "model_id": model_mod.MODEL_ID,
            "tokenizer_id": model_mod.TOKENIZER_ID,
            "warmstart_sha256": adapter_sha,
            "metric_alignment": "last object character span",
            "max_length": args.max_length,
            "seed": args.seed,
            "candidate_count": len(candidates),
        },
    )
    for name, artifact in manifest["artifacts"].items():
        print(
            f"[{ts()}] wrote {args.output / name}: rows={artifact['rows']} "
            f"sha256={artifact['sha256']}"
        )
        print(f"[{ts()}] invariants={json.dumps(artifact['invariants'], sort_keys=True)}")
        sample = artifact["sample"]
        print(
            f"[{ts()}] sample task={sample['task_descriptive']!r} "
            f"evidence={str(sample['evidence'])[:500]!r}"
        )


if __name__ == "__main__":
    main()
