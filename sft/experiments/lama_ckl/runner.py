"""Run one frozen LAMA-CKL Mamba arm with resumable cycle artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import tempfile
import time
from pathlib import Path
from collections.abc import Mapping, Sequence

from experiments.dreams.generation import (
    copy_state,
    dream_generation_seed,
    generate_replay_dreams,
    state_to,
)
from experiments.dreams.types import dream_set_sha
from experiments.lama_ckl.evaluation import score_records
from experiments.lama_ckl.protocol import (
    dream_instruction_ids,
    lama_dream_diagnostics,
    run_conversational_wake,
)
from experiments.lama_ckl.training import epoch_batches, train_document_epoch
from progress import ts


ARMS = ("frozen", "lora", "mix-review", "altrux")
WARMSTART_SHA256 = "226e95765f9e2c0a9fa335d5f70af8fb1d63bbf0f30c4427097b116375a11f3c"
CYCLES = 30
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EVIDENCE_TOKENS = 512
REPLY_TOKENS = 64
DREAM_COUNT = 300
DREAM_TOKENS = 512
DREAM_TEMPERATURE = 0.7
KL_TEMPERATURE = 1.0


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_dream_payload(dreams, generation_seeds: Sequence[int], teacher_sha: str,
                          diagnostics: Mapping[str, object]) -> dict[str, object]:
    """Keep enough to reconstruct a dream cache without retaining full logits."""
    logit_bytes = sum(
        dream.teacher_logits.nelement() * dream.teacher_logits.element_size() for dream in dreams
    )
    return {
        "teacher_sha256": teacher_sha,
        "set_sha256": dream_set_sha(dreams),
        "generation_seeds": list(generation_seeds),
        "ephemeral_teacher_logit_bytes": logit_bytes,
        "diagnostics": dict(diagnostics),
        "dreams": [{
            "sha256": dream.dream_sha,
            "token_ids": dream.dream_ids,
            "text": "".join(dream.token_texts),
            "prefix_tokens": dream.prefix_len,
            "stop_reason": dream.stop_reason,
        } for dream in dreams],
    }


def curve_summary(curve: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    if not curve:
        raise ValueError("the evaluation curve is empty")
    peak = max(curve, key=lambda row: (float(row["to_learn_accuracy"]), -int(row["cycle"])))
    learned = float(peak["to_learn_accuracy"])
    retained = float(peak["not_to_forget_accuracy"])
    return {
        "top_accuracy": learned,
        "cycle": int(peak["cycle"]),
        "not_to_forget_accuracy": retained,
        "total_knowledge": round(learned + retained, 6),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def _load_split(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for name in ("variant.jsonl", "invariant_descriptive.jsonl"):
        path = root / name
        expected = manifest["artifacts"][name]
        if file_sha(path) != expected["sha256"]:
            raise SystemExit(f"split artifact sha256 mismatch: {path}")
        loaded = _read_jsonl(path)
        if len(loaded) != expected["rows"]:
            raise SystemExit(f"split artifact row count mismatch: {path}")
        rows.append(loaded)
    return rows[0], rows[1], manifest


def _encode_documents(tokenizer, rows: Sequence[dict[str, object]]) -> list[list[int]]:
    documents: list[list[int]] = []
    for row in rows:
        encoded = tokenizer(
            str(row["evidence"]), add_special_tokens=True, truncation=True,
            max_length=EVIDENCE_TOKENS,
        )["input_ids"]
        documents.append([int(token) for token in encoded])
    return documents


def _evaluate(model, tokenizer, learned: Sequence[dict[str, object]],
              retained: Sequence[dict[str, object]], batch_size: int, device,
              cycle: int) -> dict[str, object]:
    started = time.time()
    learned_scores = score_records(
        model, tokenizer, learned, "task_descriptive", batch_size, EVIDENCE_TOKENS, device)
    retained_scores = score_records(
        model, tokenizer, retained, "task_descriptive", batch_size, EVIDENCE_TOKENS, device)
    for label, rows, scores in (("to-learn", learned, learned_scores),
                                ("not-to-forget", retained, retained_scores)):
        for index in range(min(3, len(rows))):
            print(f"[{ts()}] cycle {cycle} {label} sample {index + 1}: "
                  f"task={rows[index]['task_descriptive']!r} object={rows[index]['object']!r} "
                  f"token_accuracy={scores[index]:.6f}", flush=True)
    result = {
        "cycle": cycle,
        "to_learn_accuracy": sum(learned_scores) / len(learned_scores),
        "not_to_forget_accuracy": sum(retained_scores) / len(retained_scores),
        "to_learn_scores": learned_scores,
        "not_to_forget_scores": retained_scores,
        "evaluation_seconds": time.time() - started,
    }
    print(f"[{ts()}] cycle {cycle}: to-learn {result['to_learn_accuracy']:.6f}, "
          f"not-to-forget {result['not_to_forget_accuracy']:.6f}", flush=True)
    return result


def _save_trainable(model, path: Path, rank: int, alpha: float) -> str:
    import torch

    state = {name: value.detach().cpu() for name, value in model.named_parameters()
             if value.requires_grad}
    torch.save(state, path)
    (path.parent / "lora_config.json").write_text(json.dumps({"rank": rank, "alpha": alpha}))
    return file_sha(path)


def _save_cycle(directory: Path, model, optimizer, state, result: dict[str, object],
                wake: dict[str, object] | None, dream: dict[str, object] | None,
                rank: int, alpha: float) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    if wake is not None:
        _write_json(directory / "wake.json", wake)
    if dream is not None:
        _write_json(directory / "dreams.json", dream)
    result["adapter_sha256"] = _save_trainable(model, directory / "trainable.pt", rank, alpha)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save(state_to(copy_state(state), torch.device("cpu")), directory / "state.pt")
    result["artifact_bytes"] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    _write_json(directory / "result.json", result)


def _completed_cycles(output: Path) -> list[Path]:
    paths = sorted(path for path in output.glob("cycle-*") if (path / "result.json").exists())
    for expected, path in enumerate(paths):
        if path.name != f"cycle-{expected:02d}":
            raise SystemExit(f"non-contiguous completed cycles under {output}: {path.name}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", type=Path,
                        default=Path("../.cache/lama_ckl/mamba2_2_7b_recap050"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name", default="mamba2_2_7b")
    parser.add_argument("--init-adapter", type=Path, default=Path(
        "../models/mamba2_2_7b/checkpoints/recap050/epoch-2/step-800"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dream-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true",
                        help="engineering-only one-cycle, two-document, two-dream gate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("LAMA-CKL treatment runs require a CUDA GPU with the fused Mamba path")
    if args.dream_batch_size < 1 or args.eval_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    learned, retained, split_manifest = _load_split(args.split)
    cycles, dream_count, dream_tokens = CYCLES, DREAM_COUNT, DREAM_TOKENS
    if args.smoke:
        learned, retained = learned[:2], retained[:2]
        cycles, dream_count, dream_tokens = 1, 2, 32
    output = args.output or Path("../.cache/lama_ckl/runs") / (
        f"{args.arm}-seed-{args.seed}{'-smoke' if args.smoke else ''}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "work").mkdir(exist_ok=True)

    warm_sha = file_sha(args.init_adapter / "trainable.pt")
    if warm_sha != WARMSTART_SHA256 or split_manifest.get("warmstart_sha256") != warm_sha:
        raise SystemExit("run and split must use the pinned recap-0.5 warm start")
    settings: dict[str, object] = {
        "arm": args.arm,
        "seed": args.seed,
        "engineering_only": args.smoke,
        "cycles": cycles,
        "train_batch_size": min(TRAIN_BATCH_SIZE, len(learned)),
        "learning_rate": LEARNING_RATE,
        "evidence_tokens": EVIDENCE_TOKENS,
        "reply_tokens": REPLY_TOKENS,
        "reply_temperature": 0.0,
        "dream_count": dream_count if args.arm == "altrux" else 0,
        "dream_tokens": dream_tokens if args.arm == "altrux" else 0,
        "dream_temperature": DREAM_TEMPERATURE if args.arm == "altrux" else None,
        "kl_temperature": KL_TEMPERATURE if args.arm == "altrux" else None,
        "requested_dream_batch_size": args.dream_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "model_name": args.model_name,
        "warmstart_sha256": warm_sha,
        "split_manifest_sha256": file_sha(args.split / "manifest.json"),
    }
    manifest_path = output / "run.json"
    serialized_settings = json.dumps(settings, indent=1, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != serialized_settings:
        raise SystemExit(f"existing run uses different settings: {manifest_path}")
    manifest_path.write_text(serialized_settings)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model_mod = importlib.import_module(f"models.{args.model_name}")
    hooks = importlib.import_module(f"models.{args.model_name}.train_hooks")
    config = json.loads((args.init_adapter / "lora_config.json").read_text())
    rank, alpha = int(config["rank"]), float(config["alpha"])
    model, trainable = hooks.setup_training(device, rank, alpha, 0.0)
    from models.common import build_tokenizer
    from training.checkpoints import load_checkpoint

    load_checkpoint(model, args.init_adapter)
    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open, eoc = model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC
    optimizer = (None if args.arm == "frozen"
                 else torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01))
    if hasattr(model, "set_grad_checkpoint"):
        model.set_grad_checkpoint(True, 64)
    documents = [str(row["evidence"]) for row in learned]
    learned_tokens = _encode_documents(tokenizer, learned)
    retained_tokens = _encode_documents(tokenizer, retained)
    train_batch_size = int(settings["train_batch_size"])
    learned_batches = epoch_batches(len(learned), train_batch_size, 42)
    review_batches = epoch_batches(len(retained), train_batch_size, 0)

    completed = _completed_cycles(output)
    curve = [json.loads((path / "result.json").read_text()) for path in completed]
    state = None
    start_cycle = 0
    if completed:
        last = completed[-1]
        load_checkpoint(model, last)
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(last / "optimizer.pt", map_location="cpu", weights_only=True))
        state = state_to(torch.load(last / "state.pt", map_location="cpu", weights_only=False), device)
        start_cycle = int(curve[-1]["cycle"]) + 1
        print(f"[{ts()}] resumed after cycle {start_cycle - 1} from {last}", flush=True)
    else:
        initial = _evaluate(model, tokenizer, learned, retained, args.eval_batch_size, device, 0)
        work = Path(tempfile.mkdtemp(prefix="cycle-00.", dir=output / "work"))
        initial.update({"arm": args.arm, "seed": args.seed, "treatment": {"kind": "initial"}})
        _save_cycle(work, model, optimizer, None, initial, None, None, rank, alpha)
        os.replace(work, output / "cycle-00")
        curve.append(initial)
        start_cycle = 1

    for cycle in range(start_cycle, cycles + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cycle_started = time.time()
        work = Path(tempfile.mkdtemp(prefix=f"cycle-{cycle:02d}.", dir=output / "work"))
        wake_started = time.time()
        wake, state = run_conversational_wake(
            model, tokenizer, documents, state, user_open, asst_open, eoc,
            reply_tokens=REPLY_TOKENS, evidence_tokens=EVIDENCE_TOKENS,
        )
        wake_seconds = time.time() - wake_started
        if wake["invariants"] != {
            "turns": len(documents), "missing_assistant_eos": 0,
            "internal_eoc": 0, "closing_eoc": 1,
        }:
            raise RuntimeError(f"wake structural invariants failed: {wake['invariants']}")
        treatment: dict[str, object]
        dream_payload = None
        if args.arm == "frozen":
            treatment = {"kind": "frozen", "optimizer_steps": 0, "token_gradients": 0}
        elif args.arm in ("lora", "mix-review"):
            if args.arm == "lora":
                training_documents = learned_tokens
                batches = learned_batches
            else:
                training_documents = learned_tokens + retained_tokens
                offset = len(learned_tokens)
                batches = [learn + [offset + index for index in review]
                           for learn, review in zip(learned_batches, review_batches, strict=True)]
            treatment = {
                "kind": args.arm,
                **train_document_epoch(
                    model, optimizer, training_documents, batches,
                    pad_id=int(tokenizer.pad_token_id), device=device,
                    label=f"{args.arm} cycle {cycle}",
                ),
                "raw_documents": len(training_documents),
            }
        else:
            teacher_sha = warm_sha if cycle == 1 else str(curve[-1]["adapter_sha256"])
            generation_seed = args.seed * 100 + cycle
            dreams, topology = generate_replay_dreams(
                model, state, dream_instruction_ids(tokenizer, user_open, asst_open, device),
                count=dream_count, batch_size=args.dream_batch_size, seed=generation_seed,
                n_tokens=dream_tokens, temperature=DREAM_TEMPERATURE,
                decode_token=lambda token: tokenizer.decode([token]),
                stop_id=int(tokenizer.convert_tokens_to_ids(eoc)),
                turn_id=int(tokenizer.eos_token_id),
            )
            if len({dream.dream_sha for dream in dreams}) != len(dreams):
                raise RuntimeError("the fixed dream set contains duplicate dreams; refusing to train")
            diagnostics = lama_dream_diagnostics(dreams, learned, wake["transcript_token_ids"])
            seeds = [dream_generation_seed(generation_seed, index, 0) for index in range(dream_count)]
            dream_payload = compact_dream_payload(dreams, seeds, teacher_sha, diagnostics)
            dream_payload["batch_topology"] = topology
            dream_payload["decoded_samples"] = ["".join(dream.token_texts) for dream in dreams[:3]]
            _write_json(work / "dreams.json", dream_payload)
            for index, sample in enumerate(dream_payload["decoded_samples"], start=1):
                print(f"[{ts()}] cycle {cycle} dream sample {index}: {sample[:1000]!r}", flush=True)
            print(f"[{ts()}] cycle {cycle} dream diagnostics "
                  f"{json.dumps(diagnostics, sort_keys=True)}", flush=True)
            training_started = time.time()
            from experiments.dreams.distillation import distill_dream_set

            token_gradients = distill_dream_set(
                model, optimizer, dreams, state, None, 1, KL_TEMPERATURE,
                lambda step, loss: print(
                    f"[{ts()}] altrux cycle {cycle} dream {step + 1}/{len(dreams)} "
                    f"loss {loss:.4f}", flush=True),
                lambda index, epoch, step: None,
            )
            treatment = {
                "kind": "altrux",
                "optimizer_steps": len(dreams),
                "token_gradients": token_gradients,
                "generated_tokens": sum(len(dream.dream_ids) - dream.prefix_len for dream in dreams),
                "dreams": len(dreams),
                "set_sha256": dream_payload["set_sha256"],
                "teacher_sha256": teacher_sha,
                "diagnostics": diagnostics,
                "seconds": time.time() - training_started,
            }
            del dreams
            gc.collect()
        evaluated = _evaluate(model, tokenizer, learned, retained, args.eval_batch_size, device, cycle)
        evaluated.update({
            "arm": args.arm,
            "seed": args.seed,
            "wake_sha256": wake["transcript_sha256"],
            "wake_seconds": wake_seconds,
            "treatment": treatment,
            "cycle_seconds": time.time() - cycle_started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        })
        _save_cycle(work, model, optimizer, state, evaluated, wake, None, rank, alpha)
        os.replace(work, output / f"cycle-{cycle:02d}")
        curve.append(evaluated)
        summary = {"arm": args.arm, "seed": args.seed, "settings": settings,
                   "curve": curve, "checkpoint": curve_summary(curve)}
        _write_json(output / "summary.json", summary)
        print(f"[{ts()}] completed cycle {cycle}/{cycles}: "
              f"{json.dumps(summary['checkpoint'], sort_keys=True)}", flush=True)
    final_curve = [json.loads((path / "result.json").read_text())
                   for path in _completed_cycles(output)]
    _write_json(output / "summary.json", {
        "arm": args.arm,
        "seed": args.seed,
        "settings": settings,
        "curve": final_curve,
        "checkpoint": curve_summary(final_curve),
    })


if __name__ == "__main__":
    main()
