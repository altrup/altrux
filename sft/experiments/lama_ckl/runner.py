"""Run one frozen LAMA-CKL Mamba arm with resumable epoch artifacts.

An epoch reads the 500 to-learn documents once in the official frozen order.
Only the altrux arm wakes: it reads them as wake-sleep cycles of
--docs-per-wake documents, carrying the recurrent state across the cycles of
an epoch and starting every epoch from fresh state. Each cycle dreams
--dreams-per-cycle dreams from the open post-wake state and distils them.
lora and mix-review train one document epoch instead; frozen is the initial
evaluation alone. Both fact sets are evaluated once per epoch, after its last
cycle.

Runs resume at epoch granularity from the last completed epoch-EE/. The state
at an epoch end is the fresh state, so a partial epoch restarts from its first
cycle with the saved adapter and optimizer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from experiments.dream_generation import (
    dream_generation_seed,
    generate_replay_dreams,
)
from experiments.dream_types import dream_set_sha
from experiments.lama_ckl.evaluation import score_records
from experiments.lama_ckl.protocol import (
    WAKE_FRAME,
    apply_dream_prompt,
    dream_prompt_ids,
    lama_dream_diagnostics,
    run_conversational_wake,
)
from experiments.lama_ckl.split import DEFAULT_SPLIT, DEFAULT_WARMSTART, pinned_warmstart_sha
from experiments.lama_ckl.training import epoch_batches, train_document_epoch
from progress import fmt_duration, heartbeat, ts

ARMS = ("frozen", "lora", "mix-review", "altrux")
EPOCHS = 30
DOCS_PER_WAKE = 10
DREAMS_PER_CYCLE = 10
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EVIDENCE_TOKENS = 512
REPLY_TOKENS = 128
DREAM_TOKENS = 512
DREAM_TEMPERATURE = 0.7
KL_TEMPERATURE = 1.0


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_dream_payload(
    dreams, generation_seeds: Sequence[int], teacher_sha: str, diagnostics: Mapping[str, object]
) -> dict[str, object]:
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
        "dreams": [
            {
                "sha256": dream.dream_sha,
                "token_ids": dream.dream_ids,
                "text": "".join(dream.token_texts),
                "prefix_tokens": dream.prefix_len,
                "stop_reason": dream.stop_reason,
            }
            for dream in dreams
        ],
    }


def curve_summary(curve: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    if not curve:
        raise ValueError("the evaluation curve is empty")
    peak = max(curve, key=lambda row: (float(row["to_learn_accuracy"]), -int(row["epoch"])))
    learned = float(peak["to_learn_accuracy"])
    retained = float(peak["not_to_forget_accuracy"])
    return {
        "top_accuracy": learned,
        "epoch": int(peak["epoch"]),
        "not_to_forget_accuracy": retained,
        "total_knowledge": round(learned + retained, 6),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def _write_epoch_result(directory: Path, result: dict[str, object]) -> None:
    while True:
        _write_json(directory / "result.json", result)
        artifact_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        if result.get("artifact_bytes") == artifact_bytes:
            return
        result["artifact_bytes"] = artifact_bytes


def _load_split(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
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
            str(row["evidence"]),
            add_special_tokens=True,
            truncation=True,
            max_length=EVIDENCE_TOKENS,
        )["input_ids"]
        documents.append([int(token) for token in encoded])
    return documents


def cycles_per_epoch(documents: int, docs_per_wake: int) -> int:
    return -(-documents // docs_per_wake)


def dream_set_seed(seed: int, epoch: int, cycle: int) -> int:
    """Generation seed of one cycle's dream set, unique while cycle < 1000 and epoch < 100."""
    return seed * 100_000 + epoch * 1000 + cycle


class EpochSteps(Protocol):
    """The model-side steps of an epoch; run_epochs owns their order."""

    def wake(
        self, rows: Sequence[Mapping[str, object]], state: object | None
    ) -> tuple[dict[str, object], object]: ...

    def dream(
        self,
        epoch: int,
        cycle: int,
        rows: Sequence[Mapping[str, object]],
        open_state: object,
        wake: Mapping[str, object],
    ) -> tuple[Sequence[object], dict[str, object]]: ...

    def carry(self, open_state: object) -> object: ...

    def distil(
        self, dreams: Sequence[object], open_state: object, label: str
    ) -> dict[str, object]: ...

    def train_documents(self, epoch: int) -> dict[str, object]: ...

    def evaluate(self, epoch: int) -> dict[str, object]: ...

    def save(self, directory: Path, result: dict[str, object]) -> None: ...


def _altrux_epoch(
    epoch: int,
    directory: Path,
    rows: Sequence[Mapping[str, object]],
    docs_per_wake: int,
    steps: EpochSteps,
) -> dict[str, object]:
    """Wake-sleep cycles over rows in order; wake.json is saved before the wake is checked."""
    count = cycles_per_epoch(len(rows), docs_per_wake)
    cycles: list[dict[str, object]] = []
    state: object | None = None
    started = time.time()
    for cycle in range(count):
        cycle_started = time.time()
        cycle_rows = rows[cycle * docs_per_wake : (cycle + 1) * docs_per_wake]
        cycle_dir = directory / f"cycle-{cycle:02d}"
        cycle_dir.mkdir()
        wake, open_state = steps.wake(cycle_rows, state)
        _write_json(cycle_dir / "wake.json", wake)
        if wake["invariants"] != {"turns": len(cycle_rows), "internal_eoc": 0}:
            raise RuntimeError(
                f"epoch {epoch} cycle {cycle} wake invariants failed: {wake['invariants']}"
            )
        dreams, dream_payload = steps.dream(epoch, cycle, cycle_rows, open_state, wake)
        _write_json(cycle_dir / "dreams.json", dream_payload)
        state = steps.carry(open_state)
        distilled = steps.distil(dreams, open_state, f"altrux epoch {epoch} cycle {cycle}")
        diagnostics = dream_payload["diagnostics"]
        record: dict[str, object] = {
            "cycle": cycle,
            "documents": len(cycle_rows),
            "wake_sha256": wake["transcript_sha256"],
            "wake_tokens": len(wake["transcript_token_ids"]),
            "forced_closes": int(wake["forced_closes"]),
            "dreams": len(dreams),
            "duplicate_dreams": int(diagnostics["duplicate_dreams"]),
            "set_sha256": dream_payload["set_sha256"],
            "diagnostics": diagnostics,
            **distilled,
            "seconds": time.time() - cycle_started,
        }
        cycles.append(record)
        del dreams
        gc.collect()
        heartbeat()
        done = cycle + 1
        elapsed = time.time() - started
        print(
            f"[{ts()}] epoch {epoch} cycle {done}/{count}: "
            f"forced closes {record['forced_closes']}/{len(cycle_rows)}, "
            f"duplicate dreams {record['duplicate_dreams']}/{record['dreams']}, "
            f"{elapsed / done:.1f} s/cycle ETA {fmt_duration(elapsed / done * (count - done))}",
            flush=True,
        )

    def total(key: str) -> int:
        return sum(int(cycle[key]) for cycle in cycles)

    return {
        "kind": "altrux",
        "cycles": cycles,
        "optimizer_steps": total("optimizer_steps"),
        "token_gradients": total("token_gradients"),
        "generated_tokens": total("generated_tokens"),
        "review_tokens": 0,
        "dreams": total("dreams"),
        "forced_closes": total("forced_closes"),
        "duplicate_dreams": total("duplicate_dreams"),
    }


def run_epochs(
    output: Path,
    settings: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    steps: EpochSteps,
    curve: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Run epochs len(curve)..settings["epochs"], epoch 0 being the initial evaluation

    Each epoch is built in a work/ tempdir and moved to epoch-EE/ once its
    result is saved, so a crash leaves no partial epoch-EE/. Every epoch
    starts from fresh state. Returns the curve, one row per epoch.
    """
    arm = str(settings["arm"])
    (output / "work").mkdir(parents=True, exist_ok=True)

    def write_summary() -> dict[str, object]:
        summary = {
            "arm": arm,
            "seed": settings["seed"],
            "settings": dict(settings),
            "curve": curve,
            "checkpoint": curve_summary(curve),
        }
        _write_json(output / "summary.json", summary)
        return summary

    for epoch in range(len(curve), int(settings["epochs"]) + 1):
        started = time.time()
        work = Path(tempfile.mkdtemp(prefix=f"epoch-{epoch:02d}.", dir=output / "work"))
        wake_sha: str | None = None
        wake_tokens: int | None = None
        treatment: dict[str, object]
        if epoch == 0:
            treatment = {"kind": "initial", "optimizer_steps": 0, "review_tokens": 0}
        elif arm == "altrux":
            treatment = _altrux_epoch(epoch, work, rows, int(settings["docs_per_wake"]), steps)
            cycles = treatment["cycles"]
            wake_sha = hashlib.sha256(
                ",".join(str(cycle["wake_sha256"]) for cycle in cycles).encode()
            ).hexdigest()
            wake_tokens = sum(int(cycle["wake_tokens"]) for cycle in cycles)
        elif arm in ("lora", "mix-review"):
            treatment = {"kind": arm, **steps.train_documents(epoch)}
            heartbeat()
        else:
            raise ValueError(f"the {arm} arm has no treatment epochs")
        result = steps.evaluate(epoch)
        result.update(
            {
                "arm": arm,
                "seed": settings["seed"],
                "wake_sha256": wake_sha,
                "wake_tokens": wake_tokens,
                "treatment": treatment,
                "epoch_seconds": time.time() - started,
            }
        )
        steps.save(work, result)
        _write_epoch_result(work, result)
        os.replace(work, output / f"epoch-{epoch:02d}")
        curve.append(result)
        summary = write_summary()
        heartbeat()
        print(
            f"[{ts()}] completed epoch {epoch}/{settings['epochs']}: "
            f"{json.dumps(summary['checkpoint'], sort_keys=True)}",
            flush=True,
        )
    write_summary()
    return curve


def _evaluate(
    model,
    tokenizer,
    learned: Sequence[dict[str, object]],
    retained: Sequence[dict[str, object]],
    batch_size: int,
    device,
    epoch: int,
) -> dict[str, object]:
    started = time.time()
    learned_scores = score_records(
        model, tokenizer, learned, "task_descriptive", batch_size, EVIDENCE_TOKENS, device
    )
    retained_scores = score_records(
        model, tokenizer, retained, "task_descriptive", batch_size, EVIDENCE_TOKENS, device
    )
    for label, rows, scores in (
        ("to-learn", learned, learned_scores),
        ("not-to-forget", retained, retained_scores),
    ):
        for index in range(min(3, len(rows))):
            print(
                f"[{ts()}] epoch {epoch} {label} sample {index + 1}: "
                f"task={rows[index]['task_descriptive']!r} object={rows[index]['object']!r} "
                f"token_accuracy={scores[index]:.6f}",
                flush=True,
            )
    result = {
        "epoch": epoch,
        "to_learn_accuracy": sum(learned_scores) / len(learned_scores),
        "not_to_forget_accuracy": sum(retained_scores) / len(retained_scores),
        "to_learn_scores": learned_scores,
        "not_to_forget_scores": retained_scores,
        "evaluation_seconds": time.time() - started,
    }
    print(
        f"[{ts()}] epoch {epoch}: to-learn {result['to_learn_accuracy']:.6f}, "
        f"not-to-forget {result['not_to_forget_accuracy']:.6f}",
        flush=True,
    )
    return result


def _save_trainable(model, path: Path, rank: int, alpha: float) -> str:
    import torch

    state = {
        name: value.detach().cpu()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    torch.save(state, path)
    (path.parent / "lora_config.json").write_text(json.dumps({"rank": rank, "alpha": alpha}))
    return file_sha(path)


class _BoxSteps:
    """EpochSteps on the loaded model."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        settings: Mapping[str, object],
        model,
        tokenizer,
        optimizer,
        markers: tuple[str, str, str],
        learned: Sequence[dict[str, object]],
        retained: Sequence[dict[str, object]],
        device,
        teacher_sha: str,
    ) -> None:
        self.args = args
        self.settings = settings
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.markers = markers
        self.learned = learned
        self.retained = retained
        self.device = device
        self.teacher_sha = teacher_sha
        self.learned_tokens = _encode_documents(tokenizer, learned)
        self.retained_tokens = _encode_documents(tokenizer, retained)
        batch_size = int(settings["train_batch_size"])
        self.learned_batches = epoch_batches(len(learned), batch_size, 42)
        self.review_batches = epoch_batches(len(retained), batch_size, 0)

    def wake(self, rows, state):
        return run_conversational_wake(
            self.model,
            self.tokenizer,
            [str(row["evidence"]) for row in rows],
            state,
            *self.markers,
            reply_tokens=REPLY_TOKENS,
            evidence_tokens=EVIDENCE_TOKENS,
            device=self.device,
        )

    def dream(self, epoch, cycle, rows, open_state, wake):
        tokenizer = self.tokenizer
        count = int(self.settings["dreams_per_cycle"])
        seed = dream_set_seed(self.args.seed, epoch, cycle)
        dreams, topology = generate_replay_dreams(
            self.model,
            open_state,
            dream_prompt_ids(tokenizer, *self.markers, self.device),
            count=count,
            batch_size=self.args.dream_batch_size,
            seed=seed,
            n_tokens=int(self.settings["dream_tokens"]),
            temperature=DREAM_TEMPERATURE,
            decode_token=lambda token: tokenizer.decode([token]),
            stop_id=int(tokenizer.convert_tokens_to_ids(self.markers[2])),
            turn_id=int(tokenizer.eos_token_id),
        )
        diagnostics = lama_dream_diagnostics(dreams, rows, wake["transcript_token_ids"])
        seeds = [dream_generation_seed(seed, index, 0) for index in range(count)]
        payload = compact_dream_payload(dreams, seeds, self.teacher_sha, diagnostics)
        payload["teacher_cycles_distilled"] = cycle
        payload["batch_topology"] = topology
        payload["decoded_samples"] = ["".join(dream.token_texts) for dream in dreams[:3]]
        for index, sample in enumerate(payload["decoded_samples"], start=1):
            print(
                f"[{ts()}] epoch {epoch} cycle {cycle} dream sample {index}: {sample[:1000]!r}",
                flush=True,
            )
        print(
            f"[{ts()}] epoch {epoch} cycle {cycle} dream diagnostics "
            f"{json.dumps(diagnostics, sort_keys=True)}",
            flush=True,
        )
        return dreams, payload

    def carry(self, open_state):
        return apply_dream_prompt(
            self.model, self.tokenizer, open_state, *self.markers, self.device
        )

    def distil(self, dreams, open_state, label):
        from experiments.dream_distillation import distill_dream_set

        started = time.time()
        token_gradients = distill_dream_set(
            self.model,
            self.optimizer,
            dreams,
            open_state,
            1,
            KL_TEMPERATURE,
            lambda step, loss: print(
                f"[{ts()}] {label} dream {step + 1}/{len(dreams)} loss {loss:.4f}", flush=True
            ),
            lambda index, epoch, step: None,
        )
        return {
            "optimizer_steps": len(dreams),
            "token_gradients": token_gradients,
            "generated_tokens": sum(len(dream.dream_ids) - dream.prefix_len for dream in dreams),
            "distil_seconds": time.time() - started,
        }

    def train_documents(self, epoch):
        arm = self.args.arm
        if arm == "lora":
            documents, batches = self.learned_tokens, self.learned_batches
        else:
            documents = self.learned_tokens + self.retained_tokens
            offset = len(self.learned_tokens)
            batches = [
                learn + [offset + index for index in review]
                for learn, review in zip(self.learned_batches, self.review_batches, strict=True)
            ]
        return {
            **train_document_epoch(
                self.model,
                self.optimizer,
                documents,
                batches,
                pad_id=int(self.tokenizer.pad_token_id),
                device=self.device,
                label=f"{arm} epoch {epoch}",
            ),
            "raw_documents": len(documents),
            "review_tokens": (sum(map(len, self.retained_tokens)) if arm == "mix-review" else 0),
        }

    def evaluate(self, epoch):
        return _evaluate(
            self.model,
            self.tokenizer,
            self.learned,
            self.retained,
            self.args.eval_batch_size,
            self.device,
            epoch,
        )

    def save(self, directory, result):
        import torch

        rank, alpha = int(self.settings["lora_rank"]), float(self.settings["lora_alpha"])
        self.teacher_sha = _save_trainable(self.model, directory / "trainable.pt", rank, alpha)
        result["adapter_sha256"] = self.teacher_sha
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        result["source_tokens"] = (
            0 if result["epoch"] == 0 else int(self.settings["source_document_tokens"])
        )
        result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()


def _completed_epochs(output: Path) -> list[Path]:
    paths = sorted(path for path in output.glob("epoch-*") if (path / "result.json").exists())
    for expected, path in enumerate(paths):
        if path.name != f"epoch-{expected:02d}":
            raise SystemExit(f"non-contiguous completed epochs under {output}: {path.name}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-name", default="mamba2_2_7b")
    parser.add_argument("--init-adapter", type=Path, default=DEFAULT_WARMSTART)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--docs-per-wake",
        type=int,
        default=DOCS_PER_WAKE,
        help="documents per altrux wake-sleep cycle",
    )
    parser.add_argument(
        "--dreams-per-cycle",
        type=int,
        default=DREAMS_PER_CYCLE,
        help="dreams generated and distilled per altrux wake-sleep cycle",
    )
    parser.add_argument("--dream-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="engineering-only gate: one epoch, two documents per set, "
        "at most two documents per wake, two 32-token dreams per cycle",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("LAMA-CKL treatment runs require a CUDA GPU with the fused Mamba path")
    if min(args.dream_batch_size, args.eval_batch_size, args.docs_per_wake) < 1:
        raise SystemExit("batch sizes and --docs-per-wake must be positive")
    if args.dreams_per_cycle < 1:
        raise SystemExit("--dreams-per-cycle must be positive")
    pinned = pinned_warmstart_sha()
    torch.cuda.reset_peak_memory_stats()
    learned, retained, split_manifest = _load_split(args.split)
    epochs, docs_per_wake = EPOCHS, args.docs_per_wake
    dreams_per_cycle, dream_tokens = args.dreams_per_cycle, DREAM_TOKENS
    if args.smoke:
        learned, retained = learned[:2], retained[:2]
        epochs, docs_per_wake, dreams_per_cycle, dream_tokens = 1, min(docs_per_wake, 2), 2, 32
    if args.arm == "frozen":
        epochs = 0
    altrux = args.arm == "altrux"
    output = args.output or Path("../.cache/lama_ckl/runs") / (
        f"{args.arm}-seed-{args.seed}{'-smoke' if args.smoke else ''}"
    )
    output.mkdir(parents=True, exist_ok=True)

    warm_sha = file_sha(args.init_adapter / "trainable.pt")
    if warm_sha != pinned or split_manifest.get("warmstart_sha256") != warm_sha:
        raise SystemExit(
            f"run adapter {warm_sha}, pinned {pinned}, and split manifest "
            f"{split_manifest.get('warmstart_sha256')} must be the same warm start"
        )
    settings: dict[str, object] = {
        "arm": args.arm,
        "seed": args.seed,
        "engineering_only": args.smoke,
        "epochs": epochs,
        "docs_per_wake": docs_per_wake if altrux else None,
        "cycles_per_epoch": cycles_per_epoch(len(learned), docs_per_wake) if altrux else 0,
        "wake_frame": WAKE_FRAME if altrux else None,
        "train_batch_size": min(TRAIN_BATCH_SIZE, len(learned)),
        "learning_rate": LEARNING_RATE,
        "evidence_tokens": EVIDENCE_TOKENS,
        "reply_tokens": REPLY_TOKENS,
        "reply_temperature": 0.0,
        "dreams_per_cycle": dreams_per_cycle if altrux else 0,
        "dream_tokens": dream_tokens if altrux else 0,
        "dream_temperature": DREAM_TEMPERATURE if altrux else None,
        "kl_temperature": KL_TEMPERATURE if altrux else None,
        "requested_dream_batch_size": args.dream_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "model_name": args.model_name,
        "warmstart_sha256": warm_sha,
        "split_manifest_sha256": file_sha(args.split / "manifest.json"),
    }
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
    optimizer = (
        None
        if args.arm == "frozen"
        else torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    )
    if hasattr(model, "set_grad_checkpoint"):
        model.set_grad_checkpoint(True, 64)
    settings.update(
        {
            "gpu_model": torch.cuda.get_device_name(device),
            "gpu_count": 1,
            "visible_gpu_count": torch.cuda.device_count(),
            "lora_rank": rank,
            "lora_alpha": alpha,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "optimizer_parameters": (
                0 if optimizer is None else sum(parameter.numel() for parameter in trainable)
            ),
            "source_document_tokens": sum(map(len, _encode_documents(tokenizer, learned))),
            "review_document_tokens": sum(map(len, _encode_documents(tokenizer, retained))),
        }
    )
    manifest_path = output / "run.json"
    serialized_settings = json.dumps(settings, indent=1, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != serialized_settings:
        raise SystemExit(f"existing run uses different settings: {manifest_path}")
    manifest_path.write_text(serialized_settings)

    completed = _completed_epochs(output)
    curve = [json.loads((path / "result.json").read_text()) for path in completed]
    teacher_sha = warm_sha
    if completed:
        last = completed[-1]
        load_checkpoint(model, last)
        if optimizer is not None:
            optimizer.load_state_dict(
                torch.load(last / "optimizer.pt", map_location="cpu", weights_only=True)
            )
        teacher_sha = str(curve[-1]["adapter_sha256"])
        print(f"[{ts()}] resumed after epoch {len(curve) - 1} from {last}", flush=True)
    steps = _BoxSteps(
        args=args,
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        markers=(model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC),
        learned=learned,
        retained=retained,
        device=device,
        teacher_sha=teacher_sha,
    )
    run_epochs(output, settings, learned, steps, curve)


if __name__ == "__main__":
    main()
