"""Run one frozen LAMA-CKL Mamba arm with resumable epoch artifacts.

An epoch reads the 500 to-learn documents once in the official frozen order;
--epochs sets the count (dream cells run one). Only the dream arm wakes: it
reads them as wake-sleep cycles of --docs-per-wake documents, every wake from
fresh state with sampled replies. Each cycle dreams --dreams-per-cycle dreams
from that cycle's open post-wake state and distils them in one optimizer step,
then saves the adapter into epoch-EE/cycle-NNN/ and, every --eval-every
cycles and on the epoch's last, scores both fact sets there. lora and
mix-review train one document epoch instead; frozen is the initial evaluation
alone. Both fact sets are evaluated after every epoch; for dream the last
cycle's result is the epoch's.

An epoch is built in work/epoch-EE/ and moved to epoch-EE/ when complete. A
dream cycle is built in cycle-NNN.tmp/ and renamed to cycle-NNN/ last, with
its fp32 master adapter and, for an optimizer that holds state, optimizer.pt.
Rerunning the same command resumes after the last complete cycle, loading its
masters directly. --cycles-per-run N stops after cycle N-1 of the epoch and
--max-hours after the first cycle that ends past the cap; both write
summary.json with cycles_run and leave the epoch resumable.

--cycles N is a pre-grid acceptance check, not a grid cell: after the initial
evaluation, epoch 1 runs only its first N dream cycles, saves their adapters
without an evaluation, and the run stops. It needs a fresh --output outside
runs/.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import shutil
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
    WAKE_FRAMES,
    dream_prompt_ids,
    lama_dream_diagnostics,
    run_conversational_wake,
)
from experiments.lama_ckl.split import DEFAULT_SPLIT, DEFAULT_WARMSTART, pinned_warmstart_sha
from experiments.lama_ckl.training import epoch_batches, train_document_epoch
from progress import fmt_duration, heartbeat, ts

ARMS = ("frozen", "lora", "mix-review", "dream")
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
DREAM_LOSS_FLOOR = 0.01
WAKE_FRAME_SALT = "lama-ckl-wake-frame"
SESSION_LIMITS = ("max_hours", "cycles_per_run")
SCORE_KEYS = ("to_learn_accuracy", "not_to_forget_accuracy", "evaluation_seconds")


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


def draw_wake_frame(mode: str, epoch: int, cycle: int) -> str:
    """The cycle's wake frame; its own generator, so no reply or dream sample moves."""
    if mode == "single":
        return WAKE_FRAME
    return random.Random(f"{WAKE_FRAME_SALT}:{epoch}:{cycle}").choice(WAKE_FRAMES)


class EpochSteps(Protocol):
    """The model-side steps of an epoch; run_epochs owns their order."""

    def wake(
        self, rows: Sequence[Mapping[str, object]], state: object | None, epoch: int, cycle: int
    ) -> tuple[dict[str, object], object]: ...

    def dream(
        self,
        epoch: int,
        cycle: int,
        rows: Sequence[Mapping[str, object]],
        open_state: object,
        wake: Mapping[str, object],
    ) -> tuple[Sequence[object], dict[str, object]]: ...

    def distil(self, dreams: Sequence[object], label: str) -> dict[str, object]: ...

    def train_documents(self, epoch: int) -> dict[str, object]: ...

    def evaluate(self, epoch: int) -> dict[str, object]: ...

    def save_cycle(self, directory: Path) -> str: ...

    def restore_cycle(self, directory: Path) -> None: ...

    def save(self, directory: Path, result: dict[str, object]) -> None: ...


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _cycle_record(directory: Path) -> dict[str, object]:
    """The epoch record of one complete cycle, from its files alone."""
    wake = _read_json(directory / "wake.json")
    payload = _read_json(directory / "dreams.json")
    distilled = _read_json(directory / "distil.json")
    result = directory / "result.json"
    scores = {key: _read_json(result)[key] for key in SCORE_KEYS} if result.exists() else {}
    return {
        "cycle": int(directory.name.removeprefix("cycle-")),
        "documents": wake["invariants"]["turns"],
        "wake_sha256": wake["transcript_sha256"],
        "wake_tokens": len(wake["transcript_token_ids"]),
        "forced_closes": int(wake["forced_closes"]),
        "role_marker_closes": int(wake["role_marker_closes"]),
        "dreams": len(payload["dreams"]),
        "duplicate_dreams": int(payload["diagnostics"]["duplicate_dreams"]),
        "set_sha256": payload["set_sha256"],
        "diagnostics": payload["diagnostics"],
        **distilled,
        **scores,
        **_read_json(directory / "cycle.json"),
    }


def _complete_cycles(directory: Path) -> list[Path]:
    """Delete half-written cycle-NNN.tmp/ dirs; return the complete cycles, checked contiguous."""
    for stale in directory.glob("cycle-*.tmp"):
        shutil.rmtree(stale)
    paths = sorted(directory.glob("cycle-*"))
    for expected, path in enumerate(paths):
        if path.name != f"cycle-{expected:03d}":
            raise SystemExit(f"non-contiguous complete cycles under {directory}: {path.name}")
    return paths


def _dream_epoch(
    epoch: int,
    directory: Path,
    rows: Sequence[Mapping[str, object]],
    docs_per_wake: int,
    steps: EpochSteps,
    max_cycles: int | None = None,
    eval_every: int = 1,
    cycles_per_run: int | None = None,
    deadline: float | None = None,
) -> dict[str, object]:
    """
    Wake-sleep cycles over rows in order, after the complete ones already in directory

    wake.json is saved before the wake is checked. Every cycle saves its
    adapter; unless max_cycles is set, cycles with (cycle + 1) % eval_every == 0
    and the epoch's last are scored. Stops after cycle cycles_per_run - 1, or
    after the first cycle that ends past deadline.
    """
    count = cycles_per_epoch(len(rows), docs_per_wake)
    stop = min(count, max_cycles or count, cycles_per_run or count)
    complete = _complete_cycles(directory)
    cycles = [_cycle_record(path) for path in complete]
    if complete:
        steps.restore_cycle(complete[-1])
        print(f"[{ts()}] epoch {epoch} resumed after cycle {len(cycles) - 1}", flush=True)
    started, first = time.time(), len(cycles)
    for cycle in range(first, stop):
        cycle_started = time.time()
        cycle_rows = rows[cycle * docs_per_wake : (cycle + 1) * docs_per_wake]
        cycle_dir = directory / f"cycle-{cycle:03d}.tmp"
        cycle_dir.mkdir()
        wake, open_state = steps.wake(cycle_rows, None, epoch, cycle)
        _write_json(cycle_dir / "wake.json", wake)
        if wake["invariants"] != {"turns": len(cycle_rows), "internal_eoc": 0}:
            raise RuntimeError(
                f"epoch {epoch} cycle {cycle} wake invariants failed: {wake['invariants']}"
            )
        dreams, dream_payload = steps.dream(epoch, cycle, cycle_rows, open_state, wake)
        _write_json(cycle_dir / "dreams.json", dream_payload)
        distilled = steps.distil(dreams, f"dream epoch {epoch} cycle {cycle}")
        subjects = dream_payload["diagnostics"]["subjects_mentioned"]
        _write_json(cycle_dir / "distil.json", {**distilled, "subjects_mentioned": subjects})
        adapter_sha = steps.save_cycle(cycle_dir)
        if max_cycles is None and ((cycle + 1) % eval_every == 0 or cycle == count - 1):
            _write_json(cycle_dir / "result.json", {**steps.evaluate(epoch), "cycle": cycle})
        _write_json(
            cycle_dir / "cycle.json",
            {"adapter_sha256": adapter_sha, "seconds": time.time() - cycle_started},
        )
        final = directory / f"cycle-{cycle:03d}"
        os.replace(cycle_dir, final)
        record = _cycle_record(final)
        cycles.append(record)
        del dreams
        gc.collect()
        heartbeat()
        done = cycle + 1 - first
        elapsed = time.time() - started
        accuracy = (
            f"to-learn {record['to_learn_accuracy']:.6f}, "
            f"not-to-forget {record['not_to_forget_accuracy']:.6f}, "
            if "to_learn_accuracy" in record
            else ""
        )
        print(
            f"[{ts()}] epoch {epoch} cycle {cycle + 1}/{stop}: "
            f"forced closes {record['forced_closes']}/{len(cycle_rows)}, "
            f"role-marker closes {record['role_marker_closes']}/{len(cycle_rows)}, "
            f"duplicate dreams {record['duplicate_dreams']}/{record['dreams']}, "
            f"loss {float(record['loss']):.4f}, {accuracy}"
            f"{elapsed / done:.1f} s/cycle ETA {fmt_duration(elapsed / done * (stop - cycle - 1))}",
            flush=True,
        )
        if deadline is not None and time.time() > deadline:
            break

    def total(key: str) -> int:
        return sum(int(cycle[key]) for cycle in cycles)

    return {
        "kind": "dream",
        "cycles": cycles,
        "optimizer_steps": total("optimizer_steps"),
        "token_gradients": total("token_gradients"),
        "generated_tokens": total("generated_tokens"),
        "review_tokens": 0,
        "dreams": total("dreams"),
        "forced_closes": total("forced_closes"),
        "role_marker_closes": total("role_marker_closes"),
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

    Each epoch is built in work/epoch-EE/ and moved to epoch-EE/ once its
    result is saved, so a crash leaves no partial epoch-EE/; a dream epoch
    resumes after its last complete cycle. Every epoch starts from fresh
    state. Returns the curve, one row per epoch.

    settings["cycles"] stops the run after a truncated, unevaluated epoch 1;
    settings["cycles_per_run"] and settings["max_hours"] stop it mid-epoch
    with summary.json carrying cycles_run; max_hours also stops it after the
    first epoch that ends past the cap.
    """
    arm = str(settings["arm"])
    max_cycles = settings.get("cycles")
    max_hours = settings.get("max_hours")
    run_started = time.time()
    deadline = None if max_hours is None else run_started + float(max_hours) * 3600

    def write_summary(**extra: object) -> dict[str, object]:
        summary = {
            "arm": arm,
            "seed": settings["seed"],
            "settings": dict(settings),
            "curve": curve,
            "checkpoint": curve_summary(curve),
            **extra,
        }
        _write_json(output / "summary.json", summary)
        return summary

    for epoch in range(len(curve), int(settings["epochs"]) + 1):
        started = time.time()
        work = output / "work" / f"epoch-{epoch:02d}"
        work.mkdir(parents=True, exist_ok=True)
        wake_sha: str | None = None
        wake_tokens: int | None = None
        treatment: dict[str, object]
        if epoch == 0:
            treatment = {"kind": "initial", "optimizer_steps": 0, "review_tokens": 0}
        elif arm == "dream":
            docs_per_wake = int(settings["docs_per_wake"])
            treatment = _dream_epoch(
                epoch,
                work,
                rows,
                docs_per_wake,
                steps,
                max_cycles,
                int(settings.get("eval_every") or 1),
                settings.get("cycles_per_run"),
                deadline,
            )
            cycles = treatment["cycles"]
            if max_cycles is None and len(cycles) < cycles_per_epoch(len(rows), docs_per_wake):
                write_summary(cycles_run=len(cycles))
                print(
                    f"[{ts()}] stopped after cycle {len(cycles) - 1} of epoch {epoch}; "
                    "rerun with the same settings to resume",
                    flush=True,
                )
                return curve
            wake_sha = hashlib.sha256(
                ",".join(str(cycle["wake_sha256"]) for cycle in cycles).encode()
            ).hexdigest()
            wake_tokens = sum(int(cycle["wake_tokens"]) for cycle in cycles)
        elif arm in ("lora", "mix-review"):
            treatment = {"kind": arm, **steps.train_documents(epoch)}
            heartbeat()
        else:
            raise ValueError(f"the {arm} arm has no treatment epochs")
        acceptance = max_cycles is not None and epoch > 0
        last_cycle = work / f"cycle-{len(treatment.get('cycles', [])) - 1:03d}" / "result.json"
        if acceptance:
            result = {"epoch": epoch}
        elif last_cycle.exists():
            # The last cycle scored these same weights; its result is the epoch's.
            result = {**json.loads(last_cycle.read_text()), "epoch": epoch}
            result.pop("cycle")
        else:
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
        if acceptance:
            print(
                f"[{ts()}] acceptance epoch {epoch} saved after {max_cycles} cycles, "
                "not evaluated; stopping",
                flush=True,
            )
            return curve
        curve.append(result)
        summary = write_summary()
        heartbeat()
        print(
            f"[{ts()}] completed epoch {epoch}/{settings['epochs']}: "
            f"{json.dumps(summary['checkpoint'], sort_keys=True)}",
            flush=True,
        )
        hours = (time.time() - run_started) / 3600
        if deadline is not None and time.time() > deadline and epoch < int(settings["epochs"]):
            print(
                f"[{ts()}] soft cap of {max_hours} h reached after epoch {epoch} "
                f"({hours:.2f} h); rerun the same command to resume",
                flush=True,
            )
            return curve
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


def _save_trainable(model, path: Path, rank: int, alpha: float, optimizer=None) -> str:
    """Save the trainable tensors, the fp32 master copies where optimizer holds them."""
    import torch

    masters = (
        {}
        if optimizer is None
        else dict(zip(map(id, optimizer.params), optimizer.masters, strict=True))
    )
    state = {
        name: masters.get(id(value), value).detach().cpu()
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

    def wake(self, rows, state, epoch, cycle):
        return run_conversational_wake(
            self.model,
            self.tokenizer,
            [str(row["evidence"]) for row in rows],
            state,
            *self.markers,
            reply_tokens=REPLY_TOKENS,
            evidence_tokens=EVIDENCE_TOKENS,
            device=self.device,
            temperature=float(self.settings["reply_temperature"]),
            seed=dream_set_seed(self.args.seed, epoch, cycle),
            frame=draw_wake_frame(str(self.settings["wake_frame"]["mode"]), epoch, cycle),
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
            temperature=float(self.settings["dream_temperature"]),
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

    def distil(self, dreams, label):
        from experiments.dream_distillation import distill_dream_batch

        started = time.time()
        user_id, asst_id, eoc_id = map(self.tokenizer.convert_tokens_to_ids, self.markers)
        distilled = distill_dream_batch(
            self.model,
            self.optimizer,
            dreams,
            KL_TEMPERATURE,
            user_id=int(user_id),
            asst_id=int(asst_id),
            eos_id=int(self.tokenizer.eos_token_id),
            eoc_id=int(eoc_id),
            loss_weights=self.settings["dream_loss_weights"],
            clip_norm=self.settings["clip_norm"],
        )
        peak = distilled["max_dream_loss"]
        grad_norm = distilled["grad_norm"]
        losses = " ".join(
            "-" if value is None else f"{value:.4f}" for value in distilled["dream_losses"]
        )
        print(
            f"[{ts()}] {label} distilled {distilled['scored_dreams']}/{len(dreams)} dreams, "
            f"loss {distilled['loss']:.4f}, max {'n/a' if peak is None else f'{peak:.4f}'}, "
            f"dream losses [{losses}], "
            f"grad norm {'n/a' if grad_norm is None else f'{grad_norm:.4f}'}"
            f"{' (clipped)' if distilled['clipped'] else ''}",
            flush=True,
        )
        if peak is not None and peak < DREAM_LOSS_FLOOR:
            print(
                f"[{ts()}] WARNING: {label} max dream loss {peak:.4f} is under "
                f"{DREAM_LOSS_FLOOR}; are teacher and student the same function?",
                flush=True,
            )
        return {
            "optimizer_steps": 1 if distilled["scored_dreams"] > 0 else 0,
            "token_gradients": distilled["token_gradients"],
            "loss": distilled["loss"],
            "dream_losses": distilled["dream_losses"],
            "max_dream_loss": peak,
            "scored_dreams": distilled["scored_dreams"],
            "grad_norm": grad_norm,
            "clipped": distilled["clipped"],
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

    def save_cycle(self, directory):
        import torch

        rank, alpha = int(self.settings["lora_rank"]), float(self.settings["lora_alpha"])
        self.teacher_sha = _save_trainable(
            self.model, directory / "trainable.pt", rank, alpha, self.optimizer
        )
        state = self.optimizer.state_dict()
        # SGD without momentum holds no state; its masters are trainable.pt.
        if state["optimizer"]["state"]:
            torch.save(state, directory / "optimizer.pt")
        return self.teacher_sha

    def restore_cycle(self, directory):
        """Load the cycle's fp32 masters straight into the optimizer, never through a bf16 cast."""
        import torch

        adapter = torch.load(directory / "trainable.pt", map_location="cpu", weights_only=True)
        names = {id(value): name for name, value in self.model.named_parameters()}
        saved = directory / "optimizer.pt"
        inner = (
            torch.load(saved, map_location="cpu", weights_only=True)["optimizer"]
            if saved.exists()
            else self.optimizer.optimizer.state_dict()
        )
        self.optimizer.load_state_dict(
            {"masters": [adapter[names[id(p)]] for p in self.optimizer.params], "optimizer": inner}
        )
        self.teacher_sha = file_sha(directory / "trainable.pt")

    def save(self, directory, result):
        import torch

        rank, alpha = int(self.settings["lora_rank"]), float(self.settings["lora_alpha"])
        self.teacher_sha = _save_trainable(
            self.model, directory / "trainable.pt", rank, alpha, self.optimizer
        )
        result["adapter_sha256"] = self.teacher_sha
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        result["source_tokens"] = (
            0 if result["epoch"] == 0 else int(self.settings["source_document_tokens"])
        )
        result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()


def _resume(model, optimizer, last: Path) -> None:
    """Load the last epoch's adapter; a trained arm also restores its fp32 masters."""
    import torch

    from training.checkpoints import load_checkpoint

    load_checkpoint(model, last)
    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(last / "optimizer.pt", map_location="cpu", weights_only=True)
        )


def _completed_epochs(output: Path) -> list[Path]:
    paths = sorted(path for path in output.glob("epoch-*") if (path / "result.json").exists())
    for expected, path in enumerate(paths):
        if path.name != f"epoch-{expected:02d}":
            raise SystemExit(f"non-contiguous completed epochs under {output}: {path.name}")
    return paths


def wake_frame_settings(mode: str) -> dict[str, object]:
    if mode == "single":
        return {"mode": mode}
    return {
        "mode": mode,
        "set_sha256": hashlib.sha256(json.dumps(WAKE_FRAMES).encode()).hexdigest(),
    }


def treatment_settings(args: argparse.Namespace) -> dict[str, object]:
    """The run.json entries the dream cell's flags set."""
    dream = args.arm == "dream"
    return {
        "wake_frame": wake_frame_settings(args.wake_frames) if dream else None,
        "reply_temperature": args.dream_temperature if dream else None,
        "dream_temperature": args.dream_temperature if dream else None,
        "dream_loss_weights": args.dream_loss_weights if dream else None,
        "fp32_forward": args.fp32_forward,
        "optimizer": None if args.arm == "frozen" else args.optimizer,
        "dream_lr": args.lr,
        "clip_norm": args.clip_norm,
        "eval_every": args.eval_every if dream else None,
        "cycles_per_run": args.cycles_per_run,
        "max_hours": args.max_hours,
    }


def _same_run(existing: Mapping[str, object], current: Mapping[str, object]) -> bool:
    free = dict.fromkeys(SESSION_LIMITS)
    return {**existing, **free} == {**current, **free}


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
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--docs-per-wake",
        type=int,
        default=DOCS_PER_WAKE,
        help="documents per dream wake-sleep cycle",
    )
    parser.add_argument(
        "--dreams-per-cycle",
        type=int,
        default=DREAMS_PER_CYCLE,
        help="dreams generated and distilled per dream wake-sleep cycle",
    )
    parser.add_argument(
        "--dream-batch-size", type=int, help="dreams per generation launch; --dreams-per-cycle"
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="engineering-only gate: one epoch, two documents per set, "
        "at most two documents per wake, two 32-token dreams per cycle",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        help="dream acceptance check: after the initial evaluation, run the first N "
        "cycles of epoch 1, save them unevaluated, and stop; needs a fresh --output",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        help="stop after the first cycle or epoch that ends past H hours of this session; "
        "the same command resumes",
    )
    parser.add_argument(
        "--cycles-per-run",
        type=int,
        help="stop after cycle N-1 of the epoch (absolute) and write summary.json with "
        "cycles_run; a rerun with a larger N continues the epoch",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="score a dream cycle when (cycle + 1) %% N == 0, and the epoch's last cycle",
    )
    parser.add_argument("--wake-frames", choices=("single", "set"), default="single")
    parser.add_argument(
        "--dream-temperature",
        type=float,
        default=DREAM_TEMPERATURE,
        help="sampling temperature of the wake replies and the dreams",
    )
    parser.add_argument("--dream-loss-weights", choices=("turn", "uniform"), default="turn")
    parser.add_argument(
        "--fp32-forward",
        action="store_true",
        help="hold the whole model in fp32, so the adapter's masters are its parameters",
    )
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float, help="the dream arm's SGD learning rate")
    parser.add_argument(
        "--clip-norm", type=float, help="the dream arm's SGD gradient-norm clip, before the lr"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dream_batch_size is None:
        args.dream_batch_size = args.dreams_per_cycle
    if args.optimizer == "sgd":
        if args.arm != "dream":
            parser.error("--optimizer sgd is for --arm dream")
        if args.lr is None or args.clip_norm is None or min(args.lr, args.clip_norm) <= 0:
            parser.error("--optimizer sgd needs a positive --lr and --clip-norm")
    elif args.lr is not None or args.clip_norm is not None:
        parser.error("--lr and --clip-norm need --optimizer sgd")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")
    if args.cycles_per_run is not None and (args.cycles_per_run < 1 or args.cycles is not None):
        parser.error("--cycles-per-run needs a positive N and no --cycles")
    return args


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("LAMA-CKL treatment runs require a CUDA GPU with the fused Mamba path")
    if min(args.dream_batch_size, args.eval_batch_size, args.docs_per_wake) < 1:
        raise SystemExit("batch sizes and --docs-per-wake must be positive")
    if args.dreams_per_cycle < 1:
        raise SystemExit("--dreams-per-cycle must be positive")
    if args.cycles is not None:
        if args.arm != "dream" or args.output is None or args.cycles < 1:
            raise SystemExit("--cycles needs --arm dream, --output, and a positive N")
        if (args.output / "run.json").exists():
            raise SystemExit(f"--cycles needs a fresh --output: {args.output}")
    pinned = pinned_warmstart_sha()
    torch.cuda.reset_peak_memory_stats()
    learned, retained, split_manifest = _load_split(args.split)
    epochs, docs_per_wake = args.epochs, args.docs_per_wake
    dreams_per_cycle, dream_tokens = args.dreams_per_cycle, DREAM_TOKENS
    if args.smoke:
        learned, retained = learned[:2], retained[:2]
        epochs, docs_per_wake, dreams_per_cycle, dream_tokens = 1, min(docs_per_wake, 2), 2, 32
    if args.arm == "frozen":
        epochs = 0
    dream_arm = args.arm == "dream"
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
        "docs_per_wake": docs_per_wake if dream_arm else None,
        "cycles_per_epoch": cycles_per_epoch(len(learned), docs_per_wake) if dream_arm else 0,
        "train_batch_size": min(TRAIN_BATCH_SIZE, len(learned)),
        "learning_rate": LEARNING_RATE,
        "evidence_tokens": EVIDENCE_TOKENS,
        "reply_tokens": REPLY_TOKENS,
        "dreams_per_cycle": dreams_per_cycle if dream_arm else 0,
        "dream_tokens": dream_tokens if dream_arm else 0,
        "kl_temperature": KL_TEMPERATURE if dream_arm else None,
        "requested_dream_batch_size": args.dream_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "model_name": args.model_name,
        "warmstart_sha256": warm_sha,
        "split_manifest_sha256": file_sha(args.split / "manifest.json"),
        "cycles": args.cycles,
        "adapter_dtype": None if args.arm == "frozen" else "fp32-master",
        **treatment_settings(args),
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
    from training.master_weights import MasterAdamW, MasterOptimizer, fp32_forward

    if args.fp32_forward:
        fp32_forward(model)
    load_checkpoint(model, args.init_adapter)
    tokenizer = build_tokenizer(model_mod)
    optimizer: MasterOptimizer | None = None
    if args.optimizer == "sgd":
        optimizer = MasterOptimizer(
            trainable, torch.optim.SGD, lr=args.lr, momentum=0.0, weight_decay=0.0
        )
    elif args.arm != "frozen":
        optimizer = MasterAdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
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
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        current = json.loads(json.dumps(settings))
        existing.setdefault("cycles", None)
        # A run without the key is bf16 (refused) or frozen (no adapter to train).
        existing.setdefault("adapter_dtype", None)
        if not _same_run(existing, current):
            raise SystemExit(f"existing run uses different settings: {manifest_path}")
    manifest_path.write_text(json.dumps(settings, indent=1, sort_keys=True) + "\n")

    completed = _completed_epochs(output)
    curve = [json.loads((path / "result.json").read_text()) for path in completed]
    teacher_sha = warm_sha
    if completed:
        last = completed[-1]
        _resume(model, optimizer, last)
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
