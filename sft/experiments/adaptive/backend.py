"""Experiment: state-erasure

Model-backed adaptive experiment backend."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from experiments.adaptive.manifest import ExperimentManifest, token_sha
from experiments.adaptive.wake import CommandUserGenerator, LiveWakeHarness


class RuntimeBackend(Protocol):
    initial_state: object

    def fork_state(self, state: object) -> object: ...
    def wake(
        self, arm: str, wake: int, state: object, facts: Sequence[tuple[str, str]], scenario: str
    ) -> tuple[dict[str, object], object]: ...
    def sleep(self, arm: str, wake: int, state: object, artifact: dict[str, object]) -> object: ...
    def probe(
        self, arm: str, wake: int, state: object, artifact: dict[str, object]
    ) -> dict[str, object]: ...
    def execution_metadata(self) -> dict[str, object]: ...


def artifact_transcript_ids(artifact: dict[str, object]) -> list[int]:
    ids, recorded = artifact.get("transcript_token_ids"), artifact.get("transcript_token_sha256")
    if not isinstance(ids, list) or any(not isinstance(token, int) for token in ids):
        raise RuntimeError("wake artifact has no exact transcript token sequence")
    if token_sha(ids) != recorded:
        raise RuntimeError("wake transcript token SHA mismatch")
    return ids


def dream_cache_identity(
    teacher_sha: str,
    transcript_sha: str,
    state_sha: str,
    facts: Sequence[tuple[str, str]],
    settings: dict[str, object],
    seed: int,
    wake: int,
) -> dict[str, object]:
    value: dict[str, object] = {
        "teacher_sha256": teacher_sha,
        "transcript_token_sha256": transcript_sha,
        "wake_state_sha256": state_sha,
        "facts": [list(fact) for fact in facts],
        "dream_settings": settings,
        "seed": seed,
        "wake": wake,
    }
    value["identity_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    return value


def battery_collision_terms(
    manifest: ExperimentManifest, user_open: str, asst_open: str
) -> list[str]:
    facts = [fact for spec in manifest.wakes for fact in spec.facts]
    return (
        [spec.scenario for spec in manifest.wakes]
        + [goal for spec in manifest.wakes for goal in spec.plan.turn_goals]
        + [
            f"{user_open} What is the code for the {entity}?{asst_open} "
            f"The code for the {entity} is"
            for entity, _ in facts
        ]
        + [part for fact in facts for part in fact]
    )


def manifest_payload(manifest: ExperimentManifest) -> dict[str, object]:
    return {
        "arms": list(manifest.config.arms),
        "seeds": list(manifest.config.seeds),
        "wakes": [
            {
                "scenario": spec.scenario,
                "facts": [{"entity": entity, "code": code} for entity, code in spec.facts],
                "turn_count": spec.plan.turn_count,
                "injection_turns": list(spec.plan.injection_turns),
                "turn_goals": list(spec.plan.turn_goals),
            }
            for spec in manifest.wakes
        ],
        "generator": manifest.generator,
        "output_root": manifest.output_root,
        "batch_sizes": manifest.batch_sizes,
        "runtime": manifest.runtime,
    }


def manifest_sha(manifest: ExperimentManifest) -> str:
    return hashlib.sha256(
        json.dumps(
            manifest_payload(manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def bound_rehearsal_counts(dreams: Sequence[object], facts: Sequence[object]) -> dict[str, int]:
    from experiments.dreams.cache import binding_coverage

    counts = {str(fact.entity): 0 for fact in facts}
    for dream in dreams:
        bound, _ = binding_coverage("".join(dream.token_texts), facts)
        for entity, count in bound.items():
            counts[entity] += count
    return counts


@dataclass
class _ArmContext:
    model: object
    optimizer: object


class DreamSleepBackend:
    """Concrete adapter from the coordinator to dream_sleep's model machinery."""

    def __init__(self, manifest: ExperimentManifest, seed: int):
        import importlib

        from models.common import build_tokenizer

        import torch

        from adapters.lora import DEFAULT_DROPOUT
        from experiments.dreams.cli import build_distractors, load_init_adapter
        from experiments.facts import Fact
        from experiments.locality import (
            BATTERY_CANDIDATES,
            HELDOUT_TEXT,
            load_or_build_battery_batched,
            perplexity,
            validate_battery_candidates,
        )

        if seed not in manifest.config.seeds:
            raise RuntimeError(f"seed {seed} is not registered in the manifest")
        self.manifest, self.seed, self.initial_state = manifest, seed, None
        self.runtime, self.torch = manifest.runtime, torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = str(self.runtime["model"])
        model_mod = importlib.import_module(f"models.{model_name}")
        hooks = importlib.import_module(f"models.{model_name}.train_hooks")
        self.tokenizer = build_tokenizer(model_mod)
        self.user_open, self.asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
        self.user_id = self.tokenizer.convert_tokens_to_ids(self.user_open)
        self.asst_id = self.tokenizer.convert_tokens_to_ids(self.asst_open)
        eoc = getattr(model_mod, "EOC", None)
        self.eoc_id = self.tokenizer.convert_tokens_to_ids(eoc) if eoc else None
        self.turn_id = self.tokenizer.eos_token_id
        self.contexts: dict[str, _ArmContext] = {}
        hashes: set[str] = set()
        model_hashes: set[str] = set()
        for arm in manifest.config.arms:
            torch.manual_seed(seed)
            print(f"[{self._ts()}] loading independent {arm} model", flush=True)
            model, trainable = hooks.setup_training(
                self.device,
                int(self.runtime["lora_rank"]),
                float(self.runtime["lora_alpha"]),
                DEFAULT_DROPOUT,
            )
            digest = load_init_adapter(
                model,
                str(self.runtime["warm_start"]),
                int(self.runtime["lora_rank"]),
                float(self.runtime["lora_alpha"]),
            )
            hashes.add(digest)
            hash_started = time.time()
            print(f"[{self._ts()}] hashing independent {arm} full model", flush=True)
            model_hashes.add(self._model_hash(model))
            print(
                f"[{self._ts()}] hashed independent {arm} full model in "
                f"{time.time() - hash_started:.1f}s",
                flush=True,
            )
            model.eval()
            optimizer = torch.optim.AdamW(trainable, lr=float(self.runtime["learning_rate"]))
            self.contexts[arm] = _ArmContext(model, optimizer)
        expected = str(self.runtime["warm_start_sha256"])
        if hashes != {expected}:
            raise RuntimeError(
                f"warm-start SHA mismatch: expected {expected}, loaded {sorted(hashes)}"
            )
        if len(model_hashes) != 1:
            raise RuntimeError("independent arm models do not start from identical weights")
        self.initial_full_sha = next(iter(model_hashes))
        self.adapter_sha = expected
        self.output = Path(manifest.output_root) / f"seed-{seed}"
        self.output.mkdir(parents=True, exist_ok=True)
        manifest_value = manifest_payload(manifest)
        self.manifest_sha = manifest_sha(manifest)
        self._store_json(
            self.output / "manifest.json", {"sha256": self.manifest_sha, "manifest": manifest_value}
        )
        self.topology: list[dict[str, object]] = []
        self.effective_batches = dict(manifest.batch_sizes)
        self._batch_locked: set[str] = set()
        self.dream_seconds = 0.0
        self.dreams_generated = 0
        self.retries = 0
        self._probe_verified: set[str] = set()
        self._peak_vram = 0
        self.treatments: dict[tuple[str, int], dict[str, object]] = {}

        all_facts = [
            Fact(entity, "entity", code) for wake in manifest.wakes for entity, code in wake.facts
        ]
        self.facts = {fact.entity: fact for fact in all_facts}
        self.fact_waves = {
            entity: wake
            for wake, spec in enumerate(manifest.wakes, start=1)
            for entity, _ in spec.facts
        }
        self.distractors = build_distractors(all_facts, seed)
        validate_battery_candidates(
            BATTERY_CANDIDATES,
            lambda answer: len(self.tokenizer(" " + answer, add_special_tokens=False)["input_ids"]),
            battery_collision_terms(manifest, self.user_open, self.asst_open),
        )
        battery_path = Path(str(self.runtime["battery"]))
        self.battery = load_or_build_battery_batched(
            battery_path,
            BATTERY_CANDIDATES,
            lambda batch: [
                (generation, mean)
                for generation, mean, _ in self._probe_pairs(
                    "replay", list(batch), manifest.batch_sizes["battery"], "battery-calibration"
                )
            ],
            batch_size=manifest.batch_sizes["battery"],
            checkpoint_sha=expected,
        )
        if len(self.battery) < 100:
            raise RuntimeError(
                f"the self-calibrated knowledge battery kept {len(self.battery)} items; need 100"
            )
        self.heldout = self._encode(HELDOUT_TEXT)
        self.base_ppl = perplexity(
            self.contexts["replay"].model,
            self.heldout,
            int(self.runtime["chunk_len"]),
            "heldout baseline",
        )

    @staticmethod
    def _ts() -> str:
        from progress import ts

        return ts()

    def _model_hash(self, model: object) -> str:
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            value = parameter.detach().cpu().contiguous()
            digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _teacher_hash(self, model: object) -> str:
        digest = hashlib.sha256(self.initial_full_sha.encode())
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.detach().cpu().contiguous()
            digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
            digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _encode(self, text: str):
        return self.torch.tensor(
            [self.tokenizer(text, add_special_tokens=False)["input_ids"]],
            dtype=self.torch.long,
            device=self.device,
        )

    def _decode(self, ids: object) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _context(self, arm: str) -> _ArmContext:
        return self.contexts["replay" if arm == "shared" else arm]

    def fork_state(self, state: object) -> object:
        from experiments.dreams.generation import copy_state

        return copy_state(state)

    def _sample_reply(
        self, arm: str, wake: int, turn: int, user: str, state: object
    ) -> tuple[dict[str, object], object]:
        from experiments.dreams.generation import sample_next

        model = self._context(arm).model
        prompt = self._encode(f"{self.user_open} {user}{self.asst_open} ")
        seed_material = f"{self.seed}:{arm}:{wake}:{turn}".encode()
        reply_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (2**63 - 1)
        generator = self.torch.Generator(device=self.device).manual_seed(reply_seed)
        pieces: list[int] = []
        stop_reason: str | None = None
        model.eval()
        with self.torch.no_grad():
            logits, state = model(prompt, state=state)
            for _ in range(int(self.runtime["reply_tokens"])):
                token = sample_next(
                    logits[:, -1], float(self.runtime["reply_temperature"]), generator
                )
                value = int(token.item())
                if value == self.eoc_id:
                    raise RuntimeError("model emitted EOC inside a registered wake")
                pieces.append(value)
                logits, state = model(token, state=state)
                if value == self.turn_id:
                    stop_reason = "eos"
                    break
        if stop_reason is None:
            raise RuntimeError("assistant reply exhausted reply_tokens before EOS")
        visible = pieces[:-1] if pieces and pieces[-1] == self.turn_id else pieces
        return {
            "text": self._decode(visible),
            "prompt_token_ids": [int(token) for token in prompt[0].tolist()],
            "assistant_token_ids": pieces,
            "stop_reason": stop_reason,
        }, state.detach()

    def _state_hash(self, state: object) -> str:
        digest = hashlib.sha256()
        for attr in ("conv_states", "ssm_states"):
            for tensor in getattr(state, attr, []):
                value = tensor.detach().cpu().contiguous()
                digest.update(f"{attr}:{tuple(value.shape)}:{value.dtype}".encode())
                digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _store_state(self, arm: str, wake: int, state: object) -> dict[str, object]:
        from experiments.dreams.generation import copy_state, state_to

        path = self.output / "states" / f"{arm}_w{wake}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = self._state_hash(state)
        if path.exists():
            existing = self.torch.load(path, map_location="cpu", weights_only=False)
            if self._state_hash(existing) != digest:
                raise RuntimeError(f"wake state already exists with different content: {path}")
        else:
            self.torch.save(state_to(copy_state(state), self.torch.device("cpu")), path)
        return {"path": str(path), "sha256": digest}

    @staticmethod
    def _store_json(path: Path, value: dict[str, object]) -> None:
        serialized = json.dumps(value, indent=1, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != serialized:
            raise RuntimeError(f"artifact already exists with different content: {path}")
        if not path.exists():
            path.write_text(serialized)

    def wake(
        self, arm: str, wake: int, state: object, facts: Sequence[tuple[str, str]], scenario: str
    ) -> tuple[dict[str, object], object]:
        spec = self.manifest.wakes[wake - 1]
        generator = CommandUserGenerator(
            [str(part) for part in self.manifest.generator["command"]],
            provenance={
                key: str(self.manifest.generator[key]) for key in ("provider", "model", "version")
            },
        )
        harness = LiveWakeHarness(spec.plan, generator, self.output / "wakes")
        holder = [state]
        self.torch.manual_seed(
            self.seed * 10_000
            + wake * 10
            + (0 if arm == "shared" else self.manifest.config.arms.index(arm))
        )

        def reply(user: str, turn: int) -> dict[str, object]:
            result, holder[0] = self._sample_reply(arm, wake, turn, user, holder[0])
            return result

        def replay_turn(turn: dict[str, object]) -> None:
            model = self._context(arm).model
            with self.torch.no_grad():
                prompt = self.torch.tensor(
                    [turn["prompt_token_ids"]], dtype=self.torch.long, device=self.device
                )
                _, holder[0] = model(prompt, state=holder[0])
                for token in turn["assistant_token_ids"]:
                    value = self.torch.tensor([[token]], dtype=self.torch.long, device=self.device)
                    _, holder[0] = model(value, state=holder[0])
            holder[0] = holder[0].detach()

        artifact = harness.run(
            arm,
            wake,
            facts,
            reply,
            scenario,
            None,
            state_metadata=lambda: self._store_state(arm, wake, holder[0]),
            fact_distances=lambda turns: self._fact_distances(
                turns, facts, spec.plan.injection_turns
            ),
            replay_turn=replay_turn,
        )
        if arm == "shared":
            for target in self.manifest.config.arms:
                clone = dict(artifact)
                clone["arm"] = target
                clone["artifact_sha256"] = hashlib.sha256(
                    json.dumps(
                        {key: value for key, value in clone.items() if key != "artifact_sha256"},
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest()
                harness.store(clone)
        return artifact, holder[0]

    def _fact_distances(
        self,
        turns: Sequence[dict[str, object]],
        facts: Sequence[tuple[str, str]],
        injection_turns: Sequence[int],
    ) -> dict[str, int]:
        transcript = [
            token
            for turn in turns
            for key in ("prompt_token_ids", "assistant_token_ids")
            for token in turn[key]
        ]
        distances: dict[str, int] = {}
        consumed = 0
        targets = dict(zip(injection_turns, facts, strict=True))
        all_codes = {fact.code: fact.entity for fact in self.facts.values()}
        current_positions = {code: turn for turn, (_, code) in targets.items()}

        def future(code: str, turn: int) -> bool:
            return code not in current_positions or current_positions[code] > turn

        for index, turn in enumerate(turns, start=1):
            user = str(turn["user"])
            if index in targets:
                entity, code = targets[index]
                if user.count(code) != 1:
                    raise RuntimeError(
                        f"injection turn for {entity} states its code {user.count(code)} times"
                    )
                leaked = [
                    other
                    for other in all_codes
                    if other != code and other in user and future(other, index)
                ]
                if leaked:
                    raise RuntimeError(
                        f"injection turn for {entity} also introduces target codes {leaked}"
                    )
                prompt_text = f"{self.user_open} {user}{self.asst_open} "
                encoded = self.tokenizer(
                    prompt_text, add_special_tokens=False, return_offsets_mapping=True
                )
                prompt_ids = [int(token) for token in turn["prompt_token_ids"]]
                if encoded["input_ids"] != prompt_ids:
                    raise RuntimeError("stored wake prompt tokens do not match tokenizer offsets")
                code_end = prompt_text.index(code) + len(code)
                end_token = next(
                    (
                        position + 1
                        for position, (_, end) in enumerate(encoded["offset_mapping"])
                        if end >= code_end
                    ),
                    None,
                )
                if end_token is None:
                    raise RuntimeError(
                        f"cannot locate {entity}'s code in the exact prompt token stream"
                    )
                distances[entity] = len(transcript) - (consumed + end_token)
            else:
                leaked = [code for code in all_codes if code in user and future(code, index)]
                if leaked:
                    raise RuntimeError(
                        f"non-injection turn {index} introduces target codes {leaked}"
                    )
            consumed += len(turn["prompt_token_ids"]) + len(turn["assistant_token_ids"])
        return distances

    def _wake_facts(self, wake: int):
        return [
            self.facts[entity] for spec in self.manifest.wakes[:wake] for entity, _ in spec.facts
        ]

    def sleep(self, arm: str, wake: int, state: object, artifact: dict[str, object]) -> object:
        from experiments.dreams.cache import load_dream_cache, save_dream_cache
        from experiments.dreams.distillation import distill_dream_set, distill_sft, sft_steps
        from experiments.dreams.generation import (
            copy_state,
            dream_generation_seed,
            generate_replay_dreams,
            state_to,
        )
        from experiments.dreams.probes import dream_is_degenerate
        from experiments.dreams.types import DreamSetCache, token_sha

        context = self.contexts[arm]
        if arm == "nosleep":
            print(f"[{self._ts()}] {arm} wake {wake}: no generation and no training", flush=True)
            self.treatments[arm, wake] = {
                "dreams_generated": 0,
                "dream_passes": 0,
                "raw_transcript_passes": 0,
                "state_carry": "intact",
            }
            self.topology.append(
                {
                    "operation": "sleep",
                    "arm": arm,
                    "wake": wake,
                    "batches": [],
                    "items_per_second": None,
                    "retries": 0,
                    "peak_vram_bytes": (
                        int(self.torch.cuda.max_memory_allocated())
                        if self.torch.cuda.is_available()
                        else 0
                    ),
                }
            )
            return state
        transcript = self.torch.tensor(
            [artifact_transcript_ids(artifact)], dtype=self.torch.long, device=self.device
        )
        if arm == "sft-ref":
            steps = sft_steps(transcript.shape[1], int(self.runtime["chunk_len"]))
            started = time.time()
            context.model.train()
            distill_sft(
                context.model,
                context.optimizer,
                transcript,
                steps,
                int(self.runtime["chunk_len"]),
                lambda step, loss: print(
                    f"[{self._ts()}] sft-ref wake {wake} step {step + 1}/{steps} loss {loss:.4f}",
                    flush=True,
                ),
            )
            context.model.eval()
            elapsed = time.time() - started
            self.treatments[arm, wake] = {
                "dreams_generated": 0,
                "dream_passes": 0,
                "raw_transcript_passes": 1,
                "token_gradients": transcript.shape[1] - 1,
                "state_carry": "none",
            }
            length = transcript.shape[1] - 1
            chunks = [
                min(int(self.runtime["chunk_len"]), length - start)
                for start in range(0, length, int(self.runtime["chunk_len"]))
            ]
            self.topology.append(
                {
                    "operation": "sft",
                    "arm": arm,
                    "wake": wake,
                    "batches": chunks,
                    "seconds": elapsed,
                    "items_per_second": length / elapsed if elapsed else None,
                    "retries": 0,
                    "peak_vram_bytes": (
                        int(self.torch.cuda.max_memory_allocated())
                        if self.torch.cuda.is_available()
                        else 0
                    ),
                }
            )
            return None

        path = self.output / "dreams" / f"replay_w{wake}.pt"
        wave_facts = self._wake_facts(wake)
        teacher_sha = self._teacher_hash(context.model)
        settings = {
            key: self.runtime[key]
            for key in (
                "dream_count",
                "dream_tokens",
                "dream_temperature",
                "kl_temperature",
                "chunk_len",
            )
        }
        identity = dream_cache_identity(
            teacher_sha,
            str(artifact["transcript_token_sha256"]),
            self._state_hash(state),
            [fact for spec in self.manifest.wakes[:wake] for fact in spec.facts],
            settings,
            self.seed,
            wake,
        )
        if path.exists():
            cache = load_dream_cache(path)
            metadata = json.loads(path.with_suffix(".json").read_text())
            cached_counts = (
                bound_rehearsal_counts(cache.dreams, wave_facts)
                if isinstance(cache, DreamSetCache)
                else {}
            )
            if (
                not isinstance(cache, DreamSetCache)
                or len(cache.dreams) != 300
                or cache.generator != teacher_sha
                or cache.transcript_sha != identity["transcript_token_sha256"]
                or self._state_hash(cache.wake_state) != identity["wake_state_sha256"]
                or metadata.get("cache_identity") != identity
                or metadata.get("bound_rehearsals") != cached_counts
            ):
                raise RuntimeError(f"replay cache does not match the registered treatment: {path}")
            topology = [int(width) for width in metadata["batch_topology"]]
            loaded_batch = int(metadata["effective_batch_size"])
            if "dream" in self._batch_locked and loaded_batch != self.effective_batches["dream"]:
                raise RuntimeError("cached dream cells use inconsistent effective batch sizes")
            self.effective_batches["dream"] = loaded_batch
            self._batch_locked.add("dream")
        else:
            started = time.time()
            batch_size = self.effective_batches["dream"]
            operation_retries = 0
            while True:
                try:
                    dreams, topology = generate_replay_dreams(
                        context.model,
                        state,
                        self._encode(f"{self.asst_open} "),
                        count=300,
                        batch_size=batch_size,
                        seed=self.seed * 100 + wake,
                        n_tokens=int(self.runtime["dream_tokens"]),
                        temperature=float(self.runtime["dream_temperature"]),
                        decode_token=lambda token: self.tokenizer.decode([token]),
                        stop_id=self.eoc_id,
                        turn_id=self.turn_id,
                    )
                    break
                except self.torch.OutOfMemoryError:
                    if batch_size == 1 or "dream" in self._batch_locked:
                        raise
                    self.retries += 1
                    operation_retries += 1
                    batch_size = max(1, batch_size // 2)
                    self.effective_batches["dream"] = batch_size
                    self.torch.cuda.empty_cache()
                    print(
                        f"[{self._ts()}] replay wake {wake}: OOM, retry {self.retries} at "
                        f"dream batch {batch_size}",
                        flush=True,
                    )
            elapsed = time.time() - started
            self._batch_locked.add("dream")
            self.dream_seconds += elapsed
            self.dreams_generated += len(dreams)
            cache = DreamSetCache(
                seed=self.seed,
                transcript_ids=[int(token) for token in transcript[0].tolist()],
                wake_state=state_to(copy_state(state), self.torch.device("cpu")),
                dreams=dreams,
                distractors={fact.entity: self.distractors[fact.entity] for fact in wave_facts},
                facts=[(fact.entity, fact.category, fact.code) for fact in wave_facts],
                generator=teacher_sha,
            )
            if any(dream_is_degenerate("".join(dream.token_texts)) for dream in dreams):
                raise RuntimeError("a replay dream is malformed; refusing the treatment cache")
            role_violations = 0
            for dream in dreams:
                roles = [
                    token for token in dream.dream_ids if token in (self.user_id, self.asst_id)
                ]
                role_violations += bool(roles and roles[0] != self.asst_id)
                role_violations += sum(left == right for left, right in zip(roles, roles[1:]))
            if role_violations:
                raise RuntimeError(
                    f"replay dreams contain {role_violations} malformed role transitions"
                )
            hashes = [dream.dream_sha for dream in dreams]
            if len(set(hashes)) != len(hashes):
                raise RuntimeError("the registered replay set does not contain 300 distinct dreams")
            counts = bound_rehearsal_counts(dreams, wave_facts)
            save_dream_cache(cache, path)
            self._store_json(
                path.with_suffix(".json"),
                {
                    "arm": arm,
                    "wake": wake,
                    "dreams": 300,
                    "set_sha256": cache.set_sha,
                    "teacher_sha256": teacher_sha,
                    "cache_identity": identity,
                    "batch_topology": topology,
                    "bound_rehearsals": counts,
                    "requested_batch_size": self.manifest.batch_sizes["dream"],
                    "effective_batch_size": batch_size,
                    "retries": operation_retries,
                    "seconds": elapsed,
                    "items_per_second": len(dreams) / elapsed if elapsed else None,
                    "peak_vram_bytes": (
                        int(self.torch.cuda.max_memory_allocated())
                        if self.torch.cuda.is_available()
                        else 0
                    ),
                    "invariants": {
                        "distinct": len(set(dream.dream_sha for dream in dreams)),
                        "role_adjacency_violations": role_violations,
                        "degenerate": 0,
                    },
                    "generation_seeds": [
                        dream_generation_seed(self.seed * 100 + wake, index, 0)
                        for index in range(300)
                    ],
                    "dreams_metadata": [
                        {
                            "sha256": dream.dream_sha,
                            "tokens": len(dream.dream_ids),
                            "stop_reason": dream.stop_reason,
                        }
                        for dream in dreams
                    ],
                },
            )
            sidecar = path.with_suffix(".txt")
            sidecar.write_text(
                f"set_sha {cache.set_sha}\ntranscript_sha {token_sha(transcript[0].tolist())}\n"
                f"binding_rehearsals {json.dumps(counts, sort_keys=True)}\n"
                + "".join(
                    f"decoded_sample_{index + 1} {''.join(dream.token_texts)!r}\n"
                    for index, dream in enumerate(dreams[:3])
                )
            )
        self.topology.append(
            {
                "operation": "dream",
                "arm": arm,
                "wake": wake,
                "batches": topology,
                "requested_batch_size": self.manifest.batch_sizes["dream"],
                "effective_batch_size": max(topology) if topology else 0,
            }
        )
        training_started = time.time()
        context.model.train()
        token_gradients = distill_dream_set(
            context.model,
            context.optimizer,
            cache.dreams,
            state,
            None,
            1,
            float(self.runtime["kl_temperature"]),
            lambda step, loss: print(
                f"[{self._ts()}] replay wake {wake} dream {step + 1}/300 loss {loss:.4f}",
                flush=True,
            ),
            lambda index, epoch, step: None,
        )
        context.model.eval()
        training_elapsed = time.time() - training_started
        self.topology.append(
            {
                "operation": "dream-training",
                "arm": arm,
                "wake": wake,
                "batches": [1] * len(cache.dreams),
                "effective_batch_size": 1,
                "seconds": training_elapsed,
                "items_per_second": len(cache.dreams) / training_elapsed
                if training_elapsed
                else None,
                "retries": 0,
                "peak_vram_bytes": (
                    int(self.torch.cuda.max_memory_allocated())
                    if self.torch.cuda.is_available()
                    else 0
                ),
            }
        )
        counts = bound_rehearsal_counts(cache.dreams, wave_facts)
        self.treatments[arm, wake] = {
            "dreams_generated": 300,
            "dream_passes": 300,
            "raw_transcript_passes": 0,
            "token_gradients": token_gradients,
            "state_carry": "none",
            "dream_set_sha256": cache.set_sha,
            "bound_rehearsals": counts,
        }
        return None

    def _score_tensor_batch(
        self, model: object, prompts: object, targets: object
    ) -> list[tuple[str, float, float]]:
        from experiments.inference import generate

        generations = generate(model, prompts, None, int(self.runtime["probe_tokens"]), 0.0)
        sequence = self.torch.cat([prompts, targets], dim=1)
        with self.torch.no_grad():
            logits, _ = model(sequence, state=None)
        logprobs = self.torch.log_softmax(logits.float(), dim=-1)
        start = prompts.shape[1] - 1
        indices = self.torch.arange(start, sequence.shape[1] - 1, device=self.device)
        results = []
        for row in range(prompts.shape[0]):
            values = logprobs[row, indices, targets[row]]
            results.append(
                (self._decode(generations[row].cpu()), float(values.mean()), float(values.sum()))
            )
        return results

    def _probe_pairs(
        self, arm: str, pairs: Sequence[tuple[str, str]], batch_size: int, operation: str
    ) -> list[tuple[str, float, float]]:
        started = time.time()
        key = "battery" if operation.startswith("battery") else "probe"
        batch_size = self.effective_batches[key]
        operation_retries = 0
        model = self.contexts[arm].model
        model.eval()
        encoded = [(self._encode(prompt), self._encode(" " + answer)) for prompt, answer in pairs]
        groups: dict[tuple[int, int], list[int]] = {}
        for index, (prompt, target) in enumerate(encoded):
            groups.setdefault((prompt.shape[1], target.shape[1]), []).append(index)
        while True:
            results: list[tuple[str, float, float] | None] = [None] * len(pairs)
            batches: list[int] = []
            try:
                for indices in groups.values():
                    for start in range(0, len(indices), batch_size):
                        batch = indices[start : start + batch_size]
                        prompts = self.torch.cat([encoded[index][0] for index in batch])
                        targets = self.torch.cat([encoded[index][1] for index in batch])
                        scored = self._score_tensor_batch(model, prompts, targets)
                        if arm not in self._probe_verified and len(batch) > 1:
                            scalar = [
                                self._score_tensor_batch(
                                    model, prompts[row : row + 1], targets[row : row + 1]
                                )[0]
                                for row in range(len(batch))
                            ]
                            for batched, single in zip(scored, scalar, strict=True):
                                if batched[0] != single[0] or abs(batched[2] - single[2]) > 1e-4:
                                    raise RuntimeError(
                                        "deterministic batched probes disagree with scalar probes"
                                    )
                            self._probe_verified.add(arm)
                        for index, result in zip(batch, scored, strict=True):
                            results[index] = result
                        batches.append(len(batch))
                        print(
                            f"[{self._ts()}] {operation}: batch {len(batches)} width {len(batch)}",
                            flush=True,
                        )
                break
            except self.torch.OutOfMemoryError:
                if batch_size == 1 or key in self._batch_locked:
                    raise RuntimeError(
                        f"{operation} exceeded the locked comparable-cell batch size"
                    )
                self.retries += 1
                operation_retries += 1
                batch_size = max(1, batch_size // 2)
                self.effective_batches[key] = batch_size
                self.torch.cuda.empty_cache()
                print(
                    f"[{self._ts()}] {operation}: OOM, retry {self.retries} at batch {batch_size}",
                    flush=True,
                )
        elapsed = time.time() - started
        self._batch_locked.add(key)
        self.topology.append(
            {
                "operation": operation,
                "arm": arm,
                "batches": batches,
                "wake": next(
                    (
                        int(part[1:])
                        for part in operation.split("-")
                        if part.startswith("w") and part[1:].isdigit()
                    ),
                    None,
                ),
                "requested_batch_size": self.manifest.batch_sizes[key],
                "effective_batch_size": batch_size,
                "retries": operation_retries,
                "seconds": elapsed,
                "peak_vram_bytes": (
                    int(self.torch.cuda.max_memory_allocated())
                    if self.torch.cuda.is_available()
                    else 0
                ),
                "items_per_second": len(pairs) / elapsed if elapsed else None,
            }
        )
        if any(result is None for result in results):
            raise RuntimeError("probe batching lost a request")
        return [result for result in results if result is not None]

    def probe(
        self, arm: str, wake: int, state: object, artifact: dict[str, object]
    ) -> dict[str, object]:
        from experiments.dreams.cli import paraphrase_prompts
        from experiments.facts import exact_match
        from experiments.locality import battery_summary, perplexity, score_battery_batched

        facts = self._wake_facts(wake)
        pairs: list[tuple[str, str]] = []
        for fact in facts:
            prompt = f"{self.user_open} What is the code for the {fact.entity}?{self.asst_open} The code for the {fact.entity} is"
            pairs += [(prompt, fact.code), (prompt, self.distractors[fact.entity])]
            pairs += [
                (prompt, fact.code)
                for prompt in paraphrase_prompts(fact, self.user_open, self.asst_open)
            ]
        scored = self._probe_pairs(
            arm, pairs, self.manifest.batch_sizes["probe"], f"fact-probe-w{wake}"
        )
        fact_results: dict[str, object] = {}
        cursor = 0
        stops = (".", "\n", self.user_open, self.asst_open)
        for fact in facts:
            correct, foil, *paraphrases = scored[cursor : cursor + 6]
            cursor += 6
            margin = correct[2] - foil[2]
            fact_results[fact.entity] = {
                "fact_wave": self.fact_waves[fact.entity],
                "margin": margin,
                "greedy": correct[0],
                "exact_match": exact_match(correct[0], fact.code, stops),
                "paraphrase_rate": sum(
                    exact_match(item[0], fact.code, stops) for item in paraphrases
                )
                / 4,
                "correct_logprob": correct[2],
                "foil_logprob": foil[2],
            }
            print(
                f"[{self._ts()}] {arm} wake {wake} fact {fact.entity}: margin {margin:+.3f}, "
                f"exact {fact_results[fact.entity]['exact_match']}, "
                f"paraphrase {fact_results[fact.entity]['paraphrase_rate']:.2f}",
                flush=True,
            )

        battery_pairs = [(str(item["prompt"]), str(item["answer"])) for item in self.battery]
        battery_scored = self._probe_pairs(
            arm,
            battery_pairs,
            self.manifest.batch_sizes["battery"],
            f"battery-w{wake}",
        )
        battery_by_prompt = {
            prompt: result
            for (prompt, _), result in zip(
                battery_pairs,
                battery_scored,
                strict=True,
            )
        }
        scored_battery = score_battery_batched(
            self.battery,
            lambda prompts: [
                (battery_by_prompt[prompt][0], battery_by_prompt[prompt][1]) for prompt in prompts
            ],
            self.manifest.batch_sizes["battery"],
        )
        locality = battery_summary(scored_battery)
        locality["loss_rate"] = float(locality["lost"]) / float(locality["items"])
        for item in scored_battery:
            if not item["correct"]:
                print(
                    f"[{self._ts()}] {arm} wake {wake} battery lost {item['prompt']!r} "
                    f"delta {float(item['logprob_delta']):+.3f}",
                    flush=True,
                )
        ppl = perplexity(
            self.contexts[arm].model,
            self.heldout,
            int(self.runtime["chunk_len"]),
            f"{arm} wake {wake} heldout ppl",
        )
        if self.torch.cuda.is_available():
            self._peak_vram = max(self._peak_vram, int(self.torch.cuda.max_memory_allocated()))
        print(
            f"[{self._ts()}] {arm} wake {wake}: {sum(bool(v['exact_match']) for v in fact_results.values())}/"
            f"{len(fact_results)} exact, battery lost {locality['lost']}, dPPL {ppl - self.base_ppl:+.4f}",
            flush=True,
        )
        return {
            "facts": fact_results,
            "battery": locality,
            "ppl": ppl,
            "ppl_delta": ppl - self.base_ppl,
            "treatment": self.treatments[arm, wake],
        }

    def execution_metadata(self) -> dict[str, object]:
        return {
            "model": self.runtime["model"],
            "warm_start_sha256": self.adapter_sha,
            "initial_full_weight_sha256": self.initial_full_sha,
            "manifest_sha256": self.manifest_sha,
            "runtime": self.runtime,
            "provider": self.manifest.generator,
            "device": str(self.device),
            "torch_version": self.torch.__version__,
            "requested_batch_sizes": self.manifest.batch_sizes,
            "effective_batch_sizes": self.effective_batches,
            "batch_topology": self.topology,
            "peak_vram_bytes": self._peak_vram,
            "concurrent_workers": 1,
            "dream_throughput": (
                self.dreams_generated / self.dream_seconds if self.dream_seconds else None
            ),
            "retries": self.retries,
        }


__all__ = [
    "RuntimeBackend",
    "artifact_transcript_ids",
    "dream_cache_identity",
    "battery_collision_terms",
    "manifest_payload",
    "manifest_sha",
    "bound_rehearsal_counts",
    "DreamSleepBackend",
]
