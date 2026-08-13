"""CPU-only end-to-end smoke through the real DreamSleepBackend methods."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace


def generator() -> None:
    request = json.load(sys.stdin)
    session = request.get("session_id")
    print(json.dumps({
        "message": str(request["goal"]),
        "session_id": session or f"fake-{request['scenario']}-{request['turn']}",
        "resume_status": "resumed" if session else "started",
    }))


if __name__ == "__main__" and "--generator" in sys.argv:
    generator()
    raise SystemExit


import torch

from adaptive_multisleep import DreamSleepBackend, manifest_sha, run_registered_experiment
from adaptive_wake import ExperimentConfig, ExperimentManifest, WakePlan, WakeSpec
from dream_sleep import Fact, build_distractors


class FakeState:
    def __init__(self, batch: int):
        self.conv_states = [torch.zeros(batch, 1)]
        self.ssm_states = [torch.zeros(batch, 1)]

    def detach(self):
        state = copy.copy(self)
        state.conv_states = [value.detach() for value in self.conv_states]
        state.ssm_states = [value.detach() for value in self.ssm_states]
        return state


class FakeTokenizer:
    eos_token_id = 2
    markers = {"[USER]": 3, "[ASST]": 4, "[EOC]": 5}

    def convert_tokens_to_ids(self, token):
        return self.markers[token]

    @staticmethod
    def _char_id(char: str) -> int:
        return 10 + (ord(char) - 32) % 95

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, index = [], [], 0
        while index < len(text):
            marker = next((item for item in self.markers if text.startswith(item, index)), None)
            if marker:
                ids.append(self.markers[marker])
                offsets.append((index, index + len(marker)))
                index += len(marker)
            else:
                ids.append(self._char_id(text[index]))
                offsets.append((index, index + 1))
                index += 1
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        inverse = {value: key for key, value in self.markers.items()}
        return "".join("" if skip_special_tokens and token in inverse else
                       inverse.get(token, chr(32 + (int(token) - 10) % 95)) for token in ids)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.output = torch.nn.Linear(8, 128)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
            self.output.bias[FakeTokenizer._char_id("a")] = 0.1
            self.output.bias[2:6] = -20

    def forward(self, tokens, state=None):
        batch = tokens.shape[0]
        state = state or FakeState(batch)
        logits = self.output(self.embedding(tokens))
        ending = tokens[:, -1] == FakeTokenizer._char_id("a")
        if ending.any():
            logits = logits.clone()
            logits[ending, -1] = -100
            logits[ending, -1, 2] = 100
        update = tokens.float().sum(dim=1, keepdim=True)
        state = copy.copy(state)
        state.conv_states = [state.conv_states[0] + update]
        state.ssm_states = [state.ssm_states[0] + update]
        return logits, state


def fake_backend(manifest: ExperimentManifest, out: Path) -> DreamSleepBackend:
    import dream_sleep

    def fake_distill_set(model, optimizer, dreams, state, variant, epochs, temperature, on_step, on_boundary):
        assert len(dreams) == 300 and epochs == 1
        on_step(0, 0.0)
        return sum(len(dream.dream_ids) for dream in dreams)

    def fake_distill_sft(model, optimizer, ids, steps, chunk_len, on_step):
        assert steps >= 1
        on_step(0, 0.0)
        return ids.shape[1] - 1

    dream_sleep.distill_dream_set = fake_distill_set
    dream_sleep.distill_sft = fake_distill_sft
    backend = DreamSleepBackend.__new__(DreamSleepBackend)
    backend.manifest, backend.seed, backend.initial_state = manifest, 1, None
    backend.runtime, backend.torch = manifest.runtime, torch
    backend.device = torch.device("cpu")
    backend.tokenizer = FakeTokenizer()
    backend.user_open, backend.asst_open = "[USER]", "[ASST]"
    backend.user_id, backend.asst_id, backend.eoc_id, backend.turn_id = 3, 4, 5, 2
    torch.manual_seed(1)
    base = FakeModel()
    backend.contexts = {}
    for arm in manifest.config.arms:
        model = copy.deepcopy(base)
        backend.contexts[arm] = SimpleNamespace(model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4))
    backend.adapter_sha = "f" * 64
    backend.initial_full_sha = backend._model_hash(base)
    backend.output = out / "seed-1"
    backend.output.mkdir(parents=True, exist_ok=True)
    backend.manifest_sha = manifest_sha(manifest)
    backend.topology, backend.effective_batches, backend._batch_locked = [], dict(manifest.batch_sizes), set()
    backend.dream_seconds = backend.dreams_generated = backend.retries = backend._peak_vram = 0
    backend._probe_verified, backend.treatments = set(), {}

    def fake_score(self, model, prompts, targets):
        return [("a", -1.0, -float(targets.shape[1])) for _ in range(prompts.shape[0])]

    backend._score_tensor_batch = MethodType(fake_score, backend)
    all_facts = [Fact(entity, "entity", code) for spec in manifest.wakes for entity, code in spec.facts]
    backend.facts = {fact.entity: fact for fact in all_facts}
    backend.fact_waves = {entity: wake for wake, spec in enumerate(manifest.wakes, 1) for entity, _ in spec.facts}
    backend.distractors = build_distractors(all_facts, 1)
    backend.heldout = backend._encode("held out text for fake perplexity")
    from experiments.locality import perplexity
    backend.base_ppl = perplexity(base, backend.heldout, 64, "fake heldout baseline")
    baseline = backend._probe_pairs("replay", [(f"battery {index:03d}", "a") for index in range(100)],
                                    manifest.batch_sizes["battery"], "battery-calibration")
    backend.battery = [{"prompt": f"battery {index:03d}", "answer": "a", "logprob": score[1],
                        "greedy": score[0]} for index, score in enumerate(baseline)]
    return backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--generator", action="store_true")
    args = parser.parse_args()
    if args.generator:
        generator()
        return
    if not args.out:
        raise SystemExit("--out is required for the smoke run")
    out = Path(args.out)
    config = ExperimentConfig.from_dict({"arms": ["replay", "nosleep", "sft-ref"],
                                         "wakes": 6, "seeds": [1, 2, 3]})
    plan = WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4],
                               "turn_goals": ["open", "develop", "complicate", "close"]})
    wakes = tuple(WakeSpec(
        f"scenario-{wake}", tuple((f"entity_{wake}_{index}", f"{wake} {index} 0 0 0")
                                  for index in range(1, 5)), plan,
    ) for wake in range(1, 7))
    runtime = {"model": "fake", "warm_start": "fake", "warm_start_sha256": "f" * 64,
               "lora_rank": 1, "lora_alpha": 1, "learning_rate": 1e-4, "chunk_len": 64,
               "dream_count": 300, "dream_tokens": 12, "dream_temperature": 1.0,
               "reply_tokens": 2, "reply_temperature": 0.0, "probe_tokens": 1,
               "kl_temperature": 1.0, "battery": str(out / "battery.json")}
    manifest = ExperimentManifest(
        config, wakes, {"command": [sys.executable, __file__, "--generator"],
                        "provider": "fake", "model": "fake", "version": "1"},
        str(out), {"dream": 50, "probe": 32, "battery": 32}, runtime,
    )
    run_registered_experiment(manifest, 1, fake_backend(manifest, out))


if __name__ == "__main__":
    main()
