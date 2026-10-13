"""Experiment: lama-ckl

Warm-start checkpoint probe: wake frame census on held-out contexts and the
dreams of one wake, read-only
"""

import argparse
import importlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import torch

from experiments.dream_generation import generate_replay_dreams
from experiments.lama_ckl.protocol import (
    apply_dream_prompt,
    dream_prompt_ids,
    run_conversational_wake,
)
from experiments.lama_ckl.runner import (
    DOCS_PER_WAKE,
    DREAM_TEMPERATURE,
    DREAM_TOKENS,
    DREAMS_PER_CYCLE,
    EVIDENCE_TOKENS,
    REPLY_TOKENS,
    file_sha,
)
from experiments.verbatim import copy_fraction
from progress import heartbeat, ts


class Dream(Protocol):
    dream_ids: Sequence[int]
    prefix_len: int
    stop_reason: str


def _first_token_kind(reply: Sequence[int], eoc_id: int, asst_id: int, eos_id: int) -> str:
    return {eoc_id: "eoc", asst_id: "assistant", eos_id: "eos"}.get(reply[0], "content")


HISTOGRAM_EDGES = ((1, 1), (2, 4), (5, 16), (17, 64))


def _length_bucket(length: int) -> str:
    for low, high in HISTOGRAM_EDGES:
        if length <= high:
            return str(low) if low == high else f"{low}-{high}"
    return f"{HISTOGRAM_EDGES[-1][1] + 1}+"


def _opener(text: str) -> str:
    words = text.split()
    return words[0].lower().rstrip(".,!?:;") if words else "<empty>"


def wake_census(
    turns: Sequence[Mapping[str, object]], *, eoc_id: int, asst_id: int, eos_id: int
) -> dict[str, object]:
    """First reply token kinds, closes, reply lengths, and reply openers"""
    first = Counter({"eoc": 0, "assistant": 0, "eos": 0, "content": 0})
    lengths: list[int] = []
    for turn in turns:
        reply = list(turn["assistant_token_ids"])
        first[_first_token_kind(reply, eoc_id, asst_id, eos_id)] += 1
        lengths.append(len(reply) - bool(turn["eos_forced"]))
    marked = [turn for turn in turns if turn.get("role_marker_closed")]
    histogram = Counter(_length_bucket(length) for length in lengths)
    return {
        "turns": len(turns),
        "first_token": dict(first),
        "forced_closes": sum(bool(turn["eos_forced"]) for turn in turns),
        "role_marker_closes": len(marked),
        "role_marker_examples": [
            {"assistant": turn["assistant"], "document": str(turn["document"])[:200]}
            for turn in marked[:3]
        ],
        "reply_tokens": {
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "histogram": {
                bucket: histogram[bucket]
                for bucket in map(_length_bucket, (1, 2, 5, 17, 65))
                if histogram[bucket]
            },
        },
        "openers": dict(Counter(_opener(str(turn["assistant"])) for turn in turns).most_common(10)),
    }


def _assistant_loop(generated: Sequence[int], user_id: int, asst_id: int) -> bool:
    last = None
    for token in generated:
        if token == asst_id and last == asst_id:
            return True
        if token in (user_id, asst_id):
            last = token
    return False


def dream_census(
    dreams: Sequence[Dream], transcript_ids: Sequence[int], *, user_id: int, asst_id: int
) -> dict[str, object]:
    """Empty and verbatim-copy dreams, [USER] openers, [ASSISTANT] loops, and stop reasons"""
    generated = [list(dream.dream_ids[dream.prefix_len :]) for dream in dreams]
    return {
        "dreams": len(dreams),
        "empty": sum(not ids for ids in generated),
        "verbatim_copies": sum(copy_fraction(ids, transcript_ids) >= 0.9 for ids in generated),
        "user_opener": sum(bool(ids) and ids[0] == user_id for ids in generated),
        "assistant_loop": sum(_assistant_loop(ids, user_id, asst_id) for ids in generated),
        "stop_reasons": dict(Counter(dream.stop_reason for dream in dreams)),
        "generated_tokens": sum(map(len, generated)) / len(generated) if generated else 0.0,
    }


def verdict(wake: Mapping[str, object], dreams: Mapping[str, object]) -> str:
    """One informational line against the expectations of a good checkpoint"""
    eoc_first = int(wake["first_token"]["eoc"])
    markers, forced = int(wake["role_marker_closes"]), int(wake["forced_closes"])
    opener, opener_count = next(iter(wake["openers"].items()), ("<none>", 0))
    count, openers, loops = (
        int(dreams["dreams"]),
        int(dreams["user_opener"]),
        int(dreams["assistant_loop"]),
    )
    # Thresholds are the pin gate over the 50 held-out replies (DISCUSSION-20261013 §3.2).
    checks = {
        f"eoc_first {eoc_first} (gate 0)": eoc_first == 0,
        f"role_marker_closes {markers} (gate <=4)": markers <= 4,
        f"forced {forced} (gate <=9)": forced <= 9,
        f"top opener {opener!r} {opener_count} (gate <=25)": opener_count <= 25,
        f"user_opener {openers}/{count} (expect most)": openers * 2 > count,
        f"assistant_loop {loops} (expect 0)": loops == 0,
    }
    return "verdict: " + "; ".join(
        f"{label} {'ok' if passed else 'OFF'}" for label, passed in checks.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-adapter", type=Path, required=True)
    parser.add_argument("--documents", type=Path, default=Path("data/dolly_heldout.json"))
    parser.add_argument("--docs-per-wake", type=int, default=DOCS_PER_WAKE)
    parser.add_argument("--dreams", type=int, default=DREAMS_PER_CYCLE)
    parser.add_argument("--dream-batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "mamba2_2_7b"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.init_adapter / "probe.json"
    documents = [str(document) for document in json.loads(args.documents.read_text())]
    if not documents or args.docs_per_wake < 1:
        raise SystemExit("need at least one document and --docs-per-wake >= 1")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model_mod = importlib.import_module(f"models.{args.model_name}")
    hooks = importlib.import_module(f"models.{args.model_name}.train_hooks")
    config = json.loads((args.init_adapter / "lora_config.json").read_text())
    model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
    from models.common import build_tokenizer

    from training.checkpoints import load_checkpoint

    load_checkpoint(model, args.init_adapter)
    tokenizer = build_tokenizer(model_mod)
    markers = (model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC)
    ids = {
        "eoc_id": int(tokenizer.convert_tokens_to_ids(model_mod.EOC)),
        "asst_id": int(tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN)),
        "eos_id": int(tokenizer.eos_token_id),
    }
    print(f"[{ts()}] probe {args.init_adapter}: {len(documents)} documents", flush=True)

    all_turns: list[Mapping[str, object]] = []
    per_wake: list[dict[str, object]] = []
    state = None
    for start in range(0, len(documents), args.docs_per_wake):
        wake_index = start // args.docs_per_wake + 1
        artifact, state = run_conversational_wake(
            model,
            tokenizer,
            documents[start : start + args.docs_per_wake],
            None,
            *markers,
            reply_tokens=REPLY_TOKENS,
            evidence_tokens=EVIDENCE_TOKENS,
            device=device,
            temperature=DREAM_TEMPERATURE,
            seed=args.seed + wake_index,
        )
        turns = artifact["turns"]
        all_turns.extend(turns)
        census = wake_census(turns, **ids)
        per_wake.append(census)
        print(f"[{ts()}] wake {wake_index} census {json.dumps(census, sort_keys=True)}", flush=True)
        failures = [
            turn
            for turn in turns
            if _first_token_kind(turn["assistant_token_ids"], **ids) in ("eoc", "assistant")
        ]
        for turn in failures[:5]:
            print(
                f"[{ts()}] wake {wake_index} failure: document={turn['document'][:200]!r} "
                f"assistant={turn['assistant']!r}",
                flush=True,
            )
        for example in census["role_marker_examples"]:
            print(
                f"[{ts()}] wake {wake_index} role marker close: "
                f"document={example['document']!r} assistant={example['assistant']!r}",
                flush=True,
            )
        heartbeat()

    open_state = apply_dream_prompt(model, tokenizer, state, *markers, device)
    dreams, _ = generate_replay_dreams(
        model,
        open_state,
        dream_prompt_ids(tokenizer, *markers, device),
        count=args.dreams,
        batch_size=args.dream_batch_size,
        seed=args.seed,
        n_tokens=DREAM_TOKENS,
        temperature=DREAM_TEMPERATURE,
        decode_token=lambda token: tokenizer.decode([token]),
        stop_id=ids["eoc_id"],
        turn_id=ids["eos_id"],
    )
    dreams_summary = dream_census(
        dreams,
        artifact["transcript_token_ids"],
        user_id=int(tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN)),
        asst_id=ids["asst_id"],
    )
    print(f"[{ts()}] dream census {json.dumps(dreams_summary, sort_keys=True)}", flush=True)
    decoded = ["".join(dream.token_texts)[:1000] for dream in dreams]
    for index, text in enumerate(decoded, start=1):
        print(f"[{ts()}] dream {index}: {text!r}", flush=True)

    wake_summary = wake_census(all_turns, **ids)
    print(f"[{ts()}] wake census {json.dumps(wake_summary, sort_keys=True)}", flush=True)
    print(f"[{ts()}] {verdict(wake_summary, dreams_summary)}", flush=True)
    output.write_text(
        json.dumps(
            {
                "init_adapter": str(args.init_adapter),
                "trainable_sha256": file_sha(args.init_adapter / "trainable.pt"),
                "documents": len(documents),
                "wake": {**wake_summary, "per_wake": per_wake},
                "dreams": {**dreams_summary, "decoded": decoded},
                "seed": args.seed,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[{ts()}] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
