from types import SimpleNamespace

import pytest
import torch

from experiments.lama_ckl.protocol import (
    DREAM_INSTRUCTION,
    dream_instruction_ids,
    lama_dream_diagnostics,
    run_conversational_wake,
)


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    markers = {"[USER]": 3, "[ASSISTANT]": 4, "<EOC>": 5}

    def convert_tokens_to_ids(self, token):
        return self.markers[token]

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        ids, index = [], 0
        while index < len(text):
            marker = next((item for item in self.markers if text.startswith(item, index)), None)
            if marker:
                ids.append(self.markers[marker])
                index += len(marker)
            else:
                ids.append(10 + ord(text[index]))
                index += 1
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.markers.items()}
        return "".join(inverse[int(token)] if int(token) in inverse else chr(int(token) - 10)
                       for token in ids)


class State:
    def __init__(self, calls=0):
        self.calls = calls

    def detach(self):
        return State(self.calls)


class Model:
    def __init__(self, next_token=2):
        self.next_token = next_token

    def eval(self):
        return self

    def __call__(self, ids, state=None):
        state = State() if state is None else state
        state.calls += 1
        logits = torch.zeros(ids.shape[0], ids.shape[1], 128)
        logits[:, -1, self.next_token] = 1
        return logits, state


def test_conversational_wake_keeps_one_state_and_closes_once():
    tokenizer = Tokenizer()

    artifact, state = run_conversational_wake(
        Model(), tokenizer, ["Ada in Rome", "Bob in Oslo"], None,
        "[USER]", "[ASSISTANT]", "<EOC>", reply_tokens=4, evidence_tokens=512,
    )

    assert state.calls == 5
    assert len(artifact["turns"]) == 2
    assert all(turn["assistant_token_ids"] == [2] for turn in artifact["turns"])
    assert artifact["transcript_token_ids"][-1] == 5
    assert artifact["invariants"] == {
        "turns": 2,
        "missing_assistant_eos": 0,
        "internal_eoc": 0,
        "closing_eoc": 1,
    }


def test_conversational_wake_rejects_internal_eoc():
    with pytest.raises(RuntimeError, match="EOC inside wake"):
        run_conversational_wake(
            Model(next_token=5), Tokenizer(), ["Ada in Rome"], None,
            "[USER]", "[ASSISTANT]", "<EOC>", reply_tokens=4, evidence_tokens=512,
        )


def test_dream_instruction_is_a_user_turn_after_the_wake_boundary():
    tokenizer = Tokenizer()

    ids = dream_instruction_ids(tokenizer, "[USER]", "[ASSISTANT]", "cpu")

    assert ids[0, 0].item() == 3
    assert ids[0, -2].item() == 4
    assert tokenizer.decode(ids[0].tolist()).startswith("[USER] " + DREAM_INSTRUCTION)


def test_lama_dream_diagnostics_reports_bindings_misbindings_and_copies():
    rows = [
        {"subject": "Ada", "object": "Rome"},
        {"subject": "Bob", "object": "Oslo"},
    ]
    dreams = [
        SimpleNamespace(token_texts=["Ada remembers Rome. Bob remembers Rome."],
                        dream_ids=[1, 2, 3, 4], prefix_len=0, stop_reason="eoc", dream_sha="a"),
        SimpleNamespace(token_texts=["Bob remembers Oslo."], dream_ids=[9, 8],
                        prefix_len=0, stop_reason="max-tokens", dream_sha="b"),
    ]

    result = lama_dream_diagnostics(dreams, rows, [1, 2, 3, 7])

    assert result["correct_bindings"] == 2
    assert result["misbindings"] == 1
    assert result["unique_dreams"] == 2
    assert result["eoc_termination_rate"] == 0.5
    assert result["max_verbatim_run"] == 3
