from types import SimpleNamespace

import torch

from experiments.lama_ckl.protocol import (
    WAKE_FRAME,
    apply_dream_prompt,
    dream_prompt_ids,
    lama_dream_diagnostics,
    run_conversational_wake,
)
from experiments.lama_ckl.runner import REPLY_TOKENS


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    markers = {"[USER]": 3, "[ASSISTANT]": 4, "<EOC>": 5, "<EOS>": 2}

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
        return "".join(
            inverse[int(token)] if int(token) in inverse else chr(int(token) - 10) for token in ids
        )


class State:
    def __init__(self, calls=0):
        self.calls = calls

    def detach(self):
        return State(self.calls)


class Model:
    def __init__(self, next_token=2):
        self.next_token = next_token
        self.fed = []

    def eval(self):
        return self

    def __call__(self, ids, state=None):
        self.fed.append(ids.tolist())
        state = State() if state is None else state
        state.calls += 1
        logits = torch.zeros(ids.shape[0], ids.shape[1], 128)
        logits[:, -1, self.next_token] = 1
        return logits, state


def test_conversational_wake_keeps_one_state_and_leaves_it_open():
    tokenizer = Tokenizer()

    artifact, state = run_conversational_wake(
        Model(),
        tokenizer,
        ["Ada in Rome", "Bob in Oslo"],
        None,
        "[USER]",
        "[ASSISTANT]",
        "<EOC>",
        reply_tokens=4,
        evidence_tokens=512,
        device="cpu",
    )

    assert state.calls == 4
    assert len(artifact["turns"]) == 2
    assert all(turn["assistant_token_ids"] == [2] for turn in artifact["turns"])
    assert 5 not in artifact["transcript_token_ids"]
    assert artifact["invariants"] == {"turns": 2, "internal_eoc": 0}
    assert artifact["forced_closes"] == 0


def test_conversational_wake_puts_one_frame_before_every_document():
    tokenizer = Tokenizer()
    documents = ["Ada in Rome", "Bob lives in Oslo"]

    artifact, _ = run_conversational_wake(
        Model(),
        tokenizer,
        documents,
        None,
        "[USER]",
        "[ASSISTANT]",
        "<EOC>",
        reply_tokens=4,
        evidence_tokens=512,
        device="cpu",
    )

    prompts = [turn["prompt_token_ids"] for turn in artifact["turns"]]
    frame_ids = tokenizer(f"[USER] {WAKE_FRAME}".rstrip())["input_ids"]
    assert all(prompt[: len(frame_ids)] == frame_ids for prompt in prompts)
    for prompt, document in zip(prompts, documents, strict=True):
        assert tokenizer.decode(prompt) == f"[USER] {WAKE_FRAME}{document}[ASSISTANT] "
    assert artifact["frame"] == WAKE_FRAME


def test_conversational_wake_forces_eos_at_the_backstop_and_counts_it():
    artifact, state = run_conversational_wake(
        Model(next_token=20),
        Tokenizer(),
        ["Ada in Rome", "Bob in Oslo"],
        None,
        "[USER]",
        "[ASSISTANT]",
        "<EOC>",
        reply_tokens=REPLY_TOKENS,
        evidence_tokens=512,
        device="cpu",
    )

    turn = artifact["turns"][0]
    assert REPLY_TOKENS == 128
    assert turn["assistant_token_ids"] == [20] * 128 + [2]
    assert turn["eos_forced"] is True
    assert state.calls == 2 * (1 + 128 + 1)
    assert artifact["forced_closes"] == 2
    assert artifact["invariants"] == {"turns": 2, "internal_eoc": 0}


def test_conversational_wake_records_internal_eoc_for_the_runner_to_refuse():
    artifact, _ = run_conversational_wake(
        Model(next_token=5),
        Tokenizer(),
        ["Ada in Rome"],
        None,
        "[USER]",
        "[ASSISTANT]",
        "<EOC>",
        reply_tokens=4,
        evidence_tokens=512,
        device="cpu",
    )

    assert artifact["turns"][0]["assistant_token_ids"] == [5]
    assert artifact["invariants"]["internal_eoc"] == 1


def test_dream_prompt_is_the_wake_boundary_alone():
    ids = dream_prompt_ids(Tokenizer(), "[USER]", "[ASSISTANT]", "<EOC>", "cpu")

    assert ids.tolist() == [[5]]
    assert ids.dtype == torch.long


def test_apply_dream_prompt_feeds_the_prompt_and_leaves_the_open_state_untouched():
    model = Model()
    open_state = State(calls=4)

    closed = apply_dream_prompt(
        model, Tokenizer(), open_state, "[USER]", "[ASSISTANT]", "<EOC>", "cpu"
    )

    assert model.fed == [[[5]]]
    assert closed.calls == 5
    assert open_state.calls == 4


def test_lama_dream_diagnostics_reports_bindings_misbindings_and_copies():
    rows = [
        {"subject": "Ada", "object": "Rome"},
        {"subject": "Bob", "object": "Oslo"},
    ]
    dreams = [
        SimpleNamespace(
            token_texts=["Ada remembers Rome. Bob remembers Rome."],
            dream_ids=[1, 2, 3, 4],
            prefix_len=0,
            stop_reason="eoc",
            dream_sha="a",
        ),
        SimpleNamespace(
            token_texts=["Bob remembers Oslo."],
            dream_ids=[9, 8],
            prefix_len=0,
            stop_reason="max-tokens",
            dream_sha="b",
        ),
    ]

    result = lama_dream_diagnostics(dreams, rows, [1, 2, 3, 7])

    assert result["correct_bindings"] == 2
    assert result["misbindings"] == 1
    assert result["unique_dreams"] == 2
    assert result["eoc_termination_rate"] == 0.5
    assert result["max_verbatim_run"] == 3
