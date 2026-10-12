from types import SimpleNamespace

import pytest
import torch

from experiments.lama_ckl.probe import dream_census, wake_census
from experiments.lama_ckl.protocol import run_conversational_wake

USER, ASST, EOC, EOS = 3, 4, 5, 2


class Tokenizer:
    eos_token_id = EOS
    markers = {"[USER]": USER, "[ASSISTANT]": ASST, "<EOC>": EOC, "<EOS>": EOS}

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
    def detach(self):
        return self


class Model:
    def __init__(self, next_token):
        self.next_token = next_token

    def eval(self):
        return self

    def __call__(self, ids, state=None):
        logits = torch.zeros(ids.shape[0], ids.shape[1], 128)
        logits[:, -1, self.next_token] = 1
        return logits, state or State()


def _census(next_token, documents=("Ada in Rome", "Bob in Oslo", "Cy in Lima")):
    artifact, _ = run_conversational_wake(
        Model(next_token),
        Tokenizer(),
        list(documents),
        None,
        "[USER]",
        "[ASSISTANT]",
        "<EOC>",
        reply_tokens=4,
        evidence_tokens=512,
        device="cpu",
    )
    return wake_census(artifact["turns"], eoc_id=EOC, asst_id=ASST, eos_id=EOS)


def test_wake_census_counts_eoc_first_replies():
    census = _census(EOC)

    assert census["first_token"] == {"eoc": 3, "assistant": 0, "eos": 0, "content": 0}
    assert census["forced_closes"] == 0
    assert census["turns"] == 3


def test_wake_census_counts_eos_first_replies():
    census = _census(EOS)

    assert census["first_token"]["eos"] == 3
    assert census["reply_tokens"]["max"] == 1


@pytest.mark.parametrize("token", [ASST, 70])
def test_wake_census_counts_forced_closes_and_excludes_the_fed_eos(token):
    census = _census(token)

    assert census["forced_closes"] == 3
    assert census["first_token"]["assistant" if token == ASST else "content"] == 3
    assert census["reply_tokens"] == {"mean": 4.0, "min": 4, "max": 4}


def test_dream_census_counts_empty_user_opener_and_assistant_loop():
    dreams = [
        SimpleNamespace(dream_ids=[EOC], prefix_len=1, stop_reason="eoc"),
        SimpleNamespace(dream_ids=[EOC, USER, 70, ASST, 71], prefix_len=1, stop_reason="eoc"),
        SimpleNamespace(dream_ids=[EOC, ASST, 70, ASST, 71], prefix_len=1, stop_reason="budget"),
    ]

    census = dream_census(dreams, user_id=USER, asst_id=ASST)

    assert census["dreams"] == 3
    assert census["empty"] == 1
    assert census["user_opener"] == 1
    assert census["assistant_loop"] == 1
    assert census["stop_reasons"] == {"eoc": 2, "budget": 1}
    assert census["generated_tokens"] == pytest.approx(8 / 3)
