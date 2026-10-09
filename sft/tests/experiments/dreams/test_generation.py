import torch

from experiments.dreams.generation import _teacher_dream_batch


class State:
    def __init__(self):
        self.conv_states = [torch.zeros(1, 2)]
        self.ssm_states = [torch.zeros(1, 3)]


class Model:
    """Row 0 always argmaxes to the stop token, row 1 always to token 7."""

    def __init__(self):
        self.fed: list[list[int]] = []

    def __call__(self, ids, state=None):
        self.fed.append(ids[:, 0].tolist())
        logits = torch.zeros(ids.shape[0], 1, 16)
        logits[0, 0, 5] = 1
        logits[1:, 0, 7] = 1
        return logits, state


def test_batch_stops_rows_on_eoc_and_keeps_the_prefix_logits():
    model = Model()
    seed_ids = torch.tensor([[3, 4]])

    first, second = _teacher_dream_batch(
        model, State(), seed_ids, [1, 2], 5, 0.0, str, stop_id=5, turn_id=None
    )

    assert first.tokens.tolist() == [[3, 4]]
    assert first.logits.shape[0] == 2
    assert first.stop_reason == "eoc"
    assert first.prefix_len == 2

    assert second.tokens.tolist() == [[3, 4, 7, 7, 7]]
    assert second.logits.shape[0] == 5
    assert second.stop_reason == "max-tokens"

    # the finished row keeps feeding the last prefix token as filler
    assert model.fed == [[3, 3], [4, 4], [4, 7], [4, 7], [4, 7]]


def test_batch_stops_early_when_every_row_is_done():
    model = Model()
    seed_ids = torch.tensor([[3]])

    (dream,) = _teacher_dream_batch(
        model, State(), seed_ids, [1], 50, 0.0, str, stop_id=5, turn_id=None
    )

    assert dream.stop_reason == "eoc"
    assert len(model.fed) == 1
