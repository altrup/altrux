import inspect

import pytest
import torch

from experiments.dream_distillation import (
    distill_dream_batch,
    dream_batch_gradient,
    flat_grad,
    turn_weights,
)
from experiments.dream_types import CachedDream

USER, ASST, EOS, EOC = 1, 2, 3, 4
VOCAB = 16
IDS = {"user_id": USER, "asst_id": ASST, "eos_id": EOS, "eoc_id": EOC}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))
        )
        self.states: list[object] = []

    def forward(self, tokens, state=None):
        self.states.append(state)
        return self.table[tokens], state


class CountingSGD(torch.optim.SGD):
    steps = 0

    def step(self, closure=None):
        self.steps += 1
        self.seen = torch.cat([p.grad.flatten() for g in self.param_groups for p in g["params"]])
        return super().step(closure)


def dream(ids, stop="eoc", seed=1):
    logits = torch.randn(len(ids), VOCAB, generator=torch.Generator().manual_seed(seed))
    return CachedDream(
        dream_ids=ids,
        token_texts=[str(t) for t in ids],
        teacher_logits=logits,
        cue_flags=[False] * len(ids),
        prefix_len=1,
        stop_reason=stop,
        divergence=[],
        gate_positions=[],
        queries=[],
        spectra=[],
        ranks={},
        bases={},
    )


# turn 0 is one content token before any marker; turn 1 is [USER] + 7 content + EOS
TWO_TURNS = [EOC, 5, USER, *[6] * 7, EOS]


def run(dreams, lr=0.0):
    model = Model()
    opt = CountingSGD(model.parameters(), lr=lr)
    before = model.table.detach().clone()
    out = distill_dream_batch(model, opt, dreams, 1.0, **IDS)
    return out, opt, model, before


def test_one_optimizer_step_for_the_whole_batch():
    out, opt, model, before = run(
        [dream(TWO_TURNS), dream([EOC, ASST, 7, 8], "max-tokens")], lr=0.1
    )
    assert opt.steps == 1
    assert model.table.grad is None
    assert not torch.equal(model.table, before)
    assert out["scored_dreams"] == 2


def test_each_content_turn_gets_half_the_dream_weight():
    weights = turn_weights(dream(TWO_TURNS), **IDS)
    assert weights[0] == pytest.approx(0.5)
    assert [weights[p] for p in range(1, 10)] == pytest.approx([0.5 / 9] * 9)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_terminal_eoc_position_is_dropped_and_max_tokens_last_is_kept():
    assert 10 not in turn_weights(dream(TWO_TURNS, "eoc"), **IDS)
    kept = turn_weights(dream(TWO_TURNS, "max-tokens"), **IDS)
    assert kept[10] == pytest.approx(0.5 / 10)


def test_turn_without_content_gets_no_weight():
    weights = turn_weights(dream([EOC, USER, ASST, 5, 5, EOS]), **IDS)
    assert 0 not in weights
    assert [weights[p] for p in range(1, 5)] == pytest.approx([0.25] * 4)


def test_empty_dream_contributes_nothing_and_does_not_dilute():
    full = dream(TWO_TURNS)
    alone, *_ = run([full])
    batched, *_ = run([full, dream([EOC])])
    assert batched["scored_dreams"] == 1
    assert batched["loss"] == pytest.approx(alone["loss"])
    assert batched["scored_tokens"] == [10, 0]


def test_padding_positions_are_never_scored():
    short, long = dream([EOC, ASST, 7, 8], "max-tokens", seed=2), dream(TWO_TURNS, seed=3)
    a, *_ = run([short])
    b, *_ = run([long])
    both, *_ = run([short, long])
    assert both["loss"] == pytest.approx((a["loss"] + b["loss"]) / 2)
    assert both["token_gradients"] == a["token_gradients"] + b["token_gradients"]


def test_student_state_is_empty_in_batch():
    assert not {"init_state", "state"} & set(inspect.signature(distill_dream_batch).parameters)
    _, _, model, _ = run([dream(TWO_TURNS)])
    assert model.states == [None]


def test_dream_losses_recorded_and_floor_warning_in_batch():
    first, unscored, second = (
        dream(TWO_TURNS, seed=2),
        dream([EOC]),
        dream([EOC, ASST, 7, 8], seed=3),
    )
    alone = [run([d])[0]["loss"] for d in (first, second)]
    out, *_ = run([first, unscored, second])
    assert out["dream_losses"] == [pytest.approx(alone[0]), None, pytest.approx(alone[1])]
    assert out["max_dream_loss"] == pytest.approx(max(alone))
    assert out["loss"] == pytest.approx(sum(alone) / 2)

    # A dream's loss is its weighted KL with weights summing to 1: identical
    # per-position KL everywhere gives that KL back whatever the dream length.
    model = Model()
    flat = dream(TWO_TURNS)
    flat.teacher_logits = torch.zeros(len(TWO_TURNS), VOCAB)
    with torch.no_grad():
        model.table.copy_(torch.arange(VOCAB, dtype=torch.float).expand(VOCAB, VOCAB))
    out = distill_dream_batch(model, CountingSGD(model.parameters(), lr=0.0), [flat], 1.0, **IDS)
    log_p = torch.log_softmax(torch.zeros(VOCAB), -1)
    kl = (
        log_p.exp() * (log_p - torch.log_softmax(torch.arange(VOCAB, dtype=torch.float), -1))
    ).sum()
    assert out["dream_losses"] == [pytest.approx(kl.item())]

    none, *_ = run([dream([EOC])])
    assert none["dream_losses"] == [None] and none["max_dream_loss"] is None
    assert none["loss"] == 0.0


def test_uniform_weights_cover_the_turn_positions_equally():
    for ids, stop in (
        (TWO_TURNS, "eoc"),
        (TWO_TURNS, "max-tokens"),
        ([EOC, USER, ASST, 5, EOS], "eoc"),
    ):
        turn = turn_weights(dream(ids, stop), **IDS)
        uniform = turn_weights(dream(ids, stop), **IDS, loss_weights="uniform")
        assert list(uniform) == list(turn)
        assert list(uniform.values()) == pytest.approx([1 / len(turn)] * len(turn))
    with pytest.raises(ValueError):
        turn_weights(dream(TWO_TURNS), **IDS, loss_weights="tokens")


def _batch_grad(dreams, loss_weights="turn"):
    model = Model()
    out = dream_batch_gradient(model, dreams, 1.0, **IDS, loss_weights=loss_weights)
    return out, flat_grad(model)


@pytest.mark.parametrize("loss_weights", ["turn", "uniform"])
def test_per_dream_gradients_average_to_the_batched_gradient(loss_weights):
    dreams = [
        dream(TWO_TURNS, seed=2),
        dream([EOC]),
        dream([EOC, ASST, 7, 8], "max-tokens", seed=3),
        dream([EOC, USER, 9, 5, 6, EOS, ASST, 7, EOS], seed=4),
    ]
    # An unscored all-marker dream lengthens the padded batch without entering the loss.
    blank = dream([EOC] + [USER] * 30, "max-tokens")
    out, batched = _batch_grad(dreams, loss_weights)
    _, repadded = _batch_grad([*dreams, blank], loss_weights)
    spread = (batched - repadded).abs().max().item()

    per_dream = [_batch_grad([d], loss_weights) for d in dreams]
    scored = [g for o, g in per_dream if o["scored_dreams"]]
    assert len(scored) == out["scored_dreams"] == 3
    mean = torch.stack(scored).mean(0)
    assert (mean - batched).abs().max().item() <= max(4 * spread, 1e-6)


def test_unscored_dream_leaves_no_gradient():
    model = Model()
    out = dream_batch_gradient(model, [dream([EOC])], 1.0, **IDS)
    assert out["scored_dreams"] == 0
    assert torch.equal(flat_grad(model), torch.zeros(VOCAB * VOCAB))


def test_clip_scales_the_gradient_before_the_step():
    dreams = [dream(TWO_TURNS, seed=2), dream([EOC, ASST, 7, 8], seed=3)]
    _, raw = _batch_grad(dreams)
    norm = raw.norm().item()

    model = Model()
    opt = CountingSGD(model.parameters(), lr=0.1)
    out = distill_dream_batch(model, opt, dreams, 1.0, **IDS, clip_norm=norm / 4)
    assert out["clipped"] and out["grad_norm"] == pytest.approx(norm)
    assert opt.seen.norm().item() == pytest.approx(norm / 4, rel=1e-4)
    assert torch.allclose(opt.seen, raw / 4, rtol=1e-4, atol=1e-8)

    model = Model()
    opt = CountingSGD(model.parameters(), lr=0.1)
    out = distill_dream_batch(model, opt, dreams, 1.0, **IDS)
    assert not out["clipped"] and out["grad_norm"] == pytest.approx(norm)
    assert torch.equal(opt.seen, raw)

    model = Model()
    opt = CountingSGD(model.parameters(), lr=0.1)
    out = distill_dream_batch(model, opt, dreams, 1.0, **IDS, clip_norm=norm * 4)
    assert not out["clipped"]
    assert torch.equal(opt.seen, raw)
