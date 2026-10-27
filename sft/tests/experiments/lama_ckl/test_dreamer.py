import json
import random

import pytest
import torch

from experiments.lama_ckl import dreamer_corpus
from experiments.lama_ckl.dreamer import (
    abstention_reading,
    batch_gap,
    completed_rows,
    kept_dream_draw,
    live_reading,
    mann_whitney_greater,
    pair_seq,
    replay_draw,
    replay_or_stop,
    round_complete,
    round_dir,
    sft_batch_loss,
    sft_pairs,
    sft_weights,
    specificity_others,
    weight_readings,
    write_json,
)

EOC = 93


def _row(kind, score, gap):
    return {"kind": kind, "score": score, "gap": gap}


def test_weight_sign_from_score_magnitude_from_gap_normalised_per_kind():
    rows = [
        _row("fact", 0.5, 2.0),
        _row("fact", 0.1, 4.0),
        _row("fact", -0.2, 9.0),  # score <= 0: zero whatever its gap
        _row("fact", 0.3, -1.0),  # gap <= 0: zero whatever its score
        _row("conversation", 0.2, 30.0),
        _row("conversation", 0.2, 10.0),
    ]
    w = [o["weight"] for o in sft_weights(rows)]
    assert w[:4] == pytest.approx([2 / 3, 4 / 3, 0.0, 0.0])
    assert w[4:] == pytest.approx([1.5, 0.5])


def test_kind_without_a_positive_row_is_all_zero_and_causes_are_recorded():
    out = sft_weights([_row("fact", None, 1.0), _row("fact", -1.0, -1.0), _row("fact", 1.0, None)])
    assert [o["weight"] for o in out] == [0.0, 0.0, 0.0]
    assert [o["zero"] for o in out] == [["no_batch"], ["score", "gap"], ["gap"]]


def test_weight_readings_share_at_zero_by_cause():
    rows = [
        _row("fact", 1.0, 1.0),
        _row("fact", -1.0, 2.0),
        _row("fact", 1.0, -1.0),
        _row("fact", None, None),
    ]
    reading = weight_readings(rows, sft_weights(rows))["fact"]
    assert reading["zero_share"] == 0.75
    assert reading["zero_by_cause"] == {"no_batch": 0.25, "score": 0.25, "gap": 0.5}
    assert reading["gap_positive"] == 2


def test_specificity_excludes_the_partner_in_both_directions():
    kinds = ["fact"] * 5 + ["conversation"] * 2
    partners = [1, 2, 3, 4, 0, 6, 5]
    # row 0 took its C- from row 1, and row 4 took its C- from row 0
    assert specificity_others(0, kinds, partners) == [2, 3]
    assert specificity_others(5, kinds, partners) == []


def test_batch_gap_is_own_dot_minus_mean_of_others_with_the_clip():
    gram = torch.eye(3, dtype=torch.float64) * 4.0
    dots = torch.tensor([[3.0, 1.0, 0.0], [1.0, 1.0, 2.0], [9.0, 9.0, 9.0]], dtype=torch.float64)
    # G_B = mean of dreams 0, 1: dots (2, 1, 1), ||G_B||^2 = 8 / 4 = 2
    assert batch_gap(
        [0, 1], gram=gram, dots=dots, own=0, others=[1, 2], alpha=0.5, c=10.0
    ) == pytest.approx(0.5)
    clipped = batch_gap([0, 1], gram=gram, dots=dots, own=0, others=[1, 2], alpha=0.5, c=1.0)
    assert clipped == pytest.approx(0.5 / 2**0.5)
    assert batch_gap([0, 1], gram=gram, dots=dots, own=0, others=[], alpha=0.5, c=1.0) is None


def test_pair_targets_the_dream_body_and_the_appended_eoc():
    dream = {"ids": [EOC, 7, 8, 0], "prefix_len": 1, "stop_reason": "eoc"}
    ids, mask = pair_seq([1, 2, 3], dream, EOC)
    assert ids == [1, 2, 3, EOC, 7, 8, 0, EOC]
    assert [t for t, m in zip(ids, mask, strict=True) if m] == [7, 8, 0, EOC]
    ids, mask = pair_seq([1, 2, 3], dream | {"stop_reason": "max-tokens"}, EOC)
    assert ids[-1] == 0 and sum(mask) == 3


class _Fake(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(20, 20)
        torch.nn.init.normal_(self.emb.weight, generator=torch.Generator().manual_seed(0))

    def forward(self, tokens, state=None):
        return self.emb(tokens), None


def _token_mean(model, ids, mask):
    logits, _ = model(torch.tensor([ids]))
    nll = torch.nn.functional.cross_entropy(logits[0, :-1], torch.tensor(ids[1:]), reduction="none")
    keep = torch.tensor(mask[1:])
    return float(nll[keep].mean())


def test_pair_loss_is_per_pair_token_mean_weighted_over_the_batch_size():
    model = _Fake()
    long = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [False, False] + [True] * 9)
    short = ([3, 5, 7], [False, True, True])
    expected = (_token_mean(model, *long) + _token_mean(model, *short)) / 2
    assert float(sft_batch_loss(model, [(long, 1.0), (short, 1.0)], 2)) == pytest.approx(
        expected, rel=1e-5
    )
    weighted = (3 * _token_mean(model, *long) + _token_mean(model, *short)) / 2
    assert float(sft_batch_loss(model, [(long, 3.0), (short, 1.0)], 2)) == pytest.approx(
        weighted, rel=1e-5
    )
    micro = sft_batch_loss(model, [(long, 3.0)], 2) + sft_batch_loss(model, [(short, 1.0)], 2)
    assert float(micro) == pytest.approx(weighted, rel=1e-5)


def test_zero_weight_rows_drop_out_of_the_pairs():
    pool = [{"ids": [EOC, i, 0], "prefix_len": 1, "stop_reason": "eoc"} for i in range(4)]
    records = [
        {"transcript": [1], "pool": pool, "kept": [0, 1]},
        {"transcript": [2], "pool": pool, "kept": [2, 3]},
    ]
    pairs = sft_pairs(records, [{"weight": 0.0}, {"weight": 1.5}], EOC)
    assert [w for _, w in pairs] == [1.5, 1.5]
    assert all(seq[0][0] == 2 for seq, _ in pairs)


def test_replay_mix_is_one_to_one_and_a_short_pool_draws_what_exists():
    pool = [{"dolly_line": i} for i in range(5)]
    rows, shortfall = replay_draw(pool, 3, random.Random(0))
    assert len(rows) == 3 and shortfall == 0
    rows, shortfall = replay_draw(pool, 12, random.Random(0))
    assert sorted(r["dolly_line"] for r in rows) == list(range(5)) and shortfall == 7
    assert replay_draw([], 4, random.Random(0)) == ([], 4)


def test_short_replay_pool_stops_the_round_with_its_numbers(tmp_path):
    pool = [{"dolly_line": i} for i in range(5)]
    assert len(replay_or_stop(pool, 5, random.Random(0), round_dir(tmp_path, 0))) == 5
    assert not (round_dir(tmp_path, 0) / "readings.json").exists()
    with pytest.raises(SystemExit):
        replay_or_stop(pool, 12, random.Random(0), round_dir(tmp_path, 0))
    readings = json.loads((round_dir(tmp_path, 0) / "readings.json").read_text())
    assert readings["replay"] == {"pool": 5, "kept_pairs": 12, "shortfall": 7, "stopped": True}
    assert not round_complete(tmp_path, 0)


def test_over_abstention_is_the_unknown_share_of_the_labels(monkeypatch):
    labels = ["unknown", "answers", "cannot", "unknown"]
    monkeypatch.setattr(
        dreamer_corpus, "abstention_labels", lambda replies, device: labels, raising=False
    )
    reading = abstention_reading(["a", "b", "c", "d"], torch.device("cpu"))
    assert reading["over_abstention"] == 0.5
    assert reading["abstention_labels"] == {"unknown": 2, "cannot": 1, "answers": 1}
    assert reading["abstentions"] == ["a", "d"]


def test_kept_dream_source_draws_only_positive_weight_rows():
    records = [{"kept": [0, 1, 2]}, {"kept": [3, 4]}, {"kept": [5]}]
    weights = [{"weight": 0.0}, {"weight": 2.0}, {"weight": 0.5}]
    picks = kept_dream_draw(records, weights, 10, random.Random(0))
    assert sorted(picks) == [(1, 3), (1, 4), (2, 5)]
    assert len(kept_dream_draw(records, weights, 2, random.Random(0))) == 2


def test_resume_skips_completed_rows_and_rounds(tmp_path):
    directory = round_dir(tmp_path, 0)
    write_json(directory / "rows" / "00-fact-a.json", {"row_id": "00-fact-a"})
    (directory / "rows" / "01-fact-b.json.partial").write_text("{")
    assert list(completed_rows(directory)) == ["00-fact-a"]
    assert not round_complete(tmp_path, 0)
    write_json(directory / "readings.json", {})
    assert round_complete(tmp_path, 0) and not round_complete(tmp_path, 1)
    assert json.loads((directory / "readings.json").read_text()) == {}


def test_mann_whitney_one_sided():
    u, p = mann_whitney_greater([5.0, 6.0, 7.0], [1.0, 2.0, 3.0, 4.0])
    assert u == 12.0 and p < 0.05
    _, p_low = mann_whitney_greater([1.0, 2.0], [5.0, 6.0, 7.0])
    assert p_low > 0.9
    assert mann_whitney_greater([], [1.0]) is None


def test_live_reading_ranks_within_rows_and_pools_them():
    def record(values, revisit):
        return {
            "s": values,
            "cos": values,
            "candidates": list(range(len(values))),
            "pool": [{"revisit": i in revisit} for i in range(len(values))],
        }

    # Raw values differ in scale per row; within-row ranks put the revisit on top in both.
    reading = live_reading(
        [record([1.0, 2.0, 30.0], {2}), record([100.0, 200.0, 300.0, 400.0], {3})]
    )
    assert reading["s"]["revisiting"] == 2 and reading["s"]["other"] == 5
    assert reading["s"]["p"] < 0.1
