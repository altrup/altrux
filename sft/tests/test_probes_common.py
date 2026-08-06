"""CPU tests for the shared probe battery: grading, self-calibrated build,
the on-disk cache, and the perplexity math."""

import json

import pytest
import torch

from consolidation_null import target_logprob
from probes_common import (
    battery_hit,
    battery_summary,
    build_battery,
    code_margin,
    load_or_build_battery,
    logprob_sum,
    nll_from_logits,
    score_battery,
)


def test_battery_hit_ignores_case_space_and_trailing_text():
    assert battery_hit("  Paris, the capital city.", "Paris")
    assert battery_hit("paris", "Paris")
    assert battery_hit(" New York City is large", "New York City")


def test_battery_hit_requires_the_answer_first():
    assert not battery_hit("Lyon, actually Paris", "Paris")
    assert not battery_hit("", "Paris")


def test_battery_hit_matches_whole_words_only():
    assert not battery_hit("Parisian food", "Paris")


def test_build_battery_keeps_only_the_model_s_own_hits():
    candidates = [("a", "yes"), ("b", "yes"), ("c", "yes")]
    answers = {"a": ("yes indeed", -0.5), "b": ("no", -2.0), "c": ("yes", -0.1)}

    kept = build_battery(candidates, lambda p: answers[p])

    assert [item["prompt"] for item in kept] == ["a", "c"]
    assert kept[0]["logprob"] == -0.5


def test_load_or_build_battery_caches_to_disk(tmp_path):
    path = tmp_path / "battery.json"
    built = load_or_build_battery(path, [("a", "yes")], lambda p: ("yes", -0.5))

    def explode(prompt: str) -> tuple[str, float]:
        raise AssertionError("cached battery must not be rebuilt")

    assert json.loads(path.read_text()) == built
    assert load_or_build_battery(path, [("a", "yes")], explode) == built


def test_score_battery_reports_flips_and_logprob_drops():
    items = [{"prompt": "a", "answer": "yes", "logprob": -0.5},
             {"prompt": "b", "answer": "yes", "logprob": -1.0}]
    after = {"a": ("yes", -0.9), "b": ("no", -3.0)}

    scored = score_battery(items, lambda p: after[p])
    summary = battery_summary(scored)

    assert [r["correct"] for r in scored] == [True, False]
    assert scored[0]["logprob_delta"] == -0.4
    assert summary["items"] == 2 and summary["lost"] == 1
    assert summary["retained_rate"] == 0.5
    assert summary["mean_logprob_delta"] == -1.2


def test_nll_from_logits_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    ids = torch.randint(0, 11, (6,))

    expected = torch.nn.functional.cross_entropy(logits[:-1], ids[1:]).item()

    assert abs(nll_from_logits(logits, ids) - expected) < 1e-6


def test_nll_from_logits_is_zero_for_a_confident_correct_model():
    ids = torch.tensor([1, 2, 3])
    logits = torch.zeros(3, 4)
    logits[0, 2] = 50.0
    logits[1, 3] = 50.0
    assert nll_from_logits(logits, ids) < 1e-6


def test_margin_is_the_gap_between_the_two_summed_logprobs():
    margin, installed = code_margin(-3.0, -5.5)
    assert margin == pytest.approx(2.5)
    assert installed


def test_a_fact_installs_only_at_a_full_nat_of_margin():
    assert not code_margin(-3.0, -3.5)[1]
    assert code_margin(-3.0, -4.0)[1]  # exactly the registered threshold
    assert not code_margin(-4.0, -3.0)[1]


class _ConstModel:
    """Returns fixed logits for whatever it is handed -- enough to pin the
    reduction the margin is calibrated to."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, ids, state=None):
        return self.logits[:, : ids.shape[1]], None


def test_code_logprob_sums_over_the_whole_code_rather_than_averaging():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 9)
    model = _ConstModel(logits)
    prompt, target = torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])

    total = logprob_sum(model, prompt, target, None)

    assert total == pytest.approx(2 * target_logprob(model, prompt, target, None))
