"""CPU tests for sanity_sample's window selection -- no tokenizer, no model."""

import sys
from pathlib import Path

import torch


from sanity_sample import pick_windows


def test_prefers_windows_around_recall_credit():
    ids = torch.arange(1000)
    recall = torch.zeros(1000, dtype=torch.bool)
    recall[600:605] = True
    wins = pick_windows(n_tokens=1000, recall=recall, sleeps=torch.tensor([300]), n=1, width=100, seed=0)
    lo, hi, kind = wins[0]
    assert kind == "recall"
    assert lo <= 600 and hi >= 605
    assert 0 <= lo and hi <= 1000


def test_falls_back_to_sleeps_then_random():
    sleep_wins = pick_windows(n_tokens=1000, recall=None, sleeps=torch.tensor([300]), n=1, width=100, seed=0)
    assert sleep_wins[0][2] == "sleep"
    lo, hi, _ = sleep_wins[0]
    assert lo <= 300 <= hi

    rand_wins = pick_windows(n_tokens=1000, recall=None, sleeps=None, n=2, width=100, seed=0)
    assert len(rand_wins) == 2
    assert all(k == "random" and 0 <= lo and hi <= 1000 for lo, hi, k in rand_wins)


def test_short_sequence_clamps_and_never_exceeds_bounds():
    wins = pick_windows(n_tokens=40, recall=None, sleeps=None, n=3, width=100, seed=1)
    assert all(lo == 0 and hi == 40 for lo, hi, _ in wins)
