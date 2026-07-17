"""Fast (CPU, seconds) tests for prepare_chains.py's chain generator --
build_chains is IO-free (pool tensors in, dataset dict out) with the
tokenizer injected as a plain callable, so these run against a synthetic
episode pool and a stub tokenizer, no downloads.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_chains import build_chains

USER, ASST = "[U]", "[A]"
USER_ID, ASST_ID = 1, 2


def _episode(n_turns: int, turn_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternating user/assistant turns: each turn is a marker token followed
    by turn_len-1 filler tokens; mask True on assistant content."""
    ids, mask = [], []
    for t in range(n_turns):
        marker = USER_ID if t % 2 == 0 else ASST_ID
        ids += [marker] + [10 + t % 5] * (turn_len - 1)
        mask += [False] + [marker == ASST_ID] * (turn_len - 1)
    return torch.tensor(ids), torch.tensor(mask)


def _stub_encode(strings: list[str]) -> list[list[int]]:
    return [[ASST_ID if s.startswith(ASST) else USER_ID, 20, 21, 22] for s in strings]


def _args(**overrides):
    defaults = dict(
        min_budget=300, max_budget=300, min_wake=1, max_wake=2,
        mid_sleep_rate=0.0, mid_sleep_min_len=60,
        fact_rate=1.0, min_facts=2, max_facts=4, min_queries=1, max_queries=3,
        revise_rate=0.5, seed=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build(n_episodes=40, n_turns=6, turn_len=10, **overrides):
    pool = [_episode(n_turns, turn_len) for _ in range(n_episodes)]
    labels = [f"lbl{i}" for i in range(64)]
    return build_chains(
        [ids for ids, _ in pool], [m for _, m in pool],
        encode=_stub_encode, labels=labels,
        user_open=USER, asst_open=ASST, user_id=USER_ID, asst_id=ASST_ID,
        args=_args(**overrides),
    )


def test_every_sleep_lands_on_a_turn_boundary_after_splicing():
    dataset, _ = _build()
    assert sum(len(s) for s in dataset["sleep_positions"]) > 0
    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        assert torch.equal(sleeps, sleeps.sort().values)
        for s in sleeps.tolist():
            # A boundary is where a turn-marker token starts -- a sleep inside
            # a turn would mean the post-splice shift arithmetic is off.
            assert ids[s].item() in (USER_ID, ASST_ID)


def test_wake_of_one_with_no_inserts_sleeps_at_every_episode_boundary():
    ep_len = 6 * 10
    dataset, _ = _build(min_wake=1, max_wake=1, fact_rate=0.0)
    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        n_eps = len(ids) // ep_len
        assert sleeps.tolist() == [ep_len * i for i in range(1, n_eps)]


def test_chain_lengths_respect_the_token_budget():
    ep_len = 6 * 10
    dataset, _ = _build(fact_rate=0.0)
    for ids in dataset["ids"][:-1]:  # the final chain may be a remainder
        assert 300 <= len(ids) < 300 + ep_len


def test_queries_exist_at_all_three_distances():
    _, stats = _build(n_episodes=80)
    assert all(v > 0 for v in stats["dist_counts"].values())


def test_recall_masks_mark_exactly_the_spliced_answer_content():
    dataset, stats = _build()
    n_queries = sum(stats["dist_counts"].values())
    assert n_queries > 0
    total_true = 0
    for ids, masks, recall in zip(dataset["ids"], dataset["masks"], dataset["recall_masks"]):
        if recall is None:
            continue
        assert len(recall) == len(ids) == len(masks)
        true_idx = recall.nonzero().flatten().tolist()
        total_true += len(true_idx)
        for i in true_idx:
            assert masks[i], "recall tokens are assistant content"
        # Each marked run sits right after an assistant marker (the stub
        # answer turn is [ASST_ID, 20, 21, 22] with content marked).
        for i in true_idx:
            if i - 1 not in true_idx:
                assert ids[i - 1].item() == ASST_ID
    # 3 content tokens per stub answer turn
    assert total_true == 3 * n_queries


def test_mid_sleeps_appear_inside_long_episodes():
    dataset, _ = _build(
        n_episodes=20, n_turns=12, turn_len=10, fact_rate=0.0,
        mid_sleep_rate=1.0, mid_sleep_min_len=100,
        min_wake=4, max_wake=4, min_budget=500, max_budget=500,
    )
    ep_len = 12 * 10
    found_mid = False
    for sleeps in dataset["sleep_positions"]:
        for s in sleeps.tolist():
            if s % ep_len != 0:
                found_mid = True
    assert found_mid
