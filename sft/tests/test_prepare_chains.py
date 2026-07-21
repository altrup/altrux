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
        split_episode_rate=0.0, split_min_part=30,
        split_qa_rate=None, split_gap_min=2, split_gap_max=2,
        sentence_sleep_rate=0.0,
        fact_rate=1.0, min_facts=2, max_facts=4, min_queries=1, max_queries=3,
        revise_rate=0.5, cross_sleep_bias=0.0, seed=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build(n_episodes=40, n_turns=6, turn_len=10, pool=None, sent_end_ids=None,
           space_start_ids=None, **overrides):
    if pool is None:
        pool = [_episode(n_turns, turn_len) for _ in range(n_episodes)]
    labels = [f"lbl{i}" for i in range(64)]
    return build_chains(
        [ids for ids, _ in pool], [m for _, m in pool],
        encode=_stub_encode, labels=labels,
        user_open=USER, asst_open=ASST, user_id=USER_ID, asst_id=ASST_ID,
        args=_args(**overrides),
        sent_end_ids=sent_end_ids, space_start_ids=space_start_ids,
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


def test_split_episodes_conserve_content_and_sleep_on_boundaries():
    n_episodes, n_turns, turn_len = 20, 12, 10
    dataset, stats = _build(
        n_episodes=n_episodes, n_turns=n_turns, turn_len=turn_len, fact_rate=0.0,
        split_episode_rate=1.0,
        min_wake=4, max_wake=4, min_budget=500, max_budget=500,
    )
    assert stats["n_split"] > 0
    # Splitting reorders segments but never drops or duplicates tokens.
    out = torch.cat(dataset["ids"])
    inp = torch.cat([_episode(n_turns, turn_len)[0] for _ in range(n_episodes)])
    assert torch.equal(out.sort().values, inp.sort().values)
    # Every sleep (including each forced pre-tail sleep) is on a turn boundary,
    # and every chain with a split has at least one sleep.
    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        assert len(sleeps) > 0
        for s in sleeps.tolist():
            assert ids[s].item() in (USER_ID, ASST_ID)


def _qa_episode(filler: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Two turns only: a long user document turn, then a long answer --
    the split cut can only land on the answer boundary."""
    ids = [USER_ID] + [filler] * 79 + [ASST_ID] + [filler] * 79
    mask = [False] * 81 + [True] * 79
    return torch.tensor(ids), torch.tensor(mask)


def test_split_qa_rate_splits_only_single_qa_episodes():
    pool = [_episode(6, 20) for _ in range(10)] + [_qa_episode() for _ in range(10)]
    dataset, stats = _build(
        pool=pool, fact_rate=0.0,
        split_episode_rate=0.0, split_qa_rate=1.0,
        min_wake=4, max_wake=4, min_budget=10_000, max_budget=10_000,
    )
    # Every single-QA episode splits (cut at its answer boundary), no
    # multi-turn episode does.
    assert stats["n_split"] == stats["n_split_qa"] == 10
    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        for s in sleeps.tolist():
            assert ids[s].item() in (USER_ID, ASST_ID)


def test_split_gap_controls_episodes_between_head_and_tail():
    # One splittable episode (unique filler 100) among short unsplittable
    # ones (unique fillers 101..130), all in one chain. Enough shorts that
    # the shuffled head isn't near the chain end (the insert would clamp).
    pool = [_qa_episode(filler=100)]
    for k in range(30):
        ids = [USER_ID] + [101 + k] * 9 + [ASST_ID] + [101 + k] * 9
        pool.append((torch.tensor(ids), torch.tensor([False] * 20)))
    dataset, stats = _build(
        pool=pool, fact_rate=0.0,
        split_episode_rate=1.0, split_gap_min=3, split_gap_max=3,
        min_wake=31, max_wake=31, min_budget=10_000, max_budget=10_000,
    )
    assert stats["n_split"] == 1
    ids = torch.cat(dataset["ids"])
    head_tail = (ids == 100).nonzero().flatten().tolist()
    gap_slice = ids[head_tail[78] + 1 : head_tail[79]]  # between head's last and tail's first filler
    intervening = {v for v in gap_slice.tolist() if v > 100}
    assert len(intervening) == 3


SENT_END, SPACE_START = 30, 31


def _single_qa_episode(n_sentences: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One long user document turn of repeated sentences, then a short answer
    -- the LongAlign shape whose only turn boundaries are at 0 and the end."""
    sentence = [5] * 8 + [SENT_END, SPACE_START]
    ids = [USER_ID] + sentence * n_sentences + [ASST_ID] + [6] * 9
    mask = [False] * (1 + 10 * n_sentences) + [False] + [True] * 9
    return torch.tensor(ids), torch.tensor(mask)


def test_sentence_sleeps_land_at_sentence_starts_inside_document_turns():
    pool = [_single_qa_episode(12) for _ in range(20)]
    dataset, stats = _build(
        pool=pool, fact_rate=0.0,
        sentence_sleep_rate=1.0, mid_sleep_min_len=100,
        sent_end_ids={SENT_END}, space_start_ids={SPACE_START},
        min_wake=4, max_wake=4, min_budget=500, max_budget=500,
    )
    assert stats["n_sentence_sleeps"] > 0
    found = 0
    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        for s in sleeps.tolist():
            if ids[s].item() in (USER_ID, ASST_ID):
                continue  # between-episode sleep
            assert ids[s].item() == SPACE_START and ids[s - 1].item() == SENT_END
            found += 1
    assert found == stats["n_sentence_sleeps"]


def test_sentence_sleep_rate_zero_is_a_noop():
    pool = [_single_qa_episode(12) for _ in range(20)]
    dataset, stats = _build(
        pool=pool, fact_rate=0.0,
        sent_end_ids={SENT_END}, space_start_ids={SPACE_START},
    )
    assert stats["n_sentence_sleeps"] == 0
    for ids, sleeps in zip(dataset["ids"], dataset["sleep_positions"]):
        for s in sleeps.tolist():
            assert ids[s].item() in (USER_ID, ASST_ID)


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
