"""CPU tests for acceptance_check.py -- DISCUSSION-20260808 §2.1's warm-start
acceptance clause, counted at the token level. The clause exists because a
DECODED dream cannot distinguish the real `[USER]` special token from a
plain-text imitation of it, so every assertion here is about ids.
"""

import sys
from pathlib import Path

import pytest


from acceptance_check import MARKER_SHARE_MAX, acceptance


class _Dream:
    def __init__(self, ids, texts, prefix_len=0):
        self.dream_ids = ids
        self.token_texts = texts
        self.prefix_len = prefix_len


def _spans(*dreams):
    return list(dreams)


def test_real_markers_outnumbering_plain_brackets_passes():
    # 8 real markers, 2 plain ']' -> 20% < 35%
    ids = [50277, 50278] * 4 + [62, 62] + [100] * 20
    texts = ["[USER]", "[ASSISTANT]"] * 4 + ["]", "]"] + ["x"] * 20
    result = acceptance(_spans(_Dream(ids, texts)), user_id=50277, asst_id=50278, plain_id=62)

    assert result["plain"] == 2
    assert result["markers"] == 8
    assert result["share"] == pytest.approx(0.2)
    assert result["non_ascii"] == 0
    assert result["passed"] is True


def test_bracket_mimicry_over_the_threshold_fails():
    # the 400-step calibration point: 17 plain, 5 real -> 77%
    ids = [62] * 17 + [50277] + [50278] * 4 + [100] * 20
    texts = ["]"] * 17 + ["[USER]"] + ["[ASSISTANT]"] * 4 + ["x"] * 20
    result = acceptance(_spans(_Dream(ids, texts)), user_id=50277, asst_id=50278, plain_id=62)

    assert result["share"] == pytest.approx(17 / 22)
    assert result["share"] > MARKER_SHARE_MAX
    assert result["passed"] is False


def test_any_mojibake_fails_whatever_the_bracket_share():
    ids = [50277, 50278, 100]
    result = acceptance(_spans(_Dream(ids, ["[USER]", "[ASSISTANT]", "caf�"])),
                        user_id=50277, asst_id=50278, plain_id=62)

    assert result["non_ascii"] == 1
    assert result["share"] == 0.0  # the bracket clause on its own would pass
    assert result["passed"] is False


def test_the_steer_prefix_is_not_scored():
    # The prefix is authored, not emitted: its tokens say nothing about what
    # the warm start taught.
    ids = [62, 62, 62, 50277, 100]
    texts = ["]", "]", "]", "[USER]", "x"]
    result = acceptance(_spans(_Dream(ids, texts, prefix_len=3)),
                        user_id=50277, asst_id=50278, plain_id=62)

    assert result["plain"] == 0
    assert result["markers"] == 1
    assert result["passed"] is True


def test_no_marker_slot_emissions_at_all_is_not_a_pass():
    # Nothing to divide by: a dream that never opens a turn has not shown the
    # markers survived, so it cannot clear the clause by vacuous arithmetic.
    result = acceptance(_spans(_Dream([100] * 30, ["x"] * 30)),
                        user_id=50277, asst_id=50278, plain_id=62)

    assert result["markers"] == 0
    assert result["plain"] == 0
    assert result["passed"] is False


def test_counts_pool_over_every_dream_of_a_set():
    a = _Dream([50277, 62], ["[USER]", "]"])
    b = _Dream([50278, 50278], ["[ASSISTANT]", "[ASSISTANT]"])
    result = acceptance(_spans(a, b), user_id=50277, asst_id=50278, plain_id=62)

    assert result["markers"] == 3
    assert result["plain"] == 1
    assert result["dreams"] == 2
