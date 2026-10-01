"""CPU tests for the babilong -> item conversion. Block assembly itself is
`preparation.cram`'s and is covered by `test_prepare_cram.py`.
"""

import sys
from pathlib import Path

from preparation.needles import babilong_items

STORY = "Mary went to the bathroom. John moved to the garden. Sandra picked up the apple."


def test_target_becomes_the_whole_credited_span():
    items = babilong_items(
        [{"input": STORY, "question": "Where is Mary?", "target": "bathroom"}], "qa1"
    )
    (item,) = items
    assert item["answer"][item["span"][0] : item["span"][1]] == "bathroom"
    assert item["cue"] == "Where is Mary?"
    assert item["source"] == STORY
    assert item["meta"]["entity"] == "bathroom"


def test_a_target_already_in_the_question_is_dropped():
    assert (
        babilong_items(
            [{"input": STORY, "question": "Is Mary in the bathroom?", "target": "bathroom"}], "qa6"
        )
        == []
    )


def test_a_target_absent_from_the_story_is_dropped():
    assert (
        babilong_items(
            [{"input": STORY, "question": "Where is Daniel?", "target": "kitchen"}], "qa1"
        )
        == []
    )


def test_item_ids_are_unique_per_task_and_index():
    records = [{"input": STORY, "question": "Where is Mary?", "target": "bathroom"}] * 3
    ids = [
        it["meta"]["article"]
        for it in babilong_items(records, "qa1") + babilong_items(records, "qa2")
    ]
    assert len(set(ids)) == len(ids) == 6
