"""CPU tests for prepare_data.py's conversation formatting against the real
tokenizer (cached locally; build_tokenizer prefers local files). The point
pinned down here: role markers round-trip as the single registered
special-token ids, never as their multi-token BPE spellings -- the stale
pre-registration data/train.pt artifact carried BPE-spelled markers, and
every downstream consumer assumes the single-id encoding.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.common import build_tokenizer
from prepare_data import format_conversation

USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(
        SimpleNamespace(TOKENIZER_ID="EleutherAI/gpt-neox-20b", SPECIAL_TOKENS=[USER_OPEN, ASST_OPEN])
    )


def _bpe_spelling(tokenizer, marker: str) -> list[int]:
    return tokenizer(marker, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def _contains(ids: list[int], pattern: list[int]) -> bool:
    return any(ids[i : i + len(pattern)] == pattern for i in range(len(ids) - len(pattern) + 1))


def test_format_conversation_markers_are_single_special_ids(tokenizer):
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(ASST_OPEN)
    messages = [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "And at night?"},
        {"role": "assistant", "content": "Black."},
    ]
    ids, mask, qoff = format_conversation(messages, tokenizer, 4096, USER_OPEN, ASST_OPEN)
    assert qoff is None

    assert ids.count(user_id) == 2
    assert ids.count(asst_id) == 2
    assert ids[0] == user_id
    for marker in (USER_OPEN, ASST_OPEN):
        spelled = _bpe_spelling(tokenizer, marker)
        assert len(spelled) > 1  # sanity: the spelling really is multi-token
        assert not _contains(ids, spelled)
    assert len(ids) == len(mask)


def test_question_message_records_offset_with_identical_tokens(tokenizer):
    doc = "The ball is in the garden. Mary went to the office."
    q = "Where is the ball?"
    target = "garden"
    plain_ids, _, plain_qoff = format_conversation(
        [{"role": "user", "content": f"{doc}\n{q}"}, {"role": "assistant", "content": target}],
        tokenizer, 4096, USER_OPEN, ASST_OPEN,
    )
    split_ids, split_mask, qoff = format_conversation(
        [{"role": "user", "content": doc, "question": q}, {"role": "assistant", "content": target}],
        tokenizer, 4096, USER_OPEN, ASST_OPEN,
    )

    assert plain_qoff is None
    # The question-bearing shape tokenizes identically to today's joined shape;
    # only the metadata is new.
    assert split_ids == plain_ids
    assert len(split_mask) == len(split_ids)
    assert qoff is not None
    assert tokenizer.decode(split_ids[qoff:]).startswith(q)
    # The token right before the question is the tail of the document, so a
    # cut at qoff separates document from question with no token invented.
    assert tokenizer.decode(split_ids[:qoff]).endswith("office.\n")


def test_question_dropped_by_max_len_truncation_yields_no_offset(tokenizer):
    ids, _, qoff = format_conversation(
        [{"role": "user", "content": "word " * 100, "question": "Where?"},
         {"role": "assistant", "content": "there"}],
        tokenizer, 10, USER_OPEN, ASST_OPEN,
    )
    assert ids == []
    assert qoff is None


def test_batched_injected_turn_encoding_uses_special_ids(tokenizer):
    # The prepare_interference/prepare_chains splice path: injected turns are
    # encoded with a batched tokenizer(...) call on "<marker> content" strings.
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(ASST_OPEN)
    encoded = tokenizer(
        [f"{USER_OPEN} What was the code for river?", f"{ASST_OPEN} The code for river is 4 8 2 1 3."],
        add_special_tokens=False,
    )["input_ids"]
    assert encoded[0][0] == user_id
    assert encoded[1][0] == asst_id
    for row in encoded:
        for marker in (USER_OPEN, ASST_OPEN):
            assert not _contains(row, _bpe_spelling(tokenizer, marker))
