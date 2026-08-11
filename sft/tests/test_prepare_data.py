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
from prepare_data import format_conversation, format_pack, pack_records, recap_messages

USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
EOC = "<|endofconversation|>"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(
        SimpleNamespace(TOKENIZER_ID="EleutherAI/gpt-neox-20b",
                        SPECIAL_TOKENS=[USER_OPEN, ASST_OPEN, EOC])
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


def _conv(n: int) -> list[dict]:
    return [
        {"role": "user", "content": f"Question {n} about gardening. Second sentence."},
        {"role": "assistant", "content": f"Answer {n} about gardening. Second sentence."},
        {"role": "user", "content": f"Follow-up {n}?"},
        {"role": "assistant", "content": f"Follow-up answer {n}."},
    ]


def test_format_conversation_closes_the_conversation_with_the_boundary_token(tokenizer):
    eoc_id = tokenizer.convert_tokens_to_ids(EOC)
    ids, mask, _ = format_conversation(_conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN, eoc=EOC)

    assert ids[-1] == eoc_id
    assert ids.count(eoc_id) == 1
    assert mask[-1] is True or mask[-1] == 1  # the model has to learn to emit it
    assert len(ids) == len(mask)


def test_format_conversation_without_a_boundary_token_is_unchanged(tokenizer):
    plain, _, _ = format_conversation(_conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN)
    closed, _, _ = format_conversation(_conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN, eoc=EOC)

    assert closed == plain + [tokenizer.convert_tokens_to_ids(EOC)]


def test_packing_puts_two_or_three_conversations_behind_boundaries(tokenizer):
    import random

    records = [{"messages": _conv(i)} for i in range(30)]
    groups = pack_records(records, random.Random(0), recap_rate=0.0)

    assert all(2 <= len(g) <= 3 for g in groups[:-1])  # the tail takes what is left
    assert 1 <= len(groups[-1]) <= 3
    assert {kind for g in groups for kind, _ in g} == {"fresh"}
    # Every source conversation is used exactly once when nothing is recapped.
    assert sum(len(g) for g in groups) == len(records)

    eoc_id = tokenizer.convert_tokens_to_ids(EOC)
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    for group in groups:
        ids, mask, _, _ = format_pack(group, tokenizer, 4096, USER_OPEN, ASST_OPEN, EOC)
        assert len(ids) == len(mask)
        assert ids[-1] == eoc_id
        assert ids.count(eoc_id) == len(group)
        # Every boundary but the last opens a fresh conversation.
        for i, token in enumerate(ids[:-1]):
            if token == eoc_id:
                assert ids[i + 1] == user_id


def test_a_conversation_that_does_not_fit_is_dropped_whole(tokenizer):
    eoc_id = tokenizer.convert_tokens_to_ids(EOC)
    group = [("fresh", _conv(0)), ("fresh", _conv(1))]
    full, _, _, packed = format_pack(group, tokenizer, 4096, USER_OPEN, ASST_OPEN, EOC)
    assert packed == 2

    budget = full.index(eoc_id) + 3  # room for the first conversation and no more
    ids, mask, _, packed = format_pack(group, tokenizer, budget, USER_OPEN, ASST_OPEN, EOC)

    assert packed == 1
    assert ids.count(eoc_id) == packed
    assert ids[-1] == eoc_id
    assert len(ids) == len(mask)


def test_every_boundary_is_a_recap_at_rate_one():
    import random

    records = [{"messages": _conv(i)} for i in range(30)]
    groups = pack_records(records, random.Random(0), recap_rate=1.0)

    assert all(kind == "recap" for g in groups for kind, _ in g[1:])
    assert all(kind == "fresh" for g in groups for kind, _ in g[:1])


def test_recap_quotes_an_exchange_of_the_conversation_it_follows():
    import random

    messages = _conv(7)
    recap = recap_messages(messages, random.Random(0))

    assert [m["role"] for m in recap] == ["user", "assistant", "user", "assistant"]
    quoted = " ".join(m["content"] for m in recap)
    sources = [m["content"] for m in messages]
    # Every recap answer is a quote of a turn of the conversation it follows.
    assert any(src.split(".")[0] in quoted for src in sources)


def test_recap_of_a_conversation_with_no_complete_exchange_is_empty():
    import random

    assert recap_messages([{"role": "user", "content": "hello?"}], random.Random(0)) == []


def test_a_recap_can_target_an_earlier_conversation_not_only_the_last():
    """The failure this fixes: recapping only the conversation just closed
    teaches RECENCY recall, which is exactly the bias that left the two
    earliest facts of a wake transcript at zero spontaneous rehearsals."""
    import random

    records = [{"messages": _conv(i)} for i in range(60)]
    groups = pack_records(records, random.Random(0), recap_rate=1.0)

    triples = [g for g in groups if len(g) == 3 and g[2][0] == "recap"]
    assert triples, "no 3-pack ending in a recap to inspect"
    # Across the corpus, some recap quotes reach past the conversation
    # immediately before them to the one that opened the pack.
    reached_back = 0
    for group in triples:
        quoted = group[2][1][1]["content"]
        first_answer = group[0][1][1]["content"][:20]
        if first_answer in quoted:
            reached_back += 1
    assert reached_back > 0


def test_an_enumerating_recap_quotes_several_pairs_and_invents_nothing():
    import random

    from prepare_data import recap_messages

    # Six pairs, so the quotable head half still holds several: enumeration is
    # capped by what survives truncation, never by inventing filler.
    messages = [
        {"role": "user", "content": "First question about gardening."},
        {"role": "assistant", "content": "First answer about gardening."},
        {"role": "user", "content": "Second question about beekeeping."},
        {"role": "assistant", "content": "Second answer about beekeeping."},
        {"role": "user", "content": "Third question about composting."},
        {"role": "assistant", "content": "Third answer about composting."},
    ]
    for _ in range(len(messages)):
        messages.append({"role": "user", "content": "Tail question."})
        messages.append({"role": "assistant", "content": "Tail answer."})
    recap = recap_messages(messages, random.Random(0), n_pairs=2)

    text = " ".join(m["content"] for m in recap)
    quoted = text.split('"')[1::2]

    assert len(set(quoted)) == 4  # two distinct pairs, question and answer each
    # Every quoted fragment comes from the source conversation: the only words
    # the builder contributes are its own fixed scaffolding.
    for fragment in quoted:
        assert any(fragment in m["content"] for m in messages)


def test_a_recap_quotes_from_the_head_of_its_source_conversation():
    """--max-len truncation drops a conversation's TAIL turns, so a recap that
    quoted a tail pair would ask the model to recall text that never made it
    into the example: hallucination training. Quoting from the head keeps the
    quoted turn on the right side of any truncation."""
    import random

    from prepare_data import recap_messages

    messages = []
    for i in range(8):
        messages.append({"role": "user", "content": f"Question {i} about gardening."})
        messages.append({"role": "assistant", "content": f"Answer {i} about gardening."})

    for seed in range(25):
        recap = recap_messages(messages, random.Random(seed))
        text = " ".join(m["content"] for m in recap)
        quoted = [q for q in text.split('"')[1::2]]
        for fragment in quoted:
            index = next(i for i, m in enumerate(messages) if fragment in m["content"])
            assert index < len(messages) // 2, f"recap reached into the tail: {fragment!r}"
