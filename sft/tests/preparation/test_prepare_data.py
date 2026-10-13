"""CPU tests for preparation/conversations.py's conversation formatting against the real
tokenizer (cached locally; build_tokenizer prefers local files). The point
pinned down here: role markers round-trip as the single registered
special-token ids, never as their multi-token BPE spellings -- the stale
pre-registration data/train.pt artifact carried BPE-spelled markers, and
every downstream consumer assumes the single-id encoding.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from models.common import build_tokenizer

import pytest

from preparation.conversations import format_conversation, format_pack, pack_records, render_pack

USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
EOC = "<|endofconversation|>"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(
        SimpleNamespace(
            TOKENIZER_ID="EleutherAI/gpt-neox-20b", SPECIAL_TOKENS=[USER_OPEN, ASST_OPEN, EOC]
        )
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
        tokenizer,
        4096,
        USER_OPEN,
        ASST_OPEN,
    )
    split_ids, split_mask, qoff = format_conversation(
        [{"role": "user", "content": doc, "question": q}, {"role": "assistant", "content": target}],
        tokenizer,
        4096,
        USER_OPEN,
        ASST_OPEN,
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
        [
            {"role": "user", "content": "word " * 100, "question": "Where?"},
            {"role": "assistant", "content": "there"},
        ],
        tokenizer,
        10,
        USER_OPEN,
        ASST_OPEN,
    )
    assert ids == []
    assert qoff is None


def test_batched_injected_turn_encoding_uses_special_ids(tokenizer):
    # The prepare_interference/prepare_chains splice path: injected turns are
    # encoded with a batched tokenizer(...) call on "<marker> content" strings.
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(ASST_OPEN)
    encoded = tokenizer(
        [
            f"{USER_OPEN} What was the code for river?",
            f"{ASST_OPEN} The code for river is 4 8 2 1 3.",
        ],
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


def test_train_user_turns_masks_user_tokens_true_and_assistant_turns_stay_true(tokenizer):
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    ids, default_mask, _ = format_conversation(_conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN)
    _, mask, _ = format_conversation(
        _conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN, train_user_turns=True
    )

    user_positions = [i for i, token in enumerate(ids) if token == user_id]
    assert len(user_positions) == 2
    assert all(not default_mask[i] for i in user_positions)
    assert all(mask)
    packed_ids, packed_mask, _, _ = format_pack(
        [("fresh", _conv(0)), ("fresh", _conv(1))],
        tokenizer,
        4096,
        USER_OPEN,
        ASST_OPEN,
        EOC,
        train_user_turns=True,
    )
    assert len(packed_ids) == len(packed_mask) and all(packed_mask)


def test_format_conversation_without_a_boundary_token_is_unchanged(tokenizer):
    plain, _, _ = format_conversation(_conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN)
    closed, _, _ = format_conversation(_conv(0), tokenizer, 4096, USER_OPEN, ASST_OPEN, eoc=EOC)

    assert closed == plain + [tokenizer.convert_tokens_to_ids(EOC)]


def test_packing_at_repeat_rate_zero_uses_every_record_once_two_to_five_per_pack(tokenizer):
    import random

    records = [{"messages": _conv(i)} for i in range(60)]
    groups = pack_records(records, random.Random(0), repeat_rate=0.0)

    assert all(2 <= len(g) <= 5 for g in groups[:-1])  # the tail takes what is left
    assert 1 <= len(groups[-1]) <= 5
    assert {len(g) for g in groups[:-1]} == {2, 3, 4, 5}
    assert {kind for g in groups for kind, _ in g} == {"fresh"}
    used = [record for g in groups for _, record in g]
    assert len(used) == len(records)
    assert all(a is r for a, r in zip(used, records, strict=True))

    eoc_id = tokenizer.convert_tokens_to_ids(EOC)
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    rng = random.Random(0)
    for group in groups:
        rendered = render_pack(group, rng)
        ids, mask, _, _ = format_pack(rendered, tokenizer, 4096, USER_OPEN, ASST_OPEN, EOC)
        assert len(ids) == len(mask)
        assert ids[-1] == eoc_id
        assert ids.count(eoc_id) == len(group)
        # Every boundary but the last opens a new conversation.
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


def _repeat_sources(group) -> list[int | None]:
    """For each slot, the index of the earlier fresh slot it repeats (None if fresh)."""
    return [
        None
        if kind == "fresh"
        else next(k for k in range(j) if group[k][0] == "fresh" and group[k][1] is messages)
        for j, (kind, messages) in enumerate(group)
    ]


def test_at_repeat_rate_one_every_later_slot_repeats_while_a_source_is_eligible():
    import random

    records = [{"messages": _long_conv(6)} for _ in range(60)]
    groups = pack_records(records, random.Random(0), repeat_rate=1.0)

    assert any(kind == "repeat" for g in groups for kind, _ in g)
    for group in groups:
        assert group[0][0] == "fresh"
        sources = _repeat_sources(group)
        repeated = [s for s in sources if s is not None]
        assert len(repeated) == len(set(repeated)), "a source repeated twice in one pack"
        for j in range(1, len(group)):
            eligible = [k for k in range(j) if group[k][0] == "fresh" and k not in sources[:j]]
            assert (group[j][0] == "repeat") == bool(eligible)


def test_repeats_occur_both_adjacent_and_gapped():
    import random

    records = [{"messages": _long_conv(6)} for _ in range(400)]
    groups = pack_records(records, random.Random(0), repeat_rate=0.3)

    adjacent = gapped = 0
    for group in groups:
        for j, src in enumerate(_repeat_sources(group)):
            if src is not None:
                adjacent += src == j - 1
                gapped += src < j - 1
    assert adjacent > 0 and gapped > 0
    # Repeats never consume pool records: every record still appears once fresh.
    assert sum(kind == "fresh" for g in groups for kind, _ in g) == len(records)


def _long_conv(n_turns: int) -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}."}
        for i in range(n_turns)
    ]


def test_an_ultrachat_repeat_splits_at_an_exchange_and_replays_a_suffix_of_the_prefix():
    import random

    original = _long_conv(10)
    record, other = {"messages": original}, {"messages": _conv(0)}
    cuts: set[int] = set()
    for seed in range(400):
        rendered = render_pack(
            [("fresh", record), ("fresh", other), ("repeat", record)], random.Random(seed)
        )
        prefix, continuation = rendered[0][1], rendered[2][1]
        k = len(prefix)
        assert k % 2 == 0 and 2 <= k <= len(original) - 2
        j = len(continuation) - (len(original) - k)
        assert j % 2 == 0 and 0 <= j <= k
        assert continuation[0]["role"] == "user"
        assert continuation[:j] == prefix[k - j :]
        assert prefix + continuation[j:] == original
        cuts.add(k)
        assert rendered[1][1] == other["messages"]
    assert cuts == {2, 4, 6, 8}


def test_a_full_replay_and_a_pure_continuation_both_occur():
    import random

    original = _long_conv(6)
    record = {"messages": original}
    seen = set()
    for seed in range(200):
        rendered = render_pack([("fresh", record), ("repeat", record)], random.Random(seed))
        k = len(rendered[0][1])
        j = len(rendered[1][1]) - (len(original) - k)
        seen.add("pure" if j == 0 else "full" if j == k else "partial")
    assert seen == {"pure", "full", "partial"}


def test_a_conversation_under_six_turns_is_never_a_repeat_source():
    import random

    records = [{"messages": _conv(i)} for i in range(100)]
    records += [{"messages": _long_conv(6)} for _ in range(100)]
    random.Random(0).shuffle(records)
    groups = pack_records(records, random.Random(0), repeat_rate=1.0)

    repeated = [r for g in groups for kind, r in g if kind == "repeat"]
    assert repeated and all(len(r["messages"]) >= 6 for r in repeated)
    assert sum(kind == "fresh" for g in groups for kind, _ in g) == len(records)


def test_an_unrepeated_ultrachat_conversation_renders_whole():
    import random

    long, short = {"messages": _long_conv(8)}, {"messages": _conv(0)}
    for seed in range(50):
        rendered = render_pack([("fresh", long), ("fresh", short)], random.Random(seed))
        assert rendered[0][1] == long["messages"] and rendered[1][1] == short["messages"]


@pytest.fixture
def dolly_helpers(monkeypatch):
    """The real preparation.dolly helpers when they exist, else stand-ins with
    the same contract."""
    from preparation import dolly

    if not hasattr(dolly, "render_document"):
        monkeypatch.setattr(
            dolly,
            "render_document",
            lambda row, rng: [
                {"role": "user", "content": row["context"]},
                {"role": "assistant", "content": rng.choice(("Okay.", "Got it."))},
            ],
            raising=False,
        )
        monkeypatch.setattr(
            dolly, "RECALL_REQUESTS", ("Repeat it.", "What did I send?"), raising=False
        )
        monkeypatch.setattr(
            dolly, "RECALL_GIST_REQUESTS", ("Gist of it?", "Sum it up."), raising=False
        )
    return dolly


def _row(i: int, summary: str | None = None) -> dict:
    return {
        "context": f"Context {i} about lighthouses.",
        "instruction": f"Instruction {i}?",
        "response": f"Response {i}.",
        "summary": summary,
    }


def _dolly_record(dolly, shape: str, rows: list[dict]) -> dict:
    import random

    if shape == "plain":
        messages = [
            {"role": "user", "content": rows[0]["context"] + "\n\n" + rows[0]["instruction"]},
            {"role": "assistant", "content": rows[0]["response"]},
        ]
    else:
        rng = random.Random(0)
        messages = [m for row in rows for m in dolly.render_document(row, rng)]
    return {"messages": messages, "dolly": {"shape": shape, "rows": rows}}


def _variants(record: dict, draws: int = 300) -> dict[str, list[list[dict]]]:
    import random

    out: dict[str, list[list[dict]]] = {}
    for seed in range(draws):
        rendered = render_pack([("fresh", record), ("repeat", record)], random.Random(seed))
        assert rendered[0][1] == record["messages"]
        out.setdefault(rendered[1][0], []).append(rendered[1][1])
    return out


def _alternates(messages: list[dict]) -> bool:
    roles = [m["role"] for m in messages]
    return roles == ["user", "assistant"] * (len(roles) // 2) and len(roles) % 2 == 0


def test_every_dolly_variant_occurs_and_summary_only_where_available(dolly_helpers):
    summarised = _dolly_record(dolly_helpers, "document", [_row(0, "Summary 0.")])
    assert len(_variants(summarised)) == 5
    for record in (
        _dolly_record(dolly_helpers, "plain", [_row(0, "Summary 0.")]),
        _dolly_record(dolly_helpers, "document", [_row(0)]),
        _dolly_record(dolly_helpers, "multi", [_row(0), _row(1)]),
    ):
        variants = _variants(record)
        assert len(variants) == 4 and "summary" not in variants
    for record in (summarised, _dolly_record(dolly_helpers, "multi", [_row(0), _row(1)])):
        for rendered in (r for found in _variants(record).values() for r in found):
            assert _alternates(rendered)


def test_dolly_verbatim_question_recall_and_summary_render_what_they_name(dolly_helpers):
    rows = [_row(0, "Summary 0."), _row(1)]
    variants = _variants(_dolly_record(dolly_helpers, "multi", rows))
    record = _dolly_record(dolly_helpers, "multi", rows)

    assert all(r == record["messages"] for r in variants["verbatim"])
    for rendered in variants["question"]:
        assert rendered == [
            m
            for row in rows
            for m in (
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["response"]},
            )
        ]
        assert not any("lighthouses" in m["content"] for m in rendered)
    for rendered in variants["recall"]:
        gist, exact = rendered[0:2], rendered[2:4]
        assert gist[0]["content"] in dolly_helpers.RECALL_GIST_REQUESTS
        assert gist[1]["content"] == rows[0]["summary"]
        assert exact[0]["content"] in dolly_helpers.RECALL_REQUESTS
        assert exact[1]["content"] == rows[1]["context"]
    for rendered in variants["summary"]:
        users = [m["content"] for m in rendered if m["role"] == "user"]
        assert "Summary 0." in users[0] and rows[0]["context"] not in users[0]
        assert rows[1]["context"] in users[1]


def test_dolly_multi_turn_stacks_the_records_distinct_exchanges(dolly_helpers):
    cases = [
        ("plain", [_row(0)], 4),
        ("document", [_row(0, "Summary 0.")], 6),
        ("multi", [_row(0), _row(1)], 4),
        ("multi", [_row(0), _row(1), _row(2)], 6),
    ]
    for shape, rows, turns in cases:
        found = _variants(_dolly_record(dolly_helpers, shape, rows))["multi-turn"]
        assert found
        for rendered in found:
            assert len(rendered) == turns and _alternates(rendered)
    orders = {
        tuple(m["content"] for m in r if m["role"] == "user")
        for r in _variants(_dolly_record(dolly_helpers, "plain", [_row(0)]))["multi-turn"]
    }
    assert len(orders) > 1


def test_a_dolly_verbatim_repeat_renders_token_identical_to_its_source(tokenizer, dolly_helpers):
    import random

    eoc_id = tokenizer.convert_tokens_to_ids(EOC)
    record = _dolly_record(dolly_helpers, "document", [_row(0)])
    group = [("fresh", record), ("fresh", {"messages": _conv(1)}), ("repeat", record)]
    rendered = next(
        r
        for r in (render_pack(group, random.Random(seed)) for seed in range(100))
        if r[2][0] == "verbatim"
    )
    ids, _, _, packed = format_pack(rendered, tokenizer, 4096, USER_OPEN, ASST_OPEN, EOC)

    assert packed == 3
    cuts = [i + 1 for i, t in enumerate(ids) if t == eoc_id]
    parts = [ids[a:b] for a, b in zip([0] + cuts, cuts)]
    assert parts[2] == parts[0]
    assert parts[1] != parts[0]


def test_rendered_packs_alternate_roles_and_open_fresh(dolly_helpers):
    import random

    records = [
        _dolly_record(dolly_helpers, "document", [_row(i, f"Summary {i}.")]) for i in range(200)
    ]
    records += [{"messages": _long_conv(6 + 2 * (i % 4))} for i in range(40)]
    records += [{"messages": _conv(i)} for i in range(40)]
    random.Random(1).shuffle(records)
    rng = random.Random(0)
    groups = pack_records(records, rng, repeat_rate=0.5)
    labels = set()
    for group in groups:
        rendered = render_pack(group, rng)
        assert rendered[0][0] == "fresh"
        for label, messages in rendered:
            assert messages and _alternates(messages)
            labels.add(label.split(" k=")[0])
    assert {"fresh", "ultrachat", "verbatim", "question", "recall", "summary"} <= labels


def test_iter_records_pools_input_and_hf_dataset_shuffled_by_seed(tmp_path, monkeypatch):
    import argparse
    import json

    import datasets

    from preparation import conversations

    def record(tag: str) -> dict:
        return {
            "messages": [{"role": "user", "content": tag}, {"role": "assistant", "content": "ok"}]
        }

    local = tmp_path / "local.jsonl"
    local.write_text("".join(json.dumps(record(f"local {i}")) + "\n" for i in range(20)))
    hf_rows = [record(f"hf {i}") for i in range(50)]
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: datasets.Dataset.from_list(hf_rows)
    )

    def pool(seed: int) -> list[str]:
        args = argparse.Namespace(
            input=str(local), hf_dataset="fake/ds", hf_split="train_sft", max_examples=30, seed=seed
        )
        return [r["messages"][0]["content"] for r in conversations.iter_records(args)]

    expected = {f"local {i}" for i in range(20)} | {f"hf {i}" for i in range(30)}
    assert len(pool(0)) == 50 and set(pool(0)) == expected
    assert pool(0) == pool(0)
    assert pool(0) != pool(1)
    # Shuffled into one pool: neither source sits as a contiguous block.
    first_twenty = {tag.split()[0] for tag in pool(0)[:20]}
    assert first_twenty == {"local", "hf"}
