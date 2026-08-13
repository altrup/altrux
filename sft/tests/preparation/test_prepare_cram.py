"""Fast (CPU, seconds) tests for the cram/needle block generator in
prepare_cram.py. build_blocks is IO-free -- items and filler passages in,
dataset dict out, tokenizer injected as a plain callable -- so these run
against synthetic items and a character-level stub tokenizer, no downloads
and no model import.
"""

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


from prepare_cram import add_block_args, build_blocks, make_items, split_articles, split_sentences, validate_blocks

USER_ID, ASST_ID = 1, 2
SEP_ID, NL_ID = ord(" "), ord("\n")


def _encode(strings: list[str]) -> list[list[int]]:
    """Character-level: concatenation-exact, so token offsets are char
    offsets and decoded samples are readable."""
    return [[ord(c) for c in s] for s in strings]


def _word_encode(strings: list[str]) -> list[list[int]]:
    """Word-level with the leading space merged into the word, the way
    byte-level BPE does -- a span starting at the word's first letter only
    aligns if the resolver retries with the space included."""
    out = []
    for s in strings:
        pieces = re.findall(r" ?[^ ]+| ", s)
        out.append([1000 + (hash(p) % 100000) for p in pieces])
    return out


class _Tok:
    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)


def _items(n: int, entity=lambda i: f"Zorblat{i:03d}") -> list[dict]:
    items = []
    for i in range(n):
        ent = entity(i)
        answer = f"The regional capital is {ent} in the northern valley."
        s = answer.index(ent)
        items.append({
            "source": f"Article {i} covers a region of moderate size. {answer} "
                      f"Trade routes crossed it for centuries and the population grew steadily.",
            "cue": answer.replace(ent, "____"),
            "answer": answer,
            "span": (s, s + len(ent)),
            "meta": {"article": f"art{i}", "entity": ent, "entity_type": "LOC"},
        })
    return items


def _fillers(n: int) -> list[str]:
    return [f"Unrelated passage {i} about weather patterns, soil composition and the slow "
            f"drift of sediment along a river delta over many seasons." for i in range(n)]


def _groups(n: int, per: int) -> list[list[dict]]:
    """`n` sources, each carrying `per` items -- the multi-item shape the
    density knob draws on."""
    groups = []
    for g in range(n):
        answers = [f"The regional capital of district {j} is Zorb{g:03d}{j} in the northern valley."
                   for j in range(per)]
        source = (f"Article {g} covers a region of moderate size. " + " ".join(answers)
                  + " Trade routes crossed it for centuries and the population grew steadily.")
        group = []
        for j, answer in enumerate(answers):
            ent = f"Zorb{g:03d}{j}"
            s = answer.index(ent)
            group.append({
                "source": source, "cue": answer.replace(ent, "____"), "answer": answer,
                "span": (s, s + len(ent)),
                "meta": {"article": f"art{g}", "entity": ent, "entity_type": "LOC"},
            })
        groups.append(group)
    return groups


def _args(**overrides):
    defaults = dict(
        seed=0, gap_min=40, ceiling_start=80, ceiling_end=400,
        block_gap_ratio=4, min_block_tokens=400, max_block_tokens=8000,
        max_tail_units=2, item_rate=1.0, max_pending=32, max_items_per_block=0,
        items_per_source_start=1, items_per_source_end=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build(n_items=60, n_fillers=60, encode=_encode, items=None, fillers=None, groups=None,
           **overrides):
    if groups is None:
        groups = [[it] for it in (_items(n_items) if items is None else items)]
    return build_blocks(
        groups,
        _fillers(n_fillers) if fillers is None else fillers,
        encode,
        user_id=USER_ID, asst_id=ASST_ID, sep_id=SEP_ID, nl_id=NL_ID,
        args=_args(**overrides),
    )


# Block budgets wide enough for several turns, so a group's members can all
# reach their cue inside one block (the tiny defaults fit barely one turn).
_DENSE = dict(gap_min=150, ceiling_start=400, ceiling_end=1600, min_block_tokens=2400)


def _per_source(block: list[dict]) -> float:
    return len(block) / len({it["source_start"] for it in block})


def _all_items(dataset) -> list[dict]:
    return [it for block in dataset["items"] for it in block]


def test_curriculum_ceiling_is_non_decreasing_across_the_artifact():
    dataset, _ = _build()
    ceilings = dataset["curriculum"]["ceilings"]
    assert len(ceilings) == len(dataset["ids"]) > 2
    assert ceilings == sorted(ceilings)
    assert ceilings[0] <= 80
    assert ceilings[-1] <= 400


def test_every_item_gap_stays_under_its_block_ceiling():
    dataset, _ = _build()
    longest_unit = max(len(s) for s in _fillers(60) + [it["source"] for it in _items(60)])
    longest_answer = max(len(it["answer"]) for it in _items(60))
    for block in dataset["items"]:
        for it in block:
            assert it["target_gap"] <= it["ceiling"]
            # The cue goes in at the first turn boundary past the target gap, so
            # the realized gap overshoots by at most one passage plus one answer.
            assert it["gap"] <= it["ceiling"] + longest_unit + longest_answer + 4


def test_gaps_are_randomized_not_pinned_to_the_ceiling():
    dataset, _ = _build()
    targets = {it["target_gap"] for it in _all_items(dataset)}
    assert len(targets) > 5


def test_recall_credit_marks_exactly_the_entity_span():
    dataset, _ = _build()
    tok = _Tok()
    n = 0
    for ids, recall, block in zip(dataset["ids"], dataset["recall_masks"], dataset["items"]):
        marked = int(recall.sum())
        assert marked == sum(it["span_end"] - it["span_start"] for it in block)
        for it in block:
            assert recall[it["span_start"]:it["span_end"]].all()
            assert tok.decode(ids[it["span_start"]:it["span_end"]]).strip() == it["entity"]
            n += 1
    assert n > 10


def test_recall_credit_never_lands_on_the_copied_span_in_the_cue_or_source():
    dataset, _ = _build()
    for recall, block in zip(dataset["recall_masks"], dataset["items"]):
        for it in block:
            assert not recall[it["source_start"]:it["source_end"]].any()
            assert not recall[it["cue_start"]:it["cue_end"]].any()


def test_cue_is_not_always_the_last_thing_before_the_answer():
    dataset, _ = _build()
    items = _all_items(dataset)
    buried = [it for it in items if it["cue_end"] < it["answer_start"]]
    assert 0 < len(buried) < len(items)


def test_turns_strictly_alternate_user_assistant():
    dataset, _ = _build()
    for ids in dataset["ids"]:
        marks = [int(ids[i]) for i in ((ids == USER_ID) | (ids == ASST_ID)).nonzero().flatten()]
        assert marks[0] == USER_ID
        assert marks == [USER_ID if i % 2 == 0 else ASST_ID for i in range(len(marks))]


def test_source_always_precedes_its_cue_which_precedes_its_answer():
    for it in _all_items(_build()[0]):
        assert it["source_end"] <= it["cue_start"] < it["cue_end"] <= it["answer_start"] < it["span_start"]


def test_blocks_respect_the_token_budget_bounds():
    dataset, _ = _build(max_block_tokens=1200)
    for ids in dataset["ids"]:
        assert len(ids) <= 1200 + 2000  # budget bounds the last turn's start, not its length


def test_masks_train_every_token_including_carrier():
    dataset, _ = _build()
    for ids, mask in zip(dataset["ids"], dataset["masks"]):
        assert mask.shape == ids.shape and bool(mask.all())


def test_span_resolution_retries_with_the_leading_space_merged():
    dataset, stats = _build(encode=_word_encode)
    assert stats["n_span_unresolved"] == 0
    assert len(_all_items(dataset)) > 10


def test_items_whose_span_does_not_match_the_answer_are_dropped():
    bad = _items(4)
    for it in bad:
        it["span"] = (0, 3)  # "The" -- resolvable, but not the entity
    good = _items(4, entity=lambda i: f"Quovix{i}")
    dataset, stats = _build(items=bad + good, n_fillers=20)
    assert stats["n_span_unresolved"] == 4
    assert all(it["entity"].startswith("Quovix") for it in _all_items(dataset))


def test_validate_reports_zero_violations_on_clean_output():
    dataset, _ = _build()
    report = validate_blocks(dataset, _Tok(), user_id=USER_ID, asst_id=ASST_ID, n_samples=1)
    assert report["malformed_adjacency"] == 0
    assert report["span_text_mismatch"] == 0
    assert report["credit_visible_before_cue"] == 0
    assert report["stray_entity_occurrences"] == 0


def test_an_entity_visible_in_an_interference_passage_loses_its_credit():
    items = _items(20)
    leaked = [it["meta"]["entity"] for it in items[:10]]
    # max_pending 1 puts a filler between every pair of item sources, so each
    # block is guaranteed to carry the leaking passages.
    fillers = [f"An unrelated passage that happens to name {', '.join(leaked)} while discussing "
               f"sediment, weather and the drift of a river delta." for _ in range(40)]
    dataset, stats = _build(items=items, fillers=fillers, max_pending=1)
    assert stats["n_leaked_dropped"] > 0
    assert not (set(leaked) & {it["entity"] for it in _all_items(dataset)})
    assert any("unrelated passage" in _Tok().decode(ids) for ids in dataset["ids"])
    report = validate_blocks(dataset, _Tok(), user_id=USER_ID, asst_id=ASST_ID, n_samples=0)
    assert report["stray_entity_occurrences"] == 0


def test_validate_catches_a_credited_span_that_also_appears_between_source_and_cue():
    dataset, _ = _build()
    it = dataset["items"][-1][0]
    ids = dataset["ids"][-1]
    span = ids[it["span_start"]:it["span_end"]].clone()
    at = it["source_end"] + 1
    ids[at:at + len(span)] = span
    report = validate_blocks(dataset, _Tok(), user_id=USER_ID, asst_id=ASST_ID, n_samples=0)
    assert report["stray_entity_occurrences"] == 1
    assert report["credit_visible_before_cue"] == 1


def test_split_articles_is_a_disjoint_partition():
    titles = [f"art{i}" for i in range(200)]
    train, heldout = split_articles(titles, heldout_frac=0.1, seed=3)
    assert not (train & heldout)
    assert train | heldout == set(titles)
    assert 10 <= len(heldout) <= 30
    assert split_articles(titles, heldout_frac=0.1, seed=3) == (train, heldout)


def test_split_sentences_returns_covering_char_spans():
    text = "First one here. Second follows! Third ends?"
    spans = split_sentences(text)
    assert [text[a:b] for a, b in spans] == ["First one here.", "Second follows!", "Third ends?"]


class _Rng:
    """random.Random stand-in with deterministic choice/random for make_item."""

    def __init__(self, pick=0):
        self.pick = pick

    def choice(self, seq):
        return seq[self.pick % len(seq)]

    def shuffle(self, seq):
        pass

    def random(self):
        return 0.5


def _passage(text, article="art0"):
    return {"article": article, "text": text}


def _ents(text, *surfaces, label="LOC"):
    return [{"text": s, "label": label, "start": text.index(s), "end": text.index(s) + len(s)}
            for s in surfaces]


_THREE_ENTITIES = ("Berlin grew quickly in that period and trade expanded across the region. "
                   "The treaty was signed in Vienna by the delegates after long talks. "
                   "Later records describe the aftermath in Prague in considerable detail.")


def test_make_items_swaps_a_same_type_entity_and_blanks_only_the_cue():
    text = ("Berlin grew quickly in that period. The treaty was signed in Vienna by the delegates. "
            "Later records describe the aftermath in some detail.")
    items = make_items(_passage(text), _ents(text, "Vienna"), {"LOC": ["Marrakesh"]}, _Rng(),
                       min_sentence_words=6)
    assert len(items) == 1
    item = items[0]
    assert "Marrakesh" in item["source"] and "Vienna" not in item["source"]
    assert item["answer"][item["span"][0]:item["span"][1]] == "Marrakesh"
    assert "____" in item["cue"] and "Marrakesh" not in item["cue"]
    assert item["answer"] == item["cue"].replace("____", "Marrakesh")
    assert item["meta"]["entity_type"] == "LOC"


def test_make_items_never_substitutes_inside_a_longer_word():
    text = ("The Principality of Andorra kept its charter for centuries after the transfer. "
            "Later records describe the aftermath of that arrangement in some detail.")
    ents = [{"text": "Principal", "label": "LOC", "start": 4, "end": 13}]
    assert make_items(_passage(text), ents, {"LOC": ["Marrakesh"]}, _Rng(), min_sentence_words=6) == []


def test_make_items_declines_when_the_entity_repeats_in_its_sentence():
    text = "The delegates met in Vienna and Vienna hosted them again the following spring for talks."
    assert make_items(_passage(text), _ents(text, "Vienna"), {"LOC": ["Marrakesh"]}, _Rng(),
                      min_sentence_words=6) == []


def test_make_items_declines_without_a_same_type_replacement():
    text = "The delegates met in Vienna for a week of talks about the treaty and its terms."
    assert make_items(_passage(text), _ents(text, "Vienna"), {"LOC": ["Vienna"]}, _Rng(),
                      min_sentence_words=6) == []


def test_make_items_yields_a_group_sharing_one_substituted_source():
    text = _THREE_ENTITIES
    items = make_items(_passage(text), _ents(text, "Berlin", "Vienna", "Prague"),
                       {"LOC": ["Marrakesh", "Quito", "Osaka"]}, _Rng(),
                       min_sentence_words=6, max_items=3)
    assert len(items) == 3
    assert len({it["source"] for it in items}) == 1
    entities = {it["meta"]["entity"] for it in items}
    assert entities == {"Marrakesh", "Quito", "Osaka"}
    assert not ({"Berlin", "Vienna", "Prague"} & set(items[0]["source"].split()))
    assert len({it["answer"] for it in items}) == 3


def test_group_members_never_reveal_each_others_entity_in_a_cue_or_answer():
    text = _THREE_ENTITIES
    items = make_items(_passage(text), _ents(text, "Berlin", "Vienna", "Prague"),
                       {"LOC": ["Marrakesh", "Quito", "Osaka"]}, _Rng(),
                       min_sentence_words=6, max_items=3)
    for it in items:
        for other in items:
            if other is not it:
                assert other["meta"]["entity"] not in it["answer"]


def test_make_items_honours_max_items():
    text = _THREE_ENTITIES
    items = make_items(_passage(text), _ents(text, "Berlin", "Vienna", "Prague"),
                       {"LOC": ["Marrakesh", "Quito", "Osaka"]}, _Rng(), min_sentence_words=6)
    assert len(items) == 1


def test_a_group_emits_its_shared_source_passage_once():
    dataset, _ = _build(groups=_groups(40, 3), n_fillers=20, **_DENSE,
                        items_per_source_start=3, items_per_source_end=3)
    for ids, block in zip(dataset["ids"], dataset["items"]):
        by_source = {}
        for it in block:
            by_source.setdefault(it["source_start"], set()).add(it["source_end"])
        assert all(len(ends) == 1 for ends in by_source.values())
        assert len(block) > len(by_source)
        for g in range(40):
            marker = f"Article {g} covers"
            assert _Tok().decode(ids).count(marker) <= 1


def test_interference_density_is_higher_when_a_source_carries_more_items():
    dense, _ = _build(groups=_groups(60, 3), n_fillers=60, **_DENSE,
                      items_per_source_start=3, items_per_source_end=3)
    sparse, _ = _build(groups=_groups(60, 3), n_fillers=60, **_DENSE,
                       items_per_source_start=1, items_per_source_end=1)

    def tokens_per_item(dataset) -> float:
        n = sum(len(b) for b in dataset["items"])
        return sum(len(t) for t in dataset["ids"]) / n

    assert tokens_per_item(dense) < 0.75 * tokens_per_item(sparse)


def test_density_relaxes_as_the_curriculum_ceiling_grows():
    dataset, _ = _build(groups=_groups(120, 3), n_fillers=60, **_DENSE,
                        items_per_source_start=3, items_per_source_end=1)
    blocks = dataset["items"]
    assert len(blocks) > 3
    assert _per_source(blocks[0]) > 2.0
    assert _per_source(blocks[-1]) == 1.0


def test_items_per_source_never_drops_below_one():
    dataset, _ = _build(groups=_groups(40, 1), n_fillers=40,
                        items_per_source_start=0, items_per_source_end=0)
    assert sum(len(b) for b in dataset["items"]) > 10


def test_one_item_per_block_keeps_a_single_source_in_the_block():
    dataset, _ = _build(max_items_per_block=1)
    assert len(dataset["ids"]) > 3
    for ids, block in zip(dataset["ids"], dataset["items"]):
        assert len(block) == 1
        # No second item source may ride along as carrier either -- it would
        # answer the block's own question.
        assert _Tok().decode(ids).count("The regional capital is") == 3  # source, cue, answer


def test_needle_style_items_with_a_whole_answer_span_still_build():
    items = []
    for i in range(30):
        answer = f"bathroom{i}"
        items.append({
            "source": f"Mary went to the bathroom{i}. John moved to the garden. Sandra picked up the apple there.",
            "cue": f"Where is Mary ({i})?",
            "answer": answer,
            "span": (0, len(answer)),
            "meta": {"article": f"qa1-{i}", "entity": answer, "entity_type": "needle"},
        })
    dataset, stats = _build(items=items, n_fillers=40)
    report = validate_blocks(dataset, _Tok(), user_id=USER_ID, asst_id=ASST_ID, n_samples=0)
    assert report["span_text_mismatch"] == 0
    assert len(_all_items(dataset)) > 10


def test_block_args_parse(capsys):
    parser = argparse.ArgumentParser()
    add_block_args(parser)
    assert parser.parse_args([]).items_per_source_start >= 1
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])  # argparse rejects an unescaped % in a help string
