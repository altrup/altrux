"""CPU tests for the three-test solvability filter. The scorer is injected,
so these run against a stub that returns fixed log-probs -- no model, no GPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from filter_items import filter_dataset, interference_stream

USER_ID, ASST_ID = 1, 2


class _Tok:
    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)

    def encode(self, s: str, add_special_tokens=False) -> list[int]:
        return [ord(c) for c in s]


class _Scorer:
    """Returns per-(test, credited-text) log-probs; records every call."""

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[tuple[str, int, list]] = []

    def span_logprobs(self, ids, spans, label: str) -> list[float]:
        self.calls.append((label, len(ids), list(spans)))
        return [self.table[(label, _Tok().decode(ids[a:b]))] for a, b in spans]


def _block(entity: str = "Zorblat", stray: str = "") -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    src = f"The capital is {entity} in the north."
    cue = "The capital is ____ in the north."
    filler = "Unrelated sediment and weather text. " + stray
    text = f"{chr(USER_ID)} {src}\n{filler}\n{cue}{chr(ASST_ID)} {src}"
    ids = torch.tensor([ord(c) for c in text], dtype=torch.long)
    source_start = 2
    cue_start = text.index(cue)
    answer_start = cue_start + len(cue)
    span_start = answer_start + 2 + src.index(entity)
    recall = torch.zeros(len(ids), dtype=torch.bool)
    recall[span_start:span_start + len(entity)] = True
    item = {
        "gap": 40, "target_gap": 40, "ceiling": 448,
        "source_start": source_start, "source_end": source_start + len(src),
        "cue_start": cue_start, "cue_end": cue_start + len(cue),
        "answer_start": answer_start,
        "span_start": span_start, "span_end": span_start + len(entity),
        "credit_text": entity, "entity": entity, "entity_type": "LOC", "article": "art0",
    }
    return ids, recall, [item]


def _dataset(*blocks):
    return {
        "ids": [b[0] for b in blocks],
        "masks": [torch.ones(len(b[0]), dtype=torch.bool) for b in blocks],
        "recall_masks": [b[1] for b in blocks],
        "sleep_positions": [torch.zeros(0, dtype=torch.long) for _ in blocks],
        "items": [b[2] for b in blocks],
        "curriculum": {"ceilings": [448] * len(blocks)},
    }


def _args(**overrides):
    defaults = dict(a_min=-0.7, b_max=-1.5, c_max=-1.5, min_margin=1.5, samples=0)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run(scores, entity="Zorblat", stray="", **overrides):
    dataset = _dataset(_block(entity, stray))
    scorer = _Scorer({(k, entity): v for k, v in scores.items()})
    stats = filter_dataset(dataset, scorer, _Tok(), _args(**overrides))
    return dataset, stats, scorer


def test_well_posed_memory_required_item_keeps_its_credit():
    dataset, stats, _ = _run({"A": -0.2, "B": -3.0, "C": -2.6})
    assert stats["verdicts"]["pass"] == 1
    assert int(dataset["recall_masks"][0].sum()) == len("Zorblat")
    assert dataset["items"][0][0]["filter"]["verdict"] == "pass"


def test_an_item_the_backbone_cannot_answer_with_its_source_is_dropped():
    dataset, stats, _ = _run({"A": -2.0, "B": -3.0, "C": -2.6})
    assert stats["verdicts"]["fail_a"] == 1
    assert int(dataset["recall_masks"][0].sum()) == 0


def test_an_item_answerable_without_its_source_is_dropped():
    dataset, stats, _ = _run({"A": -0.2, "B": -0.4, "C": -2.6})
    assert stats["verdicts"]["fail_b"] == 1
    assert int(dataset["recall_masks"][0].sum()) == 0


def test_an_item_the_ssm_can_carry_in_stream_is_dropped():
    dataset, stats, _ = _run({"A": -0.2, "B": -3.0, "C": -0.3})
    assert stats["verdicts"]["fail_c"] == 1
    assert int(dataset["recall_masks"][0].sum()) == 0


def test_a_thin_margin_between_with_source_and_without_is_dropped():
    dataset, stats, _ = _run({"A": -0.5, "B": -1.6, "C": -3.0})
    assert stats["verdicts"]["fail_margin"] == 1
    assert int(dataset["recall_masks"][0].sum()) == 0


def test_a_discarded_item_keeps_its_tokens():
    before = _block()[0].clone()
    dataset, _, _ = _run({"A": -2.0, "B": -3.0, "C": -2.6})
    assert torch.equal(dataset["ids"][0], before)
    assert bool(dataset["masks"][0].all())


def test_an_entity_visible_in_the_interference_is_dropped_without_scoring():
    dataset, stats, scorer = _run({}, stray="Also mentions Zorblat for no reason. ")
    assert stats["verdicts"]["fail_leak"] == 1
    assert scorer.calls == []
    assert int(dataset["recall_masks"][0].sum()) == 0


def test_the_interference_stream_drops_every_item_source():
    ids, _, items = _block()
    stream, remap = interference_stream(ids, items)
    before_cue = _Tok().decode(stream[:remap(items[0]["cue_start"])])
    assert "The capital is Zorblat in the north." not in before_cue
    assert "Unrelated sediment" in before_cue
    assert len(stream) == len(ids) - (items[0]["source_end"] - items[0]["source_start"])
    assert _Tok().decode(stream[remap(items[0]["span_start"]):remap(items[0]["span_end"])]) == "Zorblat"


def test_scores_are_taken_from_three_distinct_contexts():
    _, _, scorer = _run({"A": -0.2, "B": -3.0, "C": -2.6})
    labels = {c[0]: c[1] for c in scorer.calls}
    assert set(labels) == {"A", "B", "C"}
    assert labels["A"] < labels["B"] < labels["C"]  # source+cue < interference < full block


def test_discard_composition_is_reported_per_test():
    blocks = [_block("Zorblat"), _block("Quovix")]
    dataset = _dataset(*blocks)
    scorer = _Scorer({("A", "Zorblat"): -0.2, ("B", "Zorblat"): -3.0, ("C", "Zorblat"): -2.6,
                      ("A", "Quovix"): -2.0, ("B", "Quovix"): -3.0, ("C", "Quovix"): -2.6})
    stats = filter_dataset(dataset, scorer, _Tok(), _args())
    assert stats["n_items"] == 2
    assert stats["verdicts"] == {"pass": 1, "fail_a": 1, "fail_b": 0, "fail_c": 0,
                                 "fail_margin": 0, "fail_leak": 0}
    assert stats["discard_rate"] == 0.5
