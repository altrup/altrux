"""CPU tests for the three-test solvability filter. The scorer is injected,
so these run against a stub that returns fixed log-probs -- no model, no GPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


from preparation.filtering import (
    BackboneScorer,
    filter_dataset,
    interference_stream,
    length_batches,
    pad_rows,
    rescore_dataset,
    score_rows,
)

USER_ID, ASST_ID = 1, 2


class _Tok:
    def decode(self, ids) -> str:
        return "".join(chr(int(i)) for i in ids)

    def encode(self, s: str, add_special_tokens=False) -> list[int]:
        return [ord(c) for c in s]


class _Scorer:
    """Returns per-(test, credited-text) log-probs; records every scored row
    and the batch each arrived in."""

    def __init__(self, table: dict):
        self.table = table
        self.calls: list[tuple[str, int, list]] = []
        self.batches: list[tuple[str, int]] = []

    def span_logprobs(self, rows, label: str) -> list[list[float]]:
        self.batches.append((label, len(rows)))
        out = []
        for ids, spans in rows:
            self.calls.append((label, len(ids), list(spans)))
            out.append([self.table[(label, _Tok().decode(ids[a:b]))] for a, b in spans])
        return out


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
    defaults = dict(a_min=-0.7, b_max=-1.5, c_max=-1.5, min_margin=1.5, samples=0, score_batch=8)
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


def test_rescore_reverdicts_from_stored_scores_and_rebuilds_credit():
    dataset, _, _ = _run({"A": -1.8, "B": -9.0, "C": -6.5})
    item = dataset["items"][0][0]
    assert item["filter"]["verdict"] == "fail_a"
    assert not dataset["recall_masks"][0].any()

    stats = rescore_dataset(dataset, _args(a_min=-2.0, min_margin=4.0))
    assert item["filter"]["verdict"] == "pass"
    assert dataset["recall_masks"][0][item["span_start"]:item["span_end"]].all()
    assert stats["verdicts"]["pass"] == 1


def test_rescore_can_also_revoke_credit_and_never_scores():
    dataset, _, _ = _run({"A": -0.2, "B": -9.0, "C": -4.0})
    item = dataset["items"][0][0]
    assert item["filter"]["verdict"] == "pass"

    stats = rescore_dataset(dataset, _args(min_margin=20.0))
    assert item["filter"]["verdict"] == "fail_margin"
    assert not dataset["recall_masks"][0][item["span_start"]:item["span_end"]].any()
    assert stats["verdicts"]["pass"] == 0


def test_rescore_leaves_leak_verdicts_alone():
    dataset, _, _ = _run({"A": -0.2, "B": -9.0, "C": -4.0}, stray="Zorblat")
    item = dataset["items"][0][0]
    assert item["filter"]["verdict"] == "fail_leak"

    rescore_dataset(dataset, _args(a_min=-99.0, b_max=99.0, c_max=99.0, min_margin=-99.0))
    assert item["filter"]["verdict"] == "fail_leak"
    assert not dataset["recall_masks"][0][item["span_start"]:item["span_end"]].any()


def test_leak_only_keeps_unleaked_items_without_scoring():
    dataset = _dataset(_block("Zorblat"), _block("Quovix", stray="Quovix"))
    scorer = _Scorer({})
    stats = filter_dataset(dataset, scorer, _Tok(), _args(leak_only=True))
    assert scorer.calls == []
    assert stats["verdicts"] == {"pass": 1, "fail_a": 0, "fail_b": 0, "fail_c": 0,
                                 "fail_margin": 0, "fail_leak": 1}
    assert dataset["items"][0][0]["filter"]["verdict"] == "pass"
    assert dataset["recall_masks"][0].any()
    assert not dataset["recall_masks"][1].any()


def test_rescore_leak_only_reinstates_scored_failures():
    dataset, _, _ = _run({"A": -12.0, "B": -13.0, "C": -12.5})
    item = dataset["items"][0][0]
    assert item["filter"]["verdict"] == "fail_a"

    rescore_dataset(dataset, _args(leak_only=True))
    assert item["filter"]["verdict"] == "pass"
    assert dataset["recall_masks"][0][item["span_start"]:item["span_end"]].all()


def test_the_a_contexts_of_several_blocks_are_scored_in_one_batch():
    dataset = _dataset(_block("Zorblat"), _block("Quovix"))
    scorer = _Scorer({(t, e): v for t, v in {"A": -0.2, "B": -3.0, "C": -2.6}.items()
                      for e in ("Zorblat", "Quovix")})
    filter_dataset(dataset, scorer, _Tok(), _args(score_batch=8))
    assert dict(scorer.batches) == {"A": 2, "B": 2, "C": 2}


def test_score_batch_1_scores_every_row_on_its_own():
    dataset = _dataset(_block("Zorblat"), _block("Quovix"))
    scorer = _Scorer({(t, e): v for t, v in {"A": -0.2, "B": -3.0, "C": -2.6}.items()
                      for e in ("Zorblat", "Quovix")})
    filter_dataset(dataset, scorer, _Tok(), _args(score_batch=1))
    assert {n for _, n in scorer.batches} == {1}


def test_length_batches_caps_rows_and_covers_every_index_once():
    batches = length_batches([10, 10, 10, 10, 10], max_rows=2)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3, 4]


def test_length_batches_keeps_a_long_row_out_of_a_short_batch():
    batches = length_batches([10, 100, 11], max_rows=8, max_ratio=2.0)
    assert sorted(sorted(b) for b in batches) == [[0, 2], [1]]


def test_length_batches_handles_no_rows():
    assert length_batches([], max_rows=4) == []


def test_pad_rows_right_pads_to_the_batch_max_and_reports_true_lengths():
    padded, lens = pad_rows([(torch.tensor([1, 2, 3]), [(1, 3)]),
                             (torch.tensor([4, 5]), [(1, 2)])], "cpu")
    assert lens == [3, 2]
    assert padded.shape == (2, 3)
    assert padded[1].tolist() == [4, 5, 0]


def test_pad_rows_rejects_a_span_reaching_past_its_own_row():
    with pytest.raises(AssertionError):
        pad_rows([(torch.tensor([1, 2, 3]), [(1, 3)]), (torch.tensor([4, 5]), [(1, 3)])], "cpu")


def test_score_rows_returns_results_in_input_order_across_batches():
    rows = [(torch.tensor([7] * n), [(1, 2)]) for n in (40, 10, 41, 11)]

    class _ByLength:
        def span_logprobs(self, batch, label):
            return [[float(len(ids))] for ids, _ in batch]

    out, fed = score_rows(_ByLength(), rows, "A", max_rows=2)
    assert out == [[40.0], [10.0], [41.0], [11.0]]


def test_score_rows_counts_real_tokens_and_not_padding():
    rows = [(torch.tensor([7] * n), [(1, 2)]) for n in (40, 10, 41, 11)]

    class _Fixed:
        def span_logprobs(self, batch, label):
            return [[0.0] for _ in batch]

    _, fed = score_rows(_Fixed(), rows, "A", max_rows=4)
    assert fed == 102


class _FakeState:
    def __init__(self, pos: int, acc: torch.Tensor):
        self.pos, self.acc = pos, acc

    def detach(self) -> "_FakeState":
        return self


class _FakeLM(torch.nn.Module):
    """Causal stand-in for the backbone: each position's logits depend on the
    absolute position and on a running function of every token fed so far, so
    a batched read that lands one position off -- or inside another row's
    padding -- cannot agree with the row-by-row read."""

    V = 11

    def forward(self, input_ids, state=None):
        B, T = input_ids.shape
        pos = 0 if state is None else state.pos
        acc = torch.zeros(B) if state is None else state.acc
        outs = []
        for t in range(T):
            acc = acc * 1.1 + input_ids[:, t].float()
            outs.append(torch.sin(acc.view(B, 1) * 0.37 + (pos + t) * 0.11
                                  + torch.arange(self.V).view(1, self.V) * 0.53))
        return torch.stack(outs, dim=1), _FakeState(pos + T, acc)


def test_batched_span_scoring_reads_the_same_values_as_row_by_row_scoring():
    """Rows of different lengths, spans inside a chunk, spans crossing a chunk
    boundary, and spans starting exactly on one (the carried-logits path)."""
    rows = [(torch.tensor([3, 1, 4, 1, 5, 9]), [(1, 3), (3, 6)]),
            (torch.tensor([2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4]), [(4, 5), (7, 11)]),
            (torch.tensor([6, 6, 6, 5]), [(2, 4)])]
    scorer = BackboneScorer(_FakeLM(), "cpu", chunk_len=4)

    batched = scorer.span_logprobs(rows, "A")
    serial = [scorer.span_logprobs([r], "A")[0] for r in rows]
    assert [len(r) for r in batched] == [len(r) for r in serial]
    assert [v for r in batched for v in r] == pytest.approx([v for r in serial for v in r], abs=1e-9)


def test_a_span_scores_its_own_tokens_against_the_preceding_positions_logits():
    """Pins the teacher-forcing convention independently of the scorer's
    chunking: the logits at position t-1 are what score the token at t."""
    ids = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
    model = _FakeLM()
    logits, _ = model(ids.view(1, -1))
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    a, b = 2, 6
    expected = sum(float(lp[t - 1, ids[t]]) for t in range(a, b)) / (b - a)

    got = BackboneScorer(model, "cpu", chunk_len=4).span_logprobs([(ids, [(a, b)])], "A")[0][0]
    assert got == pytest.approx(expected, abs=1e-9)


def test_a_source_shared_by_two_items_is_cut_once_and_remapped_once():
    """prepare_cram emits one passage for a whole group of items, so the same
    (source_start, source_end) arrives once per member."""
    ids, _, items = _block()
    it = items[0]
    stream, remap = interference_stream(ids, [it, dict(it)])
    assert len(stream) == len(ids) - (it["source_end"] - it["source_start"])
    assert _Tok().decode(stream[remap(it["span_start"]):remap(it["span_end"])]) == "Zorblat"
    assert _Tok().decode(stream[remap(it["cue_start"]):remap(it["cue_end"])]) == \
        "The capital is ____ in the north."
