import json
from pathlib import Path

import pytest
import torch

from experiments.lama_ckl.diagnostics.dream_census import (
    acquisition,
    document_trigrams,
    revisits,
    run,
)

EOC = "<|endofconversation|>"
FRAME = "Remember this for later:"


def _dream(generated: str, stop_reason: str) -> dict[str, object]:
    return {
        "text": EOC + generated,
        "token_ids": [0, 1],
        "prefix_tokens": 1,
        "stop_reason": stop_reason,
        "sha256": "x",
    }


def _cycle(
    root: Path,
    name: str,
    dreams: list[dict[str, object]],
    adapter: dict | None,
    documents: tuple[str, ...] = ("a wake document",),
) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "wake.json").write_text(
        json.dumps({"turns": [{"document": document} for document in documents]})
    )
    diagnostics = {"duplicate_dreams": 1, "mean_copy_fraction": 0.25, "subjects_mentioned": 3}
    (directory / "dreams.json").write_text(
        json.dumps({"dreams": dreams, "diagnostics": diagnostics})
    )
    if adapter is not None:
        torch.save(adapter, directory / "trainable.pt")
    return directory


def test_census_counts_dreams_and_measures_adapter_steps(tmp_path):
    warm = tmp_path / "warm.pt"
    torch.save({"b": torch.zeros(1), "a": torch.zeros(2)}, warm)
    run_dir = tmp_path / "epoch-01"
    run_dir.mkdir()
    dreams = [
        _dream(f"{FRAME} x [ASSISTANT] ok", "eoc"),
        _dream(f"[USER] {FRAME} y", "budget"),
        _dream("[USER] " + "z" * 80 + f" {FRAME} w", "budget"),
        _dream(f"[USER] hi [ASSISTANT] hello{EOC}", "eoc"),
    ]
    first = _cycle(
        run_dir, "cycle-00", dreams, {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])}
    )
    (first / "distil.json").write_text(json.dumps({"max_dream_loss": 2.5, "loss": 1.5}))
    adapter = {
        "a": torch.tensor([6.0, 0.0], dtype=torch.bfloat16),
        "b": torch.tensor([8.0], dtype=torch.bfloat16),
    }
    _cycle(run_dir, "cycle-01", dreams[:1], adapter)
    _cycle(run_dir, "cycle-02", dreams, None)

    rows = run(run_dir, warm)

    assert [row["cycle"] for row in rows] == [0, 1]
    first_row, second_row = rows
    assert first_row["frame_anywhere"] == 3
    assert first_row["frame_early"] == 2
    assert first_row["frame_strict"] == 1
    assert first_row["no_user"] == 1
    assert first_row["assistant_present"] == 2
    assert first_row["eoc_stop"] == 2
    assert first_row["duplicate_dreams"] == 1
    assert first_row["subjects_mentioned"] == 3
    assert first_row["max_dream_loss"] == 2.5
    assert first_row["step_norm"] == pytest.approx(5.0)
    assert first_row["dist_from_warm_start"] == pytest.approx(5.0)
    assert first_row["cos_lag1"] is None
    assert second_row["max_dream_loss"] is None and second_row["loss"] is None
    assert second_row["step_norm"] == pytest.approx(5.0)
    assert second_row["dist_from_warm_start"] == pytest.approx(10.0)
    assert second_row["cos_lag1"] == pytest.approx(1.0)
    assert second_row["cos_lag5"] is None

    saved = json.loads((run_dir / "census.json").read_text())
    assert saved["warm_start"] == str(warm)
    assert [row["cycle"] for row in saved["cycles"]] == [0, 1]


def test_revisit_needs_two_non_overlapping_trigrams_unique_to_the_document():
    documents = [
        "the museum of art in paris holds the water lilies by claude monet",
        "the museum of art in rome holds a bronze she wolf",
    ]
    paris, _ = document_trigrams(documents)
    assert not revisits("I visited the museum of art in paris today", paris)
    assert not revisits("the water lilies by claude", paris)
    assert revisits("Water lilies by Claude Monet, in Paris holds", paris)
    assert revisits("art in paris, then the water lilies by", paris)


def test_census_counts_revisits_short_turns_and_clip(tmp_path):
    warm = tmp_path / "warm.pt"
    torch.save({"a": torch.zeros(1)}, warm)
    run_dir = tmp_path / "epoch-01"
    run_dir.mkdir()
    documents = ("one two three four five six", "seven eight nine ten eleven twelve")
    dreams = [
        _dream(
            "[USER] four five six and one two three\n[ASSISTANT] Got it.<|endoftext|>"
            "[USER] more\n[ASSISTANT] Introduction:\nlong text follows here",
            "eoc",
        ),
        _dream("[USER] hi\n[ASSISTANT] Got it.", "eoc"),
        _dream("[USER] hi\n[ASSISTANT] Got it. Thanks for the long note you sent today", "eoc"),
    ]
    cycle = _cycle(run_dir, "cycle-00", dreams[:2], {"a": torch.ones(1)}, documents)
    (cycle / "distil.json").write_text(json.dumps({"clipped": True, "grad_norm": 3.5}))
    _cycle(run_dir, "cycle-01", dreams[2:], {"a": torch.ones(1)}, documents)

    first, second = run(run_dir, warm)

    assert first["revisits_per_turn"] == [1, 0]
    assert first["revisiting_dreams"] == 1
    assert first["short_turn_dreams"] == 2
    assert first["acknowledgement_dreams"] == 2
    assert (first["top_short_turn"], first["top_short_turn_dreams"]) == ("Got it.", 2)
    assert (first["clipped"], first["grad_norm"]) == (True, 3.5)
    assert second["short_turn_dreams"] == 0 and second["top_short_turn"] is None
    assert second["clipped"] is None


def test_cycles_sort_numerically_past_100(tmp_path):
    warm = tmp_path / "warm.pt"
    torch.save({"a": torch.zeros(1)}, warm)
    dreams = [_dream("[USER] hi", "eoc")]
    for name in ("cycle-99", "cycle-100", "cycle-101.tmp"):
        _cycle(tmp_path, name, dreams, {"a": torch.ones(1)})

    assert [row["cycle"] for row in run(tmp_path, warm)] == [99, 100]


def _result(path: Path, scores: list[float], cycle: int | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"to_learn_scores": scores}
    if cycle is not None:
        payload["cycle"] = cycle
    (path / "result.json").write_text(json.dumps(payload))


def test_acquisition_counts_acquired_kept_and_exclusions(tmp_path):
    run_dir = tmp_path / "dream-seed-42"
    run_dir.mkdir()
    settings = {"docs_per_wake": 1, "eval_every": 2, "cycles_per_epoch": 4}
    (run_dir / "run.json").write_text(json.dumps(settings))
    _result(run_dir / "epoch-00", [0.0, 0.0, 0.0, 0.0])
    work = run_dir / "work" / "epoch-01.abc"
    # rows: kept, lost, frozen flicker, scored before its wake cycle 3
    _result(work / "cycle-01", [1.0, 1.0, 0.0, 1.0], cycle=1)
    _result(work / "cycle-03", [1.0, 0.0, 1.0, 1.0], cycle=3)
    frozen = tmp_path / "frozen-seed-42-r1"
    _result(frozen / "epoch-00", [0.0, 0.0, 1.0, 0.0])

    counts = acquisition(work, [frozen])

    assert counts["acquired"] == [0, 1]
    assert counts["kept"] == [0]
    assert counts["excluded_frozen"] == [2]
    assert counts["excluded_before_wake"] == [3]
    assert len(counts["denominator"]) == 2

    (work / "cycle-01" / "result.json").unlink()
    with pytest.raises(SystemExit):
        acquisition(work, [frozen])
