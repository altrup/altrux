import json
from pathlib import Path

import pytest
import torch

from experiments.lama_ckl.diagnostics.dream_census import run

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


def _cycle(root: Path, name: str, dreams: list[dict[str, object]], adapter: dict | None) -> Path:
    directory = root / name
    directory.mkdir()
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
