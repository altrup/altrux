import importlib

import torch


def test_top_level_merge_wrapper_reexports_main():
    implementation = importlib.import_module("preparation.merge")
    legacy = importlib.import_module("merge_data")

    assert legacy.__all__ == ["main"]
    assert legacy.main is implementation.main


def test_merge_wrapper_preserves_artifact_concatenation(tmp_path, monkeypatch):
    legacy = importlib.import_module("merge_data")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "merged.pt"
    torch.save({"ids": [torch.tensor([1])], "masks": [torch.tensor([True])]}, first)
    torch.save({"ids": [torch.tensor([2])], "masks": [torch.tensor([False])], "question_offsets": [3]}, second)

    monkeypatch.setattr(
        "sys.argv",
        ["merge_data.py", str(first), str(second), "--output", str(output)],
    )
    legacy.main()

    merged = torch.load(output, map_location="cpu", weights_only=False)
    assert [item.tolist() for item in merged["ids"]] == [[1], [2]]
    assert [item.tolist() for item in merged["masks"]] == [[True], [False]]
    assert merged["question_offsets"] == [None, 3]
