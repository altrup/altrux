import torch

from preparation import merge


def test_merge_preserves_artifact_concatenation(tmp_path, monkeypatch):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "merged.pt"
    torch.save({"ids": [torch.tensor([1])], "masks": [torch.tensor([True])]}, first)
    torch.save({"ids": [torch.tensor([2])], "masks": [torch.tensor([False])], "question_offsets": [3]}, second)

    monkeypatch.setattr(
        "sys.argv",
        ["merge_data.py", str(first), str(second), "--output", str(output)],
    )
    merge.main()

    merged = torch.load(output, map_location="cpu", weights_only=False)
    assert [item.tolist() for item in merged["ids"]] == [[1], [2]]
    assert [item.tolist() for item in merged["masks"]] == [[True], [False]]
    assert merged["question_offsets"] == [None, 3]
