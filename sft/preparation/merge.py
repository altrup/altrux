"""Experiment: shared

Concatenates preparation/conversations.py outputs into one .pt artifact.

Use this when sources must form one dataset rather than separate training slices.
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple preparation/conversations.py .pt outputs into one")
    parser.add_argument("inputs", nargs="+", help="Input .pt files to merge")
    parser.add_argument("--output", default="data/train.pt", help="Output .pt file")
    args = parser.parse_args()

    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_qoffs: list[int | None] = []
    for path in args.inputs:
        data = torch.load(path, map_location="cpu", weights_only=False)
        all_ids.extend(data["ids"])
        all_masks.extend(data["masks"])
        all_qoffs.extend(data.get("question_offsets") or [None] * len(data["ids"]))
        print(f"  {path}: +{len(data['ids'])} examples")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": all_ids, "masks": all_masks, "question_offsets": all_qoffs}, out)
    print(f"Merged {len(all_ids)} examples total into {out}")


if __name__ == "__main__":
    main()
