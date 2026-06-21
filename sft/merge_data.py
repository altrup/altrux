"""Concatenates multiple prepare_data.py outputs (each {ids: [...], masks: [...]})
into one .pt file. train.py only accepts a single --data path, so mixing
sources (e.g. LongAlign-10k + babilong) needs this merge step after each is
tokenized separately.
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple prepare_data.py .pt outputs into one")
    parser.add_argument("inputs", nargs="+", help="Input .pt files to merge")
    parser.add_argument("--output", default="data/train.pt", help="Output .pt file")
    args = parser.parse_args()

    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    for path in args.inputs:
        data = torch.load(path, map_location="cpu", weights_only=False)
        all_ids.extend(data["ids"])
        all_masks.extend(data["masks"])
        print(f"  {path}: +{len(data['ids'])} examples")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": all_ids, "masks": all_masks}, out)
    print(f"Merged {len(all_ids)} examples total into {out}")


if __name__ == "__main__":
    main()
