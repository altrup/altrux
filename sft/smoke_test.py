"""Fast real-model sanity check: loads the actual model (paying the real
load/quantize cost, same as `make preflight`) but runs the gradient-wiring
check on a short synthetic sequence instead of a real dataset example.

`make preflight` picks the first valid example in the real dataset, which
for mamba2_780m_memory can be tens of thousands of tokens -- at this model's
--chunk-len (2, kept deliberately conservative pending re-measurement on
this backbone -- see models/mamba2_780m_memory/train_hooks.py), that's
thousands of slow chunks before the check tells you anything. This script
exists for the case where you just want "does the wiring still work" fast,
without waiting on dataset example length -- it does NOT replace `make
preflight` for confirming the real dataset is actually usable end-to-end.
"""

import argparse

import torch

import train


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Fast real-model gradient-wiring smoke test for {train.MODEL_NAME}")
    parser.add_argument("--length", type=int, default=10, help="Synthetic sequence length in tokens")
    parser.add_argument("--chunk-len", type=int, default=None, help="Defaults to the model's own DEFAULT_CHUNK_LEN")
    parser.add_argument("--eos-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {train.MODEL_ID} ...")
    model, trainable_params = train.hooks.setup_training(device, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Small ids are safe for any of this project's tokenizers (vocab sizes
    # are all in the tens of thousands) -- no need to know the real vocab size.
    ids = torch.randint(0, 100, (args.length,))
    mask = torch.ones(args.length, dtype=torch.bool)

    train.preflight(
        train.hooks, model, trainable_params, [ids], [mask], device,
        max_len=float("inf"), eos_weight=args.eos_weight, chunk_len=args.chunk_len,
    )


if __name__ == "__main__":
    main()
