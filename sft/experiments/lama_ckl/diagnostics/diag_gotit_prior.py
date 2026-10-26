"""Diagnostic: the empty-state next-token prior around "Got it." across saved cycle adapters.

Usage: python -m experiments.lama_ckl.diagnostics.diag_gotit_prior <adapter dir> [<adapter dir> ...]
Each adapter dir holds trainable.pt (+ lora_config.json). Prints, per adapter and per
context, the probability of EOS, " Got", "[USER]" and the top-3 tokens.
"""

import importlib
import json
import sys
from pathlib import Path

import torch

from progress import ts

CONTEXTS = {
    "doc+asst": "{U} The Battle of Alesia was fought in 52 BC.\n{A}",
    "asst Got it.": "{A} Got it.",
    "asst Got it. x2": "{A} Got it. Got it.",
    "eoc": "{EOC}",
    "eoc+user": "{EOC}{U}",
    "eoc+user Got it.": "{EOC}{U} Got it.",
}


def main() -> None:
    device = torch.device("cuda")
    model_mod = importlib.import_module("models.mamba2_2_7b")
    hooks = importlib.import_module("models.mamba2_2_7b.train_hooks")
    from models.common import build_tokenizer

    from training.checkpoints import load_checkpoint

    adapters = [Path(a) for a in sys.argv[1:]]
    config = json.loads((adapters[0] / "lora_config.json").read_text())
    model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
    tok = build_tokenizer(model_mod)
    U, A, EOC = model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC
    eos = tok.eos_token_id
    got = tok(" Got", add_special_tokens=False)["input_ids"][0]
    user = tok.convert_tokens_to_ids(U)
    model.eval()
    for adapter in adapters:
        load_checkpoint(model, adapter)
        print(f"[{ts()}] === {adapter}", flush=True)
        for name, template in CONTEXTS.items():
            ids = tok(template.format(U=U, A=A, EOC=EOC), add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                logits, _ = model(torch.tensor([ids], device=device), None)
            p = torch.softmax(logits[0, -1].float(), -1)
            top = torch.topk(p, 3)
            tops = ", ".join(
                f"{tok.decode([int(i)])!r}:{v:.3f}" for v, i in zip(top.values, top.indices)
            )
            print(
                f"[{ts()}] {name:18s} eos {p[eos]:.3f}  ' Got' {p[got]:.3f}  [USER] {p[user]:.3f}  top3 {tops}",
                flush=True,
            )


if __name__ == "__main__":
    main()
