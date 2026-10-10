"""Diagnostic: greedy wake replies from the recap-0.5 warm start with a long backstop."""
import importlib, json, sys, time
from pathlib import Path
import torch
from experiments.dream_generation import sample_next
from progress import ts

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BACKSTOP = 256
split = Path("../.cache/lama_ckl/mamba2_2_7b_recap050")
adapter = Path("../models/mamba2_2_7b/checkpoints/recap050/epoch-2/step-800")
rows = [json.loads(l) for l in (split / "variant.jsonl").read_text().splitlines()][:N]
device = torch.device("cuda")
model_mod = importlib.import_module("models.mamba2_2_7b")
hooks = importlib.import_module("models.mamba2_2_7b.train_hooks")
config = json.loads((adapter / "lora_config.json").read_text())
model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
from models.common import build_tokenizer
from training.checkpoints import load_checkpoint
load_checkpoint(model, adapter)
tok = build_tokenizer(model_mod)
U, A, EOC = model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC
eos, eoc_id = tok.eos_token_id, tok.convert_tokens_to_ids(EOC)
print(f"[{ts()}] eos_id={eos} eoc_id={eoc_id} U={U!r} A={A!r}", flush=True)
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
model.eval()
state = None
with torch.no_grad():
    for i, row in enumerate(rows):
        ev = tok(" " + row["evidence"], add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        ids = enc(U) + list(ev) + enc(f"{A} ")
        logits, state = model(torch.tensor([ids], device=device), state)
        reply, stop = [], "backstop"
        for _ in range(BACKSTOP):
            t = int(sample_next(logits[:, -1], 0.0).item())
            reply.append(t)
            logits, state = model(torch.tensor([[t]], device=device), state=state)
            if t == eos: stop = "eos"; break
            if t == eoc_id: stop = "eoc"; break
        print(f"[{ts()}] turn {i+1} stop={stop} len={len(reply)} evidence_tokens={len(ev)}\n  DOC: {row['evidence'][:200]!r}\n  REPLY: {tok.decode(reply)!r}", flush=True)
