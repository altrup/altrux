"""Diagnostic: full 500-document wake with a forced EOS at the backstop; reports natural-EOS rate and throughput."""
import importlib, json, sys, time
from pathlib import Path
import torch
from experiments.dream_generation import sample_next
from progress import ts

N = int(sys.argv[1]); BACKSTOP = int(sys.argv[2])
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
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
model.eval(); state = None
natural = eoc_n = 0; gen_tokens = 0; prompt_tokens = 0; gen_time = 0.0; t0 = time.time()
lens = []
with torch.no_grad():
    for i, row in enumerate(rows):
        ev = tok(" " + row["evidence"], add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        ids = enc(U) + list(ev) + enc(f"{A} ")
        prompt_tokens += len(ids)
        logits, state = model(torch.tensor([ids], device=device), state)
        reply, stop = [], "backstop"; g0 = time.time()
        for _ in range(BACKSTOP):
            t = int(sample_next(logits[:, -1], 0.0).item())
            reply.append(t)
            logits, state = model(torch.tensor([[t]], device=device), state=state)
            if t == eos: stop = "eos"; break
            if t == eoc_id: stop = "eoc"; break
        if stop == "backstop":
            reply.append(eos); logits, state = model(torch.tensor([[eos]], device=device), state=state)
        torch.cuda.synchronize(); gen_time += time.time() - g0; gen_tokens += len(reply); lens.append(len(reply))
        natural += stop == "eos"; eoc_n += stop == "eoc"
        if i < 5 or stop != "backstop":
            print(f"[{ts()}] turn {i+1} stop={stop} len={len(reply)} REPLY={tok.decode(reply)[:300]!r}", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            print(f"[{ts()}] wake {i+1}/{len(rows)} natural_eos={natural} eoc={eoc_n} gen_tok/s={gen_tokens/max(gen_time,1e-9):.1f} turn/s={(i+1)/el:.2f} ETA {el/(i+1)*(len(rows)-i-1)/60:.1f}m vram={torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
print(f"[{ts()}] DONE natural_eos={natural}/{len(rows)} eoc={eoc_n} prompt_tokens={prompt_tokens} gen_tokens={gen_tokens} mean_len={sum(lens)/len(lens):.1f} total={time.time()-t0:.0f}s", flush=True)
