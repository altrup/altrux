"""Diagnostic: cycle-1 wake + 300-dream set with the protocol's own functions, everything saved."""
import importlib, json, time
from collections import Counter
from pathlib import Path
import torch
from experiments.dream_generation import copy_state, generate_replay_dreams, state_to
from experiments.dream_types import dream_set_sha
from experiments.lama_ckl.protocol import dream_prompt_ids, lama_dream_diagnostics, run_conversational_wake
from progress import ts

OUT = Path("logs/diag-dreams-cycle1"); OUT.mkdir(parents=True, exist_ok=True)
split = Path("../.cache/lama_ckl/mamba2_2_7b_recap050")
adapter = Path("../models/mamba2_2_7b/checkpoints/recap050/epoch-2/step-800")
learned = [json.loads(l) for l in (split / "variant.jsonl").read_text().splitlines()]
device = torch.device("cuda")
torch.manual_seed(42)
model_mod = importlib.import_module("models.mamba2_2_7b")
hooks = importlib.import_module("models.mamba2_2_7b.train_hooks")
config = json.loads((adapter / "lora_config.json").read_text())
model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
from models.common import build_tokenizer
from training.checkpoints import load_checkpoint
load_checkpoint(model, adapter)
tok = build_tokenizer(model_mod)
U, A, EOC = model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC
documents = [str(r["evidence"]) for r in learned]
wake, open_state = run_conversational_wake(model, tok, documents, None, U, A, EOC, reply_tokens=64, evidence_tokens=512, device=device)
print(f"[{ts()}] wake invariants {wake['invariants']} transcript_sha256 {wake['transcript_sha256']}", flush=True)
(OUT / "wake.json").write_text(json.dumps(wake))
torch.save(state_to(copy_state(open_state), torch.device("cpu")), OUT / "open_state.pt")
print(f"[{ts()}] saved wake.json and open_state.pt", flush=True)
dreams, topology = generate_replay_dreams(
    model, open_state, dream_prompt_ids(tok, U, A, EOC, device), count=300, batch_size=30, seed=4201,
    n_tokens=512, temperature=0.7, decode_token=lambda t: tok.decode([t]),
    stop_id=int(tok.convert_tokens_to_ids(EOC)), turn_id=int(tok.eos_token_id))
rows = [{"index": i, "sha256": d.dream_sha, "stop_reason": d.stop_reason, "tokens": len(d.dream_ids) - d.prefix_len,
         "prefix_tokens": d.prefix_len, "text": "".join(d.token_texts), "token_ids": d.dream_ids} for i, d in enumerate(dreams)]
with (OUT / "dreams.jsonl").open("w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
groups = Counter(r["sha256"] for r in rows)
dups = {sha: n for sha, n in groups.items() if n > 1}
print(f"[{ts()}] set_sha256 {dream_set_sha(dreams)} unique={len(groups)}/300 duplicate_groups={len(dups)} duplicated_dreams={sum(dups.values())}", flush=True)
for sha, n in sorted(dups.items(), key=lambda kv: -kv[1])[:10]:
    r = next(r for r in rows if r["sha256"] == sha)
    print(f"[{ts()}] DUP x{n} stop={r['stop_reason']} tokens={r['tokens']} text={r['text'][:400]!r}", flush=True)
print(f"[{ts()}] stop reasons {Counter(r['stop_reason'] for r in rows)} length quartiles "
      f"{sorted(r['tokens'] for r in rows)[::75]}", flush=True)
diag = lama_dream_diagnostics(dreams, learned, wake["transcript_token_ids"])
(OUT / "diagnostics.json").write_text(json.dumps(diag, indent=1, sort_keys=True))
print(f"[{ts()}] diagnostics {json.dumps(diag, sort_keys=True)}", flush=True)
for r in rows[:12]:
    print(f"[{ts()}] dream {r['index']} stop={r['stop_reason']} tokens={r['tokens']} text={r['text'][:600]!r}", flush=True)
print(f"[{ts()}] DONE", flush=True)
