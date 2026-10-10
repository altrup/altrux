"""Diagnostic: the registered fallback instruction-turn dream prompt from the saved cycle-1 open state, plus subject coverage for both sets."""
import importlib, json
from collections import Counter
from pathlib import Path
import torch
from experiments.dream_generation import generate_replay_dreams, state_to
from experiments.dream_types import dream_set_sha
from experiments.lama_ckl.protocol import lama_dream_diagnostics
from progress import ts

OUT = Path("logs/diag-dreams-cycle1")
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
load_checkpoint(model, adapter); model.eval()
tok = build_tokenizer(model_mod)
U, A, EOC = model_mod.USER_OPEN, model_mod.ASST_OPEN, model_mod.EOC
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
wake = json.loads((OUT / "wake.json").read_text())
open_state = state_to(torch.load(OUT / "open_state.pt", map_location="cpu", weights_only=False), device)
INSTRUCTION = f"{EOC}{U} Dream about the preceding experience. Rehearse what matters without copying it verbatim.{A} "
seed_ids = torch.tensor([enc(INSTRUCTION)], dtype=torch.long, device=device)
print(f"[{ts()}] instruction prompt {INSTRUCTION!r} -> {seed_ids.shape[1]} tokens, decoded {tok.decode(seed_ids[0])!r}", flush=True)
dreams, _ = generate_replay_dreams(model, open_state, seed_ids, count=300, batch_size=30, seed=4201, n_tokens=512,
    temperature=0.7, decode_token=lambda t: tok.decode([t]), stop_id=int(tok.convert_tokens_to_ids(EOC)), turn_id=int(tok.eos_token_id))
rows = [{"index": i, "sha256": d.dream_sha, "stop_reason": d.stop_reason, "tokens": len(d.dream_ids) - d.prefix_len,
         "prefix_tokens": d.prefix_len, "text": "".join(d.token_texts), "token_ids": d.dream_ids} for i, d in enumerate(dreams)]
with (OUT / "dreams_instruction.jsonl").open("w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
groups = Counter(r["sha256"] for r in rows); dups = {s: n for s, n in groups.items() if n > 1}
print(f"[{ts()}] INSTRUCTION set_sha256 {dream_set_sha(dreams)} unique={len(groups)}/300 duplicate_groups={len(dups)} duplicated_dreams={sum(dups.values())} stops={Counter(r['stop_reason'] for r in rows)} length quartiles {sorted(r['tokens'] for r in rows)[::75]}", flush=True)
for sha, n in sorted(dups.items(), key=lambda kv: -kv[1])[:5]:
    r = next(r for r in rows if r["sha256"] == sha); print(f"[{ts()}] DUP x{n} text={r['text'][:300]!r}", flush=True)
diag = lama_dream_diagnostics(dreams, learned, wake["transcript_token_ids"])
(OUT / "diagnostics_instruction.json").write_text(json.dumps(diag, indent=1, sort_keys=True))
print(f"[{ts()}] INSTRUCTION diagnostics {json.dumps(diag, sort_keys=True)}", flush=True)
for r in rows[:10]:
    print(f"[{ts()}] idream {r['index']} stop={r['stop_reason']} tokens={r['tokens']} text={r['text'][:600]!r}", flush=True)

def coverage(path):
    texts = [json.loads(l)["text"] for l in Path(path).read_text().splitlines()]
    subj = [(r["subject"], r["object"]) for r in learned]
    dreams_with_subject = sum(any(s in t for s, _ in subj) for t in texts)
    subjects_covered = {s for s, _ in subj if any(s in t for t in texts)}
    pairs_covered = {s for s, o in subj if any(s in t and o in t for t in texts)}
    return {"dreams_with_any_subject": dreams_with_subject, "distinct_subjects": len(subjects_covered), "subject_and_object_in_some_dream": len(pairs_covered)}
for name in ("dreams.jsonl", "dreams_instruction.jsonl"):
    print(f"[{ts()}] COVERAGE {name} {json.dumps(coverage(OUT / name))}", flush=True)
print(f"[{ts()}] DONE", flush=True)
