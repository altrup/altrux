# Experiment notes — 2026-10-11 13:44:19 UTC: LAMA-CKL second box session (seed-42 grid)

Branch `box/20261011-074114`. Box: NVIDIA GH200 480GB (97,871 MiB VRAM), 64 CPU,
525 GiB RAM, 3.9 TB disk. Governing note:
[DISCUSSION-20261010 debrief](../discussion/DISCUSSION-20261010-lama-ckl-first-box-debrief.md),
session order per its 2026-10-10 amendment: retrain, format acceptance, pin,
split, wake acceptance, smokes, Llama-2 calibration, cells. Soft cap 20 h wall
clock (ASSUMED VALUE in the debrief; session start 13:42 UTC → cap 09:42 UTC
2026-10-12). Model `mamba2_2_7b` on every command.

## Arrival state (13:42 UTC)

- Uploaded: only `.cache/wheels/` (aarch64 mamba_ssm, causal_conv1d). No
  checkpoint (`LAMBDA_RESUME_CHECKPOINT` empty, as the debrief §3.7.8 allows),
  no LAMA archive, no TAALM env, `.cache/lama_ckl/` empty,
  `warmstart.sha256` = `unset`. Everything is regenerated here.
- Venv: torch 2.11.0+cu128, CUDA visible, `import mamba_ssm` OK.
- `HF_TOKEN` is in `~/.bashrc` after the interactive guard, so no
  non-interactive shell sees it; every tmux job exports it explicitly with
  `export $(grep -o 'HF_TOKEN=hf_[A-Za-z0-9_]*' ~/.bashrc)`. With it:
  whoami = Altrup, `meta-llama/Llama-2-7b-hf` config download OK → the
  Llama-2 calibration can run this session (it could not on 2026-10-10).
- Uncommitted `sft/uv.lock` change from `make sync` at setup (known, debrief
  §6); left alone.

## 13:44 UTC — three jobs launched

`train` (verbatim, after the HF_TOKEN export):

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start; echo "EXIT=$?" >> "$(ls -t logs/warm-start-*.log | head -1)"

Log `sft/logs/warm-start-20261011-134419.log`. Fresh start, no prior checkpoint.

`work:lama-dl`: official `data.zip` (74,619,466 bytes) → `.cache/LAMA/data/`,
EXIT=0 at 13:44. Log `sft/logs/lama-dl-20261011-134419.log`.

`work:taalm-env`: script (copied below) reproducing the 2026-10-10 TAALM
environment with its three ARM64 deviations (bitsandbytes 0.42.0 from source
with upstream `include/` guards from 73d3e7b; triton removed; peft pinned to
8665e2b5). Log `sft/logs/taalm-env-20261011-134419.log`.

```bash
#!/bin/bash
# TAALM pinned environment on ARM64 with the three deviations recorded in
# notes/experiments/EXPERIMENT_NOTES-20261010-051749.md.
set -euo pipefail
cd ~/altrux
export $(grep -o 'HF_TOKEN=hf_[A-Za-z0-9_]*' ~/.bashrc)
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
log "clone TAALM"
[ -d .cache/TAALM/.git ] || git clone https://github.com/ybseo-ac/TAALM.git .cache/TAALM
git -C .cache/TAALM checkout --detach b12f344a9dbae555c239635b1c192c555bed001b
log "venv"
uv venv --python 3.10 .cache/TAALM/.venv
PY=.cache/TAALM/.venv/bin/python
log "pinned requirements (cu128)"
UV_TORCH_BACKEND=cu128 uv pip install --python $PY -r .cache/TAALM/requirements.txt
log "DEVIATION 1: bitsandbytes 0.42.0 from source for aarch64"
rm -rf .cache/bnb-src
git clone https://github.com/TimDettmers/bitsandbytes.git .cache/bnb-src
git -C .cache/bnb-src checkout --detach 0.42.0
git -C .cache/bnb-src checkout 73d3e7b -- include/
(cd .cache/bnb-src && CUDA_VERSION=128 make cuda12x)
(cd .cache/bnb-src && uv pip install --python ~/altrux/$PY --no-deps --reinstall .)
log "DEVIATION 2: remove triton"
uv pip uninstall --python $PY triton || true
log "DEVIATION 3: peft pinned by date"
uv pip install --python $PY --no-deps git+https://github.com/huggingface/peft.git@8665e2b5719faa4e4b91749ddec09442927b53e0
log "results dir"
sudo install -d -o "$(id -u)" -g "$(id -g)" /results/lamackl
log "verify imports"
export LD_LIBRARY_PATH=$HOME/altrux/.cache/TAALM/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
$PY -c "import torch, transformers, peft, trl, accelerate, bitsandbytes as b; print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, accelerate.__version__, b.__version__, b.COMPILED_WITH_CUDA)"
log "DONE"
```

## 13:45–13:52 UTC — warm start attempt 1: stopped on a loss spike, then explained

Packer (log `sft/logs/warm-start-20261011-134419.log`): 710 examples, 2476
conversations, 1766 boundaries (476 repeat: 354 adjacent, 122 gapped; 1290
fresh), 1,926,846 tokens; all five "must be 0" invariants 0; decoded samples
at a fresh, an adjacent, and a gapped boundary read as a clean
`<|endoftext|><|endofconversation|>[USER] ...` join. `make sanity-sample
ARGS="--data data/warm_start.pt"` (`work:sanity`, log
`sft/logs/sanity-sample-20261011-134419.log`) EXIT=0; random windows read as
ordinary conversations and Dolly rows. Corpus accepted.

Training (chunk 512, batch 6, trainable 21,323,264, VRAM free 94.0 GB at
start): loss 1.83 (step 1) → 1.28 (step 4) → 2.51 (8) → 5.14 (9) → 10.15
(14) → 12.26 (20) with gnorm up to 3984 during LR warm-up (lr ≤ 6e-5), then
a slow fall: 7.4 at step 74, 6.28 at step 136. I stopped it at 13:48 (step
136) as a suspected numerical fault: AdamW at lr ≤ 3e-5 with clip 1.0 cannot
move a rank-16 LoRA that far in four steps, and a loss near ln(50k) means a
near-uniform model.

Checks while stopped: `tests/experiments/dreams/test_mixer_fused.py` on the
GH200 (`work:gputest`, log `sft/logs/gputest-mixer-fused.log`): 5 passed in
91 s. Library versions: python 3.14.8, torch 2.11.0+cu128, mamba_ssm
2.3.2.post1, triton 3.6.0 (the 08-10 warm start ran torch 2.13.0+cu129).

Teammate message at 13:50 (altrup, not watching live): this spike is the
known shape of the 2.7B warm start. They placed the 08-10 run's every-10th
step lines at `/tmp/warm-start-20260810-steps.txt` (not in the repo). Those
read: step 1 loss 2.18, step 21 12.12 (gnorm 692), step 31 10.52 (gnorm
5440), step 86 7.54, step 201 2.62, step 261 1.41. My run at step 86 read
7.36: the same curve. The 08-10 note's "5.76 → 4.77 over the first 160
steps" was that session's mis-summary. Decision: treat it as the registered
training's known transient, finish the fused test (done, passed), delete the
partial `models/mamba2_2_7b/checkpoints/epoch-1`, rerun `make warm-start`
unchanged. Open for discussion, not a stop: a transient collapse to
near-uniform during warm-up and a relearn from there is a different object
from a gently adapted base, even when step 800 passes acceptance; a warm-up
schedule or lower peak LR is a protocol change and belongs to the debrief.

## 13:52 UTC — warm start attempt 2 (verbatim, `train`, after the HF_TOKEN export)

    cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start; echo "EXIT=$?" >> "$(ls -t logs/warm-start-*.log | head -1)"

## 14:01 UTC — warm start DONE and PINNED

Log `sft/logs/warm-start-20261011-135130.log`, EXIT=0, 800 steps / 2 epochs
in 9 min (13:52–14:01), ~1.5 step/s. Attempt 2 reproduced attempt 1 step for
step (packer and trainer deterministic). Every-100th step:

| step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|
| loss | 7.17 | 4.69 | 1.44 | 1.07 | 1.18 | 1.24 | 1.06 | 0.956 |
| gnorm | 1.8 | 17.6 | 0.27 | 0.31 | 0.31 | 0.26 | 0.22 | 0.36 |

Zero non-finite warnings. Checkpoint
`models/mamba2_2_7b/checkpoints/repeat030/epoch-2/step-800` (2.2 GB tree).
**Warm-start SHA-256 (trainable.pt): `245e2089aa51d923747030feb8d77f23e691803478269a8bc222b2591cbf70fe`**,
written to `sft/experiments/lama_ckl/warmstart.sha256`, commit `29de8e3`.
GPU utilization was not sampled during the 9-minute job (same call as the
08-10 note: not worth tuning).

## 14:03–14:09 UTC — split built (`train`, verbatim, after the HF_TOKEN export)

    cd ~/altrux/sft && make lama-ckl-split ARGS="--model-name mamba2_2_7b --lama-root ../.cache/LAMA/data"

Log `sft/logs/lama-ckl-split-20261011-140317.log`, EXIT=0, 6 min. Source
candidates 13,645 (sha256 `6d787847…ea775`); scoring ~62–80 item/s. Output
`.cache/lama_ckl/mamba2_2_7b_repeat030/`: `variant.jsonl` 500 rows sha256
`f5edfdd9a6a5077d7dd404a8d245869d9621be4ca68b0b97dba6cd616e01d98d`,
`invariant_descriptive.jsonl` 500 rows sha256
`eec005e5c3ce7fc05484f17d5cbaf3c52aad1941ed9cd22640136b4e5711854c`,
`manifest.json` 4,029 bytes. Both invariant dicts zero
(`duplicate_uuid`, `missing_subject_or_object`). Samples: to-learn "Gary
Cuozzo plays in quarterback position" with a Cardinals evidence paragraph;
retained "Gloria Estefan used to communicate in Spanish" with a songwriting
evidence paragraph. Both read as well-formed task/evidence rows.

Order note: the amendment puts the §3.6.1 format acceptance before the pin
and the split. The repo's `make acceptance-check` scores a dream cache from
the retired erasure harness (`experiments/dreams/cli.py --build-dream-cache`,
which needs that harness's wake manifest), so instead §3.6.1 is scored on the
§3.6.2 acceptance dreams with the same function (`diagnostics.acceptance.
acceptance`, token-level plain-`]` share < 35 %, zero non-ASCII), after the
split. No benchmark score is read before it. Script (scratchpad, run with
`MODEL_NAME=mamba2_2_7b PYTHONPATH=.. uv run --no-sync python format_acceptance.py <dreams.json>...`):

```python
"""DISCUSSION-20261010 sec 3.6.1 on the sec 3.6.2 acceptance dreams: score a
LAMA-CKL dreams.json with diagnostics.acceptance.acceptance (token-level
plain-']' share and mojibake over the free-running span)."""
import importlib
import json
import os
import sys
from dataclasses import dataclass

from models.common import build_tokenizer

from diagnostics.acceptance import MARKER_SHARE_MAX, PLAIN_BRACKET_ID, acceptance


@dataclass
class D:
    dream_ids: list
    token_texts: list
    prefix_len: int


model_mod = importlib.import_module(f"models.{os.environ['MODEL_NAME']}")
tok = build_tokenizer(model_mod)
for path in sys.argv[1:]:
    rows = json.load(open(path))["dreams"]
    dreams = [
        D(r["token_ids"], [tok.decode([i]) for i in r["token_ids"]], r["prefix_tokens"]) for r in rows
    ]
    res = acceptance(
        dreams,
        user_id=tok.convert_tokens_to_ids(model_mod.USER_OPEN),
        asst_id=tok.convert_tokens_to_ids(model_mod.ASST_OPEN),
    )
    print(path, json.dumps(res), f"(share must be < {MARKER_SHARE_MAX}, plain id {PLAIN_BRACKET_ID})")
    for r in rows[:3]:
        print("  stop:", r["stop_reason"], "|", r["text"][: 300].replace("\n", "\\n"))
    print("  PASSED" if res["passed"] else "  FAILED")
```

Pull receipt 14:09:46 UTC: `repeat030/epoch-2/step-800/trainable.pt`
42,727,051 bytes, `optimizer.pt` 85,509,515, the split's three files. Home.

## 14:13 UTC — wake acceptance, 10 documents (`train`, verbatim)

    cd ~/altrux/sft && make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm altrux --seed 42 --docs-per-wake 10 --dreams-per-cycle 10 --cycles 1 --output ../.cache/lama_ckl/acceptance/docs10"

Log `sft/logs/lama-ckl-run-20261011-141302.log`. Fresh start; warm start
245e2089, split manifest `.cache/lama_ckl/mamba2_2_7b_repeat030/manifest.json`.

### 14:14 UTC — docs10 acceptance attempt 1 FAILED on the wake invariant

Log `sft/logs/lama-ckl-run-20261011-141302.log`, EXIT=2:
`epoch 1 cycle 0 wake invariants failed: {'turns': 10, 'internal_eoc': 2}`.
Epoch-0 evaluation first (floor, from fresh state, 34 s): to-learn 0.0040,
not-to-forget 0.9815, total 0.9855. Wake 10 turns at 0.39 turn/s.
Artifacts kept at `.cache/lama_ckl/acceptance/docs10-failed-rendering/`
(`work/epoch-01.*/cycle-00/wake.json`).

Decoding every turn's `prompt_token_ids` and `assistant_token_ids`:

- Every prompt ended `...<document>[ASSISTANT] ` as tokens `[..., 50278,
  209]`: no newline between the user turn and the assistant marker, and a
  lone space token (209) after it. The warm-start corpus
  (`preparation/conversations.py` `format_conversation`) renders
  `[USER] <content>\n` then `[ASSISTANT] <content><|endoftext|>`, where the
  space merges into the first content token. The wake prompt was therefore
  off the trained distribution at exactly the position the reply starts.
- Every reply began with token 4118 `###` and then, in 8 of 10, the
  model's own `[ASSISTANT]` (50278): the model repaired the rendering
  itself. After that: 6 substantive replies (turns 0, 4, 5, 6, 7 are
  near-verbatim copies of the document, the repeat-030 behaviour), 2 empty
  (`###[ASSISTANT]<|endoftext|>`), 2 EOC (`###<|endofconversation|>`,
  turns 8 and 9). Forced closes 0/10.

The 2026-10-10 wake (no frame, same prompt assembly) had the same
missing newline, so its 445–458/500 forced closes were measured on an
off-distribution prompt too.

## PROTECTED EDIT — `sft/experiments/lama_ckl/protocol.py` (commit `4dee2aa`)

Traceback: `runner.py:218 RuntimeError: epoch 1 cycle 0 wake invariants
failed: {'turns': 10, 'internal_eoc': 2}` (above). Change, in
`run_conversational_wake`: the prompt is now
`user_open + " " + WAKE_FRAME + document + "\n" + asst_open` (tokens:
`_encode("\n")` = `[187]` appended after the evidence, `_encode(asst_open)`
without the trailing space), the corpus rendering up to the reply. The
docstring says so. Verified on the real tokenizer that `" <doc>\n"` encodes
identically joint and split at every seam tried (`.`, `)`, digits). Tests:
`test_conversational_wake_puts_one_frame_before_every_document` now expects
`...<doc>\n[ASSISTANT]`, and the new
`test_conversational_wake_prompt_is_the_corpus_rendering_up_to_the_reply`
asserts byte-identity with `format_conversation` up to the reply. Both
failed before the fix; `tests/experiments/lama_ckl/`: 50 passed after.
`evaluation.py`, `split.py`, `data.py` untouched. **Every result from here
on is PROVISIONAL until altrup accepts `4dee2aa` into `main`.** The debrief
§3.2 frame is written on two lines, so I read this as the registered
rendering, not a protocol change.

## 14:17 UTC — docs10 acceptance attempt 2 (`train`, verbatim)

    cd ~/altrux/sft && make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm altrux --seed 42 --docs-per-wake 10 --dreams-per-cycle 10 --cycles 1 --output ../.cache/lama_ckl/acceptance/docs10"

Log `sft/logs/lama-ckl-run-20261011-141715.log`. PROVISIONAL.
