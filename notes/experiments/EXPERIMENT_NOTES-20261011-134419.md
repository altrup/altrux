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

### 14:20 UTC — docs10 acceptance attempt 2 FAILED: EOC is the first reply token, 10/10

Log `sft/logs/lama-ckl-run-20261011-141715.log`, EXIT=2,
`wake invariants failed: {'turns': 10, 'internal_eoc': 10}`. Every prompt
now ends `[..., 187, 50278]` (`\n[ASSISTANT]`), and every reply is the single
token 50279 `<|endofconversation|>`. Forced closes 0. Artifacts under
`.cache/lama_ckl/acceptance/docs10/work/epoch-01.*/cycle-00/wake.json`.

Teammate (14:25, pasted): corpus count on the box: `[ASSISTANT]` occurs
5083 times, always preceded by newline, never followed by EOC; after EOS the
split is 2611 `[USER]` / 2472 EOC. The data never teaches `[ASSISTANT]`→EOC.
Also: the 2.7B wrapper imports the 780m one, so replies after token 0 run
the new fused single-token step (`ace1155`); its equivalence test
`models/tests/test_mamba2_780m_fused_step.py` is queued on this box.

Diagnostic (`sft/logs/diag-reply-20261011-142300.log`, scratch script
`diag_reply.py`, same loader as the runner): top-8 next tokens after the
prompt and the greedy continuation. The first token comes from the chunk
path, the one evaluation uses.

| prompt | top next tokens (p) | greedy |
|---|---|---|
| wake frame, corpus rendering, doc 1 | EOC .25, " end" .07, `[ASSISTANT]` .06, "end" .06, " Remember" .04 | `<\|endofconversation\|>` |
| wake frame, corpus rendering, doc 2 | " end" .16, EOC .13, " End" .10 | `end quote<\|endoftext\|>` |
| wake frame, OLD rendering (no `\n`, lone space), doc 1 | `###` .21, `].` .19, `####` .13 | `###[ASSISTANT] Hollway was traded …` |
| no frame, corpus rendering, doc 1 | EOC .22, " The" .11, " Holl" .08 | EOC |
| document + "What is this passage about?" | `[ASSISTANT]` .18 / .50 | `[ASSISTANT]` × 40 (degenerate) |
| `[USER] What is the capital of France?` | " Paris" .33 | ` Paris<\|endoftext\|>` |
| Dolly-style context + question | " Gust" .65 | ` Gustave Eiffel<\|endoftext\|>` |

Reading: the corrected rendering is the trained one (plain chat and Dolly
prompts answer correctly and tersely, as the corpus does). Under the
registered frame the warm start's honest greedy reply to a bare document is
to close the conversation or say "end quote": the frame gives it nothing to
answer, which is the debrief §2 diagnosis ("a document with no reason to
reply to it") surviving the frame. The 10-10 wake's long replies were the
old rendering's `###[ASSISTANT]` self-repair followed by document
continuation, not replies to the frame. Note also `[ASSISTANT]` itself
carries 5–50 % next-token mass after `\n[ASSISTANT]` in several prompts and
loops under the question frame: a marker-delta/LoRA artifact of this warm
start worth a look.

Decision: this is the debrief amendment §3 case ("the no-reply wake is not
an in-session fallback") reached through `internal_eoc` rather than forced
closes, and the pre-grid acceptance fails → STOP-AND-REPORT. The
5-document run is not run formally: the first document's prompt and greedy
reply are identical at any wake size, so it fails the same invariant at
turn 1. Before shutdown: the fused-step equivalence test and a 100-document
first-token census plus the §3.6.1 EOC-alone dream check, for the
discussion.

### 14:21 UTC — teammate direction: altrux arm stop-and-report only; baselines and calibration proceed

Teammate (pasted, read the diag log): agrees it is model behaviour (EOC /
"end quote" after a pasted passage), not a bug. The wake frame is altrup's
decision; write the altrux arm up as stop-and-report with the top-8 lines
and the `[ASSISTANT]`-repeat artifact (12 % after the plain chat question),
do NOT shut the box down: frozen, lora, mix-review never wake (debrief §3.2),
so run those three seed-42 cells now in order with `--max-hours` = remaining
budget, run the Llama-2 calibration in parallel on the TAALM env, mark all
PROVISIONAL (protected edit `4dee2aa`), run the fused-step test before the
lora cell, leave the altrux cell to altrup.

Done before this: `models/tests/test_mamba2_780m_fused_step.py` on the
GH200: 7 passed, 0 skipped (`sft/logs/gputest-fused-step-20261011-142500.log`).
Soft cap: 20 h from 13:42 UTC → 09:42 UTC 2026-10-12.

Parallel GPU use is the teammate's call for this session (the one-tmux rule
is otherwise one GPU job at a time): the upstream run goes in a second
`train` window named `upstream`; the Mamba cells stay in `train:0`.

Upstream smoke (verbatim, `train:upstream`, after the HF_TOKEN export and
`export LD_LIBRARY_PATH=$HOME/altrux/.cache/TAALM/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib`):

    cd ~/altrux/sft && PATH="$PWD/../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-smoke

### 14:22 UTC — 100-document census and EOC-alone dreams (`sft/logs/diag-wake100-20261011-143000.log`)

Registered frame, corpus rendering, greedy, 128 backstop, fresh state per
document, first 100 to-learn rows: **EOC first 87, content 11, `[ASSISTANT]`
first 2, EOS first 0**; of the 11 content replies 2 hit the backstop; reply
lengths 3, 4, 5, 26, 38, 44, 75, 78, 87, 128, 128. The content replies
decode as "end of quote ]"-style closers or document copies.

10 dreams at T 0.7, 256 tokens, from `<|endofconversation|>` alone from
fresh state: none opens a `[USER]` turn; every dream starts mid-sentence
(" some of the most iconic moments in music history…", " the war.") and
5 of 10 fall into a `\n[ASSISTANT][ASSISTANT][ASSISTANT]…` loop; 3 end with
`<|endoftext|><|endofconversation|>` after one mid-sentence span.
`acceptance()`: plain 0, markers 643 (the loops), non-ASCII 2 (a curly
apostrophe), passed false. Caveat: in the corpus EOC always follows EOS, so
EOC from a fresh state is itself off-distribution; rechecked below with a
corpus-shaped turn + EOS + EOC prefix. The `[ASSISTANT]` loop after `\n`
is the same artifact as the question-frame probe and is a warm-start
quality finding regardless.

### 14:22 UTC — upstream smoke PASSED

`sft/logs/lama-ckl-upstream-smoke-20261011-142131.log`, EXIT=0: 64 rows,
one optimizer update, loss 2.216, eval ch ppl 6.20 acc 0.000 / unch ppl
0.383 acc 1.000 (the one-update smoke, numbers not meaningful).

## 14:28 UTC — launched: upstream run, frozen cell, dream recheck

`train:upstream` (verbatim, same exports):

    cd ~/altrux/sft && PATH="$PWD/../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-run

`train:0` (verbatim, after the HF_TOKEN export), fresh start, warm start
245e2089, split `mamba2_2_7b_repeat030`, cap remaining 19.2 h → `--max-hours 19`:

    cd ~/altrux/sft && make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm frozen --seed 42 --docs-per-wake 10 --dreams-per-cycle 10 --dream-batch-size 8 --eval-batch-size 16 --max-hours 19"

then `diag_dreams2.py` (scratch; prefix `[USER] What is the capital of
France?\n[ASSISTANT] Paris<|endoftext|>` + EOC, 10 dreams T 0.7, 256
tokens), log `sft/logs/diag-dreams2-20261011-143000.log`. PROVISIONAL (all).

### 14:29 UTC — frozen cell DONE (PROVISIONAL)

`sft/logs/lama-ckl-run-20261011-142839.log`, EXIT=0, 25 s. Output
`.cache/lama_ckl/runs/frozen-seed-42/`. Epoch 0: to-learn 0.0040,
not-to-forget 0.9835, total 0.9875 (eval batch 16; the acceptance runs at
their batch read 0.9815, the 1 % wobble the debrief §1.5 records).

### 14:29 UTC — in-distribution dream recheck (`sft/logs/diag-dreams2-20261011-143000.log`)

Prefix `[USER] What is the capital of France?\n[ASSISTANT] Paris<|endoftext|>`
then `<|endofconversation|>`, 10 dreams at T 0.7: **9 of 10 emit
`<|endofconversation|>` as the very next token** (an empty dream, stopped
on the second EOC); the tenth: `' capital of France<|endoftext|>[ASSISTANT]
This application was created by WizIQ.<|endoftext|><|endofconversation|>'`.
`acceptance()` passes vacuously (markers 1, plain 0, non-ASCII 0).

Cause, from the corpus mask (`data/warm_start.pt`, target-position mask as
`training/loop.py` applies it): the token after EOC is a training target
**0 of 1766** times; `[USER]` is a target **0 of 4377** times; EOC itself
is a target 2476 of 2476 times; after EOS, the 2472 EOC targets are trained
and the 2611 `[USER]` targets are not. The warm start teaches "EOS then
EOC" and never "EOC then `[USER] …`". The 08-10 recap050 warm start could
open a `[USER]` conversation after EOC only because the train mask was not
applied then (fixed in `97ecb30`, 2026-09-24, "conversation data trained on
user turns"). So the registered dream prompt (open state + EOC, the model
writes the conversation) has no supervision in the repeat030 warm start:
an altrux cycle would produce ten empty dreams even if the wake replied.
For the debrief: either the warm start trains the boundary-opening
`[USER]` turn (a masked-in boundary token, or user turns trained as the
recap050 run did), or the dream prompt changes; both are protocol
decisions. The `[ASSISTANT]`-repeat artifact (12 % after the plain
question, loops under the question frame and after `\n` in free text) is a
second warm-start finding.

## 14:33 UTC — lora cell launched (`train:0`, verbatim, after the HF_TOKEN export)

Fresh start, warm start 245e2089, split `mamba2_2_7b_repeat030`, cap
remaining 19.1 h → `--max-hours 19`:

    cd ~/altrux/sft && make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm lora --seed 42 --docs-per-wake 10 --dreams-per-cycle 10 --dream-batch-size 8 --eval-batch-size 16 --max-hours 19"

Log `sft/logs/lama-ckl-run-20261011-143328.log`. PROVISIONAL. Running
beside the upstream Llama-2 run (`train:upstream`, log
`sft/logs/lama-ckl-upstream-run-*.log`; at 14:33 ~3 it/s, GPU 68 GB used,
96 % util with both jobs).

### 14:53 UTC — Llama-2 calibration: MISS, recorded (gate did not pass)

Upstream run `train:upstream`, log `sft/logs/lama-ckl-upstream-run-20261011-1429*.log`,
EXIT=0, 14:29–14:51 (22 min, 30 epochs, ~2.5–3 it/s, single-GH200
batch-equivalent adapter: microbatch 8, accumulation 8). Summary
(`sft/logs/upstream-summarize-20261011-145500.log`, EXIT=2, pickle archived
first at `.cache/lama_ckl/upstream/finetune_qlora.pkl`, 47,802 bytes):
peak to-learn **0.105 at epoch 29**, paired not-to-forget **0.897**, total
1.002, `passes_gate` false. Against the frozen gate: peak 0.115 ± 0.02 →
inside; first peak epoch 16 ± 2 → outside (29); paired retention 0.8174 ±
0.02 → outside (higher). Curve samples: epoch=1 to_learn=0.004000 not_to_forget=0.985333;epoch=5 to_learn=0.008000 not_to_forget=0.981667;epoch=10 to_learn=0.014000 not_to_forget=0.979333;epoch=15 to_learn=0.038000 not_to_forget=0.966333;epoch=16 to_learn=0.040000 not_to_forget=0.966333;epoch=17 to_learn=0.048000 not_to_forget=0.965333;epoch=18 to_learn=0.057000 not_to_forget=0.961333;
The ARM64 environment deviations (bitsandbytes source build, triton
removed, peft pinned by date) and the one-GPU accumulation are the
candidate causes; for the debrief. Not a blocker (debrief §3.5).

### 14:55 UTC — lora cell curve so far (PROVISIONAL)

~110 s per epoch (training 500 documents + two 500-row evaluations),
30 epochs ≈ 55 min, ETA ~15:30. Live curve (`epoch N:` lines; the
`completed epoch` line prints the running checkpoint summary, i.e. the
peak-acquisition epoch and its paired retention, not the current epoch):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|---|---|
| to-learn | 0.017 | 0.021 | 0.046 | 0.058 | 0.056 | 0.058 | 0.037 |
| not-to-forget | 0.876 | 0.843 | 0.761 | 0.605 | 0.494 | 0.581 | 0.485 |

Loss on the documents falls to ~0.15 by epoch 10 (memorized) while the
object-token acquisition stays under 0.06 and retention halves: the
registered method's behaviour, recorded as data.
