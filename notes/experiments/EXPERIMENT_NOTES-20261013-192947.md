# Experiment notes — 2026-10-13 19:29:47 UTC — LAMA-CKL fourth box (ultra20k-v4)

Branch `box/20261013-185224`. GH200 480GB (97.9 GB VRAM), 64 CPU, 525 GB RAM.
Governing note: DISCUSSION-20261013 (third box debrief). Session soft cap 20 h
from session start (19:29 UTC, cap at 2026-10-14 15:29 UTC).

## State at start

- Nothing prepared: no `sft/data`, no `.cache/LAMA`, no HF cache, no checkpoints.
  Uploaded: `.cache/lama_ckl/upstream/finetune_qlora.pkl` (Llama-2 calibration, reused), `.cache/wheels`.
- `warmstart.sha256` reads `unset` (correct: split guard active until the pin).
- `sft/uv.lock` shows a 198-line local diff left by setup (not mine; left uncommitted).
- torch 2.11.0+cu128, CUDA visible.

## Session log

- 19:29 step 1, warm start, in `train`:
  ```
  cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start
  ```
  (target runs preparation.dolly, preparation.conversations `--hf-dataset HuggingFaceH4/ultrachat_200k --max-examples 20000 --input data/dolly_context.jsonl --max-len 8192 --pack --repeat-rate 0.3 --train-user-turns --output data/warm_start.pt`, then training.cli `--lr 1e-4 --chunk-len 512 --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --data data/warm_start.pt --epochs 1 --ckpt-every-tokens 2600000 --keep-ckpts 11`, then finalize into `checkpoints/ultra20k-v4/`.)
- 19:29 LAMA archive download in `work:lama` (wget dl.fbaipublicfiles.com/LAMA/data.zip, unzip to `.cache/LAMA/data`).
- 19:30 LAMA archive unpacked at `.cache/LAMA/data` (ConceptNet, Google_RE, Squad, TREx, relations.jsonl; 367 MB).
- 19:31 data prep done. dolly: records per shape document 1728 / plain 1771 / multi 366 (918 rows), document turns 2646 (410 prefaced). conversations: 8144 examples, 28469 conversations, 27.37M tokens, 20325 boundaries (4610 repeat: 3011 adjacent, 1599 gapped; 15715 fresh). Every must-be-zero invariant 0 (examples not ending at boundary, boundaries != conversations, boundaries not opening a turn, repeats without source, sources repeated twice, user turns unanswered, adjacent same-role, split replays not opening on user). 6 conversations dropped by --max-len. First occurrences: document 1728, multi 366, plain 1771, ultrachat 20000. Repeats: verbatim 253, question 230, recall 249, multi-turn 221, summary 30, ultrachat (k,j) 3700-ish spread over k=2..12.
- DATA SANITY GATE: read every decoded sample in the generator log (one per shape, variant and boundary kind) plus `make sanity-sample ARGS="--data data/warm_start.pt"` (`work:sanity`, two random windows). Verdict: OK. Packs are conversations and Dolly context rows; each repeat is its named variant (question-only drops the context, recall request returns the original text, summary variant replaces the context with the human summary, UltraChat split replays the last j turns verbatim then continues); boundaries open on `[USER]`; acknowledgements are varied short phrases ("Sure, I've had a look.", "Okay, read and understood.", "Okay, I've got the passage. <summary>"). Haiku reviewer flagged nothing structural; its flags (question variant repeats the answer verbatim, plain row's response restates its context, "0 recall-credited" in the inspection header) are the registered renderings and an unrelated legacy metric. One observation: in the summary variant of a summarization row the user turn is the summary and the acknowledgement is the summary again (30 rows); registered rendering, noted only.
- 19:32 training started: 8144 examples, 27.37M tokens, chunk_len 512, batch 6, ckpt every 2.6M tokens (expect 10 interval + final). Step 102 at 19:34: loss 7.53, gnorm 3.2. GPU 63-80 % util, 27 GB VRAM.
- 19:38 GPU health, warm start: 58-61 % utilization, 27 GB of 98 GB VRAM, ~2 step/s, ~1.6 ex/s, ETA ~85 min (same as the third box). Latency-bound at batch 6 / chunk 512. Left as is: the batch size is part of the fixed warm-start recipe (Makefile `WARM_START_TRAIN`), and raising it changes the effective optimizer step. **For discussion:** a larger batch (the VRAM allows roughly 3x) with the LR revisited would cut the warm start to well under an hour; decide before the next warm start, not mid-run. (Teammate asked for this note.)

## Warm-start probe curve (DISCUSSION-20261013 §3.2; gate at the final checkpoint only)

Probe command per checkpoint (run by a loop in `work:probe`, one at a time, in step order):
`make warm-start-probe ARGS="--init-adapter ../models/mamba2_2_7b/checkpoints/epoch-1/step-<N> --model-name mamba2_2_7b"` (defaults: 50 held-out Dolly contexts, `--docs-per-wake 5`, 10 dreams in one batch, seed 0).

| step | eoc_first /50 | role-marker closes /50 | forced /50 | top opener (count) | reply tok mean | user_opener /10 | asst loops | empty dreams | verbatim-copy dreams | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 910 | 11 | 27 | 0 | `<empty>` (38) | 11.5 | 5 | 1 | 0 | 7 | probe 19:40-19:42 (2 min beside training). 38 of 50 replies are empty: 27 emit a role marker as the first token (closed, EOS fed), 11 EOC first; 5 open with a stray `]`. Dreams: 7 of 10 are verbatim copies of a wake document under the frame (`[USER] Remember this for later: Nirvana was…`), several open `]<|endoftext|>`. Early checkpoint (11 % of the epoch); the third box's step-907 was 96 % forced with copies, so the new recipe's early habit is "say nothing" instead of "copy". |
| 1820 | 11 | 10 | 3 | `<empty>` (17) | 20.2 | 9 | 0 | 0 | 0 | probe 19:47-19:49. Content replies 33/50; replies are short answers or restatements about the document (" Sawgrass Country Club", " The five whys method is a method of problem-solving…"), none are copies. EOC-first stays at 11 (a whole 5-document wake of the ten returned EOC on every turn). Dreams: 9 open `[USER]` with UltraChat-style tasks, 0 verbatim copies, one `]<|endoftext|>`. |
| 2732 | 4 | 6 | 2 | `<empty>` (7) | 19.3 | 4 | 0 | 1 | 0 | probe 19:53-19:55. Content replies 43/50; openers spread (thanks 7, the 6, i 4, alright 3). Dreams: only 4 open `[USER]`, 1 empty, 0 copies. |
| 3644 | 1 | 7 | 0 | `the` (11) | 20.0 | 10 | 0 | 0 | 0 | probe 20:00-20:02. Content replies 47/50; EOC-first down to 1. Role-marker closes (7) are now replies that write a user-style instruction and then emit a role marker (" Extract a passage about vadmiral alexander colville", " Which materials are typically used in Montessori classrooms?") — the "document is followed by a question" prior of the third box surviving in a weaker form, closed by the marker rule. Other replies restate the document (" The noble gases are the atoms that make up the ine…"). Dreams 10/10 open `[USER]`, UltraChat-style tasks, 0 copies. |
| 4555 | 1 | 2 | 0 | `the` (8) | 14.8 | 10 | 0 | 0 | 0 | probe 20:06-20:08. First checkpoint inside every gate but EOC-first (1). Openers spread (the 8, okay 6, thanks 6, kenneth 4). Dreams 10/10 `[USER]`, 0 copies. |

Persistent per-document failures (steps 1820 to 4555): the EOC-first reply is always the same held-out context (the 2023 Covenant School shooting document), and the comma-list context ("Forgiven,X-Men,Generation X,X-Corps,…") always closes on a role marker with an empty reply. If the final checkpoint's only EOC-first is that one document, the gate as written (EOC-first 0) still fails; recorded here before the pin so the call is not made after the fact.
| 5467 | 1 | 3 | 4 | `the` (13) | 33.8 | 7 | 0 | 1 | 0 | probe 20:12-20:15. Mean reply length up (8 replies over 65 tokens, 4 forced); same EOC-first document. Dreams 7/10 `[USER]`, 1 empty, 0 copies. |
| 6377 | 1 | 1 | 5 | `the` (8) | 30.8 | 8 | 0 | 1 | 0 | probe 20:19-20:22. Content replies 49/50; the comma-list document now gets a content reply. Same EOC-first document. Dreams 8/10 `[USER]`, 1 empty, 0 copies. |

**Gate amendment from the teammate (20:27 UTC, in session):** "if it's just like 1 of 50 i think that's fine too … we can continue doesn't have to be 100%, but we should still train to end." Applied: an EOC-first count of 1 of 50 on the final checkpoint (the persistent Covenant School document) passes the pin gate; the other three counts stay as registered (role markers ≤4, forced ≤9, no opener >25). Training runs to its end; the final checkpoint remains the only candidate.
| 7289 | 1 | 1 | 2 | `the` (15) | 26.8 | 7 | 0 | 1 | 0 | probe 20:25-20:28. Stable; same EOC-first document. Dreams 7/10 `[USER]`, 1 empty, 0 copies. |
