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
| 8200 | 1 | 3 | 3 | `thanks` (9) | 19.8 | 10 | 0 | 0 | 0 | probe 20:32-20:34. Stable; same EOC-first document. Dreams 10/10 `[USER]`, 0 copies. |

- 20:41:45 warm-start training done: 9594 steps, 8144/8144 examples, 19:32 to 20:41 (69 min training), last running loss 0.93. Finalize moved `epoch-1/` into `checkpoints/ultra20k-v4/`; `ultra20k-v4/final -> epoch-1/step-9594`. Interval checkpoints: 910, 1820, 2732, 3644, 4555, 5467, 6377, 7289, 8200, 9114 (ten) plus the final step 9594. Tree size 3.3G. sha256 of final/trainable.pt: `7ba7972b308c70c777d4f592e46a781b4b9dad22717cec247a6f10821a054ad9` (pinned only after the gate). Fetch requested (`.watchdog-fetch`) for the checkpoint tree.
| 9114 | 1 | 4 | 2 | `thanks` (6) | 22.2 | 10 | 0 | 0 | 0 | probe 20:39-20:41 (wrote probe.json 13 s before the finalize move). Same EOC-first document. |
| **9594 (final)** | **1** | **1** | **2** | **`the` (12)** | 20.9 | 10 | 0 | 0 | 0 | probe 20:42-20:44 on `ultra20k-v4/epoch-1/step-9594` (= `final`). Content replies 48/50. The one EOC-first is the Covenant School document again; the one role-marker close is an empty reply to the Call of Duty document; the comma-list document now gets a content reply (an invented plot summary). Openers: the 12, okay 8, thanks 7, got 3, i 3. Replies are acknowledgements or one-sentence restatements, mean 21 tokens, 3 over 65. Dreams 10/10 open `[USER]` with UltraChat-style tasks ("Write a horror story…", "Given the text: …", "Can you paraphrase…"), 2 close on EOC, 0 copies, 0 empty, 0 assistant loops. |

**PIN GATE: PASS** under the teammate's amendment (EOC-first 1 of 50 allowed; role markers 1 ≤ 4; forced 2 ≤ 9; top opener 12 ≤ 25). Pinned `sft/experiments/lama_ckl/warmstart.sha256` = `7ba7972b308c70c777d4f592e46a781b4b9dad22717cec247a6f10821a054ad9` (`ultra20k-v4/final -> epoch-1/step-9594`).

- 20:45 step 2, split, in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-split ARGS="--model-name mamba2_2_7b --lama-root ../.cache/LAMA/data"
  ```
- 20:51 split done (20:45-20:51, 6 min; final scoring line: `[20:51:16] scored 13645/13645  zeros=8223 ones=2542  81.9 item/s  ETA 0s`). Source candidates 13645 (sha256 6d787847…), warm start sha256 7ba7972b… read from the pin. Wrote `.cache/lama_ckl/mamba2_2_7b_ultra20k-v4/`: variant.jsonl rows=500 sha256 fc753410…, invariant_descriptive.jsonl rows=500 sha256 fde0d5fb…, manifest.json. Invariants: duplicate_uuid 0, missing_subject_or_object 0 (both sets). Samples read (task + evidence pairs are LAMA sentences with their Wikipedia evidence). Fetch requested.
- 20:54 step 3, frozen cell ×3, in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-frozen-x3 SEED=42 ARGS="--model-name mamba2_2_7b"
  ```
  (expands to `make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm frozen --seed 42 --output ../.cache/lama_ckl/runs/frozen-seed-42-r{1,2,3}"`). Started fresh; warm start sha256 7ba7972b…; split manifest `.cache/lama_ckl/mamba2_2_7b_ultra20k-v4/manifest.json`.
- 20:54-20:56 frozen ×3 done (each run ~20 s of scoring after the model load):

| run | to-learn | not-to-forget | total |
|---|---|---|---|
| frozen-seed-42-r1 | 0.001 | 0.987 | 0.988 |
| frozen-seed-42-r2 | 0.001 | 0.987 | 0.988 |
| frozen-seed-42-r3 | 0.001 | 0.984 | 0.985 |

  **Evaluation resolution (spread of three): to-learn 0.000, not-to-forget 0.003.** Floor: to-learn 0.001, not-to-forget 0.986 (mean). Frozen-x3 wrote `.cache/lama_ckl/runs/frozen-seed-42-r{1,2,3}`.
- 20:57 step 4, acceptance, in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --docs-per-wake 5 --dreams-per-cycle 10 --cycles 1 --output ../.cache/lama_ckl/acceptance/dream-5"
  ```
- 20:59-21:01 acceptance (seed 42, 5 documents, 10 dreams, 1 cycle) finished, EXIT=0. Output `.cache/lama_ckl/acceptance/dream-5/epoch-01/cycle-00/`. run.json: warmstart_sha256 7ba7972b…, split_manifest_sha256 f829ea48…, reply_temperature 0.7, wake_seed 4201000, dream_temperature 0.7, dream_tokens 512, reply_tokens 128, lr 1e-4, requested_dream_batch_size 10.
  - **wake.json**: `internal_eoc` 0, `role_marker_closes` 0/5, `forced_closes` 0/5. All five replies are identical: ` Got it.<|endoftext|>` (4 tokens; sampled at 0.7 with per-turn seeds 4201000000-4). Quoted reply 3: document "Prestolee has various shops in or just outside the village, including a greengrocer's, a chip shop (now a takeaway) and a sweet shop later becoming a butcher's before ending as a village community centre." → ` Got it.`
  - **dreams.json** diagnostics: unique_dreams 10, duplicate_dreams 0, empty 0 (min 59 tokens, mean 385, max 511), eoc_termination_rate 0.4, mean_copy_fraction 0.0, max_verbatim_run 5, **subjects_mentioned 0**, correct_bindings 0, misbindings 0. The ten dreams are UltraChat-style conversations unrelated to the wake (quoted dream 7: `[USER] The Nth Day is a 2005 Spanish romantic comedy film written and directed by Javier Fesser. The film was produced by Erik Gutiérrez, the film had its world premiere at the Venice Film Festival on 2 August 2005.\n[ASSISTANT] Okay, I've …` — the document-acknowledgement shape, but an invented document, not a wake subject). Distillation: one step, loss 0.0035.
  - **Cycle wall time** (log timestamps): initial evaluation 20:59:50-20:59:58 (8 s for 1000 items); wake 20:59:58-21:00:01 (3 s, 1.48 turn/s); dreams 21:00:01-21:00:40 (39 s, one launch of 10, 0.26 dream/s); distillation 21:00:40-21:01:01 (21 s); runner's own figure 63.1 s/cycle. **Dream-arm ETA at 10 dreams: 100 × (63 + 8) s + 8 s ≈ 2.0 h** (far under the third box's 145 s/cycle: no state carry, one dream launch, one distill step).
  - **Gate reading**: `internal_eoc` 0 ok, role-marker 0/5 ok, forced 0/5 ok, dreams non-empty ok, `subjects_mentioned` 0 — **miss on the last condition**. DISCUSSION-20261010 §3.6 item 4 (still assigned to the experimenter by the skill: "a raised dream count by its rule") says the dream count may go up (20 to 50) only if the 10 acceptance dreams show no relation to the wake documents at all, decided at acceptance and recorded before any cell. That is this case. Decision pending the seed-43 repeat below.
- 21:03 robustness check (diagnostic, not a protocol change), in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 43 --docs-per-wake 5 --dreams-per-cycle 10 --cycles 1 --output ../.cache/lama_ckl/acceptance/dream-5-seed43"
  ```
- 21:05-21:07 seed-43 repeat: `internal_eoc` 0, forced 0/5, role-marker 0/5; replies varied (` Got it, thanks for the context.`, ` The see is vacant.`, a restatement of the Prestolee document, ` Sure, I've had a look.`, ` Knut Ove Lystad was born on 31 January 1946.`); dreams: unique 10, empty 0, copy fraction 0.0, **subjects_mentioned 0** again; loss 0.0033; 63.1 s/cycle. So 0 of 20 dreams over two seeds mention any wake subject, and reading all twenty confirms it: every dream is an UltraChat-style conversation about something else (the only document-shaped dreams invent their own document).

**DECISION (experimenter, before any cell): raise the dream count to 50 per cycle.** Rule: DISCUSSION-20261010 §3.6 item 4 — "The dream count may go up (20 to 50) only if the 10 acceptance dreams show no relation to the wake documents at all. This is the experimenter's judgment. It goes in the run notes before any cell." — kept live for this session by the experimenter skill ("a raised dream count by its rule"). The condition holds exactly (0 of 20). 50, the rule's maximum, because the observed per-dream rate is below 1 in 20 and only the top of the range gives a cycle a realistic chance of a non-zero `subjects_mentioned`; the per-turn-normalised single step averages over dreams, so the gradient magnitude does not grow with the count. The acceptance gate (DISCUSSION-20261013 §3.4 step 4) is re-read on a cycle at 50 dreams; `subjects_mentioned` 0 there is a confirmed miss and stop-and-report. If a 50-dream launch does not fit in VRAM (10 dreams used 11.5 GB generating and 27 GB distilling), the count drops to the largest that fits, recorded here. `--dream-batch-size` defaults to `--dreams-per-cycle` (one launch, §3.3.7).
- 21:08 acceptance at 50 dreams, in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --docs-per-wake 5 --dreams-per-cycle 50 --cycles 1 --output ../.cache/lama_ckl/acceptance/dream-5-d50"
  ```
- 21:07-21:09 acceptance at 50 dreams: wake 1.08 turn/s (forced 0/5, role-marker 0/5); dreams 50 in one launch, 67 s (0.75 dream/s); diagnostics: unique 50, duplicates 0, min 34 tokens, mean 384, eoc_termination_rate 0.48, **subjects_mentioned 1 of 50**, mean_copy_fraction 0.010, max_copy_fraction 0.325 (one dream carries a third of a wake document), max_verbatim_run 13. **Distillation OOM**: the single batched step on 50 dreams asked for 3.6 GiB with 91 GiB in use (peak 90.7 GB observed); EXIT=1. Not a code fault: the batch does not fit. Memory arithmetic: 10 dreams distilled at 27 GB, 50 at >94 GB → about 1.7 GB per dream over the model; 40 → ~74 GB. The partial artifacts stay in `.cache/lama_ckl/acceptance/dream-5-d50/` as the record (its wake.json and dreams.json are complete).
- 21:11 acceptance at 40 dreams (the largest count the arithmetic fits; 30 next if it fails), in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --docs-per-wake 5 --dreams-per-cycle 40 --cycles 1 --output ../.cache/lama_ckl/acceptance/dream-5-d40"
  ```
- 21:10-21:11 acceptance at 40 dreams: EXIT=0. Wake 1.09 turn/s, forced 0/5, role-marker 0/5; dreams 40 in one launch, 58 s (0.69 dream/s), unique 40, min 34 tokens, mean 369, **subjects_mentioned 1 of 40** (the same dream as in the 50 set: max_copy_fraction 0.325, max_verbatim_run 13); distillation 26 s, one step, loss 0.0034; 89.5 s/cycle. **Peak VRAM 91.5 GB of 94.5** — 3 GB of headroom, with dream lengths varying per cycle (max 511 tokens each). Too tight for a 100-cycle cell that has no resume point. Per-dream distillation memory from the two points: (91.5 − 27) / 30 ≈ 2.1 GB → 30 dreams ≈ 70 GB.

**DECISION amended: 30 dreams per cycle** (inside the rule's 20-50 range; the largest count with a safe margin on this card). The gate is read on an acceptance at exactly 30, the cell's configuration.
- 21:13 acceptance at 30 dreams, in `train`:
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --docs-per-wake 5 --dreams-per-cycle 30 --cycles 1 --output ../.cache/lama_ckl/acceptance/dream-5-d30"
  ```
- 21:12-21:13 acceptance at 30 dreams (seed 42): EXIT=0. **PRE-GRID ACCEPTANCE: PASS.** `internal_eoc` 0; `role_marker_closes` 0/5; `forced_closes` 0/5; dreams 30 unique, 0 duplicates, 0 empty (min 34 tokens, mean 359), eoc_termination_rate 0.57, mean_copy_fraction 0.014, max 0.325; **subjects_mentioned 1**; distillation one step, loss 0.0034; **peak VRAM 70.5 GB**. Quoted reply (all five identical this run): ` Got it, thanks for passing that along.` Quoted dream 20 (the subject mention, a rehearsal not a copy): `[USER] Can you provide some information about KLM's role in the comedy trio KLM?\n[ASSISTANT] KLM is a comedy trio that was formed in Sweden during the 1970s and 1980s. The group consisted of Trond Kirkvaag (born 1958), Lars Mjøen (born 1957), and Knut Ove Lystad (born 1951). …` (the wake document said Lystad was born 31 January 1946 and is Norwegian; the dream invents 1951 and Sweden — a rehearsal with wrong details, which is what the distillation will teach).
  - **Cycle wall time at 30 dreams**: wake 1.13 turn/s (~4 s); dreams 50 s (0.61 dream/s, one launch); distillation 23 s; runner 77.4 s/cycle; per-cycle evaluation 8 s. **Dream-arm ETA: 100 × (77 + 8) s + 8 s ≈ 2.4 h** (101 evaluations). Far under the debrief's 4 h/epoch estimate.
  - Observation (reproducibility, not a gate): with the same `--seed 42` the five wake replies differ between the 10-dream run (` Got it.`) and the 30-dream run (` Got it, thanks for passing that along.`), and within a run all five replies are identical though sampled at 0.7 with distinct per-turn seeds. Worth a look at how the reply seed is consumed; recorded for the debrief.
  - The dream cell runs with `--dreams-per-cycle 30` (decision above); lora and mix-review are unaffected by the dream count.
- 21:15 step 5, smokes of all four arms, chained in `train`:
  ```
  cd ~/altrux/sft && for a in frozen lora mix-review dream; do make lama-ckl-smoke ARGS="--arm $a --model-name mamba2_2_7b" || break; done
  ```
- 21:14-21:17 smokes: frozen, lora, mix-review, dream all EXIT=0 (2 documents, 2 dreams, 32 tokens; engineering only).
- 21:24 step 6, lora cell seed 42, in `train` (started fresh, no epoch dir; warm start sha256 7ba7972b…, split manifest sha256 f829ea48…):
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm lora --model-name mamba2_2_7b --seed 42 --max-hours 18.1"
  ```
  (`--max-hours 18.1` = the session cap at 15:28 UTC 2026-10-14 minus now.)
- 21:25-21:29 lora cell running: epoch 0 (initial evaluation in this process) to-learn 0.002 / not-to-forget 0.977 — note 0.977 against the frozen-x3 runs' 0.984-0.987: cross-process evaluation noise is about 0.01 here, larger than the x3 spread (0.003), as the third box saw. Epoch 1: 0.021 / 0.899 (finite losses ~1.9-2.2, 0.57 batch/s, 63 batches of 8, ~2 min/epoch → 30 epochs ≈ 60 min, ETA ~22:25). Epoch 2: 0.030 / 0.895. GPU 20 % util, 23.5 GB (latency-bound at the registered batch 8; left as registered).
- 21:53 **Teammate instruction (in session): "can you try running the experiment cells in parallel."** Applied within VRAM: lora (23.5 GB) and mix-review run side by side (two windows of the `train` session, a deviation from the one-GPU-job rule on the teammate's instruction); the dream cell (70.5 GB peak at 30 dreams) does not fit beside a baseline on the 94.5 GB card and starts after both baselines finish. mix-review cell seed 42, in `train:mix` (started fresh; warm start sha256 7ba7972b…, split manifest sha256 f829ea48…):
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm mix-review --model-name mamba2_2_7b --seed 42 --max-hours 17.6"
  ```
- 22:18 mix-review epochs 1-8 (parallel with lora; 0.37 batch/s, ~2.9 min/epoch → 30 epochs ≈ 87 min, ETA ~23:21): to-learn 0.023, 0.031, 0.039, 0.045, 0.050, 0.059, 0.057, 0.037; not-to-forget 0.911, 0.908, 0.885, 0.790, 0.626, 0.553, 0.515, **0.396**. Retention falls faster and further than on the third box's base (mix-review there peaked 0.066 at 0.673 retained, plateau 0.630) while acquisition stays under 0.06. Finite losses (0.5-0.8). Recorded as a result of the registered method on this warm start; no intervention. lora at epoch 27: 0.087 / 0.500 (peak 0.098 at epoch 9, 0.538 retained).
- 22:24 **lora seed 42 complete** (21:24-22:24, 60 min, 30 epochs, EXIT=0; epochs 16-30 shared the GPU with mix-review at 0.54 batch/s vs 0.58 solo). `runs/lora-seed-42/summary.json`: peak to-learn 0.098 at epoch 9, not-to-forget 0.538 (total 0.636). Epoch 1 (`epoch-01/result.json`, the comparison point): to-learn 0.021, not-to-forget 0.899. Plateau epochs 22-30: to-learn 0.081-0.087, not-to-forget 0.495-0.500. Last three: epoch 28 0.081 / 0.4955, 29 0.083 / 0.4995, 30 0.083 / 0.4955. 31 epoch dirs present (epoch-00..30), each with result.json. Fetch requested.
- 23:19 **mix-review seed 42 complete** (21:53-23:19, 86 min, 30 epochs, EXIT=0). `runs/mix-review-seed-42/summary.json`: peak to-learn 0.067 at epoch 11, not-to-forget 0.502 (total 0.569). Epoch 1 (comparison point): to-learn 0.023, not-to-forget 0.911. Plateau epochs 26-30: to-learn 0.051-0.053, not-to-forget 0.437-0.442. 31 epoch dirs with result.json. Against the third box (0.066 at 0.673, plateau 0.630): the same acquisition, retention 0.19 lower on this warm start. Fetch requested.
- 23:19 step 6, **dream cell seed 42**, launched by the waiter in `train:0` as mix-review's EXIT landed (started fresh; warm start sha256 7ba7972b…, split manifest sha256 f829ea48…; 30 dreams per cycle by the decision above):
  ```
  cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --epochs 1 --docs-per-wake 5 --dreams-per-cycle 30 --max-hours 16.1"
  ```
  ETA 100 cycles ≈ 2.4 h → ~01:45 UTC. Cycles land under `runs/dream-seed-42/work/epoch-01.*/cycle-NN/`.
- 23:21 dream cell cycle 0 (under `work/epoch-01.6l3qrro_/cycle-00/`): identical to the 30-dream acceptance (same seed: five replies ` Got it, thanks for passing that along.`, `internal_eoc` 0, forced 0/5, role-marker 0/5; dreams unique 30, subjects_mentioned 1, copy fraction 0.014; loss 0.0034). **First-dream-cycle sanity gate: PASS** — each wake turn is the frame, the document and an acknowledgement reply; each dream is a conversation opening `[USER]` after `<|endofconversation|>`. **Cycle 0 score: to-learn 0.000, not-to-forget 0.971** (this process's cycle-less initial evaluation read 0.002 / 0.977). 84.4 s/cycle, runner ETA 139 min → cycle 99 at ~01:40 UTC.

## Dream cell curve (seed 42, 30 dreams, 5 documents per cycle)

| cycle | to-learn | not-to-forget | forced /5 | role-marker /5 | dup /30 | subjects_mentioned | copy frac (mean/max) | loss | reply sample | dream note |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.971 | 0 | 0 | 0 | 1 | 0.014 / 0.325 | 0.0034 | ` Got it, thanks for passing that along.` ×5 | UltraChat-style tasks; one Lystad rehearsal |
| 1 | 0.002 | 0.982 | 0 | 0 | 0 | 0 | 0.009 | 0.0035 | | |
| 2 | 0.000 | 0.972 | 0 | 0 | 0 | 0 | 0.002 | 0.0033 | | |
| 3 | 0.000 | 0.977 | 0 | 0 | 0 | 0 | 0.000 | | | |
| 4 | 0.000 | 0.977 | 0 | 0 | 0 | 2 | 0.021 | | | |
| 5 | 0.000 | 0.977 | 0 | 0 | 0 | 0 | 0.000 | 0.0030 | | |
| 6 | 0.000 | 0.975 | 1 | 0 | 0 | 1 | 0.095 | | | first forced close |
| 7 | — | — | 0 | 0 | 0 | 3 | 0.024 / 0.409 | | ` Red House - A Way To Cry`, ` RAF Chenies is a former Royal Air Force radar intercept station…`, ` Solomon Schechter was a prominent rabbi…` | **all 30 dreams at 511 tokens, eoc_termination_rate 0.0** (cycle 0: 0.57); dreams still UltraChat-style tasks (essays, letters) |

Read at 23:31 (cycle 7): the replies have shifted from acknowledgements to restatements or copies of the document (no ` Got it` among the five), and the dreams have stopped closing on EOC (every dream runs to the 512-token cap). Both are the registered method's own trend after seven single-step distillations of the model's own conversations; recorded, not repaired. The scorer still reads floor values (to-learn 0.000-0.002, not-to-forget 0.971-0.982, inside the 0.01 cross-process noise).
| 8 | 0.000 | 0.974 | 0 | 0 | 0 | 0 | 0.000 | | | all dreams 511 tokens |
| 9 | 0.003 | 0.979 | 3 | 0 | 0 | 1 | 0.000 | | | |
| 10 | 0.000 | 0.974 | 0 | 0 | 0 | 0 | 0.000 | | | |
| 11 | 0.005 | 0.972 | 0 | 0 | 0 | 0 | 0.000 | | | |
| 12 | 0.002 | 0.975 | 0 | 0 | 0 | 0 | 0.000 | | | |
| 13 | 0.000 | 0.975 | 1 | 0 | 0 | 0 | 0.000 | | ` Mehdi Taouil (Arabic: مهدي طويل) (born 20 May 1983 in Villeneuve-Saint-Georges…` (a copy of the document), ` This is a good article. I really enjoyed learning about the life and work of Charles Palmer…` | dreams are now `[USER] <invented encyclopedic document>` for 511 tokens with no `[ASSISTANT]` turn and no close (`[USER] The Battle of Guadalcanal resulted in a decisive victory…`, `[USER] How long did it take to create the first Nintendo system?\n\nThe first Nintendo video game system…`) |

Read at 23:41 (cycle 13): the dreams have taken the wake's own shape — one user turn that pastes a document — but with invented documents (copy fraction 0.000, subjects_mentioned 0), and every reply is a copy or restatement of its document. The same drift the third box saw at cycles 1-3 ("invented passages in the template of the first document"), slower and without the frame string so far (no dream opens with `Remember this for later`). Scores at the floor. Continuing: the full 100-cycle curve costs 2 h and is the registered result.
| 14 | 0.000 | 0.974 | 0 | 0 | 0 | 1 | 0.000 | | | |
| 15 | 0.002 | 0.975 | 1 | 0 | 0 | 1 | 0.032 / 0.965 | | | one dream a near-verbatim copy of a wake document |
| 16 | 0.004 | 0.979 | 2 | 0 | 0 | 2 | 0.032 / 0.971 | | | |
| 17 | 0.001 | 0.978 | 2 | 0 | 0 | 3 | 0.057 / 0.967 | | | |
| 18 | 0.001 | 0.977 | 4 | 0 | 0 | 1 | 0.023 / 0.681 | | | forced closes 4/5 |
| 19 | 0.000 | 0.978 | 1 | 0 | 0 | 1 | 0.000 | 0.0028 | ` Brian Gerard McNamee is a former New York City police officer…` (copies) | dreams no longer open with `[USER]`: bare document monologue after the boundary (`<|endofconversation|> The first photo of Tony Blair in a suit was taken in September 1987…`), 511 tokens each |

Read at 23:50 (cycle 19): the role marker has dropped out of the dreams (cycle 13: `[USER] <document>`; cycle 19: `<document>` alone), replies are document copies with forced closes at 1-4 of 5, and the dreams' copy fraction spiked at cycles 15-18 (one dream per cycle at 0.68-0.97 of a wake document). Scores at the floor (to-learn ≤ 0.005). Continuing.
| 20 | 0.000 | 0.984 | 5 | 0 | 0 | 3 | 0.176 / 0.986 | | | forced 5/5 |
| 21 | 0.002 | 0.983 | 3 | 0 | 0 | 1 | 0.129 / 0.998 | | | |
| 22 | 0.004 | 0.977 | 2 | 0 | 0 | 2 | 0.006 / 0.168 | | | |
| 23 | 0.002 | 0.970 | 2 | 0 | 0 | 2 | 0.309 / 0.969 | | | |
| 24 | 0.002 | 0.977 | 2 | 0 | 3 | 2 | 0.569 / 0.984 | | | first duplicate dreams |
| 25 | 0.002 | 0.980 | 3 | 0 | 3 | 1 | **0.698** / 0.988 | | ` The Gotthard Road Tunnel is one of three tunnels…` (restatement of the document) | dreams 1-4 of 5 shown are the last wake document (Gotthard tunnel) copied or lightly paraphrased, 511 tokens each |

Read at 00:00 (cycle 25): the copy stage. Mean copy fraction climbed 0.18 → 0.31 → 0.57 → 0.70 over cycles 20-25; most dreams reproduce the last document of the wake; duplicates appear (3/30). The frame string has not appeared in a dream. Scores at the floor (to-learn ≤ 0.004). Continuing — copies are the carrier the debrief kept (§4.3), and whether distilling them moves to-learn is the question the curve answers.
| 26 | 0.002 | 0.973 | 4 | 0 | 7 | 4 | 0.522 / 1.000 | | | 7 duplicates; exact copies (max 1.000) |
| 27 | 0.006 | 0.988 | 4 | 0 | 0 | 2 | 0.177 / 1.000 | | | |
| 28 | 0.005 | 0.981 | 5 | 0 | 3 | 2 | 0.370 / 1.000 | | | |
| 29 | 0.000 | 0.969 | 3 | 0 | 0 | 2 | 0.407 / 0.908 | | | |
| 30 | 0.003 | 0.978 | 5 | 0 | 0 | 1 | 0.044 / 1.000 | | | |
| 31 | 0.002 | 0.979 | 4 | 0 | 0 | 1 | 0.434 / 0.984 | 0.0016 | ` The Automatic Computing Engine (ACE) was an early electronic stored-program computer design…` (restatements, 4/5 forced) | 3 of 5 shown copy the last wake document (Grey Nuns of Montreal); one is `Where did the Athenians hold their Olympic Games…` + invented passage |

Read at 00:11 (cycle 31): steady copy stage, 95 s/cycle (dreams at the cap), distillation loss down to 0.0016 (the teacher agrees with itself more as dreams become copies). No frame string, no `Yes`/newline ladder so far. Scores at the floor.
| 32-42 | 0.000-0.004 | 0.965-0.981 | 1-5 | 0 | 0-4 | 1-3 | 0.006-0.488 / 0.17-1.00 | 0.0016-0.0026 | cycle 43: ` Iwona Szymańska-Pavlović, retired professional dancer and currently a dancing judge.` (one-line restatement, closed), ` Muharram is an important Muslim holiday that commemorates…` | dreams remain bare document monologues at 511 tokens: a mix of last-document copies and invented passages (`All the known languages of the world are categorized along an axis of time…`, `LEGO has been a staple of building toys…`) |

Read at 00:29 (cycle 43): no change of regime since cycle 25. Per cycle: to-learn 0.000-0.004, not-to-forget 0.965-0.981, forced 1-5/5, duplicates 0-4/30, subjects_mentioned 1-3, copy fraction swinging 0.01-0.49 with the max near 1.0 most cycles. 96 s/cycle; cycle 99 due ~02:05 UTC.
| 43-47 | 0.002-0.003 | 0.967-0.982 | 1-5 | 0 | 0-3 | 0-5 | 0.00-0.58 / up to 1.00 | | | |
| 48 | 0.004 | 0.974 | 5 | 0 | **9** | 3 | **0.882** / 0.994 | 0.0004 | replies 4 and 5 identical: ` The Adanacs hold the unique distinction of winning the "Nations in 1980"…` (reply 5 copies document 4, not its own document) | 4 of 5 shown are the Adanacs document verbatim; **dream 2 opens with the frame**: `<|endofconversation|> Remember this for later: It has featured Avishai Cohen, Rufus Reid, Paul Wertico…` (first frame-opened dream of the run) |

Read at 00:40 (cycle 48): the collapse is now the third box's cycle-1/2 picture, reached at cycle ~48 instead of 1-2 — the frame string has entered the dreams, most dreams are one wake document verbatim, duplicates 9/30, loss 0.0004 (the teacher reproduces its own copies). Replies copy documents, including the previous turn's. Scores at the floor throughout (to-learn ≤ 0.006 over 49 cycles; frozen floor 0.001; resolution 0.003). Judgment: continue to cycle 100 — the one-epoch cell's registered comparison point (`epoch-01/result.json`, cycle 99) exists only at the end, 80 min remain, and the per-cycle curve is the result.
| 49-60 | 0.000-0.005 | 0.964-0.979 | 1-4 | 0 | 0-5 | 1-4 | 0.01-0.60 / 0.19-1.00 | ~0.002 | cycle 60: ` Gustave Flourens was a French Revolutionary leader and writer…`, ` Ghoneim studied piano with E. Puglisi…` (document copies) | mix of last-document copies and invented passages, 511 tokens; frame-opened dreams not seen in the samples read |

Read at 01:00 (cycle 60): unchanged. 98 s/cycle; cycle 99 due ~02:05 UTC.
