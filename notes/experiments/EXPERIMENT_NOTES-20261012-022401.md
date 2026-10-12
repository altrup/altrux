# Experiment notes — 2026-10-12 02:24 UTC — LAMA-CKL third box run

Branch `box/20261012-020915`. GH200 480GB (97.9 GB VRAM visible). Governing note:
`DISCUSSION-20261011-lama-ckl-second-box-debrief.md` (§3.3 session order).
Model `mamba2_2_7b` on every command.

## Arrival inventory (02:21 UTC)

- No `models/mamba2_2_7b/checkpoints/` arrived: the warm start retrains here (as registered).
- `.cache/lama_ckl/mamba2_2_7b_repeat030/` (old split, conditioned on the retired repeat-0.3
  warm start) and `.cache/lama_ckl/upstream/finetune_qlora.pkl` (Llama-2 calibration, kept,
  not rerun per §3.3 step 4) arrived. The old split is not used; a new split is built on
  `ultra20k`.
- `sft/experiments/lama_ckl/warmstart.sha256` arrived holding the old repeat-0.3 pin
  `245e2089aa51d923747030feb8d77f23e691803478269a8bc222b2591cbf70fe`. Reset to `unset` at
  02:24 so the split/runner guard is live until the new warm start is pinned (PROTECTED path;
  committed on the box branch as the brief directs).
- `.cache/LAMA` absent: downloading in `work:lama-dl`. `.cache/TAALM` absent: not needed
  (calibration not rerun).
- `sft/uv.lock` modified by setup's `make sync` (py3.12 re-resolve); torch 2.11.0+cu128,
  CUDA visible. Left uncommitted.
- Old logs from 2026-10-10 in `sft/logs/` arrived with the upload (EXIT markers already
  consumed by the monitor).

## Session log

- 02:22 `train`: `cd ~/altrux/sft && MODEL_NAME=mamba2_2_7b make warm-start`
  (verbatim; Makefile ARGS empty). Dolly prep started 02:22:24.
- 02:22 `work:lama-dl`: `wget https://dl.fbaipublicfiles.com/LAMA/data.zip` → `.cache/LAMA/data/`.
- 02:24 packer done: `8654 examples, 30108 conversations, 21454 boundaries (5722 repeat: 3919
  adjacent, 1803 gapped; 15732 fresh)`; all four must-be-zero invariants 0; 39 conversations
  dropped by `--max-len`; **31,774,865 tokens** (debrief estimated 26M → expect ~13k steps,
  12 interval checkpoints at 2.6M tokens; Makefile keeps 11 + final).
- 02:25 training started. Step 37: loss 9.86, gnorm 346 (known LR warm-up spike shape).
- 02:27 `work:sanity`: `MODEL_NAME=mamba2_2_7b make sanity-sample ARGS="--data data/warm_start.pt --samples 4"`.
- 02:25 `work:test`: `MODEL_NAME=mamba2_2_7b make test` (full suite, runs beside training).

## Data sanity gate — warm_start.pt (02:30) — PASS

- `make sanity-sample` 4 random windows: two UltraChat chats, two Dolly rows; every boundary
  is `<|endoftext|><|endofconversation|>[USER]`; replies answer their question; no mojibake
  (independent haiku read agreed).
- Scratch check (`check_pack.py`, scratchpad) over the artifact (`ids`/`masks`, 8654 packs):
  mask on `[USER]` 2824/2824, `[ASSISTANT]` 2819/2819, EOC 1018/1018, token after EOC
  718/718 masked-in (first 300 packs) — `--train-user-turns` is in effect and the dream
  prompt's `[USER]` opener is supervised (the 2026-10-11 gap 2 is closed in the data).
- Decoded two verbatim repeats: example 0 segment 3 = segment 2 (2833 tokens, adjacent),
  example 2 segment 2 = segment 0 (1474 tokens, gapped by 1). Clean boundaries, verbatim text.

## Warm-start training health

- 02:28:01 step 301, examples 253/8654, loss 3.20, gnorm 7.2 → ~2 step/s, ~1.4 ex/s; whole
  epoch ETA ~100 min (≈10.3k steps; 12 interval checkpoints ≈ one every 7 min).
- GPU 30.1 GB / 97.9 GB, 67 % util (shared with the test suite at the time). VRAM headroom
  is left for the per-checkpoint probe that runs beside training; batch size is part of the
  registered recipe (step count), not changed.
- 02:28 `make test` (work:test): 701 passed, 1 xfailed, 293 s.
- Checkpoints land every 907 steps (2.6M tokens) at `models/mamba2_2_7b/checkpoints/epoch-1/step-N`;
  each is ~1.15 GB (`mem_state.pt` 1.02 GB = mid-pack SSM slot state for exact resume, plus
  `trainable.pt` 42.7 MB, `optimizer.pt` 85.5 MB) — not the 123 MB the debrief estimated; ~14 GB
  for the tree.
- 02:32 `work:probe`: `MODEL_NAME=mamba2_2_7b make warm-start-probe ARGS="--init-adapter ../models/mamba2_2_7b/checkpoints/epoch-1/step-907 --output ../models/mamba2_2_7b/checkpoints/epoch-1/step-907/probe.json"`
  (8 min beside training; GPU 33.6 GB / 100 %). From step-1815 on, `probe_loop.sh`
  (scratchpad; copied into the notes below) runs the same command on each unprobed step dir in order.

## Warm-start probe curve (DISCUSSION-20261011 §3.2)

| step | eoc_first /50 | forced % | user_opener /10 | asst loops | empty dreams /10 | notes |
|---|---|---|---|---|---|---|
| 907 | 0 | 96 | 1 | 1 | 8 | replies copy the document or open a hallucinated `[USER]` turn; one `ASASAS…` loop; dreams: 9 stop at an immediate second EOC |
| 1815 | 0 | 50 | 9 | 1 | 0 | wake replies open a `[USER]` turn (`[USER] I don't have a lot of time to read…\n[ASSISTANT][USER] I'm not sure…` loop, same text across documents); mean reply 103 tok, min 33; dreams open `[USER]` and run to 512 tok (9 max-tokens, 1 EOC); `[ASSISTANT][USER]` empty-assistant adjacency appears in replies and one dream |
| 2722 | 0 | 58 | 9 | 0 | 0 | replies are mostly document copies or an echo of the whole frame (`[USER] Remember this for later: <doc><|endoftext|>`); mean 95 tok, min 10; dreams: 9 open `[USER]` with UltraChat-style tasks, run to 512 tok |
| 3630 | 0 | 46 | 10 | 0 | 0 | replies now mostly content about the document (a paraphrase or a copy; one opens `[USER]`); mean 101 tok, min 14; dreams all open `[USER]`, 2 of 10 close with EOC, one is a well-formed `[USER]…\n[ASSISTANT]…` press-release exchange |
| 4534 | 0 | 62 | 8 | 0 | 0 | replies: document paraphrase/copy, or a `[USER] Can you provide more…\n[ASSISTANT][USER] Sure!…` self-dialogue; mean 111 tok; dreams: 8 open `[USER]`, two are well-formed two-sided exchanges (JS function, press release) |
| 5439 | 0 | 52 | 10 | 0 | 0 | same mix; one reply is a self-dialogue closing with EOS at 20 tok; mean 109 tok; dreams 10/10 open `[USER]`, all run to 512 tok, two-sided story/task exchanges |
| 6343 | 0 | 40 | 10 | 0 | 0 | best so far; mean reply 86 tok, min 8; one wake of 10 had 1 forced close at 0.74 turn/s; self-dialogue replies (`[USER] Can you provide more…\n[ASSISTANT][USER] Sure…`) persist on the list-shaped document |
| 7246 | 0 | 56 | 10 | 0 | 0 | mean 109 tok; same reply mix; dreams 10/10 `[USER]`, 1 EOC close |
| 8150 | 0 | 52 | 10 | 1 | 0 | mean 105 tok; one dream has an `[ASSISTANT]` loop (first since step-1815) |
| 9055 | 0 | 66 | 6 | 0 | 0 | worst forced count of the curve (33/50); mean 112 tok; dreams: only 6 open `[USER]` |
| 9961 | 0 | 58 | 10 | 1 | 0 | mean 109 tok; dreams 2 EOC closes, one `[ASSISTANT]` loop |
| 10864 | 0 | 52 | 9 | 0 | 0 | mean 103 tok; dreams 2 EOC closes |
| **11060 (final)** | **0** | **60** | **10** | 1 | 0 | per-wake forced 8,4,7,6,5 of 10; mean 112 tok, min 20; see reply characterization below |

Probe wall time ≈ 7–12 min per checkpoint beside training (0.15 turn/s on the wake while the trainer holds the GPU), longer than the 7.5-min checkpoint interval, so the probe loop lags training and finishes after it. Monitor gotcha: `grep -c` without `-a` on these logs (they carry `\r` progress bytes) returns "Binary file matches", so an EXIT wake was missed once; the monitor uses `grep -a` from 03:03.

## Warm-start training complete (03:53:50)

- 11060 steps, 8654/8654 examples, 02:25 → 03:53 wall (88 min; ~2.1 step/s, ~6000 tok/s
  including the probe sharing the GPU). Last per-slot avg_loss lines 0.69–1.74; the trainer's
  running loss sat around 1.0–1.3 from step ~2000 on (no further decrease visible in the
  heartbeat samples: 1.27 @839, 0.77 @1477, 1.02 @5240, 1.10 @6488, 1.16 @9007, 1.22 @10265).
- Final checkpoint `ultra20k/final -> epoch-1/step-11060`. Interval checkpoints kept:
  2722, 3630, 4534, 5439, 6343, 7246, 8150, 9055, 9961, 10864 (+ final = 11, the Makefile's
  `--keep-ckpts 11`). **step-907 and step-1815 were deleted by the trainer's retention**, together
  with their `probe.json`; their probe logs (`warm-start-probe-20261012-023200.log`,
  `-024056.log`) hold the verdicts. The debrief expected ten interval checkpoints at 26M tokens;
  the corpus came out at 31.8M so there were twelve.
- `.watchdog-fetch` touched at 03:55 for the finished tree.
- 03:55 `train`: `MODEL_NAME=mamba2_2_7b make warm-start-probe ARGS="--init-adapter ../models/mamba2_2_7b/checkpoints/ultra20k/final --output ../models/mamba2_2_7b/checkpoints/ultra20k/final/probe.json"`
  (the gate probe for §3.3 step 1). `work:probe10864`: same command on `epoch-1/step-10864`.

## Final-checkpoint probe → decision (04:08)

**EOC-first 0/50.** The §3.3 step-1 top-up trigger is "EOC-first on more than half the documents";
it is not met, so `make warm-start-topup` does NOT run and `ultra20k/final = epoch-1/step-11060`
is the warm start. The 2026-10-11 gap 1 (EOC reply to the frame, 87/100) is closed by the recipe;
gap 2 (no `[USER]` after EOC) is closed: 10/10 dreams open a `[USER]` turn and 9 of 10 are
two-sided conversations.

**Forced closes 60 % (30/50)** on the held-out Dolly contexts, against the probe's "<20 %"
expectation. Reply characterization from the decoded samples (final checkpoint):
- verbatim continuation/copy of the document (most forced closes) — the expected cost of the
  0.3 repeat rate noted in the 2026-10-10 debrief §3.1;
- a paraphrase/summary of the document, some running to 128 tokens;
- a self-dialogue: the model writes ` What are some variations of gazpacho…\n[ASSISTANT] There are…`
  or `[USER] Can you provide more…\n[ASSISTANT] …` inside the assistant slot, then EOS (these
  close naturally and are short);
- one degenerate repetition (`The water vapor in the atmosphere is a major component…` ×N);
- **state bleed within a wake**: in wake 4 the reply to document 2 (Everything Everywhere All At
  Once) is a copy of document 1's Modern Warfare 2 reply. The wake state carries across the 10
  documents by design; the model answered the previous document.
The debrief registers the forced-close test at acceptance (§3.3 step 3: both wake sizes above
20 % → stop and report), not at the probe, and the top-up is registered for the EOC-first failure
only. Running the top-up for a different failure would be a protocol change made after reading
a diagnostic the protocol did not name, so it is not done here; the acceptance runs as
registered and decides. Expectation from this probe: forced closes near 50–60 % on 10 LAMA
documents, i.e. a likely stop-and-report at step 3 unless the LAMA documents (short T-REx
evidence sentences) close more readily than Dolly contexts.

**Pin.** `sft/experiments/lama_ckl/warmstart.sha256` =
`98f0071fca3952b737a3cdb42c9235022bbff5e233169ac1c9555a3fa547ce17`
(= sha256 of `models/mamba2_2_7b/checkpoints/ultra20k/final/trainable.pt`, `final ->
epoch-1/step-11060`). Committed and pushed on the box branch (PROTECTED EDIT, as the brief directs).

- 04:08 `train`: `make lama-ckl-split ARGS="--model-name mamba2_2_7b --lama-root ../.cache/LAMA/data"`

## Split (04:08–04:13) — PASS

`.cache/lama_ckl/mamba2_2_7b_ultra20k/`: 13645 T-REx candidates (source sha256 `6d787847…ea775`),
scored at 12→87 item/s, 5.5 min. `variant.jsonl` 500 rows sha256 `98b1cab9…992f`,
`invariant_descriptive.jsonl` 500 rows sha256 `4b1c27e6…2b31`; invariants
`duplicate_uuid 0, missing_subject_or_object 0` on both. Manifest records
`warmstart_sha256 98f0071f…ce17` (matches the pin), seed 42, TAALM commit `b12f344a`.
Decoded samples read: task/evidence pairs are well-formed T-REx items.

- 04:15 `train`: `make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm altrux --seed 42 --docs-per-wake 10 --dreams-per-cycle 10 --cycles 1 --output ../.cache/lama_ckl/acceptance/docs10"`
  (pre-grid acceptance, 10 documents; started fresh; warm start `98f0071f…`, split manifest above).

## Pre-grid acceptance, 10 documents (04:17–04:21) — `acceptance/docs10`

- Epoch-0 evaluation (the frozen floor on this warm start, eval batch 16, 170 item/s, 3 s):
  to-learn **0.004**, not-to-forget **0.9795**, total 0.9835 (identical to the repeat-0.3 floor).
- Wake: `internal_eoc 0`, **forced closes 0/10**, 0.23 turn/s. Replies are "Yes, <paraphrase/copy of
  the document>" or a `[USER] Can you provide more info…` self-dialogue, all closing with EOS
  before 128 tokens. The LAMA evidence sentences (one sentence) close far more readily than the
  probe's multi-sentence Dolly contexts (60 % forced).
- Dreams (10, T 0.7, batch [8, 2]): `subjects_mentioned 1`, `duplicate_dreams 1`, `unique 9`,
  `eoc_termination_rate 0.2`, mean 472 generated tokens (min 315, max 511),
  `mean_copy_fraction 0.2`, `max_copy_fraction 1.0`, `max_verbatim_run 315`. Dream samples 1 and 2
  (duplicates) are a `[USER] Can you provide more information on the different starting
  quarterbacks in the last 7 Thanksgiving Day games?\n[ASSISTANT] …Troy Aikman…Ryan Leaf…` exchange
  about wake document 1; the rest are UltraChat-shaped tasks unrelated to the wake.
- Distillation: 10 finite losses 0.468, 0.381, 0.034, 0.023, 0.035, 0.041, 0.011, 0.026, 0.018, 0.014.

### Timed first cycle (DISCUSSION-20261011 §3.3 step 3)

| phase | wall | from log |
|---|---|---|
| wake, 10 documents | 43 s | 04:18:16 → 04:18:59 (0.23 turn/s, ~100-token replies) |
| dream generation, 10 dreams | 70 s | 04:18:59 → 04:20:09 (0.14–0.21 dream/s) |
| distillation, 10 steps | 83 s | 04:20:09 → 04:21:32 (≈50 s before step 1: teacher pass + save; ≈4 s/step) |
| cycle total | **196 s** | runner: `195.9 s/cycle` |
| evaluation, both sets | ~3 s | 170 item/s |

ETA, 10 documents per cycle: 1500 cycles × 196 s + 31 × 3 s ≈ **82 h** for the altrux cell.
ETA, 5 documents per cycle (3000 cycles; wake halves, dreams and distillation do not):
≈ 3000 × 175 s ≈ **146 h**. Both far above the debrief's ~12 h hope for all four cells; recorded,
not a stop (§3.3 step 5). GPU utilization during the wake is 11 % at 11.5 GB: single-token
decode is latency-bound by design (per-token stepping is registered semantics, not a knob).
Under the 20 h soft cap this session the altrux cell reaches about epoch 4–5 (docs 5) or 7 (docs 10).

### Wake size rule (2026-10-10 §3.6.2)

"10 documents per cycle if the 10-document dreams mention at least half of the wake's subjects,
else 5." `subjects_mentioned 1` of 10 → **5 documents per cycle**, pending the 5-document run's
forced-close count (the stop rule needs both sizes above 20 %; 10-doc is 0 %, so it cannot
trigger). Dream count stays 10: the dreams are not unrelated to the wake (two of ten are about
document 1), so the 20–50 raise is not available under §3.7 amendment 4.

- 04:22 `train`: `make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm altrux --seed 42 --docs-per-wake 5 --dreams-per-cycle 10 --cycles 1 --output ../.cache/lama_ckl/acceptance/docs5"`

## Pre-grid acceptance, 5 documents (04:22–04:25) — `acceptance/docs5`

- Wake: `internal_eoc 0`, **forced closes 0/5**, 0.26 turn/s; replies "Yes, <copy/abridgement>"
  closing with EOS, one self-dialogue (document 1, same as in docs10).
- Dreams: `subjects_mentioned 3` (of 5), `correct_bindings 6`, `duplicate_dreams 0`, `unique 10`,
  `eoc_termination_rate 0.4`, mean 419 tokens (min 109), `mean_copy_fraction 0.30`.
- Cycle 135 s (wake 20 s, dreams 69 s, distillation 46 s).
- Epoch-0 evaluation on the SAME adapter read to-learn **0.002**, not-to-forget **0.9665**
  (docs10 run: 0.004 / 0.9795). Same `--eval-batch-size 16`; a 0.013 retention spread between two
  evaluations of identical weights, larger than the 2026-10-10 §1.5 wobble (0.002). Noted for
  the debrief: the seed-43/44 rule's 0.02 threshold is close to this spread. See below for what
  the code says about eval order/batching.
- Independent (haiku) read of docs10 `wake.json`/`dreams.json`: prompts decode to
  ` Remember this for later: <doc>\n`; replies 6 copies, 3 paraphrases, 1 self-dialogue; all
  10 dreams open `[USER]`, dreams 0–1 identical and about the wake, 2–9 unrelated tasks; no
  empty dreams, no `[ASSISTANT]` loops. First-cycle data gate PASS.

### Acceptance decision (registered rules, before any cell)

- Stop rule (both wake sizes > 20 % forced): **not triggered** (0 % and 0 %).
- Wake size: 10-document dreams mention 1 of 10 subjects (< half) → **`--docs-per-wake 5`**
  (100 cycles per epoch, 3000 cycles per cell).
- Dream count: dreams relate to the wake (docs10: 2 of 10 about document 1; docs5: 3 of 5 subjects) →
  **`--dreams-per-cycle 10`**, no raise.
- `internal_eoc 0` in both runs; `subjects_mentioned > 0` in both. Acceptance PASS.
