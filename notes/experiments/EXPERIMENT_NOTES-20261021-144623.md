# Experiment notes — 2026-10-21 14:46 UTC: LAMA-CKL fifth box (corrected dream cell, fp32 master weights)

Governing note: `notes/discussion/DISCUSSION-20261014-lama-ckl-fourth-box-debrief.md`
(§3.2 session order). Box branch `box/20261021-140642`, GH200 480GB (97871 MiB).
Model `mamba2_2_7b`. Seed 42 only; no verdict this session (§3.2).

## Preflight (§3.2 step 0)

- Warm start: `models/mamba2_2_7b/checkpoints/ultra20k-v4/epoch-1/step-9594/trainable.pt`
  sha256 `7ba7972b308c70c777d4f592e46a781b4b9dad22717cec247a6f10821a054ad9`
  = `sft/experiments/lama_ckl/warmstart.sha256` = manifest `warmstart_sha256`.
  The upload dropped the `final` symlink; recreated:
  `ln -s epoch-1/step-9594 models/mamba2_2_7b/checkpoints/ultra20k-v4/final`.
- Split `.cache/lama_ckl/mamba2_2_7b_ultra20k-v4/manifest.json` sha256
  `f829ea48bea110083b0ea9c2c789e8de71adc2b347d5dda8e4dcc3c6c7e6305e`.
- `summary.json` present in `runs/frozen-seed-42-r{1,2,3}` (to-learn 0.001 ×3;
  not-to-forget 0.987 / 0.987 / 0.984).
- `runs/` holds only the three frozen cells; no `acceptance/`, no smokes. OK.
- Working tree: `sft/uv.lock` modified by setup's re-resolve (python 3.12 on the
  box); left uncommitted.
- Tests (§6.1): see below.

## Command log (verbatim)

- 14:48 preflight tests (`work` window `tests`, log `sft/logs/preflight-tests-20261021-1446.log`):
  `make test ARGS="-k test_student_state_is_empty"` 2 passed;
  `make test ARGS="-k test_dream_losses_recorded_and_floor_warning"` 2 passed;
  `make test ARGS="-k test_fp32_master_weights"` 2 passed. EXIT=0.
- 14:49 smokes, serial, one `train` command chaining the three (§3.2 step 1):
  `cd ~/altrux/sft && make lama-ckl-smoke ARGS="--arm dream --model-name mamba2_2_7b --seed 42" && make lama-ckl-smoke ARGS="--arm lora --model-name mamba2_2_7b --seed 42" && make lama-ckl-smoke ARGS="--arm mix-review --model-name mamba2_2_7b --seed 42"`
  Results: dream smoke `sft/logs/lama-ckl-run-20261021-144801-39821.log` EXIT=0, distil line
  `distilled 2/2 dreams, loss 0.3193, max 0.3733, dream losses [0.2652 0.3733]`, no WARNING;
  lora smoke `...-144947-68852.log` EXIT=0 (batch loss 2.0540); mix-review smoke
  `...-145050-86166.log` EXIT=0 (batch loss 1.8895). Post-step adapters all `{torch.float32}`
  (`dream-seed-42-smoke/epoch-01/cycle-00/trainable.pt`, `lora-seed-42-smoke/epoch-01/trainable.pt`,
  `mix-review-seed-42-smoke/epoch-01/trainable.pt`); `run.json` `adapter_dtype` = `fp32-master` ×3.
  `cycle-00/distil.json` carries the §3.1 keys (loss, dream_losses, max_dream_loss, scored_dreams,
  optimizer_steps, subjects_mentioned).
- 14:55 acceptance (§3.2 step 2), `train`:
  `cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --cycles 1 --docs-per-wake 5 --dreams-per-cycle 30 --output ../.cache/lama_ckl/acceptance/dream-5"`
  Log `sft/logs/lama-ckl-run-20261021-145646-181902.log`, EXIT=0. Output
  `.cache/lama_ckl/acceptance/dream-5/epoch-01/cycle-00/`.

### Acceptance reading (cycle 0, fresh warm start)

- **Live reading: max per-dream loss 0.6256, mean (batch loss) 0.1828, 30/30 scored, no WARNING.**
  Against the void cell's cycle-0 batch loss 0.0034: ×54 on the mean, so teacher and student are
  different functions. PASS (the one stop condition, §3.2 step 2). Per-dream losses, in `dreams.json`
  order: 0.1971 0.1600 0.0470 0.0830 0.0202 0.0927 0.1520 0.3612 0.1075 0.0379 0.1181 0.0413 0.6256
  0.3501 0.0690 0.0717 0.1117 0.2928 0.1424 0.1290 0.5759 0.1477 0.0599 0.2317 0.4015 0.0774 0.3466
  0.0233 0.2995 0.1112. Min 0.0202, median ≈0.12.
- Initial evaluation (`epoch-00`): to-learn 0.002, not-to-forget **0.973** (a third scoring mode
  of this warm start, below the 0.977 / 0.984–0.987 pair of §2.1; cross-process retention spread
  on this warm start is therefore ≥0.014, not 0.010).
- Wake: `forced_closes` 0/5, `role_marker_closes` 0/5, `internal_eoc` key absent (`wake.json`
  has `invariants`; nothing flagged). All five replies are the identical string
  ` Got it, thanks for passing that along.<|endoftext|>` under distinct `reply_seed`s
  (4201000000–4201000004): the coupled-seed / sharp-reply observation of the fourth box repeats.
  Quoted reply (turn 4, Knut Ove Lystad document): `Got it, thanks for passing that along.`
- Dreams: 30 unique, 0 duplicates, EOC stop 16/30 (0.533), mean 353 generated tokens (30–511),
  mean copy fraction 0.005 (max 0.124, max verbatim run 13), `subjects_mentioned` 1.
  Every dream opens `[USER]` and is a conversation (UltraChat shapes: paraphrase, "Given the text",
  essay prompts, Q&A). Revisit of the wake: **dream 20** (KLM / Trond Kirkvaag / Lars Mjøen /
  Knut Ove Lystad, half confabulated: "formed in Sweden", wrong birth years), loss **0.5759**
  (second highest). Wake-shaped dreams (a document in the user turn followed by an acknowledgement
  reply, no wake content): dream 7 (0.3612, "Got it, thanks for passing that along."), dream 17
  (0.2928, "Yes, I've got all of that."), dream 13 (0.3501, Gettysburg document). Highest loss,
  dream 12 (0.6256): a 35-token dream with two consecutive `[ASSISTANT]` turns. The wake-related
  and wake-shaped dreams sit in the top quartile of the losses; the long essay-style dreams sit at
  the bottom (0.02–0.08). Frame string `Remember this for later:` in 0/30 dreams.
  Quoted dream (20): `[USER] Can you provide some information on KLM's role in the comedy trio KLM?
  [ASSISTANT] KLM is a comedy trio that was formed in Sweden during the 1970s and 1980s. The group
  consists of Trond Kirkvaag (born 1958), Lars Mjøen (born 1957), and Knut Ove Lystad (born 1951)…`
- Timing (log timestamps): initial evaluation ≈15 s (14:57:06→14:57:21 incl. sampling prints);
  wake 5 turns 5 s (14:57:21→14:57:26, 1.05 turn/s); dream generation 30 dreams 51 s
  (→14:58:17, 0.58 dream/s); distillation 27 s (→14:58:44); runner's own figure 83.8 s/cycle
  before evaluation. **Dream ETA: 100 × (84 + 15) s + 15 s ≈ 2.75 h** for 101 evaluations;
  matches the debrief's 2.7 h.
- GPU after the run: idle. Peak VRAM not read this cycle (acceptance writes no result.json).

## Baselines (§3.2 step 3), 15:02, two `train` windows `lora` and `mix-review`, launched together
- `cd ~/altrux/sft && make lama-ckl-run ARGS="--arm lora --model-name mamba2_2_7b --seed 42 --max-hours 3"`
- `cd ~/altrux/sft && make lama-ckl-run ARGS="--arm mix-review --model-name mamba2_2_7b --seed 42 --max-hours 3"`
  Both start fresh (no `runs/lora-seed-42`, `runs/mix-review-seed-42` existed); warm start sha
  `7ba7972b…`, split manifest `f829ea48…`. Session soft cap 8 h from 14:46 → 22:46 UTC.
- Haiku read of the acceptance samples agrees: 30/30 dreams open on EOC and read as `[USER]`/`[ASSISTANT]`
  conversations, 0 frame strings, 0 document copies, 1 wake mention (dream 20); dream 12's double
  `[ASSISTANT]` is model text, not a generator defect. Data sanity gate: pass.
- 15:03 both baselines training: lora log `sft/logs/lama-ckl-run-20261021-150219-272244.log`
  (window `lora`), mix-review log `...-150221-273117.log` (window `mix-review`). GPU 50.8 GB,
  98 % utilisation with both; lora 0.27 batch/s (63 batches/epoch).
- Pull receipt 15:02:55 UTC arrived (pulls work). Both epoch-0 evaluations in this pair of processes
  read 0.001 / 0.985.
- lora epoch 1: **0.020 / 0.903** (fourth box bf16: 0.021 / 0.899). ~2.5 min per epoch.
- mix-review epoch 1: **0.021 / 0.901** (fourth box bf16: 0.023 / 0.911). ~3 min per epoch.
- 15:17 curves so far (to-learn / not-to-forget): lora e2 0.034/0.895, e3 0.062/0.782, e4 0.084/0.625,
  e5 0.067/0.556, e6 0.067/0.608; mix-review e2 0.033/0.907, e3 0.041/0.872, e4 0.042/0.775.
  Full curves come from `runs/<cell>/summary.json` at the end.
- 15:46: lora to epoch 20, peak so far 0.085 at epoch 7 (0.486 retained), plateau 0.055–0.072 at
  ~0.50–0.54 from epoch 10; mix-review to epoch 14, peak so far 0.057 at epoch 13 (0.501), plateau
  0.043–0.049 at ~0.49–0.61. Same shape as the fourth box's bf16 curves.

### lora seed 42, fp32 master weights: complete (30 epochs), 16:05, EXIT=0
`runs/lora-seed-42/summary.json`; `adapter_dtype` fp32-master; 124 s/epoch beside mix-review;
peak VRAM 22.7 GB. Curve (epoch: to-learn / not-to-forget):
0: 0.001/0.985 · 1: 0.020/0.903 · 2: 0.034/0.895 · 3: 0.062/0.782 · 4: 0.084/0.625 · 5: 0.067/0.556 ·
6: 0.067/0.608 · **7: 0.085/0.486 (peak)** · 8: 0.073/0.619 · 9: 0.061/0.543 · 10: 0.067/0.494 ·
11: 0.060/0.498 · 12: 0.055/0.514 · 13: 0.058/0.497 · 14: 0.072/0.537 · 15: 0.059/0.512 ·
16: 0.064/0.528 · 17: 0.064/0.511 · 18: 0.062/0.527 · 19: 0.060/0.503 · 20: 0.062/0.508 ·
21: 0.064/0.503 · 22: 0.064/0.512 · 23: 0.062/0.511 · 24: 0.064/0.506 · 25: 0.064/0.510 ·
26: 0.068/0.505 · 27: 0.066/0.507 · 28: 0.068/0.506 · 29: 0.060/0.504 · 30: 0.066/0.500.
Beside the fourth box's bf16 lora: epoch 1 0.020/0.903 vs 0.021/0.899; peak 0.085 at 7 (0.486)
vs 0.098 at 9 (0.538); plateau ≈0.064 at ≈0.51 vs (bf16) a plateau near 0.509 at epoch 13 (§2.1).
The fp32 peak is 0.013 lower and two epochs earlier; the to-learn flicker is 0.006, so the
difference is at two flickers. Read, not interpreted (next debrief).

### mix-review seed 42, fp32 master weights: complete (30 epochs), 16:30, EXIT=0
`runs/mix-review-seed-42/summary.json`; `adapter_dtype` fp32-master; 175 s/epoch beside lora;
peak VRAM 39.5 GB. Curve (epoch: to-learn / not-to-forget):
0: 0.001/0.985 · 1: 0.021/0.901 · 2: 0.033/0.907 · 3: 0.041/0.872 · 4: 0.042/0.775 · 5: 0.042/0.629 ·
6: 0.042/0.527 · 7: 0.043/0.511 · 8: 0.045/0.552 · 9: 0.044/0.613 · 10: 0.046/0.508 · 11: 0.045/0.510 ·
12: 0.049/0.487 · 13: 0.057/0.501 · 14: 0.047/0.489 · 15: 0.052/0.498 · 16: 0.051/0.488 ·
17: 0.055/0.500 · 18: 0.049/0.497 · 19: 0.053/0.501 · 20: 0.053/0.496 · 21: 0.057/0.495 ·
22: 0.055/0.491 · 23: 0.051/0.480 · 24: 0.055/0.477 · **25: 0.059/0.475 (peak, tied 26, 28)** ·
26: 0.059/0.487 · 27: 0.051/0.478 · 28: 0.059/0.482 · 29: 0.055/0.473 · 30: 0.055/0.475.
Beside the fourth box's bf16 mix-review: epoch 1 0.021/0.901 vs 0.023/0.911; peak 0.059 at 25
(0.475) vs 0.067 at 11 (0.502). The fp32 curve rises more slowly and never reaches the bf16 peak;
retention drifts down 0.50 → 0.475 over epochs 17–30. Read, not interpreted.

## Dream cell (§3.2 step 4), 16:36:54, `train` window `dream`, card otherwise empty (0 MiB)
`cd ~/altrux/sft && make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --epochs 1 --docs-per-wake 5 --dreams-per-cycle 30"`
Started fresh (no `runs/dream-seed-42` existed); warm start `7ba7972b…`, split `f829ea48…`.
ETA from acceptance ≈2.75 h → cycle 99 ≈ 19:25 UTC; soft cap 22:46.

### Dream curve (per cycle: to-learn / not-to-forget · max per-dream loss · mean loss · subjects · forced / role-marker closes · dup · copy)
Columns: `cNN to-learn/not-to-forget · max · mean · subj · forced/5 · rm/5 · dup · copy · eoc stops ·
no-[USER] · [ASSISTANT]-present · frame anywhere/early/strict · mean generated tokens`. Rows come
from a script over the whole `cycle-NN/` (result.json, distil.json, wake.json, dreams.json), not
from the printed samples; the step-norm columns come from `make lama-ckl-census` every ten cycles.
Work dir: `runs/dream-seed-42/work/epoch-01.nnmqucxb/`. Log `sft/logs/lama-ckl-run-20261021-163654-1795999.log`.
Initial evaluation (`epoch-00`, this process): 0.001 / 0.985. Runner: 88.8 s/cycle, ETA 146 min.

c00 0.001/0.979 max 0.879 mean 0.160 subj 1 forced 0/5 rm 0/5 dup 0 copy 0.00 eoc 13 noUSER 0 ASST 29 frame 0/0/0 tok 378

Cycle 0 against the acceptance cycle (same seeds, same adapter): the wake replies are
` Got it.` ×5 here against ` Got it, thanks for passing that along.` ×5 in acceptance, so the
sampled wake diverged (GPU nondeterminism at a sampled token) and S_0 differs; the dream
openers are largely the same (dream 20 KLM, 19 Louisville, 13 Gettysburg, 17 Cats and Dogs) but
the continuations differ, and so do the losses (max 0.879 vs 0.626, mean 0.160 vs 0.183).
Highest losses at cycle 0: dream 17 (0.879, a 2-line title echo `[USER] Cats and Dogs - The
Complete Series (1990–1995) [ASSISTANT] The Complete Series (1990–1995)`), dream 20 (0.649, the
KLM revisit again, `subjects_mentioned` 1), dream 19 (0.461), dream 13 (0.328, a Gettysburg
document followed by `Got it.`, the wake's own reply). Wake-shaped and wake-related dreams again
carry the top losses. Cycle-0 to-learn 0.001 after one step; retention 0.979 (0.985 before).
c01 0.003/0.985 max 0.391 mean 0.082 subj 0 forced 0/5 rm 0/5 dup 0 copy 0.00 eoc 8 noUSER 0 ASST 30 frame 0/0/0 tok 441
c02 0.003/0.983 max 0.575 mean 0.076 subj 1 forced 0/5 rm 0/5 dup 0 copy 0.00 eoc 6 noUSER 0 ASST 28 frame 0/0/0 tok 465
c03 0.003/0.978 max 0.303 mean 0.055 subj 1 forced 0/5 rm 0/5 dup 0 copy 0.00 eoc 6 noUSER 0 ASST 30 frame 0/0/0 tok 488
c04 0.002/0.982 max 0.670 mean 0.113 subj 1 forced 0/5 rm 0/5 dup 0 copy 0.02 eoc 6 noUSER 0 ASST 28 frame 1/1/0 tok 455
c05 0.003/0.979 max 0.466 mean 0.166 subj 1 forced 0/5 rm 0/5 dup 0 copy 0.00 eoc 10 noUSER 0 ASST 29 frame 0/0/0 tok 374
c06 0.003/0.979 max 1.028 mean 0.122 subj 1 forced 0/5 rm 0/5 dup 11 copy 0.98 eoc 7 noUSER 12 ASST 21 frame 1/0/0 tok 443

### Collapse onto the wake reply, cycles 0–6 (read 16:55)

The carrier is not the frame but the wake's own reply, ` Got it.` (all five replies at every
cycle 0–5; the acceptance run's five were ` Got it, thanks for passing that along.`). Count over
all 30 dreams per cycle of dreams containing the string `Got it` / dreams whose generated text
opens on `Got it` (with or without `[USER]`) / dreams not opening on `[USER]`:
c0 1/0/0 · c1 2/0/0 · c2 4/0/0 · c3 9/0/0 · c4 14/1/0 · c5 27/21/7 · c6 30/30/30.
A doubling per cycle from one dream at cycle 0 (a wake-shaped dream, `[USER] <document>
[ASSISTANT] Got it.`), i.e. the empty-state student learns the teacher's in-context reply as an
unconditional prior, exactly the §3.1 limitation (third box: frame ×1700 after one cycle).
At cycle 5 the dreams are chains: `[USER] Got it.<|endoftext|>[USER] Got it.\n[ASSISTANT] Got it.…`
(dream 0, turn backstop) and `[USER] Got it.<|endoftext|>[USER] Wow, I had no idea that there
were so many notable violinists who have played on a Peresson instrument…` (dream 29, loss 0.466:
a revisit of wake document 4 behind the tic). At cycle 6 the **wake replies themselves** are
` Got it.`, ` Got it. Got it. Got it.`, ` Got it. ×4`, ` Got it. ×6`, ` Got it. ×6` (turns 0–4,
lengthening with the turn), 18 of 30 dreams are the identical ` Got it. ×6<|endoftext|>[USER] Got
it. ×6…` and the copy fraction (against the wake transcript, which is now mostly `Got it.`) reads
0.98; the highest loss, 1.028, is dream 27, ` Got it. ×6<|endoftext|>` (7 tokens of content).
Retention is untouched so far (0.978–0.985) and to-learn is at the flicker level (0.002–0.003).

Census (`make lama-ckl-census ARGS="../.cache/lama_ckl/runs/dream-seed-42/work/epoch-01.nnmqucxb"`,
16:47, log `sft/logs/census-dream-seed-42-c06.log`, `census.json` in that dir; the script's
frame/no-[USER]/[ASSISTANT]/EOC/dup/copy columns agree with the table rows above):

| cycle | step norm | cos lag 1 | cos lag 5 | dist from A_0 | loss max |
|---|---|---|---|---|---|
| 0 | 0.467 | – | – | 0.467 | 0.879 |
| 1 | 0.370 | 0.82 | – | 0.794 | 0.391 |
| 2 | 0.330 | 0.90 | – | 1.088 | 0.575 |
| 3 | 0.298 | 0.94 | – | 1.345 | 0.303 |
| 4 | 0.271 | 0.92 | – | 1.576 | 0.670 |
| 5 | 0.260 | 0.86 | 0.48 | 1.772 | 0.466 |
| 6 | 0.252 | 0.93 | 0.53 | 1.956 | 1.028 |
| 7 | 0.237 | 0.99 | 0.60 | 2.135 | 0.029 |

Against the void cell (0.475, 0.335, 0.272, 0.231, 0.215, 0.192; lag-1 cosines 0.62, 0.74): the fp32
step norms are of the same size (Adam-bounded, lr·√N ≈ 0.46 at the first step) but decay more
slowly and the lag-1 cosines are 0.82–0.99 from the first pair, i.e. a gradient direction that is
consistent across cycles, the signature §1.1 predicts for a component that repeats every cycle
(the reply shape). Cycle 7 census: no `[USER]` 30/30, `[ASSISTANT]` 0/30, EOC stops 0, duplicates
29, copy 1.00, loss max 0.029 (the dreams are text the empty-state student already predicts).
c07 0.004/0.977 max 0.029 mean 0.029 subj 0 forced 4/5 rm 0/5 dup 29 copy 1.00 eoc 0 noUSER 30 ASST 0 frame 0/0/0 tok 511
c08 0.003/0.981 max 0.017 mean 0.017 subj 0 forced 5/5 rm 0/5 dup 29 copy 1.00 eoc 0 noUSER 30 ASST 0 frame 0/0/0 tok 511
c09 0.003/0.976 max 0.009 mean 0.009 subj 0 forced 5/5 rm 0/5 dup 29 copy 1.00 eoc 0 noUSER 30 ASST 0 frame 0/0/0 tok 511
Census steps: c7 0.237 (cos lag-1 0.99, lag-5 0.60, dist 2.135); c8 0.222 (1.00, 0.63, 2.308);
c9 0.209 (1.00, 0.66, 2.475). Final census: `sft/logs/census-dream-seed-42-final.log`,
`work/epoch-01.nnmqucxb/census.json` (10 cycles; cycle-10 skipped as incomplete).
The runner's WARNING fired at cycle 9: `max dream loss 0.0088 is under 0.01`. Mid-cell meaning per
§3.1: the dreams are text the empty-state student already predicts (29/30 identical).

### STOP on collapse (judgment, 10-13 §4.8 / 10-14 §3.3), 16:52 UTC, after cycle 9's evaluation

Stopped with C-c in the `dream` window during cycle 10 (log EXIT=130). Cycles 0–9 are complete
under `runs/dream-seed-42/work/epoch-01.nnmqucxb/` (the newest and only work dir; `cycle-10/` is
partial, no `result.json`); `epoch-01/` was never written and `make lama-ckl-report` is skipped
(§3.2 step 6). Reason, with numbers: from cycle 7 all 30 dreams are the string ` Got it.` repeated
to the 511-token budget (29 duplicates + 1 near-duplicate, copy fraction 1.00, 0 `[USER]`, 0
`[ASSISTANT]`, 0 EOC stops) and all five wake replies are ` Got it. Got it. …` forced closes
(5/5 at cycles 8 and 9); the per-dream loss maximum fell 0.029 → 0.017 → 0.009 and the step
cosine at lag 1 reads 1.00: the gradient has one direction and it is the tic. Nothing a later
cycle sees differs from this; the wake (sampled at 0.7 from an adapter that now answers every
document with the tic) cannot produce a different S_c. One decoded reply at cycle 9 (turn 0):
` Got it. Got it. Got it. Got it. Got it. Got it. Got it. Got it. …` (forced close at 128
tokens). One decoded dream at cycle 9 (dream 0, 30/30 identical): `<|endofconversation|> Got it.
Got it. Got it. Got it. …` (511 tokens, max-tokens stop).

**Acquisition census** (to-learn documents scoring at cycle 9 that scored 0 at every evaluation
before their wake, over `epoch-00` + cycles 0–9): **zero**. The only documents that ever scored
are the four §2.1 flicker documents, all before their wake (shown at cycles 42–71):
211 (1.0 at cycles 1–2), 254 (1.0 at 3, 5, 7), 355 (0.5 at most cycles), 357 (1.0 from cycle 4).
To-learn 0.002–0.004 throughout is these. Retention over cycles 0–9: 0.979 0.985 0.983 0.978
0.982 0.979 0.979 0.977 0.981 0.976 (within the warm start's cross-process spread; the collapse
of the generation does not reach the scorer's teacher-forced reading in ten steps).

## Closing section

**What ran.** Preflight (pin, manifest, frozen ×3, no leftovers, 6/6 tests) · three smokes
(fp32 adapters confirmed) · acceptance (max per-dream loss 0.626, mean 0.183: fix landed) ·
lora seed 42 30 epochs and mix-review seed 42 30 epochs in parallel under fp32 master weights ·
dream seed 42, stopped at cycle 9 of 100 on collapse. No report (dream cell incomplete). No
protected edits, no code changes. Session 14:46–~17:05 UTC (≈2.3 h of the 8 h cap).

**Readings for the next debrief (§3.3).**
1. Live reading at acceptance: max 0.626 / mean 0.183 (void cell cycle 0: 0.0034). In the cell:
   max 0.879, 0.391, 0.575, 0.303, 0.670, 0.466, 1.028, 0.029, 0.017, 0.009 (cycles 0–9).
2. Exposure: `subjects_mentioned` 1, 0, 1, 1, 1, 1, 1, 0, 0, 0; by text, one revisiting dream at
   cycle 0 (KLM, loss 0.649, the same document the fourth box's gate passed on), one at cycle 5
   (Peresson violins, 0.466, behind a `Got it.` opener). About one in thirty holds while dreams
   exist.
3. The collapse carrier is the **wake reply** ` Got it.`, not the frame (frame-anywhere 0–1 per
   cycle, frame-strict 0 throughout). Doubling per cycle: dreams containing `Got it` 1, 2, 4, 9,
   14, 27, 30 at cycles 0–6; wake replies become `Got it.` chains at cycle 6; total collapse at 7.
   This is the §3.1 limitation (the KL from the empty state carries the in-context reply as an
   unconditional prior) realised under the corrected mechanism, three cycles later than the third
   box's frame collapse and with a different carrier (the v4 warm start removed the literal frame
   from the corpus; the reply shape has no such mitigation).
4. Step readings: norms 0.467 → 0.209 over cycles 0–9 (void: 0.475 → ~0.2 at cycle 5, 0.119 at
   11), lag-1 cosines 0.82–1.00 (void: 0.62, 0.74), distance from A_0 2.475 at cycle 9 (void: 1.88
   at 10). A consistent direction from the first step. fp32 masters: `trainable.pt` 85 MB, dtype
   float32.
5. Epoch-1 comparison: not available (no cycle 99). Cycle 9 dream: 0.003 / 0.976 against lora
   epoch 1 0.020 / 0.903 and mix-review epoch 1 0.021 / 0.901; noise 0.006 / ≥0.014 (a third
   scoring mode of the warm start, 0.973, appeared in the acceptance process).
6. Baselines under fp32 vs the fourth box's bf16: lora peak 0.085 at 7 (0.486) vs 0.098 at 9
   (0.538), epoch 1 equal within flicker; mix-review peak 0.059 at 25 (0.475) vs 0.067 at 11
   (0.502), a slower rise and a lower, later peak. Both plateaus ≈0.05–0.065 at ≈0.48–0.51.
   The dtype change did not raise either baseline.

**What I would want to discuss.**
- The empty-state student with 30 dreams and one Adam step at lr 1e-4 moves the adapter 0.47 per
  cycle in a direction fixed by the wake's reply distribution; five identical sampled replies per
  wake (the coupled-seed observation, now ` Got it.` ×5) make that signal maximally consistent. Any
  mitigation list (revisit priors, assistant-only scoring, loss mask on the acknowledgement turn,
  smaller lr, fewer wake-shaped dreams) is §4 item 8's deferred discussion; this run now motivates
  it with a measured collapse rather than a void one.
- Whether cycles 10–99 of a collapsed cell had any value: I judged not (every dream identical, the
  gradient at the floor, the wake unable to recover) and stopped to save ≈2.2 h; if the next
  debrief wants the full 100-cycle control profile, the command is in this note and reproduces
  from cycle 0.
- The GPU-nondeterministic wake: acceptance and cell cycle 0 share seeds and adapter but sampled
  different replies, so cycle-0 readings are not reproducible to the dream; per-cycle readings
  should be treated as one draw.

## Shutdown (checklist), 16:56 UTC

1. lora `epoch-30/result.json` and mix-review `epoch-30/result.json` present; dream has no
   `epoch-01/` (stopped; cycles 0–9 under `work/epoch-01.nnmqucxb/`).
2. Closing section above.
3. Notes committed and pushed on `box/20261021-140642` (`2ef2502`, plus this receipt commit). No
   code changes; `sft/uv.lock` left modified by setup's re-resolve, uncommitted.
4. Pull receipt `2026-10-21T16:55:17 UTC`: notes 22717 B; `runs/lora-seed-42/summary.json`
   304161 B and 31 epoch `result.json`; `runs/mix-review-seed-42/summary.json` 303870 B and 31
   epoch `result.json`; dream `work/epoch-01.nnmqucxb/` 10 cycle `result.json`, `census.json`
   5481 B, `cycle-09/trainable.pt` 85373579 B, `cycle-09/dreams.json` 195786 B;
   `acceptance/dream-5/epoch-01/cycle-00/distil.json` 845 B; logs
   `lama-ckl-run-20261021-163654-1795999.log` 110725 B, `census-dream-seed-42-final.log` 1999 B.
5. Nothing UNRETRIEVED; no fallback commit.
6. `.watchdog-terminate` touched after this commit.
