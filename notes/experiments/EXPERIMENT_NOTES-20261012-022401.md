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

Probe wall time ≈ 7–12 min per checkpoint beside training (0.15 turn/s on the wake while the trainer holds the GPU), longer than the 7.5-min checkpoint interval, so the probe loop lags training and finishes after it. Monitor gotcha: `grep -c` without `-a` on these logs (they carry `\r` progress bytes) returns "Binary file matches", so an EXIT wake was missed once; the monitor uses `grep -a` from 03:03.
