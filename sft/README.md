# sft — LoRA fine-tuning for mamba2-780m

LoRA supervised fine-tuning of [`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) on conversational data. Produces a LoRA adapter that can be loaded into the `continual-learning` model as a better base.

Uses `mamba_ssm` directly (not HF peft/trl) to avoid Mamba-2 loading issues.

## Setup

```bash
make sync
```

Installs torch (ROCm/CUDA auto-detected) plus `causal-conv1d` and `mamba-ssm` from source.

## Data

The default dataset is [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) — the same dataset used by the original mamba-chat fine-tunes. Download and tokenize 20k examples with:

```bash
make data
# outputs data/train.pt
```

To use a local JSONL file instead:

```bash
make prepare ARGS="--input data/raw.jsonl"
```

Each line is a single JSON object with a `messages` list of `{"role", "content"}` turns. Loss is computed on assistant turns only; user tokens are masked out.

Add `"train": false` to an assistant turn to keep it in the context but exclude it from the loss — e.g. an earlier answer that a later turn revises, so the model learns the revision without learning the original mistake. The field is optional and defaults to `true`.

```json
{
  "messages": [
    { "role": "user", "content": "Tell me about the 2024 election." },
    { "role": "assistant", "train": false, "content": "Trump won, losing the popular vote to Biden." },
    { "role": "user", "content": "Actually he beat Kamala Harris." },
    { "role": "assistant", "content": "Trump won the 2024 election, beating Kamala Harris." }
  ]
}
```

(Shown formatted for readability; in the file each object must be on a single line.)

See `python prepare_data.py --help` for all options (`--max-examples`, `--hf-split`, `--max-len`, etc.).

### Long-context data (`mamba2_2_7b_memory`)

That model's whole point is long-range recall, so its training data needs long sessions, not the short ones above. `make data-memory` (set `MODEL_NAME=mamba2_2_7b_memory` first) builds and merges two sources:

```bash
MODEL_NAME=mamba2_2_7b_memory make data-memory
# outputs data/train_memory.pt
```

- [`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k) — real long multi-turn conversations, already in the `messages` shape `prepare_data.py` expects.
- [`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong) — synthetic needle-in-haystack recall QA (`prepare_babilong.py` converts its `{input, question, target}` schema into a synthetic one-turn `messages` conversation first). Mixed in deliberately: long natural text alone doesn't force a model to actually *use* far-back information, only babilong-style tasks do, since getting the answer right depends on it.

`merge_data.py` concatenates the two tokenized outputs into one `.pt` file, since `train.py` only accepts a single `--data` path.

`--max-len 100000` here is intentionally far above any real example (LongAlign-10k's longest is ~65k tokens) — it only controls what gets written to disk, which is nearly free. A too-tight `--max-len` truncates trailing turns, which can drop a conversation entirely if a long turn precedes the assistant turn (a `--max-len 16384` once dropped roughly half of LongAlign-10k this way). Bounding training-time RAM belongs in chunked/truncated-BPTT training (`Model.forward`'s `state` param — see `models/mamba2_2_7b_memory/README.md`), not in dropping data at prep time.

## Training

```bash
make train                      # start fresh
make resume                     # resume from latest checkpoint
make resume ARGS="--epochs 2"   # resume and train an extra epoch
make preflight                  # load the real model+data, run the preflight gradient check, exit
```

Pass any `train.py` flag through any of these targets with `ARGS="..."` (e.g. `make train ARGS="--eos-weight 10"`).

`make preflight` is worth running before a real training run, especially after touching a model's `train_hooks.py` or `model.py`: it loads the actual model and dataset (so it still pays for that, unlike the synthetic-model unit test in `models/tests/`) and runs one example through the same chunked forward+backward path training uses, asserting gradients actually reached every trainable parameter — then exits before the full loop. This is what would have caught this model's "the gated-delta merge never actually ran" and "k_proj/v_proj never received gradient" bugs immediately, instead of after a full run.

**Slot-based batching** (`--batch-size`, default 4): `train.py` runs `B` examples in parallel using a slot-based loop. Each slot independently advances through its own example; when a slot finishes, it resets and picks up the next example. All slots are processed in one batched forward+backward per chunk, so the GPU sees a `(B, chunk_len)` tensor every step rather than `(1, chunk_len)` — the primary mechanism for saturating GPU utilisation on stateful models like `mamba2_2_7b_memory`, whose per-token step prevents the parallel-scan kernel from helping. Gradient accumulation (`--accum-tokens`, default 256) is specified in real tokens per slot, not chunk count, so it means the same amount of real training regardless of `--chunk-len` (same reasoning as `--ckpt-every-tokens` being token-based). The optimizer step fires every `round(accum_tokens / chunk_len)` chunks, i.e. after roughly `accum_tokens × B` real tokens (minus any padding).

The live per-chunk progress display (in-place terminal output) shows one line per active slot:

```
[14:32:01]  avg_loss 2.34
  slot 0  token   3421/65000  beta 0.6369  clear 0.9821  active 21/21  surprise 0.1922  o_t_norm 1.0752
  slot 1  token    891/12400  beta 0.5821  clear 0.9734  active 19/21  surprise 0.2103  o_t_norm 0.9841
  slot 2  token  12004/98221  beta 0.7012  clear 0.9901  active 21/21  surprise 0.1654  o_t_norm 1.1203
  slot 3  token    203/8831   beta 0.6543  clear 0.9812  active 20/21  surprise 0.1877  o_t_norm 1.0341
```

Checkpoints are saved every `--ckpt-every-tokens` tokens of training (default 2000) to `../models/{MODEL_NAME}/checkpoints/epoch-E/step-N/` (every trainable parameter + optimizer state), where `E` is the 1-indexed epoch and `MODEL_NAME` is read from `.env`. Cadence is counted in cumulative tokens rather than optimizer steps, since examples vary enormously in length (a few thousand to ~100k tokens) and a step-based cadence would mean wildly different amounts of training between checkpoints. The trigger fires after every gradient-accumulation boundary, so a checkpoint can land *mid-example* for long examples; resuming one needs the carried internal state at that point (see below), and without it the affected slot's example just restarts from the beginning. Only the last 20 checkpoints **per epoch** are kept, tunable with `--ckpt-every-tokens` and `--keep-ckpts`.

For models whose `train_hooks.py` carries state across chunks via `init_state` (currently only `mamba2_2_7b_memory`), the last `--keep-full-state` checkpoints (default 5, 0 to disable) additionally save the full batched internal state to `mem_state.pt`. Resuming from one of those with an unchanged `--batch-size` continues every slot exactly where it left off. Otherwise (older checkpoint, pruned retention window, a different `--batch-size`, or a model without `init_state`) any mid-example slot restarts from position 0 — bounded duplicated training, not lost work. Replaying the prefix forward with the model's current (already further-trained) weights was tried and dropped: the weights have since moved, so it doesn't reconstruct the state that was actually live when the prefix was first trained, and batching replay across slots needing different amounts of it means padding some rows with synthetic all-zero tokens, which produced non-finite state (`beta`/`retain`/`surprise` → NaN) in practice.

**Switching `--data` on resume**: a checkpoint's `slot_states`/`next_ptr` are indices into whatever dataset was in use when it was saved, alongside a fingerprint of that dataset (its resolved path + example count). If `--resume` is given a `--data` file whose fingerprint doesn't match the checkpoint's, those indices are discarded and every slot is assigned fresh from the start of the new dataset instead of being silently reindexed into unrelated examples — model weights, optimizer state, and the cumulative token/step counters still carry over normally. If the checkpoint predates fingerprinting (no fingerprint saved at all), `train.py` prompts on the terminal asking whether `--data` is the same dataset the checkpoint was trained on, since that case is ambiguous rather than a clear mismatch; answering no (or running non-interactively, e.g. under a script with no stdin) discards the indices the same way. You don't need to do anything special to switch datasets on resume — this is automatic (interactive prompt aside).

**EOS under-generation**: if the model doesn't emit `<|endoftext|>` to end turns, pass `--eos-weight 5` (or higher) to upweight EOS positions in the loss. EOS tokens are ~0.8% of assistant tokens so they get little gradient by default.

**LR warmup** (`--warmup-steps`, default 32): linearly ramps the learning rate from 0 to `--lr` over the first N optimizer steps, then holds at `--lr` (0 to disable). Applying the full LR from step 1 against a freshly-attached, near-randomly-initialized subsystem (LoRA adapters, and for `mamba2_2_7b_memory` the gate projections waking up alongside the beta anneal) is a plausible source of oversized early gradients — a real run hit raw (pre-clip) gradient norms over 100x the clip ceiling in its first ~20 steps before this was added. The current LR is printed on every step line (`lr 1.23e-04`) so you can confirm the ramp is happening.

**Reproducibility** (`--seed`, default 42): seeds `torch.manual_seed` once at startup, covering everything not already covered by the per-epoch data-shuffle seed (which is separately, always seeded on the epoch number) — LoRA init/dropout, and for models with per-sequence random state (`mamba2_2_7b_memory`'s neural-memory init and per-slot reset) this is otherwise a real source of run-to-run variance: two runs of the identical command can produce different training-stability outcomes (e.g. one hitting non-finite losses the other doesn't) purely from a different random draw. Pass a different `--seed` to sample a different random init deliberately, e.g. when checking whether a crash is a real bug versus an unlucky draw.

**Localizing a non-finite gradient** (`make detect-anomaly` / `--detect-anomaly`): a chunk whose loss is finite but whose gradient isn't (confirmed on a real run: `gnorm nan`, which went on to permanently corrupt the trainable weights before this existed) is normally handled automatically -- the run discards just that chunk's gradient and keeps going, no intervention needed. If you actually need to find *which* operation produced it, `make detect-anomaly` enables `torch.autograd.set_detect_anomaly`, trading the normal resilient behavior for a hard stop: the first time a chunk's gradient comes back non-finite, PyTorch raises immediately with a full traceback to the exact forward op responsible, instead of discarding and continuing. This also slows the forward pass down (extra bookkeeping on every op), so it's meant for a short, dedicated diagnostic run to pin down a real recurring crash -- not something to leave on for a long unattended run, which should use plain `make train`/`make resume`. Always resumes from the latest checkpoint (falls back to a fresh start on its own if none exists yet) rather than starting over, since the point is almost always to reproduce a crash you already hit partway through a real run.

See `python train.py --help` for all options (learning rate, rank, accumulation steps, etc.).

`train.py` is generic across every model in `models/` — it dispatches to that model's `train_hooks.py` (`models/{name}/train_hooks.py`) for the only part that genuinely differs: how to load/wrap the model for training, and how to compute loss for one chunk. Everything else — chunk iteration, shuffling, accumulation counting, checkpoint cadence/rotation (including mid-example resume), evaluation, preflight, non-finite checks — is shared, since both models here are chunked, state-threaded ones (just with very different `--chunk-len`s).

### Training `mamba2_2_7b_memory`

This model's `train_hooks.py` differs from `mamba2_780m`'s in two ways, both visible in its module docstring: `Model.forward(input_ids, state)` is stateful, so `train.py` processes examples in `--chunk-len`-token chunks with `state` carried (and detached) across chunks of the *same* example — never across different examples — bounding training RAM by chunk length rather than example length (see the "Long-context data" section above for why examples themselves aren't truncated at prep time instead). And loss is computed over every token, not just assistant turns, since for this model the content worth exercising long-range recall on is mostly in the long user turns.

The default `--batch-size 4` and `--chunk-len 12` are tuned for this model on an H100 (80 GB); the 2.7B model's per-token fast-weight snapshot is ~210 MB, so a single chunk of length 12 with B=4 costs roughly 10 GB of backward graph on top of the 5–6 GB model weight floor. Benchmark with `make preflight` before raising either.

```bash
MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt"
MODEL_NAME=mamba2_2_7b_memory make resume ARGS="--data data/train_memory.pt"
```

Settings used for a real H100 run of `mamba2_2_7b_memory`:

```bash
make resume ARGS="--data data/train_memory.pt --eos-weight 32 --batch-size 12 --chunk-len 4 --accum-tokens 1024 --ckpt-every-tokens 24576"
```

**`measure_knobs.py`** — diagnostic for the memory subsystem's data-dependent write knobs (`eta`/`theta`/`alpha`): runs a checkpoint (`--ckpt models/.../step-N`) or a fresh init (no flag) over the first `--tokens` of a real example and prints the actual per-token knob distributions, `knob_proj` pre-activations, the residual magnitude feeding them, and the fast-weight rms trajectory. Answers "is the memory actually writing/retaining" directly — the thing to check first if `surprise` in the training logs sits flat at ~1.0 (the zeroed-memory value against unit-rms targets) or `w1_abs_max` trends toward zero. Run with the same env vars as the Makefile targets:

```bash
HF_HOME=../.cache/huggingface PYTHONPATH=.. HSA_ENABLE_INTERRUPT=1 uv run --no-sync python measure_knobs.py --ckpt ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-45
```

**`probe_recall.py`** (`make probe-recall`) — recall probe against the latest checkpoint: states labeled random digit codes in early turns, runs a stretch of filler turns (`--gaps`, default `1024` tokens), queries one code back by its label, and reports its mean per-token log-prob with the neural memory intact vs ablated (fresh random `M` swapped in at query time) plus a no-prefix floor. Mamba's own SSM state carries context too, so the intact−ablated delta is what isolates the Titans memory's specific contribution; a delta near 0 means the memory isn't functionally recalling, whatever the training-log write stats say. `--n-facts` (default `64,128`) sweeps interference — how many labeled codes each conversation must hold at once; a single fact sits comfortably in the SSM state, so sweep until ablated recall degrades to find where the memory has a real job. `--n-probes` (default 8) conversations run batched per configuration. Needs ~17 GB VRAM headroom — pause training first if the training process has grown its allocator pool.

```bash
MODEL_NAME=mamba2_2_7b_memory make probe-recall ARGS="--gaps 1024 --n-facts 64,128,256"
```

**`--memory-window`** (default 1, i.e. today's exact per-token behavior): how many tokens' worth of signal `mamba2_2_7b_memory`'s memory subsystem consolidates into one gradient step *and* one `ssm_state` injection (a surprise-weighted pooling of the window's reads, not just the last token's), instead of taking one every single token. Must evenly divide `--chunk-len` — a window can't span across the chunk boundary where BPTT gets truncated. No-op for `mamba2_780m` (or any model without a `set_memory_window` method). See `docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md` for the full design and what's still deferred. The memory-window mechanism is also what the backbone's fused-kernel training path (`Model._forward_fused`, feature-detected — CUDA with `causal-conv1d` installed, see the design spec and `models/mamba2_2_7b_memory/README.md`) dispatches around: only each window's closing token still runs the manual per-token path there. There's no principled default above 1 yet — sweep small values (e.g. on `mamba2_2_7b_memory`'s synthetic-data `make smoke-test`) before committing real training hours to one.

## Using the adapter

Each checkpoint directory contains `lora_config.json` with the rank and alpha used during training, so callers don't need to hard-code them, and `trainable.pt` with every trainable parameter (LoRA adapters, plus a model's own full-gradient subsystem if it has one — e.g. `mamba2_2_7b_memory`'s `front_end`/`injections`):

```python
import json
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from lora import apply_lora

ckpt = "../models/mamba2_780m/checkpoints/epoch-1/step-N"
cfg = json.loads(open(f"{ckpt}/lora_config.json").read())
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-780m", dtype=torch.bfloat16)
model = apply_lora(model, ["in_proj", "out_proj"], cfg["rank"], cfg["alpha"])
state = torch.load(f"{ckpt}/trainable.pt", weights_only=True)
model.load_state_dict(state, strict=False)
```

(The backend's loader, `backend/app/model/lora.py:load_checkpoint`, does the same thing — wrap the model first, then load `trainable.pt` into the wrapped model, not the raw backbone, so a model with extra trainable state outside the backbone still loads correctly.)
