# sft — LoRA fine-tuning for mamba2-780m

LoRA supervised fine-tuning of [`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) on conversational data. Produces a LoRA adapter that can be loaded into the `continual-learning` model as a better base.

Uses `mamba_ssm` directly (not HF peft/trl) to avoid Mamba-2 loading issues.

## Setup

```bash
make sync
```

Installs torch (ROCm/CUDA auto-detected) plus `causal-conv1d` and `mamba-ssm` from source.

Two overrides for machines where the defaults don't fit:

- `TORCH_BACKEND` (default `auto`) — uv's torch wheel selector. `auto` works on
  the ROCm dev box but guesses wrong on some CUDA hosts; GH200 needs
  `make sync TORCH_BACKEND=cu128`.
- `MAX_JOBS` (default: all cores) — parallel compile jobs for the from-source
  builds. Each job is RAM-heavy; cap it (`make sync MAX_JOBS=12`) if the
  compile OOMs on a low-memory box.

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

- [`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k) — long-document carrier data (single-exchange episodes: one long document turn and its answer, not multi-turn conversation), already in the `messages` shape `prepare_data.py` expects.
- [`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong) — synthetic needle-in-haystack recall QA (`prepare_babilong.py` converts its `{input, question, target}` schema into a synthetic one-turn `messages` conversation first). Mixed in deliberately: long natural text alone doesn't force a model to actually *use* far-back information, only babilong-style tasks do, since getting the answer right depends on it.

`merge_data.py` concatenates the two tokenized outputs into one `.pt` file, since `train.py` only accepts a single `--data` path.

`--max-len 100000` here is intentionally far above any real example (LongAlign-10k's longest is ~65k tokens) — it only controls what gets written to disk, which is nearly free. A too-tight `--max-len` truncates trailing turns, which can drop a conversation entirely if a long turn precedes the assistant turn (a `--max-len 16384` once dropped roughly half of LongAlign-10k this way). Bounding training-time RAM belongs in chunked/truncated-BPTT training (`Model.forward`'s `state` param — see `models/mamba2_2_7b_memory/README.md`), not in dropping data at prep time.

### Retention data: cram blocks, needles, and the solvability filter

The memory run's retention pressure comes from purpose-built slices, generated separately and consumed as separate `--data` artifacts (the file split *is* the slice tag). Design and rationale: `notes/DISCUSSION-20260724-next-run-plan.md` §1.3, `notes/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md` §6–7.

```bash
make prepare-cram               # data/train_cram.pt + data/eval_cram.pt
```

`prepare_cram.py` streams Wikipedia, runs NER over each passage, and emits alternating-turn cram blocks:

- `[USER]` carries a few fresh passages plus one *earlier* passage's sentence re-shown with its entity blanked (`____`, the only text this repo writes — everything else is dataset-authored); `[ASSISTANT]` is that sentence completed. The cue sits at a varied position inside the turn, with fresh passages after it, so "answer the last thing" is not a learnable policy.
- **Entity substitution**: the blanked entity is swapped for a same-type entity (PER/ORG/LOC) from another article at word boundaries, so the association exists only in this block's passages and cannot be recalled from pretrained weights.
- **Recall credit on the entity span only** — `recall_masks` (train.py's `--recall-weight`) is True on the entity tokens of the answer and nothing else. The rest of the completed sentence is copied from the visible cue and earns nothing.
- **Gap curriculum, encoded data-side**: each block's ceiling is `--ceiling-start * (--ceiling-end/--ceiling-start)^p` where `p` is how far through the item supply the block starts, and each item's gap is drawn log-uniform in `[--gap-min, ceiling]`. Blocks are emitted in ceiling order, so *consuming the artifact in order is the curriculum*. Every item also records `gap`/`target_gap`/`ceiling`, so a consumer that shuffles can restore it by sorting blocks on `ceiling`.
- Defaults `--gap-min 192` / `--ceiling-start 448` come from the measured SSM interference capacity (`notes/RESEARCH-20260724-local-diagnostics.md` §1: plain-backbone recall is dead by ~192 tokens of dense interference) and the 512-token BPTT window the trainer uses on cram slices.
- **Held-out articles** (`--heldout-frac`, default 5%) never appear in `train_cram.pt`; their blocks go to `data/eval_cram.pt` and the title list is stored in both files as `heldout_articles`.
- Masks are True on **every** token, carrier passages included (plan §1.4: weight 1.0 everywhere, no zero-on-carrier mask), including the role markers themselves — the run trains those two embedding rows and needs the model to learn to *emit* them.
- No EOS is emitted inside a cram block; a block is one continuous stream of turns, not a sequence of conversation ends, and `--eos-weight 32` would otherwise put large weight on a token appearing once per short turn.

NER runs through `transformers`' CoNLL-03 token classifier (`--ner-model`, default `dslim/bert-base-NER`), not spaCy as the plan proposed: spaCy publishes no wheels for this venv's Python 3.14, so it would mean a source build of thinc/blis plus the documented risk that any `uv` install clobbers the ROCm torch build. `transformers` is already a dependency and gives the PER/ORG/LOC types the same-type swap needs.

```bash
make prepare-needles            # data/train_needles.pt + data/eval_needles.pt
```

`prepare_needles.py` is the RMT-proven needle form under the identical curriculum and block assembly (it calls `prepare_cram.build_blocks`; only the items differ): the source is a bAbI story from `RMT-team/babilong`'s `0k` config — the task text with no filler, so the gap is ours to control rather than the benchmark's — the cue is the dataset's own question, and the credited answer is its own target. Wikipedia passages fill the gap.

Two deliberate differences from the cram slice, both forced by bAbI's six-name vocabulary:

- **One story per block** (`--max-items-per-block 1`). A second story in the same block re-states where the apple is, which silently invalidates the first question's target. This was visible only in the decoded sample; the counts were all correct.
- **`--allow-repeated-credit`**: a target like `kitchen` recurs across stories by construction, while the binding the question asks about does not, so the string-uniqueness rule that protects fabricated cram entities would discard the whole slice. Test B of the filter decides solvability here instead.

The filler pool wraps rather than running out, and both generators report how many times it was cycled — if that number is far above 1, stream more articles (`--articles`), because one needle per block at long gaps consumes a lot of interference.

```bash
MODEL_NAME=mamba2_780m make filter-items ARGS="--data data/train_cram.pt"
# writes data/train_cram-filtered.pt
```

`filter_items.py` decides which items keep their `recall_masks` credit. Each is scored teacher-forced on its credited span with the plain backbone (which *is* the M-ablated model) in three contexts:

| test | context | must |
|------|---------|------|
| A (well-posed) | source + cue | answer (`--a-min`, default −0.7) |
| B (memory-required) | interference + cue, source absent | fail (`--b-max`, default −1.5) |
| C (SSM-insufficient) | source + interference + cue, in-stream | fail (`--c-max`, default −1.5) |

plus `--min-margin` (default 1.5 nats/token): A − max(B, C) must clear it. The margin is what does the work where the absolute thresholds cannot — a common-word answer scores high in every context, so its *difference* is the signal. All four are mean log-probs per credited token, and all four are **guesses until calibrated**: run the filter on a pilot artifact, read the printed scored samples, then set them.

Before scoring, an explicit string check drops any item whose credited answer (or a word of it ≥4 characters, so a surname counts) is already visible in the interference before the cue — those cost no forward pass.

B and C are one pass over the block each, not one per item: the cues and answers already sit in the stream in order, so a single teacher-forced pass reads every item's span at its own position. B's stream is the block with every item's source cut out; C's is the block verbatim. Only A is per-item, and its context is short.

Discarded items **keep their tokens** — passages stay as carrier/interference, only the credit is dropped — so a high discard rate costs probes, not tokens. Oversample candidates rather than raising the slice's token share. The composition is the diagnostic: mostly `fail_a` means the cloze construction is bad, mostly `fail_b` means entity substitution isn't biting, mostly `fail_c` means the gaps are too short for the interference to defeat the SSM.

Every generator run ends with a structural validation block: counts whose correct value is zero (malformed role adjacency, credited span text mismatch, the credited entity visible between its source and its cue, credit outside a recorded span) plus decoded text around the source, the cue and the credited answer span of a sample item. The filter prints the same kind of evidence for what it scored. Read the sample before using the artifact — counts confirm the generator did what it was told, never that what it was told was right.

#### Artifact schema

All four artifacts (`train_cram.pt`, `eval_cram.pt`, `train_needles.pt`, `eval_needles.pt`) use `train.py`'s dataset schema — `ids`, `masks`, `recall_masks`, `sleep_positions` (empty: cram blocks carry no sleeps) — plus:

| key | meaning |
|-----|---------|
| `items` | per block, a list of item records: `gap`, `target_gap`, `ceiling`, `source_start/end`, `cue_start/end`, `answer_start`, `span_start/end`, `credit_text`, `entity`, `entity_type`, `article`, `original_entity`, and after filtering `filter` (`a`/`b`/`c`/`verdict`) |
| `curriculum` | `ceilings` (one per block, non-decreasing), `gap_min`, `ceiling_start`, `ceiling_end` |
| `heldout_articles` | titles reserved for eval, recorded in both the train and eval artifacts |
| `slice` | `cram`, `cram-heldout`, `needles`, `needles-heldout` |
| `filter` | after filtering: thresholds used, discard rate, per-test composition |

Everything is plain tensors, lists, dicts and primitives, so the files load under `weights_only=True`.

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

**Training on several dataset slices at once**: `--data` is repeatable, and each slice may carry comma-separated per-slice overrides:

```bash
make train ARGS="\
  --data 'data/train_chains.pt,share=35' \
  --data 'data/train_ballast.pt,share=15' \
  --data 'data/train_cram.pt,share=35,chunk-len=512,batch-size=6,grad-checkpoint=1,shuffle=0' \
  --data 'data/train_needles.pt,share=15,chunk-len=512,batch-size=6,grad-checkpoint=1,shuffle=0' \
  --eos-weight 32 --batch-size 24 --chunk-len 48 --memory-window 8 --accum-tokens 1536 \
  --recall-weight 16 --recall-ramp-steps 32"
```

| Spec option | Meaning |
|---|---|
| `share=N` | This slice's requested fraction of trained tokens. Any units — shares are normalised. Give it for every slice or none (then each slice's own token count is used, i.e. one evenly-interleaved pass over everything). |
| `chunk-len=N` | Overrides `--chunk-len` for this slice only. |
| `batch-size=N` | Overrides `--batch-size` for this slice only. |
| `grad-checkpoint=0\|1` | Gradient checkpointing for this slice, via the model's optional `set_grad_checkpoint` train hook (warns and no-ops for models that don't define one). |
| `shuffle=0\|1` | Default 1 (the historical per-epoch shuffle). `shuffle=0` consumes the artifact in the order it was written — the way a generator-side curriculum (e.g. a growing recall-gap ceiling) survives training. |

Slices are grouped by their `(chunk-len, batch-size, grad-checkpoint)` triple. Slices in the same group can share a batch, so they're merged into one example stream interleaved by token share, each slice keeping its own internal order. Slices in different groups can't share a batch at all, so their groups take turns: each turn trains for up to `--mix-segment-tokens` (default 1,000,000) and then goes to whichever group is furthest behind its share. A turn stops assigning new examples once it's over budget and drains the ones still running, so a handover costs a shrinking tail rather than a cut example — raise `--mix-segment-tokens` to pay that tail less often, lower it to mix more finely. With a single config group the budget is unbounded and the loop is exactly the single-dataset one. Per-group resume state (which group was training, each group's position and token count) is saved in `state.pt` for multi-config runs; the mix picks up where it left off rather than re-deriving from zero.

Why per-slice config exists: a recall target thousands of tokens after the write that should have stored it is causally disconnected from that write under truncated BPTT (state is detached at every chunk boundary), so retention-training slices need a chunk long enough — and the gradient checkpointing to afford it — while conversational and ballast slices don't and shouldn't pay for it. See `notes/DISCUSSION-20260724-next-run-plan.md` §2.1.

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

**Switching `--data` on resume**: a checkpoint's `slot_states`/`next_ptr` are indices into whatever dataset was in use when it was saved, alongside a fingerprint of that dataset (its resolved path + example count; one per slice for a multi-slice run). If `--resume` is given a `--data` set whose fingerprint doesn't match the checkpoint's, those indices are discarded and every slot is assigned fresh from the start of the new dataset instead of being silently reindexed into unrelated examples — model weights, optimizer state, and the cumulative token/step counters still carry over normally. If the checkpoint predates fingerprinting (no fingerprint saved at all), `train.py` prompts on the terminal asking whether `--data` is the same dataset the checkpoint was trained on, since that case is ambiguous rather than a clear mismatch; answering no (or running non-interactively, e.g. under a script with no stdin) discards the indices the same way. You don't need to do anything special to switch datasets on resume — this is automatic (interactive prompt aside).

**EOS under-generation**: if the model doesn't emit `<|endoftext|>` to end turns, pass `--eos-weight 5` (or higher) to upweight EOS positions in the loss. EOS tokens are ~0.8% of assistant tokens so they get little gradient by default.

**Per-token loss weighting**: two more weight knobs, defaulting to the values validated on the 2026-07-17 run (pass 1.0 to reproduce the unweighted objective). `--recall-weight N` (default 8) multiplies the loss on tokens marked in the dataset's optional `recall_masks` tensors (`prepare_chains.py`/`prepare_interference.py` emit them, marking exactly their spliced query-answer tokens) — those answers are ~0.1% of all tokens, so without upweighting the recall signal is heavily diluted; it warns and no-ops on datasets without `recall_masks`. That multiplier is **ramped**, not applied flat: it starts at `--recall-ramp-start` (default 1.0) and reaches `--recall-weight` over `--recall-ramp-steps` optimizer steps (default 32, `0` to disable), interpolating linearly or in log space (`--recall-ramp-shape linear|geometric`). Keep the ramp window aligned with the model's beta-anneal window (`BETA_BIAS_ANNEAL_STEPS`, also 32): that window is where the gradient decides whether the memory path is worth using or cheaper to suppress, and full-strength retention pressure landing inside it amplifies the loss spike rather than the signal (`notes/DISCUSSION-20260724-next-run-plan.md` §2.2). While ramping, the step line carries a `recall_w 4.75` field; it disappears once the multiplier reaches its target. `--head-weight N` (default 4) multiplies the loss at the first token after each backbone reset — the start of each example, and each fired sleep for datasets with `sleep_positions` — decaying linearly to 1.0 over `--head-tokens` (default 1024): most tokens sit deep inside long examples, so the empty-state regime is otherwise underweighted relative to how often generation actually starts there. Checkpoint cadence counts real tokens, unaffected by any of the weight knobs.

**Freezing the parametric path**: `--freeze-lora` optimizes only the memory subsystem (the `front_end` projections and the per-layer injection modules) while holding the LoRA weights fixed at their resumed values. It isolates whether the neural memory can carry recall on its own when the parametric (LoRA) path can no longer improve and re-absorb the task. Checkpoints stay complete — every parameter keeps `requires_grad=True` (so `save_checkpoint` still writes it), only the optimizer's parameter set shrinks — so probes and further resumes work unchanged.

**Sleeps (episodic chains)**: if the dataset carries `sleep_positions` (per-example token offsets, emitted by `prepare_chains.py`), the training loop wipes that slot's backbone state at each offset via the model's `sleep_slot` hook while the neural memory persists — recall across a sleep can then only flow through the memory (see `docs/superpowers/specs/2026-07-17-episodic-chains-design.md`). Sleeps snap to the next chunk boundary (< `--chunk-len` drift) and are not re-fired when resuming from a saved internal state. Ignored (with a warning) for models whose train hooks define no `sleep_slot`.

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

**`probe_recall.py`** (`make probe-recall`, output tee'd to `logs/probe-<timestamp>.log` alongside the training logs) — recall probe against the latest checkpoint (or a specific one via `--checkpoint <step dir>`, e.g. for sweeping a probe across several checkpoints): states labeled random digit codes in early turns, runs a stretch of filler turns (`--gaps`, default `1024` tokens), queries one code back by its label, and reports its mean per-token log-prob with the neural memory intact vs ablated (fresh random `M` swapped in at query time) plus a no-prefix floor. Mamba's own SSM state carries context too, so the intact−ablated delta is what isolates the Titans memory's specific contribution; a delta near 0 means the memory isn't functionally recalling, whatever the training-log write stats say. `--n-facts` (default `64,128`) sweeps interference — how many labeled codes each conversation must hold at once; a single fact sits comfortably in the SSM state, so sweep until ablated recall degrades to find where the memory has a real job. `--n-probes` (default 8) conversations run batched per configuration. `--ablation` picks the ablated-condition control: `fresh-m` (default, historical) swaps in a fresh random `M` but leaves the injection machinery firing on it; `none` disables the memory→backbone pathway entirely (no injections, no reads — the plain backbone). Probes of record run both; their disagreement measures how much the fresh-m control itself perturbs a memory-co-adapted checkpoint. `--sleep` adds the cross-sleep conditions: after the full prefix, the backbone state is wiped via `Model.sleep_slot` (the episodic-chains training regime's sleep) and the query runs in the fresh wake — `sleep-intact` (memory kept; recall can only flow through the memory, the direct measure of the episodic tier) vs `sleep-ablated` (memory also replaced; should sit at the floor). Needs ~17 GB VRAM headroom — pause training first if the training process has grown its allocator pool.

`--gist <data.pt>` switches the probe to a natural-continuation gist eval and replaces the engineered fact/query machinery entirely: each probe row takes one real long conversation from the given `prepare_data.py` output, places the sleep at the between-turn boundary nearest the conversation's middle, feeds the preceding `--gist-prefix` (6144) tokens as prefix, then teacher-forces the conversation's actual next `--gist-cont` (1536) tokens and reports mean log-prob per continuation token under four conditions: `no-wipe` (full carried state — positive control, must clearly beat every wiped condition), `no-wipe-ablated` (full carried state, memory replaced fresh — **awake-mem** = no-wipe − no-wipe-ablated is the memory's contribution while the SSM is alive, which should stay small as the sleep deltas grow; growth means the memory is shadowing the backbone's short-term role), `sleep-intact` (backbone wiped, memory kept), `sleep-recent` (memory rebuilt from only the last `--gist-recent` (576) prefix tokens, backbone wiped — distance-grades intact: parity with it means the memory is a last-few-turns buffer, not an episodic store), and `sleep-ablated` (backbone wiped and memory replaced fresh — the floor). The headline **gist-delta = sleep-intact − sleep-ablated** is the most permissive detector of the memory having stored *anything* about the pre-sleep text (topic, entities, style, facts all improve continuation prediction), where the exact-code probe only detects verbatim recall. Deltas are row-paired with SEM, and also broken out by distance into the continuation (thirds). `--gist-distractor N` (0 = off) adds two interleaved-episode conditions: after the wipe, N tokens of an unrelated conversation's opening are fed (writing into the memory), the backbone is wiped again, and the original continuation is scored (`dist-intact` / `dist-ablated`) — **dist-delta** measures how much prefix gist survives *through* an intervening episode and its sleep, and (gist-delta − dist-delta) is the flush cost of that episode boundary.

```bash
MODEL_NAME=mamba2_2_7b_memory make probe-recall ARGS="--gaps 1024 --n-facts 64,128,256"
```

**`probe_correction.py`** (`make probe-correction`, tee'd to `logs/probe-correction-<timestamp>.log`) — natural-language fact-correction probe: a short conversation asserts a stale fact ("The current president is Joe Biden."), the user corrects it, a filler exchange pushes the correction's window past a write boundary, the backbone is wiped (`sleep_slot`), and the question is re-asked. Candidate answers are scored teacher-forced (mean log-prob per answer token; free generation is deliberately not used — window-1 serving degenerates on question-shaped prompts) under three conditions: `no-sleep` (SSM intact, memory on — the in-context ceiling), `sleep-intact` (SSM wiped, memory kept — the memory test), and `sleep-none` (SSM wiped, injections disabled — the plain-backbone floor), plus a no-context parametric baseline. Scenario A corrects to the name Pile-era backbones already favor (Donald Trump — a degenerate anchor); scenario B corrects to a counterfactual name (Kamala Harris), so the signal is tracking-the-correction rather than matching-the-prior. The verdict is sleep-intact vs sleep-none on the corrected name. Turn formatting goes through `prepare_data.format_conversation`, so the probe cannot drift from the training format. `--checkpoint <step dir>` (default: latest), `--chunk-len` (default 24; 8 fits 2.7B on the 8 GB local card, but sustained 2.7B inference there trips the gfx1102 instability — treat 2.7B runs as a box job), `--memory-window` (default 8). Non-finite scoring passes retry up to 3× (unreliable-GPU insurance, free on healthy hardware).

`--no-filler` sleeps immediately after the correction instead. Run both arms: the filler is what pushes the correction past a write boundary, but it is also the likeliest thing to evict it under the delta rule, so filler-fails/no-filler-passes means eviction (a write-schedule problem) while both-fail means the memory is not carrying the fact at all.

```bash
MODEL_NAME=mamba2_780m_memory_mix make probe-correction ARGS="--checkpoint ../models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333"
MODEL_NAME=mamba2_780m_memory_mix make probe-correction ARGS="--checkpoint ../models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333 --no-filler"
```

**`read_diagnostic.py`** (`make read-diagnostic`, output tee'd to `logs/read-diag-<timestamp>.log`) — per-token diagnostic of `mamba2_2_7b_memory`'s read path on real sequences (`--ckpt` required; `--data`/`--examples`/`--tokens`/`--chunk-len` control scope — keep `--chunk-len 8` on the 8 GB local box, larger OOMs). Captures every token's `‖o_t‖` and surprise (both already computed per-token by the model) plus the residual stream at `--probe-layer` (default 21, the injection-free floor) and at `READ_LAYER`, then reports: gate-signal viability (percentiles/CV, per-window-position means, top-`‖o_t‖` tokens decoded in context) and how much of the front-end's `READ_LAYER` input is linearly recoverable from the probe layer (held-out ridge R², plus raw and mean-centered cosine similarity in q/k/v space against a shuffled control — the front-end is linear, so linear recoverability is exactly the bar for moving it down). `--save <path>` dumps the raw captures for offline analysis. Built for the stage-2 read-out investigation (`notes/DISCUSSION-20260722-stage2-readout.md`).

**`dream_fidelity.py`** (`make dream-fidelity`, output tee'd to `logs/dream-<timestamp>.log`) — contrastive generation probe for `mamba2_2_7b_memory`: primes the memory M on `--prime` real tokens, wipes the SSM (`sleep_slot`), then generates `--gen` tokens under M-primed vs a random-M control and reports how much each generation overlaps the priming text (Jaccard over content tokens, corpus-common tokens discounted) plus decoded samples. Tests whether M can *generate* a faithful dream of what it stored — the precondition for the M→weights consolidation idea (`notes/RESEARCH-20260722-memory-consolidation-landscape.md`); a positive primed−random overlap delta means M steers generation toward the stored content. Priming runs at the checkpoint's trained memory-window (fused kernel path on CUDA); generation drops to window 1 (single-token steps). Nucleus sampling via `--temperature`/`--top-p` (seeded by `--seed`). A box tool (~17 GB, like `probe_recall`) — the 2.7B plus the memory write's transient working set doesn't fit the 8 GB local card.

**`prepare_chains.py`** (`make prepare-chains`) — builds episodic-chain training data (default `data/train.pt` + `data/train_memory.pt` → `data/train_chains.pt`), the regime the three-tier memory design trains in (see `docs/superpowers/specs/2026-07-17-episodic-chains-design.md`). Each chain is one long example: shuffled whole conversations concatenated up to a sampled token budget (log-uniform `--min-budget`/`--max-budget`, 30k/130k), carrying `sleep_positions` — token offsets where train.py wipes the slot's backbone state while the neural memory persists. Sleeps land at randomly chosen between-episode boundaries (wakes of `--min-wake`..`--max-wake` episodes, 1–4, unpredictable placement), plus one mid-conversation sleep at a between-turn boundary in `--mid-sleep-rate` (0.2) of episodes at least `--mid-sleep-min-len` (4096) tokens long — the natural-continuation signal, where post-sleep tokens are predicted better iff the memory kept the gist. `--split-episode-rate` (0.0) additionally splits that fraction of eligible episodes at a turn boundary with at least `--split-min-part` (256) tokens on each side and resumes the tail `--split-gap-min`..`--split-gap-max` (2..2) episodes later behind a forced sleep — the same signal stretched across intervening episodes, training cross-episode gist retention (interleaved continuation). Single-QA episodes (exactly two turns) instead cut at the *question start* recorded by `prepare_data.py` (carried as `question_offsets`; `prepare_babilong.py` emits the question as its own field): the head keeps the document only, suspended behind a forced sleep, and the tail re-emits a fresh `[USER]` marker + `" "` before the dataset's own question and answer, moved verbatim — read the document now, get asked about it episodes later, with every content token dataset-authored. Episodes without a recorded question never split (fail closed — e.g. LongAlign, whose questions aren't mechanically extractable, stays whole as carrier data). `--split-qa-rate` sets the single-QA split rate separately (default: follows `--split-episode-rate`). Note the mid-conversation sleep only fires on episodes ≥ `--mid-sleep-min-len` with a middle-third turn boundary — LongAlign/babilong single-QA episodes never qualify, and ultrachat only does when tokenized uncapped (`prepare_data.py --max-len`, default 32768) — check the regen log's mid-conversation counter before assuming the signal exists. `--sentence-sleep-rate` (0.0) instead places that fraction of long episodes' sleeps at a *sentence* boundary in the middle third (a sentence-ending token followed by a whitespace-starting token, from a vocab scan) — reaches inside long single-QA document turns where no turn boundary exists, training reading-persistence across a sleep. A `--fact-rate` (0.3) fraction of episodes host a fact block (`--min-facts`..`--max-facts`, 4–64, log-uniform; same heterogeneous kinds, phrasings, and `--revise-rate` revisions as `prepare_interference.py`, keys unique per chain), and each block's queries are placed at one of three distances: within the same episode, in a later episode of the same wake, or beyond a sleep — the last answerable only through the neural memory, chosen uniformly among the distances available for each query unless `--cross-sleep-bias P` (default 0.0) forces the cross-sleep distance with probability `P` when it exists. Only cross-sleep queries require the memory, so raising this concentrates the memory-requiring training signal (uniform selection leaves it ~1/3 of queries). Emits `recall_masks` like `prepare_interference.py`.

**`prepare_interference.py`** (`make prepare-interference`) — splices interference-recall structure into an already-tokenized dataset (default `data/train_memory.pt` → `data/train_memory_v2.pt`): a fraction of examples (`--fraction`, default 0.3) get a block of key/value facts near the start — count log-uniform between `--min-facts`/`--max-facts` (8/256) — plus `--min-queries`..`--max-queries` (3–8) query/answer turn pairs at random later turn boundaries. Facts are heterogeneous (digit codes, keywords, colors, weekdays, counts, owners), each with several matched statement/question/answer phrasings, so the learnable invariant is key→value binding rather than one template; `--revise-rate` (default 0.12) of facts are later revised to a new value, with their queries placed after the revision expecting the newest value (trains overwrite-on-update). Everything is spliced at the token level (only the injected turns are tokenized, batched), so regeneration takes minutes. Fact keys come from a vocab slice disjoint from `probe_recall.py`'s, keeping the probe an honest held-out eval. Query answers are assistant turns with `True` masks — the recall training signal. The output also carries `recall_masks` marking exactly those spliced answer tokens, so training can amplify them with `--recall-weight` (see "Per-token loss weighting" above).

```bash
MODEL_NAME=mamba2_2_7b_memory make prepare-interference ARGS="--fraction 0.3"
make resume ARGS="--data data/train_memory_v2.pt ..."   # fingerprint change resets example pointers, keeps weights
```

**`--memory-window`** (default 1, i.e. today's exact per-token behavior): how many tokens' worth of signal `mamba2_2_7b_memory`'s memory subsystem consolidates into one gradient step *and* one `ssm_state` injection (a surprise-weighted pooling of the window's reads, not just the last token's), instead of taking one every single token. Must evenly divide the chunk length of every `--data` slice — a window can't span across the chunk boundary where BPTT gets truncated. No-op for `mamba2_780m` (or any model without a `set_memory_window` method). See `docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md` for the full design and what's still deferred. The memory-window mechanism is also what the backbone's fused-kernel training path (`Model._forward_fused`, feature-detected — CUDA with `causal-conv1d` installed, see the design spec and `models/mamba2_2_7b_memory/README.md`) dispatches around: only each window's closing token still runs the manual per-token path there. There's no principled default above 1 yet — sweep small values (e.g. on `mamba2_2_7b_memory`'s synthetic-data `make smoke-test`) before committing real training hours to one.

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
