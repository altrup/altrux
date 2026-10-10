# sft — Mamba2-2.7B training and continual-learning experiments

The active experiment is the [LAMA-CKL reproduction and Mamba comparison](#current-experiment-lama-ckl)
on the plain `mamba2_2_7b` model. The general SFT, memory-model, and earlier
dream-sleep sections remain as historical engineering reference; they are not
the current experiment path.

Uses `mamba_ssm` directly (not HF peft/trl) to avoid Mamba-2 loading issues.

## Setup

```bash
make sync
```

Installs torch and `mamba-ssm`. It installs `causal-conv1d` only on supported
CUDA systems.

Two overrides for machines where the defaults don't fit:

- `TORCH_BACKEND` (default `auto`) — uv's torch wheel selector. `auto` works on
  the ROCm dev box but guesses wrong on some CUDA hosts; GH200 needs
  `make sync TORCH_BACKEND=cu128`.
- `MAX_JOBS` (default: all cores) — parallel compile jobs for the from-source
  builds. Each job is RAM-heavy; cap it (`make sync MAX_JOBS=12`) if the
  compile OOMs on a low-memory box.

`make fmt` sorts imports and formats `sft/`, `models/`, and `scripts/` with
ruff (spaces, 100 columns); `make lint` checks without writing. `make hooks` installs a
pre-commit hook that runs `make lint`, so an unformatted commit fails until
you run `make fmt` and restage. `make test` runs the whole
suite; for one file use `uv run --no-sync pytest tests/<path>` with the same
`HF_HOME` and `PYTHONPATH` the Makefile sets.

## Source layout

Implementation is grouped by domain: `training/` owns datasets, checkpoints,
the training loop, and CLI setup; `preparation/` owns data builders;
`experiments/` owns facts, inference, locality, consolidation, erasure,
dream, and adaptive work; `diagnostics/`, `adapters/`, `providers/`, and
`reporting/` own their matching support code. The final split paths are
`training/{datasets,checkpoints,loop,cli}.py`,
`experiments/erasure/{operators,gating,wake_items,pilot,probe}.py`,
`experiments/dreams/{types,cache,generation,distillation,probes,runner,cli}.py`,
and `experiments/adaptive/{manifest,wake,coordinator,backend,analysis,runner,cli}.py`.

Make targets are the supported command interface. They run the domain CLI
modules through uv. Shell drivers live in `scripts/` and are also exposed by
Make targets. The root contains project files and the shared `progress.py`
module only; it has no Python command wrappers.

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

Every conversation is closed with the model's `EOC` marker (`<|endofconversation|>`) where the model package defines one, so the corpus teaches conversation-end as a token. `--pack` additionally packs 2–3 conversations into each example behind those boundaries (`DISCUSSION-20260808` §2.10.9): most boundaries are followed by an unrelated conversation, while `--recap-rate` (default 1/3) of them are followed instead by a **mechanical recap** — a short follow-up exchange quoting one user/assistant pair of the conversation just closed, built by string substitution with no model in the loop. Packing exclusively unrelated conversations would train the model to ignore its state at exactly the position every dream starts from; the recap fraction makes the post-boundary distribution "maybe new, maybe recall". `--seed` fixes the packing RNG. The run prints the packing invariants (examples not ending at a boundary, boundary count vs conversations packed, boundaries not opening a new turn — all must be 0) and decoded text either side of one boundary of each kind.

Run `make prepare ARGS="--help"` for all options (`--max-examples`, `--hf-split`, `--max-len`, etc.).

### Long-context data (`mamba2_2_7b_memory`)

That model's whole point is long-range recall, so its training data needs long sessions, not the short ones above. `make data-memory` (set `MODEL_NAME=mamba2_2_7b_memory` first) builds and merges two sources:

```bash
MODEL_NAME=mamba2_2_7b_memory make data-memory
# outputs data/train_memory.pt
```

- [`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k) — long-document carrier data (single-exchange episodes: one long document turn and its answer, not multi-turn conversation), already in the `messages` shape `preparation/conversations.py` expects.
- [`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong) — synthetic needle-in-haystack recall QA (`preparation/babilong.py` converts its `{input, question, target}` schema into a synthetic one-turn `messages` conversation first). Mixed in deliberately: long natural text alone doesn't force a model to actually *use* far-back information, only babilong-style tasks do, since getting the answer right depends on it.

`preparation/merge.py` concatenates the two tokenized outputs into one `.pt` file, since `training/cli.py` accepts one path per `--data` argument.

`--max-len 100000` here is intentionally far above any real example (LongAlign-10k's longest is ~65k tokens) — it only controls what gets written to disk, which is nearly free. A too-tight `--max-len` truncates trailing turns, which can drop a conversation entirely if a long turn precedes the assistant turn (a `--max-len 16384` once dropped roughly half of LongAlign-10k this way). Bounding training-time RAM belongs in chunked/truncated-BPTT training (`Model.forward`'s `state` param — see `models/mamba2_2_7b_memory/README.md`), not in dropping data at prep time.

### Retention data: cram blocks, needles, and the solvability filter

The memory run's retention pressure comes from purpose-built slices, generated separately and consumed as separate `--data` artifacts (the file split *is* the slice tag). Design and rationale: `notes/discussion/DISCUSSION-20260724-next-run-plan.md` §1.3, `notes/discussion/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md` §6–7.

```bash
make prepare-cram               # data/train_cram.pt + data/eval_cram.pt
```

`preparation/cram.py` streams Wikipedia, runs NER over each passage, and emits alternating-turn cram blocks:

- `[USER]` carries a few fresh passages plus one *earlier* passage's sentence re-shown with its entity blanked (`____`, the only text this repo writes — everything else is dataset-authored); `[ASSISTANT]` is that sentence completed. The cue sits at a varied position inside the turn, with fresh passages after it, so "answer the last thing" is not a learnable policy.
- **Entity substitution**: the blanked entity is swapped for a same-type entity (PER/ORG/LOC) from another article at word boundaries, so the association exists only in this block's passages and cannot be recalled from pretrained weights.
- **Recall credit on the entity span only** — `recall_masks` (`training/cli.py`'s `--recall-weight`) is True on the entity tokens of the answer and nothing else. The rest of the completed sentence is copied from the visible cue and earns nothing.
- **Gap curriculum, encoded data-side**: each block's ceiling is `--ceiling-start * (--ceiling-end/--ceiling-start)^p` where `p` is how far through the item supply the block starts, and each item's gap is drawn log-uniform in `[--gap-min, ceiling]`. Blocks are emitted in ceiling order, so *consuming the artifact in order is the curriculum* — pass these slices through `training/cli.py` with `shuffle=0` (see the multi-slice `--data` table under Training) or the per-epoch shuffle throws it away. Every item also records `gap`/`target_gap`/`ceiling`, so a consumer that shuffles anyway can restore it by sorting blocks on `ceiling`.
- Defaults `--gap-min 192` / `--ceiling-start 448` come from the measured SSM interference capacity (`notes/research/RESEARCH-20260724-local-diagnostics.md` §1: plain-backbone recall is dead by ~192 tokens of dense interference) and the 512-token BPTT window the trainer uses on cram slices.
- **Interference density, `--items-per-source-start` (3) / `--items-per-source-end` (1)**: one passage can host several items — several entities swapped, each with its own host sentence, no member's answer showing another member's entity — and the whole group shares one emitted passage. A block takes the first `k` items of each group, `k` interpolated linearly from the start value in the lowest-ceiling block to the end value in the highest, so short gaps are packed with cue/answer turns and fabricated entities while long ones get their interference from distance and volume. At 3 the measured yield is 1.69 items/passage, 251 → 191 tokens per item and +47% items off the same articles. The number is a *density* knob, not a quality one: the pilot's own scores show interference composition inside a ≤512-token gap does not predict the filter's test C (`notes/discussion/DISCUSSION-20260725-implementation-state-and-box-handoff.md` §3.1) — distance does.
- **Held-out articles** (`--heldout-frac`, default 5%) never appear in `train_cram.pt`; their blocks go to `data/eval_cram.pt` and the title list is stored in both files as `heldout_articles`.
- Masks are True on **every** token, carrier passages included (plan §1.4: weight 1.0 everywhere, no zero-on-carrier mask), including the role markers themselves — the run trains those two embedding rows and needs the model to learn to *emit* them.
- No EOS is emitted inside a cram block; a block is one continuous stream of turns, not a sequence of conversation ends, and `--eos-weight 32` would otherwise put large weight on a token appearing once per short turn.

NER runs through `transformers`' CoNLL-03 token classifier (`--ner-model`, default `dslim/bert-base-NER`), not spaCy as the plan proposed: spaCy publishes no wheels for this venv's Python 3.14, so it would mean a source build of thinc/blis plus the documented risk that any `uv` install clobbers the ROCm torch build. `transformers` is already a dependency and gives the PER/ORG/LOC types the same-type swap needs.

```bash
make prepare-needles            # data/train_needles.pt + data/eval_needles.pt
```

`preparation/needles.py` is the RMT-proven needle form under the identical curriculum and block assembly (it calls `preparation.cram.build_blocks`; only the items differ): the source is a bAbI story from `RMT-team/babilong`'s `0k` config — the task text with no filler, so the gap is ours to control rather than the benchmark's — the cue is the dataset's own question, and the credited answer is its own target. Wikipedia passages fill the gap.

Two deliberate differences from the cram slice, both forced by bAbI's six-name vocabulary:

- **One story per block** (`--max-items-per-block 1`). A second story in the same block re-states where the apple is, which silently invalidates the first question's target. This was visible only in the decoded sample; the counts were all correct.
- **`--allow-repeated-credit`**: a target like `kitchen` recurs across stories by construction, while the binding the question asks about does not, so the string-uniqueness rule that protects fabricated cram entities would discard the whole slice. Test B of the filter decides solvability here instead.

The filler pool wraps rather than running out, and both generators report how many times it was cycled — if that number is far above 1, stream more articles (`--articles`), because one needle per block at long gaps consumes a lot of interference.

```bash
MODEL_NAME=mamba2_780m make filter-items ARGS="--data data/train_cram.pt"
# writes data/train_cram-filtered.pt
```

`preparation/filtering.py` decides which items keep their `recall_masks` credit. Each is scored teacher-forced on its credited span with the plain backbone (which *is* the M-ablated model) in three contexts:

| test | context | must |
|------|---------|------|
| A (well-posed) | source + cue | answer (`--a-min`, default −2.0) |
| B (memory-required) | interference + cue, source absent | fail (`--b-max`, default −1.5) |
| C (SSM-insufficient) | source + interference + cue, in-stream | fail (`--c-max`, default −1.5) |

plus `--min-margin` (default 4.0 nats/token): A − B must clear it. The margin is what does the work where the absolute thresholds cannot — a common-word answer scores high in every context, so its *difference* is the signal. A's floor is deliberately loose: entity substitution makes the span implausible on purpose, so the backbone's prior fights the copy even with the source in view; demanding near-certainty there discards exactly the items substitution was built to create. The margin is over B only — C has its own bar, and a large required A − C gap would double-count it. All four are mean log-probs per credited token, calibrated on the 2026-07-25 pilot (401 items: strict-A defaults kept 26%; these keep 48% with every fail category doing its designed job). Re-calibrating is free: `--rescore` re-verdicts a scored artifact from its stored per-item scores without a model or GPU.

Before scoring, an explicit string check drops any item whose credited answer (or a word of it ≥4 characters, so a surname counts) is already visible in the interference before the cue — those cost no forward pass.

The A/B/C battery is a **cram-slice instrument**. Needle artifacts get `--leak-only` (string check gates credit, no scoring): bAbI items are well-posed and leak-proof by construction, and the base backbone cannot express their bare-entity answer format at all — the 2026-07-25 pilot scored A between −12 and −18 in *every* context, so the tests measure format competence rather than item quality there. If per-item SSM-solvability verification for needles ever matters, the recorded option is rescoring in a base-model-native `Q:/A:` prompt format.

B and C are one pass over the block each, not one per item: the cues and answers already sit in the stream in order, so a single teacher-forced pass reads every item's span at its own position. B's stream is the block with every item's source cut out; C's is the block verbatim. Only A is per-item, and its context is short.

All three passes run batched: blocks are independent streams and A's contexts are independent sequences, so `--score-batch` (default 16) of them go through one forward, right-padded to the batch's longest and grouped by length so the padding waste stays bounded. Right-padding is inert because the backbone's recurrence is causal — nothing fed after a row's own tokens can reach the log-probs at positions before them — and every read is asserted to sit inside its own row. Chunk boundaries (`--chunk-len`) are absolute positions shared by every row, so a batched score is the serial score to within batched-matmul noise; `--score-batch 1` is the serial path exactly. The batch costs one carried SSM state (~39 MB on the 780m backbone) plus one chunk of logits (~26 MB at `--chunk-len 256`) per row, so 16 is about 1 GB above the model — raise it on a large card.

Progress rides a single `\r` line carrying block count, running verdict composition, tokens/sec (real tokens, padding excluded) and an ETA, and every printed line is timestamped, so a log alone reconstructs the run's rate and any stall.

Discarded items **keep their tokens** — passages stay as carrier/interference, only the credit is dropped — so a high discard rate costs probes, not tokens. Oversample candidates rather than raising the slice's token share. The composition is the diagnostic: mostly `fail_a` means the cloze construction is bad, mostly `fail_b` means entity substitution isn't biting, mostly `fail_c` means the gaps are too short for the interference to defeat the SSM.

Every generator run ends with a structural validation block: counts whose correct value is zero (malformed role adjacency, credited span text mismatch, the credited entity visible between its source and its cue, credit outside a recorded span) plus decoded text around the source, the cue and the credited answer span of a sample item. The filter prints the same kind of evidence for what it scored. Read the sample before using the artifact — counts confirm the generator did what it was told, never that what it was told was right.

For the cross-slice version of that read — a few decoded windows and headline counts from *every* artifact about to be trained on — use `make sanity-sample ARGS="--data data/train_chains.pt --data data/train_cram.pt …"`. Windows center on recall-credited spans, then sleeps, then random text. It's the ten-second pre-training gate (the experimenter runs it on the box and hands the output to a cheap subagent), not a substitute for the generation-time validation.

#### Artifact schema

All four artifacts (`train_cram.pt`, `eval_cram.pt`, `train_needles.pt`, `eval_needles.pt`) use `training/loop.py`'s dataset schema — `ids`, `masks`, `recall_masks`, `sleep_positions` (empty: cram blocks carry no sleeps) — plus:

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

Pass any training flag through these targets with `ARGS="..."` (for example, `make train ARGS="--eos-weight 10"`).

`make preflight` is worth running before a real training run, especially after touching a model's `train_hooks.py` or `model.py`: it loads the actual model and dataset (so it still pays for that, unlike the synthetic-model unit test in `models/tests/`) and runs one example through the same chunked forward+backward path training uses, asserting gradients actually reached every trainable parameter — then exits before the full loop. This is what would have caught this model's "the gated-delta merge never actually ran" and "k_proj/v_proj never received gradient" bugs immediately, instead of after a full run.

**Slot-based batching** (`--batch-size`, default 4): `training/loop.py` runs `B` examples in parallel using a slot-based loop. Each slot independently advances through its own example; when a slot finishes, it resets and picks up the next example. All slots are processed in one batched forward+backward per chunk, so the GPU sees a `(B, chunk_len)` tensor every step rather than `(1, chunk_len)` — the primary mechanism for saturating GPU utilisation on stateful models like `mamba2_2_7b_memory`, whose per-token step prevents the parallel-scan kernel from helping. Gradient accumulation (`--accum-tokens`, default 256) is specified in real tokens per slot, not chunk count, so it means the same amount of real training regardless of `--chunk-len` (same reasoning as `--ckpt-every-tokens` being token-based). The optimizer step fires every `round(accum_tokens / chunk_len)` chunks, i.e. after roughly `accum_tokens × B` real tokens (minus any padding).

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

Why per-slice config exists: a recall target thousands of tokens after the write that should have stored it is causally disconnected from that write under truncated BPTT (state is detached at every chunk boundary), so retention-training slices need a chunk long enough — and the gradient checkpointing to afford it — while conversational and ballast slices don't and shouldn't pay for it. See `notes/discussion/DISCUSSION-20260724-next-run-plan.md` §2.1.

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

**Switching `--data` on resume**: a checkpoint's `slot_states`/`next_ptr` are indices into whatever dataset was in use when it was saved, alongside a fingerprint of that dataset (its resolved path + example count; one per slice for a multi-slice run). If `--resume` is given a `--data` set whose fingerprint doesn't match the checkpoint's, those indices are discarded and every slot is assigned fresh from the start of the new dataset instead of being silently reindexed into unrelated examples — model weights, optimizer state, and the cumulative token/step counters still carry over normally. If the checkpoint predates fingerprinting (no fingerprint saved at all), `training/cli.py` prompts on the terminal asking whether `--data` is the same dataset the checkpoint was trained on, since that case is ambiguous rather than a clear mismatch; answering no (or running non-interactively, e.g. under a script with no stdin) discards the indices the same way. You don't need to do anything special to switch datasets on resume — this is automatic (interactive prompt aside).

**EOS under-generation**: if the model doesn't emit `<|endoftext|>` to end turns, pass `--eos-weight 5` (or higher) to upweight EOS positions in the loss. EOS tokens are ~0.8% of assistant tokens so they get little gradient by default.

**Per-token loss weighting**: two more weight knobs, defaulting to the values validated on the 2026-07-17 run (pass 1.0 to reproduce the unweighted objective). `--recall-weight N` (default 8) multiplies the loss on tokens marked in the dataset's optional `recall_masks` tensors (`preparation/chains.py`/`preparation/interference.py` emit them, marking exactly their spliced query-answer tokens) — those answers are ~0.1% of all tokens, so without upweighting the recall signal is heavily diluted; it warns and no-ops on datasets without `recall_masks`. That multiplier is **ramped**, not applied flat: it starts at `--recall-ramp-start` (default 1.0) and reaches `--recall-weight` over `--recall-ramp-steps` optimizer steps (default 32, `0` to disable), interpolating linearly or in log space (`--recall-ramp-shape linear|geometric`). Keep the ramp window aligned with the model's beta-anneal window (`BETA_BIAS_ANNEAL_STEPS`, also 32): that window is where the gradient decides whether the memory path is worth using or cheaper to suppress, and full-strength retention pressure landing inside it amplifies the loss spike rather than the signal (`notes/discussion/DISCUSSION-20260724-next-run-plan.md` §2.2). While ramping, the step line carries a `recall_w 4.75` field; it disappears once the multiplier reaches its target. `--head-weight N` (default 4) multiplies the loss at the first token after each backbone reset — the start of each example, and each fired sleep for datasets with `sleep_positions` — decaying linearly to 1.0 over `--head-tokens` (default 1024): most tokens sit deep inside long examples, so the empty-state regime is otherwise underweighted relative to how often generation actually starts there. Checkpoint cadence counts real tokens, unaffected by any of the weight knobs.

**Freezing the parametric path**: `--freeze-lora` optimizes only the memory subsystem (the `front_end` projections and the per-layer injection modules) while holding the LoRA weights fixed at their resumed values. It isolates whether the neural memory can carry recall on its own when the parametric (LoRA) path can no longer improve and re-absorb the task. Checkpoints stay complete — every parameter keeps `requires_grad=True` (so `save_checkpoint` still writes it), only the optimizer's parameter set shrinks — so probes and further resumes work unchanged.

**Sleeps (episodic chains)**: if the dataset carries `sleep_positions` (per-example token offsets, emitted by `preparation/chains.py`), the training loop wipes that slot's backbone state at each offset via the model's `sleep_slot` hook while the neural memory persists — recall across a sleep can then only flow through the memory (see `docs/superpowers/specs/2026-07-17-episodic-chains-design.md`). Sleeps snap to the next chunk boundary (< `--chunk-len` drift) and are not re-fired when resuming from a saved internal state. Ignored (with a warning) for models whose train hooks define no `sleep_slot`.

**LR warmup** (`--warmup-steps`, default 32): linearly ramps the learning rate from 0 to `--lr` over the first N optimizer steps, then holds at `--lr` (0 to disable). Applying the full LR from step 1 against a freshly-attached, near-randomly-initialized subsystem (LoRA adapters, and for `mamba2_2_7b_memory` the gate projections waking up alongside the beta anneal) is a plausible source of oversized early gradients — a real run hit raw (pre-clip) gradient norms over 100x the clip ceiling in its first ~20 steps before this was added. The current LR is printed on every step line (`lr 1.23e-04`) so you can confirm the ramp is happening.

**Step budget** (`--max-steps`, default unset): stops the run after that many optimizer steps and writes the final checkpoint, regardless of how much of the dataset is left. It counts `global_step`, so a `--resume`d run finishes the same budget instead of taking that many more steps. Used by `make warm-start` together with `--epochs`.

**Reproducibility** (`--seed`, default 42): seeds `torch.manual_seed` once at startup, covering everything not already covered by the per-epoch data-shuffle seed (which is separately, always seeded on the epoch number) — LoRA init/dropout, and for models with per-sequence random state (`mamba2_2_7b_memory`'s neural-memory init and per-slot reset) this is otherwise a real source of run-to-run variance: two runs of the identical command can produce different training-stability outcomes (e.g. one hitting non-finite losses the other doesn't) purely from a different random draw. Pass a different `--seed` to sample a different random init deliberately, e.g. when checking whether a crash is a real bug versus an unlucky draw.

**Localizing a non-finite gradient** (`make detect-anomaly` / `--detect-anomaly`): a chunk whose loss is finite but whose gradient isn't (confirmed on a real run: `gnorm nan`, which went on to permanently corrupt the trainable weights before this existed) is normally handled automatically -- the run discards just that chunk's gradient and keeps going, no intervention needed. If you actually need to find *which* operation produced it, `make detect-anomaly` enables `torch.autograd.set_detect_anomaly`, trading the normal resilient behavior for a hard stop: the first time a chunk's gradient comes back non-finite, PyTorch raises immediately with a full traceback to the exact forward op responsible, instead of discarding and continuing. This also slows the forward pass down (extra bookkeeping on every op), so it's meant for a short, dedicated diagnostic run to pin down a real recurring crash -- not something to leave on for a long unattended run, which should use plain `make train`/`make resume`. Always resumes from the latest checkpoint (falls back to a fresh start on its own if none exists yet) rather than starting over, since the point is almost always to reproduce a crash you already hit partway through a real run.

Run `make train ARGS="--help"` for all options (learning rate, rank, accumulation steps, etc.).

`training/loop.py` is generic across every model in `models/` — it dispatches to that model's `train_hooks.py` (`models/{name}/train_hooks.py`) for the only part that genuinely differs: how to load/wrap the model for training, and how to compute loss for one chunk. Everything else — chunk iteration, shuffling, accumulation counting, checkpoint cadence/rotation (including mid-example resume), evaluation, preflight, non-finite checks — is shared, since both models here are chunked, state-threaded ones (just with very different `--chunk-len`s).

### Training `mamba2_2_7b_memory`

This model's `train_hooks.py` differs from `mamba2_780m`'s in two ways, both visible in its module docstring: `Model.forward(input_ids, state)` is stateful, so `training/loop.py` processes examples in `--chunk-len`-token chunks with `state` carried (and detached) across chunks of the *same* example — never across different examples — bounding training RAM by chunk length rather than example length (see the "Long-context data" section above for why examples themselves aren't truncated at prep time instead). And loss is computed over every token, not just assistant turns, since for this model the content worth exercising long-range recall on is mostly in the long user turns.

The default `--batch-size 4` and `--chunk-len 12` are tuned for this model on an H100 (80 GB); the 2.7B model's per-token fast-weight snapshot is ~210 MB, so a single chunk of length 12 with B=4 costs roughly 10 GB of backward graph on top of the 5–6 GB model weight floor. Benchmark with `make preflight` before raising either.

```bash
MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt"
MODEL_NAME=mamba2_2_7b_memory make resume ARGS="--data data/train_memory.pt"
```

Settings used for a real H100 run of `mamba2_2_7b_memory`:

```bash
make resume ARGS="--data data/train_memory.pt --eos-weight 32 --batch-size 12 --chunk-len 4 --accum-tokens 1024 --ckpt-every-tokens 24576"
```

**`make acceptance-check`** (implementation: `diagnostics/acceptance.py`) — the warm-start acceptance check of
`notes/discussion/DISCUSSION-20260808-headline-collapse-deep-block-and-regime-bridge.md`
§2.1, scored over a dream cache's free-running spans (`--cache
data/dream_set_s1234.pt`, single-dream or set). A warm start is accepted when
plain `]` tokens are under 35% of marker-slot emissions and no token is
non-ASCII; both clauses count **token ids**, because `[USER]` decodes
identically whether it is the registered special token or the ordinary tokens
that spell it, and separating those is the whole point. The steer prefix is
excluded (authored, not emitted), a cache with no marker-slot emission at all
fails rather than passing vacuously, and the exit status is the verdict:

```bash
HF_HOME=../.cache/huggingface PYTHONPATH=.. MODEL_NAME=mamba2_2_7b \
  make acceptance-check ARGS="--cache data/dream_set_s1234.pt"
```

**`make measure-knobs`** (implementation: `diagnostics/knobs.py`) — diagnostic for the memory subsystem's data-dependent write knobs (`eta`/`theta`/`alpha`): runs a checkpoint (`--ckpt models/.../step-N`) or a fresh init (no flag) over the first `--tokens` of a real example and prints the actual per-token knob distributions, `knob_proj` pre-activations, the residual magnitude feeding them, and the fast-weight rms trajectory. Answers "is the memory actually writing/retaining" directly — the thing to check first if `surprise` in the training logs sits flat at ~1.0 (the zeroed-memory value against unit-rms targets) or `w1_abs_max` trends toward zero.

```bash
HSA_ENABLE_INTERRUPT=1 make measure-knobs ARGS="--ckpt ../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-45"
```

**`diagnostics/recall.py`** (implementation: `diagnostics/recall.py`; `make probe-recall`, output tee'd to `logs/probe-<timestamp>.log` alongside the training logs) — recall probe against the latest checkpoint (or a specific one via `--checkpoint <step dir>`, e.g. for sweeping a probe across several checkpoints): states labeled random digit codes in early turns, runs a stretch of filler turns (`--gaps`, default `1024` tokens), queries one code back by its label, and reports its mean per-token log-prob with the neural memory intact vs ablated (fresh random `M` swapped in at query time) plus a no-prefix floor. Mamba's own SSM state carries context too, so the intact−ablated delta is what isolates the Titans memory's specific contribution; a delta near 0 means the memory isn't functionally recalling, whatever the training-log write stats say. `--n-facts` (default `64,128`) sweeps interference — how many labeled codes each conversation must hold at once; a single fact sits comfortably in the SSM state, so sweep until ablated recall degrades to find where the memory has a real job. `--n-probes` (default 8) conversations run batched per configuration. `--ablation` picks the ablated-condition control: `fresh-m` (default, historical) swaps in a fresh random `M` but leaves the injection machinery firing on it; `none` disables the memory→backbone pathway entirely (no injections, no reads — the plain backbone). Probes of record run both; their disagreement measures how much the fresh-m control itself perturbs a memory-co-adapted checkpoint. `--sleep` adds the cross-sleep conditions: after the full prefix, the backbone state is wiped via `Model.sleep_slot` (the episodic-chains training regime's sleep) and the query runs in the fresh wake — `sleep-intact` (memory kept; recall can only flow through the memory, the direct measure of the episodic tier) vs `sleep-ablated` (memory also replaced; should sit at the floor). Needs ~17 GB VRAM headroom — pause training first if the training process has grown its allocator pool.

`--gist <data.pt>` switches the probe to a natural-continuation gist eval and replaces the engineered fact/query machinery entirely: each probe row takes one real long conversation from the given `preparation/conversations.py` output, places the sleep at the between-turn boundary nearest the conversation's middle, feeds the preceding `--gist-prefix` (6144) tokens as prefix, then teacher-forces the conversation's actual next `--gist-cont` (1536) tokens and reports mean log-prob per continuation token under four conditions: `no-wipe` (full carried state — positive control, must clearly beat every wiped condition), `no-wipe-ablated` (full carried state, memory replaced fresh — **awake-mem** = no-wipe − no-wipe-ablated is the memory's contribution while the SSM is alive, which should stay small as the sleep deltas grow; growth means the memory is shadowing the backbone's short-term role), `sleep-intact` (backbone wiped, memory kept), `sleep-recent` (memory rebuilt from only the last `--gist-recent` (576) prefix tokens, backbone wiped — distance-grades intact: parity with it means the memory is a last-few-turns buffer, not an episodic store), and `sleep-ablated` (backbone wiped and memory replaced fresh — the floor). The headline **gist-delta = sleep-intact − sleep-ablated** is the most permissive detector of the memory having stored *anything* about the pre-sleep text (topic, entities, style, facts all improve continuation prediction), where the exact-code probe only detects verbatim recall. Deltas are row-paired with SEM, and also broken out by distance into the continuation (thirds). `--gist-distractor N` (0 = off) adds two interleaved-episode conditions: after the wipe, N tokens of an unrelated conversation's opening are fed (writing into the memory), the backbone is wiped again, and the original continuation is scored (`dist-intact` / `dist-ablated`) — **dist-delta** measures how much prefix gist survives *through* an intervening episode and its sleep, and (gist-delta − dist-delta) is the flush cost of that episode boundary.

```bash
MODEL_NAME=mamba2_2_7b_memory make probe-recall ARGS="--gaps 1024 --n-facts 64,128,256"
```

**`diagnostics/correction.py`** (`make probe-correction`, tee'd to `logs/probe-correction-<timestamp>.log`) — natural-language fact-correction probe: a short conversation asserts a stale fact ("The current president is Joe Biden."), the user corrects it, a filler exchange pushes the correction's window past a write boundary, the backbone is wiped (`sleep_slot`), and the question is re-asked. Candidate answers are scored teacher-forced (mean log-prob per answer token; free generation is deliberately not used — window-1 serving degenerates on question-shaped prompts) under three conditions: `no-sleep` (SSM intact, memory on — the in-context ceiling), `sleep-intact` (SSM wiped, memory kept — the memory test), and `sleep-none` (SSM wiped, injections disabled — the plain-backbone floor), plus a no-context parametric baseline. Scenario A corrects to the name Pile-era backbones already favor (Donald Trump — a degenerate anchor); scenario B corrects to a counterfactual name (Kamala Harris), so the signal is tracking-the-correction rather than matching-the-prior. The verdict is sleep-intact vs sleep-none on the corrected name. Turn formatting goes through `preparation.conversations.format_conversation`, so the probe cannot drift from the training format. `--checkpoint <step dir>` (default: latest), `--chunk-len` (default 24; 8 fits 2.7B on the 8 GB local card, but sustained 2.7B inference there trips the gfx1102 instability — treat 2.7B runs as a box job), `--memory-window` (default 8). Non-finite scoring passes retry up to 3× (unreliable-GPU insurance, free on healthy hardware).

`--no-filler` sleeps immediately after the correction instead. Run both arms: the filler is what pushes the correction past a write boundary, but it is also the likeliest thing to evict it under the delta rule, so filler-fails/no-filler-passes means eviction (a write-schedule problem) while both-fail means the memory is not carrying the fact at all.

```bash
MODEL_NAME=mamba2_780m_memory_mix make probe-correction ARGS="--checkpoint ../models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333"
MODEL_NAME=mamba2_780m_memory_mix make probe-correction ARGS="--checkpoint ../models/mamba2_780m_memory_mix/checkpoints/archive-bx1-mix16/step-333 --no-filler"
```

**`diagnostics/reads.py`** (`make read-diagnostic`, output tee'd to `logs/read-diag-<timestamp>.log`) — per-token diagnostic of `mamba2_2_7b_memory`'s read path on real sequences (`--ckpt` required; `--data`/`--examples`/`--tokens`/`--chunk-len` control scope — keep `--chunk-len 8` on the 8 GB local box, larger OOMs). Captures every token's `‖o_t‖` and surprise (both already computed per-token by the model) plus the residual stream at `--probe-layer` (default 21, the injection-free floor) and at `READ_LAYER`, then reports: gate-signal viability (percentiles/CV, per-window-position means, top-`‖o_t‖` tokens decoded in context) and how much of the front-end's `READ_LAYER` input is linearly recoverable from the probe layer (held-out ridge R², plus raw and mean-centered cosine similarity in q/k/v space against a shuffled control — the front-end is linear, so linear recoverability is exactly the bar for moving it down). `--save <path>` dumps the raw captures for offline analysis. Built for the stage-2 read-out investigation (`notes/discussion/DISCUSSION-20260722-stage2-readout.md`).

**`diagnostics/dream_fidelity.py`** (`make dream-fidelity`, output tee'd to `logs/dream-<timestamp>.log`) — contrastive generation probe for `mamba2_2_7b_memory`: primes the memory M on `--prime` real tokens, wipes the SSM (`sleep_slot`), then generates `--gen` tokens under M-primed vs a random-M control and reports how much each generation overlaps the priming text (Jaccard over content tokens, corpus-common tokens discounted) plus decoded samples. Tests whether M can *generate* a faithful dream of what it stored — the precondition for the M→weights consolidation idea (`notes/research/RESEARCH-20260722-memory-consolidation-landscape.md`); a positive primed−random overlap delta means M steers generation toward the stored content. Priming runs at the checkpoint's trained memory-window (fused kernel path on CUDA); generation drops to window 1 (single-token steps). Nucleus sampling via `--temperature`/`--top-p` (seeded by `--seed`). A box tool (~17 GB, like `diagnostics/recall.py`) — the 2.7B plus the memory write's transient working set doesn't fit the 8 GB local card.

**`experiments/consolidation/null.py`** (`make consolidation-null`, output tee'd to `logs/consolidation-null-<timestamp>.log`, per-fact records to `--out`) — the transcript-consolidation null: the field-default consolidation baseline that gates every downstream continual-learning arm (`notes/discussion/DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md` §4, `notes/discussion/DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md` §4). It builds a wake transcript of `--n-facts` (40) entity→digit-code facts separated by `--filler-tokens` (200) of digit-free filler, probes it three ways *before* any training (in-context positive control — must be high or the harness is broken; fresh-state floor — must be ~0; teacher-forced code log-prob from a fresh state), then runs a brief LoRA distillation: the teacher is the frozen model over `[transcript ‖ transcript-replay]` with logits taken on the replay half, the student is the LoRA model over the replay half alone from a fresh state, and the loss is KL(teacher ‖ student) at `--kl-temp`. One `--distill-steps` step is one `--chunk-len` chunk of the replay, with state carried and detached across chunks and reset at each pass boundary (truncated BPTT, as in `training/loop.py`); the teacher's logits are cached once before the first optimizer step, while the adapters are still identity, so a single model instance serves as both. Post-distillation it re-probes from a fresh state with no context: greedy exact match, pass@`--pass-k` at `--temperature`, a three-rung cue ladder (free recall → entity-category hint → first-digit hint, recording the rung that first succeeds), and the log-prob delta against the pre-distillation floor. Structural invariants and decoded samples of the transcript print before anything runs; per-fact results stream as they are computed. `--fresh-state-replay` resets the student's state before *every* replay chunk instead of only at pass boundaries: under the carried schedule the first chunk is the only one ever distilled from a fresh state — the very condition the post-distillation probes run under — and every run of this harness so far has installed the transcript's first fact and no other. On a memory model, `--no-memory` disables the neural-memory injection so the null measures distillation into the backbone alone (an untrained memory is not inert — see `experiments/consolidation/capacity.py` below), and the verdict record carries `memory_injection`.

Verdict semantics (thresholds are module-level constants, pre-registered before the first run): **PASS** if the post-distillation greedy exact-match rate ≥ 0.30; **FAIL-UNDERPOWERED** if below that but the mean code log-prob rose by ≥ 1.0 nat/token (the distribution moved, generation did not follow); **FAIL-DEAD** if neither. A box tool — it trains a LoRA and holds a full-vocab teacher-logit cache for the whole replay, so it runs on rented CUDA hardware, not the local card.

```bash
make consolidation-null ARGS="--n-facts 40 --distill-steps 200"
make consolidation-null ARGS="--checkpoint ../models/mamba2_780m/checkpoints/epoch-1/step-500"
```

**`experiments/consolidation/capacity.py`** (`make capacity-ladder`, output tee'd to `logs/capacity-ladder-<timestamp>.log`, per-fact records to `--out`) — measures the in-context positive control alone, over a grid of transcript shapes, to find where `experiments/consolidation/null.py` has a valid premise. The null's control asks whether the transcript is held losslessly *before* asking whether distillation can move it into weights; when that control fails, the null's verdict measures the wrong thing and the run has to move to a configuration where it holds. Each `--grid` cell is `n_facts`x`filler_tokens`: it builds the same transcript `experiments/consolidation/null.py` does, primes on it, and runs one greedy in-context probe per fact — no distillation and no LoRA training, so a cell costs a single forward pass. Alongside the per-cell hit rate it reports the **late-half hit rate** and the hit indices, which separate a recency-limited state (late facts survive, early ones fall out) from a capacity-limited one (uniform loss); the rate alone cannot distinguish them. The summary names the largest `n_facts` clearing the 0.8 control threshold, or reports that no cell on the grid did. Cells varying `n_facts` at fixed filler give the capacity curve; cells varying filler at fixed `n_facts` isolate distance from fact count. Note the null itself prints this same control as its own pre-phase, so pointing the null straight at a single candidate configuration is cheaper than a ladder when you only need to test one — the ladder earns its cost when you want the shape of the curve. On a memory model, `--no-memory` disables the neural-memory injection so the cell measures the backbone alone, and every record carries `memory_injection` saying which was measured; an untrained memory is **not** inert — `beta_anneal_offset` is 0.0 until training sets it, so the gated-delta merge writes an untrained random value into `ssm_state` at every window close — so a backbone-capacity reading needs the flag, and running both ways gives the memory's ablation delta.

```bash
make capacity-ladder
make capacity-ladder ARGS="--grid 4x200,8x200,40x40 --seed 1234"
MODEL_NAME=mamba2_2_7b_memory make capacity-ladder ARGS="--no-memory"
```

**`experiments/erasure/probe.py`** (implementation: `experiments/erasure/probe.py`; `make erase-probe`, output tee'd to `logs/erase-probe-<timestamp>.log`, records to `--out`) — the erase-efficacy probe for the dream-distillation sleep protocol (`notes/discussion/DISCUSSION-20260805-dream-distillation-cl-ab.md` §3): does the rank-1 state erase `S ← S(I − γ ĉĉᵀ)` make a primed SSM forget the *targeted* fact, and only it? Pure inference, no training — fits the local 8 GB box. It primes on the consolidation-null transcript (`--n-facts`/`--filler-tokens`/`--seed`, same generator), records baseline greedy recall + code log-prob per fact, then for each fact captures every layer's read query `C` while teacher-forcing that fact's probe answer (`--span answer` captures the answer-generating positions only; `question` includes the prompt), and for each `--gammas` value applies the erase to a copy of the primed state at all captured queries and re-probes **all** facts. Reports per-γ target vs off-target log-prob deltas and greedy flips, plus the pairwise mean-|cos| overlap of the facts' read queries — the direct measure of whether the state's key geometry makes the erase surgical or blunt. `--deflate` orthogonalizes each erase direction before applying it (queries share a ~0.9-cos interference cone, so a raw erase mostly attacks the shared component): `state-svd` deflates against the current state's top `--deflate-k` right-singular directions — computable by a streaming dream loop from `(S, ĉ)` alone — and `others` deflates against the other facts' mean queries, the oracle upper bound; near-cone directions (<5% of the query surviving deflation) are skipped rather than erased as noise. Three panels measure collateral *outside* the query cone: `--bystanders N` adds N off-format facts (three non-code relation templates — who lives where, when an event is, what an object weighs) to the transcript, probed at baseline and after every erase but never captured and never erased, with a bystander that neither answers correctly nor scores above −2 nats at baseline flagged `UNBOUND` and excluded from the collateral mean; `--filler-spans` scores the teacher-forced continuation of that many digit-free filler spans (15 tokens, prompted by the preceding 15) before and after each erase, the damage to non-fact state content; and every deflated erase logs the residual norm `‖(I−vvᵀ)ĉ‖/‖ĉ‖` and `|cos(ĉ, v)|` per position/layer plus the mean per-layer fraction of Frobenius mass the erase removed — a residual near zero means the direction actually applied is noise. Tail risk gets its own instruments: `--nearcone N` adds numeric-but-off-relation bystanders (spaced digits like a code, but "the parcel from X weighs" / "the invoice from Y totals" — the hard collateral case, since they compete inside the code cone), every collateral mean is printed alongside the single worst Δlogprob and the item/erased-fact that produced it, and read queries are captured for every bystander probe and for a fixed ~200-question address-space sweep battery (`--no-sweep` to skip) so the `|cos|` distribution against each erase direction is reported — unrelated, numeric, and code-fact-paraphrase questions scored as separate classes, raw and with the state's top singular direction deflated out. 780M-only (uses `Model.c_capture`; capture drives tokens one at a time, priming uses the normal fused-where-available dispatch).

```bash
make erase-probe
make erase-probe ARGS="--gammas 0.5,1.0 --span question --seed 2345"
```

**Warm start** (`make warm-start`, output tee'd to `logs/warm-start-<timestamp>.log`) — the format warm start every dream run loads, registered in `notes/discussion/DISCUSSION-20260808-headline-collapse-deep-block-and-regime-bridge.md` §2.9.3. The `[USER]`/`[ASSISTANT]` markers and `<|endofconversation|>` are special tokens the base backbone never saw, so a dream generated after the assistant marker runs off-distribution (mojibake, bracket mimicry). The target trains their embeddings (the marker delta) plus a LoRA on ordinary chat data, and is nothing but the two existing scripts chained: `preparation/conversations.py` renders 1000 ultrachat conversations (~4000 turns, `--max-len 4096`, `--pack`) into `data/warm_start.pt` through the same `format_conversation` the dreams see, then `training/cli.py` runs `--max-steps 800 --epochs 2 --lr 1e-4 --chunk-len 512 --lora-rank 16 --lora-alpha 32 --lora-dropout 0 --keep-ckpts 1` on it. Loss is on assistant turns only (`preparation/conversations.py`'s mask). `--keep-ckpts 1` leaves exactly one checkpoint dir, `models/<MODEL_NAME>/checkpoints/epoch-2/step-800`, which is what `experiments/dreams/cli.py --init-adapter` takes; `lambda_pull.sh` brings that dir home. (`DISCUSSION-20260808` §2.1 records 400 steps as an acceptance FAIL — 77% plain-`]` marker emissions — and calibrates the pass at 800.) Read `make sanity-sample ARGS="--data data/warm_start.pt"` before training on it. Box tool (trains a LoRA on the real backbone). Acceptance before any grid cell: per §2.1's token-level threshold — plain-`]` under 35% of marker-slot emissions counted at the token level, zero non-ASCII in a decoded dream, and (post-EOC) a decode conditioned on `<|endofconversation|>` alone opening a fresh well-formed `[USER]` conversation.

```bash
make warm-start                              # -> models/<MODEL_NAME>/checkpoints/epoch-2/step-800
```

**`make adaptive-multisleep`** (implementation: `experiments/adaptive/`) runs the registered six-wake comparison of replay, no-sleep, and sequential SFT over all three registered seeds; `--seed <registered-seed>` limits a hardware smoke or scheduled job to one seed. It loads three independent model/optimizer instances. Wake 1 and its exact saved recurrent state are shared; later wakes use separate resumable user-generator sessions and live local-model replies. Reply sampling is keyed by seed, arm, wake, and turn, so an interrupted wake produces the same remaining local tokens after resume. The wake artifacts keep the exact consumed prompt and assistant token IDs, including a generated assistant EOS, and SFT trains once on those IDs without decoding and tokenizing them again. A reply that reaches its token limit without EOS is refused instead of being joined to the next user turn. Replay generates and caches all 300 uncued dreams before one pass over each dream. No-sleep keeps the wake state without generation or training. SFT makes one raw-transcript pass and clears the state. Every sleep writes fresh-state fact/paraphrase probes, the calibrated battery, PPL damage (`ppl_delta`), the `R[evaluation wake][learning wake]` matrices, batch topology, throughput, VRAM, retries, provider provenance, and immutable wake/state/dream artifacts. The three-seed aggregate keeps each seed value and reports the mean and standard error for margins, exact match, paraphrases, battery loss/log-prob summaries, and PPL damage. Its secondary rehearsal-retention output keeps every earlier-fact observation, computes one mean per seed and rehearsal count, then reports uncertainty across those seed means. Use `--aggregate-only` to rebuild it from separately completed seed JSON files without loading a model.

The manifest has no scientific defaults. It must name exactly three distinct seeds and six wake objects, with unique fact entities and codes. Each wake must include its scenario, four `{entity, code}` facts, `turn_count`, four `injection_turns`, and one deterministic `turn_goals` entry per turn. The goal remains the intent spine on an injection turn; the fact instruction is added to it. It must also pin the generator command/provider/model/version; output root; dream/probe/battery batch sizes; and every runtime value: model, warm-start path and full SHA-256, LoRA rank/alpha, learning rate, chunk length, dream count (exactly 300), dream length/temperature, reply length/temperature, probe length, KL temperature, and battery artifact path. The loader refuses a missing wake schedule; do not use the proposed 12-turn, 3/6/9/12 schedule until it is registered.

`providers/user_generator.py` adapts an existing authenticated Claude Code or Codex CLI login to the manifest's JSON command contract. It uses native resumable sessions, disables Claude tools, runs Codex read-only in a stable inert directory, and needs no API key. Pin one of these command arrays in the manifest (replace the model and work directory explicitly):

```json
{"command": ["uv", "run", "--no-sync", "python", "-m", "providers.user_generator", "--provider", "codex", "--model", "gpt-5.4", "--work-dir", "/tmp/altrux-user-codex"]}
{"command": ["uv", "run", "--no-sync", "python", "-m", "providers.user_generator", "--provider", "claude", "--model", "claude-sonnet-4-6", "--work-dir", "/tmp/altrux-user-claude"]}
```

Each completed wake turn is persisted before the next user-generator call, so an interrupted wake resumes its native session and replays the exact local tokens. The mutable partial file is removed after the immutable wake artifact is stored. Completed seed outputs are reused. If a process stops during a wake or training cell, rerunning the seed starts from the pinned base model, replays completed wake artifacts and immutable dream caches, and resumes the partial provider session. Each attempt gets a new `seed-N.resume-K.jsonl`; prior logs are not changed. This is full-seed reconstruction, not an optimizer checkpoint inside the failed cell.

Run the CPU production-boundary smoke with `make adaptive-wake-smoke ARGS="--out /tmp/adaptive-smoke"`. It covers all 18 arm/wake cells and asserts the treatment counts without loading a model or provider.

**`make dream-sleep`** (implementation: `experiments/dreams/`, output tee'd to `logs/dream-sleep-<timestamp>.log`, per-fact records to `--out`) — the continual-learning A/B: is consolidating a session by *dreaming it back out of the state* better than conventional fine-tuning, and does it forget as much? Implements the arm sequences registered in `notes/discussion/DISCUSSION-20260806-dream-distillation-ab-postmortem.md` §3, which supersede the 08-05 file's. Wake is exactly stock — the erase fires only inside a sleep, at γ = 1.0 (no flag, since it is registered), along the direction `--erase-op` selects: `deflated` (default, the g2 behaviour) cuts along the query with the state's top singular direction deflated out (k = 1, §3a's probe result), `raw` cuts along the query itself. Which operator the B arms use is the open question `notes/discussion/DISCUSSION-20260807-g2-results-erase-geometry-and-warmstart-run.md` §3.4 registers as a paired B1-raw vs B1-deflated cell, so the chosen operator is stamped on every record the run writes. Wake material is the consolidation-null generator's transcript (`--n-facts`, default 4 — the measured binding ceiling — separated by `--filler-tokens`, default 40).

**One dream per seed, shared by every arm.** `--build-dream-cache` generates this seed's teacher dream once (from a copy of the wake state, intact, at the weights the student starts that sleep with — the warm-start checkpoint under `--init-adapter`, the base otherwise) and writes `data/dream_cache_s<seed>.pt` — wake transcript ids, wake state, dream token ids, per-position teacher logits and per-layer read queries, one distractor code per fact, SHA-256 hashes of both token sequences, and the `generator` that emitted it (the `--init-adapter` SHA-256 or `base`) — plus a decoded plain-text sidecar `data/dream_s<seed>.txt` with the cue spans bracketed. **Dreams are self-terminating**: nothing is masked out of the sampled distribution, so `<|endoftext|>` is ordinary dream-internal turn structure (it ends an assistant turn, not the dream) and the dream ends when the model emits `<|endofconversation|>`. `--dream-tokens` is a hard max, and a dream that never emits `<|endofconversation|>` is bounded by a turn backstop (`DREAM_MAX_TURNS`, counted in emitted `<|endoftext|>`s). The termination reason — `eoc`, `max-tokens` or `turn-backstop` — is recorded on the cache, printed in the sidecar header and stamped on the run's `phase: "cache"` record. Every other invocation *loads* that cache and refuses to run without it; no arm generates, and each result jsonl records both hashes so `reporting/grid.py` can assert they agree across every cell of a seed (§2's process rule: a registered invariant ships with its machine check — the 08-06 grid registered a shared dream in prose and each arm process silently regenerated its own). Rebuild a seed's cache if the wake transcript changes; the cache's own hashes are re-verified on load, so a corrupted artifact is refused rather than distilled.

**Rich wake transcripts** (`--wake-bystanders N --wake-nearcone N --wake-dialogue N`, all 0 by default — `DISCUSSION-20260808` §2.10.11). The A-vs-B4 contrast is entirely about *non-fact* state content, and 40 tokens of digit-free filler starves it, so the wake transcript can carry heterogeneous distractor content alongside the facts: off-format bystanders and near-cone numeric bystanders (`experiments/erasure/wake_items.py`) plus a slice of ordinary `HuggingFaceH4/ultrachat_200k` dialogue drawn from `test_sft`, the split the warm start never trains on. The items are shuffled in with the facts and separated by the same `--filler-tokens` filler.

Two guards ride with it. The **collision guard** is automated and refuses to build: the bystander pools are filtered against the fact names and every battery answer before sampling, dialogue candidates mentioning either are skipped (with the kept/skipped counts printed), and `assert_no_collisions` then re-checks every item and exits rather than build a transcript where probing a distractor would be probing something the run measures. The **context-leakage probe class** scores each distractor item with fresh-state QA — once as a floor before the sleep, once after — and emits `phase: "leak_floor"` / `phase: "leakage"` records. Neither arm should install any of it: a rising log-prob there is untargeted consolidation, measured at its origin. The transcript report prints the distractor invariants (items not stated exactly once, duplicate labels — both must be 0) and decoded text around the first item of each class.

**Many dreams per seed — the literature-shaped regime** (`--dreams N`, `DISCUSSION-20260808` §2.10.4). `--build-dream-cache --dreams N` generates **N** dreams upfront, each from a fresh copy of the *intact* wake state at the sleep-start snapshot's weights, and writes `data/dream_set_s<seed>.pt` plus a decoded sidecar `data/dream_set_s<seed>.txt`. An arm cell picks the set up automatically when one exists for the seed (which cache was chosen is printed); pass `--dream-cache` to override. The set file holds, per dream: tokens, per-position teacher logits, the state-dependency gate's decision (every position's divergence and the positions it kept), the raw per-layer read queries at those positions, each layer's singular spectrum with **both** rank rules' answers, and the per-dream eraser itself — one orthonormal basis per layer per variant. Shared across the set: the wake transcript and state, the foils, the steer prefix, the gate threshold and the rank rule. The **set hash** (SHA-256 over the dreams' own hashes, in order) is stamped on *every* record of every result jsonl, so no cell can silently distil a different collection. Each dream's hash is re-verified on load.

The cache build **gates rather than warns**: every fact must bind in at least `--bind-min-dreams` (2) dreams of the set or the build fails outright — the per-dream gate is retired for these caches, since coverage is aggregate and no dream is penalized for wandering off the facts. The build also prints, per the "sanity-check the artifact" rule: per-dream termination reasons, per-dream basis ranks, cross-dream V-overlap per variant (the price of per-dream erasers, measured rather than argued), the gate's precision/recall against the binding scan with **per-fact contribution counts** (a zero is called out), per-fact within-dream repeat counts (B4's re-installation window), and a decoded sample of dream 0's start with the prefix and the first free tokens marked.

**Training over a set** follows §2.10.4's carry matrix. Student *weights* carry across the whole set — one optimizer trajectory, since resetting them per dream would leave only dream N's learning. Student *state* resets at every dream boundary: arm A from blank, B4 from a fresh erased copy of the wake state (never erase-once-then-carry). Each dream is **one pass** (§2.10.2's retirement of the hundreds-of-passes regime). `--dream-epochs` (1) counts passes over the **set**, not over each dream — dream 1..N, then 1..N again, the interleaved replay-literature shape the variant cell exists to price; massing a dream's repeats back to back would rebuild the very regime §2.10.2 retires. `--distill-steps` does not apply — the budget is `N × --dream-epochs` optimizer steps. The full probe round runs at **every dream boundary in every epoch** (`--probe-every-dream`, 1; pass 2 for the registered degradation) — the per-dream install/damage curve is a primary deliverable and a later epoch's boundaries are exactly where redistribution would show, which is the §1.9 step-200 lesson. Each record carries its `epoch` and `dream` index. The harness rate-checks the probes: if they cost more than the training they measure, it says so and names the flag. A set cache runs arms `replay`, `b4-raw`, `b4-deflated`, `b4-qcm` only.

**The B4 arms** (`b4-raw`, `b4-deflated`, `b4-qcm` — §2.7): erase once per dream, from the frozen teacher's aggregate, then train on the dream as **ordinary sequence training** — full BPTT through the model's own sequence path, normal-training footprint, no spine and no `--cf-batch`. B1 = (erase every token, live student query); B4 = (erase once per dream, frozen teacher aggregate). Per dream, at cache-build time: (i) the **state-dependency gate** re-scores the dream's tokens under a blank state at the same weights and keeps the positions where the with-state and blank-state next-token distributions diverge by at least `--gate-threshold` nats (KL, with-state as the reference) — a memory read is a position where the state changed the prediction, defined by the state and not by any fact list, so generic reads drop out on their own and no ground-truth fact knowledge touches the mechanism path (§2.9.1; the binding scan survives only as a printed validation overlay). (ii) One SVD per layer over those positions' raw queries gives an orthonormal basis; both rank rules — largest ratio gap σᵢ/σᵢ₊₁, and σ > 4·median(σ) — are computed and printed for every layer, and `--rank-rule` picks which one truncates. Rank is capped by the **prod-side address budget**, a fixed 1/16 of the state's address dimensions, never the fact count. (iii) The dream's start state is projected **once**: `S ← S(I − VᵀV)`, a projection and never a sum of per-query cuts (which over-subtracts along the shared cone into a sign-flipped anti-memory) and never σ-scaled (partial cuts compound as (1−γ)ᴺ across N re-applications; a projection is idempotent, so the damage is independent of N). The three variants come from that one shared SVD: `raw` takes V as-is, `deflated` deflates each direction against the state's own top singular direction and re-orthonormalizes (the aggregate port of the operator that won the 08-08 picker), and `qcm` drops v₁, the query-consensus direction. Every final basis is asserted orthonormal at build time.

**The gate pilot's capture** (`--pilot-capture`, §2.10.7) — harness-only instrumentation on a `--build-dream-cache --dreams N` build: alongside the cache it writes `data/dream_set_s<seed>.pilot.pt`, holding **every** position's per-layer read queries (fp16) and D_t — not only the positions the build's own `--gate-threshold` kept — plus the wake state, the facts, the dream texts and the battery prompts' read queries (a forward pass, so it happens here rather than offline). That is enough to score every gating scheme §2.10.7 names *offline from one run* with `experiments/erasure/pilot.py`. The (T, V) with-state/blank-state logit *pairs* are deliberately not stored: ~100 MB per dream against ~6 MB of queries, and every registered scheme is a function of D_t, which is kept in full — trying a different divergence *measure* needs a fresh capture. The prod cache is byte-for-byte unchanged, and a build without the flag writes no capture.

**`make gate-pilot`** (implementation: `experiments/erasure/pilot.py`) scores that capture **offline**, on any machine, and is the tool §2.10.7's deliverable table comes out of: `make gate-pilot ARGS="data/dream_set_s1234.pilot.pt"`. *Test 1, separability*: D_t at the binding scan's fact positions against every other position, AUC per dream and pooled — below `--min-auc` (0.6) it prints the kill-condition verdict and **exits nonzero**, because a gate concept that cannot tell a fact read from a context read is not a threshold to be tuned. *Test 2, the bake-off*: `{hard, divergence-weighted, sqrt-capped, clip-capped}` × a τ grid taken from the capture's own pooled D_t quantiles (the τ that selects a weighted scheme's positions *is* its floor — one constant, not two). Every scheme builds each dream's per-layer V through the same primitives prod uses (`experiments/erasure/gating.py`, `experiments.dreams.cache.dream_bases`), with the weights scaling the query rows entering the SVD, and is scored on **§2.10.6's decision metric**: target removal (readout removed along the oracle fact-read queries) against collateral removal (along context reads and the battery items' queries), measured by applying V to the actual wake state. The table — scheme × {AUC, precision/recall, oracle-subspace overlap, cross-dream stability, target, collateral, gated positions, rank span with the second rank rule shown where the two disagree} — streams row by row as it computes and is written next to the capture as `.gate_pilot.txt` and `.gate_pilot.jsonl`. `--variant` picks which post-processing of the shared SVD is scored. The tool **recommends** by §2.10.6's lexicographic procedure (mechanical sanity, then dominance, then the highest target removal below the knee — taken as the median — of this pilot's own collateral distribution, then simplicity), and says in the output that the freeze itself is a human decision: the experimenter proposes, the team ratifies. Precision/recall and oracle overlap are printed as **diagnostics only** (§2.10.6 names label accuracy a non-goal), and no scheme is ever scored by a downstream training outcome.

**The steer prefix** is the existing `--dream-prompt`. Its tokens are excluded from the keep/score mask in **every** arm — they condition the dream through state only — with the last prefix position kept, since it predicts the first free token, exactly as cue spans are masked. The prefix text and its token count are recorded on the cache and printed in the sidecar header.

**`--init-adapter <ckpt-dir>` warm-starts every arm from the same checkpoint.** It takes a `training/loop.py` checkpoint directory (`models/<MODEL_NAME>/checkpoints/epoch-2/step-800`, what `make warm-start` writes): the run asserts the dir's `lora_config.json` rank/alpha against this cell's `--lora-rank`/`--lora-alpha` — fatal on a mismatch rather than a silent partial load — and loads `trainable.pt` through `training.checkpoints.load_checkpoint`, i.e. the LoRA factors plus the marker-embedding delta. The load happens **before the dream cache is built, before the knowledge battery is built, and before any training**, so the cached dream, the self-calibrated battery and every arm all descend from the warm-started weights — the dream is *generated* with the adapter active, and loading a cache whose `generator` disagrees with the cell's adapter is fatal. Recalibrate the battery for a new warm start by deleting `data/knowledge_battery_<model>.json` and letting the first `--init-adapter` invocation rebuild it. `trainable.pt`'s SHA-256 and the checkpoint dir's name land on every record, and `reporting/grid.py` refuses to pool cells whose hashes disagree (including a warm-started cell beside a cold one); a grid with no `--init-adapter` anywhere pools as before.

`--arm` selects the sleep. `replay` (A) teacher-forces the student over the cached dream from a **fresh** state with KL to the cached logits and carries nothing; the registered form makes the chunk the whole dream (one optimizer step per pass), so nothing carries across chunk boundaries and no position is privileged — pass `--chunk-len 48` for the 08-06 bridge cell, where `--fresh-state-replay` resets before every chunk instead of only at pass boundaries. `drain` (B1) and `counterfactual` (B2) both teacher-force the same cached dream from a copy of the cached wake state, ablating **per layer, interleaved inside the forward** via `Model.erase_hook`: at each layer the token's own read query C becomes the ablation direction (deflated against that layer's state under `--erase-op deflated`), the carried past is ablated along it, and only then does the token's decay+write+read happen — so the logit trained on always comes from the ablated version of the state that generated that token, addressed by the *student's* query rather than a cached teacher one. The gradient runs **through** that direction's query path (§3.4: the cut follows the query, so the read gradient cannot point at dodging via remnants); only the deflation's protected subspace is stop-gradiented, at its source, so no arm — however deep its BPTT — can rotate the guard onto the fact. B1 ablates in place and carries its own detached forward; B2 ablates a copy, trains on it, discards it, and carries the intact state. They are identical at token 1 (state, query, ablation, logit and gradient) and diverge only through the carry. The carry is detached between tokens unless `--deep` (B2 only) accumulates the pass's losses for one full-BPTT optimizer step — deep-B1 is not offered, since its cross-token gradient runs through every intervening ablation projector and is pre-aimed at the deflation-protected subspace (§6). `--ce-on-dream` runs A's sequence with cross-entropy on the dream tokens instead of KL, isolating objective from data; `--sft-ref` is the CE-on-raw-text convention on the wake transcript; `--no-sleep` is the floor. `--distill-steps` is the optimizer budget, but it is **not** a cross-arm currency — one A step is a whole chunk and one B step is one token — so every cell logs `token_gradients` and the frontier is reported on both axes. `drain-live` (retired, §6) is kept for reference.

**The fused B arms** (`b2-fused-detached`, `b2-fused-deep`, `b3-fused` — `notes/discussion/DISCUSSION-20260807-g2-results-erase-geometry-and-warmstart-run.md` §3.5). B2's defining property, that nothing counterfactual carries, is what makes the per-token loop unnecessary: the whole dream can be one optimizer step. Per pass, all at that pass's fixed weights — (i) the *intact* spine is run from a copy of the wake state and every token's per-layer carried state is materialized, (ii) all dream positions are batched, each taking its own copy of those states, ablating along the student's own query for that token (same `Model.erase_hook` micro-order and same `--erase-op` semantics as the per-token arms), writing the token and reading, and (iii) the KL against the cached teacher logits at every kept position is summed into **one optimizer step per pass** — arm A's currency, so `--distill-steps` counts passes here, not tokens. The spine is materialized in two phases so its sequential depth is ~√T rather than T: whole-block forwards (`--spine-block`, 32 — the fused SSD chunk-scan wherever that kernel is usable) fix the block boundaries, then every block steps through its own tokens simultaneously as one batch. `--cf-batch` (128) is the counterfactual micro-batch, a pure memory knob — micro-batches accumulate into the same single step, and the tests assert a micro-batched pass equals an unbatched one.

`b2-fused-detached` recomputes the spine every pass and detaches it, so nothing counterfactual carries and the spine drifts pass to pass with the weights. `b2-fused-deep` keeps the spine's graph — full BPTT through the scan; `v`, the deflation basis, is still computed under no-grad at its source, which the tests assert *through this path* rather than only at the source. Depth is measured here and not on B1, whose cross-token gradient composes ablation projectors (§6); B2's ablations are leaves of the graph and never composed. `b3-fused` replaces the recomputed spine with the **dream generator's own** trajectory — materialized once at the start of each sleep from the weights that generated that sleep's dream, and constant across passes, so depth is vacuous for it and it isolates within-sleep spine drift. Because at pass 1 the student's weights *are* the generator snapshot, `b3-fused` must reproduce `b2-fused-detached` exactly there: the arm checks it itself, at every sleep's pass 1, and writes a `phase: "equivalence"` record (both losses, their difference, the verdict) to the results jsonl. All three carry the intact wake state, like B2.

**Cues.** Free generation rehearses by luck (distinct-code coverage 4/4, 0/4, 0/4 across three seeds at the best fixed temperature and length), so `--cue-every N` forces each fact's own wake-session question stem into the dream every N tokens, cycling the facts, and `--cue-greedy K` (12) decodes the K tokens after each cue greedily — the cue names which fact to recall, the state still supplies the digits. A fired cue timer waits for the next `.`/newline (up to 20 tokens) before splicing, so a cue never cuts a thought in half, and cue tokens are **loss-masked as targets** in both KL and CE: positions whose next-token target is cue text drop out of the sum, the cue stays in context, and the first post-cue answer digit is not masked. A fully-masked chunk takes no optimizer step. Because the timer restarts from the end of each spliced cue, a sizeable share of a "512-token dream" is cue text — every cell logs its actual free-generation token count. The dream seed is the assistant marker plus a literal space, per the trained chat format.

## Current experiment: LAMA-CKL

The external benchmark gate runs the authors' code at TAALM commit
`b12f344a9dbae555c239635b1c192c555bed001b`; Altrux does not copy or modify
that unlicensed trainer or evaluator. The released run used eight GPUs with
microbatch 8. The registered GH200 adapter changes only the pinned launcher's
GPU visibility and accumulation count: one GPU, microbatch 8, accumulation 8,
and the same effective batch 64. DDP numerics can differ, so this is a
single-GH200 batch-equivalent replication, not an exact hardware reproduction.
From the repository root, prepare its isolated environment and results path:

```bash
git clone https://github.com/ybseo-ac/TAALM.git .cache/TAALM
git -C .cache/TAALM checkout --detach b12f344a9dbae555c239635b1c192c555bed001b
uv venv --python 3.10 .cache/TAALM/.venv
UV_TORCH_BACKEND=cu128 uv pip install \
  --python .cache/TAALM/.venv/bin/python \
  -r .cache/TAALM/requirements.txt
sudo install -d -o "$(id -u)" -g "$(id -g)" /results/lamackl
cd sft
```

Keep the pinned dependency versions. Stop if they do not install on ARM64;
do not substitute newer packages during the registered gate. Then run:

```bash
make lama-ckl-upstream-check
PATH="$PWD/../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-smoke
PATH="$PWD/../.cache/TAALM/.venv/bin:$PATH" make lama-ckl-upstream-run
make lama-ckl-upstream-summarize RESULT=/results/lamackl/finetune_qlora.pkl
```

The check verifies the four released files by row count, structure, decoded
sample, and SHA-256. The smoke uses the first 64 verified rows, accumulation 8,
and one epoch, so it performs one optimizer update before the paid full gate.
The summary streams every full-run epoch and requires the published
Llama-2-7B QLoRA result within the frozen gate: peak TO-LEARN accuracy
`0.115 ± 0.02`, first peak at epoch `16 ± 2`, and the paired NOT-TO-FORGET
accuracy `0.8174 ± 0.02`. After the gate passes, the summary target archives
the result at `.cache/lama_ckl/upstream/finetune_qlora.pkl`. This gate must pass
before the Mamba comparison runs.

After that gate passes, unpack the official LAMA download so the supplied path
contains `relations.jsonl` and `TREx/`, then build the model-conditioned Mamba
split on the rented CUDA GPU:

```bash
mkdir -p ../.cache/LAMA
wget -O ../.cache/LAMA/data.zip https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip ../.cache/LAMA/data.zip -d ../.cache/LAMA   # unpacks to ../.cache/LAMA/data/
rm ../.cache/LAMA/data.zip
make lama-ckl-split ARGS="--model-name mamba2_2_7b --lama-root ../.cache/LAMA/data"
```

This loads the pinned recap-0.5 warm start, scores the descriptive and schematic
tasks from fresh state in GPU batches, applies the released zero/one selection
rules and seed 42, and writes an immutable 500/500 artifact under
`.cache/lama_ckl/mamba2_2_7b_recap050/`. The Mamba tokenizer encodes standalone
objects differently from sentence-internal objects, so this cross-backbone
port locates the last object character span and scores all overlapping tokens.
The manifest records that adaptation, the source-tree hash, warm-start hash,
model IDs, settings, artifact hashes, zero-valued invariants, and decoded
samples.

Run the engineering gate for each arm before a full cell. It uses two documents,
two 32-token dreams where applicable, and one cycle, and writes to a separate
`-smoke` directory:

```bash
make lama-ckl-smoke ARGS="--model-name mamba2_2_7b --arm frozen --seed 42"
make lama-ckl-smoke ARGS="--model-name mamba2_2_7b --arm lora --seed 42"
make lama-ckl-smoke ARGS="--model-name mamba2_2_7b --arm mix-review --seed 42"
make lama-ckl-smoke ARGS="--model-name mamba2_2_7b --arm altrux --seed 42"
```

After the smoke fixes batch sizes for that rented machine, run the registered
cells serially. Seeds `42`, `43`, and `44` are the frozen Mamba replication
seeds; the split itself stays the one seed-42 artifact:

```bash
for seed in 42 43 44; do
  for arm in frozen lora mix-review altrux; do
    make lama-ckl-run ARGS="--model-name mamba2_2_7b --arm $arm \
      --seed $seed --dream-batch-size 8 --eval-batch-size 16"
  done
done
```

Each full cell runs 30 cycles. Every wake has 500 `[USER]` evidence turns and
greedy `[ASSISTANT]` replies with a 64-token backstop (a reply that reaches it
is closed with a fed EOS; `wake.json` flags each such turn as `eos_forced` and
the cycle log prints the count), followed by one `<|endofconversation|>`. Native LoRA trains one fixed seed-42 pass over the 500
evidence documents per cycle. Mix-Review pairs that pass with the 500 retention
documents in the official fixed seed-0 review order. Altrux generates 300
uncued 512-token dreams at temperature `0.7` from the intact post-wake state and
distils one pass at KL temperature `1.0`. All trainable arms use AdamW at
`1e-4`; evaluation uses the published descriptive object-token accuracy from
fresh state after every cycle.

Runs resume from the last atomically completed `cycle-NN/`. Each cycle stores
the exact wake, adapter and optimizer, carried state, metric vectors, resource
counts, hashes, and decoded samples. Altrux stores exact dream token IDs, text,
seeds, stop reasons, diagnostics, teacher adapter hash, and set hash. Its
full-vocabulary teacher logits exist only in memory through that cycle's
distillation; the saved teacher adapter and tokens can reconstruct them. Dream
diagnostics never select, regenerate, stop, or tune a dream.

After all 12 cells finish, aggregate the official checkpoint metrics, full
per-cycle curves, acquisition, forgetting, source, wake, review, generated and
training tokens, all persistent artifact bytes, wall time, GPU-hours, peak
VRAM, parameter counts, LoRA configuration, and GPU identity:

```bash
make lama-ckl-report
```

The report ignores adjacent smoke directories and rejects incomplete cycle
curves, mixed split hashes, evaluation or dream batch sizes, other shared
settings, duplicate arm/seed cells, or a missing registered cell. Use
`ARGS="--allow-incomplete"` only for an interim engineering report.

**Probes.** The primary reliability metric is the **distractor-code margin** (§4): the summed log-prob of the correct code minus that of a fixed random foil code, same question, fresh state — immune to the format prior the cues inject and to the digit-counting attractor that broke greedy exact match. A fact counts installed at margin ≥ 1.0 nat; the margin, both raw sums and the verdict are logged per probe point. Greedy exact match and the four-paraphrase generality battery are reported and never gate. The full battery — margin, paraphrases, knowledge battery and held-out ΔPPL from `experiments/locality.py` — streams every `--probe-every` steps (200), so every cell yields a learned-vs-forgotten *curve* and iso-learning comparisons are read off curves rather than engineered with hyperparameters. Dream rehearsal is **binding-aware**: a code counts only where it appears in the same sentence as its own entity, and codes sitting next to a different fact's entity are reported separately as misbindings. The carried-state column stays a **diagnostic, never scored as installation**. Box tool (trains a LoRA, holds a full-vocab dream logit cache); 780M-only, since the erase is addressed through `Model.c_capture` and applied through `Model.erase_hook`.

**Multi-sleep** (`--waves K`, the registered shape is `--waves 4 --n-facts 4` — `notes/discussion/DISCUSSION-20260807-g2-results-erase-geometry-and-warmstart-run.md` §3.7). Each wave wakes on the state the last one carried, sleeps, and is probed on **every fact so far** plus the battery; probes are fresh-state only, since state continuity across sleeps is load-bearing for the B arms. Wave 1 distils the seed's shared cache; **every later sleep generates its own dream** from the state it carried in, cued on that wave's facts only — spontaneous rehearsal of earlier waves is then a measured observable rather than a cue artifact. `--wave-teacher` is mandatory above one wave and names the generator: `current` is the registered protocol (the student as of that sleep's start), `base` is the one-seed drift-contribution control, pinned at the frozen base at every sleep. Each generated dream writes a `phase: "cache"` record with its hash, its cue coverage and its generator, so the wave-≥2 dreams are recorded rather than asserted equal — they legitimately differ per arm, which is the object of the comparison. Later waves get foils of their own, drawn clear of every code already in play.

`counterfactual-commit` (B2′) is the multi-sleep-only arm: B2's training, then at sleep end a fresh-state margin check per fact and one **real** erase along that fact's own query wherever it passes — the erase as verified memory policy, not as training signal. A fact commits once; the state it hands the next wake is the intact one minus what was proved installed.

**Reporting.** After every sleep the run streams column *j* of the **R-matrix** — one `phase: "r_matrix"` record per (arm, sleep, wave) with that wave's mean margin, install count and fact count — so the log says what each earlier wave's facts are worth now, read at any point. At the end it prints the matrix and emits a `phase: "cl_summary"` record: **BWT** (the mean, over waves taught before the last sleep, of how far their margin moved between the sleep that taught them and the final one — negative is forgetting) and **cumulative installation** as what survives the final sleep beside the peak any sleep reached, whose gap is a fact installed and then destroyed.

```bash
make dream-sleep ARGS="--build-dream-cache --seed 1234 --n-facts 4 --dream-tokens 512 --dream-temp 0.7 --cue-every 32"
make dream-sleep ARGS="--arm counterfactual --seed 1234 --distill-steps 800 --probe-every 200 --out logs/g2_B2_s1234.jsonl"
make dream-sleep ARGS="--arm replay --ce-on-dream --seed 1234 --distill-steps 800"
make dream-sleep ARGS="--arm b2-fused-detached --seed 1234 --distill-steps 800 --cf-batch 128"
make dream-sleep ARGS="--arm b3-fused --seed 1234 --distill-steps 800 --erase-op raw"
make dream-sleep ARGS="--build-dream-cache --dreams 8 --seed 1234 --dream-tokens 512 --dream-temp 0.7 --gate-threshold 1.0"
make dream-sleep ARGS="--build-dream-cache --dreams 16 --seed 1234 --pilot-capture"   # + the gate-pilot capture
make dream-sleep ARGS="--arm b4-deflated --seed 1234 --out logs/g3_b4_deflated_s1234.jsonl"
make dream-sleep ARGS="--arm replay --seed 1234 --dream-epochs 3 --probe-every-dream 2"
make dream-sleep ARGS="--arm counterfactual-commit --waves 4 --wave-teacher current --seed 1234 --distill-steps 800 --out logs/ms_B2p_s1234.jsonl"
make dream-sleep ARGS="--arm replay --waves 4 --wave-teacher base --seed 1234 --distill-steps 800 --out logs/ms_A_basecontrol_s1234.jsonl"
INIT_ADAPTER=../models/mamba2_780m/checkpoints/epoch-2/step-800 ERASE_OP=deflated make grid2 ARGS=1234   # the d800 baselines for one seed
INIT_ADAPTER=../models/mamba2_780m/checkpoints/epoch-2/step-800 make ladder ARGS=3200                    # one saturation rung
make gate-pilot ARGS="data/dream_set_s1234.pilot.pt"   # the sec 2.10.7 scheme table
make summarize-grid    # scores logs/g2_*_s*.jsonl
make summarize-grid ARGS="--curves 'logs/g2_*_s1234.jsonl'"   # per-probe-step curves
```

**`make grid2` / `make ladder`** — the box-side drivers, one seed (resp. one rung) per invocation. Both **refuse to start without `INIT_ADAPTER`**: the warm start is registered for the whole session — cache, battery and every arm — so a cold grid has to be asked for by name with `INIT_ADAPTER=none`. `ERASE_OP` (default `deflated`) picks the erase operator and is echoed at start; the erase-family cells carry it in their filename, so the operator picker can run raw beside deflated at the same seed. `CUE_EVERY` (default 32) rebuilds an under-binding seed's cache at 24. The grid emits the `nosleep` floor cell every Δ metric is read against; the ladder does not and pools against the grid's.

**`make summarize-grid`** (implementation: `reporting/grid.py`) — scores the grid: one row per cell (binding-aware coverage, misbindings, `erase_op`, margin installs, greedy EM, Δlogp, paraphrase rate, `token_gradients`, battery losses, ΔPPL) and a pooled-by-arm block. `--curves` prints instead one line per cell per probe step — floor-corrected Δmargin, Δ-installs and ΔPPL — which is what the §3.4 operator picker's iso-learning rule reads (damage at *matched* Δmargin; endpoints alone cannot answer it). Each step is corrected against the seed's no-sleep cell at the same probe step, falling back to that cell's final margin where the steps don't line up. A cell whose records disagree on `erase_op`, or that contains a failed B3-fused pass-1 equivalence record, is fatal — the runtime check warns and continues so a grid cell isn't killed mid-run, and the summarizer is where it becomes a refusal to score. Cells with *different* `erase_op`s pool fine; the picker grid runs raw beside deflated on purpose. It reads the last probe point of each cell, and it **exits nonzero before printing anything if the wave-1 transcript or dream hashes disagree within a seed** — cells that did not share a dream are not comparable, and the check is scoped to wave 1 because wave-≥2 dreams legitimately differ per arm in the multi-sleep grid. The same check applies to `--init-adapter`: one warm-start SHA-256 across the whole grid, or nothing is pooled. **Multi-dream cells** (§2.10.4) score alongside single-dream ones: the `set_sha` joins the per-seed hash check (cells that distilled different dream sets refuse to pool), the `bound` column becomes **aggregate** coverage — facts bound in at least one dream of the set — `misb` totals the set's misbindings, and `rehrs` prints `-`, since a set has no single-dream rehearsal fraction. A cell's **seed is its `_s<digits>` token and everything else is its arm**, so `lad_A_s1234_d3200` is arm `lad_A_d3200` at seed 1234: reading the seed as `1234_d3200` matched no floor cell, which is why the 08-08 ladder's Δ columns came out empty and were computed by hand.

**`experiments/locality.py`** (implementation: `experiments/locality.py`; used by `experiments/dreams/cli.py`) — the locality half of that battery. The knowledge battery is **self-calibrated**: candidate simple completions are run through the base model greedily and only its own hits are kept, so a lost item is the model forgetting something it demonstrably knew rather than a question it never could answer. The kept items and their baseline answer log-probs are cached to `data/knowledge_battery_<model>.json` (override with `--battery`) and rebuilt only if that file is missing, so every arm and seed is scored on identical items; re-scoring reports correct→incorrect flips plus per-item log-prob drops as the sensitive measure. `HELDOUT_TEXT` is the fixed general-text slice behind the ΔPPL backstop — never trained on, never generated from, so a perplexity delta is attributable to the sleep alone. It also holds the installation side's arithmetic: `logprob_sum` (the **summed**, not averaged, teacher-forced log-prob of a code) and `code_margin`, whose `MARGIN_INSTALL = 1.0` nat threshold was chosen before the data.

**`preparation/chains.py`** (`make prepare-chains`) — builds episodic-chain training data (default `data/train.pt` + `data/train_memory.pt` → `data/train_chains.pt`), the regime the three-tier memory design trains in (see `docs/superpowers/specs/2026-07-17-episodic-chains-design.md`). Each chain is one long example: shuffled whole conversations concatenated up to a sampled token budget (log-uniform `--min-budget`/`--max-budget`, 30k/130k), carrying `sleep_positions` — token offsets where training/loop.py wipes the slot's backbone state while the neural memory persists. Only a `--sleep-chain-rate` (0.1) fraction of chains carries any of the engineered apparatus described below (sleeps, interleaved continuations, fact blocks); the rest are plain multi-episode concatenations with silent joins, because chains are the deployment shape and retention pressure belongs to the cram slices (`notes/discussion/DISCUSSION-20260724-next-run-plan.md` §1.3) — pass `1.0` for the fully-engineered dataset (byte-identical to pre-gate output at a given seed). The regen log reports the realized sleeping-chain fraction. Within a sleeping chain, sleeps land at randomly chosen between-episode boundaries (wakes of `--min-wake`..`--max-wake` episodes, 1–4, unpredictable placement), plus one mid-conversation sleep at a between-turn boundary in `--mid-sleep-rate` (0.2) of episodes at least `--mid-sleep-min-len` (4096) tokens long — the natural-continuation signal, where post-sleep tokens are predicted better iff the memory kept the gist. `--split-episode-rate` (0.0) additionally splits that fraction of eligible episodes at a turn boundary with at least `--split-min-part` (256) tokens on each side and resumes the tail `--split-gap-min`..`--split-gap-max` (2..2) episodes later behind a forced sleep — the same signal stretched across intervening episodes, training cross-episode gist retention (interleaved continuation). Single-QA episodes (exactly two turns) instead cut at the *question start* recorded by `preparation/conversations.py` (carried as `question_offsets`; `preparation/babilong.py` emits the question as its own field): the head keeps the document only, suspended behind a forced sleep, and the tail re-emits a fresh `[USER]` marker + `" "` before the dataset's own question and answer, moved verbatim — read the document now, get asked about it episodes later, with every content token dataset-authored. Episodes without a recorded question never split (fail closed — e.g. LongAlign, whose questions aren't mechanically extractable, stays whole as carrier data). `--split-qa-rate` sets the single-QA split rate separately (default: follows `--split-episode-rate`). On the single-QA path `--split-min-part` constrains the head (the retained document) only — a QA tail is the dataset's own question and answer, about ten tokens in babilong, so requiring the same length there would disqualify every episode. Note the mid-conversation sleep only fires on episodes ≥ `--mid-sleep-min-len` with a middle-third turn boundary — LongAlign/babilong single-QA episodes never qualify, and ultrachat only does when tokenized uncapped (`preparation/conversations.py --max-len`, default 32768) — check the regen log's mid-conversation counter before assuming the signal exists. `--sentence-sleep-rate` (0.0) instead places that fraction of long episodes' sleeps at a *sentence* boundary in the middle third (a sentence-ending token followed by a whitespace-starting token, from a vocab scan) — reaches inside long single-QA document turns where no turn boundary exists, training reading-persistence across a sleep. A `--fact-rate` (0.3) fraction of episodes host a fact block (`--min-facts`..`--max-facts`, 4–64, log-uniform; same heterogeneous kinds, phrasings, and `--revise-rate` revisions as `preparation/interference.py`, keys unique per chain), and each block's queries are placed at one of three distances: within the same episode, in a later episode of the same wake, or beyond a sleep — the last answerable only through the neural memory, chosen uniformly among the distances available for each query unless `--cross-sleep-bias P` (default 0.0) forces the cross-sleep distance with probability `P` when it exists. Only cross-sleep queries require the memory, so raising this concentrates the memory-requiring training signal (uniform selection leaves it ~1/3 of queries). Emits `recall_masks` like `preparation/interference.py`. The run ends with a structural validation pass (decoded samples of every event kind, `--validate-samples`); the dataset is written first, but a nonzero malformed-transition count exits nonzero — fact blocks currently splice unanswered `[USER]` turns into conversations, so `--fact-rate 0` is the clean configuration.

**`preparation/interference.py`** (`make prepare-interference`) — splices interference-recall structure into an already-tokenized dataset (default `data/train_memory.pt` → `data/train_memory_v2.pt`): a fraction of examples (`--fraction`, default 0.3) get a block of key/value facts near the start — count log-uniform between `--min-facts`/`--max-facts` (8/256) — plus `--min-queries`..`--max-queries` (3–8) query/answer turn pairs at random later turn boundaries. Facts are heterogeneous (digit codes, keywords, colors, weekdays, counts, owners), each with several matched statement/question/answer phrasings, so the learnable invariant is key→value binding rather than one template; `--revise-rate` (default 0.12) of facts are later revised to a new value, with their queries placed after the revision expecting the newest value (trains overwrite-on-update). Everything is spliced at the token level (only the injected turns are tokenized, batched), so regeneration takes minutes. Fact keys come from a vocab slice disjoint from `diagnostics/recall.py`'s, keeping the probe an honest held-out eval. Query answers are assistant turns with `True` masks — the recall training signal. The output also carries `recall_masks` marking exactly those spliced answer tokens, so training can amplify them with `--recall-weight` (see "Per-token loss weighting" above).

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
