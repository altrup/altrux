# Experiment notes 2026-10-12 12:49 — local replay of altrux seed-42 epoch 1 cycle 0

Local probe (RX 7700S, 8 GB) during the third-box debrief. Reproduces the
first distillation cycle of the altrux cell from the box's saved artifacts
and measures what one cycle does to the wake frame's probability.
Script: `sft/experiments/lama_ckl/diagnostics/replay_cycle.py`.

## Inputs

- Adapter: `.cache/lama_ckl/runs/altrux-seed-42/epoch-00` (the ultra20k warm
  start, sha `98f0071f…ce17`, before any cycle).
- `epoch-01/cycle-00/wake.json`: 720 transcript tokens, five documents,
  forced closes 0/5. The open state is rebuilt by a forward pass over the
  transcript.
- `epoch-01/cycle-00/dreams.json`: the ten dream token sequences. Teacher
  logits were not persisted by the box (`ephemeral_teacher_logit_bytes`),
  so they are recomputed from the rebuilt open state with the same adapter.
- Optimizer as the runner: AdamW, lr 1e-4, weight decay 0.01, fresh at
  cycle 0; one step per dream in the box's order; KL temperature 1.

Deviation from the box: the backward runs in 4-token chunks with the state
detached between chunks (truncated backprop) and gradient checkpointing off,
because a full-sequence backward of the 2.7B model does not fit in 8 GB.
The per-step KL values match the box log within 0.03, so the forward is
faithful; gradient norms and the post-step weights are approximate.

## Per-dream KL and LoRA gradient norm (sequential, as the box stepped)

| step | dream (opening) | KL | grad norm | box KL |
|---|---|---|---|---|
| 1 | `[USER] Can you provide more information on the different starting quarterbacks…` (about the wake) | 0.787 | 0.676 | 0.7895 |
| 2 | same question, second sample (about the wake) | 1.618 | 0.468 | 1.5933 |
| 3 | `[USER] Remember this for later: When Romo started…` (frame + document copy) | 1.351 | 0.541 | 1.3386 |
| 4 | `Write a narrative about a person who overcame adversity` | 0.019 | 0.157 | 0.0213 |
| 5 | `Can you discuss the types of colorants…` | 0.014 | 0.135 | 0.0173 |
| 6 | `How did the writer incorporate their personal experiences…` | 0.030 | 0.196 | 0.0364 |
| 7 | `Write a fantasy story…` | 0.015 | 0.094 | 0.0164 |
| 8 | `The Nth Day is a 2005 American romantic comedy…` | 0.025 | 0.246 | 0.0228 |
| 9 | `In what ways can you enhance your communication skills…` | 0.019 | 0.105 | 0.0201 |
| 10 | `Write a fictional short story…` | 0.011 | 0.076 | 0.0124 |

The three wake-related dreams carry 40–80× the KL of the seven normal
conversations but only 3–5× the gradient norm, and the frame copy is no
heavier than the two question-and-answer dreams about the wake. The
normal conversations are not negligible steps under Adam.

## Frame probability before and after the cycle

log P of the seven tokens `<|endofconversation|> [USER] Remember this for
later:` as the wake tokenizes them (` Remember` is one token), scored from
an empty state and from the rebuilt cycle-0 open state:

| condition | before | after 10 steps | factor |
|---|---|---|---|
| empty state | −22.28 | −14.85 | ×1700 |
| cycle-0 wake state | −3.35 | −1.16 | ×9 (3.5 % → 31 %) |

A first version of the probe encoded the frame as `" "` + `"Remember"`
(a different tokenization) and read −33.6/−19.4 before and −26.1/−16.1
after; discarded.

## Reading

1. The wake state alone, before any training, gives the whole frame a 3.5 %
   chance as the first thing a dream writes after `[USER]`. Cycle 0 dreamed
   one frame copy in ten. In-context imitation of the five pasted frames is
   the seed.
2. One cycle's ten steps raise that to 31 % in context and by ×1700
   unconditionally. Cycle 1 then dreamed from a state holding the carried
   cycle-0 state plus five new frames and produced eight frame imitations in
   ten, all variations of cycle 0's first document ("When Romo started…" →
   "When Primal Madness was canceled…", "When PRBC hosted…"). None related
   to cycle 1's own documents.
3. The gradient is shared about equally by the three wake-related dreams,
   not carried by the copy alone. The question-and-answer dreams teach
   content specific to cycle 0's documents, which the next wake's state no
   longer holds; the frame is identical in every cycle, so only it compounds.
4. The runner carries the open state across cycles, so the in-context frame
   prior grows with the cycle count independently of training.
5. The warm start never saw the frame: 0 occurrences of "remember this for
   later" in the UltraChat train split, no acknowledgement rows in the pool,
   the top-up never ran. The warm start's `--repeat-rate 0.3` (5722 verbatim
   repeat boundaries) is a separate suspect for the copying habit; not tested.

## Artifacts

- `epoch-01/cycle-00/replay-trainable.pt`: the post-step LoRA tensors from
  this replay, for later probes without the 25-minute rerun.
- Log: scratchpad `replay_cycle0b.log` (not kept).

Open for the debrief: mask wake-transcript tokens out of the distill loss;
bound or reset the carried state; one batched optimizer step per cycle;
save the adapter and score after every cycle at one epoch.
