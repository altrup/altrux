# Discussion notes — 2026-10-14: LAMA-CKL fourth box run debrief

Debrief of the fourth GH200 session (2026-10-13 19:29 UTC to 2026-10-14
02:45 UTC, run note `notes/experiments/EXPERIMENT_NOTES-20261013-192947.md`,
box branch `box/20261013-185224`, squash-merged into `main` as `e5d6cc6`).
It amends
[the 2026-10-13 third box debrief](DISCUSSION-20261013-lama-ckl-third-box-debrief.md),
which stays governing for everything this note does not change: warm-start
recipe (§3.1), probe (§3.2), the wake, dream-prompt and distillation
amendments (§3.3), the goal block and the four curves (§3.5), and the
rejected list (§4). Where the two differ, this note wins.

This note is short on purpose. The fourth box's dream cell measured nothing
(§1), so the scientific discussion of the dream arm waits for a corrected
run (§3). What this note registers is the defect, what remains valid, the
corrected mechanism, the fp32 optimizer change it needs to be a fair test,
the session that reruns the cell and its two trained baselines, and the
process changes that came out of the failure. Cycles are numbered 0 to 99; "the
last cycle" is cycle 99, which the 10-13 note calls "cycle 100".

## 1. What happened

### 1.1 The dream cell trained the student from the teacher's state

In every cycle of the fourth box's dream arm, the student replayed the
dreams from `open_state`, the wake state the teacher had generated them
from, with the same adapter. Teacher and student were one function on one
input. The KL between them was bf16 rounding noise between the single-step
decode path (teacher, token by token under `no_grad`) and the chunked
forward (student, one padded batch with gradients): 0.0034 at cycle 0,
0.0004–0.0035 across all 100 cycles
(`sft/logs/lama-ckl-run-20261013-231911.log`). For comparison, the third
box's acceptance cycle (fresh-state student, one sequential step per dream,
ten dreams, the ultra20k warm start) read 0.47 and 0.38 on its two
wake-revisiting dreams (duplicates of one dream, so one carrier counted
twice) and 0.011–0.041 on the eight unrelated ones
(`EXPERIMENT_NOTES-20261012-022401.md` lines 157–163).

The weights still moved, because AdamW normalises the gradient: with
N = 21,323,264 adapter parameters and lr 1e-4, Adam's first step has norm
about lr·√N ≈ 0.46 (the measured 0.475 exceeds it because the saved
adapters are bf16-rounded; later steps are not bounded per coordinate). Measured from the saved cycle
adapters (`cycle-NN/trainable.pt` holds A_{c+1}; the warm start is A_0):
the first step (warm start to cycle 0) was 0.475, then 0.335, 0.272,
0.231, 0.215, 0.192, 0.119 at cycle 11, and 0.06–0.13 (mean 0.081) over
cycles 20–99. Consecutive steps were correlated through Adam's momentum
(cosine 0.62 between the first two, 0.74 between the second and third;
over cycles 20–99 the mean cosine between steps one cycle apart is 0.79,
five apart 0.47, twenty apart 0.08, against the 0.9 / 0.59 / 0.12 that
β₁ = 0.9 predicts for independent gradients: the same shape, 20–35 % lower
at the longer lags). One caveat on all of these
step readings: **the adapter is bf16** (every `trainable.pt`, in every arm,
holds bf16 tensors; median |A| = 0.0063, where one bf16 ulp is ≈3e-5), and
a steady-state Adam step of ≈2.3e-5 per coordinate is under one ulp, so
most coordinates do not move at all in a given cycle (64 % unchanged
between cycles 50 and 51; median non-zero change 2.1e-5) and weight decay
at 1e-6·|A| is a no-op. The step norms and cosines above are a quantised
readout, not a clean measure of gradient consistency. The corrected run
saves fp32 adapters (§3.1), so its step readings are not quantised and a
difference from the void profile is partly the dtype. Worse, in the void
cell the rounding was a filter: at the full
per-coordinate step of 1e-4 (a gradient consistent across cycles) 0.08 %
of coordinates cannot move; at the noise-level step of 2.3e-5 (a gradient
new every cycle) 41 % cannot. The component that changes every cycle is
the document facts; the component that repeats is the wake's shape. So
a bf16 adapter preferentially discards the thesis signal and keeps the
frame transfer, which is why §3.1 moves the trainable set to fp32 master
weights. These coordinate fractions are a calculation from the ulp
spacing at the measured weight magnitudes and the Adam ratio, not an
observation on a real gradient; the baseline rerun is the first
measurement of the effect.
The adapter's distance from A_1 (the `cycle-00` file) reached 0.57 at
cycle 2, 1.61 at 10, 2.20 at 20, 2.88 at 48, 3.63 at 99 (from A_0, the
warm start: 0.93, 1.88, 2.44, 3.07, 3.78). A walk of lr-sized, noise-directed,
momentum-smoothed steps accounts for the "drift" the run observed (no
zero-lr control was run; the attribution is by mechanism, not by
control): acknowledgement replies for six cycles, then document
restatements and copies; dreams stop closing on EOC by cycle 5, the frame
appears in dreams from cycle 4 and as a `[USER] Remember this for later:`
opener from cycle 6, the `[USER]` marker leaves most dreams by cycle 18,
last-document copies rise from cycle 20 (mean copy fraction 0.18, 0.13,
0.31 at cycles 20, 21, 23) with duplicates from cycle 24, and dreams opening
on the bare frame are common from cycle 44 (the full census is the §3.3
table). None of it could have been learning, because the gradient was a
numeric-path difference between the teacher's single-step decode and the
student's chunked forward; the cosines above are consistent with no
direction across cycles, though as a bf16-quantised readout they cannot
carry that claim alone. Structure within a cycle is not excluded, and the drift had
content from S_c (the last document copied, the frame), so "a perturbed
model that copies from its state" is the alternative description; the
attribution to the walk is by mechanism, without a zero-lr control. The
void cell's batch loss tracked its copy fraction (Pearson −0.96 over the
100 cycles; every dip under 0.001 sits on one of the seven highest
copy-fraction cycles): copies are low-entropy text, and two numeric paths
agree more closely on a peaked distribution. That is a statement about
the numerics floor, not about state dependence, and it is no reference
for the corrected run's loss. The void cell is
the **no-consistent-direction control** for the corrected run (§3.3): same
warm start, same optimizer, same step count. It does not control for step
size: with AdamW's defaults (β 0.9, 0.999) a zero-mean
gradient settles the per-coordinate ratio |m̂|/√v̂ near 0.23, which puts
the step near 0.1 (measured 0.06–0.13 after cycle 20), while a gradient
that is consistent from cycle to cycle pushes the ratio toward 1 and the
step toward 0.46. Adam's step size responds to consistency across cycles,
not to information. The fix guarantees a real gradient, not a consistent
one: a gradient whose direction changes with each cycle's five documents
also settles near the void step size, while a component that is the same
every cycle (the frame, the user-turn opener, the wake's rhythm) raises
it. The step readings in §3.3 are recorded, not interpreted, with that in
mind.

### 1.2 How the defect entered

- 2026-10-09: in conversation, the analyst described the core of the method
  as "feed the same tokens to the student from the same state". False, and
  the code at that time did the opposite (`distill_dream_set` →
  `distill_replay(init_state=None)`, a fresh-state student, the form the
  08-05 consolidation-null work had found necessary and
  `DISCUSSION-20260805-dream-distillation-cl-ab.md` §3 registers: "from a
  **fresh state**").
- 2026-10-13 debrief §2 repeated the false premise in prose ("state fixed
  during distillation") and §3.3.6 registered "one batched optimizer step
  per cycle" without naming the student's initial state.
- The implementation brief for `distill_dream_batch` said "from
  `init_state` repeated B times" without saying which state; the runner
  brief said "call `distill_dream_batch(model, opt, dreams, init_state,…)`"
  and "dreams still generate from that cycle's open state". The runner
  agent passed `open_state`, the variable already in that argument slot
  (where the old function used it only to read the device). Commit
  `0ff672d`; the person committed the protected file.
- The new test passes a dummy `State()` and never checks the student's
  state. The dry run read the file and did not flag it. The experimenter
  recorded loss 0.0034 at cycle 0 without comment and at cycle 31 (loss
  0.0016) wrote "the teacher agrees with itself".
  The analyst verified mid-run that teacher and student were the same
  function and recorded it as "a plausible drift mechanism" instead of a
  defect against the 08-05 spec. The cell ran to completion.
- The teammate never said the student should see the teacher's state. The
  question "should we keep the state?" (2026-10-13 03:37) concerned the
  wake state carried across cycles and became the per-cycle reset.

### 1.3 Which runs it voids

Only the fourth box's dream cell and its five acceptance runs.
`distill_dream_batch` entered `runner.py` in `0ff672d`; of the four box
branches only `box/20261013-185224` contains it. The first, second and
third boxes ran the fresh-state student, and their results, including the
third box's three-cycle collapse, stand as measured.

## 2. What the fourth box established

### 2.1 Valid and banked

| artifact | where | note |
|---|---|---|
| Warm start `ultra20k-v4`, step 9594, sha256 `7ba7972b…` | `models/mamba2_2_7b/checkpoints/ultra20k-v4/final` → `epoch-1/step-9594`; pin in `sft/experiments/lama_ckl/warmstart.sha256` | probe at the pin: EOC-first 1/50 (the Covenant School document, allowed by the teammate), role-marker closes 1, forced 2, no copies, 48/50 content replies |
| Split, manifest sha256 `f829ea48…` | `.cache/lama_ckl/mamba2_2_7b_ultra20k-v4/` | 500/500, invariants 0 |
| Frozen ×3 | `runs/frozen-seed-42-r1`, `-r2`, `-r3` | to-learn 0.001 ×3 (spread 0.000); not-to-forget 0.987/0.987/0.984 (spread 0.003). The triplet understates the cross-process noise: the same warm start was scored in 12 separate full evaluations (`epoch-00/result.json` of every run, acceptance and frozen) and they fall in two modes, 0.977 / 0.002 (8 times; document 357 scores) and 0.984–0.987 / 0.001 (4 times; document 355 scores 0.5), so the retention noise on this warm start is **0.010**, as the run note also says, and the to-learn flicker is 0.006 (below) |
| lora seed 42, 30 epochs | `runs/lora-seed-42` | peak 0.098 at epoch 9 (0.538 retained); epoch 1: 0.021 / 0.899 |
| mix-review seed 42, 30 epochs | `runs/mix-review-seed-42` | peak 0.067 at epoch 11 (0.502 retained); epoch 1: 0.023 / 0.911 |
| lora seed 43 to epoch 20, mix-review seed 43 to epoch 13 | `runs/{lora,mix-review}-seed-43` | same plateaus as seed 42 (lora 43 peaks earlier, 0.086 at epoch 4); resumable |

The lora and mix-review rows are the fourth box's record and the only
30-epoch curves on this warm start; they were trained with a bf16 adapter
(§1.1) and are rerun under fp32 master weights in the next session
(§3.2), which supplies the comparator for the dream cell.

Two readings from these that stand on their own:

- **The scorer's to-learn resolution is three documents (0.006), not the
  frozen triplet's 0.000.** Only five of the 500 to-learn documents ever
  scored above 0 in the dream cell's 100 evaluations. Four flicker before
  their wake: document 357 scores 1.0 at cycle 1, in the dream cell's own
  `epoch-00`, and in seven other fresh-process evaluations of the frozen
  weights, and the wake did not show it until cycle 71; 211 and 254 behave
  the same way (shown at cycles 42 and 50, scoring from cycles 9 and 22);
  355 flickers at 0.5. The fifth, 292, scored once, at cycle 90, after its
  wake at 58. To-learn
  values up to 0.006 are these borderline documents under GPU
  nondeterminism, not acquisition. Any claim of acquisition needs documents
  that score after their wake and keep scoring (§3.3).
- **Mix-review's retention looks warm-start dependent, one seed per base.**
  On the third box's `ultra20k` base it plateaued at 0.630 retained
  (10-13 §1 item 7); on `ultra20k-v4`, a neighbouring recipe, at 0.441,
  0.05 below lora's plateau. Seed 43 at epoch 13 reads 0.507
  against lora's 0.509, equal within noise, so the second seed neither
  confirms nor denies it. A reading to check, not a result.

### 2.2 Void

The dream cell (`runs/dream-seed-42`, 100 cycles) and the acceptance runs
(`acceptance/dream-5*`). Their curves are the walk of §1.1 and carry no
information about the dream arm. They move to
`.cache/lama_ckl_archive/fourth-box/` (§6.2), not deleted, as the record of
the defect.

Two protocol observations from the void cell are still worth keeping,
because the corrected arm will face them:

- At 10 dreams per cycle, 0 of 20 dreams over two seeds mentioned a wake
  subject. Under item 4 of the 2026-10-10 note's amended list (20 to 50
  dreams at the experimenter's judgment) the experimenter chose 50; 50 ran out of memory in the single distillation step; torch's
  `peak_vram_bytes` (`acceptance/dream-5-d40/epoch-01/result.json`) read
  93.3e9 B (86.9 GiB) at 40 dreams and 71.4e9 B at 30, ≈2.2e9 B per dream,
  on a card the run note gives as 94.5 GiB (the run note's 91.5 GB at 40
  is `nvidia-smi`'s reserved figure, a different quantity); 30 was the
  memory decision. At 30, 40 and 50 the same one
  dream (a question-and-answer revisit of the Knut Ove Lystad document,
  half confabulated) was the only mention, so the gate passed on it. The
  gate is weak: it passes on one dream in thirty, and raising the count
  only widens the net. Dreams revisit the wake rarely; the warm start's
  conversations all open with `[USER]`, so a revisit is a user decision
  the model has to make, and the variants that revisit a document in a
  later user turn are 249 recall and 230 question rows in 4610 repeats
  (the 221 multi-turn and 30 summary variants continue a conversation
  rather than ask about an earlier document; a corpus count from the
  warm-start generator log, `sft/logs/warm-start-20261013-192947.log`; the one
  revisiting dream's text is in
  `acceptance/dream-5-d30/epoch-01/cycle-00/dreams.json`).
- The seed rule (10-13 §3.3.8) fired on a 0.060 retention difference that
  only meant "the dream arm learned nothing". No seed rule applies to this
  run (§3.2); its replacement is decided after the corrected run.

## 3. Standing direction

### 3.1 The fix: student replays from an empty state; fp32 master weights

This is the mechanism spec block for the dream arm. It is what the
implementer's brief quotes and what the PROTECTED EDIT report maps to
`file:line`. Everything else about the arm is as 10-13 §3.3 registers it.

**Ledger, one cycle.** S_c is the open state at the end of the wake: after
the last sampled reply, before the EOC dream prompt. "Empty" is the state
`Model.forward(…, state=None)` builds (zero-initialised), the same state the
wake and the scorer start from. "Adapter A_c" is the trainable set as
today, the LoRA matrices plus `marker_delta` (`train_hooks.py`), LoRA
dropout 0, so train and eval mode agree, **held and optimised in fp32
master copies**. The model's trainable tensors stay bf16 and the forward
pass is unchanged (`apply_lora`'s matmul takes bf16 on both sides). An
optimizer wrapper owns an fp32 copy of each trainable tensor and the AdamW
state over those copies; its `step()` copies the bf16 gradients up into
the fp32 copies' `.grad`, runs AdamW, and casts the copies back into the
bf16 tensors; its `zero_grad()` clears the model's gradients. The two
protected functions that call `opt.step()` / `opt.zero_grad()`
(`distill_dream_batch`, `train_document_epoch`) are not edited; the runner
builds the wrapper where it builds the optimizer today. The wrapper's
`state_dict()` holds the fp32 copies and the AdamW state, so
`optimizer.pt` carries them and a resume restores the masters from
`optimizer.pt` (a resume whose `optimizer.pt` is missing stops with the
`FileNotFoundError` from loading it; `load_checkpoint` of a `trainable.pt`
into the bf16 model stays as today and is only the cast). `_save_trainable` saves the fp32 copies, so
`trainable.pt` is float32; the bf16 warm start is upcast once at process
start when the wrapper is built. `run.json` settings record
`"adapter_dtype": "fp32-master"` for the trained arms (`null` for the
frozen arm, which has no optimizer and keeps its bf16 adapter, so the
banked frozen ×3 still pass the settings check), and a leftover bf16 run
directory of a trained arm fails the resume settings check instead of
resuming. This applies to every
trained arm (dream, lora, mix-review) in this session, which is why the
two baselines rerun (§3.2). Reason: §1.1, a bf16 adapter rounds away the
steady-state Adam step in about 41 % of coordinates and keeps only the
cross-cycle-consistent component, which loads the run against the thesis.
Cost: 85 MB for the copies plus ≈85 MB for AdamW state moving from bf16 to
fp32, ≈170 MB of GPU memory; `trainable.pt` 85 MB instead of 43 MB,
`optimizer.pt` about twice its size. Every per-dream loss also carries
the numerics floor between the teacher's decode path and the student's
chunked forward (0.0034 at the void cell's cycle 0); it does not go away
under the fix and adds a direction-free component to every step.

| forward pass | initial state | weights | grad | recorded |
|---|---|---|---|---|
| initial evaluation, once per run before cycle 0 (`epoch-00/result.json`) | empty | A_0 (the warm start) | off | to-learn, not-to-forget per row |
| wake: 5 documents under the frame, replies sampled at 0.7, turn i seeded by `dream_generation_seed(dream_set_seed(seed, epoch, cycle), i, 0)` (unchanged by the fix) | empty | adapter A_c | off | transcript, S_c |
| dream generation: 30 dreams from the EOC prompt in one batch (`--dream-batch-size` = dreams per cycle = 30), single-step decode, sampled at 0.7, dream i seeded by `dream_generation_seed(dream_set_seed(seed, epoch, cycle), i, 0)` | a copy of S_c per dream | A_c | off | dream tokens (EOC prompt token first, then the generated tokens), teacher logits at every generated position, kept in float |
| student replay: the recorded dream token sequences, teacher-forced, one batched forward with gradient checkpointing as today | **empty**: `state=None` (the model builds a zero state for the batch); the device is taken from the model, not from a state | A_c | on | per-dream losses, batch loss |
| scorer: both 500-row sets, each row from the start | empty | A_{c+1} | off | to-learn, not-to-forget per row |

**Student input.** Row 3's sequence is exactly row 2's recorded tokens: the
EOC prompt token first, then the generated tokens, right-padded to the
longest dream with that dream's last token. The position predicting the
first generated token (the one after the prompt) is scored; the position
that would predict the prompt token, the position that predicts a terminal
EOC (when the dream stopped on EOC; 10-13 §3.3.6), and padding positions
are not. **Plumbing:** the student's state is not a parameter.
`EpochSteps.distil(dreams, label)` and `distill_dream_batch(model, opt,
dreams, kl_temp, …)` take no state argument; the forward is called with
`state=None`. The slot that carried the defect no longer exists.

**Loss.** For dream d and scored position t,
KL(softmax(teacher_{d,t}/τ) ‖ softmax(student_{d,t}/τ)) · τ²,
teacher from row 2, student from row 3, τ = `KL_TEMPERATURE` = 1.0. Weights
as 10-13 §3.3.6 and today's `turn_weights`: the dream splits into turns at
role markers (EOS closes a turn; text before the first marker is turn 0;
text after an EOS with no marker is an unmarked turn), each turn with
content gets 1/T of the dream's weight, each token 1/(T·n) within its
turn, so each dream's weights sum to 1. The **per-dream loss** is that
dream's weighted sum (weights summing to 1, so it is comparable across
dreams and does not shrink with the batch size); a dream is **scored**
when it has at least one scored position (its `turn_weights` are
non-empty), and `dream_losses` holds one entry per dream in `dreams.json`
order, `null` for an unscored dream, so indices match the dream text; the
**batch loss** is the mean of the per-dream losses over scored dreams; the
maximum is over non-null entries. Both are recorded in the cycle's own
record (`cycle-NN/distil.json`, below) **and** in the epoch record, under
the same keys, so a stopped cell keeps them. The quantity backpropagated is the batch loss (the mean over scored
dreams); gradients land on the bf16 tensors and are copied up. One AdamW
step at lr 1e-4, weight decay 0.01, on the fp32 master copies, then the
bf16 cast into the model, gives A_{c+1}; the wrapper is built fresh at
process start and its state carries across cycles.

**Event sequence.**
1. State reset: nothing carries from cycle c−1 except A_c and the
   optimizer state.
2. Wake from the empty state with A_c; keep S_c.
3. Generate 30 dreams from copies of S_c with A_c; record tokens and
   logits.
4. Student forward over the padded dream batch from empty states with A_c,
   gradients on.
5. Loss as above; backward.
6. Optimizer step → A_{c+1}; zero grad.
7. Save A_{c+1} to `cycle-NN/trainable.pt`; score both sets from empty.
8. Discard S_c.

**Live reading, every cycle.** The per-dream losses are printed on the
distil log line with their maximum and recorded twice, under the same
keys: in `cycle-NN/distil.json`, written at step 6 before the save and the
score so that a stopped or crashed cell keeps them, and in the epoch
record (`epoch-EE/result.json`, `treatment.cycles[]`; an acceptance run
writes that file too, and in `--cycles` mode `distil.json` is the only
per-cycle file). Keys, identical in both: `loss` (the batch loss, the
existing key that `report.py` reads; it is not renamed), `dream_losses`,
`max_dream_loss`, `scored_dreams`, `optimizer_steps`, plus
`subjects_mentioned` copied from the dream diagnostics so exposure sits
beside the loss. The reading is the **maximum
per-dream loss of the cycle**, and it answers one question only: are
teacher and student different functions? From an empty state every dream's
first scored position already sees a different context from the teacher's,
so a correct student puts a dream's early positions above rounding noise
(a long low-entropy dream can still average near the floor as the
student's state converges on the teacher's, which is why the reading is
the maximum, not every dream). References,
each from a different run and under a per-token loss on the ultra20k warm
start, so comparable in kind and not in value: the third box's acceptance
cycle read 0.011–0.041 on eight unrelated dreams and 0.47 / 0.38 on one
revisiting dream counted twice (`EXPERIMENT_NOTES-20261012-022401.md` line
163); its cell's cycle 0 read 0.01–0.03 on ordinary dreams and 0.8–1.6 on
three wake-related dreams (10-13 §1 item 3). The void cell's cycle-0 batch
loss was 0.0034 (its 100-cycle mean 0.0022), its per-dream maximum
unmeasured, so 0.01 is a guess at three times the cycle-0 value, and the
maximum of 30 noise values could clear it: the floor is a smoke alarm
that can miss the defect, not a detector; the §6.1 tests are the
detector. The runner logs a WARNING when the maximum is under
**0.01** and continues. At acceptance (cycle 0, fresh warm start) the
experimenter reads the maximum and the mean together against the void
cell's cycle-0 batch loss of 0.0034: under the defect the per-dream values are
rounding noise around that mean and their maximum is unmeasured (it may
reach 0.01); under the fix the third box's unrelated dreams alone read
0.011–0.041 each. The stop rule is one condition: **maximum under 0.01**.
The mean is recorded beside it and read, not a stop. Mid-cell it means something else: the
dreams' tokens have become text the empty-state student already predicts,
whether because the dreams stopped depending on the state or because they
collapsed to low-entropy copies. The void cell is no reference for this
(§1.1: its loss tracked copy fraction through the numerics floor alone).
Record it, read the text, continue on judgment.
Whether any dream
carried the wake is **not** readable from the loss and is read from the
dream text (§3.2 step 2). A cycle with no scored dream takes no optimizer
step and records `null` for `max_dream_loss`, `0.0` for `loss` (as today),
and an all-`null` `dream_losses`; the distil line prints `max n/a` and no
WARNING fires. The WARNING is a `print` line that contains the word
`WARNING`, as the runner prints everything; the test captures stdout. The floor is a warning, not the proof that the
fix landed: the proof is the §6.1 test, run on the box in the preflight.
The expected levels above are references, not thresholds: the experimenter
reads the maxima beside the text and judges.

**What this does to the signal, and its known limitation.** The KL from the
empty state carries everything the teacher's context contributed: the
document facts, which is the thesis, and the shape of the wake conversation
itself (the frame "Remember this for later:", the user-turn opener, the
document-then-acknowledgement rhythm, the model's own replies), which is
not. The loss cannot separate the two. The third box had this student and
after one cycle the probability that a dream from an empty state opened
with the frame rose ×1700 after cycle 0's ten distill steps (10-13 §1
item 2); it collapsed in three cycles. The
fourth warm start was the partial mitigation (document shapes and repeat
variants trained in, no literal frame in the corpus), and the per-cycle
reset, the sampled wake and the single step per cycle differ from the third
box too. A bf16 adapter would sharpen it (§1.1): the repeating shape component
takes full-size steps that clear the rounding, the per-cycle fact
component mostly does not; the fp32 master weights remove that part.
**The limitation itself stands for this run; it is what it is.** The
readings for it are the dream census from cycle 0 on, the per-cycle
adapter step norm and step cosines, and the per-cycle count of dreams that
mention a wake subject (§3.3), beside the void cell's values. None of them
separates frame transfer from the thesis signal: the census columns
degrade under both as they did under the walk, and the step readings rise
above the void profile under any cross-cycle-consistent gradient (§1.1)
and differ from it in dtype besides. **This run has no reading that
attributes its drift.** The gradient from the dreams that do not revisit
the wake (most of them, if the acceptance rate of about one in thirty
holds) has no registered objective: it is tolerated, not intended, and
its share of each step is what the subject count per cycle records. It records what the drift looks like and how fast
it comes beside the void cell's values; the attribution, and any
mitigation, are the next debrief's. Copies remain the
strongest carrier under this mechanism, and 10-13 §4.3 (no loss mask on
copies) stands.

### 3.2 Session order (replaces 10-13 §3.4 for this run)

The dream cell plus the two trained baselines, all under fp32 master
weights (§3.1), seed 42 only. No warm start, no split, no frozen: banked
(§2.1) and uploaded with the launch; the frozen arm has no optimizer, so
its ×3 stands. The fourth box's bf16 lora and mix-review runs stay in the
archive as that box's record and are not the comparator. No seeds 43 and
44, no verdict: the cells run, the readings are recorded, and the next
debrief decides. Expected ≈ 5 h (smokes and acceptance ~0.5 h, baselines
~1.5 h in parallel, dream cell ~2.7 h, report); soft cap 8 h.

0. Preflight on the box, all of it before anything runs:
   - `sft/experiments/lama_ckl/warmstart.sha256` equals the sha256 of
     `models/mamba2_2_7b/checkpoints/ultra20k-v4/final/trainable.pt`
     (`7ba7972b…`), and `.cache/lama_ckl/mamba2_2_7b_ultra20k-v4/manifest.json`
     carries that value as `warmstart_sha256`. Recreate the `final` symlink
     if the upload dropped it:
     `ln -s epoch-1/step-9594 models/mamba2_2_7b/checkpoints/ultra20k-v4/final`.
   - `summary.json` present in each of `.cache/lama_ckl/runs/frozen-seed-42-r1`,
     `frozen-seed-42-r2`, `frozen-seed-42-r3`.
   - **`.cache/lama_ckl/runs/dream-seed-42`, `runs/lora-seed-42`,
     `runs/mix-review-seed-42`, every `runs/*-smoke`, and
     `.cache/lama_ckl/acceptance/` absent.** The runner resumes a run whose
     directory exists; a leftover fourth-box directory would make a step
     "complete" without training. If any is present, stop and report; do
     not delete on the box.
   - The §6.1 tests exist and pass on the box, one selection each so that
     a missing test cannot hide behind the other (pytest exits 5 on an
     empty selection; the `test` target forwards `ARGS` after §6.4):
     `cd ~/altrux/sft && make test ARGS="-k test_student_state_is_empty"`
     (2 tests), `make test ARGS="-k test_dream_losses_recorded_and_floor_warning"`
     (2 tests) and `make test ARGS="-k test_fp32_master_weights"` (2 tests).
     This is the proof the fixes landed; the loss floor is not.
1. Smokes, serial, one per trained arm (the training path changed for all
   three):
   `cd ~/altrux/sft && make lama-ckl-smoke ARGS="--arm dream --model-name mamba2_2_7b --seed 42"`,
   then the same with `--arm lora` and `--arm mix-review`. EXIT=0 each; the
   dream smoke shows a distil line with per-dream losses; a post-step
   adapter is float32:
   `cd ~/altrux/sft && uv run --no-sync python -c "import torch,sys; print({v.dtype for v in torch.load(sys.argv[1],map_location='cpu',weights_only=True).values()})" ../.cache/lama_ckl/runs/dream-seed-42-smoke/epoch-01/cycle-00/trainable.pt`
   and the same on `runs/lora-seed-42-smoke/epoch-01/trainable.pt`
   (`epoch-00/` is saved before any step and proves nothing). A rerun of
   any smoke goes to a fresh `--output …/runs/<arm>-seed-42-smoke-r2`,
   since the runner resumes a completed smoke directory without training.
2. Acceptance, one cycle, the live reading's first check:
   `make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --cycles 1 --docs-per-wake 5 --dreams-per-cycle 30 --output ../.cache/lama_ckl/acceptance/dream-5"`.
   Read, from the log: the per-dream losses, their maximum and the mean
   (§3.1 live reading); a maximum under 0.01 is stop-and-report, the fix
   did not land; the mean is recorded beside it. If the acceptance run
   crashes after writing `run.json`, rerun it with `--output
   ../.cache/lama_ckl/acceptance/dream-5-r2` (the runner refuses to reuse
   a started `--cycles` output) and say so in the note.
   Read, from `wake.json` and `dreams.json`: the wake replies
   (acknowledgements or short content, the 10-13 §3.2 probe census applied
   by eye), `internal_eoc` 0, and every decoded dream, noting which dreams
   revisit the wake and what their losses are against the unrelated ones.
   Record all of it; none of it stops the session except the floor. The
   `subjects_mentioned > 0` condition and the dream-count rule are
   suspended for this run (§4.2); the dream count is 30.
3. Baselines, seed 42, in parallel in two `train` windows (GH200 grid
   concurrency per `sft/AGENTS.md`):
   `make lama-ckl-run ARGS="--arm lora --model-name mamba2_2_7b --seed 42 --max-hours 3"`
   and `make lama-ckl-run ARGS="--arm mix-review --model-name mamba2_2_7b --seed 42 --max-hours 3"`.
   About 1 h and 1.5 h on the fourth box. Record epoch 1 and the peak of
   each beside the fourth box's bf16 values (lora 0.021 / 0.899, peak 0.098
   at 9; mix-review 0.023 / 0.911, peak 0.067 at 11): the difference is the
   first measurement of what bf16 rounding cost the baselines.
4. Dream cell, seed 42, 100 cycles, after both baselines have their EXIT
   line (30 dreams need ~71 GB; do not share the card):
   `make lama-ckl-run ARGS="--arm dream --model-name mamba2_2_7b --seed 42 --epochs 1 --docs-per-wake 5 --dreams-per-cycle 30"`
   (`--dream-batch-size` defaults to `--dreams-per-cycle`, so all 30 dreams
   generate in one batch). About 95 s per cycle on the fourth box, 2.7 h. `--max-hours` is not
   passed: the runner checks the cap only at an epoch end, so it cannot
   stop a one-epoch cell; the experimenter stops it by hand (C-c) if it
   must. A stopped cell's cycles are under
   `runs/dream-seed-42/work/epoch-01.<random>/cycle-NN/` (they move to
   `epoch-01/` only at epoch end), the report never sees them, and
   rerunning the same command restarts at cycle 0 in a new temp dir: read
   a stopped cell by hand from those directories, name the directory read
   (the newest by mtime after a rerun) in the run note, and say so.
   Per cycle: save, score, and in the run note's curve table the two
   accuracies, the maximum per-dream loss, forced and role-marker closes,
   duplicates, copy fraction, subjects mentioned, and the §3.3 dream
   census, the last from
   `cd ~/altrux/sft && make lama-ckl-census ARGS="../.cache/lama_ckl/runs/dream-seed-42/work/<the epoch-01.* dir>"`
   every ten cycles (it skips a cycle directory the runner is still
   writing and writes its table as `census.json` in the directory it was
   given). Read the text (replies and dreams) at cycles 0, 5 and
   9, then every tenth cycle, and at any cycle where one of those columns
   leaves the min–max range of the previous ten cycles.
5. Report: `make lama-ckl-report`. It writes `runs/report.json` with the
   per-arm peaks and epoch-1 values (dream cycle 99 beside this session's
   lora and mix-review epoch 1).
6. Closing section of the run note with the §3.3 readings, then the
   shutdown checklist as the experimenter skill registers it. Nothing else
   launches in this session. If the cell was stopped before cycle 99, skip
   the report (it would ingest an epoch-0-only dream arm), name the `work/`
   directory read, give the last completed cycle's accuracies and census in
   the closing section, and leave the epoch-1 comparison to the next
   debrief.

### 3.3 Readings to record (no verdict this run)

The next debrief decides what the corrected run means. The experimenter
records these so that it can:

- **Live reading**: the per-dream loss maximum every cycle (§3.1); the
  WARNING under 0.01; stop-and-report at acceptance only; mid-cell, a
  reading to record whose meaning is open. `cycle-NN/distil.json` fields
  are the §3.1 keys: `loss`, `dream_losses` (list, `null` for unscored),
  `max_dream_loss` (`null` when nothing scored), `scored_dreams`,
  `optimizer_steps`, `subjects_mentioned`.
- **Exposure**: the per-cycle `subjects_mentioned` (the runner's dream
  diagnostic; in the void cell 0–5 per cycle, mostly 1–3, much of it
  copies) and, from the text, how many of a cycle's dreams revisit the
  wake. The "about one in thirty" rate rests on one acceptance cycle; this
  run measures it over 100.
- **The four 10-13 §3.5 curves per cycle**, with the dream census and the
  loss maximum as extra columns. The census, over all 30 dreams of a cycle
  from `dreams.json` (the runner prints three), counted on the decoded
  `text` field after removing the prompt: `prefix_tokens` is a count, so
  strip that many ids from `token_ids` and decode the rest, or equivalently
  strip the literal `<|endofconversation|>` that opens `text` (the prompt
  is exactly that one token today): **frame-anywhere** (the frame string
  `Remember this for later:` appears anywhere in the dream), **frame-early**
  (the frame string starts within the first 60 characters of the generated
  text, with or without a
  leading `[USER]` or whitespace; the `[USER] Remember this for later:` form
  is the wake's own rendering and was the first form the void cell
  produced), **frame-strict** (the generated text, whitespace stripped,
  starts with the frame string), **no `[USER]`** (no `[USER]` token anywhere
  in the dream), **`[ASSISTANT]` present**, **stopped on EOC**
  (`stop_reason == "eoc"`), duplicates and copy fraction (both from the
  runner's diagnostics).
  `experiments/lama_ckl/diagnostics/dream_census.py` (§6.3) prints it per
  cycle, with the adapter step norm ‖A_{c+1} − A_c‖ and the cosines between
  the step and the steps 1, 5 and 20 cycles earlier, from the saved
  `trainable.pt` files (`cycle-NN/trainable.pt` is A_{c+1}; the warm start
  `final/trainable.pt` is A_0, passed as `--warm-start` with
  `DEFAULT_WARMSTART` as the default, and the script loads it for the first
  step; distances are reported from A_0). The loss maximum comes from
  `cycle-NN/distil.json`, which exists for a stopped cell too.
- **The void cell as control**, counted over all 30 dreams of every cycle
  of `runs/dream-seed-42` (fourth box, archived under §6.2; the earlier
  timeline in the run note came from the three printed samples and put
  the first frame-strict dream at cycle 48; the full count has
  frame-strict first at 24 and frame-early at 6). The EOC, duplicate and
  copy-fraction rows come from the runner's recorded diagnostics; the
  other rows and the step readings come from `dream_census.py` (§6.3) run
  over the full run in this debrief session (its output is
  `.cache/lama_ckl_archive/fourth-box/runs/dream-seed-42/epoch-01/census.json`,
  archived with the run):

  | column | void cell, all 30 dreams per cycle |
  |---|---|
  | stopped on EOC | 17 at cycle 0, 5 at 1, 2 at 2 and 3, 1 at 4, 0 at every cycle from 5 |
  | `[ASSISTANT]` present | 29 at cycles 0–1, 14 at 5, 7 at 8, 2–4 at 16–19, 0–3 after (0 in 35 of cycles 44–99) |
  | no `[USER]` | 0 through cycle 15; 3, 7, 23, 23, 19 at cycles 16–20; 23–29 at 21–25; 30 at every cycle from 26 except 29 at 75 and 24 at 78 |
  | frame anywhere in a dream | first at cycle 4 (1); 3 at 6, 2 at 7, 1 at 16 and 17, 3 at 20 and 24; 0–18 after (14 at 48, max 18 at 78; 0 in 23 of cycles 44–99) |
  | frame-early | first at cycle 6 (3, the `[USER] Remember…` form), 2 at 7, 1 at 16, 1 at 24; 0 at 25–43; 0–11 over cycles 44–99 (non-zero in 29 of 56; max 11 at 72 and 78) |
  | frame-strict | first at cycle 24 (1); 0 at 25–43; 0–11 over cycles 44–99 (same cycles as frame-early) |
  | duplicates | 0 through cycle 23; 3, 3, 7, 0, 3 at 24–28; 0–11 after (max 11 at 64; 0 in 42 of cycles 44–99) |
  | mean copy fraction | ≤0.03 through cycle 15 except 0.09 at 6; 0.18 at 20, 0.57 at 24, 0.70 at 25; 0.00–0.88 over cycles 26–99 (≥0.5 in 16 of them) |
  | adapter step norm | 0.475 (A_0 → A_1), 0.335, 0.272, 0.231, 0.215, 0.192; 0.119 at cycle 11; 0.06–0.13 (mean 0.081) over cycles 20–99 |
  | step cosines, cycles 20–99 | lag 1 mean 0.79 (0.43–0.90), lag 5 mean 0.47 (0.28–0.69), lag 20 mean 0.08 (−0.07–0.17) |

  The corrected run's columns are recorded beside these. The next debrief
  reads the comparison; this note registers no rule for it (§3.1).
- **Acquisition census** (a sanity reading behind the to-learn number, not
  the acquisition readout; with dreams revisiting the wake about once in
  thirty and one step per cycle, the expected count is low under the
  thesis too, so a low census does not count against it; the same holds
  for the to-learn number itself: this run can confirm the thesis by a
  surprise, it cannot refute it, and the next debrief reads a null as
  "not shown", not as "shown false"): the to-learn
  documents that score at the last
  completed cycle and scored 0 at every evaluation before their wake
  (including the initial `epoch-00` evaluation). This removes the flicker
  documents of §2.1, which turn on before they are shown. "Scores" means
  a to-learn token accuracy above 0 (355 flickers at 0.5); the census
  records the value (355 is a real row that flickers at 0.5; 357, 211, 254
  flicker at 1.0). On the void cell the census gives zero documents
  (292, shown at cycle 58, scored once at cycle 90 and not at 99), and one
  near miss, so noise can produce single post-wake passers; the bands are
  uncalibrated: 0 is what the void cell gave; 1–4 is within reach of
  flicker on one run's evidence; five or more is beyond anything the void
  cell showed. The census is biased toward documents shown early (a
  document shown at cycle k has k+1 evaluations in which to be caught
  flickering before its wake; every flicker document the void cell caught
  was shown at cycle 42 or later) and reads one final
  evaluation (a document that scores in a third of cycles, as 211 did, is
  flicker whichever side of the last cycle it lands on), so it is a
  reading, not a verdict. Document indices in `result.json`'s
  `to_learn_scores` are row indices of `variant.jsonl`; rows are shown in
  file order, five per cycle, so document i was shown at cycle i // 5.
- **The epoch-1 comparison** from `report.json`: dream cycle 99 beside
  this session's lora and mix-review epoch 1 (the fourth box's bf16 values,
  0.021 / 0.899 and 0.023 / 0.911, are recorded beside them, not used as
  the comparator), with this box's noise stated next to it: to-learn
  flicker 0.006 and retention cross-process spread 0.010 (§2.1, the two
  scoring modes of the same warm start; the frozen triplet's 0.003 landed
  in one mode). This is the
  comparison the next debrief weighs first, together with the acquisition
  census and the dream text; it is what could count against the dream
  arm, and this note registers no threshold for it.
- **Kill on collapse**: as 10-13 §4.8, the experimenter stops on judgment
  with the text in the note, using the dream census as the checklist. A
  stop for any other reason (crash, OOM, operator) is rerun if the cap
  allows, from cycle 0, and recorded either way.

## 4. Explicitly considered and rejected

1. **Raise an error at the live-reading floor.** Rejected by the teammate: a
   logged warning. The floor is a reading, not a proof, and an error in a
   protected file turns a judgment into a crash.
2. **Acceptance condition `subjects_mentioned > 0` with the dream-count
   rule, for this run.** Suspended: the pair is weak (§2.2), and the
   quantity that matters for the corrected mechanism is the per-dream KL,
   which the acceptance run now reads instead. Whether the condition
   returns is decided after the corrected run.
3. **A separate SPEC file written before implementation.** Rejected in
   favour of a mandatory mechanism spec block inside the DISCUSSION note
   (§5): a second document drifts from the first; what was missing was a
   section with fixed fields and a review that maps it to the code, not a
   file.
4. **Running the baselines again on this box.** Rejected: banked, pinned by
   sha, and the report reads their `summary.json` and `result.json`.
5. **Seeds 43 and 44 in this session**, for any arm, and **a pass/fail
   criterion read in-session**. Rejected by the teammate: the session is
   scoped to the rerun of the failed cell; what it means is the next
   debrief's question.
6. **Calibrating the live-reading floor by a local replay of cycle 0 with
   the fresh-state student.** Rejected: the floor only has to separate
   "same function" from "different function", and the experimenter judges
   everything above it from the text.
7. **Designing a mitigation for the frame transfer (§3.1 limitation) before
   the run.** Deferred by the teammate: known limitation, partly mitigated
   by the fourth warm start; the corrected run measures it.
8. **Discussing the dream arm's design (revisit priors, assistant-only
   scoring, frozen world turns, projected gradients, learning-rate or
   optimizer changes against the noise walk of §1.1) in this note.**
   Deferred: every such proposal was motivated by a cell that measured
   nothing. The corrected run decides whether any of it is needed.
9. **Deleting the void artifacts.** Rejected: archived under
   `.cache/lama_ckl_archive/fourth-box/` as the record of §1.
10. **Running the corrected cell with the bf16 adapter and deferring the
    dtype question.** Considered first, then rejected by the teammate once
    §1.1's filter argument was on the table: a null from a bf16 run could
    not be told from rounding, and the run exists to give the thesis one
    clean shot. The two baselines rerun so the comparison stays like for
    like.
11. **Re-training the warm start in fp32.** Rejected: it is a pinned
    artifact whose behaviour the probe measured and the split was selected
    against; the dtype change concerns how the arms train from it.

## 5. Process changes (done this debrief, commits `ef4a06b`, `e7f7c90`)

- **Mechanism spec block** required in every DISCUSSION note that
  registers or changes a mechanism (`.agents/skills/altrux-debrief/SKILL.md`
  step 4): at least a ledger per forward pass (initial state, weights, grad,
  recorded), the loss with each input naming its ledger row, the numbered
  event sequence, and the cycle-0 live reading. "At least": more detail is
  welcome. §3.1 is the first instance.
- **Derived numbers** come from a computation over the whole artifact and
  the note names it (root `AGENTS.md`, "Sanity-check the artifact"); the
  heavier "save every derived number as a file" rule was tried and dropped
  the same day as process for a failure that never happened.
- **Adversarial read** as a command, `/altrux-adversarial-read <note>`
  (`.agents/skills/altrux-adversarial-read/SKILL.md`): an Opus subagent
  with no conversation context reads the note alone (pass one:
  completeness and clarity), then against the repo (pass two: truth and
  support), and reports findings in four ranked categories: pointless or
  self-defeating, wrong, unsupported, unclear. Pass-one findings are
  append-only; the code doing it right today never removes one. The
  debrief flow runs it before implementation and brings the findings to
  the teammate with the analyst's position on each. Its first run, on the
  draft of this note, produced 20 findings; the material ones were a floor
  calibrated against a misread number (now the per-dream maximum), the
  frame-transfer limitation (now §3.1), a floor that claimed to read
  "no revisit" when it can only read "same function", a per-dream loss
  whose normalisation would have made the floor 30× wrong, and a resume
  hazard on the box (now preflight item 3 of §3.2). Its second run removed
  the in-session verdict.
- **Debrief flow** reordered: bank, analyze, discuss, write, adversarial
  read and discuss, repeat, implement with review, dry run (after
  implementation, against the real code), close.
- **Protected-edit rule** (root `AGENTS.md`, "Protected paths"): the
  implementer's brief quotes the mechanism spec block verbatim; the
  PROTECTED EDIT report maps every ledger row to `file:line`; a forward
  pass whose initial state the brief does not name is not implemented.

## 6. Housekeeping

Pending, by consent, in this order:

1. **Runner and distillation fix** (protected, the teammate commits):
   - `EpochSteps.distil` and `_BoxSteps.distil` drop the state parameter;
     `distill_dream_batch` drops `init_state`, calls the model with
     `state=None`, and takes its device from the model's parameters. The
     runner's cycle loop calls `steps.distil(dreams, label)`.
   - `distill_dream_batch` returns the per-dream losses (each dream's
     weights summing to 1; the batch loss their mean); the runner prints
     them on the distil line with their maximum, records them in
     `cycle-NN/distil.json` at step 6 and in the treatment record
     (`treatment.cycles[].dream_losses`), `null` for an unscored dream, and
     logs a WARNING when the maximum is under 0.01.
   - **fp32 master weights** as §3.1 specifies: `MasterAdamW` in
     `sft/training/master_weights.py` (shared, unprotected) owning fp32
     copies and the AdamW state, with `step()` (grad copy up, AdamW step,
     cast back), `zero_grad()`, `state_dict()` / `load_state_dict()`; the
     runner (protected) builds it in place of `torch.optim.AdamW`, saves
     the copies in `_save_trainable`, restores them from `optimizer.pt` in
     `_resume` (a missing file raises), and records `adapter_dtype` in
     settings (`"fp32-master"`, `null` for frozen). `train_document_epoch`
     is untouched. **Done (uncommitted; the teammate commits the protected
     files):** `distill_dream_batch` and the runner as above; tests
     `test_student_state_is_empty` (+ `_in_batch`),
     `test_dream_losses_recorded_and_floor_warning` (+ `_in_batch`),
     `test_fp32_master_weights` (+ `_saved_and_required_on_resume`); full
     suite 741 passed, 8 skipped, 1 xfailed; `sft/README.md` updated.
   - Tests first, names fixed because the §3.2 preflight selects them:
     `test_student_state_is_empty` (drives `_BoxSteps.dream` and
     `_BoxSteps.distil` directly with a fake model and tokenizer whose
     forward records its `state` argument; asserts that the first dream
     forward's state is value-equal to the wake state repeated over the
     batch (`_repeat_state` copies, so identity cannot be asserted) and
     that the distil forward's state is `None`; the
     existing `run_epochs` test with a fake `EpochSteps` never reaches a
     model, so it cannot carry this), the same assertion at the function
     level in `test_dream_distillation.py`, and
     `test_dream_losses_recorded_and_floor_warning` (per-dream losses with
     weights summing to 1, batch loss their mean, the record field, the
     WARNING under 0.01; at the `distill_dream_batch` level for the
     losses and at the runner level for the record and the print), and
     `test_fp32_master_weights` (on the wrapper: a gradient whose Adam
     step is below one bf16 ulp of the parameter moves the fp32 copy and
     accumulates over two steps while the bf16 tensor changes once the
     copy crosses a spacing; `state_dict` round-trips the copies; on the
     runner: `_save_trainable` writes float32 and a resume without
     `optimizer.pt` raises). The function-level twin of the state test is
     named `test_student_state_is_empty_in_batch` so the preflight `-k`
     selects it.
2. **Slim `.cache/lama_ckl` before launch** (done 2026-10-15; `.cache/lama_ckl`
   is 124 MB). The launch uploads and the
   pull rsyncs the whole directory (`LAMBDA_CACHE_ARTIFACTS="lama_ckl …"`).
   Move `runs/dream-seed-42`, `runs/lora-seed-42`, `runs/mix-review-seed-42`,
   `runs/*-smoke`, `runs/*-seed-43`, `runs/report.json` and `acceptance/`
   to `.cache/lama_ckl_archive/fourth-box/` (beside the existing
   `third-box-ultra20k/` archive); keep only `runs/frozen-seed-42-r1`,
   `-r2`, `-r3` (41 MB each; the frozen arm has no optimizer and is not
   rerun). Set
   `LAMBDA_RESUME_CHECKPOINT="models/mamba2_2_7b/checkpoints/ultra20k-v4/epoch-1/step-9594"`
   in `scripts/.env`; move `models/mamba2_2_7b/checkpoints/epoch-1` (the
   fourth box's raw training checkpoints, which `ultra20k-v4/epoch-1`
   duplicates) to `.cache/lama_ckl_archive/fourth-box/`, since the launch
   preflight warns on `checkpoints/epoch-*` and the pull would merge into
   it. The report's
   `persistent_artifact_bytes` for the slimmed baselines will no longer
   match the banked runs; it is a diagnostic field and does not enter any
   comparison.
3. **Dream census script** `sft/experiments/lama_ckl/diagnostics/dream_census.py`,
   with a `lama-ckl-census` Makefile target (`ARGS` = the directory):
   given an `epoch-NN/` or a `work/epoch-NN.*/` directory, for every
   `cycle-NN/` that has a `dreams.json` and a `trainable.pt` (an
   incomplete cycle is skipped and named), prints the §3.3 census over all
   30 dreams, `max_dream_loss` and `subjects_mentioned` from
   `cycle-NN/distil.json` when present (the archived void cell has none;
   those columns stay empty), and the adapter step norm, the cosines to
   the steps 1, 5 and 20 cycles earlier, and the distance from A_0, from
   `--warm-start` (default `DEFAULT_WARMSTART`, resolved from `sft/`) and
   consecutive `trainable.pt` files; writes its table as `census.json` in
   the given directory. Done; run on the void cell, and the §3.3 table and
   §1.1 step figures agree with its output.
4. **`sft/Makefile`:** the `test` target forwards `ARGS` to pytest (the
   §3.2 preflight selects three tests with `-k`); `lama-ckl-census` (item
   3); and two `make lama-ckl-run` in the same second tee into one log, so
   the PID joins the log name.
5. **Run-note items deferred** from the fourth box's "things to discuss":
   warm-start batch size (60 % GPU at batch 6; decide before the next warm
   start, none planned), the distillation step's memory (≈2.1 GB per
   511-token dream caps the count near 40 on a 94.5 GB card;
   micro-batching inside the one step is a protected change for a later
   note), the Covenant School document's EOC-first reply, the probe's
   "role marker inside a reply" census; the bf16 question for the **warm
   start's own training** (`training/loop.py` trains the same bf16 LoRA
   tensors; with consistent gradients over 9594 steps the rounding cost is
   smaller than in the one-step-per-cycle arms, but unmeasured; a
   fifth warm start, if one is ever trained, should use fp32 master
   weights too); and the coupled seeds: wake turn i
   and dream i draw from the same CPU generator seed
   (`dream_generation_seed(dream_set_seed(seed, epoch, cycle), i, 0)` in
   both `protocol.py` and `dream_generation.py`), and the fourth box saw
   five identical wake replies under distinct per-turn seeds (sharp
   distribution, not a seed bug, but the coupling is worth separating).
6. **Experimenter skill** (done, commit `73f3d09`), every line that conflicts with §3.2: the
   session-order table points at this note's §3.2 for this run and its
   acceptance row reads `--dreams-per-cycle 30`; the PRE-GRID ACCEPTANCE
   block's pass conditions (`subjects_mentioned > 0`, `role_marker_closes
   0 of 5`, `forced closes 0 of 5`, "a miss is stop-and-report") are marked
   suspended for this run per §4.2 and §3.2 step 2, where the only stop is
   the loss floor; the "launch every cell with `--max-hours`" line gains
   the one-epoch exception of §3.2 step 4 (the baselines in step 3 do
   pass it); the per-dream loss maximum and
   the census join the curve table columns; the skill states that the
   governing note's rules, not only its commands, win where they differ.
7. **Reading path:** `notes/README.md` names this note first, then the
   10-13 debrief (done).

Dry run (debrief step 7, 2026-10-15): a fresh experimenter session read
the repo as a local invocation and audited the launch gate: every §3.2
step-0 check passes on the local copy; the gate is closed only by the
uncommitted protected edits of item 1 and the unpushed `main`, both the
teammate's actions. No misreading of the note surfaced.
