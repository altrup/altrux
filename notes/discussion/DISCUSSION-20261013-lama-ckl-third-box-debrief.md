# Discussion notes — 2026-10-13: LAMA-CKL third box run debrief

Debrief of the third LAMA-CKL box session
([`EXPERIMENT_NOTES-20261012-022401`](../experiments/EXPERIMENT_NOTES-20261012-022401.md),
branch `box/20261012-020915`, merged) and of the local replay of its first
cycle
([`EXPERIMENT_NOTES-20261012-124900`](../experiments/EXPERIMENT_NOTES-20261012-124900-cycle0-replay.md)).
It amends [`DISCUSSION-20261011`](DISCUSSION-20261011-lama-ckl-second-box-debrief.md)
§3.1 (warm start), §3.2 (probe), §3.3 (session order) and the
[wake/dream protocol](DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md)
(state carry, wake decoding, distillation loss, epoch count). Where this
note and an older note disagree, this note wins. The
[self-teaching end goal](DISCUSSION-20261010-self-teaching-end-goal.md) is
the goal every choice below is argued from.

## 1. What the run established

All four arms ran on the ultra20k warm start. Lora and mix-review completed
30 epochs, frozen its one, altrux was stopped at epoch 3 cycle 8 by the
teammate after the model had degenerated.

1. **The warm start fixed both 2026-10-11 gaps and exposed a third.**
   EOC-first replies 0 of 50 at every checkpoint; dreams open with `[USER]`
   10 of 10. But in cycle 0 of the altrux cell the reply to the first
   document was `[USER] Can you provide more information on the different
   starting quarterbacks…` followed by `[ASSISTANT]` and an answer: the
   model, given `[ASSISTANT]`, wrote a user question and answered it, 105
   tokens, EOS on its own. The warm-start probe had measured this as
   "`[USER]` openers 8 to 10 of 10" on Dolly contexts and the registered
   gate (EOC-first only) let it through. Cause: every Dolly row in the
   corpus is context, blank line, instruction, so a document with nothing
   asked of it is a turn the model has never seen, and with user turns
   trained it fills in the missing question itself.
2. **The altrux loop collapsed in three cycles, measured.** Cycle 0: 7 of 10
   dreams are ordinary conversations, 3 are replays of the wake transcript
   (turns dropped, reordered, duplicated, the assistant side sometimes
   missing). After the ten distill steps of cycle 0 the frame
   `Remember this for later:` as a dream opener goes from 3.5 % to 31 % from
   the wake state and up ×1700 from an empty state (local replay, per-step
   KL within 0.03 of the box). Cycle 1: 8 of 10 dreams open with the frame,
   5 of them invented passages in the template of cycle 0's first document
   ("When Primal Madness was canceled…"), 2 the cycle-0 document verbatim
   then this cycle's documents verbatim. Cycle 2: wake replies are verbatim
   copies, one copies the previous turn's document and hits the backstop.
   Cycle 3: 5 of 5 replies forced, copy plus confabulation to 128 tokens;
   5 of 10 dreams empty. Then a frequency ladder: `Yes` (cycle 14), `Yes`
   with spaces and `20` (cycle 99), newline (epoch 2 cycle 17), `20` alone
   (epoch 2 cycle 50), `e` (epoch 2 cycle 99 to epoch 3). Each rung is the
   most frequent token pattern of the previous rung's transcript.
3. **The gradient is shared, not carried by the copy.** At cycle 0 the three
   wake-related dreams carry KL 0.8 to 1.6 and LoRA gradient norms 0.5 to
   0.7; the seven ordinary conversations KL 0.01 to 0.03 and norms 0.08 to
   0.25. The frame-copy dream is no heavier than the two Q&A dreams, and
   the Q&A dreams are copies of wake reply 0, not dreams about the wake.
4. **The state was two thirds the model's own text by cycle 2.** Documents
   are 50 to 70 tokens; replies from cycle 2 are 129 tokens every turn
   (backstop plus fed EOS). The state is carried across cycles and never
   reset, so the in-context frame prior grows with the cycle count
   independently of training.
5. **Nothing in the loop could see the collapse.** Distillation loss stayed
   0.006 to 0.011 because the teacher is the collapsed model. The scorer
   read 0.932 (epoch 1) and 0.8765 (epoch 2) retention on a model emitting
   `Yes` and newlines, because it scores one object token at a fixed
   position. Only the decoded replies and dreams showed it, and they were
   read a day late.
6. **The wake decodes greedily, the dreams at 0.7.** Greedy is the argmax
   lock: once `Yes` is the top token at every position every reply is `Yes`
   forever. Reproducibility was the reason, and the run has none end to
   end: identical weights scored 0.9795 / 0.9665 / 0.9765 retention in
   three processes (likely Triton autotune choosing kernels per process),
   so a sampled dream diverges at the first token anyway.
7. **Baselines on the ultra20k base** (seed 42, 30 epochs): frozen
   0.000 / 0.9765; lora peak 0.092 at epoch 9 with 0.5175 retained, plateau
   0.074 / 0.465; mix-review peak 0.066 at epoch 5 with 0.6728 retained,
   plateau 0.047 / 0.630. Review holds retention 0.16 above lora's plateau
   on this base. The seed-43/44 rule's 0.02 threshold is about 1.5× the
   cross-process noise and was not evaluable (no complete altrux curve).
8. **Cost.** 145 s per cycle at 5 documents (wake ~20 to 43 s, dreams 70 s
   in two launches of 8, distill 46 to 83 s one dream per step), 4.0 h per
   epoch, 121 h per 30-epoch cell. GPU 11 % utilised.
9. **The warm start never saw the frame.** 0 occurrences of "remember this
   for later" in the UltraChat train split; no acknowledgement rows; the
   top-up never ran. The frame as a dream mode came from the state and the
   loop alone. The packer's `--repeat-rate 0.3` (5722 verbatim repeat
   boundaries) is a trained habit of reproducing an earlier conversation
   from state word for word; not tested as a cause, removed below.

## 2. Reinterpretation

The method as registered is a closed loop with no stabiliser: the wake
emits the model's most probable continuation, distillation raises that
continuation's probability in the weights, the next wake emits more of it,
and the carried state accumulates every turn. The collapse is not a
property of self-generated dreams; it is a property of greedy decoding,
length-weighted replay of the model's own text, a state that is never
cleared, and a warm start that taught "a document is followed by a question"
and "an earlier conversation comes back verbatim". Each of those is a
choice, and each is changed below without touching the two things the
teammate holds fixed: cycle n+1's teacher is cycle n's student, and no dream
is filtered.

The sleep literature maps onto this loop closely enough to use as a
checklist (2026-10-12 discussion): replay is sparse and selective; sleep
renormalises; cortex does not learn from itself during consolidation; old
memories are interleaved with new; the world comes back every morning. Of
these, "cortex does not learn from itself" is already true within a cycle
(teacher logits recorded at dream time, state fixed during distillation) and
the brain's teacher drifts across days as ours does. What the brain
guarantees and we broke is what fills the fast store: the day's episodes,
not the cortex's echo. The changes below restore that (short replies, state
reset, sampled wake) and add interleaving through the warm start, and leave
selection and renormalisation as the next levers if those are not enough.

The altrux arm is at a disadvantage against lora by construction: lora gets
500 documents verbatim under cross-entropy; altrux gets whatever ten dreams
per cycle say. That is the experiment. The line not to cross is handicapping
the dream path on top of that, which is why the loss mask on copied spans
(§4) is held and not registered.

## 3. Standing direction

### 3.1 Warm start, fourth recipe (replaces 2026-10-11 §3.1)

From scratch. Same LoRA config, one epoch, ten interval checkpoints, probe
on each (§3.2). No top-up stage. No string from the LAMA-CKL protocol in
the corpus.

**Sources.** UltraChat `train_sft`, the first 20,000 conversations. Dolly
context-bearing categories (`summarization`, `closed_qa`,
`information_extraction`): the pool of 4,417 rows after the fixed-seed
hold-out of 50 (`HELD_OUT_SEED = 0`, unchanged). The summarization rows
(about 1,200) carry a human-written summary as their response; it is used
below as a summary of the context.

**Dolly first occurrence.** Every pool row appears once in one of three
shapes, drawn per row with the probabilities given:

| shape | share | rendering |
|---|---|---|
| plain | 0.40 | `[USER] <context>\n\n<instruction>` → `<response>`; in half of these the instruction precedes the context |
| document only | 0.40 | `[USER] <context>` → `<acknowledgement>` |
| multi-document | 0.20 | two or three document-only rows in one conversation, each acknowledged in turn; rows grouped at random |

In the document-only and multi-document shapes, 15 % of document turns carry
a preface before the context (a short phrase such as a user would type
before pasting); 85 % are the bare context. The acknowledgement is: for
summarization rows, the human summary, in half of them preceded by an
opener; for other rows, an opener alone, or a one-sentence restatement where
the Dolly response is one. Openers, prefaces and recall requests (below)
each come from a set of at least 40 distinct phrasings written in
`preparation/dolly.py`, drawn uniformly, with a test asserting the set size
and that no set contains the protocol frame (`sft/AGENTS.md`, "No
hard-coded phrases in training data"). `ACK_TEMPLATES` and `--acknowledge`
are deleted.

**Dolly second occurrence (repeats).** The packer repeats a Dolly
conversation later in the same pack with probability 0.30, as today, but a
repeat is a variant drawn uniformly from those available to the row, never a
plain copy unless the draw picks "verbatim":

| variant | available to | rendering |
|---|---|---|
| verbatim | all | the first conversation again |
| question only | all | `[USER] <instruction>` → `<response>`, context absent |
| summary for document | summarization rows, document-turn shapes | the first conversation with the context replaced by its summary |
| recall request | all | `[USER] <recall request>` → `<context>` (or `<summary>` for summarization rows, "gist" phrasing) |
| multi-turn | all | as many of the row's distinct exchanges as it has (two for plain rows, three for summarization rows, up to three across documents for multi-document rows) stacked in one conversation in random order |

**UltraChat second occurrence.** Conversations with at least 6 turns (about
two thirds of the first 20k) are eligible. With probability 0.30 an eligible
conversation is split: k is drawn uniformly over the turn boundaries that
leave at least one exchange on each side; the first occurrence is turns 1 to
k; the second occurrence, later in the pack, replays the last j of those
turns verbatim (j uniform over 0 to k) and continues with turns k+1 to n.
j = k is the full-replay case; j = 0 the pure continuation. No separate
verbatim-repeat knob.

**Packer.** `--pack`, 2 to 5 conversations per example, `--max-len 8192`,
`--train-user-turns`, `--repeat-rate 0.3`. The flag keeps its name and its
value; what a repeat is changes: a Dolly repeat is a variant from the table
above and an UltraChat repeat is the (k, j) split-and-replay, and the plain
verbatim copy exists only as the "verbatim" Dolly variant or j = k. The
checkpoint tree is `models/mamba2_2_7b/checkpoints/ultra20k-v4/`, the split
`.cache/lama_ckl/mamba2_2_7b_ultra20k-v4`; `split.py`'s defaults and the
Makefile's `WARM_START_DIR` move with it, and `warmstart.sha256` is reset to
`unset` at the start of the local work so the split guard refuses a skipped
re-pin. The log prints a count per first-occurrence
shape, per Dolly variant, per UltraChat (k, j), prefaced turns, and the
must-be-zero invariants, plus one decoded example of every shape and
variant. No dataset goes to the box until those samples are read.

Rationale, each against the end goal (learn from conversations, converse
while doing it): the plain flip and the document-only shape remove the
"document is followed by a question" prior; the acknowledgement teaches a
short reply that keeps the state mostly document; the multi-document shape
is the wake's own shape; the question-only, summary and recall variants
teach reuse of what was read, in the state across other conversations,
without unprompted replay; the UltraChat continuation teaches the same for
things said in a chat, so the model does not learn that documents are
remembered and chats are not; the verbatim variants keep exact reproduction
available on request, which the scorer's single object token needs.

### 3.2 Probe (amends 2026-10-11 §3.2)

`make warm-start-probe` on every checkpoint, held-out Dolly rows under the
real frame, unchanged in method. It reports, per checkpoint:

- EOC-first, forced closes, and (new) **role marker inside a reply**: a reply
  whose tokens contain `[USER]` or `[ASSISTANT]`, counted and printed in
  full for the first three.
- (new) **reply length** distribution, and the **reply opener census**: the
  first word of each reply and its count, so a dominant opener is visible
  before the pin.
- The dream census as before, plus (new) the count of dreams that are
  verbatim copies of the probe document and of dreams with no content
  (EOC first).

Gate at the pin, over the 50 held-out replies: EOC-first 0, role markers at
most 4, forced closes at most 9, no single opener on more than 25. A miss is
stop-and-report; there is no top-up.

### 3.3 Protocol amendments (protected files; the teammate commits)

Each is a PROTECTED EDIT with its reason in the commit. The run settings
(`run.json`) record every value.

1. **One epoch per altrux cell.** A new `--epochs` flag on the runner,
   default 30; altrux cells launch with `--epochs 1`. Lora and
   mix-review keep 30 epochs; the comparison is at their epoch-1 point,
   which `epoch-01/result.json` already holds. Reason: production is one
   pass; the collapse happened inside epoch 1; 100 scored cycles are the
   curve.
2. **Save and score after every cycle.** The adapter (`trainable.pt`, not
   the optimizer) and a full evaluation after each cycle, into
   `epoch-01/cycle-NN/`. Cost ~43 MB and ~14 s per cycle. The epoch-end
   save keeps the optimizer for resume. The acceptance run (`--cycles 1`)
   stays unevaluated, as the flag is today.
3. **State reset at every cycle boundary.** Each wake starts from an empty
   state; dreams generate from that cycle's open state only; `carry` is
   removed. Amends the 2026-08-23 protocol. Reason: §1.4; the scorer probes
   from an empty state, so the carried state added no measured recall and
   two amplifiers.
4. **Wake replies sampled.** Temperature equal to `DREAM_TEMPERATURE`
   (0.7), with a per-cycle, per-turn seed derived like `dream_set_seed`,
   recorded in `wake.json` and `run.json`. Reason: §1.6.
5. **Role marker in a reply closes the turn.** A generated `[USER]` or
   `[ASSISTANT]` ends the reply at the marker (the marker is not fed), EOS
   is fed, and the turn is counted in a new `role_marker_closes` field
   beside `forced_closes`. Reason: §1.1; chat APIs treat role tokens as stop
   sequences.
6. **Distillation: one optimizer step per cycle, per-turn normalised.** All
   dreams of a cycle in one batch, one AdamW step at the registered
   learning rate. The loss is a weighted KL over scored positions: the dream
   is split into turns at role markers (EOS closes a turn), each turn with
   content gets 1/T of the dream's weight, each token 1/(T·n) within its
   turn; a turn with no tokens after its marker gets no share. The dream's
   terminal EOC is dropped from the scored positions. Reason: §1.3, §1.4;
   length-proportional weighting is a frequency weighting and the collapse
   was a frequency collapse; the terminal EOC was the source of EOC-first
   dreams. Empty dreams contribute nothing, as today.
7. **Dreams in one launch.** `--dream-batch-size` equal to
   `--dreams-per-cycle` (10). Reason: §1.8; the GPU is latency-bound.
8. **Frozen cell scored three times per box.** Three processes, outputs
   `runs/frozen-seed-42-r1`, `-r2`, `-r3`; the report reads them as one
   frozen arm and prints the mean and the spread; the spread is recorded in
   the run notes as the evaluation resolution. The seed-43/44 rule's
   threshold becomes 0.05 absolute (2026-10-10 §3.7 item 5 amended), compared
   at the end of epoch 1: altrux at cycle 100 against each baseline's
   `epoch-01/result.json`, on acquisition and on paired retention. Altrux's
   per-cycle peak is reported but is not the comparison point (a maximum
   over 100 noisy points is not comparable to one).
9. **Report.** `make lama-ckl-report` accepts a one-epoch altrux curve with
   per-cycle points and compares arms at matched epochs; it refuses nothing
   it can describe.

Not changed: `--docs-per-wake 5` (passed explicitly; the code default is
10) and `--dreams-per-cycle 10` (the 2026-10-11 wake-size rule stands); KL temperature 1.0; the dream prompt (one EOC); the
teacher (the current weights at dream time, logits recorded); AdamW 1e-4.

### 3.4 Session order (replaces 2026-10-11 §3.3)

1. Warm start (§3.1), probe on each checkpoint (§3.2). Final checkpoint is
   the candidate; gate at the pin. Pin the SHA.
2. Split (`--lama-root ../.cache/LAMA/data`).
3. Frozen cell three times (§3.3.8); record the spread before any other
   cell.
4. Acceptance, `--arm altrux --cycles 1 --docs-per-wake 5`: `internal_eoc`
   0, `role_marker_closes` 0 of 5, forced closes 0 of 5, dreams non-empty
   with `subjects_mentioned` above 0; read the five replies and ten dreams
   and quote one of each in the run notes. Record the cycle's wall time per
   phase and the altrux ETA for 100 cycles with per-cycle evaluation. A
   miss: stop and report.
5. Smokes of all four arms.
6. Cells, seed 42: lora (30 epochs), mix-review (30 epochs), altrux
   (`--epochs 1`, 100 cycles, per-cycle save and score). Each cell launches
   with `--max-hours` set to the remaining session budget. The session soft
   cap is 20 h wall clock (confirmed 2026-10-13; it was an assumed value
   since 2026-10-10).
7. Seeds 43 and 44 for all three trained arms if, at seed 42, altrux differs
   from the best baseline by at least 0.05 on peak acquisition or paired
   retention at epoch 1. Otherwise seed 42 only.
8. Shutdown checklist.

Stopping mid-cell stays the experimenter's judgment under the rule committed
2026-10-12 (`.agents/skills/altrux-experimenter/SKILL.md`): finish the
current cycle's evaluation, then stop and report with the numbers and one
decoded reply and dream. There is no hard-coded collapse trigger. The
check-in reads decoded replies and dreams, not only counts.

### 3.5 What the run must answer

With the cycle as the unit: does acquisition rise above the frozen floor
plus the evaluation resolution within 100 cycles, and at what retention?
Does the reply stay an acknowledgement (length, opener census,
`role_marker_closes`) through the epoch? Do dreams keep mentioning wake
subjects (`subjects_mentioned` per cycle), and what fraction are verbatim
copies versus rehearsals? Those four curves, per cycle, are the result; the
epoch-1 comparison against lora and mix-review is the headline.

## 4. Explicitly considered and rejected

1. **Frozen teacher per epoch** (snapshot the adapter, distil from its
   logits). Rejected by the teammate: cycle n+1's teacher being cycle n's
   student is the method. The brain's consolidation teacher drifts across
   days too (§2).
2. **Dream filter** (distil only dreams with `subjects_mentioned` above 0).
   Rejected: no filtering. It would also hide the loop's health behind a
   gate.
3. **Loss mask on verbatim wake-transcript spans.** Held, not registered.
   It removes copies from the gradient entirely, and copies are the most
   faithful carrier of a fact into the weights; per-turn weighting reduces
   their dominance and keeps them. Revisit if §3.5 shows copies still
   dominating dreams with per-turn weighting in place.
4. **Mixing real documents into the distill step.** Rejected: makes altrux
   a worse mix-review.
5. **No-reply wake.** Rejected 2026-10-11, stays rejected: the model must
   converse. "Dream from a documents-only state, converse on the full
   state" was proposed and withdrawn: the model's own replies belong in
   the state, their length is the problem, and §3.1 fixes the length.
6. **Sequential dreams, each from the state after the previous.** Rejected:
   writes the model's own output into the state (~5000 tokens of dreams on
   ~300 of documents), the echo problem ten times over; loses batching.
7. **Empty dream prompt or no EOC token.** Rejected: the EOC is the slot the
   warm start now trains (document, acknowledgement, EOC, reuse); without
   it the dream is one turn long or has no end, the boundary diagnostics
   go, and the end goal needs a conversation boundary. EOC-first dreams
   are addressed by dropping the terminal EOC from the loss (§3.3.6).
8. **Hard-coded collapse stop rule.** Rejected (2026-10-12): the experimenter
   stops on judgment.
9. **Acknowledgement template rows and a fixed preface.** Rejected: one
   literal becomes a tic (`Yes` → `YesYesYes…`). Diverse sets or human text
   only.
10. **UltraChat summaries, Dolly summaries for closed-QA and extraction
    rows** ("same conversation, document replaced by its summary" for rows
    without a human summary). Needs a generation pass; optional later.
11. **Batched wake decoding.** Not possible: the five turns go through one
    state in order, and batching across cycles is the frozen teacher.
12. **Fp32 or batch-1 scorer, pinned autotune.** Deferred: measuring the
    spread and widening the threshold (§3.3.8) is enough until arms differ
    by under 0.05.
13. **Top-up stage.** Removed: its trigger (EOC-first) was the wrong gate,
    and the fourth recipe teaches the acknowledgement directly.

## 5. Local work before the next box

This is the debrief's work, done here with the teammate under the standing
go-ahead gate; an experimenter session that finds it undone stops and
reports (dry run, 2026-10-13). Tests first. In order:

1. Done (`fa803f2`). `preparation/dolly.py`: emit rows carrying `context`, `instruction`,
   `response`, `summary` (summarization rows) as separate fields; the three
   first-occurrence shapes with the shares in §3.1; openers, prefaces and
   recall-request sets of at least 40 each with a test on size and on the
   absence of the protocol frame; delete `ACK_TEMPLATES` and
   `--acknowledge`. Held-out partition unchanged (test pins it).
2. Done (`fa803f2`). `preparation/conversations.py`: repeat variants for Dolly rows and the
   (k, j) split-and-replay for UltraChat; `--repeat-rate` keeps its name
   and now draws variants; per-shape and per-variant counters and one
   decoded example of each in the log; invariants (role adjacency,
   unanswered user turns) extended to the new shapes.
3. Done (`ef8e8c1`, `1693e96`). `experiments/lama_ckl/probe.py`: role-marker count with full prints,
   reply length, opener census, verbatim-copy and empty-dream counts.
4. Done (`0ff672d`, committed by the teammate). PROTECTED EDITS: `runner.py`
   (one-epoch altrux, per-cycle save and score, state reset, dream batch =
   dreams per cycle, frozen ×3 target), `protocol.py` (wake temperature and
   seed, role-marker close and count), `dream_distillation.py` (batched
   single step, per-turn weights, terminal EOC dropped), `report.py`
   (one-epoch curves, matched-epoch comparison). Tests for each live beside
   the existing ones. `report.py` is not protected: done (`0189519`).
5. Done (`fa803f2`). `sft/Makefile`: `warm-start` target with the new recipe
   (flags unchanged, `--repeat-rate 0.3` kept); remove `warm-start-topup`; a `frozen-x3` convenience
   if the runner does not provide it.
6. Done (`dbc0848`). `.agents/skills/altrux-experimenter/SKILL.md`, stale against this note
   in these places (dry run): the top-up rule (gone), the pin gate (§3.2's
   four counts), acceptance (5 documents only), cell length (one epoch, 100
   cycles, ETA for 100 cycles plus 101 evaluations), session order (frozen
   ×3 before acceptance), the seed threshold (0.05 at epoch 1), the
   quick-reference paths (`ultra20k-v4`), `role_marker_closes` in the
   healthy-cycle check and the monitor line, and the check-in reading decoded
   replies and dreams.
7. Done (`fa803f2`, `ef8e8c1`, `0189519`). `sft/README.md` and `sft/.env.example` for every flag above.
8. Dry run by a fresh experimenter session (debrief skill step 5), again
   after items 1 to 7 are in; the first dry run (2026-10-13, before any of
   them) is folded in above. The second dry run ran 2026-10-13; its
   findings are folded into the skill: EXIT markers in the Makefile targets,
   `grep -a`, no cap or resume inside the one-epoch cell, probe at finalize,
   probe default 5 per wake, seed-rule wording, report before seed 43,
   archive old runs. Code fixes from it: the report compared
   `reply_temperature` across arms and refused any report holding altrux
   beside a baseline (`84bd7ac`); probe defaults (`6c4e7b3`); EXIT markers
   (`074d0cc`).

## 6. Housekeeping

Done this debrief (commits on `main`):

- `c46d1d7`, `387bec6`: `sft/logs/` gitignored and untracked.
- `767c1ed`: one-off scripts live under their experiment folder
  (`AGENTS.md`, experimenter skill); the second-box diagnostics moved to
  `sft/experiments/lama_ckl/diagnostics/`.
- `2deb201` (box branch, merged): the branch fallback commit happens only
  for artifacts missing from the pull receipt; the 240k-line forced copy
  left the branch.
- `dc99565`: the cycle-0 replay note and `replay_cycle.py`.
- `e52f434`: experimenter may stop a cell on judgment.
- Global `CLAUDE.md` (`e85fcd0`): argue from the stated goal; ponytail
  governs code only. `sft/AGENTS.md`: no hard-coded phrases in training
  data. Debrief skill: read the decoded text behind every count.

Open:

- The three stale local stashes and the local `altrux` tmux session.
- The watchdog delay-file change agreed 2026-10-10, still after this run.
- Per-cycle resume for the one-epoch altrux cell (a crash at cycle n
  restarts at cycle 0): not implemented; decide before a box with less than
  a full ETA of budget.
- Done 2026-10-13: the third-box `runs/`, `acceptance/` and both old splits
  moved to `.cache/lama_ckl_archive/third-box-ultra20k/`, outside the
  `lama_ckl` upload pattern.
