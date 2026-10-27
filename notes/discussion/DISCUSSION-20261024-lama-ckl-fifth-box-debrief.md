# Discussion notes — 2026-10-24: LAMA-CKL fifth box run debrief

Debrief of the fifth GH200 session (2026-10-21 14:46 to about 17:05 UTC,
run note `notes/experiments/EXPERIMENT_NOTES-20261021-144623.md`, box
branch `box/20261021-140642`, merged into `main` as `dbc78a8`). It amends
[the 2026-10-14 fourth box debrief](DISCUSSION-20261014-lama-ckl-fourth-box-debrief.md)
and [the 2026-10-13 third box debrief](DISCUSSION-20261013-lama-ckl-third-box-debrief.md),
which stay governing for everything this note does not change. Where they
differ, this note wins. The
[self-teaching end goal](DISCUSSION-20261010-self-teaching-end-goal.md) is
the goal every choice below is argued from.

What this note does: reads the corrected dream cell, which collapsed in
seven cycles on a new carrier (§1); names the cause as the mechanism's
registered limitation realised, plus a second structural finding, recency
(§2); changes the direction from "fix the loss" to "train the dreamer"
(§3), with the one-document wake as the cell protocol that follows; records
the design discussion so the next session does not redo it (§4); and lists
the local work (§6). Cycles are numbered 0 to 99. Discussed 2026-10-24 to
2026-10-26 (the dreamer reward thread of §3.2 and §4 is the 10-25 and 10-26 part).

Every derived number below comes from a computation over the whole artifact
run in this session, named in brackets after the number. The artifacts are
`.cache/lama_ckl/runs/dream-seed-42/work/epoch-01.nnmqucxb/cycle-NN/`
(`wake.json`, `dreams.json`, `distil.json`, `result.json`, cycles 0–9),
`runs/dream-seed-42/epoch-00/result.json`, `census.json` beside the
cycle directories (step norms and cosines), and
`runs/{lora,mix-review}-seed-42/summary.json`. Two text rules recur and
are pinned here: a dream **revisits** a wake document when it shares at least two
distinct word trigrams with it at non-overlapping positions (lowercase
`\w+` tokens, over the dream text with the EOC prompt token stripped and
the turn's `document` field of `wake.json`), counting each distinct trigram at its first occurrence in the dream and
only trigrams that occur in exactly one document of the **document set**
the rule is applied over (here the run's 50 wake documents; for a cell,
its split's 500 documents; for a dreamer round, its 50 rows plus the prod
rows, a fact row's document being its passage and a conversation row's
its prefix text; a trigram two documents share, `museum of art`, `of the roman`, is
generic and does not count; the 10-24 rule had no such filter and counted
four generic matches, §5); a dream carries a **short assistant turn** when the text of
some `[ASSISTANT]` turn, read to the next newline, EOS or role marker, is
one to eight words, and that turn is an **acknowledgement** when it opens
on Got it / Okay / Sure / Thanks / Understood / Noted.

## 1. What the run established

### 1.1 The fix landed, and the loop collapsed anyway, on the wake reply

The corrected mechanism (10-14 §3.1: student replays from an empty state,
fp32 master weights) did what it was built to do. Acceptance read a
per-dream loss maximum of 0.626 and a mean of 0.183 against the void cell's
0.0034 (run note); the cell's cycle 0 read 0.879 / 0.160 (`distil.json`).
Teacher and student were different functions. The 10-14 §6.1 tests passed
on the box.

The cell then collapsed in seven cycles. The carrier was the wake's own
reply, ` Got it.`, not the frame. Counts over all 30 dreams of each cycle
[`'Got it' in text` over `dreams.json`, prompt token stripped]:

| cycle | dreams containing "Got it" | dreams opening on "Got it" | dreams with no `[USER]` | wake replies |
|---|---|---|---|---|
| 0 | 1 | 0 | 0 | ` Got it.` ×5 |
| 1 | 2 | 0 | 0 | ` Got it.` ×5 |
| 2 | 4 | 0 | 0 | ` Got it.` ×4, ` Thanks, I've got it.` |
| 3 | 9 | 0 | 0 | ` Got it.` ×5 |
| 4 | 14 | 1 | 0 | ` Got it.` ×5 |
| 5 | 27 | 21 | 0 | ` Got it.` ×5 |
| 6 | 30 | 30 | 12 | ` Got it.`, then ` Got it.` ×3, ×4, ×6, ×6 |
| 7–9 | 30 | 30 | 30 | ` Got it. Got it. …` forced close at 128, 4/5 then 5/5 |

Growth of 1, 2, 4, 9, 14, 27, 30 from one dream at cycle 0. The wake broke at cycle 6
(replies lengthen with the turn), and from cycle 7 every dream is the
string ` Got it.` repeated to the 511-token budget (`unique_dreams` 1: all
30 identical, copy fraction 1.00 against a transcript that is itself
mostly `Got it.`). The per-dream loss maximum fell 0.029 → 0.017 → 0.009
over cycles 7–9 because the empty-state student predicts the chain exactly;
the 0.01 WARNING fired at cycle 9. The experimenter stopped during cycle 10
on the kill-on-collapse judgment (10-13 §4.8). Cycles 0–9 are complete;
`epoch-01/` was never written; no report.

The scorer saw none of it. Retention over `epoch-00` and cycles 0–9: 0.985,
0.979, 0.985, 0.983, 0.978, 0.982, 0.979, 0.979, 0.977, 0.981, 0.976
[`result.json`], within the warm start's cross-process spread. To-learn
0.001–0.004 throughout, which is the four flicker rows of 10-14 §2.1 (211,
254, 355, 357 are the only rows that ever scored). **Acquisition census at cycle 9: zero** [an in-session check over
`result.json`: a to-learn row counts as acquired when its descriptive accuracy is 1.0 (the scorer's "known" rule) at the
first evaluation at or after the cycle it was shown (wake order is row
order, cycle = row index // documents per wake) and 0 at every
earlier evaluation; it is kept when it is also 1.0 at the last
evaluation; rows at any accuracy above 0 at any evaluation of the split's frozen ×3, or at any evaluation before their own wake cycle, are flicker rows and are left out of the count and of the denominator (the frozen runs alone miss rows that flicker between modes: the v4 frozen ×3 score only row 355, while 211, 254 and 357 flicker 8:4 over twelve evaluations, 10-14 §2.1, so the before-wake test is the second detector, and a row that first flickers after its wake is the residual, recorded); the only rows that ever scored are 211, 254, 355 and 357,
outside rows 0–49, so the zero is exact; §3.1 and §6.2b carry the same definition]. Free generation was destroyed while the
teacher-forced readout was untouched: the two measure different things,
and the census, not the curve, is what reads the loop's health.

### 1.2 Where the shape came from, cycle by cycle

Read from the text [every dream of cycles 0–2 with a short assistant turn
or containing `Got it`, by the pinned rules; losses from `distil.json`]:

- **Cycle 0** (before any step): seven dreams carry a short assistant turn
  (7, 12, 13, 15, 17, 18, 28); four of those are titles or list heads
  (`Introduction:`, `1. Hummus`, a series title). Three are
  acknowledgement-shaped and two of them acknowledge a document: the
  wake-shaped `[USER] <Gettysburg sentence> [ASSISTANT] Got it.` (dream 13,
  loss 0.328) and `Okay, I've read it.` (dream 7, 0.205). The warm start's
  natural rate of acknowledgement rows. One dream revisits a wake document
  (§1.3). Acknowledgement-shaped short turns to revisits at cycle 0: three
  to one by the pinned rule (dream 12's `Sure, here you go:` counts as one
  though it acknowledges no document), document acknowledgements two to
  one.
- **Cycle 1**: two dreams contain the string, both wake-shaped (a pasted
  document, then `Got it.`: dreams 14 and 28, losses 0.118 and 0.065).
  Nothing is *about* saying Got it.
- **Cycle 2**: the first dreams that are nothing but the exchange: `[USER]
  I've got it. [ASSISTANT] Got it.` (dream 9, 0.575, the highest loss of
  the cycle) and `[USER] I've read the passage. [ASSISTANT] Got it. …`
  (dream 20, 0.340, third). Second is the cycle's one revisit (dream 15,
  0.458, the French car industry document). Two Adam steps turned an
  in-context habit into a weight habit.
- **Cycle 4**: 14 dreams carry the exchange, 10 of them with no wake
  content by the revisit rule (dreams 1, 5, 7, 12, 14, 17, 19, 22, 26,
  29), e.g.
  dream 1: `[USER] That's great to know! [ASSISTANT] Got it.<eos>[USER]
  Wow, it's amazing how much art …` and then a normal conversation. The
  wake's exchange with the document removed.
- **Cycle 5**: `Got it.` as the dream's opener on 21 dreams (14 as `[USER]
  Got it`, 7 with no marker before it); chains
  (`[USER] Got it.<eos>[USER] Got it. [ASSISTANT] Got it. …`); a
  confabulated NHL document acknowledged on both sides and repeated.

The mechanism, in one line: from an empty state, the only thing the teacher's
behaviour consistently looks like is "a user turn is followed by Got it".
The documents are the part that changes every cycle, so the part that
accumulates least across steps (§2). Step norms fell 0.467 → 0.209 over cycles 0–9 (`census.json`; the
cycle-0 value 0.46662 sits 1 % above Adam's first-step bound lr·√d =
0.4618, §4.26, which the tenth read's local recomputation over the banked
adapters puts down to reduction spread in the census, not to the step)
and the lag-1 cosines between consecutive steps were 0.82, 0.90, 0.94,
0.92 for the first four pairs and up to 1.00 later (`census.json`
`cos_lag1`). Adam's momentum alone (β₁ 0.9, β₂ 0.999, bias-corrected) gives a
lag-1 cosine of 0.60 for steps 1→2, 0.74 for 2→3 and 0.80 for 3→4 on
independent Gaussian gradients [simulated this session, d = 2·10⁵], which
is consistent with the fourth box's void cell (0.62, 0.74; a bf16
quantised readout, 10-14 §1.1, so a rough match, not a clean one). The run's
early cosines sit above that null, so part of the direction was shared
from the first step; `cos_lag5` of 0.48–0.66 over cycles 5–9 says it was
not one fixed direction. The cosines point the
same way as the text and do not by themselves name the carrier.

**Why one "Got it." became a chain.** Measured from an empty state under the
saved adapters [`sft/experiments/lama_ckl/diagnostics/diag_gotit_prior.py`,
run this session on the local GPU, output in
`sft/logs/diag_gotit_prior-20261025.log`; `cycle-NN/trainable.pt` is the
adapter after cycle NN's step. A first run on 10-24, not saved, gave the same picture with entries
differing by up to 0.04 (e.g. 0.127 against 0.158 after cycle 1): the forward is not bit-reproducible on this GPU, and the table is
the saved log's run]:

| context → next token | warm start | after cycle 1 | after cycle 4 | after cycle 5 |
|---|---|---|---|---|
| `[USER] <sentence> [ASSISTANT]` → ` Got` | 0.004 | 0.158 | 0.958 | 0.981 |
| `[ASSISTANT] Got it.` → EOS | 0.142 | 0.288 | 0.609 | 0.601 |
| `[ASSISTANT] Got it.` → ` Got` | 0.004 | 0.005 | 0.024 | 0.104 |
| `[ASSISTANT] Got it. Got it.` → ` Got` | 0.148 | 0.171 | 0.156 | 0.173 |
| `<EOC>[USER]` → ` Got` | 0.000 | 0.000 | 0.038 | 0.829 |

The probe's contexts are one sentence with no frame and no document, so
they read the prior in a reduced context, not the wake's; the direction of
every change is the finding, not the absolute values.

The repetition prior was in the warm start: after two `Got it.` a third was
already 15 % likely (ordinary in-context repetition bias), and that row
barely moved (0.148 → 0.173, inside the 0.04 run-to-run spread), while
EOS after two `Got it.` rose from 0.070 to 0.471 against chains [the
log's `asst Got it. x2` rows]. What distillation changed is the first
step: ` Got` after a document ×240, after `<EOC>[USER]` from 0 to 0.83,
and after one `Got it.` ×26 (0.004 → 0.104), while EOS after one `Got it.`
rose only to 0.60. EOS has to win every time, and the second ` Got` is
where it lost. Cycle 5's dreams
were sampled from the adapter after cycle 4, where a second ` Got` after
`Got it.` stood at 2.4 %; over 30 dreams of up to 511 tokens that is
enough draws for a chain to start, and once started its own context pushes
the next to 15.6 % (the warm start's own repetition level, unchanged by
distillation) and up. Cycle 5's dreams contained such same-turn runs (dream 19, three `Got
it.` in one turn; dreams 15, 17 and 21, two) [`(?:Got it\.\s*){2,}` within
one turn over `dreams.json`]; distilling them taught the student that chains are normal
text (10 % by the next adapter), and the wake replies chained at cycle 6. A
habit that the warm start held in balance (say it once, then stop) was
tipped by raising one side of the balance everywhere.

### 1.3 Revisits: more than counted, and only the last document

Two findings behind the exposure reading that the governing note got wrong.

**`subjects_mentioned` does not count dreams.** `protocol.py` counts the
wake *rows* (max 5) whose LAMA probe subject string appears in at least one
dream. The probe subject is not the document's topic: the cycle-4 Rauschner
document's subject is `Historic New England`, present in 3 of the 6
dreams revisiting the Rauschner document, reported as 1. The 10-14 note's "about one in thirty" (§3.1, §3.3) rests on that counter. Dreams revisiting a wake document, by
the pinned trigram rule [over `dreams.json` × `wake.json` turns]:

| cycle | per wake turn 0..4 | dreams revisiting any | `subjects_mentioned` | loss ratio, revisits / rest |
|---|---|---|---|---|
| 0 | 0 0 0 0 1 | 1 | 1 | 4.5 |
| 1 | 0 0 0 0 0 | 0 | 0 | – |
| 2 | 0 0 0 0 1 | 1 | 1 | 7.4 |
| 3 | 1 0 0 0 1 | 2 | 1 | 6.3 |
| 4 | 0 0 0 1 6 | 7 | 1 | 4.0 |
| 5 | 0 0 0 0 2 | 2 | 1 | 2.7 |
| 6 | 0 0 0 0 1 | 1 | 1 | 1.4 |

The revisiting dreams tend to carry the top losses of their cycle (the
revisit is the cycle's maximum in three of the six cycles with at least one and
second in two; cycle 0: 0.649 on the Lystad revisit, second to a 0.879
title, against a mean of 0.14 on the rest; cycle 4: 0.670 on a Rauschner
revisit, the cycle's maximum). Pooled over cycles 0–6, the 14 revisiting dreams average 3.34× the loss
of the 196 others [`dream_losses` in `distil.json` indexed by the revisit
sets]; the cycle-2 top loss is the pure exchange, not the revisit (§1.2). The loss responds to context-dependent
content, facts and shape alike; it does not prefer the facts.

**Mostly the last document.** In seven cycles, 12 of 14 revisiting dreams
read from wake turn 4, the most recent document; the exceptions are cycle
3 dream 15 (turn 0) and cycle 4 dream 3 (turn 3). In four of the seven
cycles (0, 2, 5, 6) no document but the last was revisited. Whatever the
distillation does, acquisition under the five-document wake is concentrated
on the most recent row. This is a recency property of the state the dream
starts from, not of the loss, and no change to the loss addresses it.

### 1.4 Baselines under fp32 master weights: recorded, no claim

[`summary.json` curves, `adapter_dtype` = `fp32-master` in both]:

| arm, seed 42 | epoch 1 | peak | epoch 30 | fourth box, bf16 |
|---|---|---|---|---|
| lora | 0.020 / 0.903 | 0.085 at epoch 7 (0.486) | 0.066 / 0.500 | 0.021 / 0.899; peak 0.098 at 9 (0.538) |
| mix-review | 0.021 / 0.901 | 0.059 at epoch 25 (0.475) | 0.055 / 0.475 | 0.023 / 0.911; peak 0.067 at 11 (0.502) |

Epoch 1 equal within the 0.006 flicker. Peaks lower under fp32 by 0.013
and 0.008, one seed each. The 0.013 is outside two flickers, but the
cross-seed spread of the same arm under bf16 is the same size: the fourth
box's seed 43 reached 0.086 (lora peak, epoch 4) and 0.064 (mix-review at
epoch 13, the last epoch its partial run reached, not a peak) against seed
42's 0.098 and 0.067 (10-14 §2.1, line 161; the fourth-box run note
`EXPERIMENT_NOTES-20261013-192947.md:245` carries the mix-review value and
lora seed 43 at epoch 20). Mix-review's rise is slower under fp32 and its retention drifts 0.50 →
0.475 over epochs 17–30. The dtype change did not raise either baseline and
this run cannot say it lowered them. These are the comparators for any cell
on `ultra20k-v4`; a cell on a different checkpoint reruns them (§3.4).

### 1.5 Two observations for the record

- **The wake is not reproducible across processes.** Acceptance and cell
  cycle 0 share seeds and adapter and sampled different replies (` Got it,
  thanks for passing that along.` ×5 against ` Got it.` ×5), so S_0
  differed and the losses differed (max 0.626 vs 0.879). Per-cycle readings
  are one draw. Five identical replies under five distinct seeds is a sharp
  reply distribution at temperature 0.7, as the fourth box also saw.
- **Timing.** About 67 s per cycle before evaluation at 5 documents and
  30 dreams [cycle log timestamps, cycles 2–7: 68–74 s per cycle with
  evaluation; the run note's 88.8 s (line 141) is the runner's
  `elapsed/done` after cycle 1 with the scoring and first-cycle overhead
  inside it; the acceptance cycle read 84 s (wake 5 s, dreams 51 s,
  distillation 27 s); after the collapse a cycle took about 117 s, the
  511-token chains]; evaluation 6.6–6.9 s per cycle [`evaluation_seconds`
  in `result.json`; the run note's 15 s was the acceptance evaluation with
  its prints]. Dream generation is the clock.

## 2. Reinterpretation

**The collapse is the registered limitation, realised.** 10-14 §3.1 said
the KL from the empty state carries everything the teacher's context
contributed and cannot separate the document facts from the shape of the
wake conversation, and deferred the mitigation (§4.7, §4.8). This run
shows the two signals side by side at cycle 0: two dreams acknowledge a
document, one revisits one [§1.2, §1.3]. The acknowledgement points the
same way every cycle (the same reply after every document); the revisit
points somewhere new every cycle. A consistent direction accumulates and a
changing one does not, whatever the per-cycle ratio (so a dreamer that only lowers the shape's share does not by this argument survive; §3's bet, stated as a hypothesis in §4.38, is that selection, the hinges and replay push the share under the rate at which 100 steps accumulate it, and the three cells test exactly that), and the collapse
followed.

**Two boxes, two carriers, one cause.** Third box: the frame, collapse at
cycle 3. Fifth box: the reply, collapse at cycle 7. Removing the literal frame from the corpus (fourth recipe) changed the
carrier; the third box's cycle carried ten distillation steps with a
carried state (10-13 §1.2), so the two collapses are about 30 steps
against 7, under at least four protocol differences, and no cycle count
compares them. The model's own repeated text is the most consistent
thing in S_c, and whichever literal is most consistent is the next
carrier. Removing literals one at a time does not converge.

**The learning rate is not the cause.** Adam normalises each coordinate's
step toward lr; a consistent direction gets a full step every cycle, a
changing one averages toward zero. The shape-to-fact ratio of movement is
set by the data, not by lr, so a lower lr delays the collapse in cycles
and moves the facts by the same smaller amount per cycle: in this linear
picture each document's own movement is proportional to lr (it is shown
once) and only the total over all documents before collapse is free of
lr. Once the wake replies change (cycle 6) the loop feeds back on itself,
and lr then sets whether that drift runs away.

**The optimizer is part of the mechanism.** Every arm trains with AdamW
(lr 1e-4, weight decay 0.01, `runner.py:808`; 08-23 protocol, 10-13 §3.3). Two Adam
properties bear on a habit: per-coordinate normalisation equalises step
sizes across coordinates, so a coordinate that carries only the small,
consistent habit component steps as far as one carrying a large fact
component (inside one coordinate the habit-to-fact ratio is unchanged);
and β₁ momentum carries a consistent direction across cycles. Plain SGD
with momentum 0 has neither: a component small in the gradient moves the
weights slowly, and nothing carries it between cycles, though a
consistent component still accumulates linearly. This run's readings
cannot measure the difference (the lag-1 cosines, §1.2, sit above the
momentum null but do not name what was shared), so SGD for the dream arm
(§3.4) is a treatment whose effect this run cannot predict, not a known
fix, and no AdamW dream control is scheduled (§4.2). The change also buys
two things: the dreamer's reward (§3.2) then describes the step the cell
takes, and the gate (§3.5) reads the one-step quantity behind the
argument, the gradient norms of revisiting against acknowledgement
dreams. What decides the argument is consistency across cycles, which no
one-step reading measures; the cell census's per-cycle acknowledgement
count is where it is read. The premise that the habit is small in the
gradient is also not clean: the cycle-2 top loss was the pure exchange
(0.575) and the cycle-0 wake-shaped acknowledgement read 0.328 (§1.2).
The loss weighting is part of the same mechanism, for one shape only:
`turn_weights` weights the turns of the *dream*, so inside a long
wake-shaped dream (a pasted document, then `Got it.`, the cycle 0–2 shape)
the per-turn weights gave the 4-token ` Got it.<eos>` turn the same weight
as the document turn beside it, about 10× per token (the document turns
inside the wake-shaped dreams are confabulated one-liners, dream 13 about
26 words and dream 7 about 40 against a 2–4-word acknowledgement; the
exact per-token ratio over all wake-shaped dreams is a §6.1 census
computation, not yet run; the wake's own documents are longer, median
prompt 78 tokens, range 47–432 [`prompt_token_ids` over `wake.json`,
cycles 0–9]). A short dream that
is nothing but the exchange (cycle 2 dream 9, the shape that dominated
from cycle 4) has about 1/N per position under both schemes, so the
uniform weighting of §3.4 does not touch it; §4.25 records that cost and
selection in the dreamer is what addresses it.

**Gradient descent on text cannot tell a fact from a habit** (teammate's
concern, recorded). Lora with no dreams at all costs half the retention by
its peak. Any method that trains on text installs whatever the text
consistently contains; the dreamer track and the ruler are both "choose the
text", which is the largest lever and the one not yet pulled. The
knowledge-editing literature (ROME, MEMIT, MEND) and complementary learning
systems point at a separate fast store with slow interleaved consolidation
if choosing the text is not enough. That architecture was retired in
October with reasons; the condition that reopens the question is a
selection-trained dreamer that still erodes free generation over a
one-document epoch (§3.4).

**Two problems, not one.** The shape transfer collapsed the loop; the
recency of what dreams revisit would have capped it had it survived.
Neither is a property of the loss. Both are properties of what is
distilled: the dream text, and the reply-holding teacher state the dream
is read from (§4.38: ` Got` after a document rose 0.004 → 0.158 by the
adapter after cycle 1, when 3 of the 60 distilled dreams held the string,
so the reply can reach the student through the teacher's logits at every
scored position, whatever the dream says; the run cannot separate that
path from §2's own Adam account, three acknowledgement dreams at full
per-turn weight under per-coordinate normalisation, and the SGD cells
with uniform weights are where the two come apart). Training the dreamer changes
the first; only the wake reply changes the second. The direction changes
from fixing the distillation to training the dreamer (§3) as the
hypothesis §4.38 states, with the ruler (§4.1) as the next lever if the
three cells collapse alike.

**What the thesis still has.** An empty-state student is pushed toward what the
teacher's context changed, facts and shape alike (§1.3: 3.3× loss on
revisits, pooled). The distillation does its job on whatever it is given;
the dreams fed to it are not chosen for it.

## 3. Standing direction

### 3.1 Goal block (prime directive from the 10-10 self-teaching note; amends 10-13 §3.5, "what the run must answer")

- **Prime directive, unchanged:** a model that reads conversations while
  awake and installs their facts in its weights while asleep, from its own
  dreams, with retention; measured on LAMA-CKL as to-learn against
  not-to-forget, compared with lora and mix-review at epoch 1 on the same
  500 documents.
- **What the dreamer is for:** producing, after the conversation boundary
  and with no runtime instruction, the text whose one empty-state
  distillation step moves the weights toward what the conversation's
  continuation needs (its answers, its later turns) and away from nothing
  the model already knows or does. That is the whole objective of the
  dreamer track. Its proxies, in lexicographic order: (1) the loop survives
  the epoch (census: EOC stops, `[USER]` present, no duplicate run; "no
  literal above 3 of 10 dreams" is a reading, the kill rule stays 10-13
  §4.8); (2) the acquisition census reads documents that score after their
  wake and keep scoring (acquired: descriptive accuracy 1.0, the scorer's "known" rule, at the
  first evaluation at or after its wake cycle and 0 at every earlier one;
  kept: also 1.0 at the run's last evaluation; read against the same split's frozen ×3 flicker rows, which are subtracted; one definition, in §1.1, here and §6.2b); (3) to-learn at epoch 1 against the baselines, at
  the stated retention.
- **Non-goals:** tuning any dreamer or cell knob on the LAMA-CKL to-learn
  or not-to-forget sets (the dreamer trains on a disjoint corpus, §3.3, and
  the retained pool the hinge protects is not the scorer's retention set,
  §4.24); making the arm a better mix-review by feeding it real documents
  (10-13 §4.4 stands); a runtime dream prompt (10-13 §4.7 stands: one EOC);
  tuning the sampling temperature (sample at 1.0, §3.2); tuning any
  threshold on the dreamer's own outcome (§3.2 has none; every cut is a
  reading of the round's own distribution; the constants the note does
  carry are listed with their status in §4.29); paying the dreamer for the
  shape of a reply (the contrast form of g_new, §3.2 row 3, exists so that
  only content is paid, and abstention is protected by the hinge, never
  paid for, §4.30).
- **Named risk, accepted:** the dreamer corpus's question style leaks into
  the dreams (format leakage). SEAL accepted the same with SQuAD train and
  eval splits. Any gain is read as "on this question format" until a
  second format says otherwise. The conversation rows of §3.3 are the
  second format, in the same corpus. A second named gap: the reward's
  target is a `[USER] q [ASSISTANT] a` reply and the scorer's readout is a
  marker-free cloze (`data.py:77` renders it, `runner.py:416-420` scores
  it); a
  dreamer that improves one and not the other is possible, and the cell's
  curve against its own cycle census is where that would show.

### 3.2 The dreamer track: two dreamers from `ultra20k-v4`, compared against it

This is the new warm start. It is a training stage for the skill of
dreaming usefully. The facts it installs are the dreamer corpus's own (its
SFT trains on dreams about them); what keeps the LAMA measurement clean is
the corpus's disjointness from LAMA (§3.3), not the absence of learning. It
replaces the supervised revisit recipes of 10-13 §3.1 as the next warm-start
step; the `ultra20k-v4` checkpoint is its starting policy.

Three arms, one cell each under one protocol (§3.4): **(a) ReST-EM
dreamer**, selection among the model's own dreams by the reward below;
**(b) dense dreamer**, the gradient-matching loss through the dreamer's
logits, gated by §3.5 item 2b; **(c) `ultra20k-v4` itself**, the untrained
dreamer, **rerun under §3.4** with kill-on-collapse (the fifth box's v4
cell ran 5 documents, 30 dreams, AdamW, per-turn weights and no clip, so it
cannot stand in as the control, §4.27). (a) is built first and is the
sign-off block below; (b) is specified in the research note §6 and is
built only if its gate passes and (a)'s round-0 random-batch specificity
reading (step 2) is positive on most rows, a debrief judgment. Chained (multi
step) training of either dreamer is deferred with its full design in §4.13.

**Prior art.** [`RESEARCH-20261024-learned-dreamers`](../research/RESEARCH-20261024-learned-dreamers.md)
holds the literature and the derivations this section rests on: SEAL (read
this session, numbers verified against the paper), ReST-EM, the
first-order reward and its second-order correction, A-GEM, gradient
matching and its per-token reading, truncated backpropagation through a
chain of steps, and the self-consuming model-collapse results behind §2's
stability paragraph. The decisive facts for the design: SEAL's trained
dreamer gained 7 points over its untrained self (39.7 → 47.0 no-context
SQuAD accuracy) and matched a frontier model as data writer; its unprompted
control, our EOC dream's nearest relative, was the worst format they tried
(13.8 → 18.9 after RL); it used ReST-EM because PPO and GRPO were unstable;
its only protection against drift was structural (SFT on the model's own
fluent samples, two small rounds); it never measured generation quality or
general capability after training; and it never trained the dreamer in the
stepped states deployment puts it in, which is where its sequential-edit
decline appeared.

**Mechanism spec block (arm a) — signed off 2026-10-26 by the teammate
after the tenth adversarial read**, on the record of the 10-25 and 10-26
conversation in which every decision below was made; this block as
written is what any implementation brief quotes (debrief step 3).

Notation: θ the trainable set (LoRA + `marker_delta`, fp32 masters as
10-14 §3.1), flattened as one vector for every inner product below; d its
length. **Every forward pass in this block runs under the fp32 forward of
§3.4** (the same path as the gate and the cells, so the reward describes
the step the cell takes; the bf16-cast path of `master_weights.py:47-50`
is not used by the dreamer). The round checkpoint
`dreamer-a-r<NN>/final/trainable.pt` holds the fp32 masters, as the
runner's `trainable.pt` does (`runner.py:449-463`), and the pinned sha
covers that file. A training **row** is a conversation prefix W (the wake) and a
continuation C (the turns after the boundary), from the dreamer corpus
(§3.3), with a wrong continuation C⁻ (§3.3). "Target tokens" of a row are
the tokens of `a` for a fact row and the tokens of the assistant reply
after the boundary for a conversation row (the user follow-up is context,
not loss: with the same follow-up on both sides of the contrast its
gradient cancels, so it cannot be paid for, §4.41), in both cases without
`<eos>`, `<eoc>` and the role markers. `format_conversation` puts the
marker and EOS inside the trainable turn (`conversations.py:113-121`), so
the dreamer builds **its own target mask** over the rendered ids; the
renderer is reused, its mask is not. The reference batch R is
fixed per round (row 4). n is the cell's dreams per cycle (10, §3.4); K·n
is the pool per row (K = 4: the smallest pool in which a best-of-n batch
is a real choice, four candidates per slot; a reading, raised if the kept
batch is the pool's top-n by sᵢ on most rows); α the cell's SGD learning rate (§3.4); c the
cell's gradient clip norm (§3.4). Every gradient below is of a **per-token
mean** cross-entropy or KL, so all inner products are in nats and the
cell's step θ ← θ − αG' changes a loss by about −α⟨G', g⟩ nats to first
order. "Scored positions" of a dream are the positions `scored_keep`
returns and `turn_weights` weights (`dream_distillation.py:34,118`): the
dream's own tokens after the prompt, cue positions excluded, `[USER]`
turns included. The wake **frame** is drawn per wake from the frame set
(§3.4), never one literal.

| forward pass | initial state | weights | grad | recorded |
|---|---|---|---|---|
| 1. wake: W rendered as the cell renders it — for a fact row: a drawn frame + the passage as the one user turn, reply sampled at 1.0, EOC; for a conversation row: the prefix turns teacher-forced as written, **no frame, no sampled reply** (the prefix ends on an assistant turn), then EOC. Fact passages are cut at `EVIDENCE_TOKENS` (512) as the cell cuts them; the fact wake is built by the protocol's own token builder (`run_conversational_wake`, `protocol.py:202`, frame and document joins tokenized as the cell tokenizes them, imported by the dreamer module, §6.6); the conversation prefix is rendered by the warm start's `format_conversation` (`preparation/conversations.py:58`), the renderer the markers were trained under; the wake's closing EOC **is** the dream prompt of row 2, written once | empty | θ | off | transcript, S_W |
| 2. dream pool: K·n dreams from the EOC prompt in one launch, sampled at 1.0, `DREAM_TOKENS` 512 | a copy of S_W per dream | θ | off | dream tokens, teacher logits per generated position |
| 3. continuation gradient g_new, **contrast form**: C rendered alone through the warm start's `format_conversation` (`preparation/conversations.py:58`, the renderer the markers were trained under; not the scorer's cloze), per-token mean cross-entropy over the row's target tokens, **`<eos>` and `<eoc>` excluded** (the stop tokens would weigh 2/(|a|+2) and leave a "stop here" component when C and C⁻ differ in length; stopping is protected on the g_old side, never paid for here); the same for C⁻, rendered after **this row's** context (fact row: the same question, another fact row's answer; conversation row: the same user follow-up as context, another conversation row's assistant reply as the target; the wrong text is scored under this row's prefix, so lowering it here touches nothing in its own context, §4.35); **g_new = ∇L(C) − ∇L(C⁻)**; both row kinds. The contrast is the **direction**; every measured drop, floor and curvature in the gate is on the true-reply loss **L(C)** (§4.47) | empty, both | θ | on, two backwards per row | g_new (∇L(C) alone is a gate quantity on the gate rows, §3.5 item 2, and is not kept for the P rows) |
| 4. reference gradients g_old^k, **one per source k** (decided 10-25: one pooled direction lets a step that damages kept dreams pass when it helps the facts by more): per row, the per-token mean cross-entropy over that row's loss tokens; g_old^k is the gradient of the mean over the source's rows, so a four-token fact weighs as much as a 500-token dream. R = 100 rows drawn fresh each round, 25 from each of four sources (§3.3), 200 draws over two rounds from a corpus of 175 g_old rows (§3.3; round 1's kept-dream source is not in the corpus): **retained-pool facts** (`[USER] q [ASSISTANT] a<eos><eoc>`, loss on the answer tokens the checkpoint predicts correctly in the (vii) pass and both stop tokens, §4.51, from empty); **abstention rows** (`[USER] q [ASSISTANT] <abstention><eos><eoc>`, the unanswerable question **without its passage**, loss on the reply and both stop tokens, from empty; the behaviour the dreams must not erode); **warm-start rows** (Dolly plain rows and trained-on UltraChat rows in the warm start's own proportion, about 1:4 by conversation count (≈ 4,500 Dolly rows against 20,000 UltraChat), one conversation each, unpacked, with the `<eoc>` the packer appends (`conversations.py:258-263`; `format_conversation` alone does not emit it, so every renderer here appends it explicitly), **loss on user and assistant turns** as the warm start trained them, from empty; R and the replay draw are disjoint within a round); **the previous round's kept dreams** (25 drawn at random from the kept batches of rows with a positive SFT weight; zero-weight rows' batches are excluded, since row 6 did not train toward them; **in the two registered rounds this source guards nothing**: round 0 has no previous round, and in round 1 θ was just fit to those strings, so the source has no slope, and with no gradient through the prefix it cannot see a step that damages how the wake is read; it is specified here for a third round and for the kept-dream likelihood reading, not as a guard the cells rely on, research note §5), each teacher-forced after its **recorded** wake transcript under the current θ with no gradient through the prefix, **cross-entropy** on the dream's scored positions (no teacher logits are kept across rounds). Round 0: the first three sources, 34/33/33 | empty / empty / empty / the dream's wake state | θ | on, four backwards once per round | g_old^facts, g_old^abst, g_old^warm, g_old^dreams |
| 5. per-dream student gradient ∇ᵢ: dream i teacher-forced from empty, the cycle's KL loss (`KL_TEMPERATURE` 1.0) with the recorded teacher logits as soft targets, **uniform weight 1/N over the dream's N scored positions**, the set `turn_weights` already weights (content-free turns and the position that predicts the terminal EOC excluded, `dream_distillation.py:118-158`; the flag changes the weights, never the set); a dream with N = 0 is not scored and not counted in n's divisor, as the cell does (`:219`) | **empty** | θ | on, one backward per dream, no `create_graph` | ∇ᵢ, and the pool's Gram matrix ⟨∇ᵢ, ∇ⱼ⟩ |
| 4b. readings with gradient off: the held-out loss (200 warm-start rows) and the abstention rate (50 held-out **v4-best-known** fact rows, §4.51, no passage, greedy decoding to `<eos>`, detector = the mined stems; the rows are facts the checkpoint answers, so an abstention on them is over-abstention and a rising rate is the damage reading; on unknown rows an abstention would be correct and the reading would have the wrong sign) are forward passes of the round checkpoint. The gate's sweep, floor and step rows take steps and are §3.5 item 2's, not this row's; the retained-pool pass is gradient-off and is §3.5 item 2 (vii) | empty | θ | off | readings |
| 6. policy update: SFT of θ on (wake transcript → each dream of the kept batch) pairs, cross-entropy on the dream tokens only (the frame is in the input, never in the loss; the frame-string count among kept dreams is a round reading, §4.31), mixed 1:1 with replay rows (Dolly plain and trained-on UltraChat rows, one conversation each, unpacked, loss on user and assistant turns); trained set = LoRA + `marker_delta` with fp32 masters, **continuing from round r−1's checkpoint** (not re-fine-tuned from v4 each round as ReST-EM's paper does: the warm start's conversational behaviour is the thing replay protects, and a fresh start would discard round r−1's dreaming); input = wake transcript through the EOC prompt token, targets = the dream tokens with their `<eos>` and `<eoc>`; max length 2,560 (prefix ≤ 2,048 + dream ≤ 512); **loss per pair = per-token mean over that pair's target tokens, batch loss = the SFT-weighted mean over the pairs** (one pair, one experience, as §3.4 weights dreams; a 500-token dream does not outweigh a 50-token one), replay rows weight 1, zero-weight pairs dropped before the 1:1 mix is formed; **gradient through the whole prefix in one unchunked backward** (decided 10-26, §4.48: the warm start's loop detaches the state at every 512-token chunk boundary, `training/loop.py:627`, and with the loss on the dream only a chunk that holds prefix alone gets no gradient, so under chunking most of a 2,048-token prefix is never trained on; the full backward over ≤ 2,560 tokens is what lets the SFT teach reading W; its memory at batch 8 is unmeasured, the gate's local run measures it at 128 tokens first, and the batch drops, with gradient accumulation to the same 8 pairs, if 2,560 tokens do not fit); AdamW 1e-4, weight decay 0.01 (the runner's, `runner.py:808`), through `MasterAdamW`, no warmup, constant lr, optimizer state fresh per round, LoRA dropout 0, 2 epochs, batch 8 (dreamer pins, §4.29; not the warm start's settings, which were 1 epoch, batch 6); a short loop in the dreamer module, since `training/loop.py` has neither the master-weight wrapper's interface (`loop.py:718,951` reads `param_groups`) nor per-example weights | empty | θ | on | round checkpoint |

Per-dream scalars, from rows 3–5 (one backward per dream, then dot
products): sᵢ = α⟨∇ᵢ, g_new⟩ (nats: the first-order drop in the
continuation's loss if the cell stepped on dream i alone, unclipped),
oᵢ^k = α⟨∇ᵢ, g_old^k⟩ (the same for each reference source), ‖∇ᵢ‖, and the Gram matrix
for ‖G_B‖. Cosines are logged beside them, never scored.

Batch score, for a batch B of exactly n dreams with **G_B = (1/n)
Σ_{i∈B} ∇ᵢ** (the cell's loss is the mean over scored dreams,
`dream_distillation.py:219`, so this is the gradient the cell computes)
and the clipped gradient G'_B = G_B · min(1, c / ‖G_B‖) (the step the cell
takes):

```
score(B) = α⟨G'_B, g_new⟩  −  ½ α κ ‖G'_B‖²  −  Σ_k max(0, −α⟨G'_B, g_old^k⟩)
```

Term by term, all in nats. The first is the first-order predicted drop in
the continuation loss after the cell's SGD step on B; magnitude counts,
because the cell's step scales with it. The second is the second-order
correction (research note §4): a step that is too large stops paying off.
κ = α·h, with h the scalar curvature of the **true-reply loss L(C)**
along the step, fitted in the gate (§3.5 item 2 (v)) over the lr sweep's
steps, which vary the rate. The score is therefore a first-order drop in
the contrast minus a second-order penalty from L(C): not the Taylor
expansion of either loss alone, and stated as such (the contrast's own
curvature h(C) − h(C⁻) can be non-positive by construction, §4.47; the
L(C) penalty is the larger of the two when h(C⁻) > 0, so the score
favours small steps a little more than the contrast would). **h rule**
(10-26, analyst's call): h is reused as h, and κ = α·h with the
checkpoint's own α; the session stops only when the sweep found an edge
and the fit on the rungs below it still gives h ≤ 0 (the loss bent the
wrong way); when the sweep found no edge (every rung linear) h is the
sign of noise, so the fit is used if positive and otherwise κ = 0 for
that checkpoint, logged, and the session continues; a later checkpoint
with a failed fit reuses the last good h. h is fitted on fact-row revisit
batches and applied to every batch and both row kinds; isotropic
curvature is an assumption, and the ratio reading below, taken per kind,
is its check: a conversation-row ratio outside the fact-row ratio's range
across rows is the reading that would move h to per kind, a debrief
call, no stop. The teammate keeps the quadratic
term (10-25): without it nothing in the score sees step size. Its reading: the ratio of
the quadratic term to the learn term on the **random** batches of the
specificity reading (kept batches are selected to pass, so they cannot
read it), logged per round. The third is four hinges, one per reference source; each fires only when the batch's step would raise that source's loss (A-GEM's half-space as a selection criterion, held per source), in nats per reference row against nats per new row, so each weight is 1 and not a knob (the adversarial read's objection that a mean
over 100 old rows is "100× weaker" than one new row is recorded in §4.32;
the teammate keeps 1 unless the cells show a reason). A row whose
reference loss has no slope contributes nothing to g_old, and the second
term does not protect it either: h is the curvature of L(C) on the gate rows,
not of old rows. Flat old facts have no first-order guard; that is A-GEM's
known limit, and the cell's retention curve is what measures the damage.
Nothing here is normalised by
‖g_new‖ or ‖g_old‖: the §3.2 draft of 10-24 divided by them, which made
the three terms three different units and hid a weight of
‖g_new‖/‖g_old‖ in the hinge.

Batch assembly (greedy, arithmetic on the scalars): start empty; add the
pool dream that raises score(B) most, where score is evaluated on the
partial batch with G as its mean; continue to exactly n dreams (the cell
distils n every cycle, so a smaller batch is not a step the cell takes);
then one pass of swaps (each chosen dream against each unchosen, keep any
swap that raises the score). The first term is additive
across dreams; the quadratic term (Gram cross-terms), the clip and the
hinge are not, so greedy plus one swap pass is a heuristic, not an exact
maximiser, and the swap pass is what lets a rehearsal dream with positive
oᵢ release the hinge.
The best batch per row is kept, always: ReST-EM with no keep threshold.
Its **SFT weight** (row 6; decided 10-26, §4.45) has two parts. The sign
is absolute: a row weighs zero unless score(B*) > 0, the full score with
its hinges (a row on which no batch is predicted to help has as its best
batch the one with the smallest gradient, the most document-free dreams,
and training toward it would teach the opposite of §3.1; a batch that
also damages a reference source is not trained toward). The magnitude is
relative: the row's **random-batch specificity gap** of step 2, the mean
over 20 unselected n-batches of the pool of (score against this row's
g_new minus the mean of score against the g_new of every other row of
the same kind in the round, the C⁻ partner rows excluded; only the first
term depends on the row, so each gap is α⟨G'_B, g_new^i − mean_j
g_new^j⟩, a dot product per batch), with a gap at or below zero weighing
zero; the positive weights are then divided by their mean **over the
positive rows** of the same kind (fact and conversation rows have
different ‖g_new‖ scales and are normalised apart). The gap is taken on
random batches and not on B* because B* is selected on its own-row term,
so its gap carries that selection upward under any reward, noise
included (tenth read); the random-batch gap is unbiased and reads a dead
reward as zero. The weight thus says "this row's pool carries its
document"; B* decides the sign and supplies the strings. The gap is zero for
a batch that helps every row alike (format, register, "reply briefly")
and positive for one that carries this document, so the weight pays for
what is specific to the row and not for what is generic. The reason the
magnitude is neither the score nor B*'s own gap: B* is the best of many
batches, so under a dead or format-only reward its score and its gap are
positive on most rows and a weight sized on either would train on noise
at full strength (ninth and tenth reads). Two readings per round: the share of rows at zero weight, split
by cause (score ≤ 0, gap ≤ 0), and the gap's distribution; a round 0
with almost no positive rows is the case the §4.8 bridge exists for,
decided at the debrief. If no row of a kind has a positive weight, that
kind's weights are all zero and nothing is divided. A pool dream with
N = 0 scored positions is never picked, so G_B is always a mean of n
scored dreams; a row whose pool has fewer than n such dreams keeps no
batch, weighs zero and is logged.

Event sequence, one round (P = 50 rows, fresh each round, 35 fact rows
and 15 conversation rows, §3.3; the 20 **prod rows** are a fixed separate
set, never in P, never trained on):
1. Measure α, c, κ and the retained pool on the round's starting
   checkpoint with §3.5 item 2's script, **on the box at 512-token
   dreams** (the box gate run at the start of the dreamer session is the verdict on the reward and round 0's instrument; the local 128-token run is a smoke of the code and the hand-written ranking by sᵢ, never a verdict); the fact rows of the round
      are drawn from this pass's unknown side; the same instrument steps run once more on the last round's final checkpoint, so every checkpoint a cell starts from has its own α, c and κ. Draw R and compute the four g_old^k (row 4). The prod reading's R is
   round 0's, kept as rows; its gradients and the prod rows' g_new are
   recomputed under the current θ (the reading measures the model under a
   fixed instrument and fixed data, not a stored vector).
2. For each row: wake (row 1); pool of K·n dreams (row 2); g_new (row 3);
   ∇ᵢ and the Gram matrix (row 5); the scalars; the assembled batch B*.
      Then the **specificity reading**: for 20 random n-batches of the
   pool (unselected, so no winner's bias), the mean of score against this
   row's g_new minus the mean of score against the g_new of every other
   row of the same kind in the round (never the row that supplied this
   row's C⁻, and never a row whose C⁻ came from this row, since that
   row's g_new carries −∇L(C_i); the mean over rows rather than one
   random row, since every g_new is in memory and only one dot product
   depends on the row). Dreams that carry this document help their own
   row and not the others, so the gap is positive; dreams that carry only
   format help all alike, so it is near zero; it can go negative.    This
   random-batch gap is the row's SFT weight magnitude (above), the arm
   (b) gate and the dead-reward reading; the same gap for B* is logged
   beside it as the selection view, biased upward by selection, and
   decides nothing. All 50 g_new vectors are computed before any pool
   (4.25 GB in fp32), so the other rows' g_new exist when row i's
   gradients are in memory. The reading changes nothing about what is
   kept.
3. **Live reading, round 0 only, logged, no stop** (teammate, 10-25: the
   box gate at the session's start is the liveness verdict on hand-written
   dreams, and the session then runs to its end): over the first ten fact rows, the pool dreams
   that revisit the row's passage by the revisit rule (preamble) against
   those that do not, on sᵢ and on cos(∇ᵢ, g_new) (the second is
   norm-free, since revisits carry larger gradients, §1.3). Ranks taken
   within each row, pooled, one-sided Mann–Whitney U; the statistic, its
   p and the revisiting group's size are logged and read at the debrief
   (the fifth box's cycle 0 had one revisit in thirty, so a group under
   five dreams is itself the reading). This is a reading of sampled
   dreams and is separate from the hand-written ranking of §3.5 item 2.
4. Policy update (row 6) on the P kept batches with the replay rows.
5. Round readings (all logged, none a stop): the specificity reading per
   row and its sign count over the round; per row type, the mean kept score **under round 0's α, c, κ** (the fact-row mean is the one that must rise; the conversation rows' mean rising or not decides their share in later rounds); the 100 per-row predicted old-loss changes ⟨G'_{B*}, g_old,row⟩ for the kept batches (100 single-row backwards once per round, about 8 min at 5 s each, in the budget below; the reading that would move the hinge to per row, §4.36); the
   **prod reading**: on the 20 prod rows, dreamed from the round's starting checkpoint and, after the last round, from its final checkpoint too, the mean over the 20 rows of the score of one random n-batch per row, **computed with round 0's α, c, κ and R** so that re-measurement
   and the redraw (step 1) cannot move it, with the mean cosine
   cos(G_B, g_new) of the same batches beside it (the magnitude-free form,
   since the round SFT changes ‖∇ᵢ‖ and ‖g_new‖) (what a cycle's unselected step would score). The decision reading is
   the **cosine**, the norm-free form, since the round SFT moves ‖∇ᵢ‖ and
   ‖g_new‖; the score is read beside it; both carry the standard error
   over the 20 rows, and "rises" means the round-to-round change is
   outside that spread, a debrief judgment, not a rule. The kept-score
   mean is a best-of-many statistic that rises with variance alone
   (§4.40), so it is read against the random-batch mean of the same
   round, never alone; hinge fire rate; clip fire rate; mean ‖∇ᵢ‖ of kept
   dreams against the pool mean (the zero-gradient optimum shows here);
   kept dreams' mean and max length (SEAL's RL lengthened its self-edits);
   the frame-string and acknowledgement counts among kept dreams; held-out
   loss on 200 warm-start rows (disjoint from the replay rows) under the
   round checkpoint (the drift reading for the replay guard); over-abstention
   rate of the round checkpoint on the 50 held-out v4-best-known fact rows with
   no passage (row 4b; detector: the mined abstention stems, §3.3); the dream
   census (§6.1) over all pool dreams; five decoded kept dreams and five
   rejected per round. Rounds: 2 as SEAL, a third only if the prod reading still rises (the
   debrief's call). Whether round 0's readings warrant the bridge (§4.8)
   is decided at the debrief. Rows are independent, so a round is
   resumable per row: each row's kept batch and readings are written when
   the row completes, and a restarted round skips completed rows.

Budget, per round on the GH200, from the fourth box's measured 40-dream
launch (58 s generation, 26 s distillation,
`EXPERIMENT_NOTES-20261013-192947.md:100`): 70 rows per round (50 P +
20 prod) × 40 dreams = 2,800 dreams in 70 launches ≈ 68 min generation;
2,800 single-dream backwards at 0.17–2.1 s (the fifth box's empty-state path distilled 30 dreams in about 5 s, `sft/logs/lama-ckl-run-20261021-163654-1795999.log` cycle 1, 16:39:45 → 16:39:50, about 0.17 s per dream batched; the 0.65–2.1 s are the fourth box's figures under the voided teacher-state path; a single-dream backward with a flat
gradient copy and the Gram update is **not measured** and is the first
number the session logs) ≈ 8–98 min; 140 g_new backwards (35 fact and 15 conversation rows × 2, 20 prod rows × 2), four g_old
of 100 rows in four source batches, the 100 per-row g_old backwards for the hinge log, and in round 1 the recompute of round 0's R (four more; each a 2.7B backward over at most 2,560 tokens, unmeasured, allow 5 s) ≈ 25 min; the step 1 re-measurement (noise floor, five 7-rate sweeps, the retained-pool pass over 2,000 candidates) ≈ 30 min; specificity scoring
is dot products; SFT on 500 pairs + 500 replay rows, 2 epochs ≈ 13 min
(2,000 sequence passes at the warm start's rate, 9,594 steps of batch 6
in 69 min, the fourth-box run note `EXPERIMENT_NOTES-20261013-192947.md` lines 26 and 230: 27.4M tokens at batch 6 and chunk 512, about 6.6k tokens/s, applied here to 2,560-token sequences as an estimate; 2,000 passes ≈ 5.1M tokens ≈ 13 min); live reading,
census and held-out readings ≈ 20 min. **About 2.7–4.2 h per round**, plus the final-checkpoint prod pass after the last round (20 launches, 800 dreams, 800 single-dream backwards: 28–47 min at the same rates) and its instrument steps; **two rounds in one session of 12 h or less**, all at bf16 rates with the fp32 forward's cost unmeasured (§3.4). Memory per row: 40 flat
gradients of the 85 MB trainable set, 3.4 GB in fp32, the 50 g_new
vectors (4.25 GB), plus the pool's teacher logits
(`ephemeral_teacher_logit_bytes` 3.09 GB for 30 dreams at 511 tokens, so
≈ 4 GB for 40); the gradients are dropped after the scalars, the Gram
matrix and the specificity dot products are taken.

Known dangers, each with its reading: the zero-gradient optimum (step 4
norm reading); length growth (step 4 length reading; the lexicographic
length rule of §4.14 returns if kept length rises across rounds); the
dreamer's own stepped states are never seen in training (the cell's
per-cycle trigram revisit rate is the reading, §3.4; §4.13 is the
response); g_old from a sample is a noisy direction, so the hinge is a
filter against blatant interference, not a forgetting guarantee (the cell
measures the real retention); a short dream of any kind carries the same
total weight as a long revisit under the uniform per-dream weighting; in
the dreamer, selection keeps acknowledgements out of the kept set (the
kept-set count is the reading); in the cell, which distils unselected
samples, nothing does, and the cell census's per-cycle acknowledgement
count is the reading there (§4.25);
format leakage (§3.1 named risk).

Dreams are independent draws, so the policy can learn a mixture (mostly
new-document dreams, some rehearsal, in learned proportions), not a
composed batch (§4.5). Batch-level scoring is what lets a rehearsal dream
earn its place beside an interfering one.

### 3.3 Dreamer corpus (decided 2026-10-25)

Two data roles, and no source serves both: **g_new rows**, one renderer (`format_conversation`), P = 50 per round:

| rows | source | wake W | target C (loss tokens) | wrong C⁻ | share |
|---|---|---|---|---|---|
| fact | SQuAD 2.0 train (`rajpurkar/squad_v2`, HF cache under `.cache/huggingface`), answerable questions the checkpoint does **not** answer without the passage (the complement of the retained-pool test, same pass), passages with ≥ 2 questions | a drawn frame + the passage as the one user turn | `[USER] q [ASSISTANT] a<eos><eoc>` for one question of the passage, drawn per row; loss on `a` only (row 3) | the same q, a random other fact row's `a` | 35 |
| conversation | UltraChat `train_sft` examples at index ≥ 20,000 (the warm start loads the first 20,000, `sft/Makefile:324`), split at a turn boundary with ≥ 1 assistant turn on each side; among the valid boundaries one is drawn per row (seeded, recorded) | the turns before the boundary, no frame, capped at 2,048 tokens (rows whose prefix is longer are dropped) | the assistant reply after the boundary, with the user follow-up as context; loss on the reply, stop tokens and markers excluded (§4.41: the follow-up as a target is deferred to the personality rows) | the same follow-up, a random other conversation row's assistant reply | 15 |

Rules on the g_new rows: C⁻ differs from C as a string (a coincidence of
answers gives a near-zero g_new and the row is redrawn), and the row that
supplies C⁻ is never the row used for the specificity reading; **no target
turn is an acknowledgement** by the pinned rule, so a conversation
boundary whose target assistant turn is one is redrawn among the row's
other valid boundaries, else the row is dropped; conversation targets with
dangling references ("swap day 2") are accepted as a lower ceiling, not
filtered (§4.21). The **unanswerable questions** of SQuAD 2.0 (an empty
answer list) are **not g_new rows** (§4.30): they are the abstention rows
of g_old, with the reply drawn from the **abstention set**, mined from the
trained-on UltraChat rows: assistant turns of at most 30 words that open
on a refusal-to-answer stem (`I don't know`, `I'm not sure`, `I don't
have`, `I'm sorry, but I`, `I cannot`, `I can't`), deduplicated, written
to the corpus file and read in the review; a set under 20 phrasings is a
reading before round 0, not a reason to fall back to one string (§4.23).
The same stems are the abstention detector of the round readings.

**g_old and replay rows:** the **retained pool** (facts the checkpoint
already knows: SQuAD 2.0 train answerable questions, disjoint from every
g_new, prod and held-out row, **known** by the one rule of §3.5 item 2 (vii): at least one answer token the checkpoint predicts correctly, teacher-forced with no passage in context, and its g_old loss is the cross-entropy on those currently-correct answer tokens and the two stop tokens (decided 10-26, §4.51: the accuracy-1.0 rule found 4 known passages in 879 on v4); recomputed on every checkpoint a dreamer round starts from, since the round SFT installs facts (a cell does not use g_old); written to
`data/retained_pool-<sha8>.json`); the **abstention rows** (above); the
**Dolly plain rows** (`data/dolly_context.jsonl`, shape `plain`,
regenerated locally by `preparation/dolly.py` with its fixed seed); the
**warm-start UltraChat rows** (indices < 20,000, the warm start's source
set; packing truncated some (`conversations.py:125-126`), split others and dropped some whole (`:262-263`), so "trained on" holds for most rows, exactly for none, and not at all for an uncounted dropped share (the generator logs how many of its draws the packer's seeded pass would have dropped),
which is why the drift reading on them is read as a change across rounds,
never as an absolute). Dolly's document and multi shapes are not used
anywhere: their targets are acknowledgements. Multi-document rows of any source are left out because of recency (§1.3): a five-document state concentrates revisits on the last document, and the first dreamer measurement should not carry that. Conversation rows are also a wake the cell never runs (a frameless teacher-forced prefix) and are kept on purpose as §3.1's second format, the one the long goal is about.

**Fixed sets, all disjoint from each other and from every round's P:** 20
prod rows (fact, v4-unknown), 50 held-out **v4-best-known** fact rows (the 50 highest-accuracy rows of the v4 scan, §4.51; the over-abstention reading, row 4b), 200 held-out warm-start rows (from the
warm-start UltraChat rows, outside the replay draw), 5 gate rows (fact,
v4-unknown). Disjointness is at the **passage** level across every set
and every round: an abstention row never shares a passage with any P row
of any round (else the hinge charges for learning that passage), each
round's P draw and each checkpoint's retained-pool redraw exclude every
earlier round's P passages, and the fixed sets' passages are excluded
from every draw. Corpus size for two rounds (a third is the debrief's call): 100 g_new
rows, 175 g_old rows (round 1 adds the kept-dream source), 275 fixed, plus the replay draw; SQuAD 2.0 train has about 130k questions; the word-bounded LAMA filter (country, city and language names among its objects) will remove many passages and prefixes, and the generator's log reports the surviving fraction per pool, the number that says whether supply is a constraint.

Requirements for all rows: **no overlap with the LAMA-CKL candidate
pool** (every row `build_candidates` can emit, not the one split, since
the split is recomputed per checkpoint, §3.4), by a case-insensitive, word-bounded match of every LAMA candidate's
subject and object string against the row's passage and question text
(short strings such as `Spain` will exclude many passages; the excluded
count is a reading of the generator, supply is not the constraint). The corpus is a generator with the
"sanity-check the artifact" log (root `AGENTS.md`): overlap count 0, five
decoded rows of each kind with C⁻ beside C, the abstention set, the
retained pool's size and accuracy distribution. Its files
(`data/dreamer_corpus.jsonl`, `data/dreamer_abstentions.json`,
`data/retained_pool-*.json`, `data/dreamer_fixed_sets.json`) reach the box through the `LAMBDA_DATA_ARTIFACTS` line of §6.8 (the
script's defaults are not changed, §6.7c). **The decoded samples are
read together by the analyst and the teammate before the dreamer session
is scheduled**; that read is a debrief step, not a log the experimenter
checks.

### 3.4 Cell protocol after the dreamer: one document per wake

Amends the five-document wake (10-10 first-box debrief §3, 10-13 §3.3) where they
differ. Reasons: §1.3, recency (12 of 14 revisits read the last document);
and the reward of §3.2 is defined per row, so the cell that measures the
dreamer should feed it rows. Production has gaps between conversations;
this is a fine-grained sleep, not wake learning (no gradient during the
conversation). The multi-document regime is deferred, not rejected (§4.9).
**All three arms run this protocol**, the v4 arm with kill-on-collapse.

- `--docs-per-wake 1`, `--dreams-per-cycle 10` (the 30 was to fish one
  revisit out of a five-document state), `--dream-batch-size 10`, dreams
  and reply sampled at **1.0** by a new `--dream-temperature` flag, which sets both the reply and the dream temperature (one constant sets both today, `runner.py:511,527`; default
  0.7 so every v4 command stays reproducible; `probe.py` imports the
  constant and is untouched; `run.json` records both `reply_temperature`
  and `dream_temperature`), **100 cycles per cell** (below), of a 500-cycle epoch.
- **Wake frame from a set** (protected edit, `protocol.py:20`): a new
  `WAKE_FRAMES` list of at least 40 phrasings, with `WAKE_FRAME` kept as
  its first entry so nothing that imports it changes (`test_dolly.py:8`,
  `test_protocol.py:7`, `replay_cycle.py:27`), and the frame counts in `dream_census.py:14`, `replay_cycle.py:115` and `test_dolly.py:75` extended to the whole set; a new runner flag `--wake-frames {single,set}` (default
  `single`, so every v4 command still reproduces; `run.json` records the
  mode and `wake.json` the drawn frame). The set: user-side requests in the register of the Dolly `PREFACES` (`preparation/dolly.py:70`, the user-turn prefaces; `OPENERS` at `:23` are the assistant acknowledgements, the carrier), **disjoint from `PREFACES` by exact string** so the warm start never saw any frame verbatim; one drawn per wake from a generator of its own, seeded from (epoch, cycle, a fixed frame salt) and touched by nothing else (the wake seeds one generator per turn, `protocol.py:225-226,293-300`, and a draw from any of those would move the reply sample), and passed to `run_conversational_wake`, which takes the frame as a parameter (today it reads the constant, `protocol.py:268`); under `single` no draw is made, so every v4 command stays bit-reproducible (`protocol.py:293-295`); recorded in `wake.json`, with `settings["wake_frame"]` recording the mode and the set's hash under `set`. One literal seen every cycle is the
  tic the `sft/AGENTS.md` rule names; the set keeps the request and
  removes the literal. The dreamer rows (§3.2 row 1) draw from the same
  set. The set stays out of the warm-start corpus by that exact-string rule; paraphrase overlap is accepted (the request itself must be learnable). Every entry ends in the same trailing space convention as `WAKE_FRAME` (`protocol.py:268` appends the document after it), and the census and test matchers (`dream_census.py:15`) match the stripped strings.
- **fp32 forward** for the dream cells (decided 10-25; new flag
  `--fp32-forward`, the trainable set and the backbone computed in fp32,
    about 11 GB, the whole 2.72B-parameter model in fp32, which is "more"
  only if the bf16 copy stays resident beside it, a choice §6.4's helper
  makes, and the memory sums below assume the worse case): the cell's forward otherwise reads a bf16 cast of
  the fp32 masters (`master_weights.py:19,47-50`), with a bf16 spacing of
  a few 1e-5 per coordinate at the adapter's magnitudes (the gate script
  reports the median; the tenth read's local pass over
  `ultra20k-v4/final/trainable.pt` found a median spacing of about 3e-5),
  so a one-off update under half a spacing never reaches the
  forward while a consistent one accumulates in the master until it
  crosses: a filter for exactly the carrier class under SGD at a small α,
      and a step the reward does not describe. The baselines keep the bf16
  forward with masters; the report states the asymmetry. **One owner:**
  the fp32 forward is a mode of the master-weight helper (§6.4; the
  wrapper's parameters and the backbone held in fp32, no bf16 cast in
  the loop), and the gate, the dreamer and the runner's `--fp32-forward`
  all call it, so "the same path" holds by construction and not by three
  implementations agreeing. **Loading:** the runner's cell start
  (`runner.py:800-803`, `load_checkpoint`) and the dreamer's
  round-to-round load cast the stored fp32 masters into bf16 parameters,
  after which the wrapper rebuilds its masters from bf16
  (`master_weights.py:18-19`), the same hazard as resume (below); under
  the fp32 forward every load of a `trainable.pt` goes straight into
  fp32, so the pinned sha covers the weights the cell runs.
- **Cell ledger** (the states the 7b brief names, so the 10-13 failure
  mode has no gap to enter; the current code already does this,
  `dream_distillation.py:203` passes `state=None` for the student):

  | forward pass | initial state | weights | grad |
  |---|---|---|---|
  | wake: frame + document, reply sampled, EOC | empty | θ_c | off |
  | dreams: n from the EOC prompt | a copy of the wake state S_c per dream | θ_c | off (teacher logits recorded) |
  | student: each dream teacher-forced, KL to the recorded logits, uniform 1/N | **empty** | θ_c | on, one step |
- **Starting point:** the dreamer checkpoint's LoRA and marker delta are
  loaded as the cell's trainable set and trained on (no merge, no fresh
  adapter), as `ultra20k-v4/final` is today. Its sha is pinned in
  `warmstart.sha256` and **its own LAMA split is generated first**
  (`split.py` selects by what the pinned adapter knows; the runner refuses
  a sha mismatch), one protected script run per scored checkpoint; the v4
  arm keeps the banked v4 split. Arms are therefore on different splits by
  construction, which the report states beside the comparison.
- **Loss weighting: uniform per token within a dream**, behind a flag
  (`--dream-loss-weights {turn,uniform}`, default `turn`, so every
  fifth-box command still reproduces; `uniform` makes `turn_weights`
  return 1/N for each of the dream's N scored positions; the batch loss
  stays the mean over scored dreams). Each dream is one experience of
  equal weight whatever its length, and inside it every token counts the
  same. Rejected forms in §4.15; the cost, §4.25.
- **Optimizer: plain SGD** for the dream arm (`torch.optim.SGD`, momentum
  0, weight decay 0, fp32 master weights as today through a generalised
  wrapper, §6.4), replacing AdamW; new flags `--optimizer {adamw,sgd}` (default adamw), `--lr` and `--clip-norm` (no defaults under `sgd`: the runner refuses `--optimizer sgd` without both; `--clip-norm` is refused under `adamw`), all in `run.json` and in the `settings` comparison that guards resume.
  `--lr` applies to the dream arm only; the baselines' `LEARNING_RATE`
  (`runner.py:60`) is untouched. **Learning rate α:** the edge of the
  linear regime, measured by the gate (§3.5 item 2): over a sweep of lr
  on one 10-dream revisit step, the largest lr at which the measured drop
  in the row's true-reply loss L(C) is at least half the first-order
  prediction lr·⟨G, ∇L(C)⟩ (decided 10-26, §4.47: L(C) is bounded below,
  so it is the loss that "stops paying off"; the contrast is the
  direction, not the ruler; the half is a constant, listed in §4.29); the
  cell's α is the smallest of the five gate rows' values. (The 10-24 anchor "first step norm in 0.3–0.5" is withdrawn: Adam's
  first step is lr·√d = 0.4618 for d = 21,323,264, a parameter count,
  §4.26.) Measured locally at 128 tokens for the gate's verdict; **re-measured on
  the box at the cell's 512-token dreams on each checkpoint before its
  cell**, and those are the values the cell runs with; never on LAMA. **Gradient clip c** (new flag
  `--clip-norm`, a **gradient** norm: the clip acts on G before the lr, so
  the step norm is α·min(‖G‖, c)): the largest ‖G‖ over the five gate rows' 10-dream revisit batches (a
  gradient norm does not depend on lr; G is a mean over the n dreams, so
  n does not scale it), so the clip fires only on a gradient larger than
  any the gate validated; applied between
  `backward` and `opt.step()` in `distill_dream_batch`; logged per cycle,
  fire rate in the census (a reading).
  The baselines keep AdamW 1e-4: lora and
  mix-review are the standard method with its standard optimizer, and the
  report states the asymmetry beside the comparison.
- **Score every fifth cycle** (new runner flag, protected edit:
  `--eval-every N`, default 1; score when `(cycle + 1) % N == 0` and
  always on the epoch's last cycle, so the runner's epoch-end fallback
  evaluation never adds a 101st; the adapter is still saved every cycle).
  20 evaluation points over 100 cycles; the acquisition census (§6.2b) uses the cycle a row was shown and the first evaluation at or after it.
- **Per-cycle resume** (protected edit, decided 10-25, crash insurance):
  every epoch's work directory is a fixed path (`work/epoch-NN/`, not a
  `mkdtemp`, `runner.py:336`, `epoch-00` included); each cycle is written to `cycle-NNN.tmp/`
  and renamed to `cycle-NNN/` as its last act, so a half-written cycle is
  a `.tmp` directory that resume deletes; resume counts the complete
  cycle directories and continues at the next, loading
  the last complete cycle's `cycle-NNN/trainable.pt`, which already holds
  the fp32 masters (`runner.py:449-463`), **directly into the wrapper's
  masters**: the hazard is `load_checkpoint`, which casts them into the
  bf16 parameters, after which the wrapper rebuilds its masters from bf16
  (`master_weights.py:18-19`), so resume bypasses that cast; an
  optimizer with moments (AdamW) also writes its state beside every
  cycle's adapter and loads the last complete cycle's; resume is automatic when `--output` exists with matching settings; cycle directories are `cycle-NNN` (`:03d`) where they are created (`_dream_epoch`, `runner.py:218`), where the last is checked (`run_epochs`, `runner.py:357`, else the epoch-end fallback evaluation fires) and in the census; the in-memory epoch record (`runner.py:340-349`: cycle list,
  totals, `wake_sha256`) is rebuilt from the complete cycles' `wake.json`, `dreams.json`, `distil.json` and, where present, `result.json`; a `--cycles-per-run` stop writes `summary.json` with `cycles_run` and leaves the epoch directory incomplete (no epoch rename at `runner.py:378`, no entry in `_completed_epochs`, `:665-670`), so a resume with a larger N continues the same epoch instead of skipping it; the existing per-(epoch, cycle) seeding (`dream_set_seed`, `runner.py:165-167,512,518`) already makes a resumed cycle draw what the original would have; `--max-hours` is checked after every cycle, not only at epoch
  end (`runner.py:395`), and stops with the partial epoch resumable.
- Cost, from the fourth box's measured 10-dream cycle (wake 3 s, 10 dreams
  39 s, distillation 21 s under the voided teacher-state path, 63.1 s
  before an 8 s evaluation; the fifth box's empty-state path distilled 30
  dreams in about 5 s, so the 21 s is an upper bound): ≈ 45–65 s per
  cycle with a score every fifth, before any length effect of sampling at
  1.0 and with the per-dream backward unmeasured (§3.5 item 1). **The
  fp32 forward's cost is unmeasured**: every timing here is bf16, the
  manual Mamba2 path in fp32 may be several times slower, and the first
  cycle of the first cell is the measurement; `--max-hours 4` is the
  backstop so that a slower cell is not cut before cycle 100. **Every
  cell runs 100 cycles** (teammate, 10-25: the cells are expected to
  collapse or not within 50; a cell that survives 100 is the finding, and
  its full epoch is the next session's job by per-cycle resume), ≈ 1.8 h
  each at bf16 rates; the acquisition census and the shown-row comparison against the
  baselines read on the 100 shown documents, and the full-epoch
  comparison waits for a survivor. **One session of 12 h or less for every cell whose checkpoint exists**
  (arm (c) and arm (a) after the first dreamer session; arm (b)'s cell
  joins when its checkpoint exists, in that session or a later one), in
  this order: the v4 arm's cell under the banked v4 pin
  (α and c measured on the box at 512 tokens first, ≈ 25 min), then for
  each dreamer checkpoint: `warmstart.sha256` re-pinned on the box branch
  (`split.py:115-117` and `runner.py:760-763` refuse any other adapter, so no two arms overlap under one pin; the experimenter edits `sft/experiments/lama_ckl/warmstart.sha256` (protected; the box branch allows it) on the box branch, where the hook allows it, and the pin returns to v4 only if the v4 cell is resumed), its split generated (`make lama-ckl-split ARGS="--lama-root <the LAMA root the v4 split was built from> --init-adapter <ckpt>/final --output ../.cache/lama_ckl/mamba2_2_7b_<ckpt name>"`, `--lama-root` being required, `split.py:93-96`, the "scorer artifact" being that directory's split and probe files), its baselines' outputs under `runs/<arm>-<ckpt name>-seed-42` (the frozen target's hard-coded `runs/frozen-seed-$(SEED)-r$$run`, `sft/Makefile:276`, gains a `RUN_PREFIX`), α and c re-measured, its
  cell run with its baselines **one at a time beside it, lora then
  mix-review** (measured bf16 peaks: 27 GB for a 10-dream distillation,
  22.7 GB fp32 lora, 39.5 GB fp32 mix-review; all three together would be
  89 of 94.5 GB). Three cells ≈ 5.5 h plus two acceptance cycles ≈ 0.2 h; per dreamer checkpoint, lora and mix-review at the fifth box's measured 124 s and 175 s per epoch (measured sharing the card with each other, not with a cell) ≈ 2.5 h, frozen ×3 ≈ 2 min for all three (the fourth box's measurement, `EXPERIMENT_NOTES-20261013-192947.md:66`; the fifth box ran none), beside the cells; splits and re-measures ≈ 1 h. A baseline pair (≈ 2.5 h) is longer than the 1.8 h cell it runs beside, so the session's serial length is about 8–9 h at bf16 rates (v4 cell, two checkpoints' pairs, splits and re-measures), not the sum of the cells. Memory: the 27 GB is a five-document bf16 distillation; with the fp32 forward's ≈ 11 GB the cell is ≈ 38 GB, ≈ 77.5 GB beside mix-review, so one baseline at a time still fits; these are estimates from other configurations, not a measurement of this plan. Disk: 100
  adapters × 85 MB ≈ 8.5 GB per cell under SGD (no optimizer state; an AdamW control would double it). `--max-hours` stays on as a backstop
  at 4 (covers a cell up to about 2.2× the bf16 rate; a slower cell stops
  resumable and the session plan is re-cut on the measured first cycle,
  since three 4 h cells would fill the session).
  Full cell command (existing flags `--arm --split --init-adapter --output
  --docs-per-wake --dreams-per-cycle --dream-batch-size --epochs --seed
  --max-hours`, `runner.py:677-720`; **not** `--cycles`, which is the
  acceptance flag that saves cycles unevaluated and refuses resume,
  `runner.py:706-711`; `--cycles-per-run N` is a new flag: stop after cycle N−1 of the epoch (absolute, not "N more"), excluded from the resume `settings` guard like `max_hours` (`runner.py:834`), so a survivor's full epoch is a resume with a larger N, `summary.json` written at the stop with `cycles_run` and a partial curve that `report.py` accepts when `cycles_run` is present (`report.py:99-100` refuses an incomplete curve today); the shown-row comparison is the dream arm's to-learn **restricted to rows 0–99 at its last evaluation** against the baselines' to-learn on the same rows at epoch 1 (same point for both, so a row learned and then forgotten counts 0 on both sides; the acquisition census is the separate per-row health reading, §6.2b); the
  rest are §6.7b's new flags; the experimenter fills the paths):

  ```
  make lama-ckl-run ARGS="--arm dream --split <split dir> --init-adapter <dreamer ckpt>/final \
    --output ../.cache/lama_ckl/runs/dream-<ckpt name>-seed-42 \
    --docs-per-wake 1 --dreams-per-cycle 10 --dream-batch-size 10 --dream-temperature 1.0 \
    --wake-frames set --dream-loss-weights uniform --fp32-forward --optimizer sgd --lr <α> --clip-norm <c> \
        --eval-every 5 --cycles-per-run 100 --epochs 1 --seed 42 --max-hours 4"
  ```

  `run.json` records the dream arm's `--lr` under `dream_lr` and the clip under `clip_norm`; `learning_rate` stays the baselines' constant (`runner.py:774`); `--lr` and `--clip-norm` are refused under `adamw`.
- **One α and one c for all three cells** (decided 10-26, §4.49): every
  checkpoint's instrument steps are run, but the cells run at the
  smallest α and the largest c over the arms' checkpoints, so a survival
  is not a smaller step (§4.39); each checkpoint's own α and the ratio to
  the common one are reported beside the comparison.
- One optimizer step per document (100 steps over 100 cycles against
  lora's 63 per epoch over 500 documents); retention is read after 100
  steps of drift on 100 documents against the baselines' epoch 1 on 500,
  with that stated beside the comparison. §3.1's proxies (1) and (3) are
  read at 100 cycles; "survives the epoch" means the 100-cycle run until
  a survivor's full epoch exists.
- Baselines: frozen ×3, lora, mix-review rerun on each dreamer checkpoint
  under fp32 master weights, same session order as 10-14 §3.2 with the
  dreamer checkpoint in place of `ultra20k-v4` and its sha pinned in
  `warmstart.sha256`. Paths: the dreamer checkpoints are
  `models/mamba2_2_7b/checkpoints/dreamer-a-r<NN>/final`, NN two digits from 00 as the dreamer numbers its rounds (arm b:
  `dreamer-b-r<NN>`); "the dreamer checkpoint" of an arm is its last
  round's; its split and scorer artifact live under
  `.cache/lama_ckl/mamba2_2_7b_dreamer-a-r<round>/`, the v4 ones stay
  where they are.
- Readings per cycle, in the census: trigram revisit count of the wake
  document (the dreamer's habit under its own steps; a decay across cycles
  is the trigger for §4.13), short-turn and acknowledgement counts, clip
  fire, step norm. Adapters are 85 MB each; `LAMBDA_CACHE_ARTIFACTS` pulls the whole `lama_ckl` tree (≈ 8.5 GB per cell), the census runs on the box over all of them.
- Kill on collapse as 10-13 §4.8; the census of §1.1 is the checklist;
  the per-dream loss floor stays a warning.

### 3.5 Order of work and gates

1. **Protected loss edits first** (§6.7a): the `--dream-loss-weights`
   flag in `turn_weights`, and `distill_dream_batch`
   (`dream_distillation.py:161-230`) split into a per-dream gradient
   function (loss, backward, flat gradient out, no step) and the step
   (clip, `opt.step()`), so the dreamer trainer and the gate import the
   cell's gradient and row 5 of §3.2 is the cell's gradient by
   construction, not a copy. Then the **corpus** (§3.3) with its overlap
   check and the shared read of its samples, committed before anything
   runs.
2. **Alignment gate (2.7B `ultra20k-v4`, batch 1; decided 10-25: the
   step measurements run **on the box with an fp32 forward**, since the
      bf16 cast of the fp32 masters has a spacing of a few 1e-5 and at the
   sweep's lrs most coordinates may not move at all, so a bf16 "measured
   drop" can be cast noise and its deficit, linear in lr, would be fitted
   as curvature; the local run at 128 tokens does the parts that need no
   step: the hand-written ranking by sᵢ, the code paths, the fit on the
   box's logged numbers; the box run is the first 30 minutes of the
   dreamer session and aborts it on a fail):** for five fact rows (q = the row's first answerable question, **unknown** to the checkpoint by the (vii) rule, as every drawn fact question is; the (vii) pass runs first), **nine hand-written dreams each, three per kind** (decided 10-26, §4.46), 45 in all, written by the analyst and committed beside the gate script as `diagnostics/gate_dreams.json`: three **revisits** restating the passage **without the row's answer span** (a dream that quotes the answer makes the test trivial), three **acknowledgements** (`[USER] <sentence> [ASSISTANT] Got it.`-shaped, different sentences and acknowledgement strings), three **unrelated** dreams (UltraChat-shaped conversations about other things), at most 128 tokens each; a box rerun at 512 tokens uses the same 45 dreams, each in a batch with the row's own nine 512-token pool samples as fillers. Each hand-written dream gets a teacher row by forcing it
   through the model from the row's S_W (row 1 of §3.2) and recording the
   logits at its scored positions, as row 2 would have. Every "measured drop" below is the drop in the row's **true-reply loss L(C)** (decided 10-26, §4.47: bounded below, so its curvature is the one that "stops paying off"; the contrast loss's drop is logged beside it), and every prediction is lr·⟨G, ∇L(C)⟩. The script
   runs in this order, (vii) before (i), since (iii)'s oᵢ needs the gate's
   R (round 0's R draw, 34/33/33 from the first three sources of row 4,
   drawn by the gate from the corpus and handed to round 0 as its R). Any
   step that needs two processes (the floor's "two processes", and every
   instrument rerun) is run by the script itself, which spawns the second
   process and merges; the experimenter runs one command. (i) **lr sweep**, one per row, on the row's first revisit batch (the revisit dream plus nine pool samples of the row; each row has its own nine, the same nine for all of that row's batches): lr over {1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
   3e-3, 1e-2}, one **unclipped** SGD step each from the same θ, measured
   drop against the prediction; the row's α is the largest lr whose measured drop is at least half the prediction
   (§3.4); the cell's α is the **smallest** of the five rows' values; steps (iii) and (iv) use each row's own α. The fraction of trainable coordinates whose
   bf16 cast changes at each lr is logged beside the sweep, with the coordinate RMS and median |θ| over `ultra20k-v4/final/trainable.pt`, **which the gate script computes** (no figure is pinned here; the 10-14 note's median 0.0063 is LoRA A alone; the bf16-spacing argument for the fp32 forward of §3.4 is read against the script's figures).
   If every lr passes, the row's α is the top of the sweep and the row
   reports "no edge found", a reading; a row on which no lr passes **fails
   the row**; the cell's α is the minimum over the rows that passed, and
   the gate as a whole fails when fewer than 4 of 5 rows reach (iv) with a
   passing row (the one pass rule, stated once here). **c** is the largest ‖G‖ among the five rows' first revisit batches (§3.4 states the same once). (ii) **Noise floor**, per row: the first revisit batch's measured drop at the (iv) rung (one rung below the row's α) computed three times in each of two processes under the fp32 forward; the floor is the standard deviation over the six (the known non-reproducibility
   is across processes, §1.5 and §1.2; in-process repeats may be
   identical). A row whose prediction at that rung is not above its floor fails the row (nothing measurable to pass). (iii) Per hand-written dream: sᵢ in both forms
   of g_new (contrast and plain), oᵢ, ‖∇ᵢ‖, cos(∇ᵢ, g_new). (iv) **Discrimination.** Per row, each of the nine batches (one hand-written dream plus the row's nine fillers) stepped **at one sweep rung below that row's α** (at α itself when α is the bottom rung, and that row then contributes one point to (v)) (still inside the half rule, so near-linear, with
   more signal than the lowest passing rung: the tested dream is one of
   ten, so the between-batch difference is about a tenth of the batch
   drop), **unclipped**, from the same θ, one shared before-step loss per
   row, **measured fresh** (not the sweep's own values, which carry a
   selection bias). Statistic per row: the mean drop of the three revisit
   batches minus the mean drop of the six others. **Pass per row** by a
   permutation test: over all 84 ways of labelling three of the nine
   batches "revisit", the observed statistic is among the top 4 (p ≤
   0.05; the smallest attainable p is 1/84), and sᵢ (contrast) ranks each
   revisit above every acknowledgement and unrelated dream of the row.
   **Gate pass**: at least 4 of 5 rows. The null is the spread across
   dreams, which is the spread §4.40 names (acknowledgements vary widely
   dream to dream); the noise floor of (ii) is the measurement's own
   spread and is reported beside the statistic, not used as the test (the
   ninth read: under the fp32 forward the floor can be near zero, and the
   √2-floor rule then reduced to a sign test on one dream per kind, which
      a content-blind reward passes on 4 of 5 rows between one time in
   twenty and one time in five, depending on whether a row had to beat
   both other dreams or one).
   Per-row p values and the acknowledgement-against-unrelated statistic
   are readings. Fail → the reward is not live on this
   model; §3.2 (a) does not proceed, and the fallback is §4.8's supervised
   bridge first. (v) **h** by least squares with an intercept of (measured drop −
   lr·⟨G, ∇L(C)⟩) on −½·lr²·‖G‖² over the five rows' lr sweeps, **the rungs at or below each row's α
   only** (the rungs past the edge would dominate an lr²-weighted fit and
   overstate the curvature at α; this is what identifies curvature, where
   the step-(iv) batches share nine dreams and one lr and cannot);
      κ = α·h with the cell's α; the stop and no-edge rules are §3.2's (stop only when an edge was found and h ≤ 0; no edge → the fit if positive, else κ = 0, logged; a later checkpoint reuses the last good h). The §3.2 ratio reading is the
   in-run check. (vi) The gradient norms of revisit
   against acknowledgement dreams, a reading for §2's optimizer paragraph.
   (vii) The **retained pool** pass for this checkpoint (§3.3): 2,000 answerable questions drawn from the overlap-checked corpus pool outside every fixed set, rendered
   `[USER] q [ASSISTANT]` with no passage, teacher-forced token accuracy on the answer as the scorer computes it (`evaluation.py:65-84`) (SQuAD has one question per fact and no schematic template, so the scorer's rule `data.py:110-113` is applied on the one rendering): unknown at accuracy 0.0, known when at least one answer token is predicted correctly, the per-token correctness recorded for the retained-pool loss and the accuracy for the held-out ranking (§4.51) (the unknown side is the fact-row supply; a fact row's drawn q must be unknown, and "≥ 2 questions" counts the passage's answerable questions; the 2,000 candidates are redrawn per checkpoint, disjoint from the fixed sets).
      Steps (i), (ii), (v) and (vii) rerun on the box at 512 tokens on every
   checkpoint a dreamer round starts from and on the last round's final
   checkpoint (§3.2 step 1), and (i), (ii), (v)
   on every checkpoint a cell runs on (§3.4); the gate CLI takes
   `--init-adapter`, `--instrument-only` (steps (i), (ii), (v), (vii), no
   verdict) and `--output <dir>` for its `gate.json`. A rerun is an instrument
   measurement, not a verdict: "no edge found" sets α to the sweep top and
   is logged; a revisit drop below the floor on a dreamer checkpoint is
   logged and the round or cell runs with that α. Only the box gate on v4, at the dreamer session's start, decides whether arm (a) proceeds; the local run cannot step and does not decide. 780M is not a substitute: it has no v4 behaviour to measure.
   2b. **Double-backward gate (780M, local):** `torch.autograd.grad(L_in,
   θ, create_graph=True)` followed by a backward of the research note §6
   loss into the dreamer's logits runs through the manual Mamba2 mixer
   with gradient checkpointing (`use_reentrant=False`); the memory per
   128-token dream is recorded and extrapolated to 512; the entry point carries a finite-difference check of d⟨G, g⟩/dπ, since the Triton norms drop the second-order path without an error (research note §6, checked 10-26: fails as the models stand; the prerequisite is `model.torch_norms = True`, `models/norms.py`, added 10-26, and the 2b entry point sets it). Fail → arm (b) is not built; nothing else changes. **Run 10-27 (local, `make lama-ckl-gate ARGS="double-backward --tokens N --block 4 --fp32-forward"`, logs `sft/logs/alignment_gate-20261027-03*.log`):** the graph reaches π under `torch_norms` (fp32 relative error 1.2e-2; bf16 reads 1.0, so the check is read in fp32), but the memory above the 780M model is 1.85 GB at 4 tokens, 3.49 GB at 8, and 12 tokens does not fit the 8 GB box: about 0.41 GB per dream token, about 210 GB at 512 tokens by the linear extrapolation the entry point prints. Blockwise checkpointing gives nothing here, since `create_graph=True` keeps the recomputed graph of the first backward for the second. Read as a fail on memory for the manual mixer path at 512 tokens; the question for arm (b) is whether the fused CUDA path is twice differentiable (§4.52, §4.54).
3. **Dreamer session (box, one session of 12 h or less):** arm (a), two
   rounds per §3.2 with the live reading before round 0's update; arm (b)
   in a second dreamer session if 2b passed and (a)'s round-0 specificity
   reading was positive on most rows (a debrief judgment), same round
   structure, budget to be measured in that session (no estimate here);
   round logs pulled. The cycle-0 acceptance at one document on each dreamer checkpoint runs
   in the cell session after its re-pin (item 4), since the runner refuses
   an unpinned adapter: `make lama-ckl-run ARGS="--arm dream --cycles 1 --split <its split dir> --init-adapter <ckpt>/final --docs-per-wake 1 --dreams-per-cycle 10 --dream-batch-size 10 --dream-temperature 1.0 --wake-frames set --dream-loss-weights uniform --fp32-forward --optimizer sgd --lr <α> --clip-norm <c> --seed 42 --output <fresh dir>"` (`--cycles` evaluates nothing, so `--eval-every` has no place here) (the one distillation step it takes is under the cell's own settings), then
   the revisit rate by the revisit rule over the split's documents (more
   than half of 10 dreams is the reading, not a stop).
4. **Cell session** (§3.4): the v4 arm's cell, then per dreamer
   checkpoint the re-pin, split, baselines and 100-cycle cell; arm (b)'s
   cell in the same session if its checkpoint exists by then, else in a
   second cell session. Sessions in total after the gate: one or two
   dreamer sessions plus one or two cell sessions, so two to four.
5. **Thinking test** (recurrent depth, 07-23 roadmap) only after a cell
   survives its epoch: it changes the sampler every dream comes from and
   would confound attribution (§4.8).

## 4. Explicitly considered, rejected or deferred

1. **A reference-state weighting on the current loss ("the ruler").** A
   second teacher-forced wake with the documents cut out gives a shape-only
   state S_c'; weight each scored position by KL(teacher(S_c) ‖
   teacher(S_c')), so positions the documents did not change get about
   zero weight. One extra teacher forward over the dreams per cycle, about
   a third more time. **Recorded, not scheduled** by the teammate: it
   solves a problem the dreamer's reward also solves, and two fixes for one
   symptom muddy attribution. Returns if the dreamer track fails its gate.
2. **Lower learning rate under AdamW.** Rejected as the fix (§2): it
   delays the collapse in cycles and cuts the fact signal by the same
   factor. The optimizer change (§3.4) is registered instead. An AdamW
   dream cell on the dreamer checkpoint is acceptable as a control if a
   session has spare hours; none scheduled.
3. **Gradient matching (double backward) as the only dreamer signal.**
   Was deferred on 10-24 (dense local credit does not reach the
   sequence-level choice of topic; the double backward is unverified on
   this stack). On 10-25 promoted to arm (b) of §3.2, a second dreamer
   built beside ReST-EM rather than a term inside it, because the
   three-way comparison (selection, dense, untrained) is the cleanest
   reading of which signal makes dreams worth distilling. Gate §3.5 2b. Its
   math, with the sign convention corrected this session (a helpful token
   has ⟨∇ log p(v), g_new⟩ < 0), is research note §6; the KL guard to v4 is load-bearing for it (soup optimum), where arm (a) uses replay. Decided 10-26: arm (b)'s block carries the four g_old hinges of §3.2 in the same linear form beside its KL guard, so the arms differ in the learning signal and not in the guard.
4. **SEAL's reward (fine-tune, evaluate, grade).** Rejected: 30–45 s per
   dream for a scalar; the first-order alignment is the same quantity for
   one step at a small fraction of the cost (estimate: a fourteenth to a two-hundredth, 0.17–2.1 s against 30–45 s) and
   runs on the fused kernels.
5. **Sequential dream generation (each dream from the state after the
   previous).** Stays rejected (10-13 §4.6), strengthened: the echo moves
   into the state (dream k conditioned on k−1 of the model's own dreams);
   the empty-state student of dream k then distils the earlier dreams'
   content; 10 × 500 serial tokens is about 8 min per cycle. Coordination
   lives in the batch reward as a learned mixture. Greedy decoding is
   impossible anyway: ReST-EM selects among samples.
6. **No-reply wake.** Stays rejected (10-13 §4.5): the model must converse.
7. **RL from the raw base.** Rejected: rejection sampling cannot amplify a
   behaviour with zero hit rate (no markers, no EOC, no conversation).
8. **Supervised revisit warm start before RL.** Rejected as the first move:
   it bakes in our guess of the form, which the reward is meant to choose,
   and costs a session before any RL evidence. Kept as the §3.2 bootstrap
   bridge: one epoch of SFT on recall and question rows written from the
   dreamer corpus (the document, then a dream that restates it and asks
   and answers its questions; 10-13 recipe shapes), not LAMA, then back to
   round 0.
9. **Five documents per wake for the next cell.** Deferred: recency (§1.3)
   concentrates the dreams on the last row and the dreamer's first
   measurement should not carry that. Returns once a one-document cell
   survives, with g_new summed over the wake's rows.
10. **Thinking test before the dreamer.** Rejected (§3.5 item 5).
11. **Tuning the sampling temperature.** Rejected: 1.0 in RL and in the
    cell; the 0.7 was the third box's answer to greedy collapse and has no
    role under a trained policy.
12. **Running cycles 10–99 of the collapsed cell.** The experimenter's stop
    is endorsed: 30 of 30 identical dreams, gradient at the floor, wake
    unable to produce a different state. The full control profile is
    reproducible from the run note's command if ever wanted.
13. **Chained dreamer training (deferred, design recorded).** The cell
    distils the dreamer on its own dreams, and nothing in §3.2 trains or
    selects for what the dreamer does after that step; the fifth box's
    damage was exactly there. The design, for when the cell's per-cycle
    trigram revisit rate decays: within a round, chain L = 5 rows; dream
    row 1 from θ₀, take **prod's step** (a random n of the pool, not the
    kept batch, so the chain's states are the cell's states), dream row 2
    from θ₁, and so on; reset to θ₀ per chain. Every dream is generated
    once and used twice: as row i's candidates and as the reading of the
    step before it. Readings: row i+1's kept score against the same row
    dreamed from θ₀ (the **paired control**, 2× generation if done for
    every row; sampled at one row in five for the reading alone; a fitted
    prediction of the fresh score from the row's features is the cheap
    stand-in); and row 6 dreamed from θ₅, scored, never stepped, for the
    whole chain. Criterion, constant-free: the paired difference, stepped
    minus fresh. Its sign test is the exact statement of "maintain" but
    strikes half of harmless batches by sampling noise (both are best-of-40
    draws); its soft form weights the SFT loss by max(0, 1 + d/σ_d), d the
    difference and σ_d the round's spread, striking nothing. Rejected
    forms: a constant floor on the next row's score (a 10th-percentile
    floor strikes 10 % of harmless batches by construction, and with 10 chains per round (P = 50 at L = 5) the percentile
    is near the minimum); branching (K
    stepped futures per row, 4× generation); roll-back-and-repick on a
    failed reading (one extra generation only when it fires; the response
    if the sign test is adopted). For arm (b) the chain is truncated
    backpropagation through time with window W (research note §6): W = 1
    at L = 5 first, W = 2 if memory allows; credit reaches W rows back and
    no further.
14. **Length in the reward.** A reward that rises as length falls (cos −
    μ·len, cos / len) has the acknowledgement as its optimum: an
    acknowledgement's gradient is small but consistent, and a tiny length
    rescues a weak score. Rejected. A one-sided hinge above the wake's
    own length (ν · max(0, len − L_target) / `DREAM_TOKENS`) was proposed
    and withdrawn for the lexicographic rule, "among batches that clear
    the bar, the shortest", which has no ν and no target; with the keep
    threshold removed (§4.16) the lexicographic rule has nothing to be
    lexicographic under, so length is a reading in round 0 (§3.2 step 5) and the rule returns, as a tie-break on the greedy assembly's last
    slot, if kept length rises across rounds. The reason length matters
    at all: fewer dream tokens per wake token lowers the self-consuming
    gain of research note §7 (fifth box: 17–40 dream tokens per wake token,
    cycles 0–6 [`token_ids` minus `prefix_tokens` over `dreams.json`
    against `prompt_token_ids` + `assistant_token_ids` over `wake.json`]).
15. **Loss weighting forms** (§3.4 picks per-dream per-token). Per-dream
    gradient normalisation (unit ‖∇ᵢ‖): rejected, it removes the
    self-limiting property (a learned dream has a near-zero gradient and
    contributes nothing; cycles 7–9 at loss 0.009 would have taken full
    steps) and makes the reward score a step the cell does not take.
    Batch per-token mean (`training.py:83`, the baselines' form):
    rejected for the dream arm, a dream's pull scales with its length, so
    long chains dominate and the reward's marginal favours long dreams.
    Per-turn weights (10-14 §3.1): replaced, the ≈10× per-token weight on
    a four-token acknowledgement turn beside the confabulated document turn
    of a wake-shaped dream (§2; the exact ratio is a census computation).
16. **Keep thresholds (τ on the batch score, a floor on the lookahead).**
    Rejected for round 0: best-of-pool with no threshold is plain ReST-EM
    and works whenever the reward is alive; a dead reward is a reading (§3.2 steps 2 and 5), not a selection rule. The teammate's standing
    preference, recorded: no constant cut anywhere; every cut is relative
    to the round's own distribution or is a weight.
17. **Cosine-only learning term.** Replaced by the projection (§3.2): the
    cell's SGD step scales with ‖G_B‖, so two batches of equal direction
    and different size make different progress, and a cosine cannot see
    it. The cosine was chosen on 10-24 against magnitude gaming (long
    dreams, soup); that risk belongs to arm (b), which optimises token
    probabilities directly; arm (a) selects among the policy's own samples, where the magnitude
    levers are per-token surprise and length (under the uniform per-token
    mean of N weakly correlated position gradients, ‖∇ᵢ‖ falls roughly as
    1/√N, so a short dream of equal per-position KL has the larger norm
    and the magnitude-sensitive first term favours it; the quadratic term
    pulls the other way; the kept-length reading is where the net shows).
    Cosine stays logged and is the live reading's second statistic.
18. **Two-sided reference term (reward for lowering the reference loss).**
    Deferred. It would reward rehearsal directly; its cost is that the
    dreamer can then earn reward by rehearsing what the model knows,
    ignoring the row. One-sided plus the learning term forces every kept
    batch to learn something. Trigger: the cell's revisit rate holds but
    LAMA retention drifts; then the term goes two-sided with a small
    weight.
19. **KL guard to v4 in the ReST-EM policy update.** Replaced by replay
    (row 6): the same job (keep the dreamer's conversational behaviour)
    with no reference forward and no coefficient; the held-out warm-start
    loss is its reading. The KL guard returns for arm (a) if that loss
    rises across rounds; it is load-bearing for arm (b) regardless.
20. **Actual lookahead as the primary signal** (take the step, dream the
    next row, score). The chain of §4.13 is this made efficient; the
    standalone form (one extra generation per candidate batch, about 4×
    round time) is dominated by it.
21. **Filtering or re-anchoring conversation targets with dangling
    references** ("swap day 2" names nothing without the prefix). Rejected
    (10-25): the reward is relative within a row, so the row still ranks
    dreams correctly at a lower ceiling; a hand-written cue guesses what a
    user would say. Accepted as noise, small share (15 of 50), the
    conversation rows' kept-score mean is the reading.
22. **Trained-on rows as g_new sources** (Dolly plain rows, the first
    20,000 UltraChat conversations). Rejected: a continuation the warm
    start trained on is partly memorised and is not a new fact. They go to
    g_old and replay, where memorised is the point.
23. **One fixed abstention string as the unanswerable target.** Rejected
    (teammate, 10-25): a hard-coded phrase pays for the phrase, not for
    abstaining. The abstention set is mined from the trained-on UltraChat
    rows and drawn per row (§3.3).
24. **The LAMA retention set as the retained pool.** Rejected: the hinge
    would then protect exactly the facts the scorer measures, and the
    retention curve would be partly trained on (a §3.1 non-goal). The retained pool is the same known-or-unknown test
    on SQuAD, and §4.18 stands beside it: fact rows are drawn from the
    unknown side, so a known fact is never a learning target that pays
    for rehearsal.
25. **Uniform per-dream weighting's cost** (recorded, not changed): a
    short acknowledgement dream carries the same total pull as a 500-token
    revisit, about 100× per token. It is the price of per-dream equal
    weight over the batch per-token mean (§4.15), chosen by the teammate;
    selection is what keeps acknowledgements out of the kept set, and the
    acknowledgement count among kept dreams is the reading.
26. **The 0.3–0.5 first-step-norm anchor for the SGD learning rate**
    (10-24 draft). Withdrawn: Adam's first step on d = 21.3M coordinates
    has norm lr·√d = 0.4618 whatever the data (the census's cycle-0 value
    0.46662 sits 1 % above the bound; the tenth read's sub-readers
    recomputed the norm locally over `cycle-00/trainable.pt` against the
    pinned v4 and got 0.4601 in float64 with every coordinate's |Δ| at or
    under 1.0004e-4, and 0.4551 through the census's own float32 code, so
    the excess is reduction spread in the census, not a property of the
    step; the recomputation is not a committed script, which is why it is
    cited as the reader's and not as a note number), so the anchor carried
    no information. Replaced by the edge-of-linear-regime rule
    (§3.4), whose one constant (half) is named as such.
27. **The fifth box's v4 cell as arm (c).** Rejected: it ran 5 documents,
    30 dreams, AdamW, per-turn weights, temperature 0.7 and no clip.
    Nothing a dreamer cell does under §3.4 would be attributable to the
    dreamer against it. The v4 arm is rerun under §3.4 with
    kill-on-collapse.
28. **Scoring g_new against the true reply alone** (10-24 draft).
    Replaced by the contrast form (§3.2 row 3) after the second
    adversarial read: a SQuAD reply is a short span then a stop, so a
    large part of its gradient says "reply briefly, then stop", which an
    acknowledgement dream teaches exactly. The plain form is logged beside
        the contrast form in the gate for the comparison.
29. **The constants this note carries**, each a pin or a reading, none a
    keep rule on the dreamer's own outcome: the half in the lr rule (§3.4,
    a pin); "4 of 5 rows" in the gate's pass (§3.5, a pin on a 5-row
    test); the scorer's known-or-unknown rule for the retained pool (0.0 unknown; known at one or more correct answer tokens, §4.51; on SQuAD's one rendering, `data.py:110-113`); the 2,048-token prefix cap and the 100-cycle cell length (scheduling
    pins, §3.4); p ≤ 0.05 in the gate's permutation test (top 4 of 84 labellings; a verdict needs a cut, and the per-row p is logged beside it) and three hand-written dreams per kind per row (a sample size); 20 random batches for the
    specificity reading and K = 4 (sample sizes, readings); at most 30
    words and under 20 phrasings
    for the abstention set (a mining filter and a reading); at least 2
    questions per passage (a corpus filter against the "dream the pair"
    optimum); "more than half of 10 dreams revisit" in the acceptance
    cycle (a reading); the two-trigram, non-overlapping, one-document
    revisit rule (a measurement definition); the last good κ as the fallback
    when a round's fit fails (§3.2); P = 50 (35/15), R = 100 (25 per
    source), 20 prod, 5 gate and 50 held-out fact rows, 200 held-out
    warm-start rows, 2,000 retained-pool candidates (sample sizes); 1:1
    replay, 2 epochs, batch 8, AdamW 1e-4, weight decay 0.01, 512-token
    chunks (dreamer SFT pins, §3.2 row 6; the warm start ran 1 epoch at
    batch 6, `sft/Makefile:305-326`, so these are not "reused"). Removed this revision: the 0.25× norm
    flag, the "more than half of rows" bootstrap stop, and "more than a
    few percent" of clip fires, all now plain readings. The 10-24 claim
    that the half was "the one constant" was false.
30. **Unanswerable questions as g_new rows** (10-25 morning draft).
    Rejected the same day: the contrast "abstain rather than guess" points
    the same way on every such row, a consistent reply-shape direction,
    the carrier class §2 names, and a §3.1 non-goal. They are g_old rows
    instead: abstention is protected by the hinge and never paid for.
31. **One literal wake frame.** Replaced by a frame set (§3.4): the
    `sft/AGENTS.md` rule against a literal the model sees hundreds of
    times applies to the frame as much as to an opener, and the dreamer's
    SFT (row 6) conditions on the frame, so a literal would become the
    dreaming cue. The frame is never a loss target; the frame-string count
        among kept dreams is the reading for the copy risk.
32. **Hinge weight above 1.** The adversarial read (fifth, 10-25) argued
    that a hinge on the mean over 100 reference rows is "100× weaker" than
    a learn term on one row. Both terms are nats per row, so weight 1 is
    one old row against one new row; a larger weight is a design choice.
    The teammate keeps 1 and revisits if a cell's retention falls while
    its dreams stay healthy.
33. **An automatic stop on the live reading** (fourth and fifth drafts).
        Dropped by the teammate: the box gate at the session's start is the
    liveness check (the local run cannot step, §3.5 item 2), and the
    session runs to its end with the reading logged. "Median above" was a
    coin flip under a dead reward in any case.
34. **Stop tokens in the learning target.** Removed (§3.2 row 3): with
    the per-token mean the two stop tokens weigh 2/(|a|+2), so a wrong
    answer of a different length leaves a "stop here" component in the
    contrast. Length-matching C⁻ was the alternative; excluding the
    tokens cancels exactly. Answer type (a date against a name) survives
        the contrast and is a recorded residual. That residual can also make
    the specificity reading positive with no document content, so the
    arm (b) decision reads it beside the kept dreams' text, not alone.
35. **Contrast for conversation rows.** Dropped on the sixth read's
    objection (a contrast against another real conversation pays for
    raising its loss) and restored the same day with the reply swapped
    and the context kept: the wrong reply is scored after *this* row's
    follow-up, so lowering it here touches nothing in its own context,
    exactly as a wrong span under a fact question does. Without the
    contrast 15 of 50 rows would pay for reply style, the §2 carrier. The
    residual: an off-topic reply carries a small "be relevant" component
    the true reply does not share, which points toward attending to the
    conversation.
36. **One pooled hinge.** Replaced by one hinge per reference source
    (§3.2 row 4): pooled, a step that raises the kept-dream loss and
    lowers the fact loss by more passes unpenalised, and the kept-dream
    source was added as the guard on dreaming itself. **One hinge per
    row** (100) was considered: it costs little, but a one-sided hinge
    over 100 noisy single-row gradients keeps about half of the noise as
    small rises, a tax that grows with step size and favours small
    gradients again. Four for now (teammate, 10-25); the per-row changes
    are logged every round, and batches that hurt single rows while the
    four hinges stay silent are the trigger for per row. The kept-dream
    hinge guards the likelihood of last round's kept strings, a proxy for
    dreaming, not dreaming itself.
37. **Full-weight SFT on every kept batch.** Replaced by the
    positive-score weight, then by the gated specificity weight (§3.2,
    §4.45): on a dead row the best batch is the
    smallest-gradient one, so full weight trains toward document-free
    dreams at the share of dead rows. A danger the fourth draft logged
    and did not act on.
38. **Format convergence in the dreamer** (recorded danger): a format
    component orthogonal to g_new is neither paid nor charged, so
    selection can converge on one winning dream shape, which by §2's own
    argument is the next carrier in a cell that distils unselected
    samples. No term measures cross-cycle consistency; the dreamer's
    kept-dream census (literal counts, acknowledgement and frame counts,
    `unique_dreams`) per round and the cell census per cycle are the
    readings. **The wake-reply leak is in all three arms**: the teacher's state holds the sampled reply, so its distribution puts mass on ` Got` at every assistant position whatever the dream says (§1.2: 0.004 → 0.158 in the adapter after cycle 1, two steps whose 60 distilled dreams held three with the string), and the student copies that distribution at every scored position. Selection changes which dreams are distilled, not that leak; only sampling the reply at 1.0 varies it. The three cells therefore test the hypothesis that dream content outweighs the leak; if all three collapse alike on the reply, the next lever is the wake reply itself (the ruler of §4.1, or the no-reply wake 10-13 §4.5 rejected), not the dreamer.
39. **The α rule and survival** (recorded objection): the rule picks the
    largest step inside the linear regime, which by §2's own argument is
    also the fastest accumulation of whatever is consistent. No reading
    ties α to survival; the cell census's acknowledgement trend per cycle
    is where it would show, and a smaller α is the first knob if the
    dreamer cells collapse with healthy kept dreams.
40. **Selection against high-variance acknowledgements** (recorded):
    under the contrast form an acknowledgement's expected sᵢ is near zero
    but its spread is the largest (‖∇ᵢ‖ ∝ 1/√N, §4.17), and best-of-40
    favours spread. Whether the quadratic term and the hinges win is not
        shown; the acknowledgement count among kept dreams is the reading.
41. **The user follow-up as a target** (10-25 morning draft, "loss on
    both" turns). Withdrawn: with the same follow-up on both sides of the
    contrast its gradient cancels, so no dream was ever paid for it; a
    second contrast (the follow-up against another row's follow-up, both
    from empty) would pay for it at two more backwards per row. The
    teammate chose the simple form, reply only, and defers "learn about
    the user" to the personality rows (§3.3).
42. **bf16 forward in the dream cells.** Replaced by `--fp32-forward`
    (§3.4): the 10-14 masters fixed the stored step; the forward still
    read a bf16 cast, which under SGD at a small α filters one-off updates
    and passes consistent ones. The case arose with the move to SGD, not
    with the 10-14 decision.
43. **Baselines at 30 epochs** for a comparison read at epoch 1
    (recorded): kept, so the dreamer checkpoints' lora and mix-review
    curves are comparable to the banked v4 ones at every epoch; 2.5 h per
    checkpoint in parallel with the cells.
44. **The "unknown without passage" filter and underspecified questions**
    (recorded): "What year did it open?" is unknown for want of a
    referent, not of a fact, and g_new then pays for binding an
    underspecified question to a span, the §4.21 case on the fact side.
    The generator logs the share of fact rows whose question has no
        capitalised noun, as the reading.
45. **The SFT weight as the score's size** (fourth to eighth drafts).
        Replaced (10-26) by the two-part rule of §3.2: sign from the full
    score (absolute, zero the reference), magnitude from the row's
    random-batch specificity gap (relative, own row against the mean of
    the other rows of the kind, on unselected batches), normalised within
    kind over the positive rows. A first draft took B*'s own gap; the
    tenth read showed it carries the selection bias the rule exists to
    remove (B* maximises its own-row term, the other-row mean is not
    selected on), so under a dead reward it is positive on most rows too. The ninth read showed that B*,
    a best-of-many, scores positive on most rows under a dead reward, so
    a score-sized weight trains on noise at full strength and the
    zero-weight share cannot read a dead reward. Pure specificity was
    considered and rejected: the quadratic and hinge terms cancel in the
    gap, so a batch that moves away from its own target but further from
    the others would be trained toward, with no hinge on the weight; SFT
    only pulls, and the best of a bad pool is still bad (ReST-EM filters
    on absolute reward for this reason). The cost accepted: a row whose
    pool is all bad gives the dreamer no signal that round, and a row at
    zero across rounds is itself the collapse signal. The analyst's
    recorded objection to the within-kind normalisation: dividing by the
    mean rescales a round in which nothing is specific to full strength
    and erases the absolute size of the gap; the alternative is a
    measured scale, the gap divided by the box gate's own revisit-minus-
    others statistic at α (median over the five rows), so that weight 1
    means "as row-specific as a hand-written revisit". The teammate keeps
    the mean; the gap's distribution is logged per round so the scale can
    be read.
46. **The gate's √2-floor pass** (fifth to eighth drafts). Replaced by
    the permutation test over nine hand-written dreams per row (§3.5 item
    2 (iv)). The floor measured the repeat spread of one measurement; the
    confound is the spread across dreams of one kind (§4.40), which one
    dream per kind cannot see, so the row could pass or fail on which
    acknowledgement was written. Cost: 30 more step-and-measure runs, a
    few minutes. The 0.05 is a constant and is listed in §4.29.
47. **α and κ measured on the contrast loss** (second to eighth drafts).
    Replaced (10-26): the contrast L(C) − L(C⁻) has no lower bound, its
    curvature is h(C) − h(C⁻) with the wrong answer's Gauss-Newton part
    subtracting, so h ≤ 0 was a structural possibility and the gate could
    fail by construction. α, the floor, the discrimination drops and h
    are measured on L(C), which is bounded below; g_new stays the
    contrast, so the direction selected on is still content and not reply
    shape (§4.28). The plain ∇L(C) therefore becomes load-bearing in the
        gate and is measured on the gate rows (§3.2 row 3).
48. **Chunked backward in the dreamer's SFT** (first 10-26 draft).
    Replaced by one unchunked backward over the pair (§3.2 row 6): the
    warm start's loop detaches the recurrent state at every 512-token
    chunk (`training/loop.py:627`), and with the loss on the dream only,
    prefix chunks get no gradient, so the dreamer would learn what to
    dream and not how to read the wake. Memory at 2,560 tokens is
    unmeasured; the teammate accepts the cost (10-26) and the batch drops
    if it does not fit.
49. **Per-checkpoint α in the cells** (10-25 draft). Replaced by one
    common α and c over the arms (§3.4): with α re-measured per
    checkpoint, arm (a) could survive by a smaller step and not by its
    dreams (§4.39), and no reading would separate the two. The cost: the
    arm with the largest own α runs below its edge. Each checkpoint's own
    α is reported beside the comparison.
50. **Gate readings recorded, no rule** (tenth read): the gate verifies
    drops in L(C) while the score's first term predicts a drop in the
    contrast, so a pass shows the step is described, not that the
    selected quantity is; the contrast drop is logged beside L(C)'s at
    every gate step, and the sᵢ(contrast) ranking is the one contrast
    check in the pass. The L(C) drop at α on the gate rows is also logged
    against the drop a row needs to go from accuracy 0 to 1.0 (the row's
    L(C) itself, in nats); if one step at α is a small fraction of it,
    "survived, acquired zero" in every arm is a step-size result and not
    a content result, read at the debrief.

51. **The known rule for the retained pool and the held-out rows** (10-26,
    after the generator's first scan). The pinned rule, answer accuracy
    1.0 with no passage, found 4 known passages among the 879 LAMA-clean
    SQuAD passages on v4 (`sft/logs/lama-ckl-corpus-20261026-193741.log`:
    291 unknown, 584 middle, 4 known), so the retained pool and the 50
    held-out rows could not be filled. SQuAD answers are multi-token spans
    and v4 reproduces few exactly; most facts are partly known. Revised,
    both relative, no new constant: (1) a retained-pool row is a question
    with at least one answer token the checkpoint predicts correctly, and
    its g_old loss is the cross-entropy on those currently-correct answer
    tokens and the two stop tokens, so the guard holds what the model has,
    token by token; a question with no correct token has nothing to retain
    (the unknown side, accuracy 0.0, stays the fact-row supply); (2) the
    50 held-out over-abstention rows are the 50 highest-accuracy rows of
    the v4 scan, ties in the seeded order, excluded from every pool.
    Consequences, stated: the guard protects partial knowledge, charging a
    step that raises the loss on a token the model already gets right and
    reading nothing on the tokens it gets wrong; the held-out rows are the
    facts v4 knows best, not facts it answers exactly, so the
    over-abstention rate is read as a change across rounds, like the
    held-out loss. Considered and not taken: "answered" by the greedy
    reply's abstention stem (no correctness in the selection, and a
    greedy decode of every candidate); accuracy-weighted row losses (the
    gradient then carries the tokens the model gets wrong, a learning
    direction inside a retention guard).

52. **Arm (b) outline** (10-26, agreed in conversation; its mechanism
    spec block is still to be written into §3.2 and read before it is
    built, §6.6). (1) Built, after gate 2b under `model.torch_norms = True` (research note §6, the plain-torch norm path added 10-26). (2) Objective: arm
    (a)'s score(B) exactly (learn term, quadratic term, clip, four
    hinges) on the sampled batch of n dreams, maximised by gradient
    through the teacher logits πᵢ instead of by selection: policy forward
    from S_W over the sampled dream with the graph kept gives πᵢ; the
    student forward from empty gives ∇ᵢ = `autograd.grad(KL(πᵢ, student),
    θ, create_graph=True)`; G_B, the clip and the score as in arm (a)
    with g_new and g_old^k constants; `(−score(B) + KL guard).backward()`
    steps θ. (3) The student is the fixed instrument: it runs on a
    detached copy of θ (`torch.func.functional_call`), so the score moves
    θ through πᵢ only. (4) Credit flows into the soft targets at the
    scored positions the sample visited, uniform 1/N as row 5; the token
    choices are sampled and not differentiated, so a dream on the wrong
    topic gets no signal to change topic (the research note's known
    limit, the reason arm (a) stays beside it). (5) The KL guard to v4
    on the dream distribution, its coefficient set at each round start so
    that its gradient norm on the pool equals the learn term's (a
    normalisation measured in the gate, no constant; open to a better
    normaliser). (6) Same replay 1:1, rounds, optimizer, lr and corpus
    rows as arm (a), so the arms differ in the optimiser of the same
    score. Known costs: a double backward over 2.7B per dream, memory two
    to three times the single backward, slower norms; gate 2b measures
    them, and its finite-difference pass is read against the two-process
    noise of the same quantity rather than a fixed relative error.

53. **Corpus generator readings, 10-26, three open decisions for the
    teammate** (generator built and run on v4, §6.3; logs
    `sft/logs/lama-ckl-corpus-20261026-223225.log` and
    `-220427.log`). (a) **The LAMA filter removes almost everything
    conversational.** The word-bounded match of every candidate subject
    and object string (11,599 distinct) hits common words: the most
    frequent first matches over the 20,000 warm-start conversations are
    "life" (886), "language" (805), "word" (638), "many" (489), "key"
    (434), "home", "history", "brand", "yes", "management". Survivors: 127
    of 20,000 warm-start UltraChat conversations, 60 of 1,771 Dolly plain
    rows, 1,000 conversation candidates from 154,190 read (0.6%, leaning
    to code help), 975 of 19,029 SQuAD passages. So the **warm-start
    replay pool is empty** and the held-out warm-start set holds 127 of
    200. Analyst's recommendation: a text teaches a LAMA fact only if it
    carries both its subject and its object, so match the filter on the
    **pair** (subject and object of one candidate both present,
    word-bounded, case-insensitive) rather than on either string alone; a
    conversation that says "yes" or "many" leaks nothing. The fixed sets
    drawn under the current filter stay valid under the pair rule (it is
    strictly looser), so the gate rows and dreams keep. (b) **The mined
    abstention set is the wrong kind of refusal.** The six stems find 20
    phrasings after the filter, 19 of them "I don't have ..." and nearly
    all capability refusals of the chat model ("I don't have the
    capability to visit places", "I don't have childhood memories"), so an
    abstention row reads `Why is Symbol not useful for all instances?` →
    "I don't have childhood memories." Recommendation: keep only stems in
    the knowledge sense (`I don't know`, `I'm not sure`, `I don't have
    (enough |any )?information`, `I'm not aware`, `I cannot (say|tell)`)
    and read the set size again; if it is under ten, the abstention source
    is a reading and the hinge runs on three sources until a better set
    exists. (c) **One-token known rows.** Of 1,271 known retained-pool
    rows, 531 are known on one answer token, often punctuation or an
    article (`"clean picture"` correct only on the closing quote; `from a
    pre-industrial …` on ` a`, `-`, `industrial`). Recommendation: a
    correct token counts for the known rule and the guard loss only if it
    contains a letter or a digit (no constant; punctuation and
    whitespace carry no fact). Also recorded: the v4 scan at batch 4 gave
    290 unknown where batch 8 gave 291 (bf16 padding numerics); the 50
    held-out rows span accuracy 0.72–1.00; the retained pool is 1,271
    known / 729 unknown of 2,000; §4.44's reading is 58.9% of fact
    questions with no capitalised word after the first.

4.54. **Night of 10-26/27, build state and local readings (recorded 10-27).**
    Built and committed: the alignment gate (§6.5) with step (vii) under
    `--fp32-forward` writing `retained_pool-<sha8>-fp32.json`, the warm-start
    R rows at 7 Dolly : 26 UltraChat (filled from the other pool with a
    WARNING when one is short), the §4.51 facts loss, `torch_norms` in 2b;
    the dreamer trainer arm (a) (§6.6, `make lama-ckl-dreamer`); `report.py`
    8b; the runner 7b edit (protected, uncommitted, for the teammate's
    review). Readings: (1) the 2.7B gate does not fit this box, the 128-token
    fit probe fails at 7.72 GiB in the checkpointed backward, so the gate's
    first run is the box run (§3.5 item 2); no 780M adapter in the gate's
    format exists, so there is no local end-to-end smoke; (2) gate 2b: see
    §3.5 2b, memory fail at 512 tokens on the manual path; (3) the archive
    move of §6.8 is done (`.cache/lama_ckl_archive/fifth-box/{runs,acceptance}`),
    so §1's artifact paths now resolve under that prefix. Open for the
    teammate beside §4.53 (a)–(c): (d) the dreamer SFT batch loss follows
    row 6 literally, Σ w·L / Σ w over each batch of 8, which cancels the
    weights inside a batch whose rows share one weight; the builder
    recommends Σ w·L / batch size (the weights already have mean 1 within a
    kind), and the choice is the teammate's; (e) with the corpus as it
    stands (0 `warm_start` rows), R's warm-start source is 33 Dolly rows and
    the replay pool is the Dolly plain rows minus R (about 27 to 53 rows
    against about 500 kept pairs), logged as WARNINGs and run on; whether a
    short replay pool stops the round is undecided (the adversarial read of
    10-27 ranks this first). The adversarial read's merged report is in the
    session handoff, not in this note.

## 5. Process

- **Banking** (done, the teammate's merge `dbc78a8`; `7b863a0` is the
  skill fix): the box commits its run note and
  shutdown receipt on `origin/box/<id>`; banking is the teammate's merge
  of that branch into `main`, never an analyst commit of the rsync'd copy
  (`.agents/skills/altrux-debrief/SKILL.md` step 1). This session's first
  action was that wrong commit; it was undone and the teammate merged.
- **Counters have semantics.** `subjects_mentioned` sat behind "one in
  thirty" for a debrief and a run. A diagnostic's name is not its
  definition; the note that cites a counter names what it counts (§6.1
  adds the dream-level revisit count beside it).
- **The adversarial read (10-24) found arithmetic in the note that was
  not computed over the artifact** (an acknowledgement ratio from a line
  count, a recency count of 17 of 19 that was 15 of 18 under the 10-24
  rule and is 12 of 14 under the pinned one, a loss ratio of "4–5×" that
  pools to 2.9× and then 3.3×) and a sign error in the dense term's
  per-token reading. All corrected in this revision; the rules behind the
  counts are now pinned in the preamble so a recount is mechanical.
- **The reward was redesigned in conversation (10-25)** on the teammate's
  questions, not on the analyst's draft: continuation target, magnitude in
  the score, no constant thresholds, prod's step as the chain's step.
  Record of the thread is §4.13–§4.20.
- **The second adversarial read (10-25)** found the score's units mixed
  (three terms normalised three ways, a hidden hinge weight), the batch
  gradient written as a sum where the cell takes a mean, the lr anchor
  that was a parameter count, the cycle cost halved against the fourth
  box's measured 10-dream cycle, a "control" that ran a different
  protocol, and a reward that would pay for reply shape. All folded in
  (§3.2, §3.4, §4.21–§4.28). The derivations were right and the
  bookkeeping around them was not; the lesson for the brief is that every
    quantity in a formula names its unit and its source row.
- **The third adversarial read (10-25)** found the dead-reward reading
  non-negative by construction, the gate's pass unable to attribute a
  drop to the one dream it tested, the unanswerable rows paying for a
  reply shape, the clip a step norm where the code clips a gradient, the
  revisit rule counting generic phrases (four of eighteen counts), and a
  list of unpinned flags, paths and orders. All folded in (§3.2–§3.5,
  §4.29–§4.31, §6). The pattern across the three reads: the mechanism
  survives; what fails is a reading that cannot move (a comparison whose
  sign is fixed), a count whose rule admits junk, and a unit written
    without its factor.
- **The fourth adversarial read (10-25)** found the specificity reading
  biased by selection (fixed: random batches), the second-order term
  claimed as a guard for old rows it cannot see (withdrawn), the prod
  reading moving with the re-measured instrument (fixed: round 0's
  instrument), the parallel baselines over the GH200's memory (fixed:
    one at a time), and the gate's reruns, pins and open cases. Folded in.
- **The fifth adversarial read (10-25)** confirmed every run number and
  found the registered cell command scoring nothing (`--cycles` is the
  acceptance flag), the live-reading stop a coin flip, the κ fit
  unidentifiable from fifteen steps at one lr, the wrong-answer length
  leaving a stop component in the contrast, and a false "plus weight
  decay".   Folded in; §4.32–§4.34 record the three decisions.
- **The sixth to ninth adversarial reads (10-25 and 10-26)** confirmed
  every run number against the artifacts (the ninth recounted every §1
  table cell, the probe table, the Adam null and the baselines) and moved
  on to the plan: the conversation contrast paying for another
  conversation (§4.35), the pooled hinge (§4.36), full-weight SFT
  (§4.37), the reply leak shared by all arms (§4.38), α against survival
  (§4.39), high-variance acknowledgements (§4.40), the follow-up target
  (§4.41), the bf16 forward (§4.42), then in the ninth a kept-dream guard
  that never acts in the registered rounds, a gate floor that measured
  the wrong variance, an SFT weight that could not see a dead reward, a
  κ that could fail by construction, a §2 premise that §4.38 contradicted,
  and some forty pins (states, dtypes, paths, orders, arithmetic). Folded
  in; §4.45–§4.47 record the three decisions. The pattern of the later
  reads: the mechanism holds, and what fails is a statistic whose null is
  the wrong spread, a weight that cannot read the case it exists for, and
    a plan quantity stated in two places two ways.
- **The tenth adversarial read (10-26)** confirmed every §1 number again
  and found the 10-26 material inconsistent with itself: §4.47 applied in
  §3.5 but not in §3.2; the new SFT weight taking B*'s gap, which carries
  the selection bias it exists to remove; row 6's chunked backward unable
  to reach the prefix; the h-sign stop firing on noise when no edge is
  found; the fourth box's voided distillation timings still in the
  budgets; the flicker exclusion unable to catch three of the four
  flicker rows; and the arms at different α. Folded in; §4.48–§4.50 and
  the §3.2 h rule record the decisions.




## 6. Housekeeping

Pending, by consent, in this order (item 7a comes before item 5, which
imports it):

1. **Dream census additions** (`sft/experiments/lama_ckl/diagnostics/dream_census.py`,
   unprotected): per cycle, the §1.3 dream-level revisit count per wake
   turn and the §1.2 short-turn and acknowledgement counts by the pinned
   rules, beside `subjects_mentioned`; clip fire and step norm per cycle;
   the per-cycle summary prints the literal-count column for the most
   frequent short `[ASSISTANT]` string. Cycle directories sort numerically
   (`cycle-{:02d}` breaks the lexical sort at cycle 100).
2. **`diag_gotit_prior.py`** (written 10-24, §1.2): committed under
   `diagnostics/`; its output is in `sft/logs/diag_gotit_prior-20261025.log`
   (the 10-25 rerun, adapters `cycle-{01,04,05}/` and the warm start).
   **`adam_momentum_null.py`** (written 10-25, §1.2): the Adam lag-1
   cosine null, output `sft/logs/adam_momentum_null-20261025.log`.
   `sft/logs/` is gitignored, so the table in §1.2 is the durable record
   and names the script and the adapters that produced it.
2b. **Acquisition census** (`dream_census.py`): the §1.1 count has no
      code yet; add it with the one definition of §1.1 and §3.1 (acquired: a
   to-learn row at descriptive accuracy 1.0 at the first evaluation at or
   after its wake cycle, cycle = row index // documents per wake, and 0 at
   every earlier evaluation; kept: also 1.0 at the run's last evaluation;
      rows above 0 at any frozen ×3 evaluation or at any evaluation before
   their wake cycle are left out of the count and the denominator, §1.1),
   reading `docs_per_wake` and
   `eval_every` from `run.json` and the frozen runs from `--frozen <dir>...`.
3. **Dreamer corpus generator** (§3.3,
   `sft/experiments/lama_ckl/dreamer_corpus.py`, unprotected): SQuAD 2.0
   from the HF cache, UltraChat by index, Dolly plain rows from
      `data/dolly_context.jsonl`; writes `data/dreamer_corpus.jsonl` (rows
   with `kind`, W, C, C⁻), `data/dreamer_abstentions.json`,
   `data/dreamer_fixed_sets.json` (prod, held-out, gate rows), and the
   retained pool per checkpoint (`data/retained_pool-<sha8>.json`, sha8 =
   the first eight hex digits of the checkpoint's `trainable.pt` sha256, the
   known-or-unknown pass as its own entry point, §3.5 item 2 (vii), run on
   the box for each checkpoint a dreamer round starts from). The corpus
   file holds **candidate pools** (fact candidates with all their
   questions, conversation candidates with their valid boundaries, the
   fixed sets), the overlap check (`build_candidates` on the `--lama-root` the v4 split
   was built from, passed to the generator the same way) applied once at generation; the dreamer draws each
   round's rows from the pools (fact rows from the checkpoint's unknown
   side, C⁻ paired at draw time, duplicate passages allowed and counted),
   and the shared read covers the pools' decoded samples. Prod, gate and
   held-out rows are fixed under v4 and not re-filtered; the generator runs the known-or-unknown pass on v4 locally (the 2.7B forward fits in 8 GB in bf16) so the prod and gate rows are v4-unknown and the 50 held-out fact rows are **v4-best-known** (the 50 highest-accuracy rows of the scan, §4.51; they read over-abstention: an abstention on a fact the model answers is the damage; on an unknown fact it would be correct, and the reading would have the wrong sign). The corpus file materialises **every row the dreamer reads**: the P pools, the g_old sources (retained-pool candidates, abstention rows, Dolly plain rows, the warm-start UltraChat rows), the replay draw's pool and the fixed sets, so nothing is loaded from HF on the box. The decoded-sample
   log for the shared read.
4. **Master-weight wrapper generalised** (`sft/training/master_weights.py`,
   unprotected): `MasterAdamW` becomes a wrapper over any optimizer class,
   so SGD runs through the same fp32 masters.
5. **Alignment gate script** `sft/experiments/lama_ckl/diagnostics/alignment_gate.py`
      (§3.5 item 2): five fact rows from the corpus, the 45 hand-written
   dreams in `diagnostics/gate_dreams.json` (analyst writes, committed),
   teacher rows by forcing from S_W, sᵢ (both g_new forms) / oᵢ / ‖∇ᵢ‖ per
   dream, the lr sweep → α, c, the noise floor, the 45 batch steps with
   measured L(C) drops and the per-row permutation test, the κ fit,
   revisit-vs-acknowledgement norms, the coordinate statistics; flags
   `--init-adapter --instrument-only --output --tokens 128|512`; output to `sft/logs/alignment_gate-<date>.log` and the summary
   table into this note; the 2b double-backward check as a second entry
   point. Local 2.7B fit at 128 tokens, batch 1 is the first thing it reports; `--fp32-forward` for the box's step measurements (model weights cast to fp32 for the gate only, 11 GB).
6. **Dreamer trainer** (new, `sft/experiments/lama_ckl/dreamer.py`,
   target `make lama-ckl-dreamer ARGS=...`, flags `--corpus --init-adapter
   --rounds --output` (at each round start after round 0 the dreamer imports the gate module and runs its instrument steps on the starting checkpoint, writing `round-NN/gate.json` with α, c, h, κ, the noise floors and the retained pool; round 0's instrument **is** the box gate's `gate.json`, passed as `--gate <path>`, never re-measured; the common cell α and c of §3.4 are taken by the experimenter over all checkpoints' `gate.json`),
   `--pool-k 4 --dreams-n 10 --rows 50` as pins, restart by rerunning the
   same command on an existing `--output` (completed rows and rounds are
   skipped); `--output` is `.cache/lama_ckl/dreamer/<arm>/`, inside the
   tree `LAMBDA_CACHE_ARTIFACTS` already pulls; output `<output>/round-NN/{rows/<row>.json (wake transcript,
   pool, scalars, kept batch, readings), readings.json, gate.json}` and `<output>/final/gate.json` for the last checkpoint plus the round
   checkpoint written directly to
   `models/mamba2_2_7b/checkpoints/dreamer-a-r<NN>/final` (the tree
   `lambda_pull.sh` already mirrors, so it comes home with the session and
   goes back up with `LAMBDA_RESUME_CHECKPOINT` pointed at it); round r+1's
   row 4 reads `round-NN/rows/`; unprotected
   until it writes a loss the cell depends on): §3.2 arm (a) after
   sign-off, with the round readings of step 5 and the §6.1 census over
   the pool. The gate script gets `make lama-ckl-gate ARGS=...` with
   `--tokens 128|512` for the local and box modes. Arm (b) as a second mode behind gate 2b, **after its own mechanism spec block** (ledger, loss with the projection form of §4.17, the four g_old hinges of §3.2 (§4.3, decided 10-26), KL-guard coefficient, lr, rounds, the manual-mixer requirement for the double backward) is written and signed off; research note §6 is a derivation, not that block. `dreamer.py` and `alignment_gate.py` set the cells' starting checkpoint, α and c, and `master_weights.py` will carry the SGD step every dream cell takes, so all three are proposed for `PROTECTED_PATHS` from their first commit (teammate's call; until then they are unprotected).
7. **Protected edits** (the teammate commits; brief quotes §3.2 and §3.4
   verbatim; the PROTECTED EDIT report maps each ledger row to
   `file:line`). **7a, first, before the gate script and the dreamer
   trainer** (§3.5 item 1): `dream_distillation.py` — the
   `--dream-loss-weights` flag in `turn_weights` (row 5 of §3.2),
   `distill_dream_batch` split into a per-dream gradient function and the
   step with the clip, and `protocol.py` `WAKE_FRAMES` with its users (`protocol.py:268` builds the frame tokens, `:363` and `runner.py:46,772` record it) taking the drawn frame. Test: the per-dream function's gradients, averaged over the
   scored dreams, equal the batched function's gradient on a padded batch to a tolerance measured by running the batched call under two different paddings (repeats in one process can be bit-identical, §3.5 item 2 (ii); padding changes kernel shapes and gives the rounding spread the test should tolerate); the cell keeps the batched path, the dreamer and the gate use the per-dream one. **7b, before the first cell:** runner `--wake-frames` (§3.4), `--cycles-per-run`, `--eval-every N` with the `(cycle + 1) % N`
   rule, `--dream-temperature`, `--dream-loss-weights`, `--fp32-forward`, `--optimizer` / `--lr` for the dream arm,
   the gradient clip with its per-cycle log, per-cycle resume; all
   recorded in `run.json`. Tests: an eval-every run writes `result.json`
   only on the rule's cycles and never twice for the last; the adapter and
   optimizer state are written every cycle and a killed run resumes at the
   next cycle; `run.json` carries the temperatures, optimizer, lr, clip, frame mode and cycles-per-run; `--wake-frames set` draws a recorded frame per wake; `uniform` weights cover the same positions as `turn`; the clip acts on the gradient before the step; `--optimizer sgd` without `--lr`/`--clip-norm` is refused, as is `--clip-norm` under adamw; a `--cycles-per-run` run writes `summary.json` and resumes.
   α and c come from the gate (§3.5 item 2), so the runner edits land
   after the gate has run on v4 and before the first cell; each dreamer
   checkpoint's own α and c are re-measured and passed as flags.
7c. **Data artifacts**: the defaults in `scripts/lambda_data_artifacts.sh` are not changed; the `.env` line of item 8 is the one mechanism by which the corpus files reach the box.
8. **Archive the fifth box's dream cell** (a move, not a copy):
   `runs/dream-seed-42`, `runs/*-smoke` and `acceptance/` to
   `.cache/lama_ckl_archive/fifth-box/`; keep
   `runs/{lora,mix-review}-seed-42` and `runs/frozen-seed-42-r{1,2,3}` as
   the v4 baselines; `scripts/.env` gains `LAMBDA_DATA_ARTIFACTS="dreamer_*.json* retained_pool-*.json dolly_context.jsonl"` (today's `"none-*"` disables the defaults, `lambda_data_artifacts.sh:18`, so without this no corpus file reaches the box; `dolly_context.jsonl` is regenerated locally first, §3.3).
8b. **`report.py`, the partial epoch and the Makefile** (unprotected except the runner's epoch result): the runner writes `summary.json` with `cycles_run` at the `--cycles-per-run` stop without marking the epoch complete (§3.4; `runner.py:357-378`, protected, in 7b); the partial curve is epoch 0's `result.json` followed by every `cycle-NNN/result.json` the eval-every rule wrote, keyed by cycle (`runner.py:236-238` writes them), and `report.py` accepts it when `cycles_run` is present and distinguishes runs by output name, not seed alone (`report.py:94-95` refuses same-seed runs under one root today); `lama-ckl-frozen-x3` takes a `RUN_PREFIX`; the v4 cell's output is `runs/dream-ultra20k-v4-seed-42` (the `dream-<ckpt name>-seed-42` template); the shown-row comparison (§3.4) is a `report.py` column. `dream_census.py --warm-start` is passed the cell's own checkpoint's `trainable.pt` (its default is v4, `dream_census.py:99`). The dreamer checkpoint directory carries `lora_config.json` beside `trainable.pt`, as `runner.py:795` expects; `<arm>` is `a` or `b` and `<round>` is `r<NN>`.
8c. **Adversarial-read brief** (proposed to the teammate, not done): `scripts/.env` holds a live API key; the skill brief names it as a file the reader never opens.
9. **Experimenter skill:** the session-order table gains the dreamer
   session (§3.5 item 3), the per-checkpoint split run, and the one-session, three-cell,
   100-cycle plan (§3.4) with the baselines one at a time beside the
   cells and the pin edited on the box branch, the
   v4 arm's kill-on-collapse cell, the per-cycle resume and the whole-tree adapter pull; the acceptance row's exposure reading
   is the trigram revisit count, not `subjects_mentioned`.
10. **Reading path:** `notes/README.md` names this note first; this note's
    and the research note's index entries; `RESEARCH-20260805`'s SEAL line
    points at `RESEARCH-20261024-learned-dreamers` (done 10-24).
11. **Deferred from the run note:** the coupled wake/dream seeds (same CPU
    generator seed for wake turn i and dream i; harmless, worth separating
    when the runner is next edited); the GPU-nondeterministic wake (§1.5;
    per-cycle readings are one draw, no action).
