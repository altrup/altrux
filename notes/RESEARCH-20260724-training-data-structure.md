# Research note — 2026-07-24: what our training data actually is, and what comparable memory work trains on

Not a run log, not standing direction — findings only. A structural audit of
our own training data (with decoded evidence) plus a literature scan of what
comparable memory work trains on, prompted by the debrief question *"what
should we change about the training data to get the 780M memory working?"*

Read alongside `RESEARCH-20260722-memory-consolidation-landscape.md` (the
consolidation/end-goal framing this note serves). This note supersedes
20260722's note that its `[S]` items were unread — those papers have now been
read in full. What to *do* about any of this belongs in the accompanying
DISCUSSION note, not here.

**Confidence markers:** `[M]` = measured locally this session, method in the
text; `[V]` = verified against the paper's own text by a subagent that read the
PDF/HTML in full; `[NOT IN PAPER]` = searched for and absent, not merely
un-found by us.

## 1. What our data actually is [M]

Measured on `data/train.pt` + `data/train_memory*.pt`, and on a 31-chain
regeneration from a 360-episode subset using the exact 780M-screen recipe
(`--cross-sleep-bias 0.75 --seed 7 --split-episode-rate 0.15 --split-qa-rate 0.9
--split-gap-min 1 --split-gap-max 4`).

| Property | Value |
|---|---|
| ultrachat (`train.pt`) | 18,187 eps, 14.4M tok, median 830, **capped at 1024** |
| LongAlign | 9,846 eps, median 16,161, max 99,314, **100% exactly two turns** |
| babilong | 2,000 eps, median 3,865, **100% exactly two turns** |
| Document share of training tokens | **94%** (214.9M of 229.3M) |
| Malformed role transitions in generated chains | **9.0%** (4 `[USER]→[USER]`, 4 `[ASSISTANT]→[ASSISTANT]` of 89) |
| Of those, with no sleep between | 2 |
| Max concurrent suspended episodes | 2 |
| Retention-bearing tokens | ~1% of corpus, against ~99% local-context loss |
| Mid-conversation sleeps | **0**, in every dataset trained on to date |

### 1a. Long implies single-turn; multi-turn implies short

Every LongAlign and babilong episode is one large user turn plus one answer —
9,846/9,846 and 2,000/2,000 respectively have exactly two role markers. The
only multi-turn data in the pool is ultrachat, capped at 1024 tokens. **There
is no long multi-turn conversational data anywhere in the corpus.**

The `Makefile`'s `data-memory` comment describes LongAlign-10k as "real long
multi-turn conversations." As tokenized here it is not.

### 1b. The episode splices are malformed, and it is the majority case

`prepare_chains.py:136` collects cut points at *any* role marker. For a
two-turn episode (`is_qa = len(b) == 2`) the only candidate inside the middle
third is the assistant marker, so the cut lands immediately before
`[ASSISTANT]`: the head is a dangling user turn, the tail a bare answer. The
realized token stream:

```
[USER] document + question\n                                    <- head, unanswered
[USER] unrelated question\n[ASSISTANT] unrelated answer<EOS>     <- intervening episode
[ASSISTANT] answer to the document<EOS>                          <- tail, unaddressed
```

Two structures that never occur in well-formed conversation: `[USER]` directly
followed by `[USER]`, and `<EOS>` directly followed by `[ASSISTANT]` with no
user turn addressing it.

Decoded instance from the regeneration: at offset 23,251 a Chinese question
about 金鸡独立 is cut off **with no sleep**; an unrelated English Punch/London
Charivari document begins immediately; 32,730 tokens later, after an unrelated
English assistant answer closes with `<EOS>`, the Chinese answer appears with
nothing addressing it.

At full screen scale this is 5,498 splits, of which **1,580 were multi-turn
conversations** (5,498 total less 3,918 single-QA). Cutting a dialogue is
structurally worse than cutting a document: if the preceding assistant turn
ended in a question, the tail opens with a user turn answering something
invisible — and that case is not detectable syntactically, so no boundary rule
fixes it.

Both malformations come from the same cut. Cutting at a `[USER]` marker yields
two well-formed halves; cutting at an `[ASSISTANT]` marker produces both the
dangling head and the unaddressed tail.

### 1c. Role markers are encoded inconsistently across corpora

`[USER]` as a registered special token is id 50277 (one token). In the local
`data/train.pt` it is `[60, 23131, 62]` = `[`, `USER`, `]` — three ordinary BPE
tokens. All 18,187 local ultrachat episodes have zero special markers; all
11,846 `train_memory` episodes have them.

The local file predates special-token registration. The box's 0723/0724 regens
went through `build_tokenizer`, so they did produce single-token markers — this
particular file is not what trained the screen. Two consequences remain: no
local validation can use `train.pt` as-is, and the local reproduction above was
consequently *milder* than the real data, since with no markers found in
ultrachat `prepare_chains` could only split documents (22/22 single-QA locally,
versus 71% at full scale).

Separately, `models/common.py:74`'s `extend_embeddings` initializes new
embedding rows *"from the existing embedding's std"* — i.e. randomly — and
`TARGET_LORA_MODULES = ["in_proj", "out_proj"]` excludes embeddings, while
`tie_embeddings: true` makes `lm_head` share those same rows. So the marker
tokens are read as random frozen vectors, and their output rows are also random
and frozen, meaning the model cannot learn to emit them at all. The three-token
BPE spelling carries better representations than the single special token does.

### 1d. Counts cannot catch a structural bug

Every count in every regen log was correct and reproducible while the splices
were malformed: `2753 chains, 230.9M tokens, 15857 sleeps, 5498 split-tails
(3918 = 71% single-QA) — exact match to EXPERIMENT_NOTES-20260723`. Counts
confirm the generator did what it was told, never that what it was told was
right. The bug was only visible by decoding tokens at a join.

## 2. What the literature trains on [V]

Read in full this session; every claim below is from the paper's own text, with
absences marked.

### 2a. Titans (2501.00663) — our architectural basis — contains no data recipe

FineWeb-Edu, 15B tokens (170M/340M/400M) and 30B (760M), Llama-2 tokenizer,
**training sequence length 4K**, batch 0.5M tokens, AdamW lr 4e-4 cosine.
Training procedure inherited wholesale from Gated DeltaNet (2412.06464).

- **[NOT IN PAPER]** How documents are packed into the 4K sequences. No
  separator, no boundary token, and no statement about whether memory state is
  reset or carried between training sequences. Gated DeltaNet is **equally
  silent** — its text gives only sequence length and batch size.
- Titans constructs **no** data in which anything must survive a state reset.
  S-NIAH is eval-only (2K/4K/8K/16K). BABILong is delegated entirely: *"We
  follow the original experimental setup and training process in the
  benchmark."*
- MAC/MAG/MAL differ **architecturally only** — same corpora, same token
  budgets. Only MAC is fine-tuned for BABILong, an experimental-scope choice
  rather than a described data difference.
- **The "2M context window" claim appears only in the abstract, introduction,
  and conclusion, attached to no experiment.** The longest numerically stated
  evaluation in v1 is 16K. v1 is also the only version.
- No official code release located, so the packing/reset question is only
  answerable from `fla-org/flash-linear-attention` + `flame` trainer configs.

### 2b. Do Language Models Need Sleep? (2605.26099) — the closest prior art

CMU/UMD. Three procedurally-generated training sets (Rule 110 cellular
automaton, Depo multi-hop graph retrieval, GSM-Infinite); no natural-language
pretraining corpus. Pretrained *initializations* reused for GSM-Infinite
(Jet-Nemotron 2B, Ouro 1.4B). This paper does what our chains do.

Where it matches our design:

- **Concatenates unrelated episodes deliberately** — *"we train the model on
  four independent length-24 binary strings… The four states are unrelated to
  each other (i.e., they are not obtained by unrolling the previous state)."*
  Four unrelated sub-episodes in one T=100 sequence, one per eviction window.
- **No boundary token.** Delimitation is purely structural — split into
  non-overlapping chunks of length ≤ L. The `|` in their figures is
  typographic: *"The hard eviction boundary is denoted by |"*, and the automaton
  vocabulary contains only `0` and `1`.
- **KV cache cleared at every boundary; SSM fast weights NOT reset** —
  zero-initialized once per sequence and carried across all chunks. *That
  carry-over is the memory mechanism.* Same substrate split as ours (backbone
  wiped, M persists), and they reset at **every** boundary, with no cadence
  sampling.

Where it differs:

- **A hard 0/1 loss mask.** Consolidation-phase chunks receive N recurrent
  passes and **zero gradient**; only the prediction-phase chunk is supervised
  with masked cross-entropy. *"By construction every supervised token is
  answerable only from post-eviction fast weights."* Loss-bearing tokens: **4%**
  (automaton, stated), ~2.8% (Depo, inferred), **<1%** (GSM-Infinite, inferred)
  — but that is the *entire* objective. Our ~1% retention signal competes
  against ~99% local-context signal; theirs competes against nothing.
- **Question placed before the context** in GSM-Infinite, explicitly *"so the
  model can consolidate selectively."* CoT traces excluded from the loss.
  `prepare_babilong.py:52` does the reverse (`{input}\n{question}`).
- **N recurrent consolidation passes** = "longer sleep", with monotonic gains.
  Largest single result is sliding-window eviction at L=512: **0.596 → 0.905**
  (Ouro 1.4B). We perform exactly one pass and have no parameter for this.
- A prediction-phase latency constraint: one forward pass per answer token, no
  CoT, no looping at prediction time.

Calibration: accuracy evaluations top out at T ≈ 2,000–3,300 tokens; the only
12K figure is a throughput benchmark.

### 2c. Language Models Need Sleep (2606.03979) — a different paper, same title

Google Research/Cornell, on the Hope/Nested-Learning architecture. Name
collision is real: 2605.26099's v3 renamed itself to avoid confusion with this
one.

- Post-training/continual-learning only; no pretraining corpus, no token
  counts, no stated training sequence length. Backbones Llama-3B/3-8B,
  Qwen3-1.7B/8B, Llama-3.2-1B.
- **[NOT IN PAPER]** Example delimitation, boundary tokens, packing, or state
  reset at data boundaries. Every reset it describes is architectural and on a
  fixed step-count schedule (consolidation period, chunk lengths), decoupled
  from where an example ends.
- **[NOT IN PAPER]** Any loss weighting or masking for cross-boundary
  retention. Its only masking is inside Learning-to-Imitate — a prefix/
  continuation split *within one generated sample*.
- One unmarked concatenation is stated: the SQuAD continued-pretraining setting
  aggregates 200 unrelated passages × 5 "dreams" into one training set, with no
  described transition marking or state reset.
- Consolidation explicitly does **not** replay raw data: *"the memory
  consolidation step should not simply replay raw data; instead, it needs to
  explore and extract abstractions of knowledge acquired during active (waking)
  steps."* It trains on self-generated samples from the model's pre-update
  state, scored by SFT-gradient importance (top-k plus random samples for
  diversity), into an isolated LoRA expert, optimized with ReST^EM.
- BABILong to 10M tokens, but *"All small models are fine-tuned using the
  official BABILong training protocol (Kuratov et al. 2024)"*, and both this and
  the Nested Learning paper note small models degrade sharply without that
  fine-tuning. The recipe is upstream of the headline.

Nested Learning (2512.24695, the Hope paper): ~50B tokens FineWeb-Edu plus
unnamed "long-context documents", 32K vocab; also reported as 760M/30B and
1.3B/100B. **[NOT IN PAPER]** document delimitation or per-example state reset;
its resets are fixed-period and data-agnostic. Its CTNL task teaches Manchu then
Kalamang sequentially with no separator and no reset, treating survival of that
transition as the test — but those are transitions between *complete* units,
which is a different thing from an unaddressed dangling turn.

### 2d. BABILong (2406.10149) — the one concrete cross-boundary recipe

bAbI task sentences hidden among PG19 background: *"we 'hide' the sentences of
the original task between the sentences of irrelevant text that is drawn from
another closely related distribution. Examples are constructed by gradually
adding new sentences from the background dataset in their natural order until
the augmented sample reaches the desired length."* Splits to 10M tokens,
evaluated to 50M.

Its RMT/ARMT baselines (Appendix C) are the training recipe every strong
long-context number in §2b–2c rests on:

- Segment size **512 tokens**; 16 memory tokens (RMT), 10 (ARMT); GPT-2 137M
  backbone.
- Memory carries across segments within a sample and **resets between samples**
  — the same design as our `reset_slot`.
- **Curriculum over segment count**: 1, 2, 4, 6, 8, 16, 32 (ARMT
  2-3-5-8-16-32), and *"For each curriculum step we chose the number of
  segments randomly from 1 to N for every batch to prevent overfitting to a
  certain context size."*
- Batch 64, AdamW, lr 5e-5/3e-5, 1000 warmup, ≤10k steps per stage, no gradient
  stopping, three seeds with different memory initializations.

We sample chain budget log-uniform 30k–130k from step zero, with no curriculum.

### 2e. MemTrain (2606.03197) — the cleanest scalable construction

Self-supervised recall-forcing data from unlabeled Wikipedia, no human
annotation. Pick a passage, mix it with N−n₁−1 random passages, mask a target
entity with `[MASK]`, then *"segment the long document into fixed-length chunks
{c₁...c_T}, where each chunk corresponds to an interaction step"*. The model
processes chunks sequentially and must infer the masked entity by long-range
aggregation, so it *"cannot simply copy the answer from the document"*. A second
task, **Intermediate Memory Recall**, requires reconstructing masked entities
from an *earlier* chunk c_l (l < k) from memory alone. RL-optimized with group
advantages.

### 2f. Supersede (2606.27472) — templated correction data is saturated

Closest published match to multi-session training with enforced state
boundaries. *"The agent processes one session at a time and maintains a bounded
memory (a notes field capped at B characters); crucially, raw sessions are never
re-fed. After the final session, the agent answers the query using its memory
alone."* GRPO on Qwen2.5-3B; held-out supersession accuracy 9.0% → 16.7%. Data:
LongMemEval knowledge-update subset (78 questions; `s` split ~48 sessions /
~122k tokens).

The negative result is the load-bearing one: *"We first built a procedural
generator of templated supersession timelines (explicit 'X now Y' updates). With
the history in context, frontier models score 100% across configurations: they
resolve clean, explicit updates by a last-mention scan. Synthetic templated
supersession is therefore saturated and cannot surface the failure… We
accordingly use real conversational data throughout, where updates are implicit
and paraphrased."* They still find the generator useful as a training curriculum
that transfers to real data.

`probe_correction.py`'s own phrasing — *"No, that is outdated. X won the most
recent election"* — is an instance of the saturated form.

### 2g. Beyond Perplexity (2607.00368) — our correction probe is bridge-tier

CMU/MBZUAI. Separates **proxy** (support reconstruction, local ΔNLL) →
**bridge** (answer ΔNLL) → **target behavior** (generation after the support
context is removed). Of the middle tier: *"tests whether probability mass moves
toward the target answer, but not whether the model will produce that answer in
open-ended use."*

- Headline: one-step LoRA improves support NLL **and** answer NLL at all three
  scales (Qwen3 1.7B/4B/8B) while **greedy free-form recall stays at 0.0%** for
  direct, paraphrased, and delayed prompting. A 16-step rank-8 stress point
  reached 35/48 direct recall while locality collapsed from **141/144 to
  14/144**.
- Its correction/overwrite probe is nearly identical in construction to
  `probe_correction.py`, with two elements ours lacks: **four-way scoring**
  (corrected-only / stale-only / both / neither — one-step LoRA returns
  *neither* in **72/72** conflicts, while replacement memory is corrected-only
  in 68/72) and a **locality control** of 144 unrelated prompts. Probe items are
  nonce access-code facts, 48 prompts per type, 24 conflicts.
- Audit of 24 coded papers screened from >40, through April 2026: TTT-family
  work reaches stream-level, PERK/SEAL/MEMORYLLM reach bridge-level,
  **nothing reaches the deployment-behavior tier.**
- Recommends reset-vs-stream controls as a methodological requirement, and
  builds no dataset — multi-session testing is listed as a deployment-grade
  requirement and explicitly not implemented.

### 2h. Multi-Session Chat (2107.07567) — real session structure, narrow content

*"We collect two lengths of training conversation: 4000 episodes with 3
sessions, and 1001 episodes with 4 sessions"*; validation and test extend to 5
sessions. Session 1 reuses PersonaChat. Prior-session content reaches the model
only as truncated dialogue context (compared at 128/512/1024 tokens) or as a
**gold summary** (extended personas).

Characterization relevant to us:

- Short open-domain chitchat; nothing near the 30k–130k token regime.
- What must be remembered is thin persona facts ("I'm a nurse", "I have two
  dogs"), mattering at a handful of token positions — so MSC carries the **same
  dilution problem** as our data under an unmasked objective.
- **Its own baselines do not train parametric memory** — the paper's mechanism
  is truncated context or a gold summary. It measures whether a model can use a
  summary, not whether a memory can hold something.
- 2021 crowdworker data, distributionally distant from serving.
- What it uniquely contains: genuine, human-written, *implicit* multi-session
  structure, which per §2f cannot be synthesized without saturating.

### 2i. SCM (2604.20943) — no training at all

*"It requires no training or fine-tuning; all components use existing
pretrained models or algorithmic logic."* ~3,000 lines of Python, Llama 3.2 via
Ollama for concept extraction, MiniLM embeddings, NetworkX graph, SQLite. Eval
is a hand-built 8-test harness (22 facts, 55 concepts, 360 concepts for
latency), all tests scoring 1.00 with zero variance, which the paper attributes
to a deterministic pipeline and calls a lower bound. No corpus, no loss
function, no context-length axis, code not released. No data recipe exists here.

### 2j. SuRe (2511.22367) — surprise selects what to replay, not when to read

UCL + Huawei Noah's Ark. Classification-benchmark continual learning (AG News,
Amazon, DBpedia, Yahoo; plus a large-task suite), T5-Large and Llama 3.1 8B,
LoRA r=8 on Q/V, one epoch, replay every other step.

- **Whole examples, never packed.** SuRe puts no two examples in one training
  sequence; current-task and replayed examples co-occur in a mini-batch, not in
  a token stream. There is no in-sequence boundary to mark and no recurrent
  state to reset.
- **Surprise = mean per-token NLL** under the model (averaged, not summed —
  ablated: 74.22 vs 72.84). Selection is **deterministic top-K per task**; once
  stored, replay draws are uniform. Buffer is 2% of dataset size with an equal
  per-task quota, which is what requires known task boundaries.
- **Plain unweighted cross-entropy** over the mixed batch. No masking, no
  per-token weighting. Retention pressure comes from *which examples* are
  replayed and how often, not from the loss shape.
- Recomputing surprise on each replay ("aging") performed *worse*: *"dynamic
  surprise is not necessary in this specific setting."* Scoring on labels only
  collapsed performance.
- Confirms 20260722's reading: surprise's defensible role is choosing what to
  consolidate, not gating a read.

## 3. Open questions in the evidence

1. **No existence proof.** §2g's audit finds nothing at the deployment-behavior
   tier across 24 papers. The gap between bridge-level and behavioral memory is
   unclosed in the literature, not merely unclosed by us.
2. **The transferability of §2b's loss mask is untested for our setting.** It
   was demonstrated training from scratch on procedurally-generated tasks with
   character-level or small vocabularies. Whether it holds when LoRA-tuning a
   pretrained backbone on natural language, withholding gradient from ~99% of
   tokens, is not addressed by any paper read here.
3. **The BABILong "official training protocol"** that §2c's 10M-token result
   depends on may not be identical to §2d's RMT/ARMT baseline configuration.
   Unresolved which one the Hope paper means.
4. **N recurrent consolidation passes** (§2b, §2c) show monotonic gains in both
   papers and have no counterpart in our implementation.
5. **Titans' packing and reset behaviour** is unresolvable from the papers
   (§2a) and would need the `flash-linear-attention`/`flame` configs.
6. **Whether §1b's malformation affected the 780M screen's outcome** is not
   established by anything measured here; §1b reports only the structure of the
   data, not its effect on the result.
