# Discussion notes — 2026-08-11: multi-sleep with adaptive wakes and provider-neutral agents

Team debrief (altrup + Codex) of the 2026-08-10 GH200 session
(`EXPERIMENT_NOTES-20260810-231500.md`, banked in `96e5f48`). This note
supersedes the B-family and next-run direction in
`DISCUSSION-20260808-headline-collapse-deep-block-and-regime-bridge.md`.
It also registers the repository changes required before the next paid run.

The standing objective is unchanged: compare installation against forgetting.
Raw installation, exact match, or training speed alone does not rank an arm.

## 1. What the 2.7B run established

The clean matched-token-gradient block used the recap-0.5 warm start at
`models/mamba2_2_7b/checkpoints/epoch-2/step-800`, with adapter SHA-256
`226e95765f9e2c0a9fa335d5f70af8fb1d63bbf0f30c4427097b116375a11f3c`.
It compared one-pass-per-dream A against B4 at the same token-gradient budget.

| arm | margin change | floor-corrected margin | installs | paraphrase | dPPL |
|---|---:|---:|---:|---:|---:|
| no-sleep | +1.79 | — | 3/4 | 0.00 | +0.000 |
| A, 300 distinct dreams | +10.59 | +8.80 | 4/4 | 0.25 | -0.173 |
| B4-raw | +2.02 | +0.23 | 4/4 raw | 0.00 | +0.040 |

B4-deflated, B4-qcm, and B4-sigma also remained at about 0.00–0.23
floor-corrected margin. The extra erase loss did not buy learning. The sigma
variant caused the most loss and the worst dPPL. At matched token gradients,
A learned about 38 times more floor-corrected margin than B4-raw, with no
damage disadvantage in this run.

This fires the registered B4 failure condition. It also closes the current
erase-on-read arm program. B1, B2, B3, and B4 do not enter multi-sleep. B3
remains useful only as a harness equivalence check. The local erase geometry
tools remain diagnostics, not arm selectors.

The positive result is narrower than a final claim. It is one sleep and one
training seed. A's near-zero measured damage and cumulative retention still
need confirmation across sleeps and seeds.

## 2. Scientific decision: the next experiment

Run a six-wake, three-seed comparison of these three arms:

1. **A (`replay`)** — context distillation from many distinct, uncued,
   self-generated dreams.
2. **No-sleep** — no dream generation and no weight training.
3. **Sequential SFT (`sft-ref`)** — one pass of next-token cross-entropy on
   the complete realized raw wake transcript after each wake.

This design answers two separate questions. No-sleep measures the recurrent
state and probe floor without weight updates. Sequential SFT controls for
ordinary training on the same experience. A tests whether self-generated
replay consolidates the experience with less cumulative damage.

Do not add a general-corpus rehearsal arm to the primary experiment. Add it
only as a registered rescue if all trained arms show unacceptable cumulative
damage. Do not reopen B, cue splicing, the recap-0.8 warm start, or repeated
training over one dream.

### 2.1 Experimental unit and fork

Each seed starts from the exact pinned warm-start weights. Wake 1 is generated
once. The local experimental model responds to every user turn. Save:

- the full realized user/assistant transcript;
- the exact recurrent state after the final assistant response;
- the scenario, fact, injection, model, sampling, and generator metadata.

Fork all three arms from these same weights and this exact post-wake recurrent
state. Sleep 1 is the first treatment. Wakes 2–6 are arm-specific because the
arms have different weights, states, and replies after Sleep 1.

Later wakes share the scenario specification, four target facts, injection
turn schedule, user-generator provider/model, and generation settings. They
do not share literal user messages. Each user simulator adapts to the replies
from its own arm. This preserves matched intent without creating incoherent
dialogue.

### 2.2 Wake construction

Use a held-out real dialogue as a deterministic scenario, persona, and intent
spine. An external user simulator realizes the user turns against the local
model's live replies. Do not assemble a wake from shuffled isolated question
and answer pairs. Do not author or force-feed assistant messages.

Each wake introduces four new entity-to-five-digit-code facts. The user
simulator must communicate each fact naturally at its registered injection
turn. The local model is allowed to respond. Non-injection turns continue the
scenario and may repair conversational confusion, but they must not introduce
future target facts. Do not put `<|endofconversation|>` inside one wake; the
sleep boundary ends the conversation.

The exact wake length and injection positions were not fixed in this debrief.
Freeze them before implementation approval and before any outcome is seen.
The current proposed default is 12 user turns with injections at turns 3, 6,
9, and 12. Counterbalance fact-to-position assignments across seeds and wakes.
Log each fact's token distance from the wake end so position effects remain
visible.

Six wakes give 24 facts and 15 backward-retention comparisons. Four wakes are
acceptable only as a smoke test; they are too small for the primary run.

### 2.3 Live user generation

Give each arm and wake a separate user-generator session. Wake 1 has one
shared session. Within a wake, resume that session and send only:

- the latest local assistant reply, quoted as untrusted transcript data; and
- the next registered turn goal, including a fact only at an injection turn.

The generator returns only the next user message. The harness stores the full
transcript locally. Provider resume state supplies the previous generator
context and allows provider-side prompt caching. A resume failure stops the
cell. It must never silently start a new session.

The adapter boundary is a command that accepts one normalized JSON request and
returns one normalized JSON result. The provider adapter translates this
request into the provider's native CLI, API, or local-model format. Claude CLI
uses its session identifier; Codex CLI uses its thread identifier. The common
schema is internal to this repository. Neither CLI is expected to accept the
schema directly.

Store provider, model, command version, session/thread identifier, request
hashes, response hashes, token usage when available, and resume status. Treat
the generator as part of the experimental data path. Pin it across comparable
arms and seeds.

### 2.4 Sleep treatments

**A:** Generate 300 distinct dreams after each wake from A's intact post-wake
state. Use no cue splice, steer prefix, or fact-aware selection. Generate the
complete set before training. Freeze and cache teacher logits and all dream
metadata. Train one pass over each dream. Student weights carry across dreams;
student recurrent state starts fresh for each dream. The teacher is the A
model at the start of that sleep.

N=300 is an intention-to-treat volume, not an adaptive coverage target. Do not
regenerate or select dreams after inspecting target-fact coverage. Record the
number of correctly bound rehearsals per fact. Report the primary result over
all facts and a secondary retention analysis by observed rehearsal count.

**No-sleep:** Perform no generation and no training. Carry the intact recurrent
wake state into the next wake. Run all evaluation probes from fresh state so
they measure weight retention and provide the floor for the trained arms.

**Sequential SFT:** Train once on every token in that wake's complete realized
raw transcript. Start training from fresh recurrent state and carry/detach the
training state only across transcript chunks. Do not train on dreams. The next
wake begins with the arm semantics already used by `sft-ref`: no carried
training state.

### 2.5 Evaluation and analysis

After every sleep, evaluate every fact introduced so far from fresh state.
Record the full retention matrix `R[evaluation wake, learning wake]`. Report:

- installation after the fact's own sleep;
- backward transfer and retention at every later sleep;
- cumulative installed count and floor-corrected margin;
- greedy exact match as a secondary metric;
- paraphrase retrieval;
- the knowledge battery and held-out perplexity damage measures.

Every learned-fact comparison must subtract the matched no-sleep floor. Keep
the existing installation threshold of at least 1.0 nat floor-corrected
margin. Do not tune it on these results. Report per-seed values and uncertainty;
do not pool away seed failures.

## 3. Expand the unrelated-fact battery

The current battery starts with 40 hand-written candidates and kept 23 items
under the 2.7B warm start. It does not enforce one-token answers. It calibrates
on the warm-start model from fresh state, keeps candidates whose greedy answer
is correct, then records expected-answer log probability before and after
training. A correct-to-incorrect transition counts as a lost fact.

Twenty-three retained items are too weak for a multi-sleep damage claim. With
zero observed losses, the approximate 95% upper bound on the true loss rate is
13% at n=23. It is about 3% at n=100.

Before the run:

- expand the static candidate bank to 200 verified items across geography,
  science, history, language, arithmetic, everyday knowledge, and culture;
- use short answers of one to three tokens under the experiment tokenizer;
- require at least 100 self-calibrated retained items or stop;
- reject collisions with wake scenarios, target entities, codes, and prompts;
- build one immutable calibration artifact keyed by warm-start checkpoint SHA
  and candidate-bank hash, then share it across all arms and seeds.

After every sleep, report retained/lost counts, loss rate, mean and median
answer-log-probability change, the lower-tail change, and held-out perplexity.
The lower-tail summary must be fixed before launch; use the 10th percentile
unless implementation review finds a numerical reason not to.

## 4. Batching is an experimental execution parameter

All independent GPU work on the box must expose and log a batch-size setting.
This includes dream generation, teacher-logit collection, fact probes,
paraphrase probes, battery calibration/scoring, and other independent scoring
sets. The 2026-08-10 box found a generation throughput knee near eight
concurrent generators; ten was worse. This is a hardware observation, not a
universal constant.

The experimenter must run a short hardware smoke before paid science, choose
batch sizes from throughput and VRAM only, then freeze them across comparable
arms. Log batch size, concurrent workers, throughput, peak VRAM, retry count,
and stochastic batch topology. If a batch fails for resource reasons, reduce
the batch without changing data, thresholds, or arm semantics.

Do not batch operations whose order carries meaning: turns within one live
conversation, sleeps within one arm, optimizer steps, or student dream
training. Independent arm conversations may run concurrently. Stochastic
generation can change with batching, so cache and hash every realized dream
and wake artifact. Add a deterministic batch-equivalence check for probes.

## 5. Provider-neutral repository migration

The repository must support Claude Code and Codex without maintaining two
independent experiment procedures.

Current state: only the Claude experimenter command exists. There is no
`.agents/skills/altrux-experimenter/SKILL.md`, so a clean Codex experimenter
invocation cannot load the workflow. This is an expected migration gap, not a
valid reason to guess at the procedure.

Use one canonical, provider-neutral source for each workflow and thin wrappers
for the two clients:

- Claude commands remain available as `.claude/commands/altrux-debrief.md`
  and `.claude/commands/altrux-experimenter.md`;
- Codex repo skills live at `.agents/skills/altrux-debrief/SKILL.md` and
  `.agents/skills/altrux-experimenter/SKILL.md`;
- each wrapper loads the same canonical workflow text and adds only the
  client-specific invocation details.

Do not duplicate the scientific rules in both wrappers. Choose the canonical
file location during implementation, then test both entry points from a clean
repository session.

Add `EXPERIMENTER_AGENT=claude|codex` to `scripts/.env`. Update the remote
launcher and setup scripts so this value selects the installed CLI, uploaded
configuration/authentication, initial experimenter prompt, and provider-neutral
tmux/window labels. Keep the remote tmux session named `experimenter`.

Treat Claude and Codex credentials as passwords. Upload only the selected
provider's required files and never log their contents. Claude credentials
already exist on the current rented box; Codex credentials will be available
after migration.

The remote experimenter provider and the adaptive-wake user-generator provider
are separate settings. Changing `EXPERIMENTER_AGENT` must not change wake data.
Give the wake harness its own explicit generator command/provider/model
configuration.

## 6. Required local work before the next box run

Work locally, test, commit by sub-feature, and push only after explicit user
approval. The implementation docket is:

1. Expand and test the knowledge battery and immutable calibration artifact.
2. Add batched calibration/probes/generation with deterministic equivalence
   tests where applicable and complete execution metadata.
3. Replace authored wake assembly with the live adaptive-wake harness, its
   resumable command adapter, artifact schema, and failure rules.
4. Update multi-sleep to the three registered arms, six wakes, four new facts
   per wake, three seeds, full retention matrix, and no-sleep corrections.
5. Migrate the debrief/experimenter workflows and remote launcher to the
   provider-neutral design in §5.
6. Fix `summarize_grid.floor_deltas`: it currently looks for the exact arm
   name `nosleep` and misses run names such as `g6_nosleep`.
7. Run a local end-to-end fake-backbone smoke, then a short real-hardware
   throughput/VRAM smoke. Freeze the run manifest only after both pass.

The implementation must not begin until the user gives a separate explicit
go-ahead. Before that approval, resolve the one open scientific parameter in
§2.2: exact wake length and injection positions.

## 7. Stop conditions and interpretation

Stop before the science block if the battery keeps fewer than 100 items, the
warm-start SHA differs, Wake 1 cannot be reproduced exactly across arms, a
provider resume silently resets, deterministic batched probes disagree, or
the hardware smoke cannot find one stable batch configuration for comparable
cells.

Do not stop because one A dream omits a fact. That omission is part of the
fixed-volume treatment. Stop and inspect only for malformed generation,
incorrect artifact identity, non-finite training, or a violated registered
invariant.

The primary evidence is the three-arm retention matrix across six sleeps and
three seeds. A wins the program's present claim only if it installs above the
no-sleep floor and retains earlier facts with less battery/PPL damage than
sequential SFT. If A and SFT retain similarly, the dream mechanism has no
demonstrated advantage. If no-sleep retains most facts without weight change,
the wakes or probes do not isolate consolidation strongly enough. If all
trained arms accumulate damage, register a general-rehearsal rescue rather
than changing the primary run after seeing its results.

## 8. Cold experimenter dry run

A clean Terra agent received only the repository and the instruction to load
the experimenter workflow and narrate a dry run. It made no changes. It found
the expected unresolved parameter: wake length and injection positions. It
correctly refused to promote the proposed 12-turn, 3/6/9/12 schedule to a
decision.

The repository experimenter skill, Claude/Codex launcher, battery and batching
work, resumable adaptive-wake runtime, three-arm coordinator, floor correction,
and 18-cell fake-backbone smoke are implemented through `3504363`. The cold
agent recovered the correct next science block: replay versus no-sleep versus
sequential SFT, six wakes, four new facts per wake, and three seeds. The
remaining launch gates are the frozen manifest and real-hardware
throughput/VRAM smoke.
