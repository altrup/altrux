# Research — 2026-08-25: desktop multimodal action-state tokenization

## Decision

Model desktop control as a causal sequence of synchronized screen states and
actions:

```text
screen_t -> action_t -> execution_t -> screen_t+1
```

Start with a fixed, normalized action grammar and a pre-trained visual encoder.
Train the policy with one action per observed state and add next-state or
screen-change prediction as an auxiliary objective. Do not train a new action
tokenizer first. Test learned action and screen-effect tokenizers only as
ablations against this simpler baseline.

This is a separate research branch from the current LAMA-CKL continual factual
learning program. It does not change the governing Altrux experiment.

## 1. How existing multimodal systems represent different modalities

Current systems use three main integration patterns. The important difference
is where each modality becomes compatible with the language model.

### 1.1 Project visual features into a language model

[LLaVA](https://arxiv.org/abs/2304.08485) connects a frozen visual encoder to a
language model through a learned projection. The image stays continuous until
its visual features are mapped into the language model embedding space. This
is a simple and effective pattern for image-conditioned generation.

[BLIP-2](https://arxiv.org/abs/2301.12597) instead uses a small querying
transformer between a frozen image encoder and a frozen language model. The
query module compresses image information into a small set of features that
the language model can consume.

This family does not require image pixels to share a tokenizer with text or
actions. It is the lowest-risk starting point for desktop control because a
pre-trained visual encoder already supplies useful screen features.

### 1.2 Interleave continuous visual states with text or action tokens

[Flamingo](https://arxiv.org/abs/2204.14198) adds gated cross-attention layers
that let a language model attend to interleaved visual inputs.
[PaLM-E](https://arxiv.org/abs/2303.03378) injects continuous sensor and visual
embeddings into the same sequence as language embeddings. These systems keep
modality-specific encoders but train the model to use ordered multimodal
context.

This pattern fits desktop histories well: a screen embedding can precede a
small symbolic action, and the resulting screen embedding can follow it. The
action does not need to be forced into the visual representation.

### 1.3 Use one discrete vocabulary for several modalities

[Chameleon](https://arxiv.org/abs/2405.09818) trains an early-fusion model over
mixed text and discrete image tokens. [Unified-IO
2](https://arxiv.org/abs/2312.17172) represents text, images, audio, and actions
as token sequences for one encoder-decoder model.

This is the closest architectural precedent for a joint desktop tokenizer,
but it is not evidence that a new joint vocabulary is necessary. Discrete
visual tokenizers can spend many tokens on changes that do not matter to the
task. They also make reconstruction quality and token allocation new sources
of error.

## 2. How embodied systems represent actions

Embodied agents show that action tokenization is useful when it gives a model a
stable output space, not when it imitates raw input events.

- [Gato](https://arxiv.org/abs/2205.06175) serializes observations and actions
  from many tasks into a common sequence model.
- [RT-1](https://arxiv.org/abs/2212.06817) discretizes robot action dimensions
  into bins. [RT-2](https://arxiv.org/abs/2307.15818) expresses robot actions as
  text-like tokens so a vision-language model can predict them.
- [VIMA](https://arxiv.org/abs/2210.03094) uses object-centric prompts and
  autoregressive robot actions.
- [OpenVLA](https://arxiv.org/abs/2406.09246) maps normalized robot actions
  through the base language-model tokenizer. Its [official
  implementation](https://github.com/openvla/openvla) is a practical example
  of reusing an existing vocabulary instead of training a separate tokenizer.
- [Octo](https://arxiv.org/abs/2405.12213) keeps a general policy backbone while
  adapting action heads to different robot embodiments.

The transferable lesson is to normalize action semantics first. A desktop
agent needs a compact set of intended operations. It does not initially need a
vocabulary for every observed mouse sample or operating-system event.

## 3. How GUI agents close the action-observation loop

Modern GUI systems use an explicit external control loop. The model receives a
screenshot, emits an action, the client executes it, and the next screenshot is
returned to the model.

- [OpenAI computer use](https://developers.openai.com/api/docs/guides/tools-computer-use)
  returns structured UI actions. The client executes them, captures the updated
  screen, and sends it as the next tool result.
- [Anthropic computer use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)
  exposes screenshot, click, cursor movement, drag, type, key, scroll, wait, and
  zoom operations. Batched actions execute in order before a new screenshot is
  normally returned.
- [Gemini computer use](https://ai.google.dev/gemini-api/docs/generate-content/computer-use)
  returns a function action with normalized coordinates. The client executes it
  and sends a new screenshot as the result.

Research systems use similar histories. [ShowUI](https://arxiv.org/abs/2411.17465)
interleaves visual observations and actions; after action `i`, screenshot
`i+1` enters the history. [Aguvis](https://arxiv.org/abs/2412.04454) and
[UI-TARS](https://arxiv.org/abs/2501.12326) train general GUI agents from
multimodal trajectories. [ScreenAI](https://arxiv.org/abs/2402.04615) develops
screen-specific visual representations and tasks. [Agent
S](https://openreview.net/pdf?id=lIVRgt4nLv) demonstrates computer interaction
with experience and planning over a broad desktop benchmark.

The synchronization proposed here is therefore established practice. The open
research question is which representation best captures the causal effect of
an action on the next screen.

## 4. The correct training unit

Use an atomic transition as the minimum training example:

```text
(screen_t, action_t, execution_t, screen_t+1)
```

The action is the model output. The post-action screen is the next input and a
possible auxiliary target. `execution_t` records whether the action was
accepted, failed, or timed out. This prevents a failed click from being
silently labeled as a valid state transition.

[UI-Oceanus](https://arxiv.org/abs/2604.02345) directly studies atomic GUI
transitions `(s_t, a_t, s_t+1)`. On its shared transition corpus, the reported
forward-dynamics objective, predicting `s_t+1` from `s_t` and `a_t`, is more
useful than inverse or backward objectives. [Beyond
Syntax](https://arxiv.org/abs/2506.17697) also defines action semantics through
the state transition produced by an action. [ViMo](https://arxiv.org/abs/2504.13936)
predicts the next GUI image from the current GUI and action.

These results support an auxiliary forward objective. They do not show that
full-pixel prediction is always the best target. [How Mobile World Model
Guides GUI Agents](https://arxiv.org/abs/2605.10347) compares text deltas, full
text, diffusion images, and renderable code as predicted next-state forms. Its
results motivate testing a compact screen-effect representation rather than
assuming that pixel reconstruction is necessary.

## 5. Proposed action grammar

Use these semantic actions:

```text
CLICK(x, y, button)
DOUBLE_CLICK(x, y, button)
MOVE(x, y)
DRAG(x1, y1, x2, y2, button)
SCROLL(dx, dy)
TYPE(text)
KEY(chord)
WAIT(duration_bin)
DONE(status)
```

Normalize screen coordinates to integer bins from `0` to `999` on each axis.
Record the viewport dimensions separately so execution can recover native
coordinates. Keep the bin count configurable because display scaling and small
targets can change the required precision.

Use `TYPE(text)` for ordinary text entry. Use `KEY(chord)` for special keys and
shortcuts such as `ENTER`, `TAB`, or `CTRL+S`. This keeps natural text in the
existing language tokenizer and prevents long typed strings from becoming
hundreds of key-down and key-up tokens.

Do not retain raw cursor telemetry by default. Retain a movement only when it
changes the screen through hover, participates in a drag or drawing gesture,
or is required by the task. Raw motor jitter increases sequence length without
adding action semantics.

## 6. Synchronization and recording contract

Every transition needs stable identifiers and monotonic timestamps:

| Record | Required fields |
| --- | --- |
| Observation | `observation_id`, capture start/end, image hash, viewport, cursor position, focused surface, stability flag |
| Action | `action_id`, parent `observation_id`, decoded semantic action, model timestamp |
| Execution | `action_id`, execution start/end, accepted/failed/timed-out status, error class |
| Result | new `observation_id`, parent `action_id`, first-frame delay, stable-frame delay, image hash |

For each action:

1. Capture the pre-action frame.
2. Execute exactly one semantic action.
3. Record execution acknowledgement or failure.
4. Capture an immediate post-action frame.
5. Continue to capture frames until the screen is stable or a timeout expires.
6. Store intermediate frames or their deltas, and mark whether stability was
   reached.

Define stability before data collection. A practical initial rule is a small
perceptual-image difference for a fixed quiet interval, with a maximum wait.
Animations, video, blinking cursors, and clocks can prevent strict pixel
equality, so the threshold and quiet interval must remain calibration knobs.

One action per observation is the clean initial condition. Action batches save
latency but make it unclear which action caused each visible change. Batch
execution can be a later deployment optimization after the transition model is
validated.

## 7. What to tokenize

Three representations should remain separate hypotheses.

### 7.1 Fixed action tokens — baseline

Encode the action name and normalized numeric arguments with a fixed grammar.
This gives exact decoding, explicit validation, and a small output space. It
also makes invalid-action rate measurable.

### 7.2 Learned action tokenizer — ablation

A learned tokenizer could compress repeated action chunks or discover common
gestures. Its main risk is that frequency-based compression joins events that
are common but not causally meaningful. It also makes execution and validation
harder. Test it only after the fixed grammar establishes a task-success floor.

### 7.3 Learned screen-effect tokenizer — higher-value ablation

Train a tokenizer on the screen change between `screen_t` and `screen_t+1`,
conditioned on `action_t`. Candidate targets include changed patches, latent
feature deltas, structured text deltas, or renderable UI descriptions. This
could remove unchanged pixels and focus capacity on action consequences.

This is the stronger tokenizer hypothesis because it targets the uncertain
part of the problem: what changed and whether the action had the intended
effect.

## 8. Registered minimal experiment

### 8.1 Research question

Does synchronized forward-dynamics training improve held-out desktop task
success, and does learned tokenization add value beyond a fixed action grammar?

### 8.2 Arms

1. **Policy baseline:** pre-trained visual encoder, fixed action grammar, and
   next-action loss.
2. **Forward model:** baseline plus next-screen or screen-delta prediction.
3. **Shuffled transition control:** forward-model arm with post-action screens
   shuffled within matched task classes. This tests whether the model uses the
   causal pairing rather than screen statistics.
4. **Screen-effect tokenizer:** forward-model arm with learned discrete
   screen-change targets.
5. **Action tokenizer:** replace the fixed action grammar with a learned action
   vocabulary while holding the policy backbone and data fixed.
6. **Unsynchronized control:** pair actions with delayed or adjacent screens
   using a registered offset. This measures the cost of incorrect temporal
   alignment.

Use the same trajectory split, action budget, visual encoder, policy backbone,
optimizer budget, and evaluation harness in every arm. Hash the split and the
ordered transition manifest into each result artifact, and make the scorer
reject unequal hashes.

### 8.3 Floors

Score these before any treatment result:

- no-op policy;
- most-common-action policy;
- screen-only policy with no action-observation history.

The primary metric is held-out task success. Secondary metrics are valid-action
rate, target-hit rate, key-entry success, steps, wall time, stale-observation
rate, recovery after failed actions, and performance under display scaling or
small layout changes. Forward-model metrics must include their do-nothing or
unconditioned prediction floor.

Run at least three seeds and report confidence intervals. Pre-register the
minimum useful effect before treatment results are visible. A reasonable pilot
gate is an absolute five-point increase in held-out task success with no
material increase in invalid actions or action count, but the pilot variance
must set the final threshold.

### 8.4 Stop conditions

Stop work on a learned action tokenizer if it does not improve held-out task
success across three seeds over the fixed grammar, or if it increases invalid
actions or required steps. Stop full-image forward prediction if its auxiliary
loss improves while task success does not; move to a compact screen-effect
target instead.

### 8.5 Artifact sanity checks

Before training, the generator must print timestamped progress, invariant
counts, and decoded transitions. Required zero-valued invariants include:

- actions without a parent observation;
- result screens without a parent action;
- duplicate identifiers;
- negative or reversed timestamps;
- coordinates outside the normalized range;
- successful actions without a post-action frame;
- cross-split image or trajectory duplicates.

Print representative decoded samples for clicks, typing, keys, scrolls, drags,
waits, failures, timeouts, and unstable screens. Show the pre-frame, semantic
action, immediate frame, stable frame, timing, and visible change. Counts alone
cannot establish that the causal pairs are correct.

## 9. Evaluation environments and leakage controls

[OSWorld](https://github.com/xlang-ai/OSWorld) provides a real-computer
benchmark across applications. [WebArena](https://arxiv.org/abs/2307.13854)
and its [official release](https://github.com/web-arena-x/webarena) provide a
reproducible browser environment with functional tasks. These are useful
external evaluations after the transition objective is fixed.

For a screenshot-only claim, do not expose DOM nodes, accessibility trees,
widget identifiers, OCR boxes, or evaluator state to the policy. Such metadata
can be retained in a sealed evaluation channel for scoring. Split by task
family and layout template, not only by trajectory. Use perceptual hashes to
remove near-duplicate screens across splits.

Run collection in isolated virtual machines with synthetic credentials and no
access to personal data or destructive host operations. Require human approval
for purchases, account changes, deletion, credential entry, and other
high-impact actions. This isolation is part of experimental validity, not only
deployment safety.

## 10. Relation to Altrux

The current Altrux work studies continual factual learning through the
LAMA-CKL wake/dream protocol. Desktop control studies multimodal policy learning
and environment dynamics. Combining them before either result is established
would confound the source of improvement.

If the desktop baseline later works, Mamba is a plausible sequence backbone
for long action-observation histories. That comparison must hold the visual
encoder, fixed action grammar, synchronized data, and evaluation harness
constant. The first desktop experiment should not also test continual
consolidation, memory architecture, and learned tokenization.

## 11. Answers future sessions can rely on

- The causal order `screen_t -> action_t -> screen_t+1` is established in
  deployed computer-use loops and GUI-agent research.
- The key open question is representation, not whether actions and resulting
  screens should be synchronized.
- A fixed semantic action grammar is the correct baseline. Raw cursor and key
  telemetry is not a useful default training vocabulary.
- Forward next-state or screen-change prediction is the first auxiliary
  objective to test.
- A learned screen-effect tokenizer is a more informative ablation than a
  learned action tokenizer.
- One action per observation is the clean experimental unit. Batching belongs
  after causal learning is established.
- This branch does not change the current LAMA-CKL direction.

## Sources checked

Primary papers and official releases checked for this note: LLaVA, Flamingo,
BLIP-2, PaLM-E, Chameleon, Unified-IO 2, Gato, RT-1, RT-2, VIMA, OpenVLA,
Octo, ScreenAI, Aguvis, ShowUI, UI-TARS, Agent S, UI-Oceanus, Beyond Syntax,
ViMo, How Mobile World Model Guides GUI Agents, OSWorld, and WebArena. Official
computer-use documentation from OpenAI, Anthropic, and Google was also checked.

The repository-local starting points were the [notes index](../README.md), the
[current LAMA-CKL protocol](../discussion/DISCUSSION-20260823-lama-ckl-wake-dream-protocol.md),
the [current GPU handoff](../experiments/EXPERIMENT_NOTES-20260823-lama-ckl-gpu-handoff.md),
and the [modern continual-learning evaluation review](RESEARCH-20260823-modern-llm-continual-learning-evaluations.md).
