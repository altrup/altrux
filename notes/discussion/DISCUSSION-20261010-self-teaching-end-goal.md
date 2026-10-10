# Discussion notes — 2026-10-10: the self-teaching end goal (context, not a plan)

Recorded from a conversation on 2026-10-10 (altrup). Nothing here is built,
scheduled, or registered. It exists so the LAMA-CKL work is read against the
end goal. More detail comes when any of it is picked up.

## 1. End goal

A continual-learning model that teaches itself, with a human or an external
model talking to it as the source of reward. No fine-tuning phases: new
behaviour, new tokens included, enters through the same wake learning as a
new fact. First environment: tmux, but designed for general computer use
from the start (pixels in, keys and mouse out), so tmux is a first
environment, not a special case.

## 2. Reward at wake

- Wake is new learning; sleep (dreams) is consolidation. The reward token
  is emitted during wake.
- After something happens, the model may emit a reward token with a
  continuous value attached. Gradation matters: slightly good and
  incredibly good must not collapse to one symbol. Three discrete tokens
  were considered and rejected.
- The value is anchored outside the model: the user's next input and the
  environment are what change in response to an action. The value can only
  improve against something the model does not control. A past reward is
  revised by a later counter-reward, not edited in place.
- Learning rule: an eligibility trace. One gradient-sized buffer per
  trainable parameter, decayed every generated token, plus that token's
  log-probability gradient; a reward token applies the buffer times the
  value. Recent tokens dominate, old ones fade, memory is constant. Start
  with positive weighting; the negative side is the unstable half.
- Decay only, no reset at a reward, so an earlier reward token stays in the
  trace. Open.
- Internal thinking (recurrent depth, [DISCUSSION-20260723](DISCUSSION-20260723-780m-integration-screen.md);
  the soft-token feedback variant is the alternative) emits nothing scored
  on its own. The gradient of the token that exits the loop flows back
  through every iteration and enters the trace as one entry.

## 3. Output vector

The model emits one vector per position; a slice is the vocabulary logits
and two untied rows are the reward value. Two input dimensions on every
token, zero except at reward positions, let the model read rewards back.
Same arithmetic as a separate head on the shared hidden state; the
vocabulary tie stays intact.

## 4. Encoder and decoder

- Pixels in through a pre-trained visual encoder from the start; actions out
  through a fixed grammar for keys and mouse. The 2026-08-25 research note
  ([RESEARCH-20260825](../research/RESEARCH-20260825-desktop-multimodal-action-state-tokenization.md))
  registers this baseline.
- Wanted property: the round trip through the environment returns to where
  it started, encode(render(decode(e))) close to e, so "I typed l" and "l
  appeared" are the same thing to the model. That is a cycle-consistency
  objective with the environment in the middle; with a non-trivial decoder
  it needs the forward model of the screen (the registered auxiliary
  objective).
- Open: whether the encoder keeps learning during continual use or is
  aligned once and frozen. The case for frozen: memories consolidated
  against a moving encoder quietly stop meaning what they meant.

## 5. What wake and sleep write to

Open. Two candidates:

- A dedicated learner: wake writes a fast adapter (the LoRA already in
  use), sleep distils it into the slow weights and resets it. Smallest step
  from the current apparatus; measures cleanly. First to try.
- Magnitude-based plasticity: wake favours under-used weights, sleep
  consolidates strong ones and pushes weak ones toward zero. Magnitude is a
  weak proxy for importance; ablation.

## 6. How this depends on the current work

Installing a new symbol from a few uses is the same ability as installing a
fact from one evidence document. The LAMA-CKL grid
([DISCUSSION-20261010](DISCUSSION-20261010-lama-ckl-first-box-debrief.md))
is the gate: if distilling dreams does not separate altrux from frozen, a
reward on wake learning has nothing to consolidate, and user-taught tokens
do not take.

## 7. Open questions

1. Trace decay constant, and whether sleep clears the trace.
2. Thinking: recurrent depth or soft-token feedback.
3. Encoder: live or frozen; re-alignment when the screen distribution shifts.
4. Measurement of "teaches itself": success rate on tasks with a checkable
   end state, over time, with the teacher held fixed.
