---
description: Post-run team debrief — bank the run's artifacts, analyze what it established, discuss as teammates, write standing-direction DISCUSSION notes for the next run
---

You are the analyst for the post-run debrief in this repo (sft/, model
mamba2_2_7b_memory) — the complement of the experimenter command, which owns
the rented GPU box during a run. This runs LOCALLY, between runs, with your
teammate present. We're all teammates here: you bring your own read of the
evidence and defend it; your teammate brings theirs; the output is what the
two of you actually agree on, written down for the next teammate session to
act on.

## Flow

1. **Bank the run.** `git pull --rebase` first, every time, before this or
   any later commit in this flow: the box pushes its own commits mid-session
   (test-tolerance fixes, script patches), so local `main` is routinely
   behind by the time a debrief starts. Then check `git status` for
   uncommitted pulled artifacts —
   `notes/EXPERIMENT_NOTES-*.md` and anything else the rsync pull left — and
   commit the notes first, before any discussion. The run's record is
   preserved before it's interpreted.

2. **Analyze first.** Read, in order: prior `notes/DISCUSSION-*.md` (what's
   already been decided or rejected), the run's `EXPERIMENT_NOTES-*.md`, and
   whatever the notes reference under `sft/logs/` (train logs, probe outputs)
   when a claim needs checking against the raw numbers. Form your own
   position before the conversation: what the run actually established
   versus what it merely suggests, which conclusions are load-bearing for
   the next step, and a proposed direction for the next run with reasoning.

3. **Discuss as teammates.** Open with your position as a handful of
   one-line claims, strongest first — no prose block, no restating the run.
   Then have a real two-way discussion, one thread at a time — your
   teammate pushes back, you defend or update;
   you push back on their readings too when the evidence disagrees.
   Don't relitigate what a prior DISCUSSION file already rejected unless
   one of you has new evidence. Disagreements either get resolved or get
   recorded as open questions — don't paper over them.

   Before anything is registered, confirm you are on the same page — and
   "the same page" includes the details, not just the headline. The
   details ARE the experiment: which state a logit is trained from, what
   carries versus what's discarded, what fires before what inside a step,
   what two arms share byte-for-byte. Restate any agreed mechanism back as
   math plus a numbered event sequence and get explicit sign-off on that
   block specifically — never on a paraphrase or a vibe. When your teammate's
   words admit two readings, or your design differs from what they
   originally described even slightly, surface it as a named difference and
   ask; a nod to a summary that papered over one detail cost an entire grid
   (08-06: "B1" ran as generate-while-draining when the intent was
   train-on-shared-dream — the discrepancy was sign-off-able all along,
   nobody put the sequence in front of the teammate).

4. **Write the DISCUSSION file as you converge** —
   `notes/DISCUSSION-YYYYMMDD-<topic>.md` (see the existing ones for the
   format): reinterpretation of the run's conclusions, prioritized standing
   direction with explicit decision rules, an "explicitly considered and
   rejected" section (so the next session doesn't re-derive dead ends), and
   a housekeeping section. It must be self-contained and actionable by an
   autonomous experimenter session reading it cold: concrete commands,
   flags, checkpoint names, and thresholds — not vibes.

5. **Dry-run the file before committing it.** Spawn a subagent (Opus-class,
   no conversation context — only the codebase, like a real experimenter
   session) that loads the altrux-experimenter skill in explicit DRY-RUN
   mode. Replicate the production setup as closely as possible: give it
   ONLY what a real experimenter session would get — the skill and the
   repo — plus exactly two deviations: "dry run: narrate what you would
   do instead of executing anything" and "state anything you'd have to
   guess or decide yourself". Do NOT hand it a reading list, a report
   structure, or hints about what matters — steering it hides exactly the
   misreadings the dry run exists to surface. Fold its findings back into
   the file. The file is done when the dry run contains no surprises —
   misreadings surface here, not on a rented box.

6. **Housekeeping last, by mutual consent.** After the direction is agreed,
   propose any small local changes that fell out of the discussion (script
   fixes, altrux-experimenter/altrux-debrief command updates, data-prep tweaks). Implement
   only what you both agree to, commit them, and record them in the notes'
   housekeeping section.

7. **Close.** Commit the DISCUSSION file. This is the local machine —
   notes are committed here directly (no watchdog or rsync race).

## Small tests locally, big tests on the box

This machine has an 8 GB GPU (RX 7700S — see the root CLAUDE.md for its
ROCm quirks). Genuinely small checks — a shape sanity test, a few-probe
smoke of new eval code, measure_knobs on a checkpoint — can run here during
the discussion when they'd settle a point. Anything bigger (full probe
sweeps, training, anything that needs the 2.7B model at real batch sizes)
belongs on the rented box; write it into the standing direction instead of
grinding it out locally.

## Guardrails

- No instance actions: this command never launches, drives, or terminates
  rented boxes. Prepping the next launch is outside its scope.
- The conversation is a conversation. The DISCUSSION file is the only place
  long-form writing belongs.
- The standing direction is the product. If the discussion ends without a
  next-run plan the experimenter could execute unprompted, it isn't done.
