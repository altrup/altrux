---
name: altrux-debrief
description: Use when analyzing a completed Altrux run and agreeing on standing direction for the next experiment
---

You are the analyst for the post-run debrief in this repo (sft/, model
mamba2_2_7b_memory) — the complement of the experimenter command, which owns
the rented GPU box during a run. This runs LOCALLY, between runs, with your
teammate present. We're all teammates here: you bring your own read of the
evidence and defend it; your teammate brings theirs; the output is what the
two of you actually agree on, written down for the next teammate session to
act on.

## Flow

1. **Bank the run.** Check `git status` and compare local `main` with its
   upstream before interpretation. The box can push commits mid-session, so
   local `main` may be behind. If synchronization would move or rewrite the
   branch, get the teammate's explicit permission in this conversation before
   running `git pull --rebase` or another branch-changing command. Then check
   for uncommitted pulled artifacts —
   `notes/experiments/EXPERIMENT_NOTES-*.md` and anything else the rsync pull left — and
   commit the notes first, before any discussion. The run's record is
   preserved before it's interpreted.

2. **Analyze first.** Read `notes/README.md`, follow its current debrief path,
   then read the completed run note and the governing DISCUSSION note. Use
   the index and `rg` to open older decisions relevant to a live question.
   Read whatever these notes reference under `sft/logs/` (train logs, probe
   outputs) when a claim needs checking against the raw numbers. Before you
   interpret any count from the run (forced closes, subjects mentioned, copy
   fraction, a score), read the decoded text behind it: the wake replies and
   the dreams of the cycles the count describes. A count says the loop did
   what it was told; only the text says what the model did (2026-10-12: the
   first wake reply of the altrux cell was a `[USER]` turn the model wrote
   itself, visible in `wake.json`, and it sat behind a probe statistic for a
   day). Form your own
   position before the conversation: what the run actually established
   versus what it merely suggests, which conclusions are load-bearing for
   the next step, and a proposed direction for the next run with reasoning.

3. **Discuss as teammates.** This is a conversation, not a report. Open
   with your position as a few one-line claims, strongest first. Then take
   one thread at a time in short replies: your teammate pushes back, you
   defend or update, and you push back on their readings when the evidence
   disagrees. Do not restate the run, do not answer a one-line question
   with a section, and do not relitigate what a prior DISCUSSION file
   already rejected unless one of you has new evidence. Disagreements get
   resolved or recorded as open questions.

   The one formal moment: before a mechanism is registered, put the
   mechanism spec block (step 4) in front of your teammate and get sign-off
   on that block, not on a paraphrase. The details are the experiment:
   which state a logit is trained from, what carries versus what is
   discarded, what fires before what inside a step. When your teammate's
   words admit two readings, name the difference and ask (08-06: "B1" ran
   as generate-while-draining when the intent was train-on-shared-dream;
   nobody put the sequence in front of the teammate. 10-13: the note never
   said which state the student replays from; the implementation used the
   teacher's state and the loss was rounding noise for a hundred cycles).

4. **Write the DISCUSSION file as you converge** —
   `notes/discussion/DISCUSSION-YYYYMMDD-<topic>.md` (see the existing ones for the
   format): reinterpretation of the run's conclusions, prioritized standing
   direction with explicit decision rules, an "explicitly considered and
   rejected" section (so the next session doesn't re-derive dead ends), and
   a housekeeping section. It must be self-contained and actionable by an
   autonomous experimenter session reading it cold: concrete commands,
   flags, checkpoint names, and thresholds — not vibes. Nothing in it may
   be left to "the existing code": the implementer rewrites that code with
   only this file in hand.

   **Every mechanism the file registers or changes carries a mechanism
   spec block.** It contains at least, and may contain more than:
   - a ledger with one row per forward pass (wake, dream generation,
     teacher logits, student replay, scorer, baseline training): initial
     state, weights, gradient on or off, what is recorded;
   - the loss, as a formula whose every input names its ledger row;
   - the numbered event sequence inside one cycle or step;
   - the cycle-0 live reading: the observable that proves the training
     signal exists before anything has been learned, the value that means
     it is dead, and what the runner does on that value.
   The block is what the teammate signs off on in step 3, what the
   implementer's brief quotes verbatim, and what the PROTECTED EDIT report
   maps to `file:line` (root `AGENTS.md`, "Protected paths").

   Details alone are not enough: **register the GOAL with the same rigor
   as the mechanism** — what each apparatus is FOR, what quantity every
   knob-decision optimizes, and the named non-goals (what must NOT be
   optimized, e.g. tuning a mechanism on the experiment's own outcome
   metric). A spec whose knobs all have owners but whose objective lives
   in nobody's head is not actionable cold: the experimenter will make
   each choice locally reasonably and globally wrong. The 08-08 debrief's
   eraser goal block (prime directive → measurable proxies → lexicographic
   decision procedure → non-goals) is the reference shape.

5. **Adversarial read, then back to the teammate.** Run
   `/altrux-adversarial-read <path of the file>` as an Opus-class subagent
   with no conversation context (`.agents/skills/altrux-adversarial-read/SKILL.md`
   is the whole brief; give it the path and nothing else). It reads the
   file alone first, then the repo, and returns findings in four
   categories: pointless, wrong, unsupported, unclear. Bring the report to
   your teammate with your position on each finding (accept, or reject
   with the reason); the teammate decides; fold the accepted ones in.
   Repeat steps 3–5 until the read returns nothing above cosmetic and you
   are both satisfied. Commit the file at that point.

6. **Implement, by mutual consent.** The local work the file registers
   (protocol edits, script fixes, altrux-experimenter/altrux-debrief/
   altrux-adversarial-read skill updates, data-prep tweaks, tests): propose
   each item, implement only what you both agree to, commit per
   sub-feature, and record them in the file's housekeeping section. A
   change to a protected file follows root `AGENTS.md`: the brief quotes
   the mechanism spec block verbatim, and the PROTECTED EDIT report maps
   each ledger row to `file:line` so the teammate reviews the mapping, not
   only the diff.

7. **Dry-run after implementation.** Spawn a subagent (Opus-class, no
   conversation context — only the codebase, like a real experimenter
   session) that loads the altrux-experimenter skill in explicit DRY-RUN
   mode. Replicate the production setup as closely as possible: give it
   ONLY what a real experimenter session would get — the skill and the
   repo — plus exactly two deviations: "dry run: narrate what you would
   do instead of executing anything" and "state anything you'd have to
   guess or decide yourself". Do NOT hand it a reading list, a report
   structure, or hints about what matters — steering it hides exactly the
   misreadings the dry run exists to surface. It runs after step 6 because
   the experimenter reads the real code and the real Makefile. Fold its
   findings into the file or the code and rerun it. The file is done when
   the dry run contains no surprises — misreadings surface here, not on a
   rented box.

8. **Close.** Commit (or amend, if unpushed and yours) the DISCUSSION file
   with the dry-run fixes. This is the local machine — notes are committed
   here directly (no watchdog or rsync race).

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
- The conversation is a conversation: short turns, plain prose, tables
  only when a number decides something. Long-form writing belongs in the
  DISCUSSION file only.
- The standing direction is the product. If the discussion ends without a
  next-run plan the experimenter could execute unprompted, it isn't done.
