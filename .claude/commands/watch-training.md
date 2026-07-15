---
description: Babysit a training run on a rented GPU instance — monitor, debug, keep costs bounded
---

Watch over the training run in this repo (sft/, model mamba2_2_7b_memory),
started with: cd sft && make resume

This is a rented GPU instance billed hourly. Wasted idle time is wasted
money, but a wasted *run* (training garbage for hours) is worse — prefer
catching problems early over maximizing uptime.

THE WATCHDOG: a watchdog on the owner's machine (scripts/lambda_watchdog.sh
run there, probing this instance over ssh) terminates the instance 30
minutes after train.py stops, whatever the reason. While you're actively
investigating with training stopped, run `touch scripts/.watchdog-delay` —
deliberately, when you check in on your work, at least every 25 minutes. If
you're done (fixed and training restarted, or concluded it's unfixable),
stop touching it.

PERSISTENCE: everything on this instance is DESTROYED at termination. Two
things survive: what you git push, and what the owner's local machine
rsyncs down via scripts/lambda_pull.sh (sft/logs/, models/*/checkpoints/,
notes/). Therefore:

- Any code change: commit AND push promptly. Never leave fixes only in the
  working tree.
- Write observations (health checks, anomalies, fixes, open questions) to
  notes/WATCH_NOTES.md as you go, not at the end.
- The rsync pull runs every ~5 minutes, so anything you write needs the
  instance alive that much longer to survive. Whenever you finish your LAST
  writes before going quiet (final notes, a fix you just pushed), touch
  scripts/.watchdog-delay once more — that guarantees a full watchdog
  window (~6 pull cycles) before termination, so nothing is written and
  then immediately lost.

MONITORING, every ~15 min while training runs: tail the newest
sft/logs/train-*.log. Healthy: loss trending down, "surprise" NOT flat at
~1.0, w1_abs_max neither collapsing to 0 nor growing unboundedly, few/no
"non-finite" warnings. Repeated "non-finite loss/gradient, skipping chunk"
warnings are a known failure mode — if more than rare: stop the run,
reproduce with `make detect-anomaly` (halts at the offending op with a
traceback), investigate. sft/measure_knobs.py diagnoses the memory write
knobs on a checkpoint or fresh init.

IF TRAINING CRASHES:

1. Diagnose from the traceback and log tail before restarting.
2. Clean fix -> apply, commit, push, `make resume`.
3. Same failure twice after a fix attempt, or you're guessing: stop. Write
   everything you learned to notes/WATCH_NOTES.md, push any commits, verify
   the latest checkpoint is intact, touch the delay file one final time (so
   the rsync pull has a full window to collect what you wrote), then go
   silent and let the watchdog terminate the instance. An unsolved bug at
   3am is the owner's problem tomorrow, not a reason to bill more hours.

THERE IS NO LAMBDA API KEY ON THIS INSTANCE — termination is controlled
entirely from the owner's machine, and this instance holds no credentials
to the owner's Lambda account. Never attempt to obtain such credentials or
control instances by any other route; if you believe the run needs
different resources, write that in notes/WATCH_NOTES.md for the owner to
decide.
