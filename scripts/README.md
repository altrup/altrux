# scripts/

Standalone helper scripts that don't belong to a specific subproject.

## Typical run night (rented Lambda GPU)

One-time setup: `cp scripts/.env.example scripts/.env` and fill in
`LAMBDA_API_KEY`. Everything below assumes exactly one active instance (the
scripts find its IP via the API) and the repo cloned at `~/altrux` on it.

On the **local** machine — must stay awake and online for the whole run,
it's the only thing that can stop the billing:

```bash
./scripts/lambda_watchdog.sh &      # terminates the instance when idle; key stays local
./scripts/lambda_pull.sh --follow   # rescues logs/checkpoints/notes every 5 min
```

That's the whole local side — the watchdog does a final pull (including
`mem_state.pt`) by itself right before it terminates, so nothing needs to be
run by hand at the end. The `--follow` loop is still worth running alongside:
it bounds what a *hard* crash loses, where no graceful terminate ever happens.

On the **instance**: clone the repo, `make sync`, verify a trivial `git push`
works, prepare data, then start training (`cd sft && make resume`) —
monitored by a Claude Code session started with `/experimenter`. No
watchdog, no terminate chain, no API key on the instance.

## `lambda_terminate.sh`

Terminates the current Lambda Cloud GPU instance via the Lambda Cloud API — a
Lambda instance's own terminal can shut down the OS but cannot stop billing;
only the API's terminate endpoint does that.

Setup:

```bash
cp scripts/.env.example scripts/.env
# fill in LAMBDA_API_KEY (generate at https://cloud.lambdalabs.com/api-keys)
```

The usual way to invoke it is indirectly, via `lambda_watchdog.sh` (below),
which calls it once nothing has been training for its timeout — covering
normal completion, crashes, and forgotten idle instances with one mechanism.
It can also be run by hand, or chained directly after a run (use `;`, not
`&&`, so terminate still fires when training *fails* — the exact case it
exists for):

```bash
cd sft && { make resume; ../scripts/lambda_terminate.sh; }
```

By default it looks up the running instance by matching this machine's public
IP, so it must be run from inside the instance. To terminate a specific
instance (e.g. from a different machine), set `LAMBDA_INSTANCE_ID` in
`scripts/.env` instead.

## `lambda_pull.sh`

Pulls training artifacts — `sft/logs/`, every `models/*/checkpoints/`, and
`notes/` (free-form observations written by whoever is monitoring on the
instance; committed only from the local machine after a run, so rsync is
how it travels off the instance) — down from a running
instance to this machine via rsync, so they survive termination (everything
on the instance otherwise dies with it). Run it from the **local** machine:

```bash
./scripts/lambda_pull.sh                      # one-shot; IP via the API
./scripts/lambda_pull.sh 1.2.3.4              # one-shot; IP explicit
./scripts/lambda_pull.sh --follow             # re-pull every 5 min until terminated
./scripts/lambda_pull.sh --follow --interval 60
./scripts/lambda_pull.sh --with-mem-state     # final pull, before terminating
```

`mem_state.pt` — a checkpoint's full internal model state, written by
`sft/train.py` — is excluded unless `--with-mem-state` is passed. It's large
enough that pulling it every interval wouldn't finish between pulls, and a
checkpoint without it still resumes (mid-example slots restart rather than
continuing exactly). Pass `--with-mem-state` for one last pull before
terminating, when exact mid-example resume is worth the transfer.

Any of those three that don't exist on the instance are skipped — a run that
hasn't written its first checkpoint yet isn't an error. A missing *repo* is,
and fails immediately (including under `--follow`, so a wrong
`LAMBDA_REMOTE_REPO` can't masquerade as a terminated instance).

`--follow` keeps pulling until the instance stops answering (terminated) or
Ctrl-C, bounding what a hard crash can lose to one interval. Keep its
`--interval` (default 300s) under the watchdog's `--timeout` (default
1800s): the watchdog's grace period after training exits is exactly the
window the final pull happens in.

Assumes the repo lives at `~/altrux` on the instance (override with
`LAMBDA_REMOTE_REPO` in `scripts/.env`) and ssh access as `ubuntu@`.

## `lambda_watchdog.sh`

Runs on the **local** machine and terminates the instance (via
`lambda_terminate.sh`) once nothing has been training there for `--timeout`
seconds (default 1800 — generous enough that a monitoring session's
deliberate delay touches don't need to be unrealistically frequent, while a
forgotten instance still dies within half an hour). Keeping it local means
the Lambda API key never exists on the instance at all — nothing running
there (including an autonomous monitoring session) holds credentials to
launch, resize, or terminate instances. The tradeoff: this machine must stay
awake and online for the whole run, or nothing stops the billing. Run it
alongside the pull loop:

```bash
./scripts/lambda_watchdog.sh &
./scripts/lambda_pull.sh --follow
```

Right before terminating, it runs `lambda_pull.sh` twice: once for the
resume-critical files, then once with `--with-mem-state` for the large
`mem_state.pt`. A graceful terminate is the only moment that knows a run is
over, so it's the only place `mem_state.pt` can be rescued automatically —
the `--follow` loop deliberately skips it. Both pulls are best-effort and
separately bounded (`--pull-timeout`, default 900s; `--mem-state-timeout`,
default 3600s): the terminate happens whether they succeed, fail, or time
out, because an unbounded billing leak is the one thing this script exists to
prevent. Small files go first so a timeout can't starve the files a resume
actually needs. `--no-mem-state` skips the second pull (useful on a slow
link — 8GB of `mem_state.pt` is ~8 min at 130 Mbit/s but ~14 hours at
1.5 Mbit/s); `--no-pull` skips both. The unreachable path never pulls —
there's nothing to pull from an instance that won't answer ssh.

"Training" = a process matching `--pattern` (default `train.py`) exists on
the instance, probed over ssh every `--interval` (60s). An instance that
stops answering ssh while the API reports it active is terminated after
`--unreachable-timeout` (900s) — unreachable can't be trained on, and
shouldn't bill. Anyone working interactively on the instance between runs
(e.g. a Claude Code session) can push termination back by touching the delay
file **on the instance**:

```bash
touch ~/altrux/scripts/.watchdog-delay
```

A touch grants at most one `--timeout` window from the moment of the touch
(future-dated mtimes are rewritten to now), so the timer can only be
*delayed* — repeatedly, by touching again every <timeout seconds while
actively working — never paused outright. `--terminate-cmd "echo boom"`
dry-runs the countdown without a real terminate. This is a guardrail against
a forgotten idle instance, not a security boundary: anything on the instance
could also just kill the watchdog process.

The inverse also exists — touching the terminate file **on the instance**:

```bash
touch ~/altrux/scripts/.watchdog-terminate
```

makes the next probe terminate immediately (with the usual final pulls)
instead of waiting out `--timeout`. It's how a session on the instance, which
holds no API key, says "this run is over, stop billing now" — requesting
termination can only *stop* billing, so it's the one instance-side control
that's safe to grant. Only touches made after the watchdog's first successful
probe count; a stale file left over from a previous run can't kill a healthy
run at startup.

The instance is found via the API (expects exactly one active instance);
set `LAMBDA_INSTANCE_ID`/`LAMBDA_INSTANCE_IP` in `scripts/.env` to target
one explicitly. `--terminate-cmd "echo boom"` dry-runs the countdown.

The `/experimenter` slash command (`.claude/commands/experimenter.md`)
is the standing brief for a Claude Code session monitoring the run on the
instance — it encodes the watchdog contract, the commit-and-push +
`notes/EXPERIMENT_NOTES.md` persistence rules, and the give-up criteria.

## `lambda_check_instance.sh`

Dry-runs `lambda_terminate.sh`'s instance resolution — reports every instance
the key can see and which one terminate would target, without terminating
anything:

```bash
./scripts/lambda_check_instance.sh
```

Exits non-zero when nothing resolves, so a misconfigured setup surfaces before
the terminate matters. Run from the **local** machine it only matches when
`LAMBDA_INSTANCE_ID` is set (terminate's IP lookup expects to run on the
instance).

## `lambda_check_key.sh`

Checks whether `LAMBDA_API_KEY` (in `scripts/.env`) is valid, without
terminating anything. Useful after generating or rotating a key. The Lambda
Cloud API has no dedicated "validate key" endpoint, so this hits the
lightweight, always-available `GET /instance-types` and reports whether the
key was accepted:

```bash
./scripts/lambda_check_key.sh
```
