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

On the **instance**: clone the repo, `make sync`, verify a trivial `git push`
works, prepare data, then start training (`cd sft && make resume`) —
monitored by a Claude Code session started with `/watch-training`. No
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
instance; gitignored, so rsync is how it travels) — down from a running
instance to this machine via rsync, so they survive termination (everything
on the instance otherwise dies with it). Run it from the **local** machine:

```bash
./scripts/lambda_pull.sh                      # one-shot; IP via the API
./scripts/lambda_pull.sh 1.2.3.4              # one-shot; IP explicit
./scripts/lambda_pull.sh --follow             # re-pull every 5 min until terminated
./scripts/lambda_pull.sh --follow --interval 60
```

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

The instance is found via the API (expects exactly one active instance);
set `LAMBDA_INSTANCE_ID`/`LAMBDA_INSTANCE_IP` in `scripts/.env` to target
one explicitly. `--terminate-cmd "echo boom"` dry-runs the countdown.

The `/watch-training` slash command (`.claude/commands/watch-training.md`)
is the standing brief for a Claude Code session monitoring the run on the
instance — it encodes the watchdog contract, the commit-and-push +
`notes/WATCH_NOTES.md` persistence rules, and the give-up criteria.

## `lambda_check_key.sh`

Checks whether `LAMBDA_API_KEY` (in `scripts/.env`) is valid, without
terminating anything. Useful after generating or rotating a key. The Lambda
Cloud API has no dedicated "validate key" endpoint, so this hits the
lightweight, always-available `GET /instance-types` and reports whether the
key was accepted:

```bash
./scripts/lambda_check_key.sh
```
