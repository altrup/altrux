# scripts/

Standalone helper scripts that don't belong to a specific subproject.

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

Pulls training artifacts — `sft/logs/` and every `models/*/checkpoints/` —
down from a running instance to this machine via rsync, so the run's logs
survive termination (they otherwise die with the instance). Run it from the
**local** machine before `lambda_terminate.sh`:

```bash
./scripts/lambda_pull.sh                      # one-shot; IP via the API
./scripts/lambda_pull.sh 1.2.3.4              # one-shot; IP explicit
./scripts/lambda_pull.sh --follow             # re-pull every 5 min until terminated
./scripts/lambda_pull.sh --follow --interval 60
```

`--follow` keeps pulling until the instance stops answering (terminated) or
Ctrl-C, bounding what a hard crash can lose to one interval. Keep its
`--interval` (default 300s) under the watchdog's `--timeout` (default 600s):
the watchdog's grace period after training exits is exactly the window the
final pull happens in.

Assumes the repo lives at `~/altrux` on the instance (override with
`LAMBDA_REMOTE_REPO` in `scripts/.env`) and ssh access as `ubuntu@`.

## `lambda_watchdog.sh`

Terminates the instance (via `lambda_terminate.sh`) once nothing has been
training for `--timeout` seconds (default 600). Run it on the instance in the
background at the start of a session:

```bash
nohup ./scripts/lambda_watchdog.sh >> watchdog.log 2>&1 &
```

"Training" = a process matching `--pattern` (default `train.py`) exists.
Anyone working interactively on the instance between runs (e.g. a Claude Code
session) can push termination back by touching the delay file:

```bash
touch scripts/.watchdog-delay
```

A touch grants at most one `--timeout` window from the moment of the touch
(future-dated mtimes are rewritten to now), so the timer can only be
*delayed* — repeatedly, by touching again every <timeout seconds while
actively working — never paused outright. `--terminate-cmd "echo boom"`
dry-runs the countdown without a real terminate. This is a guardrail against
a forgotten idle instance, not a security boundary: anything on the instance
could also just kill the watchdog process.

## `lambda_check_key.sh`

Checks whether `LAMBDA_API_KEY` (in `scripts/.env`) is valid, without
terminating anything. Useful after generating or rotating a key. The Lambda
Cloud API has no dedicated "validate key" endpoint, so this hits the
lightweight, always-available `GET /instance-types` and reports whether the
key was accepted:

```bash
./scripts/lambda_check_key.sh
```
