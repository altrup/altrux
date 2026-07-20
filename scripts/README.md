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

To bring an **instance** up, run `./scripts/lambda_launch.sh` from the local
machine — it provisions the GPU, waits for ssh, then runs `lambda_setup.sh`
on it (clone, `make sync`, install the Claude Code CLI). Launch also
uploads your global Claude config (`~/.claude/` CLAUDE.md, status line, skills —
not settings.json, credentials, or history) so the instance session behaves
like your local one; setup wires the status line into the instance's settings. With `CLAUDE_CODE_OAUTH_TOKEN` set the `/experimenter` session starts
itself; otherwise ssh in, run `claude` to authenticate, and start it. No
watchdog, no terminate chain, no API key on the instance.

## `lambda_launch.sh`

Provisions a Lambda Cloud GPU instance and hands it off to `lambda_setup.sh`.
Runs on the **local** machine (it needs `LAMBDA_API_KEY`, which by design
never lives on the instance). Set `LAMBDA_INSTANCE_TYPE` and
`LAMBDA_SSH_KEY_NAME` in `scripts/.env` (region auto-picks an available one if
`LAMBDA_REGION` is unset), then:

```bash
./scripts/lambda_launch.sh            # launch + configure end to end
./scripts/lambda_launch.sh --dry-run  # resolve region + print payload only
./scripts/lambda_launch.sh --no-setup # launch + wait for ssh, skip setup
./scripts/lambda_launch.sh --no-watch # don't auto-start the local watchdog/pull
```

If the instance type is sold out, launch **polls** every
`LAMBDA_CAPACITY_POLL_INTERVAL` seconds (default 60) until capacity frees up —
`LAMBDA_CAPACITY_MAX_WAIT` (default 0 = forever) caps the wait. A 60s poll is
well within normal API use (the watchdog polls at the same rate all run long).

Unless `--no-watch` is passed, launch also starts a **local tmux session
`altrux`** with everything in one place:

- window `watch` — the billing-protection stack: `lambda_pull.sh --follow`
  (top pane) plus `lambda_watchdog.sh --arm-after-training` (bottom pane),
  both targeting the instance it just launched. So you don't have to remember
  to start them by hand — the box is protected from the moment it's up.
  `--arm-after-training` keeps the watchdog from terminating the box during
  the minutes-long setup before `train.py` exists (see below).
- window `train` — ssh'd into the remote `train` tmux (training/setup output,
  live). Waits for the remote session to exist, then attaches.
- window `claude` — ssh'd into the remote `experimenter` tmux (the Claude
  session). Same wait-then-attach.

Set `LAMBDA_RESUME_CHECKPOINT` to one or more local checkpoint step dirs
(space-separated, e.g.
`../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-168` — several when the
plan evals across checkpoints) — launch scp's them up to a staging dir and
setup drops them into `models/<model>/checkpoints/<epoch>/` so `make resume`
and the probes find them. Checkpoints
are gitignored and too big for GitHub (a single `optimizer.pt` exceeds the
100MB file limit), so this direct copy is the only sane transfer. Unset →
training starts fresh.

It waits for the instance to boot and accept ssh, then starts `lambda_setup.sh`
inside a detached tmux session named `train` and returns immediately, printing
the attach commands — so setup survives a dropped connection without holding
your terminal. Setup also creates a second session, `experimenter`. **Training runs in
`train`; the Claude Code `/experimenter` session runs in `experimenter` and
drives training in `train` via `tmux send-keys`** — so the run survives the
monitoring session ending, and monitoring never blocks on the run. Reattach to
either with `ssh … -t tmux attach -t <train|experimenter>`. The result is a box
that only needs `claude` auth and `/experimenter` to start. Region selection uses `regions_with_capacity_available` from the API,
so a launch fails fast with a clear message when there's no capacity rather
than erroring mid-launch.

Config (`GITHUB_TOKEN`, `HF_TOKEN`, `TORCH_BACKEND`, `MAX_JOBS`, and the
`LAMBDA_REPO_*` / `LAMBDA_SETUP_*` overrides) is forwarded to the instance via a
temporary env file, not the command line, so tokens don't appear in its process
list. `TORCH_BACKEND=cu128` is needed on GH200, where uv's `auto` backend
guesses the wrong torch wheel.

## `lambda_setup.sh`

Configures a freshly-launched instance: install `uv` if absent, wire up
`GITHUB_TOKEN` / `HF_TOKEN` if provided, clone-or-pull the repo at `~/altrux`,
write `sft/.env` (`MODEL_NAME=mamba2_2_7b_memory` — it's gitignored, so a clone
has none), `make sync` (and verify torch sees a CUDA GPU), verify `git push`
auth, and install the Claude Code CLI. Data prep is deliberately not part of
setup — which data to build (and with what flags) is an experimental decision,
so the experimenter runs it from the notes' plan.
Runs **on the instance** — either invoked automatically by `lambda_launch.sh`,
or by hand after ssh-ing in:

```bash
ssh ubuntu@<ip> 'bash lambda_setup.sh'   # after scp-ing it up
```

Idempotent, so a half-failed run is just re-run. Config is via `LAMBDA_REPO_*`
/ `LAMBDA_SETUP_*` env vars (see `scripts/.env.example`).

**Experimenter startup** depends on `CLAUDE_CODE_OAUTH_TOKEN`:

- **Set** (generate with `claude setup-token` on a logged-in machine — works on
  a Pro/Max subscription): setup auto-starts an autonomous Claude Code
  `/experimenter` session in the `experimenter` tmux, with a preamble noting the
  teammates may be AFK. It reads past notes, starts training, and monitors.
  Runs with `--dangerously-skip-permissions` (no teammate to approve tool calls);
  the watchdog bounds cost, the brief bounds behaviour.
- **Unset**: the `experimenter` session is created empty — attach, run `claude`,
  authenticate interactively, and invoke `/experimenter` by hand. No long-lived
  credential on the box.

Caveat: an interactive `claude` session runs its first turn then waits — the
auto-start *bootstraps* the run (notes → training → first health check)
unattended, but continuous hours-long monitoring still needs the session
driven (a teammate attaching, or a self-scheduling loop). `claude -p` is not used
because it exits after one turn.

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
awake and online for the whole run, or nothing stops the billing.

`--arm-after-training` holds the idle countdown until `train.py` is first seen
(bounded by `--arm-cap` minutes, default 120, 0 = forever), so the watchdog can
be started *before* training exists — during a long setup/data-gen — without
terminating the box prematurely. `lambda_launch.sh` uses this when it
auto-starts the watchdog; a plain manual `lambda_watchdog.sh` alongside an
already-training run doesn't need it. Run it alongside the pull loop:

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
