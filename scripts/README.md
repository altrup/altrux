# scripts/

Standalone helper scripts that don't belong to a specific subproject.

## Typical run night (rented Lambda GPU)

One-time setup: `cp scripts/.env.example scripts/.env` and fill in
`LAMBDA_API_KEY`. Everything below assumes exactly one active instance (the
scripts find its IP via the API) and the repo cloned at `~/altrux` on it.

On the **local** machine — must stay awake and online for the whole run,
it's the only thing that can stop the billing:

```bash
./scripts/lambda_watchdog.sh   # pulls every 5 min; terminates when idle; key stays local
```

That's the whole local side. The watchdog owns the pulling — on a schedule,
on demand when the instance touches `scripts/.watchdog-fetch`, and once more
(including `mem_state.pt`) right before it terminates — so nothing needs to be
run by hand at the end. `lambda_pull.sh --follow` does the scheduled half on
its own, for a watchdog-less session.

To bring an **instance** up, run `./scripts/lambda_launch.sh` from the local
machine — it provisions the GPU, waits for ssh, then runs `lambda_setup.sh`
on it (clone, `make sync`, install the selected agent CLI). Set
`EXPERIMENTER_AGENT=claude|codex`. Launch uploads only that provider's config
and credentials. The remote tmux session is always named `experimenter`; the
agent loads the shared repository experimenter workflow. No watchdog, terminate
chain, or Lambda API key lives on the instance.

## Training-data artifacts: generate once, reuse forever

Datasets under `sft/data/` are gitignored and expensive to build, so they
travel by rsync in both directions and are archived on the local machine:

1. The box generates an artifact the local machine doesn't have.
2. `lambda_pull.sh` brings it home — every 5 minutes under the watchdog, and
   the moment generation or filtering finishes if the box touches
   `scripts/.watchdog-fetch`, so the artifacts are safe before the long
   training phase starts.
3. The next `lambda_launch.sh` uploads it again, path-preserving, alongside the
   resume checkpoints. **The box only generates what isn't in that upload.**

The pre-launch confirm screen is the guard: it lists every artifact it's about
to upload, with sizes, before anything bills. What's listed there does not need
regenerating on the box, and `lambda_setup.sh` prints the same list again as it
places the files. There's no manifest or hashing — rsync's own change detection
decides what moves, so re-running either direction is free.

`scripts/lambda_data_artifacts.sh` holds the one include list both ends read
(`LAMBDA_DATA_ARTIFACTS` in `scripts/.env` overrides it; empty syncs no data).
It takes the finished `train_*`/`eval_*` datasets and their scored
`*-filtered` variants — the scored files carry per-item scores, so any later
threshold policy is a local `filter_items.py --rescore` away and the box's
final filtered variant needn't come home separately. Scratch (`pilot_*`,
`smoke_*`) and raw intermediates stay put.

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
`LAMBDA_CAPACITY_POLL_INTERVAL` seconds (default 30) until capacity frees up —
`LAMBDA_CAPACITY_MAX_WAIT` (default 0 = forever) caps the wait. The poll rate is
well within normal API use (the watchdog polls at a similar rate all run long).
When capacity appears but the launch call loses the race (scarce types sell out
in seconds), or the API answers with a non-JSON body (CDN rate limiting), launch
drops straight back to polling — the next poll re-checks availability before
trying again, and the interval doubles as the rate-limit cool-off.

If `.cache/wheels/` (repo root, gitignored) holds compatible wheels — harvested
from a running instance's uv cache with `./scripts/lambda_harvest_wheels.sh`
(grabs the compiled `mamba_ssm` and `causal_conv1d` wheels; run it any time
after the instance's `make sync` finishes) — select them with
`LAMBDA_CACHE_ARTIFACTS` and setup
installs from them via `UV_FIND_LINKS`, skipping the multi-minute CUDA
compile. For example, a GH200 host uses
`LAMBDA_CACHE_ARTIFACTS="lama_ckl wheels/*aarch64.whl"`. A wheel with a
mismatched Python or platform tag is ignored automatically, but a
torch major-version bump isn't detectable from the filename — clear the stash
when torch changes, or the import will fail at runtime.

Unless `--no-watch` is passed, launch also starts a **local tmux session
`altrux`** with everything in one place:

- window `watch` — the billing-protection stack:
  `lambda_watchdog.sh --arm-after-training` against the instance it just
  launched, pulling artifacts and terminating when idle. So you don't have to
  remember to start it by hand — the box is protected from the moment it's up.
  `--arm-after-training` keeps the watchdog from terminating the box during
  the minutes-long setup before `train.py` exists (see below).
- window `train` — ssh'd into the remote `train` tmux (training/setup output,
  live). Waits for the remote session to exist, then attaches.
- window `agent` — ssh'd into the remote `experimenter` tmux (Claude or Codex).
  Same wait-then-attach.
- window `work` — ssh'd into the remote `work` tmux, where the experimenter
  runs everything that isn't training (data prep, filtering, probes, pulls),
  one named window per job. Same wait-then-attach — the experimenter creates
  that session the first time it needs one, so this window usually waits
  longer than the other two.

Set `LAMBDA_RESUME_CHECKPOINT` to one or more local checkpoint step dirs
(space-separated, e.g.
`../models/mamba2_2_7b_memory/checkpoints/epoch-1/step-168` — several when the
plan evals across checkpoints) — launch rsyncs them up to a staging dir
(excluding `optimizer.pt`: probes never read it, and it'd be ~2/3 of the
bytes for nothing) and
setup drops them into `models/<model>/checkpoints/<epoch>/` so `make resume`
and the probes find them. Put the checkpoint training will resume from in
`LAMBDA_RESUME_CHECKPOINT_FULL` instead — same format, uploaded *with*
`optimizer.pt`, so resume keeps its optimizer state instead of starting a
fresh one (a loss transient). A step dir listed in both gets its optimizer.
Checkpoints
are gitignored and too big for GitHub (a single `optimizer.pt` exceeds the
100MB file limit), so this direct copy is the only sane transfer. Unset →
training starts fresh.

Local `sft/data/` artifacts ride the same staging route automatically (see
"Training-data artifacts" above) — no env var needed to opt in, and the confirm
screen lists what's going up with sizes.

`LAMBDA_CACHE_ARTIFACTS` adds a symmetric allowlist for selected files,
directories, or globs below the repo's `.cache/`. Missing entries are skipped
on upload and can still be pulled after the instance creates them. Broad or
escaping entries are rejected; the launcher never mirrors all of `.cache/`.

It waits for the instance to boot and accept ssh, then starts `lambda_setup.sh`
inside a detached tmux session named `train` and returns immediately, printing
the attach commands — so setup survives a dropped connection without holding
your terminal. Setup also creates a second session, `experimenter`. **Training
runs in `train`; the selected agent runs in `experimenter` and drives training
in `train` via `tmux send-keys`** — so the run survives the monitoring session
ending, and monitoring never blocks on the run. Reattach to either with
`ssh … -t tmux attach -t <train|experimenter>`. Region selection uses
`regions_with_capacity_available` from the API,
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
write `sft/.env` with `MODEL_NAME` deliberately blank (it's gitignored, so a
clone has none; a run that doesn't name a model inline fails at import rather
than silently training a default — the experimenter passes `MODEL_NAME=<arm>`
inline per command), `make sync` (and verify torch sees a CUDA GPU), verify `git push`
auth, and install the selected Claude or Codex CLI. It also places whatever launch staged —
resume checkpoints and `sft/data/` artifacts — printing each one, so the
experimenter can see which datasets already exist. Data *prep* is deliberately
not part of setup: which data to build (and with what flags) is an experimental
decision, so the experimenter runs it from the notes' plan, for the artifacts
that didn't arrive with the upload.
Runs **on the instance** — either invoked automatically by `lambda_launch.sh`,
or by hand after ssh-ing in:

```bash
scp scripts/lambda_setup.sh scripts/experimenter_agent.sh ubuntu@<ip>:
ssh ubuntu@<ip> 'bash lambda_setup.sh'
```

Idempotent, so a half-failed run is just re-run. Config is via `LAMBDA_REPO_*`
/ `LAMBDA_SETUP_*` env vars (see `scripts/.env.example`).

**Experimenter startup** depends on `EXPERIMENTER_AGENT` and its credentials:

- **Claude** auto-starts when `CLAUDE_CODE_OAUTH_TOKEN` is set. Otherwise,
  authenticate after attaching and invoke `/altrux-experimenter`.
- **Codex** auto-starts when `codex login status` finds the uploaded or existing
  login cache. Otherwise, use `codex login --device-auth` after attaching and
  invoke `$altrux-experimenter`.

Caveat: an interactive agent session runs its first turn then waits — the
auto-start *bootstraps* the run (notes → training → first health check)
unattended, but continuous hours-long monitoring still needs the session
driven by its own workflow or a teammate.

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

Pulls training artifacts — `sft/logs/`, every `models/*/checkpoints/`,
`notes/` (free-form observations written by whoever is monitoring on the
instance; committed only from the local machine after a run, so rsync is
how it travels off the instance), and the `sft/data/` datasets listed in
`lambda_data_artifacts.sh` — down from a running
instance to this machine via rsync, so they survive termination (everything
on the instance otherwise dies with it). Run it from the **local** machine:

```bash
./scripts/lambda_pull.sh                      # one-shot; IP via the API
./scripts/lambda_pull.sh 1.2.3.4              # one-shot; IP explicit
./scripts/lambda_pull.sh --follow             # re-pull every 5 min until terminated
./scripts/lambda_pull.sh --follow --interval 60
./scripts/lambda_pull.sh --with-mem-state     # final pull, before terminating
./scripts/lambda_pull.sh --dry-run            # list what would transfer
```

Every pull is idempotent — rsync moves only what changed, and each run
itemizes the files it actually brought down — so it's safe to fire off
repeatedly during a session. Do exactly that when data generation or filtering
finishes: it archives the new artifacts locally (see "Training-data artifacts"
above) before the hours-long training phase, rather than betting them on the
box surviving. Normally the watchdog fires it for you (on its schedule, or
when the box touches `scripts/.watchdog-fetch`).

Every successful pull writes a receipt back to
`<remote_repo>/scripts/.pull-receipt` on the instance: a UTC timestamp, this
machine's hostname and repo path, and one `size path` line per pulled file **as
it exists here**. A session on the instance can't see this disk, so the receipt
is its only evidence that what it produced arrived — the shutdown checklist in
`.claude/commands/altrux-experimenter.md` has it read one before terminating. A
receipt that fails to write is a warning, never a failed pull.

`mem_state.pt` — a checkpoint's full internal model state, written by
`sft/train.py` — is excluded unless `--with-mem-state` is passed. It's large
enough that pulling it every interval wouldn't finish between pulls, and a
checkpoint without it still resumes (mid-example slots restart rather than
continuing exactly). Pass `--with-mem-state` for one last pull before
terminating, when exact mid-example resume is worth the transfer.

Any of those that don't exist on the instance are skipped — a run that
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
(bounded by `--arm-cap` minutes, default 90, 0 = forever), so the watchdog can
be started *before* training exists — during a long setup/data-gen — without
terminating the box prematurely. `lambda_launch.sh` uses this when it
auto-starts the watchdog; a plain manual `lambda_watchdog.sh` alongside an
already-training run doesn't need it.

It also owns the pulling: `lambda_pull.sh` runs every `--pull-interval`
seconds (default 300), and immediately whenever a probe finds
`scripts/.watchdog-fetch` on the instance — how a session there asks for its
artifacts to go home *now*, e.g. the moment a cache finishes building. The
marker is deleted once consumed, pull or no pull; the fresh
`scripts/.pull-receipt` is what says it worked.

Right before terminating, it runs the final pull (retried twice, ~2 min apart)
and then once more with `--with-mem-state` for the large `mem_state.pt`. A
graceful terminate is the only moment that knows a run is over, so it's the
only place `mem_state.pt` can be rescued automatically. Every stage is bounded
(`--pull-timeout`, default 900s; `--mem-state-timeout`, default 3600s) and the
terminate happens whether they succeed, fail, or time out, because an
unbounded billing leak is the one thing this script exists to prevent. Small
files go first so a timeout can't starve the files a resume actually needs; if
the final pull never succeeded, the `mem_state.pt` stage is skipped (three
failures in a row means ssh is gone, not slow) and a
`scripts/PULL-FAILED-<UTC timestamp>` file is left **here**, naming the
instance and the last error, so a lost run shows up in `git status` the same
day instead of being discovered weeks later. `--no-mem-state` skips the
mem_state stage (useful on a slow link — 8GB is ~8 min at 130 Mbit/s but ~14
hours at 1.5 Mbit/s); `--no-pull` skips all pulling, scheduled included. The
unreachable path never pulls — there's nothing to pull from an instance that
won't answer ssh.

"Training" = a process matching `--pattern` (default `train.py`; alternation
works, e.g. `train.py|probe_recall.py` — what launch's auto-started watchdog
passes, so eval/probe runs count as activity too) exists on the instance,
probed over ssh every `--interval` (60s). An instance that
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

The shared experimenter workflow (`.agents/skills/altrux-experimenter/SKILL.md`,
linked from `.claude/commands/altrux-experimenter.md`) is the standing brief
for either agent monitoring the run. It encodes the watchdog contract, the commit-and-push +
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
