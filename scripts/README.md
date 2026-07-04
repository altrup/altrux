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

Chain it after a training run so a crash doesn't leave the instance (and the
bill) running:

```bash
cd sft && make train && ../scripts/lambda_terminate.sh
```

By default it looks up the running instance by matching this machine's public
IP, so it must be run from inside the instance. To terminate a specific
instance (e.g. from a different machine), set `LAMBDA_INSTANCE_ID` in
`scripts/.env` instead.

## `lambda_check_key.sh`

Checks whether `LAMBDA_API_KEY` (in `scripts/.env`) is valid, without
terminating anything. Useful after generating or rotating a key. The Lambda
Cloud API has no dedicated "validate key" endpoint, so this hits the
lightweight, always-available `GET /instance-types` and reports whether the
key was accepted:

```bash
./scripts/lambda_check_key.sh
```
