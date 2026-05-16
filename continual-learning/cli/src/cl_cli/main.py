"""CLI client for the continual-learning backend API.

All commands talk to a running backend via HTTP; no model loading happens here.
Server URL defaults to http://localhost:8000 or CL_SERVER_URL env var.
"""
import json
import sys
import click
import httpx


def _server(ctx: click.Context) -> str:
    return ctx.obj["server"]


@click.group()
@click.option(
    "--server",
    default=None,
    envvar="CL_SERVER_URL",
    help="Backend API URL (default: http://localhost:8000).",
)
@click.pass_context
def cli(ctx: click.Context, server: str | None) -> None:
    ctx.ensure_object(dict)
    ctx.obj["server"] = (server or "http://localhost:8000").rstrip("/")


@cli.command()
@click.argument("prompt")
@click.option("--max-tokens", default=256, show_default=True, help="Max new tokens to generate.")
@click.option("--no-stream", is_flag=True, help="Use the non-streaming endpoint.")
@click.pass_context
def generate(ctx: click.Context, prompt: str, max_tokens: int, no_stream: bool) -> None:
    """Generate a response from the model."""
    server = _server(ctx)

    if no_stream:
        resp = httpx.post(
            f"{server}/generate",
            json={"prompt": prompt, "max_new_tokens": max_tokens},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        click.echo(data["response"])
        click.echo(f"\nCritic reward: {data['estimated_reward']:.3f}", err=True)
        return

    url = f"{server}/generate/stream"
    params = {"prompt": prompt, "max_new_tokens": max_tokens}
    with httpx.Client(timeout=None) as client:
        with client.stream("GET", url, params=params) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                sys.stdout.write(data["chunk"])
                sys.stdout.flush()
                if "estimated_reward" in data:
                    click.echo(f"\n\nCritic reward: {data['estimated_reward']:.3f}", err=True)
    click.echo()


@cli.group()
def train() -> None:
    """Training commands."""


@train.command("critic")
@click.option("--prompt", required=True, help="Prompt used to generate the response.")
@click.option("--response", required=True, help="Model response to train on.")
@click.option(
    "--reward",
    required=True,
    type=click.FloatRange(-1.0, 1.0),
    help="Your reward signal in [-1, 1].",
)
@click.pass_context
def train_critic(ctx: click.Context, prompt: str, response: str, reward: float) -> None:
    """Phase 1: train the critic to predict your reward."""
    server = _server(ctx)
    resp = httpx.post(
        f"{server}/train/critic",
        json={"prompt": prompt, "response": response, "user_reward": reward},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    click.echo(f"Critic step done — loss: {data['loss']:.4f}, step: {data['step']}")


@train.command("policy")
@click.option("--prompt", required=True, help="Prompt used to generate the response.")
@click.option("--response", required=True, help="Model response to train on.")
@click.pass_context
def train_policy(ctx: click.Context, prompt: str, response: str) -> None:
    """Phase 2: update the base model using the critic's reward."""
    server = _server(ctx)
    resp = httpx.post(
        f"{server}/train/policy",
        json={"prompt": prompt, "response": response},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    click.echo(
        f"Policy step done — reward: {data['reward']:.4f}, loss: {data['loss']:.4f}, step: {data['step']}"
    )


@cli.command()
@click.pass_context
def save(ctx: click.Context) -> None:
    """Manually save a checkpoint."""
    server = _server(ctx)
    resp = httpx.post(f"{server}/checkpoint/save", timeout=300)
    resp.raise_for_status()
    data = resp.json()
    click.echo(f"Checkpoint saved at step {data['step']}.")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current training status."""
    server = _server(ctx)
    resp = httpx.get(f"{server}/checkpoint/status", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    click.echo(f"Step:           {data['step']}")
    click.echo(f"Checkpoint dir: {data['checkpoint_dir']}")
    click.echo(f"Last saved:     {data['last_saved_step'] or 'never'}")


@cli.command()
@click.option("--n", default=10, show_default=True, help="Number of recent entries to show.")
@click.pass_context
def history(ctx: click.Context, n: int) -> None:
    """Show recent training history."""
    server = _server(ctx)
    resp = httpx.get(f"{server}/train/history", params={"n": n}, timeout=10)
    resp.raise_for_status()
    entries = resp.json()["entries"]
    if not entries:
        click.echo("No training history yet.")
        return
    click.echo(f"{'Step':>5}  {'Phase':>5}  {'User Reward':>11}  {'Predicted':>9}  {'Loss':>8}")
    click.echo("-" * 48)
    for e in entries:
        ur = f"{e['user_reward']:.3f}" if e["user_reward"] is not None else "   —  "
        click.echo(
            f"{e['step']:>5}  {e['phase']:>5}  {ur:>11}  {e['predicted_reward']:>9.3f}  {e['loss']:>8.4f}"
        )


def main() -> None:
    cli()
