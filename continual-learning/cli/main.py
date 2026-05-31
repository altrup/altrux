import sys
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.rule import Rule

app = typer.Typer(no_args_is_help=True)
mode_app = typer.Typer()
session_app = typer.Typer()
app.add_typer(mode_app, name="mode")
app.add_typer(session_app, name="session")

console = Console()

_DEFAULT_URL = "http://localhost:8000"


def _client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=120.0)


# ------------------------------------------------------------------
# health
# ------------------------------------------------------------------


@app.command()
def health(url: Annotated[str, typer.Option()] = _DEFAULT_URL) -> None:
    with _client(url) as c:
        r = c.get("/health")
        r.raise_for_status()
    console.print_json(r.text)


# ------------------------------------------------------------------
# mode
# ------------------------------------------------------------------


@mode_app.command("get")
def mode_get(url: Annotated[str, typer.Option()] = _DEFAULT_URL) -> None:
    with _client(url) as c:
        r = c.get("/mode")
        r.raise_for_status()
    console.print_json(r.text)


@mode_app.command("set")
def mode_set(
    value: str,
    url: Annotated[str, typer.Option()] = _DEFAULT_URL,
) -> None:
    with _client(url) as c:
        r = c.post("/mode", json={"mode": value})
        r.raise_for_status()
    console.print_json(r.text)


# ------------------------------------------------------------------
# session
# ------------------------------------------------------------------


@session_app.command("get")
def session_get(url: Annotated[str, typer.Option()] = _DEFAULT_URL) -> None:
    with _client(url) as c:
        r = c.get("/session")
        r.raise_for_status()
    sess = r.json()
    for t in sess["tokens"]:
        console.print(f"  {t['id']:6d}  {repr(t['text'])}")
    console.print(f"\n[dim]{len(sess['tokens'])} tokens — {repr(sess['text'])}[/dim]")


@session_app.command("reset")
def session_reset(
    text: Annotated[str | None, typer.Option()] = None,
    url: Annotated[str, typer.Option()] = _DEFAULT_URL,
) -> None:
    with _client(url) as c:
        r = c.post("/session/reset", json={"text": text})
        r.raise_for_status()
    console.print_json(r.text)


# ------------------------------------------------------------------
# chat
# ------------------------------------------------------------------


@app.command()
def chat(
    url: Annotated[str, typer.Option()] = _DEFAULT_URL,
    reset: Annotated[bool, typer.Option("--reset", help="Start a fresh session instead of continuing")] = False,
    temperature: Annotated[float, typer.Option("--temperature", "-t", help="Sampling temperature (0.0 = greedy)")] = 0.8,
    top_p: Annotated[float, typer.Option("--top-p", "-p", help="Nucleus sampling threshold")] = 0.95,
) -> None:
    with _client(url) as c:
        existing = c.get("/session").raise_for_status().json()
        has_session = bool(existing["tokens"])

        if reset or not has_session:
            initial = typer.prompt("Enter initial text", default="")
            c.post("/session/reset", json={"text": initial or None}).raise_for_status()
        else:
            console.print(f"[dim]Continuing session ({len(existing['tokens'])} tokens)[/dim]")

        while True:
            # Generate next token
            gen_resp = c.post("/generate", json={"temperature": temperature, "top_p": top_p})
            gen_resp.raise_for_status()
            gen = gen_resp.json()
            last_token = gen["tokens"][-1]

            # Fetch full session for context display
            sess_resp = c.get("/session")
            sess_resp.raise_for_status()
            sess = sess_resp.json()

            # Build display: all text up to generated token, then highlight it
            tokens = sess["tokens"]
            if tokens:
                pre_text = "".join(t["text"] for t in tokens[:-1])
                last_text = tokens[-1]["text"]
                display = pre_text + f"[bold green][{last_text}][/bold green]"
            else:
                display = ""

            console.print(Rule())
            console.print(display)
            console.print(Rule())

            if last_token.get("is_eos"):
                console.print("[dim]<end of text>[/dim]")
                break

            if last_token["critic_reward"] is not None:
                console.print(
                    f"critic: [dim]{last_token['critic_reward']:.4f}  {last_token.get('critic_reward_note', '')}[/dim]"
                )

            raw = typer.prompt("Reward [-1..1, Enter=skip, q=quit]", default="")

            if raw.strip().lower() == "q":
                break

            if raw.strip() == "":
                continue

            if raw.strip().lower() == "history":
                for t in sess["tokens"]:
                    console.print(f"  {t['id']:6d}  {repr(t['text'])}")
                continue

            try:
                reward = float(raw.strip())
            except ValueError:
                console.print("[red]Invalid reward — enter a number in [-1, 1] or press Enter to skip.[/red]")
                continue

            if not -1.0 <= reward <= 1.0:
                console.print("[red]Reward must be in [-1, 1].[/red]")
                continue

            reward_resp = c.post("/reward", json={"reward": reward})
            reward_resp.raise_for_status()
            rdata = reward_resp.json()
            saved_flag = rdata.get("saved", False)
            if saved_flag:
                console.print(f"  [green]✓ saved (total: {rdata['total_records']})[/green]")
            else:
                console.print("  [dim](unfrozen mode — reward acknowledged but not stored)[/dim]")


if __name__ == "__main__":
    app()
