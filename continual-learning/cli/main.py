import json
from typing import Annotated

import httpx
import typer
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.text import Text

app = typer.Typer(no_args_is_help=True)
session_app = typer.Typer()
app.add_typer(session_app, name="session")

console = Console()

_DEFAULT_URL = "http://localhost:8000"

def _prompt(label: str) -> str:
    message = HTML(
        f"<ansibrightblack>Enter = new line  |  Meta+Enter (or Esc then Enter) = submit\n"
        f"{label}\n</ansibrightblack>"
    )
    return pt_prompt(message, multiline=True)


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
        c.delete("/session").raise_for_status()
        if text:
            r = c.put("/session", json={"text": text})
            r.raise_for_status()
        else:
            r = c.get("/session")
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
            c.delete("/session").raise_for_status()
            initial = _prompt("Inject initial text into session:")
            if initial:
                c.put("/session", json={"text": initial}).raise_for_status()
        else:
            console.print(f"[dim]Continuing session ({len(existing['tokens'])} tokens)[/dim]")

        while True:
            # Render the existing context, then stream the new response into it,
            # updating the display live as each token arrives. EOS (id 0) renders as a
            # newline rather than the literal <|endoftext|>, so turns separate cleanly.
            # Display-only — the session keeps the real token, so the model still
            # conditions on the trained <EOS>[USER] format.
            sess = c.get("/session").raise_for_status().json()
            display = Text()
            for t in sess["tokens"]:
                display.append("\n" if t["id"] == 0 else t["text"])

            console.print(Rule())
            with Live(display, console=console, auto_refresh=False) as live:
                live.refresh()
                with c.stream(
                    "POST", "/generate/stream",
                    json={"temperature": temperature, "top_p": top_p},
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.strip():
                            continue
                        tok = json.loads(line)
                        display.append(
                            "\n" if tok["token_id"] == 0 else tok["token"],
                            style="bold green",
                        )
                        live.refresh()
            console.print(Rule())

            user_msg = _prompt("Inject text into session (Enter on empty = quit):")
            if not user_msg.strip():
                break
            c.put("/session", json={"text": user_msg}).raise_for_status()


if __name__ == "__main__":
    app()
