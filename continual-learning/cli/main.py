import json
from typing import Annotated

import httpx
import typer
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.text import Text

app = typer.Typer(no_args_is_help=True)
session_app = typer.Typer()
app.add_typer(session_app, name="session")

console = Console()

_DEFAULT_URL = "http://localhost:8000"

_HELP = "Enter = new line  |  Meta+Enter (or Esc then Enter) = submit  |  empty = quit"


def _user_prompt() -> str:
    """Inline `[USER] ` prompt; the typed text stays in the transcript."""
    return pt_prompt(HTML("<ansicyan>[USER] </ansicyan>"), multiline=True)


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


@session_app.command("message")
def session_message(
    content: str,
    role: Annotated[str, typer.Option(help="user or assistant")] = "user",
    url: Annotated[str, typer.Option()] = _DEFAULT_URL,
) -> None:
    """Append a chat turn — the server wraps it with the configured role openers."""
    with _client(url) as c:
        r = c.put("/session/message", json={"role": role, "content": content})
        r.raise_for_status()
    console.print_json(r.text)


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

        console.print(f"[dim]{_HELP}[/dim]")

        if reset or not has_session:
            c.delete("/session").raise_for_status()
        else:
            # Replay the existing transcript so the conversation reads continuously.
            # EOS (id 0) renders as a newline rather than the literal <|endoftext|>.
            transcript = Text()
            for t in existing["tokens"]:
                transcript.append("\n" if t["id"] == 0 else t["text"])
            console.print(transcript, end="", soft_wrap=True)

        # The transcript flows inline: each turn prints the `[USER] ` prompt (typed in
        # place), then streams the assistant response — which begins with the model's
        # own `[ASSISTANT] ` opener — token by token, in green, right below it.
        while True:
            user_msg = _user_prompt()
            if not user_msg.strip():
                break
            c.put("/session/message", json={"content": user_msg}).raise_for_status()

            with c.stream(
                "POST", "/generate/stream",
                json={"temperature": temperature, "top_p": top_p},
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    tok = json.loads(line)
                    if tok["is_eos"]:
                        break
                    console.print(Text(tok["token"], style="bold green"), end="", soft_wrap=True)
            console.print()  # end the assistant turn


if __name__ == "__main__":
    app()
