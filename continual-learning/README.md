# Continual Learning Model

Goal is to build toward an LLM that can revise its own output based on user feedback. The model is fine-tuned to **emit a `<revise>` tag after a user's response**, which becomes the trigger for a revision step.

## Base Model

[state-spaces/mamba2-780m](https://huggingface.co/state-spaces/mamba2-780m)

Optionally an SFT LoRA checkpoint is applied on top of the base model at load time (see `backend/README.md`).

## Architecture

The model is a thin wrapper around the base Mamba LM — it generates tokens until EOS, nothing more.

```
Input Tokens
     │
     ▼
[Embedding Layer]
     │
     ▼
[Mamba Layers]
     │
     ▼
[LM Head → Logits]
     │
     ▼
[Sample until EOS]
```

## The `<revise>` Tag

The model is fine-tuned to generate a `<revise>` tag after completing a response. For now the tag is treated like any other token — generation simply runs until EOS and the tag flows through as ordinary text.

**TODO:** detect the `<revise>` tag during generation and act on it (trigger a revision pass) rather than treating it as plain output.

---

## Components

| Directory  | Description |
|------------|-------------|
| `backend/` | FastAPI server — loads the Mamba model and serves the chat API |
| `frontend/` | React Router v7 web UI — chat interface for the model |
| `cli/`     | Command-line client |

---

## Open Questions / TODO

- **Tag detection:** Detect the `<revise>` tag at generation time and branch on it instead of emitting it as plain text.
- **Revision step:** Define what happens once `<revise>` is detected — how the model incorporates feedback and produces a revised response.
- **Catastrophic forgetting:** Select a mitigation once online updates are in scope (e.g. KL-regularization to the frozen base, replay, EWC, or constrained/LoRA updates).
