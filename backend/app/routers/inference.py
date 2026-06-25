import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..model.registry import registry
from ..schemas import (
    GenerateRequest,
    GenerateResponse,
    GeneratedToken,
    SessionInputRequest,
    SessionMessageRequest,
    SessionResponse,
    TokenInfo,
)

router = APIRouter()


def _session_response() -> SessionResponse:
    return SessionResponse(
        text=registry.get_session_text(),
        tokens=[TokenInfo(**t) for t in registry.get_session_tokens()],
        messages=list(registry.messages),
    )


@router.get("/session", response_model=SessionResponse)
async def get_session() -> SessionResponse:
    return _session_response()


@router.delete("/session", response_model=SessionResponse)
async def reset_session() -> SessionResponse:
    async with registry.lock:
        registry.reset_session()
    return _session_response()


@router.put("/session", response_model=SessionResponse)
async def session_input(req: SessionInputRequest) -> SessionResponse:
    async with registry.lock:
        registry.append_input(req.text)
    return _session_response()


@router.put("/session/message", response_model=SessionResponse)
async def session_message(req: SessionMessageRequest) -> SessionResponse:
    """Append a chat turn, wrapped with the configured role openers (USER_OPEN /
    ASST_OPEN) so you don't have to type them by hand. Defaults to the user role."""
    async with registry.lock:
        registry.append_message(req.role, req.content)
    return _session_response()


async def _iter_generate(req: GenerateRequest) -> AsyncIterator[dict]:
    """Yield one generated token at a time, holding the lock for the whole response.

    Generation runs until the model emits EOS or max_tokens is reached.
    Appends the completed assistant turn to registry.messages when done.
    """
    accumulated = ""
    async with registry.lock:
        for _ in range(req.max_tokens):
            result = await asyncio.to_thread(
                registry.generate_one_token, temperature=req.temperature, top_p=req.top_p
            )
            if not result["is_eos"]:
                accumulated += result["generated_token"]
            yield result
            if result["is_eos"]:
                break
        content = accumulated
        if content.startswith(registry.asst_open):
            content = content[len(registry.asst_open):]
        registry.messages.append({"role": "assistant", "content": content.strip()})


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    generated = [
        GeneratedToken(
            token=result["generated_token"],
            token_id=result["token_id"],
            is_eos=result["is_eos"],
        )
        async for result in _iter_generate(req)
    ]

    return GenerateResponse(
        tokens=generated,
        generated_text="".join(t.token for t in generated if not t.is_eos),
    )


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest) -> StreamingResponse:
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def body() -> AsyncIterator[str]:
        async for result in _iter_generate(req):
            yield json.dumps({
                "token": result["generated_token"],
                "token_id": result["token_id"],
                "is_eos": result["is_eos"],
            }) + "\n"

    return StreamingResponse(body(), media_type="application/x-ndjson")
