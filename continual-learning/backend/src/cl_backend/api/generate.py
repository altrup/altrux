import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from cl_backend.schemas import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    model = request.app.state.model
    response, reward = await asyncio.to_thread(model.generate, body.prompt, body.max_new_tokens)
    return GenerateResponse(response=response, estimated_reward=reward)


@router.get("/generate/stream")
async def stream_generate(
    prompt: str,
    max_new_tokens: int = 256,
    request: Request = None,
) -> StreamingResponse:
    model = request.app.state.model
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _prefill_cb(layer_idx: int, total_layers: int) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "prefill", "layer": layer_idx, "total": total_layers},
        )

    def _produce() -> None:
        for chunk, reward in model.generate_stream(prompt, max_new_tokens, prefill_callback=_prefill_cb):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "chunk", "text": chunk, "reward": reward},
            )
        loop.call_soon_threadsafe(queue.put_nowait, None)

    async def _event_stream():
        # Tokenize prompt so the frontend can reveal one token at a time
        token_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [model.tokenizer.decode([tid]) for tid in token_ids]
        yield f"data: {json.dumps({'status': 'prefill', 'tokens': tokens})}\n\n"
        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            if item["type"] == "prefill":
                yield f"data: {json.dumps({'prefill_layer': item['layer'], 'total_layers': item['total']})}\n\n"
            else:
                chunk, reward = item["text"], item["reward"]
                if reward is None:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                else:
                    yield f"data: {json.dumps({'chunk': chunk, 'estimated_reward': reward})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
