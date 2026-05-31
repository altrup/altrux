import asyncio

from fastapi import APIRouter, HTTPException

from ..model.registry import registry
from ..schemas import (
    GenerateRequest,
    GenerateResponse,
    GeneratedToken,
    ModeRequest,
    ModeResponse,
    RewardRequest,
    RewardResponse,
    SessionResetRequest,
    SessionResponse,
    TokenInfo,
)

router = APIRouter()

_CRITIC_NOTE = "(random — critic not yet trained)"


@router.get("/mode", response_model=ModeResponse)
async def get_mode() -> ModeResponse:
    return ModeResponse(mode=registry.mode)


@router.post("/mode", response_model=ModeResponse)
async def set_mode(req: ModeRequest) -> ModeResponse:
    async with registry.lock:
        registry.set_mode(req.mode)  # type: ignore[arg-type]
    return ModeResponse(mode=registry.mode)


@router.get("/session", response_model=SessionResponse)
async def get_session() -> SessionResponse:
    return SessionResponse(
        text=registry.get_session_text(),
        tokens=[TokenInfo(**t) for t in registry.get_session_tokens()],
        pending_token_id=registry.pending_token_id,
    )


@router.post("/session/reset", response_model=SessionResponse)
async def reset_session(req: SessionResetRequest) -> SessionResponse:
    async with registry.lock:
        registry.reset_session(req.text)
    return SessionResponse(
        text=registry.get_session_text(),
        tokens=[TokenInfo(**t) for t in registry.get_session_tokens()],
        pending_token_id=None,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if registry.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    generated: list[GeneratedToken] = []

    async with registry.lock:
        for _ in range(req.max_tokens):
            result = await asyncio.to_thread(
                registry.generate_one_token, run_critic=req.run_critic, temperature=req.temperature, top_p=req.top_p
            )
            note = _CRITIC_NOTE if req.run_critic and result["critic_reward"] is not None else None
            generated.append(GeneratedToken(
                token=result["generated_token"],
                token_id=result["token_id"],
                critic_reward=result["critic_reward"],
                critic_reward_note=note,
                is_eos=result["is_eos"],
            ))
            if result["is_eos"]:
                break

    return GenerateResponse(
        tokens=generated,
        generated_text="".join(t.token for t in generated),
    )


@router.post("/reward", response_model=RewardResponse)
async def submit_reward(req: RewardRequest) -> RewardResponse:
    if registry.pending_token_id is None:
        raise HTTPException(
            status_code=422, detail="No pending token — call POST /generate first"
        )

    async with registry.lock:
        total = await asyncio.to_thread(registry.save_reward, req.reward)

    return RewardResponse(saved=registry.mode == "frozen", total_records=total)
