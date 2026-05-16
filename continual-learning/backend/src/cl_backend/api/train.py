import asyncio
from fastapi import APIRouter, Request

from cl_backend.schemas import (
    CriticTrainRequest,
    CriticTrainResponse,
    PolicyTrainRequest,
    PolicyTrainResponse,
    HistoryEntry,
    HistoryResponse,
)

router = APIRouter()


@router.post("/train/critic", response_model=CriticTrainResponse)
async def train_critic(body: CriticTrainRequest, request: Request) -> CriticTrainResponse:
    trainer = request.app.state.trainer
    async with request.app.state.train_lock:
        loss = await asyncio.to_thread(
            trainer.critic_step, body.prompt, body.response, body.user_reward
        )
    return CriticTrainResponse(loss=loss, step=len(trainer.history))


@router.post("/train/policy", response_model=PolicyTrainResponse)
async def train_policy(body: PolicyTrainRequest, request: Request) -> PolicyTrainResponse:
    trainer = request.app.state.trainer
    async with request.app.state.train_lock:
        reward, loss = await asyncio.to_thread(
            trainer.policy_step, body.prompt, body.response
        )
    return PolicyTrainResponse(reward=reward, loss=loss, step=len(trainer.history))


@router.get("/train/history", response_model=HistoryResponse)
async def get_history(n: int = 10, request: Request = None) -> HistoryResponse:
    history = request.app.state.trainer.history
    entries = [
        HistoryEntry(
            step=i + 1,
            phase=h["phase"],
            user_reward=h.get("user_reward"),
            predicted_reward=h["predicted_reward"],
            loss=h["loss"],
        )
        for i, h in enumerate(history[-n:], start=max(0, len(history) - n))
    ]
    return HistoryResponse(entries=entries)
