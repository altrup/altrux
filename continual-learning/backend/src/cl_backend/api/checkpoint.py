import asyncio
from fastapi import APIRouter, Request

from cl_backend.schemas import CheckpointStatusResponse

router = APIRouter()


@router.post("/checkpoint/save")
async def save_checkpoint(request: Request) -> dict:
    trainer = request.app.state.trainer
    async with request.app.state.train_lock:
        await asyncio.to_thread(trainer.save_now)
    return {"ok": True, "step": len(trainer.history)}


@router.get("/checkpoint/status", response_model=CheckpointStatusResponse)
async def checkpoint_status(request: Request) -> CheckpointStatusResponse:
    trainer = request.app.state.trainer
    history = trainer.history
    step = len(history)

    last_saved_step: int | None = None
    if history:
        save_every = trainer.config.save_every
        if save_every and step % save_every == 0:
            last_saved_step = step
        else:
            for s in range(step, 0, -1):
                if save_every and s % save_every == 0:
                    last_saved_step = s
                    break

    return CheckpointStatusResponse(
        step=step,
        checkpoint_dir=trainer.config.checkpoint_dir,
        last_saved_step=last_saved_step,
    )
