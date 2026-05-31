import torch
from fastapi import APIRouter

from ..config import get_device
from ..model.loader import registry
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        mode=registry.mode,
        model_loaded=registry.model is not None,
        records_collected=registry.total_records() if registry.model is not None else 0,
    )


@router.get("/device")
async def device_info() -> dict:
    info: dict = {
        "configured_device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [
            {"index": i, "name": torch.cuda.get_device_name(i)}
            for i in range(torch.cuda.device_count())
        ],
    }
    if registry.model is not None:
        p = next(registry.model.parameters())
        info["model_device"] = str(p.device)
        info["model_dtype"] = str(p.dtype)
    return info
