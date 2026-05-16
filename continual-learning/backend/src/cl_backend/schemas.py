from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(256, ge=16, le=512)


class GenerateResponse(BaseModel):
    response: str
    estimated_reward: float


class CriticTrainRequest(BaseModel):
    prompt: str
    response: str
    user_reward: float = Field(..., ge=-1.0, le=1.0)


class CriticTrainResponse(BaseModel):
    loss: float
    step: int


class PolicyTrainRequest(BaseModel):
    prompt: str
    response: str


class PolicyTrainResponse(BaseModel):
    reward: float
    loss: float
    step: int


class CheckpointStatusResponse(BaseModel):
    step: int
    checkpoint_dir: str
    last_saved_step: Optional[int]


class HistoryEntry(BaseModel):
    step: int
    phase: int
    user_reward: Optional[float]
    predicted_reward: float
    loss: float


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
