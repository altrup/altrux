from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    mode: str
    model_loaded: bool
    records_collected: int


class ModeResponse(BaseModel):
    mode: str


class ModeRequest(BaseModel):
    mode: str = Field(pattern="^(frozen|unfrozen)$")


class TokenInfo(BaseModel):
    id: int
    text: str


class SessionResponse(BaseModel):
    text: str
    tokens: list[TokenInfo]
    pending_token_id: int | None


class SessionResetRequest(BaseModel):
    text: str | None = None


class GenerateRequest(BaseModel):
    run_critic: bool = False
    max_tokens: int = Field(default=1, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


class GeneratedToken(BaseModel):
    token: str
    token_id: int
    critic_reward: float | None
    critic_reward_note: str | None
    is_eos: bool = False


class GenerateResponse(BaseModel):
    tokens: list[GeneratedToken]
    generated_text: str


class RewardRequest(BaseModel):
    reward: float = Field(ge=-1.0, le=1.0)


class RewardResponse(BaseModel):
    saved: bool
    total_records: int
