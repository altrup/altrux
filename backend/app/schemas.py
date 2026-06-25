from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class TokenInfo(BaseModel):
    id: int
    text: str


class SessionInputRequest(BaseModel):
    text: str


class SessionMessageRequest(BaseModel):
    role: str = Field(default="user", pattern="^(user|assistant)$")
    content: str


class GenerateRequest(BaseModel):
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


class GeneratedToken(BaseModel):
    token: str
    token_id: int
    is_eos: bool = False


class GenerateResponse(BaseModel):
    tokens: list[GeneratedToken]
    generated_text: str


class SessionResponse(BaseModel):
    text: str
    tokens: list[TokenInfo]
    messages: list[dict]
