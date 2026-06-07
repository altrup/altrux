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


class ReviseEntry(BaseModel):
    at_turn: int  # message index of the latest model response when this revision was recorded
    revision: str  # full "<revise back=N>...</revise weight=0.5>" tag string


class SessionResponse(BaseModel):
    text: str
    tokens: list[TokenInfo]
    messages: list[dict]
    revise_suggestions: list[ReviseEntry]


class ReviseRequest(BaseModel):
    n: int = Field(ge=1)  # how many model messages back the revision targets
    revision: str


class ReviseResponse(BaseModel):
    ok: bool
