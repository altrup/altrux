from typing import Literal, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CL_", env_file=".env", extra="ignore")

    model: str = "ibm-granite/granite-4.0-h-micro"
    quantize: Optional[Literal["4bit", "8bit"]] = None

    @field_validator("quantize", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        return None if v == "" else v
    checkpoint_dir: str = "../checkpoints"
    no_resume: bool = False
    critic_lr: float = 1e-4
    base_lr: float = 1e-6
    save_every: int = 10
    keep_checkpoints: int = 1
    devices: Optional[str] = None  # comma-separated CUDA indices, e.g. "0,1"
    port: int = 8000

    def device_ids(self) -> list[int] | None:
        if self.devices is None:
            return None
        return [int(d.strip()) for d in self.devices.split(",") if d.strip()]
