import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

_raw = os.getenv("DEVICE", "auto").lower()


def get_device() -> str:
    if _raw == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return _raw


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "mamba2_780m")


def get_sft_checkpoint() -> Path | None:
    val = os.getenv("SFT_CHECKPOINT", "").strip()
    return Path(val) if val else None


def get_revise_data_path() -> Path:
    val = os.getenv("REVISE_DATA_PATH", "").strip()
    p = Path(val) if val else Path(__file__).parent.parent.parent / "data" / "revise.jsonl"
    if not p.is_absolute():
        p = (Path(__file__).parent.parent / p).resolve()
    return p


def get_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173").strip()
    return [o.strip() for o in raw.split(",") if o.strip()]
