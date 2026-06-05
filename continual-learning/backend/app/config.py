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


def get_sft_checkpoint() -> Path | None:
    val = os.getenv("SFT_CHECKPOINT", "").strip()
    return Path(val) if val else None
