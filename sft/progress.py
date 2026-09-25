"""Experiment: shared

Dependency-free progress formatting shared across SFT domains."""

from datetime import datetime


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fmt_duration(seconds: float) -> str:
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes}m{remaining:02d}s" if minutes else f"{remaining}s"
