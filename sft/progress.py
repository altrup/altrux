"""Dependency-free progress formatting shared across SFT domains."""

import time
from datetime import datetime
from pathlib import Path

# scripts/lambda_watchdog.sh reads this file's mtime as the box's liveness signal.
DELAY_FILE = Path(__file__).resolve().parent.parent / "scripts" / ".watchdog-delay"
HEARTBEAT_INTERVAL = 60.0
_last_heartbeat: float | None = None


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fmt_duration(seconds: float) -> str:
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes}m{remaining:02d}s" if minutes else f"{remaining}s"


def heartbeat() -> None:
    """Touches the watchdog delay file, at most once per HEARTBEAT_INTERVAL."""
    global _last_heartbeat
    now = time.monotonic()
    if _last_heartbeat is not None and now - _last_heartbeat < HEARTBEAT_INTERVAL:
        return
    _last_heartbeat = now
    try:
        DELAY_FILE.touch()
    except OSError:
        pass
