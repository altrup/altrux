import os

import progress


def test_heartbeat_touches_the_delay_file_at_most_once_a_minute(tmp_path, monkeypatch):
    delay = tmp_path / ".watchdog-delay"
    clock = [1000.0]
    monkeypatch.setattr(progress, "DELAY_FILE", delay)
    monkeypatch.setattr(progress, "_last_heartbeat", None)
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])

    progress.heartbeat()
    assert delay.exists()
    os.utime(delay, (0, 0))

    clock[0] += 59
    progress.heartbeat()
    assert delay.stat().st_mtime == 0

    clock[0] += 1
    progress.heartbeat()
    assert delay.stat().st_mtime > 0


def test_heartbeat_swallows_an_unwritable_path(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "DELAY_FILE", tmp_path / "missing-dir" / ".watchdog-delay")
    monkeypatch.setattr(progress, "_last_heartbeat", None)
    progress.heartbeat()


def test_delay_file_is_the_watchdog_path():
    assert progress.DELAY_FILE.parts[-2:] == ("scripts", ".watchdog-delay")
    assert (progress.DELAY_FILE.parent.parent / "sft" / "progress.py").exists()
