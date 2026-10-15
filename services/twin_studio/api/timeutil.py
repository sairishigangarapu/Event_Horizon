"""Simulated-time helpers shared by the Phase 4 APIs (no real wall-clock time is ever used)."""
from __future__ import annotations

from datetime import datetime, timedelta


def parse_utc(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return dt


def seconds_since(epoch_iso: str, iso: str) -> int:
    delta = (parse_utc(iso) - parse_utc(epoch_iso)).total_seconds()
    if delta != int(delta):
        raise ValueError("timestamp is not on a whole simulated second")
    return int(delta)


def at(epoch_iso: str, seconds: int) -> str:
    return (parse_utc(epoch_iso) + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z")
