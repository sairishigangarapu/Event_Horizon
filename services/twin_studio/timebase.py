"""Time semantics for Twin Studio engine output (twin_venue_v1 contract).

Engine loop (engine/simulation.py `simulate`): for t = 0..duration it spawns agents scheduled
at t, moves every agent once by its per-second speed, then appends rows[t] and frames[t].
So engine step k covers the simulated interval [k, k+1) s, and rows[k] / frames[k] describe
the state at the END of that interval, i.e. the instant k+1 s (end-of-interval convention,
approved decision D2).

The instant 0 has no engine frame: no agent exists before the first step, so the state at
instant 0 is the empty venue.

The main forecasting pipeline samples every 5 s. A canonical sample at instant T (a multiple
of the sampling interval) takes instantaneous fields from engine step T-1 and aggregates
interval fields over engine steps T-dt .. T-1, i.e. the interval [T-dt, T).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

ENGINE_STEP_SECONDS = 1


def step_interval(step_index: int) -> tuple[int, int]:
    """Simulated interval [start, end) in seconds covered by engine step `step_index`."""
    if step_index < 0:
        raise ValueError("step_index must be >= 0")
    return step_index * ENGINE_STEP_SECONDS, (step_index + 1) * ENGINE_STEP_SECONDS


def step_instant(step_index: int) -> int:
    """Instant (seconds) whose state rows[step_index] / frames[step_index] describe."""
    return step_interval(step_index)[1]


def instant_step(instant_s: int) -> int | None:
    """Engine step whose output is the state at `instant_s`; None for instant 0 (empty venue)."""
    if instant_s < 0 or instant_s % ENGINE_STEP_SECONDS:
        raise ValueError("instant must be a non-negative multiple of the engine step")
    return None if instant_s == 0 else instant_s // ENGINE_STEP_SECONDS - 1


def last_instant(duration_s: int) -> int:
    """The engine runs steps 0..duration inclusive, so the final state is at duration+1 s."""
    return step_instant(duration_s)


def canonical_instants(duration_s: int, sample_seconds: int = 5) -> list[int]:
    """Sampling instants 0, dt, 2dt, ... up to the last engine state (instant 0 = empty venue)."""
    if sample_seconds <= 0 or sample_seconds % ENGINE_STEP_SECONDS:
        raise ValueError("sample_seconds must be a positive multiple of the engine step")
    return list(range(0, last_instant(duration_s) + 1, sample_seconds))


def interval_steps(instant_s: int, sample_seconds: int = 5) -> range:
    """Engine steps aggregated for interval fields of the canonical sample at `instant_s`: [T-dt, T)."""
    if instant_s < sample_seconds:
        return range(0)  # instant 0 has no preceding interval
    return range(instant_s - sample_seconds, instant_s)


def instant_to_utc(epoch_utc: datetime, instant_s: int) -> datetime:
    """Map a simulated instant to a UTC timestamp. The epoch is SIMULATED time, not a real collection time."""
    if epoch_utc.tzinfo is None:
        raise ValueError("epoch_utc must be timezone-aware")
    return epoch_utc.astimezone(timezone.utc) + timedelta(seconds=instant_s)
