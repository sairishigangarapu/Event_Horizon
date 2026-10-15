"""
utils.py – Shared datetime utilities for event_horizon_p4.

NOTE: Shared datetime normalization utilities for event_horizon_p4.

Centralises helpers that were previously duplicated across evaluation.py,
comparison.py, and sequence.py to prevent divergent implementations.
"""

from __future__ import annotations

from datetime import datetime, timezone


def to_naive_utc(dt: datetime) -> datetime:
    """
    Converts a datetime to a timezone-naive UTC datetime.

    If *dt* already lacks timezone info it is returned unchanged (assumed
    to already represent UTC).  This avoids the subtle bug of comparing
    an aware datetime against a naive one, which raises TypeError in Python.

    Args:
        dt: Any datetime, aware or naive.

    Returns:
        A timezone-naive datetime in UTC.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt
