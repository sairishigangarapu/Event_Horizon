"""
Observation-age calculation and fixed freshness thresholds.

These thresholds are fixed defaults (P0/P1). They are named constants so a later
scenario-dependent explainable scheme (P2) can be compared against them.
A report older than POTENTIALLY_OUTDATED_AGE_S is 'potentially outdated' and
needs verification — it is NEVER asserted to be false solely because of age.
"""
from datetime import datetime, timezone
from typing import Any, Optional

# Fixed default thresholds (seconds unless noted).
RECENT_VERIFIED_MAX_AGE_S = 15.0       # newest independent report this fresh => recently verified
POTENTIALLY_OUTDATED_AGE_S = 45.0      # older than this => needs verification (NOT false)
INSUFFICIENT_EVIDENCE_WINDOW_S = 90.0  # no independent report within this => insufficient evidence
CONFLICT_WINDOW_S = 30.0               # comparable-report disagreement window
CONFLICT_REL_DEVIATION = 0.4           # |v - mean| / mean above this => disagreement
CONFLICT_MEAN_EPSILON = 0.1            # conflicts only evaluated when mean exceeds this
RAPID_CHANGE_WINDOW_S = 60.0           # window for rapid-change detection
RAPID_CHANGE_REL_STD = 0.25            # relative std above this => rapidly changing
RAPID_CHANGE_MIN_REPORTS = 3           # need at least this many independent reports


def _field(report: Any, name: str, default: Any = None) -> Any:
    if isinstance(report, dict):
        return report.get(name, default)
    return getattr(report, name, default)


def _as_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def observation_age_seconds(report: Any, as_of: datetime) -> float:
    """
    Age of a report relative to as_of, based on observed_at (when the evidence
    was captured). Falls back to received_at if observed_at is unavailable.
    """
    ref = _field(report, "observed_at") or _field(report, "received_at")
    if ref is None:
        return 999.0
    return max(0.0, (_as_naive_utc(as_of) - _as_naive_utc(ref)).total_seconds())


class FixedExpiryFreshnessBaseline:
    """
    Simple fixed-expiry baseline: a report is fresh iff its age is at most
    max_age_seconds. Kept as the named baseline for later comparison against
    scenario-dependent thresholds.
    """

    VERSION = "fixed_expiry_v1"

    def __init__(self, max_age_seconds: float = POTENTIALLY_OUTDATED_AGE_S):
        self.max_age_seconds = max_age_seconds

    def is_fresh(self, report: Any, as_of: datetime) -> bool:
        return observation_age_seconds(report, as_of) <= self.max_age_seconds

    def label(self, report: Any, as_of: datetime) -> str:
        return "fresh" if self.is_fresh(report, as_of) else "expired"
