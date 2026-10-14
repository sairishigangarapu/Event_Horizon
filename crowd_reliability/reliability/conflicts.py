"""
Comparable-report grouping, duplicate/copy collapse, and disagreement checks.

Two forwarded copies of one claim must NOT count as independent confirmations.
Reports are collapsed by content identity (zone, variable, rounded value,
5-second-normalized observed_at): identical content at the same time is the
same claim regardless of how many channels forwarded it. When source metadata
is present it is recorded per group; when absent, dedup is flagged as
'content_hash_only'.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from crowd_reliability.reliability.freshness import (
    CONFLICT_MEAN_EPSILON,
    CONFLICT_REL_DEVIATION,
    CONFLICT_WINDOW_S,
    _field,
    _as_naive_utc,
)

NORM_EPOCH = datetime(2026, 1, 1)


def norm_ts_5s(dt: datetime) -> int:
    """5-second rounding shared with the comparator/evaluator alignment convention."""
    dt = _as_naive_utc(dt)
    return int(round((dt - NORM_EPOCH).total_seconds() / 5.0) * 5)


def deduplicate_reports(reports: List[Any]) -> Tuple[List[Any], Dict[str, List[str]], str]:
    """
    Collapse report copies into independent reports.

    Returns:
        independent: one representative report per unique claim
        duplicates:  {kept_report_id: [duplicate_report_ids]}
        mode: 'source_id' when all reports carry source metadata,
              'content_hash_only' when some do not, 'empty' for no reports
    """
    if not reports:
        return [], {}, "empty"

    has_source_meta = all(_field(r, "source_id") for r in reports)
    mode = "source_id" if has_source_meta else "content_hash_only"

    groups: Dict[Tuple, List[Any]] = {}
    for r in reports:
        observed_at = _field(r, "observed_at") or _field(r, "received_at")
        key = (
            _field(r, "zone_id"),
            _field(r, "variable"),
            round(float(_field(r, "value", 0.0)), 3),
            norm_ts_5s(observed_at) if observed_at else None,
        )
        groups.setdefault(key, []).append(r)

    independent: List[Any] = []
    duplicates: Dict[str, List[str]] = {}
    for key in sorted(groups.keys(), key=lambda k: (str(k[0]), str(k[1]), k[3] or 0)):
        group = groups[key]
        kept = min(group, key=lambda r: str(_field(r, "report_id")))
        independent.append(kept)
        dup_ids = [str(_field(r, "report_id")) for r in group if r is not kept]
        if dup_ids:
            duplicates[str(_field(kept, "report_id"))] = sorted(dup_ids)

    return independent, duplicates, mode


def group_comparable(reports: List[Any], as_of: datetime, window_s: float = CONFLICT_WINDOW_S) -> Dict[Tuple[str, str], List[Any]]:
    """
    Groups independent reports by (zone_id, variable) for those observed within
    window_s of as_of — the comparable source/zone/variable/time sets used for
    disagreement checks.
    """
    as_of_naive = _as_naive_utc(as_of)
    groups: Dict[Tuple[str, str], List[Any]] = {}
    for r in reports:
        observed_at = _field(r, "observed_at") or _field(r, "received_at")
        if observed_at is None:
            continue
        age = (as_of_naive - _as_naive_utc(observed_at)).total_seconds()
        if 0 <= age <= window_s:
            key = (_field(r, "zone_id"), _field(r, "variable"))
            groups.setdefault(key, []).append(r)
    return groups


def detect_conflicts(group: List[Any]) -> Tuple[bool, List[str]]:
    """
    Disagreement check over one comparable group: any report deviating more
    than CONFLICT_REL_DEVIATION from the group mean (when the mean exceeds
    CONFLICT_MEAN_EPSILON) is conflicting.
    """
    if len(group) < 2:
        return False, []
    values = [float(_field(r, "value")) for r in group]
    mean_val = sum(values) / len(values)
    conflict_ids: List[str] = []
    if mean_val > CONFLICT_MEAN_EPSILON:
        for r in group:
            v = float(_field(r, "value"))
            if abs(v - mean_val) / (mean_val + 1e-4) > CONFLICT_REL_DEVIATION:
                conflict_ids.append(str(_field(r, "report_id")))
    return (len(conflict_ids) > 0), sorted(conflict_ids)
