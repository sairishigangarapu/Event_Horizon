"""
assess_information(reports, as_of) -> per_report / per_zone freshness, conflicts,
coverage, and flags.

Five labels are provided independently where supported; multiple may apply at
once. There is deliberately NO synthetic single truth number. A stale report is
flagged as potentially outdated (needs verification) — never as false.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from crowd_reliability.reliability.conflicts import (
    deduplicate_reports,
    detect_conflicts,
    group_comparable,
)
from crowd_reliability.reliability.coverage import compute_coverage
from crowd_reliability.reliability.freshness import (
    INSUFFICIENT_EVIDENCE_WINDOW_S,
    POTENTIALLY_OUTDATED_AGE_S,
    RAPID_CHANGE_MIN_REPORTS,
    RAPID_CHANGE_REL_STD,
    RAPID_CHANGE_WINDOW_S,
    RECENT_VERIFIED_MAX_AGE_S,
    _as_naive_utc,
    _field,
    observation_age_seconds,
)

FRESHNESS_LABELS = [
    "recently_verified",
    "potentially_outdated",
    "conflicting_evidence",
    "rapidly_changing",
    "insufficient_current_evidence",
]


def _empty_labels() -> Dict[str, bool]:
    return {label: False for label in FRESHNESS_LABELS}


def _relative_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    if abs(mean_val) < 1e-6:
        mean_val = 1e-6
    var = sum((v - mean_val) ** 2 for v in values) / len(values)
    return (var ** 0.5) / abs(mean_val)


def assess_information(
    reports: List[Any],
    as_of: datetime,
    expected_coverage: Optional[Dict[Tuple[str, str], List[str]]] = None,
) -> Dict[str, Any]:
    """
    Assesses evidence freshness/conflict/coverage over a set of reports.

    Returns:
        {
          "as_of": iso str,
          "dedup_mode": "source_id" | "content_hash_only" | "empty",
          "per_report": [{report_id, zone_id, variable, age_seconds,
                          is_duplicate_of, labels}],
          "per_zone":   [{zone_id, variable, age_seconds, independent_report_count,
                          conflict_ids, coverage_ratio, labels}],
        }
    Labels are independent booleans; age_seconds / conflict_ids / coverage_ratio
    are separate fields, not a fused score.
    """
    as_of_naive = _as_naive_utc(as_of)
    as_of_reports = [
        r for r in reports
        if (_field(r, "received_at") is None or _as_naive_utc(_field(r, "received_at")) <= as_of_naive)
    ]

    independent, duplicates, dedup_mode = deduplicate_reports(as_of_reports)
    dup_of: Dict[str, str] = {}
    for kept_id, dup_ids in duplicates.items():
        for d in dup_ids:
            dup_of[d] = kept_id

    # Disagreement checks over independent (deduped) comparable groups.
    comparable = group_comparable(independent, as_of)
    conflict_ids_by_key: Dict[Tuple[str, str], List[str]] = {}
    for key, group in comparable.items():
        is_conflict, ids = detect_conflicts(group)
        if is_conflict:
            conflict_ids_by_key[key] = ids
    conflicting_report_ids = {rid for ids in conflict_ids_by_key.values() for rid in ids}

    # Rapid-change detection per (zone, variable) over the rapid window.
    rapid_keys = set()
    by_key_all: Dict[Tuple[str, str], List[Any]] = {}
    for r in independent:
        by_key_all.setdefault((_field(r, "zone_id"), _field(r, "variable")), []).append(r)
    for key, group in by_key_all.items():
        recent = [r for r in group if observation_age_seconds(r, as_of) <= RAPID_CHANGE_WINDOW_S]
        if len(recent) >= RAPID_CHANGE_MIN_REPORTS:
            values = [float(_field(r, "value")) for r in recent]
            if _relative_std(values) > RAPID_CHANGE_REL_STD:
                rapid_keys.add(key)

    coverage = compute_coverage(independent, expected_coverage, as_of)

    # Per zone-variable labels (independent reports only).
    per_zone: List[Dict[str, Any]] = []
    zone_labels: Dict[Tuple[str, str], Dict[str, bool]] = {}
    for key in sorted(by_key_all.keys()):
        zone_id, variable = key
        group = by_key_all[key]
        ages = [observation_age_seconds(r, as_of) for r in group]
        latest_age = min(ages)

        labels = _empty_labels()
        labels["recently_verified"] = latest_age <= RECENT_VERIFIED_MAX_AGE_S
        labels["potentially_outdated"] = latest_age > POTENTIALLY_OUTDATED_AGE_S
        labels["conflicting_evidence"] = key in conflict_ids_by_key
        labels["rapidly_changing"] = key in rapid_keys
        labels["insufficient_current_evidence"] = latest_age > INSUFFICIENT_EVIDENCE_WINDOW_S

        cov_entry = coverage["per_zone_variable"].get(key, {})
        cov_ratio = cov_entry.get("coverage_ratio")
        if cov_ratio is not None and cov_ratio < 0.5:
            labels["insufficient_current_evidence"] = True

        zone_labels[key] = labels
        per_zone.append({
            "zone_id": zone_id,
            "variable": variable,
            "age_seconds": round(latest_age, 2),
            "independent_report_count": len(group),
            "conflict_ids": conflict_ids_by_key.get(key, []),
            "coverage_ratio": cov_ratio,
            "labels": labels,
        })

    # Zones with no reports at all are represented in coverage (if expected
    # coverage was supplied) and flagged insufficient at the overall level.
    if expected_coverage:
        for key in sorted(set(expected_coverage.keys()) - set(by_key_all.keys())):
            zone_id, variable = key
            labels = _empty_labels()
            labels["insufficient_current_evidence"] = True
            zone_labels[key] = labels
            per_zone.append({
                "zone_id": zone_id,
                "variable": variable,
                "age_seconds": None,
                "independent_report_count": 0,
                "conflict_ids": [],
                "coverage_ratio": 0.0,
                "labels": labels,
            })

    per_report: List[Dict[str, Any]] = []
    for r in sorted(as_of_reports, key=lambda r: str(_field(r, "report_id"))):
        report_id = str(_field(r, "report_id"))
        key = (_field(r, "zone_id"), _field(r, "variable"))
        age = observation_age_seconds(r, as_of)
        z_labels = zone_labels.get(key, _empty_labels())
        labels = {
            "recently_verified": age <= RECENT_VERIFIED_MAX_AGE_S,
            # Age alone => needs verification, never 'false'.
            "potentially_outdated": age > POTENTIALLY_OUTDATED_AGE_S,
            "conflicting_evidence": report_id in conflicting_report_ids,
            "rapidly_changing": key in rapid_keys,
            "insufficient_current_evidence": z_labels["insufficient_current_evidence"],
        }
        per_report.append({
            "report_id": report_id,
            "zone_id": key[0],
            "variable": key[1],
            "source_id": _field(r, "source_id"),
            "age_seconds": round(age, 2),
            "is_duplicate_of": dup_of.get(report_id),
            "labels": labels,
        })

    return {
        "as_of": as_of.isoformat(),
        "dedup_mode": dedup_mode,
        "label_names": list(FRESHNESS_LABELS),
        "per_report": per_report,
        "per_zone": per_zone,
        "coverage_overall": coverage["overall"],
    }
