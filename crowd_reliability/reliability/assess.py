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
    ConflictGraph,
    build_conflict_graph,
    deduplicate_reports,
    detect_conflicts,
    group_comparable,
)
from crowd_reliability.reliability.coverage import compute_coverage
from crowd_reliability.reliability.freshness import (
    INSUFFICIENT_EVIDENCE_WINDOW_S,
    RAPID_CHANGE_MIN_REPORTS,
    RAPID_CHANGE_REL_STD,
    RAPID_CHANGE_WINDOW_S,
    _as_naive_utc,
    _field,
    observation_age_seconds,
)
from crowd_reliability.reliability.thresholds import (
    DEFAULT_FRESHNESS_THRESHOLDS,
    FreshnessThresholds,
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
    thresholds: Optional[FreshnessThresholds] = None,
    scenario_type: Optional[str] = None,
    zone_type_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Assesses evidence freshness/conflict/coverage over a set of reports.

    Uses scenario/zone-dependent explainable freshness thresholds (P2).

    Returns:
        {
          "as_of": iso str,
          "dedup_mode": "source_id" | "content_hash_only" | "empty",
          "thresholds": {variable: [recent_sec, outdated_sec]},
          "per_report": [{report_id, zone_id, variable, age_seconds,
                          is_duplicate_of, labels}],
          "per_zone":   [{zone_id, variable, age_seconds, independent_report_count,
                          conflict_ids, conflict_provenance, threshold_used,
                          coverage_ratio, labels}],
          "conflict_graph": {num_nodes, num_conflict_edges, num_duplicate_edges, ...},
        }
    Labels are independent booleans; age_seconds / conflict_ids / coverage_ratio
    are separate fields, not a fused score. A stale report is flagged
    potentially_outdated (needs verification) — never as false.
    """
    thresholds_obj = thresholds or DEFAULT_FRESHNESS_THRESHOLDS
    zone_type_lookup = zone_type_lookup or {}
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

    # Richer conflict provenance graph (P2): conflict + duplicate edges with
    # source provenance. Built once; per-zone views are derived below.
    graph = build_conflict_graph(as_of_reports, as_of)
    provenance_by_zone: Dict[str, List[Any]] = {}
    for edg in graph.conflict_edges:
        provenance_by_zone.setdefault(edg.zone_id, []).append({
            "reports": edg.reports,
            "variable": edg.variable,
            "mean": round(edg.mean, 6),
            "relative_deviation": round(edg.relative_deviation, 6),
            "observed_at": edg.observed_at,
        })

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
        zone_type = zone_type_lookup.get(zone_id)
        recent_s, outdated_s = thresholds_obj.for_variable(
            variable, zone_type, scenario_type)

        labels = _empty_labels()
        labels["recently_verified"] = latest_age <= recent_s
        labels["potentially_outdated"] = latest_age > outdated_s
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
            "conflict_provenance": provenance_by_zone.get(zone_id, []),
            "threshold_used": {"recent_sec": round(recent_s, 3),
                               "outdated_sec": round(outdated_s, 3)},
            "coverage_ratio": cov_ratio,
            "labels": labels,
        })

    # Zones with no reports at all are represented in coverage (if expected
    # coverage was supplied) and flagged insufficient at the overall level.
    if expected_coverage:
        for key in sorted(set(expected_coverage.keys()) - set(by_key_all.keys())):
            zone_id, variable = key
            zone_type = zone_type_lookup.get(zone_id)
            recent_s, outdated_s = thresholds_obj.for_variable(
                variable, zone_type, scenario_type)
            labels = _empty_labels()
            labels["insufficient_current_evidence"] = True
            zone_labels[key] = labels
            per_zone.append({
                "zone_id": zone_id,
                "variable": variable,
                "age_seconds": None,
                "independent_report_count": 0,
                "conflict_ids": [],
                "conflict_provenance": provenance_by_zone.get(zone_id, []),
                "threshold_used": {"recent_sec": round(recent_s, 3),
                                   "outdated_sec": round(outdated_s, 3)},
                "coverage_ratio": 0.0,
                "labels": labels,
            })

    per_report: List[Dict[str, Any]] = []
    for r in sorted(as_of_reports, key=lambda r: str(_field(r, "report_id"))):
        report_id = str(_field(r, "report_id"))
        key = (_field(r, "zone_id"), _field(r, "variable"))
        age = observation_age_seconds(r, as_of)
        zone_type = zone_type_lookup.get(key[0])
        recent_s, outdated_s = thresholds_obj.for_variable(
            _field(r, "variable"), zone_type, scenario_type)
        z_labels = zone_labels.get(key, _empty_labels())
        labels = {
            "recently_verified": age <= recent_s,
            # Age alone => needs verification, never 'false'.
            "potentially_outdated": age > outdated_s,
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
        "scenario_type": scenario_type,
        "thresholds": {var: [recent, outdated] for var, (recent, outdated)
                       in thresholds_obj.defaults().items()},
        "label_names": list(FRESHNESS_LABELS),
        "per_report": per_report,
        "per_zone": per_zone,
        "coverage_overall": coverage["overall"],
        "conflict_graph": graph.to_dict(),
    }
