"""
Evidence coverage: observed vs expected relevant sources/variables.

Coverage is reported as its own field — it is never fused with freshness or
conflict into a synthetic single number.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from crowd_reliability.reliability.freshness import _field, _as_naive_utc


def compute_coverage(
    reports: List[Any],
    expected: Optional[Dict[Tuple[str, str], List[str]]],
    as_of: datetime,
    window_s: float = 90.0,
) -> Dict[str, Any]:
    """
    Args:
        reports: independent (deduplicated) reports to count.
        expected: {(zone_id, variable): [expected source_ids]}. If None, only
            the observed side is reported and ratios are None.
        window_s: a report counts as covering only if observed within this window.

    Returns:
        {"per_zone_variable": {(zone, var): {...}}, "overall": {...}}
        Ratios are separate fields; nothing is combined with freshness labels.
    """
    as_of_naive = _as_naive_utc(as_of)
    observed: Dict[Tuple[str, str], List[str]] = {}
    for r in reports:
        observed_at = _field(r, "observed_at") or _field(r, "received_at")
        if observed_at is None:
            continue
        age = (as_of_naive - _as_naive_utc(observed_at)).total_seconds()
        if 0 <= age <= window_s:
            key = (_field(r, "zone_id"), _field(r, "variable"))
            observed.setdefault(key, [])
            src = _field(r, "source_id")
            if src and src not in observed[key]:
                observed[key].append(src)

    per_zone_variable: Dict[Tuple[str, str], Dict[str, Any]] = {}
    keys = set(observed.keys())
    if expected:
        keys |= set(expected.keys())

    total_observed = 0
    total_expected = 0
    for key in sorted(keys):
        zone_id, variable = key
        exp_sources = list(expected.get(key, [])) if expected else []
        obs_sources = observed.get(key, [])
        covered = [s for s in exp_sources if s in obs_sources] if expected else obs_sources
        missing_sources = [s for s in exp_sources if s not in obs_sources]
        total_observed += len(covered) if expected else len(obs_sources)
        total_expected += len(exp_sources)
        per_zone_variable[key] = {
            "zone_id": zone_id,
            "variable": variable,
            "observed_sources": sorted(obs_sources),
            "expected_sources": sorted(exp_sources),
            "missing_sources": sorted(missing_sources),
            "observed_count": len(covered) if expected else len(obs_sources),
            "expected_count": len(exp_sources) if expected else None,
            "coverage_ratio": (len(covered) / len(exp_sources)) if (expected and exp_sources) else None,
        }

    return {
        "per_zone_variable": per_zone_variable,
        "overall": {
            "observed": total_observed,
            "expected": total_expected if expected else None,
            "coverage_ratio": (total_observed / total_expected) if (expected and total_expected) else None,
        },
    }
