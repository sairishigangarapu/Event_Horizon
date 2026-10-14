"""
Comparable-report grouping, duplicate/copy collapse, and disagreement checks.

Two forwarded copies of one claim must NOT count as independent confirmations.
Reports are collapsed by content identity (zone, variable, rounded value,
5-second-normalized observed_at): identical content at the same time is the
same claim regardless of how many channels forwarded it. When source metadata
is present it is recorded per group; when absent, dedup is flagged as
'content_hash_only'.
"""
from dataclasses import dataclass, field
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


@dataclass
class ConflictEdge:
    """An edge in the conflict provenance graph: two reports disagree."""

    reports: List[str]
    zone_id: str
    variable: str
    mean: float
    deviation: float
    relative_deviation: float
    observed_at: Optional[str] = None


@dataclass
class DuplicateEdge:
    """A dedup edge: one report is a forwarded copy of another."""

    kept: str
    duplicate: str
    zone_id: str
    variable: str
    source_id: Optional[str] = None
    dedup_method: str = "content_hash"


@dataclass
class ConflictGraph:
    """Richer provenance for disagreement and duplication.

    Nodes are reports; edges are conflicts (disagreement) or duplicates
    (forwarded copies of the same claim). Source provenance is attached to
    every edge so downstream consumers can ask *which sources* disagree.
    """

    nodes: List[str] = field(default_factory=list)
    conflict_edges: List[ConflictEdge] = field(default_factory=list)
    duplicate_edges: List[DuplicateEdge] = field(default_factory=list)
    source_provenance: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_conflict_edges": len(self.conflict_edges),
            "num_duplicate_edges": len(self.duplicate_edges),
            "conflict_edges": [
                {
                    "reports": e.reports,
                    "zone_id": e.zone_id,
                    "variable": e.variable,
                    "mean": e.mean,
                    "deviation": e.deviation,
                    "relative_deviation": e.relative_deviation,
                    "observed_at": e.observed_at,
                    "sources": self.source_provenance.get(e.reports[0], []),
                }
                for e in self.conflict_edges
            ],
            "duplicate_edges": [
                {
                    "kept": e.kept,
                    "duplicate": e.duplicate,
                    "zone_id": e.zone_id,
                    "variable": e.variable,
                    "source_id": e.source_id,
                    "method": e.dedup_method,
                }
                for e in self.duplicate_edges
            ],
        }


def build_conflict_graph(
    reports: List[Any],
    as_of: Optional[datetime] = None,
    window_s: float = CONFLICT_WINDOW_S,
    rel_deviation: float = CONFLICT_REL_DEVIATION,
) -> ConflictGraph:
    """Assemble the conflict + duplication provenance graph for a report set.

    Conflict edges are computed over the comparable (same zone/variable/time)
    groups; duplicate edges come from :func:`deduplicate_reports`. No edge is
    created between two forwarded copies of the same claim (they are collapsed
    first), so copies never count as independent confirmations.
    """
    graph = ConflictGraph(nodes=[str(_field(r, "report_id")) for r in reports])

    # Duplicate / forwarded-copy edges (collapsed first so copies don't
    # participate in conflict comparisons).
    independent, duplicates, _ = deduplicate_reports(reports)
    seen: Dict[str, str] = {}  # report_id -> kept id
    for kept, dups in duplicates.items():
        seen[kept] = kept
        for d in dups:
            seen[d] = kept
    for kept, dups in duplicates.items():
        for d in dups:
            rep_k = next((r for r in reports if str(_field(r, "report_id")) == kept), None)
            rep_d = next((r for r in reports if str(_field(r, "report_id")) == d), None)
            graph.duplicate_edges.append(DuplicateEdge(
                kept=kept,
                duplicate=d,
                zone_id=str(_field(rep_k, "zone_id")) if rep_k else "",
                variable=str(_field(rep_k, "variable")) if rep_k else "",
                source_id=str(_field(rep_k, "source_id")) if rep_k else None,
                dedup_method="content_hash" if rep_k else "content_hash",
            ))
            graph.source_provenance.setdefault(kept, [])
            if rep_k:
                graph.source_provenance[kept].append(str(_field(rep_k, "source_id")))

    # Conflict edges over independent (de-duplicated) comparable groups.
    groups = group_comparable(independent, as_of, window_s=window_s)
    for key, group in groups.items():
        if len(group) < 2:
            continue
        is_conflict, conflict_ids = detect_conflicts(group)
        if not is_conflict:
            continue
        values = [float(_field(r, "value")) for r in group]
        mean_val = sum(values) / len(values)
        worst = max(group, key=lambda r: abs(float(_field(r, "value")) - mean_val))
        rel = abs(float(_field(worst, "value")) - mean_val) / (mean_val + 1e-4)
        sources = [str(_field(r, "source_id")) for r in group]
        for r in conflict_ids:
            graph.source_provenance.setdefault(r, sources)
        oat = _field(worst, "observed_at")
        graph.conflict_edges.append(ConflictEdge(
            reports=sorted(conflict_ids),
            zone_id=str(_field(worst, "zone_id")),
            variable=str(_field(worst, "variable")),
            mean=mean_val,
            deviation=abs(float(_field(worst, "value")) - mean_val),
            relative_deviation=rel,
            observed_at=_as_naive_utc(oat).isoformat() if oat is not None else None,
        ))
    return graph
