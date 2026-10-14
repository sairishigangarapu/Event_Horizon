"""
P2 coverage: explainable scenario/zone-dependent freshness thresholds and the
richer conflict provenance graph. These are the stretch deliverables layered on
top of the P0/P1 reliability core.
"""
import pytest
from datetime import datetime, timezone, timedelta

from crowd_reliability.reliability import (
    FRESHNESS_LABELS,
    assess_information,
)
from crowd_reliability.reliability.conflicts import (
    build_conflict_graph,
    deduplicate_reports,
)
from crowd_reliability.reliability.thresholds import (
    DEFAULT_FRESHNESS_THRESHOLDS,
    DEFAULT_THRESHOLDS_SEC,
)

T_NOW = datetime(2026, 10, 14, 12, 5, 0, tzinfo=timezone.utc)


def _report(rid, value, age_s, zone="concourse", var="density", source="cam_1"):
    return {
        "report_id": rid, "scenario_id": "s1", "zone_id": zone, "source_id": source,
        "variable": var, "value": value,
        "observed_at": T_NOW - timedelta(seconds=age_s),
        "published_at": T_NOW - timedelta(seconds=max(0.0, age_s - 1)),
        "received_at": T_NOW - timedelta(seconds=max(0.0, age_s - 1)),
    }


# ---------- explainable thresholds ----------

def test_thresholds_order_by_physical_timescale():
    """Event-driven states stale faster; bulk density is most tolerant."""
    order = ["exit_open", "inflow", "speed", "queue_length", "density"]
    outdated = [DEFAULT_THRESHOLDS_SEC[v][1] for v in order]
    # Monotonically non-decreasing: exit_open < inflow < speed < queue < density
    for a, b in zip(outdated, outdated[1:]):
        assert a <= b
    assert outdated[-1] == DEFAULT_THRESHOLDS_SEC["density"][1]  # slowest to go stale


def test_zone_type_and_scenario_multipliers_widen_narrow_windows():
    # exit zones require fresher observations (multiplier < 1)
    base_recent, _ = DEFAULT_FRESHNESS_THRESHOLDS.for_variable("density")
    exit_recent, _ = DEFAULT_FRESHNESS_THRESHOLDS.for_variable("density", zone_type="exit")
    assert exit_recent < base_recent
    # high-confidence scenario widens the fresh window (multiplier > 1)
    assert DEFAULT_FRESHNESS_THRESHOLDS.for_variable("density", scenario_type="normal_flow") != \
        DEFAULT_FRESHNESS_THRESHOLDS.for_variable("density", scenario_type="sudden_stop")


def test_stale_exit_open_flagged_at_15s_not_density_15s():
    # exit_open outdated boundary is 15s; density is 45s. At age 20s, exit_open is
    # outdated but density is still acceptable.
    out_exit = assess_information([_report("e1", 1.0, 20.0, var="exit_open")], T_NOW)
    out_den = assess_information([_report("d1", 1.0, 20.0, var="density")], T_NOW)
    exit_labels = out_exit["per_zone"][0]["labels"]
    den_labels = out_den["per_zone"][0]["labels"]
    assert exit_labels["potentially_outdated"] is True
    assert den_labels["potentially_outdated"] is False


def test_assess_emits_thresholds_and_threshold_used():
    out = assess_information([_report("r1", 1.0, 5.0)], T_NOW)
    assert "thresholds" in out
    assert set(out["thresholds"].keys()) == set(DEFAULT_THRESHOLDS_SEC.keys())
    assert all(len(v) == 2 for v in out["thresholds"].values())
    zone = out["per_zone"][0]
    assert "threshold_used" in zone
    assert set(zone["threshold_used"].keys()) == {"recent_sec", "outdated_sec"}


# ---------- conflict provenance graph ----------

def test_conflict_graph_captures_duplicate_and_conflict_edges():
    original = _report("orig", 2.0, 5.0, source="cam_1")
    forward = _report("fwd", 2.0, 5.0, source="warden_relay")  # forwarded copy
    a = _report("ra", 4.0, 6.0, source="cam_1")      # independent
    b = _report("rb", 0.5, 6.0, source="warden_patrol")  # independent, disagrees

    graph = build_conflict_graph([original, forward, a, b], T_NOW)

    assert set(graph.nodes) == {"orig", "fwd", "ra", "rb"}
    # forwarded copy collapsed -> one duplicate edge, NOT a conflict edge
    assert len(graph.duplicate_edges) == 1
    assert graph.duplicate_edges[0].kept in ("orig", "fwd")
    # ra and rb disagree and are independent -> one conflict edge
    assert len(graph.conflict_edges) == 1
    edge = graph.conflict_edges[0]
    assert set(edge.reports) == {"ra", "rb"}
    # source provenance: both disagreeing sources are recorded
    assert {"cam_1", "warden_patrol"} <= set(
        graph.source_provenance[edge.reports[0]]) | set(graph.source_provenance[edge.reports[1]])
    assert graph.to_dict()["num_duplicate_edges"] == 1
    assert graph.to_dict()["num_conflict_edges"] == 1


def test_two_copies_never_become_independent_confirmations():
    original = _report("orig", 2.0, 5.0, source="cam_1")
    forward = _report("fwd", 2.0, 5.0, source="warden_relay")
    graph = build_conflict_graph([original, forward], T_NOW)
    assert graph.conflict_edges == []  # identical copies never conflict
    independent, _, _ = deduplicate_reports([original, forward])
    assert len(independent) == 1


def test_assess_includes_conflict_provenance_per_zone():
    a = _report("ra", 4.0, 5.0, zone="exit_1", source="cam_1", var="speed")
    b = _report("rb", 0.5, 6.0, zone="exit_1", source="warden", var="speed")
    out = assess_information([a, b], T_NOW)
    zone = out["per_zone"][0]
    assert zone["labels"]["conflicting_evidence"] is True
    assert zone["conflict_provenance"]
    assert zone["conflict_provenance"][0]["variable"] == "speed"
    assert out["conflict_graph"]["num_conflict_edges"] >= 1
