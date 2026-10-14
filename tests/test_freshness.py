"""
Tests for evidence freshness / conflict / coverage:
- stale reports are flagged potentially outdated (needs verification), never false
- forwarded copies of one claim are not independent confirmations
- the five labels are independent and may co-occur
- fixed-expiry baseline behaves as a simple threshold
"""
import pytest
from datetime import datetime, timezone, timedelta

from crowd_reliability.reliability import (
    FRESHNESS_LABELS,
    FixedExpiryFreshnessBaseline,
    assess_information,
    compute_coverage,
    observation_age_seconds,
)

T_NOW = datetime(2026, 10, 14, 12, 5, 0, tzinfo=timezone.utc)


def _report(rid, value, age_s, zone="concourse", var="density", source="cam_1"):
    return {
        "report_id": rid,
        "scenario_id": "s1",
        "zone_id": zone,
        "source_id": source,
        "variable": var,
        "value": value,
        "observed_at": T_NOW - timedelta(seconds=age_s),
        "received_at": T_NOW - timedelta(seconds=max(0.0, age_s - 1)),
    }


def test_stale_report_flagged_outdated_never_false():
    out = assess_information([_report("r_old", 1.5, 100.0)], T_NOW)
    zone = out["per_zone"][0]

    assert zone["labels"]["potentially_outdated"] is True
    assert zone["labels"]["insufficient_current_evidence"] is True
    assert zone["labels"]["recently_verified"] is False

    # Age alone must never assert the report is false
    payload = str(out).lower()
    assert "false_report" not in payload and '"is_false"' not in payload
    assert not any(k in zone["labels"] for k in ("false", "incorrect", "invalid"))

    per_report = out["per_report"][0]
    assert per_report["labels"]["potentially_outdated"] is True
    assert per_report["age_seconds"] == pytest.approx(100.0)


def test_duplicate_copies_are_not_independent_confirmations():
    # Same claim forwarded by two channels: identical content + time.
    original = _report("r1", 2.0, 5.0, source="cam_1")
    forwarded = _report("r2", 2.0, 5.0, source="warden_relay")
    forwarded["received_at"] = T_NOW  # arrived later, still a copy

    out = assess_information([original, forwarded], T_NOW)
    zone = out["per_zone"][0]

    assert zone["independent_report_count"] == 1
    assert zone["labels"]["conflicting_evidence"] is False

    by_id = {r["report_id"]: r for r in out["per_report"]}
    dup_entries = [r for r in out["per_report"] if r["is_duplicate_of"] is not None]
    assert len(dup_entries) == 1
    assert dup_entries[0]["is_duplicate_of"] in by_id


def test_conflicting_independent_reports_flagged():
    a = _report("ra", 4.0, 5.0, source="cam_1")
    b = _report("rb", 0.5, 6.0, source="warden_patrol")
    out = assess_information([a, b], T_NOW)
    zone = out["per_zone"][0]

    assert zone["labels"]["conflicting_evidence"] is True
    assert len(zone["conflict_ids"]) > 0
    # Fresh + conflicting: multiple labels may apply simultaneously
    assert zone["labels"]["recently_verified"] is True


def test_labels_are_independent_and_all_five_present():
    out = assess_information([_report("r1", 2.0, 5.0)], T_NOW)
    for entry in out["per_report"] + out["per_zone"]:
        assert set(entry["labels"].keys()) == set(FRESHNESS_LABELS)
        assert all(isinstance(v, bool) for v in entry["labels"].values())


def test_insufficient_evidence_with_expected_coverage():
    expected = {("concourse", "density"): ["cam_1"], ("exit_1", "exit_open"): ["door_sensor"]}
    out = assess_information([], T_NOW, expected_coverage=expected)

    assert out["per_report"] == []
    assert len(out["per_zone"]) == 2
    for zone in out["per_zone"]:
        assert zone["labels"]["insufficient_current_evidence"] is True
        assert zone["coverage_ratio"] == 0.0
    assert out["coverage_overall"]["expected"] == 2
    assert out["coverage_overall"]["observed"] == 0


def test_rapidly_changing_label():
    reports = [
        _report("r1", 0.5, 10.0, source="cam_1"),
        _report("r2", 3.5, 20.0, source="cam_1"),
        _report("r3", 0.8, 30.0, source="cam_1"),
        _report("r4", 3.0, 40.0, source="cam_1"),
    ]
    out = assess_information(reports, T_NOW)
    zone = out["per_zone"][0]
    assert zone["labels"]["rapidly_changing"] is True


def test_fixed_expiry_freshness_baseline():
    baseline = FixedExpiryFreshnessBaseline(max_age_seconds=45.0)
    fresh = _report("r1", 1.0, 10.0)
    stale = _report("r2", 1.0, 60.0)
    assert baseline.is_fresh(fresh, T_NOW) is True
    assert baseline.is_fresh(stale, T_NOW) is False
    assert baseline.label(stale, T_NOW) == "expired"


def test_observation_age_uses_observed_at():
    report = _report("r1", 1.0, 30.0)
    report["received_at"] = T_NOW - timedelta(seconds=5)
    assert observation_age_seconds(report, T_NOW) == pytest.approx(30.0)


def test_coverage_observed_vs_expected():
    reports = [
        _report("r1", 2.0, 5.0, source="cam_1"),
        _report("r2", 2.1, 5.0, source="cam_2"),  # near-duplicate in time, different value
    ]
    expected = {("concourse", "density"): ["cam_1", "cam_2", "cam_3"]}
    cov = compute_coverage(reports, expected, T_NOW)

    entry = cov["per_zone_variable"][("concourse", "density")]
    assert entry["observed_count"] == 2
    assert entry["expected_count"] == 3
    assert entry["missing_sources"] == ["cam_3"]
    assert entry["coverage_ratio"] == pytest.approx(2 / 3)
