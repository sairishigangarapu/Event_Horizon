"""
test_forecast_alignment.py – Tests for Stream A/B comparison alignment.

Validates:
  - Forecasts pair correctly on (scenario_id, zone_id, variable, target_at).
  - Missing Stream A channels are labelled 'unavailable'.
  - Unit mismatches are skipped and flagged.
  - Causal filter (issued_at / received_at <= evaluation_time) is enforced.
  - Signed and absolute differences are both reported; no merged trust score.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

try:
    from person4.event_horizon_p4.comparison import ForecastComparator, summarise_comparison
    from person4.event_horizon_p4.config import STREAM_A_ID, STREAM_B_ID, VAR_UNITS
    from person4.event_horizon_p4.inference import ForecastRecord
except ImportError:
    from event_horizon_p4.comparison import ForecastComparator, summarise_comparison
    from event_horizon_p4.config import STREAM_A_ID, STREAM_B_ID, VAR_UNITS
    from event_horizon_p4.inference import ForecastRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    zone_id:    str,
    variable:   str,
    target_at:  datetime,
    value:      float,
    stream_id:  str,
    issued_at:  datetime = None,
    unit:       str      = None,
    horizon_sec: int = 30,
) -> ForecastRecord:
    issued_at = issued_at or datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    unit = unit or VAR_UNITS.get(variable, "p/m2")
    return ForecastRecord(
        forecast_id   = f"test_{zone_id}_{variable}",
        stream_id     = stream_id,
        scenario_id   = "sc_test",
        zone_id       = zone_id,
        variable      = variable,
        unit          = unit,
        issued_at     = issued_at,
        received_at   = issued_at,
        target_at     = target_at,
        value         = value,
        model_version = "test_v1",
        horizon_sec   = horizon_sec,
    )


_T0  = datetime(2026, 10, 1, 8, 0, 30, tzinfo=timezone.utc)
_NOW = datetime(2026, 10, 1, 8, 5, 0,  tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Basic alignment
# ---------------------------------------------------------------------------

class TestForecastAlignment:
    def test_matched_pair_produces_differences(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID)]
        a_recs = [_make_record("zone_00", "density", _T0, 1.5, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 1
        diff = report.paired[0]
        assert diff.signed_diff   == pytest.approx(0.5, abs=1e-5)
        assert diff.absolute_diff == pytest.approx(0.5, abs=1e-5)

    def test_negative_signed_diff_when_b_lower(self):
        b_recs = [_make_record("zone_00", "density", _T0, 1.0, STREAM_B_ID)]
        a_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert report.paired[0].signed_diff   == pytest.approx(-1.0, abs=1e-5)
        assert report.paired[0].absolute_diff == pytest.approx(1.0,  abs=1e-5)

    def test_different_variables_do_not_pair(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID)]
        a_recs = [_make_record("zone_00", "speed",   _T0, 1.5, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 0

    def test_different_zones_do_not_pair(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID)]
        a_recs = [_make_record("zone_01", "density", _T0, 1.5, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 0

    def test_different_target_times_do_not_pair(self):
        t1 = _T0
        t2 = _T0 + timedelta(seconds=30)
        b_recs = [_make_record("zone_00", "density", t1, 2.0, STREAM_B_ID)]
        a_recs = [_make_record("zone_00", "density", t2, 1.5, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 0

    def test_multiple_zones_all_paired(self):
        b_recs = [_make_record(f"zone_0{i}", "density", _T0, float(i), STREAM_B_ID)
                  for i in range(4)]
        a_recs = [_make_record(f"zone_0{i}", "density", _T0, float(i) + 0.1, STREAM_A_ID)
                  for i in range(4)]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 4


# ---------------------------------------------------------------------------
# Unavailable Stream A channels
# ---------------------------------------------------------------------------

class TestUnavailableChannels:
    def test_missing_stream_a_channel_labelled_unavailable(self):
        b_recs = [
            _make_record("zone_00", "density",  _T0, 2.0, STREAM_B_ID),
            _make_record("zone_00", "speed",    _T0, 1.0, STREAM_B_ID),
        ]
        # Stream A only provides density
        a_recs = [_make_record("zone_00", "density", _T0, 1.8, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs)

        assert len(report.paired) == 1
        assert ("zone_00", "speed") in report.unavailable
        assert report.unavailable[("zone_00", "speed")] == "unavailable"

    def test_all_a_missing_all_labelled_unavailable(self):
        b_recs = [_make_record("zone_00", var, _T0, 1.0, STREAM_B_ID)
                  for var in ["density", "speed", "inflow"]]
        report = ForecastComparator().compare(b_recs, [])
        assert len(report.unavailable) == 3
        assert len(report.paired) == 0

    def test_summary_shows_unavailable_for_missing_vars(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID)]
        report = ForecastComparator().compare(b_recs, [])
        summary = summarise_comparison(report)
        # density should show unavailable since it has no pairs
        assert summary.get("density", {}).get("status") == "unavailable"


# ---------------------------------------------------------------------------
# Unit mismatch
# ---------------------------------------------------------------------------

class TestUnitMismatch:
    def test_unit_mismatch_skipped_not_paired(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID, unit="p/m2")]
        a_recs = [_make_record("zone_00", "density", _T0, 0.5, STREAM_A_ID, unit="persons/m2_WRONG")]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 0
        assert len(report.unit_skipped) == 1
        assert report.unit_skipped[0][2] == "density"

    def test_unit_match_pairs_correctly(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID, unit="p/m2")]
        a_recs = [_make_record("zone_00", "density", _T0, 1.5, STREAM_A_ID, unit="p/m2")]
        report = ForecastComparator().compare(b_recs, a_recs)
        assert len(report.paired) == 1
        assert len(report.unit_skipped) == 0


# ---------------------------------------------------------------------------
# Causal timestamp filter
# ---------------------------------------------------------------------------

class TestCausalFilter:
    def test_future_issued_at_filtered_out(self):
        future_time = _NOW + timedelta(hours=1)
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID,
                               issued_at=future_time)]
        a_recs = [_make_record("zone_00", "density", _T0, 1.5, STREAM_A_ID)]
        report = ForecastComparator().compare(b_recs, a_recs, evaluation_time=_NOW)
        assert len(report.paired) == 0

    def test_past_issued_at_passes_filter(self):
        past_time = _NOW - timedelta(hours=1)
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID,
                               issued_at=past_time)]
        a_recs = [_make_record("zone_00", "density", _T0, 1.5, STREAM_A_ID,
                               issued_at=past_time)]
        report = ForecastComparator().compare(b_recs, a_recs, evaluation_time=_NOW)
        assert len(report.paired) == 1

    def test_evaluation_time_none_includes_all(self):
        future_time = _NOW + timedelta(hours=100)
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID,
                               issued_at=future_time)]
        a_recs = [_make_record("zone_00", "density", _T0, 1.5, STREAM_A_ID,
                               issued_at=future_time)]
        # None evaluation_time → no filter applied
        report = ForecastComparator().compare(b_recs, a_recs, evaluation_time=None)
        assert len(report.paired) == 1


# ---------------------------------------------------------------------------
# No merged trust probability
# ---------------------------------------------------------------------------

class TestNoMergedTrustProbability:
    def test_summary_has_signed_and_absolute_not_just_trust(self):
        b_recs = [_make_record("zone_00", "density", _T0, 2.0, STREAM_B_ID)]
        a_recs = [_make_record("zone_00", "density", _T0, 1.5, STREAM_A_ID)]
        report  = ForecastComparator().compare(b_recs, a_recs)
        summary = summarise_comparison(report)
        d = summary.get("density", {})
        assert "mean_signed_diff" in d, "signed_diff must be reported"
        assert "mean_abs_diff" in d,    "absolute_diff must be reported"
        assert "trust_probability" not in d, (
            "A merged trust_probability must NEVER be the sole comparison result"
        )
