"""
test_evaluation_delay.py – Tests for label availability delay in evaluation.

Validates:
  - Reference labels with label_available_at > evaluation_time are skipped.
  - Labels available at or before evaluation_time are included.
  - n_label_wait and n_matched counts are accurate.
  - Exit classification metrics are computed separately.
  - MAE/RMSE are computed correctly on matched numeric pairs.
"""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timedelta, timezone

from person4.event_horizon_p4.evaluation import ReferenceEvaluator, EvaluationReport
from person4.event_horizon_p4.inference import ForecastRecord
from person4.event_horizon_p4.synthetic import MockReferenceState
from person4.event_horizon_p4.config import VAR_UNITS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)


def _fc(zone_id: str, var: str, target_at: datetime, value: float,
        issued_at: datetime = None, horizon_sec: int = 30) -> ForecastRecord:
    issued_at = issued_at or _BASE
    return ForecastRecord(
        forecast_id   = f"ev_{zone_id}_{var}",
        stream_id     = "stream_b_architecture",
        scenario_id   = "sc_eval",
        zone_id       = zone_id,
        variable      = var,
        unit          = VAR_UNITS.get(var, "p/m2"),
        issued_at     = issued_at,
        received_at   = issued_at,
        target_at     = target_at,
        value         = value,
        model_version = "test_v1",
        horizon_sec   = horizon_sec,
    )


def _ref(zone_id: str, var: str, state_at: datetime, value: float,
         label_available_at: datetime) -> MockReferenceState:
    return MockReferenceState(
        scenario_id       = "sc_eval",
        zone_id           = zone_id,
        variable          = var,
        value             = value,
        unit              = VAR_UNITS.get(var, "p/m2"),
        state_at          = state_at,
        label_available_at = label_available_at,
    )


_T30 = _BASE + timedelta(seconds=30)
_T60 = _BASE + timedelta(seconds=60)


# ---------------------------------------------------------------------------
# Label availability delay enforcement
# ---------------------------------------------------------------------------

class TestLabelAvailabilityDelay:
    def test_label_not_yet_available_is_skipped(self):
        """Label available 10 minutes in the future → not scored."""
        fc  = [_fc("zone_00", "density", _T30, 2.0)]
        ref = [_ref("zone_00", "density", _T30, 1.5,
                    label_available_at=_BASE + timedelta(minutes=10))]
        eval_time = _BASE + timedelta(minutes=1)

        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_time)
        assert report.n_matched    == 0
        assert report.n_label_wait == 1

    def test_label_just_available_is_scored(self):
        """Label available exactly at evaluation_time → scored."""
        avail_t   = _BASE + timedelta(minutes=2)
        eval_time = avail_t  # exactly equal

        fc  = [_fc("zone_00", "density", _T30, 2.0)]
        ref = [_ref("zone_00", "density", _T30, 1.5,
                    label_available_at=avail_t)]

        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_time)
        assert report.n_matched    == 1
        assert report.n_label_wait == 0

    def test_label_past_available_is_scored(self):
        avail_t   = _BASE + timedelta(seconds=5)
        eval_time = _BASE + timedelta(minutes=30)

        fc  = [_fc("zone_00", "density", _T30, 2.0)]
        ref = [_ref("zone_00", "density", _T30, 2.0,
                    label_available_at=avail_t)]

        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_time)
        assert report.n_matched == 1

    def test_none_evaluation_time_scores_all_labels(self):
        future_avail = _BASE + timedelta(days=365)
        fc  = [_fc("zone_00", "density", _T30, 2.0)]
        ref = [_ref("zone_00", "density", _T30, 2.0,
                    label_available_at=future_avail)]

        report = ReferenceEvaluator().score(fc, ref, evaluation_time=None)
        assert report.n_matched == 1

    def test_mixed_available_and_waiting(self):
        avail_t = _BASE + timedelta(seconds=5)
        far_t   = _BASE + timedelta(days=1)
        eval_t  = _BASE + timedelta(minutes=5)

        fc  = [
            _fc("zone_00", "density", _T30, 2.0),   # label available
            _fc("zone_00", "speed",   _T30, 1.0),   # label not yet available
        ]
        ref = [
            _ref("zone_00", "density", _T30, 1.5, label_available_at=avail_t),
            _ref("zone_00", "speed",   _T30, 1.2, label_available_at=far_t),
        ]

        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_t)
        assert report.n_matched    == 1
        assert report.n_label_wait == 1


# ---------------------------------------------------------------------------
# Numeric MAE / RMSE correctness
# ---------------------------------------------------------------------------

class TestNumericMetricsCorrectness:
    def test_perfect_prediction_mae_is_zero(self):
        avail = _BASE
        fc  = [_fc("zone_00", "density", _T30, 2.0)]
        ref = [_ref("zone_00", "density", _T30, 2.0, label_available_at=avail)]
        report = ReferenceEvaluator().score(fc, ref, evaluation_time=_BASE + timedelta(seconds=60))
        m = next(m for m in report.numeric if m.variable == "density")
        assert m.mae  == pytest.approx(0.0, abs=1e-5)
        assert m.rmse == pytest.approx(0.0, abs=1e-5)

    def test_known_mae_rmse(self):
        avail = _BASE
        eval_t = _BASE + timedelta(minutes=5)
        errors = [1.0, 2.0, 3.0]   # MAE = 2.0, RMSE = sqrt(14/3) ≈ 2.160

        fc  = [_fc("zone_00", "density", _T30 + timedelta(seconds=i), e + 1.0)
               for i, e in enumerate(errors)]
        ref = [_ref("zone_00", "density", _T30 + timedelta(seconds=i), 1.0,
                    label_available_at=avail)
               for i in range(len(errors))]

        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_t)
        m = next(m for m in report.numeric if m.variable == "density")
        expected_mae  = sum(abs(e) for e in errors) / len(errors)
        expected_rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        assert m.mae  == pytest.approx(expected_mae,  abs=1e-4)
        assert m.rmse == pytest.approx(expected_rmse, abs=1e-4)

    def test_no_label_match_reports_n_no_label(self):
        fc  = [_fc("zone_00", "density", _T30, 2.0)]
        ref = []   # empty — no label
        report = ReferenceEvaluator().score(fc, ref, evaluation_time=None)
        assert report.n_no_label == 1
        assert report.n_matched  == 0


# ---------------------------------------------------------------------------
# Exit classification metrics
# ---------------------------------------------------------------------------

class TestExitClassificationMetrics:
    def test_perfect_exit_prediction_accuracy_is_one(self):
        avail  = _BASE
        eval_t = _BASE + timedelta(minutes=5)
        # Predict prob=1.0, true=1.0 → perfect
        fc  = [_fc("zone_04", "exit_open", _T30, 1.0)]
        ref = [_ref("zone_04", "exit_open", _T30, 1.0, label_available_at=avail)]
        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_t)
        e = next(m for m in report.exit_cls if m.horizon_sec == 30)
        assert e.accuracy    == pytest.approx(1.0, abs=1e-5)
        assert e.brier_score == pytest.approx(0.0, abs=1e-5)

    def test_wrong_exit_prediction_accuracy_is_zero(self):
        avail  = _BASE
        eval_t = _BASE + timedelta(minutes=5)
        # Predict prob=0.9 → binary=1, true=0 → wrong
        fc  = [_fc("zone_04", "exit_open", _T30, 0.9)]
        ref = [_ref("zone_04", "exit_open", _T30, 0.0, label_available_at=avail)]
        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_t)
        e = next(m for m in report.exit_cls if m.horizon_sec == 30)
        assert e.accuracy == pytest.approx(0.0, abs=1e-5)

    def test_exit_and_numeric_reported_separately(self):
        avail  = _BASE
        eval_t = _BASE + timedelta(minutes=5)
        fc  = [
            _fc("zone_00", "density",   _T30, 2.0),
            _fc("zone_04", "exit_open", _T30, 0.8),
        ]
        ref = [
            _ref("zone_00", "density",   _T30, 2.0, label_available_at=avail),
            _ref("zone_04", "exit_open", _T30, 1.0, label_available_at=avail),
        ]
        report = ReferenceEvaluator().score(fc, ref, evaluation_time=eval_t)
        # Numeric and exit reported separately
        assert len(report.numeric)  >= 1
        assert len(report.exit_cls) >= 1
        num_vars = [m.variable for m in report.numeric]
        # density belongs in numeric metrics
        assert "density" in num_vars
        # exit_open must NOT appear in numeric metrics (separate exit_cls bucket)
        assert "exit_open" not in num_vars
        # exit_cls contains ExitMetrics with horizon_sec (not a .variable field)
        assert all(hasattr(m, "horizon_sec") for m in report.exit_cls)
        assert all(hasattr(m, "brier_score") for m in report.exit_cls)
