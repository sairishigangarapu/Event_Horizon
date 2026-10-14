"""
evaluation.py – Delayed reference scoring: MAE/RMSE + exit classification metrics.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

Contract:
  - Reference metrics are calculated ONLY after label_available_at time.
  - Forecasts are matched to ReferenceState on (scenario_id, zone_id,
    variable, state_at == target_at).
  - Separate metrics for numeric regression (MAE, RMSE) and exit classification
    (accuracy, log-loss / Brier score) are reported.
  - Metrics are grouped by (variable, horizon_sec).
  - No future labels are used at issue time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import HORIZONS_SEC, VAR_NAMES, VAR_UNITS
from .inference import ForecastRecord
from .synthetic import MockReferenceState


# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NumericMetrics:
    """MAE and RMSE for one (variable, horizon) bucket."""
    variable:    str
    horizon_sec: int
    n:           int
    mae:         float
    rmse:        float

    def to_dict(self) -> Dict:
        return {
            "variable":    self.variable,
            "horizon_sec": self.horizon_sec,
            "n":           self.n,
            "mae":         self.mae,
            "rmse":        self.rmse,
        }


@dataclass
class ExitMetrics:
    """Accuracy and Brier score for exit_open channel."""
    horizon_sec:  int
    n:            int
    accuracy:     float   # fraction of predictions correctly binarised
    brier_score:  float   # mean squared probability error

    def to_dict(self) -> Dict:
        return {
            "variable":    "exit_open",
            "horizon_sec": self.horizon_sec,
            "n":           self.n,
            "accuracy":    self.accuracy,
            "brier_score": self.brier_score,
        }


@dataclass
class EvaluationReport:
    """Full evaluation result for a set of forecasts and reference labels."""
    numeric:      List[NumericMetrics] = field(default_factory=list)
    exit_cls:     List[ExitMetrics]    = field(default_factory=list)
    n_matched:    int = 0
    n_label_wait: int = 0   # records skipped because label not yet available
    n_no_label:   int = 0   # records with no matching reference at all

    def to_dict(self) -> Dict:
        return {
            "numeric":      [m.to_dict() for m in self.numeric],
            "exit_cls":     [m.to_dict() for m in self.exit_cls],
            "n_matched":    self.n_matched,
            "n_label_wait": self.n_label_wait,
            "n_no_label":   self.n_no_label,
        }


# ---------------------------------------------------------------------------
# Reference lookup helper
# ---------------------------------------------------------------------------

def _as_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def build_reference_lookup(
    reference_states: List[MockReferenceState],
) -> Dict[Tuple[str, str, str, datetime], MockReferenceState]:
    """
    Builds a lookup dictionary keyed by:
        (scenario_id, zone_id, variable, state_at)

    This allows O(1) matching from ForecastRecord.target_at == state_at.
    """
    lookup: Dict[Tuple[str, str, str, datetime], MockReferenceState] = {}
    for ref in reference_states:
        key = (
            ref.scenario_id,
            ref.zone_id,
            ref.variable,
            _as_naive(ref.state_at),
        )
        lookup[key] = ref
    return lookup


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ReferenceEvaluator:
    """
    Matches forecast records to reference labels and computes metrics.

    Numeric channels: MAE and RMSE per (variable, horizon_sec).
    Exit channel:     Accuracy and Brier score per horizon_sec.
    """

    def score(
        self,
        forecasts:        List[ForecastRecord],
        reference_states: List[MockReferenceState],
        evaluation_time:  Optional[datetime] = None,
    ) -> EvaluationReport:
        """
        Scores forecasts against reference labels.

        Args:
            forecasts:        ForecastRecord list produced by inference.
            reference_states: MockReferenceState list (labels).
            evaluation_time:  Current wall-clock time.  Labels with
                              label_available_at > evaluation_time are skipped.
                              If None, all labels are considered available.

        Returns:
            EvaluationReport with separate numeric and exit metrics.
        """
        ref_lookup = build_reference_lookup(reference_states)
        et = _as_naive(evaluation_time) if evaluation_time else None

        # Accumulate per (variable, horizon_sec)
        # For numeric: lists of (pred, true)
        # For exit:    lists of (prob, binary_true)
        numeric_buckets: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
        exit_buckets:    Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        n_matched    = 0
        n_label_wait = 0
        n_no_label   = 0

        for fc in forecasts:
            key = (
                fc.scenario_id,
                fc.zone_id,
                fc.variable,
                _as_naive(fc.target_at),
            )
            ref = ref_lookup.get(key)

            if ref is None:
                n_no_label += 1
                continue

            # Enforce label availability delay
            ref_avail = _as_naive(ref.label_available_at)
            if et is not None and ref_avail > et:
                n_label_wait += 1
                continue

            n_matched += 1

            if fc.variable == "exit_open":
                exit_buckets[fc.horizon_sec].append((fc.value, ref.value))
            else:
                numeric_buckets[(fc.variable, fc.horizon_sec)].append(
                    (fc.value, ref.value)
                )

        # Compute numeric metrics
        numeric_metrics: List[NumericMetrics] = []
        for (var, h_sec), pairs in sorted(numeric_buckets.items()):
            preds = np.array([p[0] for p in pairs], dtype=np.float64)
            trues = np.array([p[1] for p in pairs], dtype=np.float64)
            mae  = float(np.mean(np.abs(preds - trues)))
            rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
            numeric_metrics.append(NumericMetrics(
                variable    = var,
                horizon_sec = h_sec,
                n           = len(pairs),
                mae         = round(mae, 5),
                rmse        = round(rmse, 5),
            ))

        # Compute exit classification metrics
        exit_metrics: List[ExitMetrics] = []
        for h_sec, pairs in sorted(exit_buckets.items()):
            probs   = np.array([p[0] for p in pairs], dtype=np.float64)
            trues_b = np.array([p[1] for p in pairs], dtype=np.float64)
            binary_preds = (probs >= 0.5).astype(float)
            accuracy    = float(np.mean(binary_preds == trues_b))
            brier       = float(np.mean((probs - trues_b) ** 2))
            exit_metrics.append(ExitMetrics(
                horizon_sec = h_sec,
                n           = len(pairs),
                accuracy    = round(accuracy, 5),
                brier_score = round(brier, 5),
            ))

        return EvaluationReport(
            numeric      = numeric_metrics,
            exit_cls     = exit_metrics,
            n_matched    = n_matched,
            n_label_wait = n_label_wait,
            n_no_label   = n_no_label,
        )


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------

def export_metrics_csv(report: EvaluationReport, filepath: str) -> None:
    """
    Exports numeric and exit metrics to a CSV file.

    Columns: metric_type, variable, horizon_sec, n, mae, rmse,
             accuracy, brier_score.
    """
    import csv
    import os

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    rows = []
    for m in report.numeric:
        rows.append({
            "metric_type": "numeric",
            "variable":    m.variable,
            "horizon_sec": m.horizon_sec,
            "n":           m.n,
            "mae":         m.mae,
            "rmse":        m.rmse,
            "accuracy":    "",
            "brier_score": "",
        })
    for m in report.exit_cls:
        rows.append({
            "metric_type": "exit_classification",
            "variable":    "exit_open",
            "horizon_sec": m.horizon_sec,
            "n":           m.n,
            "mae":         "",
            "rmse":        "",
            "accuracy":    m.accuracy,
            "brier_score": m.brier_score,
        })

    fieldnames = ["metric_type", "variable", "horizon_sec", "n",
                  "mae", "rmse", "accuracy", "brier_score"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[evaluation] Metrics exported to {filepath}")
