"""
baseline.py – Persistence and moving-average forecasting baselines.

NOTE: Persistence and moving-average forecasting baselines for Person 4 comparisons.

Baselines:
  1. PersistenceBaseline     – last observed value repeated for all horizons.
  2. MovingAverageBaseline   – simple unweighted mean of the last N steps per zone.

These baselines are used for fair comparison against the GRU model output.
They consume the same WindowBatch input so comparisons are apple-to-apple.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

from .config import (
    HORIZONS_SEC,
    NUM_NUMERIC_CHANNELS,
    STREAM_B_ID,
    U_DIM,
    VAR_NAMES,
    VAR_UNITS,
)
from .inference import ForecastRecord
from .sequence import WindowBatch
from .synthetic import is_exit_zone


# ---------------------------------------------------------------------------
# X vector slice helpers
# ---------------------------------------------------------------------------

#: Slice indices of X (raw observation values) within U vector.
_X_START = 0
_X_END   = 6   # NUM_CHANNELS = 6


def _extract_x_from_u(u_history: np.ndarray) -> np.ndarray:
    """
    Extracts the X (raw observation) slice from u_history.

    Args:
        u_history: (HISTORY_STEPS, U_DIM)

    Returns:
        x_history: (HISTORY_STEPS, 6)
    """
    return u_history[:, _X_START:_X_END]


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

class PersistenceBaseline:
    """
    Persistence forecast: repeats the last observed value for all horizons.

    This is the simplest possible baseline.  For exit_open (binary), the last
    value is rounded to 0 or 1.  Non-exit zones do not emit exit_open.

    The 'stream_id' is always 'baseline_persistence' to distinguish from
    Stream A/B.
    """

    STREAM_ID = "baseline_persistence"

    def forecast(
        self,
        batch:       WindowBatch,
        scenario_id: str,
    ) -> List[ForecastRecord]:
        """
        Produces persistence forecasts for all zones and horizons.

        Uses the last step of u_history ([:, -1, :]) as the current observation.
        """
        records: List[ForecastRecord] = []
        issue_time = batch.issue_time

        for z_idx, zone_id in enumerate(batch.zone_ids):
            last_u = batch.u_history[z_idx, -1, :]   # (U_DIM,)
            x_last = last_u[_X_START:_X_END]          # (6,)
            _is_exit = is_exit_zone(zone_id)

            for h_sec, target_at in batch.target_times.items():
                for v_idx in range(NUM_NUMERIC_CHANNELS):
                    var = VAR_NAMES[v_idx]
                    val = float(max(0.0, x_last[v_idx]))
                    records.append(ForecastRecord(
                        forecast_id   = f"pb_{uuid.uuid4().hex[:10]}",
                        stream_id     = self.STREAM_ID,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = round(val, 5),
                        model_version = "persistence_v1",
                        horizon_sec   = h_sec,
                    ))

                if _is_exit:
                    var = "exit_open"
                    val = round(float(x_last[5]))  # binary persistence
                    records.append(ForecastRecord(
                        forecast_id   = f"pb_{uuid.uuid4().hex[:10]}",
                        stream_id     = self.STREAM_ID,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = val,
                        model_version = "persistence_v1",
                        horizon_sec   = h_sec,
                    ))

        return records


# ---------------------------------------------------------------------------
# Moving-average baseline
# ---------------------------------------------------------------------------

class MovingAverageBaseline:
    """
    Simple unweighted moving-average forecast over the last `window` steps.

    Useful for distinguishing the GRU's temporal modelling from naive smoothing.
    For exit_open, the mean is clipped to [0, 1] and represents a smoothed
    probability, not a hard binary.
    """

    STREAM_ID = "baseline_moving_average"

    def __init__(self, window: int = 12):
        """
        Args:
            window: Number of most-recent history steps to average (default 12,
                    corresponding to 60 seconds at 5 s/step).
        """
        self.window = window

    def forecast(
        self,
        batch:       WindowBatch,
        scenario_id: str,
    ) -> List[ForecastRecord]:
        """
        Produces moving-average forecasts for all zones and horizons.
        """
        records: List[ForecastRecord] = []
        issue_time = batch.issue_time

        for z_idx, zone_id in enumerate(batch.zone_ids):
            u_hist = batch.u_history[z_idx]             # (HISTORY_STEPS, U_DIM)
            x_hist = _extract_x_from_u(u_hist)          # (HISTORY_STEPS, 6)
            recent = x_hist[-self.window:]               # (window, 6)
            ma     = recent.mean(axis=0)                 # (6,)
            _is_exit = is_exit_zone(zone_id)

            for h_sec, target_at in batch.target_times.items():
                for v_idx in range(NUM_NUMERIC_CHANNELS):
                    var = VAR_NAMES[v_idx]
                    val = float(max(0.0, ma[v_idx]))
                    records.append(ForecastRecord(
                        forecast_id   = f"ma_{uuid.uuid4().hex[:10]}",
                        stream_id     = self.STREAM_ID,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = round(val, 5),
                        model_version = f"moving_average_w{self.window}_v1",
                        horizon_sec   = h_sec,
                    ))

                if _is_exit:
                    var = "exit_open"
                    val = float(np.clip(ma[5], 0.0, 1.0))
                    records.append(ForecastRecord(
                        forecast_id   = f"ma_{uuid.uuid4().hex[:10]}",
                        stream_id     = self.STREAM_ID,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = round(val, 5),
                        model_version = f"moving_average_w{self.window}_v1",
                        horizon_sec   = h_sec,
                    ))

        return records
