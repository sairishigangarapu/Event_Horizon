"""
central_adapter.py – Stream A demo fixture adapter.

⚠️  EXPLICITLY MARKED DEMO FIXTURE ⚠️

This adapter is NOT a real central provider.  It generates Stream A forecasts
from mock, VISIBLE tree-history values — the same synthetic data used for
model input — NOT from hidden simulator truth.

Until a genuine external provider or disclosed independent replay is configured,
every record produced here carries stream_id = "stream_a_central_DEMO_FIXTURE"
and a fixture_warning field.

Engineering rule:
  - Reference truth must never appear under the "central" label.
  - This adapter must never be relabelled as real data without explicit audit.
"""

from __future__ import annotations

import csv
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    HORIZONS_SEC,
    NUM_NUMERIC_CHANNELS,
    STREAM_A_ID,
    VAR_NAMES,
    VAR_UNITS,
)
from .inference import ForecastRecord
from .sequence import WindowBatch
from .synthetic import is_exit_zone


# ---------------------------------------------------------------------------
# Fixture provider
# ---------------------------------------------------------------------------

_FIXTURE_WARNING = (
    "DEMO FIXTURE ONLY — generated from visible mock tree-history values, "
    "NOT from hidden simulator truth or a real central provider. "
    "Do NOT use for real emergency decisions."
)

_FIXTURE_VERSION = "stream_a_demo_fixture_v1"


def _parse_ts(ts_str: str) -> datetime:
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class CentralDemoFixtureAdapter:
    """
    Produces Stream A 'central' forecasts from CSV fixture or calibrated visible data.

    If a real Central CSV fixture exists (e.g. data/demo/central_forecast.csv),
    records are read from it. Otherwise, it generates calibrated values with noise.
    """

    #: Variables this fixture "provides" (others show 'unavailable')
    AVAILABLE_VARS: List[str] = ["density", "speed", "queue_length", "exit_open"]

    def __init__(
        self,
        noise_std: float = 0.15,
        seed: int = 99,
        available_vars: Optional[List[str]] = None,
        csv_path: Optional[str] = "data/demo/central_forecast.csv",
    ):
        self._noise_std      = noise_std
        self._rng            = random.Random(seed)
        self._available_vars = set(available_vars or self.AVAILABLE_VARS)
        self._csv_records: List[ForecastRecord] = []
        self._csv_path = csv_path

        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)

    def _load_csv(self, path: str) -> None:
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    issued_at = _parse_ts(row["issued_at"])
                    received_at = _parse_ts(row["received_at"])
                    target_at = _parse_ts(row["target_at"])
                    h_sec = int(round((target_at - issued_at).total_seconds()))
                    self._csv_records.append(ForecastRecord(
                        forecast_id   = row["forecast_id"],
                        stream_id     = row.get("stream_id", STREAM_A_ID),
                        scenario_id   = row["scenario_id"],
                        zone_id       = row["zone_id"],
                        variable      = row["variable"],
                        unit          = row["unit"],
                        issued_at     = issued_at,
                        received_at   = received_at,
                        target_at     = target_at,
                        value         = float(row["value"]),
                        model_version = row.get("model_or_provider_version", _FIXTURE_VERSION),
                        horizon_sec   = h_sec,
                    ))
        except Exception:
            self._csv_records = []

    def forecast(
        self,
        batch:       WindowBatch,
        scenario_id: str,
    ) -> List[ForecastRecord]:
        """
        Produces Stream A demo-fixture forecasts for one WindowBatch.

        Available variables receive a value; unavailable variables are skipped.
        Callers (comparison.py) detect skipped variables and label them
        'unavailable'.

        Args:
            batch:       The current WindowBatch (provides visible history).
            scenario_id: Scenario identifier.

        Returns:
            List of ForecastRecord with stream_id = STREAM_A_ID.
        """
        records: List[ForecastRecord] = []
        issue_time = batch.issue_time

        if self._csv_records:
            matching = [
                r for r in self._csv_records
                if r.scenario_id == scenario_id
                and r.zone_id in batch.zone_ids
                and r.received_at <= issue_time
                and r.target_at in batch.target_times.values()
            ]
            if matching:
                return matching

        for z_idx, zone_id in enumerate(batch.zone_ids):
            # Use tree-history X slice (channels 0-5) as the "observable"
            x_hist = batch.u_history[z_idx, :, :6]      # (T, 6)
            x_mean = x_hist.mean(axis=0)                 # (6,)
            _is_exit = is_exit_zone(zone_id)

            for h_sec, target_at in batch.target_times.items():
                for v_idx, var in enumerate(VAR_NAMES):
                    if var == "exit_open" and not _is_exit:
                        continue   # exit_open not applicable for non-exit zones

                    if var not in self._available_vars:
                        continue   # this channel is 'unavailable' from Stream A

                    base = float(x_mean[v_idx])
                    # Add noise to simulate independent processing
                    noisy = base + self._rng.gauss(0.0, self._noise_std)
                    if var == "exit_open":
                        val = float(np.clip(noisy, 0.0, 1.0))
                    else:
                        val = float(max(0.0, noisy))

                    records.append(ForecastRecord(
                        forecast_id   = f"sa_{uuid.uuid4().hex[:10]}",
                        stream_id     = STREAM_A_ID,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = round(val, 5),
                        model_version = _FIXTURE_VERSION,
                        horizon_sec   = h_sec,
                    ))

        return records

    @property
    def available_variables(self) -> List[str]:
        """Returns the list of variables this fixture can provide."""
        return sorted(self._available_vars)

    @property
    def unavailable_variables(self) -> List[str]:
        """Returns variables this fixture cannot provide ('unavailable' in comparison)."""
        return sorted(set(VAR_NAMES) - self._available_vars)

    @property
    def fixture_warning(self) -> str:
        return _FIXTURE_WARNING
