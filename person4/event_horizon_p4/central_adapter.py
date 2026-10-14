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

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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


class CentralDemoFixtureAdapter:
    """
    Produces Stream A 'central' forecasts from mock visible data.

    The fixture mimics a central provider by adding calibrated Gaussian noise
    to the tree-history mean values.  This represents a provider that has seen
    the same observable data but processes it independently.

    Channels marked 'unavailable' are omitted and reported separately so that
    comparison.py shows "unavailable" for those channels instead of a merged
    or inflated trust score.
    """

    #: Variables this fixture "provides" (others show 'unavailable')
    AVAILABLE_VARS: List[str] = ["density", "speed", "queue_length", "exit_open"]

    def __init__(
        self,
        noise_std: float = 0.15,
        seed: int = 99,
        available_vars: Optional[List[str]] = None,
    ):
        """
        Args:
            noise_std:      Gaussian noise standard deviation applied to
                            tree-history means to simulate a separate provider.
            seed:           RNG seed for reproducibility.
            available_vars: Override the set of variables this fixture provides.
        """
        self._noise_std      = noise_std
        self._rng            = random.Random(seed)
        self._available_vars = set(available_vars or self.AVAILABLE_VARS)

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
