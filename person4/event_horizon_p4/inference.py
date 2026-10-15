"""
inference.py – Checkpoint loading and variable-resolved ForecastRecord production.

NOTE: Checkpoint loading and forecast generation. Variable schemas and
exit masks are aligned with the production schema and pipeline contract.

Public interface:
    forecaster = TemporalForecaster.from_checkpoint(path)
    records    = forecaster.forecast(history_batch, issue_time, scenario_id)

Output contract:
  - Returns a list of ForecastRecord dataclasses with:
      forecast_id, stream_id, scenario_id, zone_id, variable, unit,
      issued_at, received_at, target_at, value, model_version.
  - Exit applicability mask is applied: exit_open records are omitted for
    non-exit zones (NOT set to zero — omitted entirely).
  - Inference reverses numeric StandardScaler and applies Sigmoid to exit logit.
  - No future-time inputs are accepted (enforced by WindowBatch construction).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import (
    HORIZONS_SEC,
    NUM_CHANNELS,
    NUM_NUMERIC_CHANNELS,
    STREAM_B_ID,
    U_DIM,
    VAR_NAMES,
    VAR_UNITS,
)
from .model import EXIT_LOGIT_IDX, TemporalGRUForecaster
from .sequence import WindowBatch
from .synthetic import is_exit_zone


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastRecord:
    """
    Immutable forecast record, fully resolved by:
      scenario / zone / variable / unit / horizon / target_time / stream / version.

    Aligns with the ForecastRecord SQLAlchemy schema in crowd_reliability.models.schema
    but is kept as a plain dataclass to avoid introducing a DB dependency here.
    """
    forecast_id:             str
    stream_id:               str
    scenario_id:             str
    zone_id:                 str
    variable:                str
    unit:                    str
    issued_at:               datetime
    received_at:             datetime
    target_at:               datetime
    value:                   float
    model_version:           str
    horizon_sec:             int   # informational; derivable from target_at - issued_at

    def to_dict(self) -> Dict:
        return {
            "forecast_id":   self.forecast_id,
            "stream_id":     self.stream_id,
            "scenario_id":   self.scenario_id,
            "zone_id":       self.zone_id,
            "variable":      self.variable,
            "unit":          self.unit,
            "issued_at":     self.issued_at.isoformat(),
            "received_at":   self.received_at.isoformat(),
            "target_at":     self.target_at.isoformat(),
            "value":         self.value,
            "model_version": self.model_version,
            "horizon_sec":   self.horizon_sec,
        }


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class TemporalForecaster:
    """
    Wraps TemporalGRUForecaster with scaler reversal and applicability masking.

    The numeric outputs of the raw GRU are in scaler-normalised space;
    this class reverses that transformation to return physical-unit values.
    The exit-open logit is converted to a probability via Sigmoid.
    """

    def __init__(
        self,
        model:         TemporalGRUForecaster,
        scaler_mean:   Optional[np.ndarray] = None,
        scaler_scale:  Optional[np.ndarray] = None,
        model_version: str = "gru_forecaster_v1",
    ):
        self._model        = model
        self._model.eval()
        self._scaler_mean  = scaler_mean   # (6,) or None
        self._scaler_scale = scaler_scale  # (6,) or None
        self._version      = model_version

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        scaler_path: Optional[str] = None,
    ) -> "TemporalForecaster":
        """
        Loads model checkpoint and (optionally) scaler parameters.

        Args:
            checkpoint_path: Path to the .pt model checkpoint.
            scaler_path:     Path to the _scaler.npy file (optional).
        """
        model = TemporalGRUForecaster.load(checkpoint_path)

        mean = scale = None
        if scaler_path:
            data  = np.load(scaler_path, allow_pickle=True).item()
            mean  = data.get("mean")
            scale = data.get("scale")
        elif checkpoint_path.endswith(".pt"):
            # Try default location
            sp = checkpoint_path.replace(".pt", "_scaler.npy")
            try:
                data  = np.load(sp, allow_pickle=True).item()
                mean  = data.get("mean")
                scale = data.get("scale")
            except FileNotFoundError:
                pass

        return cls(
            model         = model,
            scaler_mean   = mean,
            scaler_scale  = scale,
            model_version = model.version,
        )

    # ------------------------------------------------------------------
    # Inverse scaling
    # ------------------------------------------------------------------

    def _inverse_scale_numeric(self, values: np.ndarray) -> np.ndarray:
        """
        Reverses StandardScaler for numeric channels [0:5].

        Args:
            values: (..., 5) array in scaler-normalised space.

        Returns:
            (..., 5) array in physical units.
        """
        if self._scaler_mean is None or self._scaler_scale is None:
            return values  # no scaler fitted — return as-is
        return values * self._scaler_scale[:NUM_NUMERIC_CHANNELS] + self._scaler_mean[:NUM_NUMERIC_CHANNELS]

    # ------------------------------------------------------------------
    # Core forecast method
    # ------------------------------------------------------------------

    def forecast(
        self,
        batch:       WindowBatch,
        scenario_id: str,
        stream_id:   str = STREAM_B_ID,
    ) -> List[ForecastRecord]:
        """
        Produces ForecastRecord list from a pre-assembled WindowBatch.

        Args:
            batch:       WindowBatch produced by SequenceWindowAssembler.step().
            scenario_id: Scenario identifier to embed in every record.
            stream_id:   Stream label (default: stream_b_architecture).

        Returns:
            List of ForecastRecord, one per (zone, horizon, variable).
            exit_open records are omitted for non-exit zones.
        """
        u_tensor = torch.tensor(
            batch.u_history, dtype=torch.float32
        )   # (Z, T, U_DIM)

        with torch.no_grad():
            raw_out = self._model(u_tensor).cpu().numpy()  # (Z, H, 6)

        # Reverse scaling on numeric channels
        num_phys   = self._inverse_scale_numeric(raw_out[..., :NUM_NUMERIC_CHANNELS])
        num_phys   = np.maximum(0.0, num_phys)   # physical non-negativity

        # Sigmoid on exit logit
        exit_probs = torch.sigmoid(
            torch.tensor(raw_out[..., EXIT_LOGIT_IDX])
        ).numpy()   # (Z, H)

        records: List[ForecastRecord] = []
        issue_time = batch.issue_time

        for z_idx, zone_id in enumerate(batch.zone_ids):
            _is_exit = is_exit_zone(zone_id)

            for h_idx, h_sec in enumerate(batch.target_times):
                # batch.target_times is a dict {h_sec: datetime}
                target_at = batch.target_times[h_sec]

                # Numeric channels
                for v_idx in range(NUM_NUMERIC_CHANNELS):
                    var = VAR_NAMES[v_idx]
                    val = float(num_phys[z_idx, h_idx, v_idx])
                    records.append(ForecastRecord(
                        forecast_id   = f"f_{uuid.uuid4().hex[:10]}",
                        stream_id     = stream_id,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = round(val, 5),
                        model_version = self._version,
                        horizon_sec   = h_sec,
                    ))

                # Exit-open channel: omit for non-exit zones
                if _is_exit:
                    var = "exit_open"
                    val = float(exit_probs[z_idx, h_idx])
                    records.append(ForecastRecord(
                        forecast_id   = f"f_{uuid.uuid4().hex[:10]}",
                        stream_id     = stream_id,
                        scenario_id   = scenario_id,
                        zone_id       = zone_id,
                        variable      = var,
                        unit          = VAR_UNITS[var],
                        issued_at     = issue_time,
                        received_at   = issue_time,
                        target_at     = target_at,
                        value         = round(val, 5),
                        model_version = self._version,
                        horizon_sec   = h_sec,
                    ))

        return records


# ---------------------------------------------------------------------------
# Convenience: run inference over all steps in a scenario
# ---------------------------------------------------------------------------

def forecast_scenario(
    forecaster:  TemporalForecaster,
    scenario,    # MockScenario
    scenario_id: str,
) -> List[ForecastRecord]:
    """
    Runs inference on every timestep of a MockScenario.

    Returns all ForecastRecord objects.  These may be persisted before
    their labels arrive (label availability is enforced in evaluation.py).
    """
    from .sequence import build_all_windows

    batches = build_all_windows(scenario)
    all_records: List[ForecastRecord] = []
    for batch in batches:
        recs = forecaster.forecast(batch, scenario_id=scenario_id)
        all_records.extend(recs)
    return all_records
