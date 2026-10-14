import os
import uuid
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from crowd_reliability.models.schema import ForecastRecord, ReferenceStateRecord
from crowd_reliability.simulation.scenarios import VenueLayout, get_default_venue
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES

class CentralSourceAdapter:
    """
    Adapter for central-source forecast feed (Stream A).
    In evaluation / demo mode, replays pre-generated forecasts from CSV
    or generates realistic simulated central forecasts (labelled 'simulated central forecast').
    """
    def __init__(self, mode: str = "replay", fixture_path: Optional[str] = None):
        self.mode = mode
        self.fixture_path = fixture_path
        self.provider_version = "simulated_central_v1.0"
        self._cached_forecasts: List[ForecastRecord] = []

    def generate_simulated_forecasts(
        self,
        scenario_id: str,
        ref_records: List[ReferenceStateRecord],
        venue: VenueLayout,
        horizons_sec: List[int] = None,
        step_interval_sec: int = 15
    ) -> List[ForecastRecord]:
        """
        Generates baseline central forecasts from historical trend with smoothing.
        Simulated central providers often have slower reaction times to sudden local changes.
        """
        if horizons_sec is None:
            horizons_sec = [30, 60, 120]

        # Organize reference records by (zone_id, variable, state_at)
        ref_map: Dict[Tuple[str, str, str], float] = {}
        unique_times: List[datetime] = []
        time_set = set()

        for r in ref_records:
            t_iso = r.state_at.isoformat()
            ref_map[(r.zone_id, r.variable, t_iso)] = r.value
            if t_iso not in time_set:
                time_set.add(t_iso)
                unique_times.append(r.state_at)

        unique_times.sort()
        forecasts: List[ForecastRecord] = []

        # Central source issues forecasts every step_interval_sec
        units = {
            "density": "p/m2",
            "speed": "m/s",
            "inflow": "p/s",
            "outflow": "p/s",
            "queue_length": "persons",
            "exit_open": "binary"
        }

        for t_issue in unique_times:
            # Issue every step_interval_sec
            sec = int(t_issue.timestamp())
            if sec % step_interval_sec != 0:
                continue

            for z_data in venue.zones:
                z_id = z_data["zone_id"]
                has_exit = z_data.get("has_exit", False)

                for h_sec in horizons_sec:
                    target_at = t_issue + timedelta(seconds=h_sec)
                    target_iso = target_at.isoformat()

                    for var_name in TARGET_VAR_NAMES:
                        if var_name == "exit_open" and not has_exit:
                            continue

                        current_val = ref_map.get((z_id, var_name, t_issue.isoformat()), 1.0)
                        future_true_val = ref_map.get((z_id, var_name, target_iso), current_val)

                        # Central forecast simulates an inertia / moving average model:
                        # Blends current reading with slower trend (exhibiting lag during surges)
                        if var_name == "exit_open":
                            # Central provider has delayed notice of exit closures
                            val = current_val  # Naive persistence
                        else:
                            # Slower exponential smoothing prediction
                            lag_factor = 0.65
                            val = lag_factor * current_val + (1.0 - lag_factor) * future_true_val

                        f_rec = ForecastRecord(
                            forecast_id=f"f_cent_{uuid.uuid4().hex[:8]}",
                            stream_id="stream_a_central",
                            scenario_id=scenario_id,
                            zone_id=z_id,
                            variable=var_name,
                            unit=units[var_name],
                            issued_at=t_issue,
                            received_at=t_issue + timedelta(seconds=2.0),  # Ingestion lag
                            target_at=target_at,
                            value=round(max(0.0, float(val)), 4),
                            model_or_provider_version=self.provider_version
                        )
                        forecasts.append(f_rec)

        self._cached_forecasts = forecasts
        return forecasts

    def export_to_csv(self, forecasts: List[ForecastRecord], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rows = []
        for f in forecasts:
            rows.append({
                "forecast_id": f.forecast_id,
                "stream_id": f.stream_id,
                "scenario_id": f.scenario_id,
                "zone_id": f.zone_id,
                "variable": f.variable,
                "unit": f.unit,
                "issued_at": f.issued_at.isoformat(),
                "received_at": f.received_at.isoformat(),
                "target_at": f.target_at.isoformat(),
                "value": f.value,
                "model_or_provider_version": f.model_or_provider_version
            })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

    def load_from_csv(self, filepath: str) -> List[ForecastRecord]:
        if not os.path.exists(filepath):
            return []
        df = pd.read_csv(filepath)
        forecasts = []
        for _, row in df.iterrows():
            forecasts.append(
                ForecastRecord(
                    forecast_id=str(row["forecast_id"]),
                    stream_id=str(row["stream_id"]),
                    scenario_id=str(row["scenario_id"]),
                    zone_id=str(row["zone_id"]),
                    variable=str(row["variable"]),
                    unit=str(row["unit"]),
                    issued_at=datetime.fromisoformat(row["issued_at"]),
                    received_at=datetime.fromisoformat(row["received_at"]),
                    target_at=datetime.fromisoformat(row["target_at"]),
                    value=float(row["value"]),
                    model_or_provider_version=str(row["model_or_provider_version"])
                )
            )
        self._cached_forecasts = forecasts
        return forecasts
