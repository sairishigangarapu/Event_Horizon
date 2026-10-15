import hashlib
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from crowd_reliability.models.schema import ForecastRecord, ObservationRecord
from crowd_reliability.simulation.scenarios import VenueLayout, get_default_venue
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES

# Simulated central provider: smoothing parameters (documented, not tuned).
CENTRAL_EWMA_ALPHA = 0.35          # weight of each newer report in the smoothed level
CENTRAL_LOOKBACK_SECONDS = 90.0    # reports observed earlier than this are ignored
CENTRAL_INGESTION_LAG_SECONDS = 2.0
CENTRAL_PROVIDER_VERSION = "simulated_central_obs_ewma_v2"

_UNITS = {
    "density": "p/m2",
    "speed": "m/s",
    "inflow": "p/s",
    "outflow": "p/s",
    "queue_length": "persons",
    "exit_open": "binary",
}


def _grid_floor(t: datetime, step_sec: int) -> datetime:
    """Floor t to a step_sec grid anchored at 2026-01-01 (same tz-awareness as t)."""
    epoch = datetime(2026, 1, 1, tzinfo=t.tzinfo)
    offset = int((t - epoch).total_seconds() // step_sec) * step_sec
    return epoch + timedelta(seconds=offset)


def issue_grid(times: List[datetime], step_sec: int) -> List[datetime]:
    """Grid instants (every step_sec, anchored at 2026-01-01) spanning [min(times), max(times)]."""
    if not times:
        return []
    first, last = min(times), max(times)
    t = _grid_floor(first, step_sec)
    if t < first:
        t += timedelta(seconds=step_sec)
    grid = []
    while t <= last:
        grid.append(t)
        t += timedelta(seconds=step_sec)
    return grid


def _forecast_id(scenario_id: str, zone_id: str, variable: str, issued_at: datetime, horizon_sec: int) -> str:
    key = f"{scenario_id}|{zone_id}|{variable}|{issued_at.isoformat()}|{horizon_sec}"
    return "f_cent_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class CentralSourceAdapter:
    """
    Adapter for the central-source forecast feed (Stream A).

    In evaluation / demo mode it replays forecasts from CSV, or generates a
    *simulated central forecast* from the observation stream. The simulated
    provider is an independent, deliberately simple forecaster: it never reads
    simulator reference states, so it cannot see current or future truth.
    """
    def __init__(self, mode: str = "replay", fixture_path: Optional[str] = None):
        self.mode = mode
        self.fixture_path = fixture_path
        self.provider_version = CENTRAL_PROVIDER_VERSION
        self._cached_forecasts: List[ForecastRecord] = []

    def generate_simulated_forecasts(
        self,
        scenario_id: str,
        observations: List[ObservationRecord],
        venue: VenueLayout,
        horizons_sec: Optional[List[int]] = None,
        step_interval_sec: int = 15,
        issue_times: Optional[List[datetime]] = None,
    ) -> List[ForecastRecord]:
        """
        Simulated central forecasts built only from observations.

        At each issue time t (every step_interval_sec over the span of received
        reports, unless issue_times is given), for every zone and variable:
          - candidates are reports with received_at <= t and observed_at within
            CENTRAL_LOOKBACK_SECONDS of t (nothing received after t is used);
          - numeric variables: exponentially weighted level of the candidates in
            observed_at order (a lagging, smoothed provider);
          - exit_open: the most recently observed report (naive persistence);
          - the level is held constant across all horizons.
        A zone/variable with no candidate report gets no forecast: it is
        unavailable, never filled with a default. exit_open is only issued for
        exit zones. Forecast IDs are deterministic.
        """
        if horizons_sec is None:
            horizons_sec = [30, 60, 120]

        by_key: Dict[Tuple[str, str], List[ObservationRecord]] = {}
        for o in observations:
            if o.scenario_id not in (None, scenario_id) or o.variable not in _UNITS:
                continue
            by_key.setdefault((o.zone_id, o.variable), []).append(o)
        if not by_key:
            self._cached_forecasts = []
            return []

        if issue_times is None:
            issue_times = issue_grid(
                [o.received_at for reports in by_key.values() for o in reports], step_interval_sec
            )

        forecasts: List[ForecastRecord] = []
        lookback = timedelta(seconds=CENTRAL_LOOKBACK_SECONDS)
        for t_issue in sorted(issue_times):
            for z_data in venue.zones:
                z_id = z_data["zone_id"]
                has_exit = z_data.get("has_exit", False)
                for var_name in TARGET_VAR_NAMES:
                    if var_name == "exit_open" and not has_exit:
                        continue
                    candidates = sorted(
                        (o for o in by_key.get((z_id, var_name), [])
                         if o.received_at <= t_issue and o.observed_at >= t_issue - lookback),
                        key=lambda o: (o.observed_at, o.received_at, o.report_id),
                    )
                    if not candidates:
                        continue  # unavailable: no evidence received by t_issue
                    if var_name == "exit_open":
                        val = float(candidates[-1].value)
                    else:
                        val = float(candidates[0].value)
                        for o in candidates[1:]:
                            val = CENTRAL_EWMA_ALPHA * float(o.value) + (1.0 - CENTRAL_EWMA_ALPHA) * val

                    for h_sec in horizons_sec:
                        forecasts.append(ForecastRecord(
                            forecast_id=_forecast_id(scenario_id, z_id, var_name, t_issue, h_sec),
                            stream_id="stream_a_central",
                            scenario_id=scenario_id,
                            zone_id=z_id,
                            variable=var_name,
                            unit=_UNITS[var_name],
                            issued_at=t_issue,
                            received_at=t_issue + timedelta(seconds=CENTRAL_INGESTION_LAG_SECONDS),
                            target_at=t_issue + timedelta(seconds=h_sec),
                            value=round(max(0.0, val), 4),
                            model_or_provider_version=self.provider_version,
                        ))

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
