from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional
from crowd_reliability.models.schema import ForecastRecord, ObservationRecord, AssessmentRecord

class ComparisonResult:
    def __init__(
        self,
        zone_id: str,
        variable: str,
        unit: str,
        target_at: datetime,
        arch_forecast: Optional[ForecastRecord] = None,
        cent_forecast: Optional[ForecastRecord] = None,
        signed_diff: Optional[float] = None,
        abs_diff: Optional[float] = None
    ):
        self.zone_id = zone_id
        self.variable = variable
        self.unit = unit
        self.target_at = target_at
        self.arch_forecast = arch_forecast
        self.cent_forecast = cent_forecast
        self.signed_diff = signed_diff
        self.abs_diff = abs_diff

class ForecastComparator:
    """
    Aligns Stream A (Central) and Stream B (Architecture) by (zone_id, variable, target_at).
    Computes per-variable differences and evaluates evidence freshness and report conflicts.
    """
    def align_and_compare(
        self,
        arch_forecasts: List[ForecastRecord],
        cent_forecasts: List[ForecastRecord],
        as_of_time: Optional[datetime] = None
    ) -> List[ComparisonResult]:
        # Filter by as_of_time if provided (only use forecasts received by as_of_time)
        if as_of_time:
            arch_pool = [f for f in arch_forecasts if f.received_at <= as_of_time]
            cent_pool = [f for f in cent_forecasts if f.received_at <= as_of_time]
        else:
            arch_pool = arch_forecasts
            cent_pool = cent_forecasts

        def norm_ts(dt: datetime) -> int:
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            epoch = datetime(2026, 1, 1)
            return int(round((dt - epoch).total_seconds() / 5.0) * 5)

        # Map by (zone_id, variable, norm_target_at)
        arch_map: Dict[Tuple[str, str, int], ForecastRecord] = {
            (f.zone_id, f.variable, norm_ts(f.target_at)): f for f in arch_pool
        }
        cent_map: Dict[Tuple[str, str, int], ForecastRecord] = {
            (f.zone_id, f.variable, norm_ts(f.target_at)): f for f in cent_pool
        }

        all_keys = set(arch_map.keys()).union(set(cent_map.keys()))
        results: List[ComparisonResult] = []

        for key in sorted(all_keys):
            z_id, var, t_norm = key
            arch_f = arch_map.get(key)
            cent_f = cent_map.get(key)
            t_target = arch_f.target_at if arch_f else cent_f.target_at

            unit = arch_f.unit if arch_f else (cent_f.unit if cent_f else "")

            signed_diff = None
            abs_diff = None
            if arch_f is not None and cent_f is not None:
                signed_diff = round(arch_f.value - cent_f.value, 4)
                abs_diff = round(abs(signed_diff), 4)

            results.append(
                ComparisonResult(
                    zone_id=z_id,
                    variable=var,
                    unit=unit,
                    target_at=t_target,
                    arch_forecast=arch_f,
                    cent_forecast=cent_f,
                    signed_diff=signed_diff,
                    abs_diff=abs_diff
                )
            )

        return results

    def assess_information(
        self,
        observations: List[ObservationRecord],
        evaluated_at: datetime,
        zone_id: str,
        variable: str,
        target_ref_id: str = "obs_eval"
    ) -> AssessmentRecord:
        """
        Assesses evidence freshness, coverage, and report conflicts for a specific zone and variable.
        """
        zone_var_obs = [
            o for o in observations
            if o.zone_id == zone_id and o.variable == variable and o.received_at <= evaluated_at
        ]

        if not zone_var_obs:
            return AssessmentRecord(
                target_ref_id=target_ref_id,
                target_type="observation",
                evaluated_at=evaluated_at,
                observation_age_seconds=999.0,
                conflict_ids=[],
                coverage=0.0,
                per_variable_flags={"stale": True, "conflict": False, "missing": True}
            )

        # Most recent observation
        latest_obs = max(zone_var_obs, key=lambda o: o.observed_at)
        age_sec = max(0.0, (evaluated_at - latest_obs.observed_at).total_seconds())

        # Check for conflicts among recent reports (within last 30s)
        recent_window = [o for o in zone_var_obs if (evaluated_at - o.observed_at).total_seconds() <= 30.0]
        conflict_ids = []
        is_conflict = False

        if len(recent_window) > 1:
            values = [o.value for o in recent_window]
            mean_val = sum(values) / len(values)
            for o in recent_window:
                if mean_val > 0.1 and abs(o.value - mean_val) / (mean_val + 1e-4) > 0.4:
                    conflict_ids.append(o.report_id)
                    is_conflict = True

        is_stale = (age_sec > 45.0)
        coverage = 1.0 if not is_stale else max(0.1, 1.0 - (age_sec - 45.0) / 100.0)

        return AssessmentRecord(
            target_ref_id=target_ref_id,
            target_type="observation",
            evaluated_at=evaluated_at,
            observation_age_seconds=round(age_sec, 2),
            conflict_ids=conflict_ids,
            coverage=round(coverage, 3),
            per_variable_flags={
                "stale": is_stale,
                "conflict": is_conflict,
                "missing": False
            }
        )
