from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional
from crowd_reliability.models.schema import ForecastRecord, ObservationRecord, AssessmentRecord
from crowd_reliability.reliability.conflicts import deduplicate_reports, detect_conflicts
from crowd_reliability.reliability.freshness import (
    INSUFFICIENT_EVIDENCE_WINDOW_S,
    RAPID_CHANGE_MIN_REPORTS,
    RAPID_CHANGE_REL_STD,
    RAPID_CHANGE_WINDOW_S,
    observation_age_seconds,
)
from crowd_reliability.reliability.thresholds import DEFAULT_FRESHNESS_THRESHOLDS

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
        Delegates to crowd_reliability.reliability; per_variable_flags carries both the
        legacy flags (stale/conflict/missing) and the five independent freshness labels.
        A stale report is flagged for verification — never asserted false due to age.
        """
        zone_var_obs = [
            o for o in observations
            if o.zone_id == zone_id and o.variable == variable and o.received_at <= evaluated_at
        ]

        missing_flags = {
            "stale": True, "conflict": False, "missing": True,
            "recently_verified": False,
            "potentially_outdated": False,
            "conflicting_evidence": False,
            "rapidly_changing": False,
            "insufficient_current_evidence": True,
        }
        if not zone_var_obs:
            return AssessmentRecord(
                target_ref_id=target_ref_id,
                target_type="observation",
                evaluated_at=evaluated_at,
                observation_age_seconds=999.0,
                conflict_ids=[],
                coverage=0.0,
                per_variable_flags=missing_flags
            )

        # Collapse forwarded copies: duplicates never count as independent confirmations.
        independent, _, _ = deduplicate_reports(zone_var_obs)

        # Most recent independent observation
        age_sec = min(observation_age_seconds(o, evaluated_at) for o in independent)

        # Check for conflicts among independent recent reports (within last 30s)
        recent_window = [o for o in independent if observation_age_seconds(o, evaluated_at) <= 30.0]
        is_conflict, conflict_ids = detect_conflicts(recent_window)

        # Rapid change: relative std over the rapid-change window
        rapid_window = [o for o in independent if observation_age_seconds(o, evaluated_at) <= RAPID_CHANGE_WINDOW_S]
        is_rapid = False
        if len(rapid_window) >= RAPID_CHANGE_MIN_REPORTS:
            values = [o.value for o in rapid_window]
            mean_val = sum(values) / len(values)
            if abs(mean_val) > 1e-6:
                rel_std = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5 / abs(mean_val)
                is_rapid = rel_std > RAPID_CHANGE_REL_STD

        # Variable/scenario-aware freshness thresholds (P2): event-driven states go
        # stale faster than bulk density. Density keeps its 45s boundary.
        thresholds_obj = DEFAULT_FRESHNESS_THRESHOLDS
        recent_th, outdated_th = thresholds_obj.for_variable(variable)
        is_stale = age_sec > outdated_th
        coverage = 1.0 if not is_stale else max(0.1, 1.0 - (age_sec - outdated_th) / 100.0)

        return AssessmentRecord(
            target_ref_id=target_ref_id,
            target_type="observation",
            evaluated_at=evaluated_at,
            observation_age_seconds=round(age_sec, 2),
            conflict_ids=conflict_ids,
            coverage=round(coverage, 3),
            per_variable_flags={
                # Legacy flags (kept for API/dashboard compatibility)
                "stale": age_sec > outdated_th,
                "conflict": is_conflict,
                "missing": False,
                # Five independent freshness labels
                "recently_verified": age_sec <= recent_th,
                "potentially_outdated": age_sec > outdated_th,
                "conflicting_evidence": is_conflict,
                "rapidly_changing": is_rapid,
                "insufficient_current_evidence": age_sec > INSUFFICIENT_EVIDENCE_WINDOW_S,
            }
        )
