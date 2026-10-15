from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional
from crowd_reliability.contracts import utc
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
    Aligns Stream A (Central) and Stream B (Architecture) by (scenario_id, zone_id, variable, unit, issued_at, target_at).
    Computes per-variable differences and evaluates evidence freshness and report conflicts.
    """
    def align_and_compare(
        self,
        arch_forecasts: List[ForecastRecord],
        cent_forecasts: List[ForecastRecord],
        as_of_time: Optional[datetime] = None
    ) -> List[ComparisonResult]:
        def index(records):
            out={}
            for f in records:
                if as_of_time and (utc(f.received_at)>utc(as_of_time) or utc(f.issued_at)>utc(as_of_time)):continue
                if utc(f.issued_at)>utc(f.received_at) or utc(f.issued_at)>=utc(f.target_at):raise ValueError('Invalid forecast timestamps.')
                # Exact physical quantity and target; latest available revision wins.
                key=(f.scenario_id,f.zone_id,f.variable,f.unit,utc(f.issued_at),utc(f.target_at))
                rank=(utc(f.issued_at),utc(f.received_at),str(f.forecast_id))
                if key not in out or rank>out[key][0]:out[key]=(rank,f)
            return {k:v[1] for k,v in out.items()}
        arch_map=index(arch_forecasts);cent_map=index(cent_forecasts)
        all_keys=set(arch_map)|set(cent_map)
        results: List[ComparisonResult] = []

        for key in sorted(all_keys):
            _, z_id, var, _, _, _ = key
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
        target_ref_id: str = "obs_eval",
        zone_type: Optional[str] = None,
        scenario_type: Optional[str] = None,
    ) -> AssessmentRecord:
        """
        Assesses evidence freshness, coverage, and report conflicts for a specific zone and variable.
        Delegates to crowd_reliability.reliability; per_variable_flags carries both the
        legacy flags (stale/conflict/missing) and the five independent freshness labels.
        A stale report is flagged for verification — never asserted false due to age.
        """
        zone_var_obs = [
            o for o in observations
            if o.zone_id == zone_id and o.variable == variable and utc(o.received_at) <= utc(evaluated_at) and utc(o.observed_at) <= utc(evaluated_at)
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

        from crowd_reliability.reliability import assess_information as canonical_assessment
        # The known roster is inferred from sources received so far, not from future
        # reports. Coverage is the fresh fraction of that roster, never an age proxy.
        roster=sorted({o.source_id for o in zone_var_obs if o.source_id})
        result=canonical_assessment(zone_var_obs,evaluated_at,
            expected_coverage={(zone_id,variable):roster},scenario_type=scenario_type,
            zone_type_lookup={zone_id:zone_type} if zone_type else None)
        row=next(r for r in result['per_zone'] if r['zone_id']==zone_id and r['variable']==variable)
        labels=row['labels']
        record=AssessmentRecord(target_ref_id=target_ref_id,target_type='observation',
            evaluated_at=evaluated_at,observation_age_seconds=row['age_seconds'],
            conflict_ids=row['conflict_ids'],coverage=row['coverage_ratio'] or 0.0,
            per_variable_flags=dict(labels,stale=labels['potentially_outdated'],
                conflict=labels['conflicting_evidence'],missing=False))
        record.coverage_basis='fresh sources / historically received source roster'
        record.conflict_graph=result['conflict_graph']
        return record
