"""Transparent staleness assessment for Twin Studio v3 reports (Phase H, policy eh-v3-staleness-1).

For every source's latest report about an entity (route or zone) and variable, at replay time T:

  age_s                = T - observed_at_s
  delivery_latency_s   = available_at_s - observed_at_s
  max_age_s            = delay_s + k * interval_s          (k = age_multiplier, candidate default 2)

Separate fields (never one combined score):
  freshness_status            FRESH | STALE_BY_AGE
  conflict_status             NONE | CONFLICT | NOT_COMPARABLE   comparable = same entity and variable, other
                              source, observations within the larger of the two sources' intervals
  evidence_coverage           SUFFICIENT | INSUFFICIENT          >= min_support fresh sources for entity/variable
  current_state_disagreement  True | False | None                newer report from another source materially
                              differs (None: no newer evidence). A validated model estimate may add
                              evidence only when its status is validated; it is labelled separately.
  forecast_change_warning     NONE | WARNING | NO_FORECAST | UNAVAILABLE_MODEL_UNVALIDATED
  state                       CONFLICTING_EVIDENCE > POTENTIALLY_OUTDATED > STALE_BY_AGE >
                              INSUFFICIENT_EVIDENCE > FORECAST_CHANGE_WARNING > FRESH (first that applies)
  recommended_action          REVERIFY (conflict / newer evidence differs), VERIFY (old, thin or
                              forecast change), NO_ACTION
  reason_codes                each code with the evidence ids and timestamps behind it

Semantics: an old report can be correct; a fresh report can be wrong; a conflict does not say
which source is right; a forecast does not prove the report is already false.

Offline evaluation (evaluate_assessments) labels a report "materially inconsistent" when it
differs from the hidden true state at T by more than the variable's tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.twin_studio.v3.ingest import Measurement
from services.twin_studio.v3.replay import AuthorizedSnapshot

POLICY_VERSION = "eh-v3-staleness-1"
ALERT_STATES = ("CONFLICTING_EVIDENCE", "POTENTIALLY_OUTDATED", "STALE_BY_AGE", "INSUFFICIENT_EVIDENCE",
                "FORECAST_CHANGE_WARNING")
# Semantic tolerances (absolute floor, relative): "materially different" means beyond both.
TOLERANCES = {
    "density": (0.25, 0.25), "speed": (0.15, 0.25), "inflow": (0.3, 0.3), "outflow": (0.3, 0.3),
    "queue_length": (2.0, 0.3), "visible_people": (3.0, 0.2), "entry_rate": (2.0, 0.5), "exit_rate": (2.0, 0.5),
    "exit_open": (0.5, 0.0),
}
CATEGORICAL = {"movement_category"}


def materially_different(variable: str, a: Any, b: Any) -> bool:
    if variable in CATEGORICAL:
        return a != b
    floor, rel = TOLERANCES[variable]
    return abs(float(a) - float(b)) > max(floor, rel * max(abs(float(a)), abs(float(b))))


@dataclass(frozen=True)
class StalenessPolicy:
    age_multiplier: float = 2.0
    min_support: int = 1
    version: str = POLICY_VERSION

    def max_age(self, delay_s: float, interval_s: float) -> float:
        return delay_s + self.age_multiplier * interval_s


@dataclass
class Assessment:
    report_id: str
    source_id: str
    source_kind: str
    entity_type: str
    entity_id: str
    variable: str
    value: Any
    observed_at_s: float
    available_at_s: float
    evaluated_at_s: float
    age_s: float
    delivery_latency_s: float
    max_age_s: float
    freshness_status: str
    conflict_status: str
    evidence_coverage: str
    current_state_disagreement: bool | None
    forecast_change_warning: str
    state: str
    recommended_action: str
    reason_codes: list = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @property
    def alert(self) -> bool:
        return self.recommended_action != "NO_ACTION"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def assess_snapshot(snapshot: AuthorizedSnapshot, policy: StalenessPolicy = StalenessPolicy(), *,
                    forecasts: dict | None = None, forecasts_validated: bool = False,
                    model_estimates: dict | None = None, model_estimates_validated: bool = False) -> list[Assessment]:
    """Assess each source's latest report per entity/variable visible in the snapshot.

    forecasts / model_estimates: optional {(entity_type, entity_id, variable): (value, model_version)}.
    They are used only when the corresponding *_validated flag is True; otherwise the field says so.
    """
    T = snapshot.as_of_s
    sources = {s.source_id: s for s in snapshot.sources}
    latest: dict[tuple, Measurement] = {}
    for m in snapshot.measurements:
        k = (m.entity_type, m.entity_id, m.variable, m.source_id)
        cur = latest.get(k)
        if cur is None or (m.observed_at_s, m.report_id) > (cur.observed_at_s, cur.report_id):
            latest[k] = m
    by_target: dict[tuple, list[Measurement]] = {}
    for (et, eid, var, _), m in latest.items():
        by_target.setdefault((et, eid, var), []).append(m)

    out = []
    for (et, eid, var), ms in sorted(by_target.items()):
        fresh_sources = [m for m in ms if T - m.observed_at_s <= policy.max_age(sources[m.source_id].delay_s,
                                                                                 sources[m.source_id].interval_s)]
        for m in sorted(ms, key=lambda x: x.source_id):
            src = sources[m.source_id]
            age = T - m.observed_at_s
            max_age = policy.max_age(src.delay_s, src.interval_s)
            reasons = []
            freshness = "FRESH" if age <= max_age else "STALE_BY_AGE"
            if freshness == "STALE_BY_AGE":
                reasons.append({"code": "AGE_EXCEEDS_LIMIT", "age_s": age, "max_age_s": max_age,
                                "observed_at_s": m.observed_at_s})
            others = [o for o in ms if o.source_id != m.source_id]
            conflict, conflict_ids = "NOT_COMPARABLE" if not others else "NONE", []
            for o in others:
                window = max(src.interval_s, sources[o.source_id].interval_s)
                if abs(o.observed_at_s - m.observed_at_s) <= window:
                    if conflict == "NOT_COMPARABLE":
                        conflict = "NONE"
                    if materially_different(var, m.value, o.value):
                        conflict = "CONFLICT"
                        conflict_ids.append(o.report_id)
            if conflict == "CONFLICT":
                reasons.append({"code": "COMPARABLE_REPORTS_DISAGREE", "with": conflict_ids})
            newer = [o for o in others if o.observed_at_s > m.observed_at_s]
            disagreement = None
            if newer:
                differing = [o for o in newer if materially_different(var, m.value, o.value)]
                disagreement = bool(differing)
                if differing:
                    reasons.append({"code": "NEWER_REPORT_DIFFERS", "with": [o.report_id for o in differing],
                                    "observed_at_s": [o.observed_at_s for o in differing]})
            estimate = (model_estimates or {}).get((et, eid, var))
            if estimate is not None and model_estimates_validated and materially_different(var, m.value, estimate[0]):
                disagreement = True
                reasons.append({"code": "VALIDATED_MODEL_ESTIMATE_DIFFERS", "estimate": estimate[0], "model": estimate[1]})
            coverage = "SUFFICIENT" if len(fresh_sources) >= policy.min_support else "INSUFFICIENT"
            if coverage == "INSUFFICIENT":
                reasons.append({"code": "NO_FRESH_SOURCE", "fresh_sources": len(fresh_sources)})
            forecast = (forecasts or {}).get((et, eid, var))
            if forecast is None:
                fc = "NO_FORECAST"
            elif not forecasts_validated:
                fc = "UNAVAILABLE_MODEL_UNVALIDATED"
            elif materially_different(var, m.value, forecast[0]):
                fc = "WARNING"
                reasons.append({"code": "FORECAST_PREDICTS_CHANGE", "forecast": forecast[0], "model": forecast[1]})
            else:
                fc = "NONE"
            if conflict == "CONFLICT":
                state, action = "CONFLICTING_EVIDENCE", "REVERIFY"
            elif disagreement:
                state, action = "POTENTIALLY_OUTDATED", "REVERIFY"
            elif freshness == "STALE_BY_AGE":
                state, action = "STALE_BY_AGE", "VERIFY"
            elif coverage == "INSUFFICIENT":
                state, action = "INSUFFICIENT_EVIDENCE", "VERIFY"
            elif fc == "WARNING":
                state, action = "FORECAST_CHANGE_WARNING", "VERIFY"
            else:
                state, action = "FRESH", "NO_ACTION"
            out.append(Assessment(m.report_id, m.source_id, m.source_kind, et, eid, var, m.value, m.observed_at_s,
                                  m.available_at_s, T, age, m.available_at_s - m.observed_at_s, max_age, freshness,
                                  conflict, coverage, disagreement, fc, state, action, reasons))
    return out


# ---------------------------------------------------------------- offline evaluation

ROUTE_TRUTH_COLUMN = {"density": "mean_density_p_m2", "speed": "mean_speed_m_s", "visible_people": "active",
                      "entry_rate": "spawned", "exit_rate": "exited", "queue_length": "queued"}


def route_movement_category(row: dict) -> str:
    active, speed = float(row["active"]), float(row["mean_speed_m_s"])
    return "slow" if active and speed < 0.2 else "moving" if active else "no_people_seen"


def truth_value(a: Assessment, zone_state_at_T: dict | None, route_row_at_T: dict | None,
                zone_window_states: list | None = None):
    """Hidden true value of the assessed entity/variable at T (offline only); None if undefined.

    zone_window_states: for zone flow variables, the zone states whose 5 s windows make up the
    reporting source's counting window ending at T (so a 10 s counter is compared with a 10 s rate).
    """
    if a.entity_type == "zone":
        if zone_state_at_T is None:
            return None
        z = zone_state_at_T["zones"][a.entity_id]
        if a.variable == "exit_open":
            return z["exit_open"] if z["exit_applicable"] else None
        if a.variable in ("inflow", "outflow") and zone_window_states:
            vals = [st["zones"][a.entity_id][a.variable] for st in zone_window_states]
            return None if any(v is None for v in vals) else sum(vals) / len(vals)
        return z.get(a.variable)
    if a.variable == "movement_category":
        return route_movement_category(route_row_at_T)
    col = ROUTE_TRUTH_COLUMN.get(a.variable)
    return float(route_row_at_T[col]) if col else None


def score(records: list[tuple[bool, bool]]) -> dict:
    """records: (alert, inconsistent). Precision, recall, false-positive rate."""
    tp = sum(1 for a, t in records if a and t)
    fp = sum(1 for a, t in records if a and not t)
    fn = sum(1 for a, t in records if not a and t)
    tn = sum(1 for a, t in records if not a and not t)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"n": len(records), "positives": tp + fn, "alerts": tp + fp, "precision": p, "recall": r,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "f1": 2 * p * r / (p + r) if p and r else 0.0}
