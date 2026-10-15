"""Strict ingestion and normalisation boundary for Twin Studio v3 reports (Phase C).

Every report that may reach a model input, a notice or an assessment passes through
``ReportNormalizer.normalize``. It validates the report against the source-specific schema, the
known run/source/route/zone identities, value domains, units and timestamps, and expands it into
one ``Measurement`` per variable in a single normalised unit. Anything malformed raises
``IngestError``; nothing is coerced, dropped or defaulted silently.

Two report families are accepted:
  * canonical route-level reports produced by the v3 archive's ``observations.py``
    (``camera``, ``sensor``, ``warden``; schema ``eh-v3-route-reports``), and
  * zone-level reports from the versioned extension ``eh-v3-zone-sensing-1``
    (``zone_camera``, ``zone_counter``, ``gate_sensor``; see zone_sensing.py).

Time: simulation-relative seconds are kept as integers/floats. ``to_utc`` maps them to UTC only
with an explicit timezone-aware run epoch; naive datetimes are rejected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

ROUTE_SCHEMA = "eh-v3-route-reports"
ZONE_SCHEMA = "eh-v3-zone-sensing-1"

ROUTE_FIELDS = frozenset({"observation_id", "source_id", "source_kind", "route_id", "observed_at_s",
                          "available_at_s", "values"})
ZONE_FIELDS = frozenset({"observation_id", "source_id", "source_kind", "zone_id", "observed_at_s",
                         "available_at_s", "values", "schema_version"})

# Required and optional value keys per source kind.
VALUE_SCHEMAS: dict[str, tuple[frozenset, frozenset]] = {
    "camera": (frozenset({"visible_people_estimate", "mean_speed_estimate_m_s", "density_estimate_p_m2"}), frozenset()),
    "sensor": (frozenset({"entries_count", "exits_count", "count_window_s"}), frozenset()),
    "warden": (frozenset({"queue_estimate", "movement_report"}), frozenset()),
    "zone_camera": (frozenset({"density_estimate_p_m2", "queue_estimate"}), frozenset({"mean_speed_estimate_m_s"})),
    "zone_counter": (frozenset({"inflow_count", "outflow_count", "count_window_s"}), frozenset()),
    "gate_sensor": (frozenset({"exit_state"}), frozenset()),
}
ROUTE_KINDS = frozenset({"camera", "sensor", "warden"})
ZONE_KINDS = frozenset({"zone_camera", "zone_counter", "gate_sensor"})
MOVEMENT_CATEGORIES = frozenset({"slow", "moving", "no_people_seen"})
EXIT_STATES = {"open": 1.0, "closed": 0.0}

# Normalised variables and their single canonical unit.
UNITS = {
    "visible_people": "persons",
    "density": "persons/m2",
    "speed": "m/s",
    "entry_rate": "persons/s",
    "exit_rate": "persons/s",
    "queue_length": "persons",
    "movement_category": "category",
    "inflow": "persons/s",
    "outflow": "persons/s",
    "exit_open": "probability",
}
# Physical sanity bounds (well above anything the v3 engine can produce; they catch unit errors).
MAX_DENSITY_P_M2 = 20.0
MAX_SPEED_M_S = 10.0
MAX_COUNT = 10_000
MAX_TIME_S = 7 * 24 * 3600


class IngestError(ValueError):
    """A report violates the ingestion contract."""


@dataclass(frozen=True)
class SourceSpec:
    """Operational metadata of one reporting source (inference-safe: no scenario events)."""
    source_id: str
    kind: str
    interval_s: int
    delay_s: int
    coverage: frozenset          # route ids (route kinds) or zone ids (zone kinds)
    schema: str

    @property
    def entity_type(self) -> str:
        return "route" if self.kind in ROUTE_KINDS else "zone"


@dataclass(frozen=True)
class Measurement:
    """One normalised variable from one validated report."""
    report_id: str
    source_id: str
    source_kind: str
    entity_type: str            # "route" | "zone"
    entity_id: str
    variable: str
    value: Any                  # float, or str for categorical variables
    unit: str
    observed_at_s: float
    available_at_s: float
    schema: str


@dataclass
class IngestContext:
    """Known identities for one run. Built by the operator from inference-safe metadata only."""
    run_id: str
    route_ids: frozenset
    zone_ids: frozenset
    gate_ids: frozenset
    sources: dict[str, SourceSpec] = field(default_factory=dict)


def _finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _count(name: str, v: Any) -> int:
    if not (_finite_number(v) and v == int(v) and 0 <= v <= MAX_COUNT):
        raise IngestError(f"{name} must be a whole number in [0, {MAX_COUNT}], got {v!r}")
    return int(v)


def _bounded(name: str, v: Any, upper: float) -> float:
    if not (_finite_number(v) and 0 <= v <= upper):
        raise IngestError(f"{name} must be a finite number in [0, {upper}], got {v!r}")
    return float(v)


def _time(name: str, v: Any) -> float:
    if not (_finite_number(v) and 0 <= v <= MAX_TIME_S):
        raise IngestError(f"{name} must be a finite, nonnegative simulation time, got {v!r}")
    return float(v)


def to_utc(epoch_utc: datetime, seconds: float) -> datetime:
    """Simulation seconds -> timezone-aware UTC. The epoch must be timezone-aware."""
    if not isinstance(epoch_utc, datetime) or epoch_utc.tzinfo is None or epoch_utc.utcoffset() is None:
        raise IngestError("run epoch must be a timezone-aware datetime")
    return epoch_utc + timedelta(seconds=float(seconds))


class ReportNormalizer:
    """Validates reports for one run and converts them to measurements, once per report id."""

    def __init__(self, ctx: IngestContext):
        self.ctx = ctx
        self._seen: dict[str, dict] = {}

    def normalize(self, report: Any) -> list[Measurement]:
        """Returns measurements for a new report, [] for an exact duplicate, raises otherwise."""
        if not isinstance(report, dict):
            raise IngestError("report must be a JSON object")
        rid = report.get("observation_id")
        if not isinstance(rid, str) or not rid:
            raise IngestError("observation_id must be a non-empty string")
        if rid in self._seen:
            if self._seen[rid] != report:
                raise IngestError(f"duplicate observation_id {rid!r} with different content")
            return []
        measurements = self._validate(report)
        self._seen[rid] = dict(report)
        return measurements

    def normalize_many(self, reports: Iterable[Any]) -> list[Measurement]:
        out: list[Measurement] = []
        for r in reports:
            out.extend(self.normalize(r))
        return out

    # ------------------------------------------------------------------ validation
    def _validate(self, r: dict) -> list[Measurement]:
        kind = r.get("source_kind")
        if kind in ROUTE_KINDS:
            expected_fields, entity_key, schema = ROUTE_FIELDS, "route_id", ROUTE_SCHEMA
        elif kind in ZONE_KINDS:
            expected_fields, entity_key, schema = ZONE_FIELDS, "zone_id", ZONE_SCHEMA
            if r.get("schema_version") != ZONE_SCHEMA:
                raise IngestError(f"zone report schema_version must be {ZONE_SCHEMA!r}")
        else:
            raise IngestError(f"unknown source_kind {kind!r}")
        if set(r) != expected_fields:
            extra, missing = sorted(set(r) - expected_fields), sorted(expected_fields - set(r))
            raise IngestError(f"report fields not in the contract: extra={extra} missing={missing}")

        source = self.ctx.sources.get(r["source_id"])
        if source is None:
            raise IngestError(f"unknown source_id {r['source_id']!r}")
        if source.kind != kind:
            raise IngestError(f"source {source.source_id!r} is a {source.kind}, report says {kind}")

        entity = r[entity_key]
        known = self.ctx.route_ids if entity_key == "route_id" else self.ctx.zone_ids
        if entity not in known:
            raise IngestError(f"unknown {entity_key} {entity!r}")
        if entity not in source.coverage:
            raise IngestError(f"{entity_key} {entity!r} is outside the coverage of {source.source_id!r}")
        if kind == "gate_sensor" and entity not in self.ctx.gate_ids:
            raise IngestError(f"gate_sensor report for non-gate zone {entity!r}")

        observed = _time("observed_at_s", r["observed_at_s"])
        available = _time("available_at_s", r["available_at_s"])
        if not observed <= available:
            raise IngestError("observed_at_s must be <= available_at_s")
        if available - observed != source.delay_s:
            raise IngestError(f"delivery latency {available - observed}s does not match source delay {source.delay_s}s")
        if observed != int(observed) or int(observed) % source.interval_s:
            raise IngestError(f"observed_at_s {observed} is not on the source's {source.interval_s}s reporting grid")
        if rid_expected(source.source_id, entity, int(observed)) != r["observation_id"]:
            raise IngestError("observation_id does not match source, entity and observation time")

        values = r["values"]
        if not isinstance(values, dict):
            raise IngestError("values must be an object")
        required, optional = VALUE_SCHEMAS[kind]
        if not required <= set(values) <= required | optional:
            extra, missing = sorted(set(values) - required - optional), sorted(required - set(values))
            raise IngestError(f"{kind} values not in the contract: extra={extra} missing={missing}")

        base = dict(report_id=r["observation_id"], source_id=source.source_id, source_kind=kind,
                    entity_type=source.entity_type, entity_id=entity, observed_at_s=observed,
                    available_at_s=available, schema=schema)
        return [Measurement(variable=var, value=val, unit=UNITS[var], **base)
                for var, val in self._values(kind, values, source)]

    @staticmethod
    def _values(kind: str, v: dict, source: SourceSpec) -> list[tuple[str, Any]]:
        if kind == "camera":
            return [("visible_people", float(_count("visible_people_estimate", v["visible_people_estimate"]))),
                    ("speed", _bounded("mean_speed_estimate_m_s", v["mean_speed_estimate_m_s"], MAX_SPEED_M_S)),
                    ("density", _bounded("density_estimate_p_m2", v["density_estimate_p_m2"], MAX_DENSITY_P_M2))]
        if kind == "sensor":
            window = _count("count_window_s", v["count_window_s"])
            if window != 1:
                raise IngestError("canonical sensor count_window_s must be 1")
            return [("entry_rate", _count("entries_count", v["entries_count"]) / window),
                    ("exit_rate", _count("exits_count", v["exits_count"]) / window)]
        if kind == "warden":
            if v["movement_report"] not in MOVEMENT_CATEGORIES:
                raise IngestError(f"unknown movement_report {v['movement_report']!r}")
            return [("queue_length", float(_count("queue_estimate", v["queue_estimate"]))),
                    ("movement_category", v["movement_report"])]
        if kind == "zone_camera":
            out = [("density", _bounded("density_estimate_p_m2", v["density_estimate_p_m2"], MAX_DENSITY_P_M2)),
                   ("queue_length", float(_count("queue_estimate", v["queue_estimate"])))]
            if "mean_speed_estimate_m_s" in v:
                out.append(("speed", _bounded("mean_speed_estimate_m_s", v["mean_speed_estimate_m_s"], MAX_SPEED_M_S)))
            return out
        if kind == "zone_counter":
            window = _count("count_window_s", v["count_window_s"])
            if window != source.interval_s:
                raise IngestError("zone_counter count_window_s must equal the source interval")
            return [("inflow", _count("inflow_count", v["inflow_count"]) / window),
                    ("outflow", _count("outflow_count", v["outflow_count"]) / window)]
        if v["exit_state"] not in EXIT_STATES:
            raise IngestError(f"unknown exit_state {v['exit_state']!r}")
        return [("exit_open", EXIT_STATES[v["exit_state"]])]


def rid_expected(source_id: str, entity_id: str, observed_s: int) -> str:
    """Report identity used by both v3 report families: '<source>:<entity>:<observed second>'."""
    return f"{source_id}:{entity_id}:{observed_s}"


def route_source_specs(canonical_sources: list[dict], route_ids: Iterable[str]) -> dict[str, SourceSpec]:
    """SourceSpecs for the archive's route-level sources (validated by observations.sources_for)."""
    routes = frozenset(route_ids)
    out = {}
    for s in canonical_sources:
        out[s["id"]] = SourceSpec(s["id"], s["kind"], int(s["interval_s"]), int(s["delay_s"]),
                                  frozenset(s.get("route_ids", sorted(routes))), ROUTE_SCHEMA)
    return out
