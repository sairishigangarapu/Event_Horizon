"""Causal synthetic observation reports generated from one run's zone-state history (Phase 2C).

Ports the repaired v0.2 generator design (event_horizon_member1_v0.2/src/data/replay.py,
`observation_records`) to Twin Studio zone states:
  observed_at  = a state instant T on the observation grid; the value comes from the state at T only
  published_at = observed_at + publish_latency (>= 1 s)
  received_at  = published_at + transport latency (>= 0 s)   =>  observed_at < published_at <= received_at
Error model (v0.2 decision D4 defaults, uncalibrated): multiplicative camera noise, additive count
noise when truth > 0, zero-count false positives, gate-state misclassification. Undefined truth
(empty-zone speed, interval fields at instant 0, exit state outside gates) produces no report.

Differences from v0.2, both for causality and reproducibility:
  - every report draws from its own RNG seeded by (seed, source, zone, variable, instant), so a
    report never depends on any other state, earlier or later;
  - report IDs are derived from the same key, independent of which other reports were dropped.
Camera B coverage in v0.2 was specific to that venue; no Twin coverage has been approved, so it
defaults to no zones (configurable).
Reports never contain incident schedules, future states or reference-only fields. They also omit
run_mode and seed (Phase 3 fix): either would reveal hidden incident information. Both still key
the per-report RNG internally, so values are unchanged.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION
from services.twin_studio.timebase import instant_to_utc

SCHEMA_VERSION = "eh-twin-schema-1.0"
OBSERVATION_SOURCE_TYPE = "SYNTHETIC_OBSERVATION"
CAMERA_VARIABLES = ("density", "speed", "inflow", "outflow", "queue_length")
GATE_VARIABLES = ("probability_exit_open",)
UNITS = {"density": "persons/m2", "speed": "m/s", "inflow": "persons/s", "outflow": "persons/s",
         "queue_length": "persons", "probability_exit_open": "probability"}
FORBIDDEN_FIELDS = ("events", "incident", "gateId", "state_at", "label_available_at", "occupancy",
                    "accounting", "exit_open_during_interval", "gate_open_fraction",
                    # Phase 3 fix: run_mode reveals whether incidents exist; seed + scene allows
                    # recomputing the seeded random-collapse schedule. Both stay reference-only.
                    "run_mode", "seed")


@dataclass(frozen=True)
class ObservationConfig:
    """Defaults: v0.2 `DEFAULT_OBSERVATION_ERRORS` and the v0.2 `normal` scenario timing."""
    observation_every_seconds: int = 15
    dropout_probability: float = 0.05
    base_delay_seconds: int = 7                 # minimum total delay observed -> received
    publish_latency_seconds: int = 3
    transport_jitter_seconds: int = 14
    camera_sigma: dict = field(default_factory=lambda: {"sim_camera_A": 0.06, "sim_camera_B": 0.10})
    count_noise_sigma_persons: float = 0.5
    zero_false_positive_probability: float = 0.02
    false_positive_max_persons: int = 2
    exit_state_error_probability: float = 0.02
    camera_a_zones: tuple | None = None         # None = every zone
    camera_b_zones: tuple = ()                  # no approved Twin coverage yet
    outages: tuple = ()                         # ({source_id, zone_id, from_second, until_second}, ...)

    def validate(self, zone_ids: set[str], gate_ids: set[str], sample_seconds: int) -> None:
        if self.publish_latency_seconds < 1:
            raise ValueError("publish_latency_seconds must be >= 1 (strict publication after observation)")
        if self.base_delay_seconds < self.publish_latency_seconds:
            raise ValueError("base_delay_seconds (minimum total delay) must be >= publish_latency_seconds")
        if self.transport_jitter_seconds < 0:
            raise ValueError("transport_jitter_seconds must be >= 0")
        if self.observation_every_seconds <= 0 or self.observation_every_seconds % sample_seconds:
            raise ValueError("observation_every_seconds must be a positive multiple of the state sampling interval")
        for name in ("dropout_probability", "zero_false_positive_probability", "exit_state_error_probability"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.false_positive_max_persons < 1:
            raise ValueError("false_positive_max_persons must be >= 1")
        if any(s < 0 for s in self.camera_sigma.values()) or self.count_noise_sigma_persons < 0:
            raise ValueError("noise sigmas must be >= 0")
        for zs in (self.camera_a_zones or (), self.camera_b_zones):
            unknown = set(zs) - zone_ids
            if unknown:
                raise ValueError(f"coverage names unknown zones {sorted(unknown)}")
        for o in self.outages:
            if o["source_id"] not in ("sim_camera_A", "sim_camera_B", "sim_gate_sensor") or o["zone_id"] not in zone_ids:
                raise ValueError(f"invalid outage {o}")
            if o["source_id"] == "sim_gate_sensor" and o["zone_id"] not in gate_ids:
                raise ValueError(f"gate sensor outage on non-gate zone {o}")
            if not (0 <= o["from_second"] < o["until_second"]):
                raise ValueError(f"outage interval must satisfy 0 <= from < until: {o}")


@dataclass(frozen=True)
class RunContext:
    """Shared identity of one simulation run; the deterministic run_id belongs to Phase 3."""
    scenario_id: str
    run_mode: str
    seed: int
    epoch_utc: Any            # timezone-aware datetime: SIMULATED wall-clock origin, not a real time
    sample_seconds: int = 5


def _iso(ctx: RunContext, seconds: int) -> str:
    return instant_to_utc(ctx.epoch_utc, seconds).isoformat(timespec="seconds").replace("+00:00", "Z")


def _key(ctx: RunContext, source: str, zone: str, variable: str, t: int) -> bytes:
    return f"{ctx.scenario_id}|{ctx.run_mode}|{ctx.seed}|{source}|{zone}|{variable}|{t}".encode()


def _rng(key: bytes) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(b"obs|" + key).digest()[:8], "big"))


def generate_observations(states: list[dict], venue: dict, ctx: RunContext,
                          config: ObservationConfig | None = None, report_id_fn=None) -> list[dict]:
    """Reports for every observation instant, each derived only from the state at its observed_at.

    report_id_fn(source, zone, variable, t) -> str supplies keyed report IDs (Phase 4). When omitted the
    Phase 3 unkeyed ID is used (local bundles only). Report IDs never feed the noise RNG, so values,
    dropouts and delays are identical under either ID scheme."""
    config = config or ObservationConfig()
    zones = venue["zones"]
    zone_ids = {z["zone_id"] for z in zones}
    gate_ids = {z["zone_id"] for z in zones if z["zone_type"] == "exit"}
    config.validate(zone_ids, gate_ids, ctx.sample_seconds)
    area = {z["zone_id"]: z["effective_walkable_area_m2"] for z in zones}
    coverage = {
        "sim_camera_A": set(config.camera_a_zones) if config.camera_a_zones is not None else zone_ids,
        "sim_camera_B": set(config.camera_b_zones),
        "sim_gate_sensor": gate_ids,
    }
    scale = {"density": lambda z: 1.0 / area[z], "inflow": lambda z: 1.0 / ctx.sample_seconds,
             "outflow": lambda z: 1.0 / ctx.sample_seconds, "queue_length": lambda z: 1.0}
    transport_min = config.base_delay_seconds - config.publish_latency_seconds
    reports = []
    for state in states:
        t = state["simulation_second"]
        if t % config.observation_every_seconds:
            continue
        for source in ("sim_camera_A", "sim_camera_B", "sim_gate_sensor"):
            variables = GATE_VARIABLES if source == "sim_gate_sensor" else CAMERA_VARIABLES
            for zone in sorted(coverage[source]):
                zstate = state["zones"][zone]
                for variable in variables:
                    truth = zstate[variable]
                    if truth is None:
                        continue  # undefined or inapplicable truth is not measurable
                    if any(o["source_id"] == source and o["zone_id"] == zone and o["from_second"] <= t < o["until_second"]
                           for o in config.outages):
                        continue
                    key = _key(ctx, source, zone, variable, t)
                    rng = _rng(key)
                    if rng.random() < config.dropout_probability:
                        continue
                    if variable == "probability_exit_open":
                        value = 1.0 - truth if rng.random() < config.exit_state_error_probability else truth
                    else:
                        value = float(truth) * (1.0 + rng.gauss(0, config.camera_sigma[source]))
                        if variable in scale:
                            unit_scale = scale[variable](zone)
                            if truth > 0:
                                value += rng.gauss(0, config.count_noise_sigma_persons) * unit_scale
                            elif rng.random() < config.zero_false_positive_probability:
                                value = rng.randint(1, config.false_positive_max_persons) * unit_scale
                        value = max(0.0, value)
                        if variable == "queue_length":
                            value = int(round(value))
                    transport = transport_min + rng.randint(0, config.transport_jitter_seconds)
                    reports.append({
                        "report_id": (report_id_fn(source, zone, variable, t) if report_id_fn
                                      else "R-" + hashlib.sha256(b"id|" + key).hexdigest()[:20]),
                        "scenario_id": ctx.scenario_id,
                        "zone_id": zone, "source_id": source, "variable": variable,
                        "value": round(value, 6) if isinstance(value, float) else value,
                        "unit": UNITS[variable],
                        "simulation_second": t,
                        "observed_at": _iso(ctx, t),
                        "published_at": _iso(ctx, t + config.publish_latency_seconds),
                        "received_at": _iso(ctx, t + config.publish_latency_seconds + transport),
                        "source_type": OBSERVATION_SOURCE_TYPE,
                        "environment_id": ENVIRONMENT_ID, "map_version": MAP_VERSION, "schema_version": SCHEMA_VERSION,
                    })
    reports.sort(key=lambda r: (r["received_at"], r["report_id"]))
    return reports
