"""Twin Studio -> main architecture pipeline adapter (Phase 5).

Converts serving-safe Twin Studio observation records (bundle file or observation API response)
into the main pipeline's ObservationRecord contract and runs crowd_reliability's
ArchitecturePipeline on the frozen twin_venue_v1 graph.

Contract:
  * Input is observation records ONLY. Each record must carry exactly the Phase 3 observation
    allowlist (bundle.OBS_RECORD_FIELDS) and environment eh_twin_studio / twin_venue_v1 /
    eh-twin-schema-1.0. Any extra field (e.g. a reference or label field), unknown variable, unit,
    zone, environment or schema raises AdapterError - nothing is dropped or guessed silently.
  * Six distinct channels are kept: Twin density/speed/inflow/outflow/queue_length/
    probability_exit_open map 1:1 onto the pipeline variables density/speed/inflow/outflow/
    queue_length/exit_open (tree channels estimated_* / probability_exit_open). Units are
    converted by name only; values are unchanged (both sides use persons/m2, m/s, persons/s,
    persons, 0/1).
  * Venue capacity is unknown (null) in twin_venue_v1. Zones are passed with
    capacity_definition=None, capacity_status="unknown"; the graph builder then uses 0.0 and
    reports them. No capacity is invented.
  * Environment separation: nothing is loaded from models/production. The decision-tree bank
    trained on the main simulator is NOT used: UnavailableTreeBank returns all six channels with
    mask 0 (unavailable). The hypernetwork/GNN/GRU are deterministic seeded initialisations,
    status "untrained_initial". See models/environments/eh_twin_studio/environment_manifest.json.
  * Determinism: torch init is seeded inside a forked RNG, and all modules are put in eval()
    mode. The upstream GRU (crowd_reliability/ml/temporal.py, Person 4) is constructed in train
    mode with inter-layer dropout and step() does not call eval(); this adapter calls
    temporal_model.eval() itself instead of modifying Person 4's module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION
from services.twin_studio.bundle import OBS_RECORD_FIELDS
from services.twin_studio.observations import SCHEMA_VERSION

TWIN_TO_PIPELINE_VARIABLE = {"density": "density", "speed": "speed", "inflow": "inflow", "outflow": "outflow",
                             "queue_length": "queue_length", "probability_exit_open": "exit_open"}
PIPELINE_TO_TWIN_VARIABLE = {v: k for k, v in TWIN_TO_PIPELINE_VARIABLE.items()}
TWIN_UNIT_TO_PIPELINE = {"persons/m2": "p/m2", "m/s": "m/s", "persons/s": "p/s", "persons": "persons",
                         "probability": "binary"}
EXPECTED_TWIN_UNIT = {"density": "persons/m2", "speed": "m/s", "inflow": "persons/s", "outflow": "persons/s",
                      "queue_length": "persons", "probability_exit_open": "probability"}
SUPPORTED_ZONE_TYPES = ("entrance", "corridor", "assembly", "stairway", "exit")
ENV_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "environments" / ENVIRONMENT_ID
STREAM_B_ID = "stream_b_architecture"
UNAVAILABLE_TREE_VERSION = "unavailable_for_eh_twin_studio"


class AdapterError(ValueError):
    """A record or venue field is unsupported by the pipeline contract. Never silently patched."""


def _utc(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise AdapterError("timestamps must be timezone-aware")
    return dt.astimezone(timezone.utc)


def venue_layout(venue: dict):
    """twin_venue_v1 -> crowd_reliability VenueLayout (capacity explicitly unknown)."""
    from crowd_reliability.simulation.scenarios import VenueLayout
    if venue.get("environment_id") != ENVIRONMENT_ID or venue.get("map_version") != MAP_VERSION:
        raise AdapterError("venue is not the frozen eh_twin_studio / twin_venue_v1 map")
    zones = []
    for z in venue["zones"]:
        if z["zone_type"] not in SUPPORTED_ZONE_TYPES:
            raise AdapterError(f"zone type {z['zone_type']!r} of {z['zone_id']} is unsupported by the pipeline")
        if z.get("capacity") is not None:
            raise AdapterError("twin_venue_v1 capacity is expected to be null; refusing an unreviewed value")
        zones.append({"zone_id": z["zone_id"], "name": z["name"], "zone_type": z["zone_type"],
                      "capacity_definition": None, "capacity_status": "unknown",
                      "area_m2": z["effective_walkable_area_m2"], "has_exit": bool(z["has_exit"]),
                      "map_version": MAP_VERSION})
    connections = [{"edge_id": c["edge_id"], "from_zone": c["from_zone"], "to_zone": c["to_zone"],
                    "width_m": c["width_m"], "distance_m": c["distance_m"], "direction": c["direction"]}
                   for c in venue["connections"]]
    return VenueLayout(zones=zones, connections=connections)


def to_observation_records(reports: Iterable[dict], *, run_id: str, zone_ids: set[str]) -> list:
    """Twin observation records -> transient (unsaved) ObservationRecord objects."""
    from crowd_reliability.models.schema import ObservationRecord
    out = []
    for r in reports:
        if set(r) != set(OBS_RECORD_FIELDS):
            extra, missing = sorted(set(r) - set(OBS_RECORD_FIELDS)), sorted(set(OBS_RECORD_FIELDS) - set(r))
            raise AdapterError(f"observation record fields not in the contract: extra={extra} missing={missing}")
        if (r["environment_id"], r["map_version"], r["schema_version"]) != (ENVIRONMENT_ID, MAP_VERSION, SCHEMA_VERSION):
            raise AdapterError("observation record from another environment, map or schema version")
        if r["run_id"] != run_id:
            raise AdapterError("observation record belongs to another run")
        if r["variable"] not in TWIN_TO_PIPELINE_VARIABLE:
            raise AdapterError(f"unsupported observation variable {r['variable']!r}")
        if r["unit"] != EXPECTED_TWIN_UNIT[r["variable"]]:
            raise AdapterError(f"unexpected unit {r['unit']!r} for {r['variable']}")
        if r["zone_id"] not in zone_ids:
            raise AdapterError(f"unknown zone {r['zone_id']!r}")
        if not isinstance(r["value"], (int, float)) or isinstance(r["value"], bool):
            raise AdapterError("observation value must be numeric")
        observed, published, received = _utc(r["observed_at"]), _utc(r["published_at"]), _utc(r["received_at"])
        if not observed < published <= received:
            raise AdapterError("observation timestamps violate observed < published <= received")
        out.append(ObservationRecord(
            report_id=r["report_id"], scenario_id=r["scenario_id"], zone_id=r["zone_id"], source_id=r["source_id"],
            variable=TWIN_TO_PIPELINE_VARIABLE[r["variable"]], value=float(r["value"]),
            unit=TWIN_UNIT_TO_PIPELINE[r["unit"]], observed_at=observed, published_at=published, received_at=received))
    return out


def _unavailable_tree_bank():
    from crowd_reliability.ml.tree_bank import TREE_CHANNELS, DecisionTreeBank

    class UnavailableTreeBank(DecisionTreeBank):
        """No tree bank exists for eh_twin_studio. All six channels are returned with mask 0
        (unavailable) instead of the unfitted bank's heuristic values with mask 1."""
        version = UNAVAILABLE_TREE_VERSION

        def predict_zone(self, feature_row, has_exit: bool = False):
            return [0.0] * len(TREE_CHANNELS), [0] * len(TREE_CHANNELS)

    return UnavailableTreeBank()


def environment_manifest() -> dict:
    path = ENV_MODEL_DIR / "environment_manifest.json"
    if not path.is_file():
        raise AdapterError("models/environments/eh_twin_studio/environment_manifest.json is missing")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("environment_id") != ENVIRONMENT_ID or doc.get("map_version") != MAP_VERSION:
        raise AdapterError("environment manifest does not describe eh_twin_studio / twin_venue_v1")
    return doc


@dataclass
class ForecastRow:
    zone_id: str
    variable: str            # Twin variable name (probability_exit_open, ...)
    horizon_seconds: int
    issue_second: int
    target_second: int
    value: float
    unit: str


class TwinArchitectureRunner:
    """Stream B for one Twin run: observations -> ArchitecturePipeline.step -> forecasts."""

    def __init__(self, venue: dict, run_id: str, epoch_utc: str, *, seed: int = 0,
                 horizons_seconds=(30, 60, 120)):
        import torch
        from crowd_reliability.ml.pipeline import ArchitecturePipeline
        from crowd_reliability.models.manifest import PipelineManifest
        self.manifest = environment_manifest()
        self.layout = venue_layout(venue)
        self.zone_ids = {z["zone_id"] for z in self.layout.zones}
        self.run_id, self.epoch = run_id, _utc(epoch_utc)
        pm = PipelineManifest(pipeline_version=self.manifest["pipeline_version"], tree_version=UNAVAILABLE_TREE_VERSION,
                              scaler_version="none", scenario_split="not_trained", is_active=False,
                              training_cutoff="none", training_status=dict(self.manifest["training_status"]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)            # GRU init is unseeded upstream; seed it here, locally
            self.pipeline = ArchitecturePipeline(venue=self.layout, manifest=pm,
                                                 horizons_sec=list(horizons_seconds), spatial_seed=seed)
        self.pipeline.tree_bank = _unavailable_tree_bank()
        self.pipeline.temporal_model.eval()     # see module docstring (Person 4 dependency)
        self.pipeline.hypernet.eval()
        self.pipeline.gnn.eval()
        self.unknown_capacity_zones = list(self.pipeline.graph_builder.unknown_capacity_zones)

    def step(self, issue_second: int, reports: list[dict]) -> list[ForecastRow]:
        """Forecasts issued at `issue_second`; only reports received by then are used."""
        as_of = self.epoch + timedelta(seconds=issue_second)
        records = [r for r in to_observation_records(reports, run_id=self.run_id, zone_ids=self.zone_ids)
                   if r.received_at <= as_of]
        _, forecasts = self.pipeline.step(self.run_id, as_of, records)
        rows = []
        for f in forecasts:
            target = int(round((f.target_at - self.epoch).total_seconds()))
            twin_var = PIPELINE_TO_TWIN_VARIABLE[f.variable]
            rows.append(ForecastRow(f.zone_id, twin_var, target - issue_second, issue_second, target,
                                    float(f.value), EXPECTED_TWIN_UNIT[twin_var]))
        return rows


def observation_persistence(issue_second: int, reports: list[dict], epoch_utc: str, *, horizons_seconds,
                            gate_ids: set[str]) -> list[ForecastRow]:
    """Twin Stream A (observation-only baseline): latest received report per zone/variable, held
    constant over every horizon. Uses observations received by issue time only."""
    epoch = _utc(epoch_utc)
    latest: dict = {}
    for r in reports:
        received = int((_utc(r["received_at"]) - epoch).total_seconds())
        if received > issue_second:
            continue
        observed = int((_utc(r["observed_at"]) - epoch).total_seconds())
        k = (r["zone_id"], r["variable"])
        if k not in latest or (observed, received, r["report_id"]) > latest[k][0]:
            latest[k] = ((observed, received, r["report_id"]), float(r["value"]))
    rows = []
    for (zone, variable), (_, value) in sorted(latest.items()):
        if variable == "probability_exit_open" and zone not in gate_ids:
            continue
        for h in horizons_seconds:
            rows.append(ForecastRow(zone, variable, h, issue_second, issue_second + h, value, EXPECTED_TWIN_UNIT[variable]))
    return rows


def forecast_digest(rows: list[ForecastRow]) -> str:
    payload = json.dumps([r.__dict__ for r in rows], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
