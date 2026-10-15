"""Operator-side assembly of replay sessions (never imported by model-side code).

The operator reads run outputs and passes the model only an AuthorizedSnapshot stream. From the
truth-bearing scene it extracts inference-safe metadata only (``safe_metadata``): route ids,
duration and reporting-source settings. Incidents, seeds, route populations and anything about
future events are not forwarded.
"""
from __future__ import annotations

import json
from pathlib import Path

from services.twin_studio.v3 import engine
from services.twin_studio.v3.ingest import IngestContext, route_source_specs
from services.twin_studio.v3.replay import ReplayController, topology_from_venue
from services.twin_studio.v3.zone_sensing import ZoneSource, source_specs
from services.twin_studio.zones import load_venue


def safe_metadata(scene: dict) -> dict:
    """The only scene content that may reach the model side."""
    return {"route_ids": [r["id"] for r in scene["routes"]], "duration_s": int(scene["duration"]),
            "route_sources": engine.observation_sources(scene)}


def build_context(run_id: str, meta: dict, zone_sources, venue: dict) -> IngestContext:
    zone_ids = [z["zone_id"] for z in venue["zones"]]
    gate_ids = [z["zone_id"] for z in venue["zones"] if z["zone_type"] == "exit"]
    sources = {**route_source_specs(meta["route_sources"], meta["route_ids"]),
               **source_specs(zone_sources, zone_ids, gate_ids)}
    return IngestContext(run_id, frozenset(meta["route_ids"]), frozenset(zone_ids), frozenset(gate_ids), sources)


def end_time(meta: dict, zone_sources) -> int:
    delays = [s["delay_s"] for s in meta["route_sources"]] + [s.delay_s for s in zone_sources]
    return meta["duration_s"] + max(delays, default=0)


def session_from_reports(session_id: str, reports: list[dict], meta: dict, zone_sources, *, venue: dict | None = None,
                         step_s: int = 5, end_s: int | None = None):
    venue = venue or load_venue()
    ctx = build_context(session_id, meta, zone_sources, venue)
    return ReplayController.create(session_id, reports, ctx, topology_from_venue(venue), step_s=step_s,
                                   end_s=end_s if end_s is not None else end_time(meta, zone_sources))


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def session_from_run_dir(run_dir: Path, mode: str, *, step_s: int = 5):
    """Replay session for one mode of a rendered run (canonical + zone-sensing observation streams)."""
    run_dir = Path(run_dir)
    scene = json.loads((run_dir / "hidden_ground_truth" / "scenario.json").read_text(encoding="utf-8"))
    meta = safe_metadata(scene)
    zone_sources = [ZoneSource(**s) for s in json.loads((run_dir / "hidden_ground_truth_ext" / "zone_sources.json").read_text(encoding="utf-8"))]
    reports = _jsonl(run_dir / "observable" / f"{mode}.jsonl") + _jsonl(run_dir / "observable_ext" / "zone_sensing" / f"{mode}.jsonl")
    return session_from_reports(f"{run_dir.name}:{mode}", reports, meta, zone_sources, step_s=step_s), meta, zone_sources
