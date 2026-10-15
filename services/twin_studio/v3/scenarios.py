"""Monte Carlo scenario variation for Twin Studio v3 (Phase E).

Each scenario varies documented scene parameters, not only the seed, around the canonical example
scene's two routes (their geometry is kept: it is the geometry validated against twin_venue_v1).
Every parameter is recorded with the scenario. Regimes are explicit metadata used for coverage
analysis and stratified evaluation; they never select a model at inference time.

  regime               incidents                                   reporting
  normal               none                                        typical settings
  surge                1-2 surges                                  typical settings
  blockage             1-2 collapses (half placed on a gate)       typical settings
  observation_outage   none                                        high missingness, longer delays, partial coverage
  compound             surge + collapse                            typical or degraded settings

Splits are by whole scenario, stratified by regime (no adjacent-timestamp leakage).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from services.twin_studio.v3.engine import ARCHIVE_DIR, simulation_module
from services.twin_studio.v3.zone_sensing import ZoneSource
from services.twin_studio.zones import ZoneAssigner, load_venue

REGIMES = ("normal", "surge", "blockage", "observation_outage", "compound")
SPLIT_FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
GENERATOR_VERSION = "eh-v3-mc-1"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    regime: str
    split: str
    seed: int
    params: dict
    scene: dict
    zone_sources: tuple

    def record(self) -> dict:
        return {"scenario_id": self.scenario_id, "regime": self.regime, "split": self.split, "seed": self.seed,
                "generator_version": GENERATOR_VERSION, "params": self.params,
                "zone_sources": [s.to_dict() for s in self.zone_sources]}


def base_scene() -> dict:
    return json.loads((ARCHIVE_DIR / "examples" / "event_horizon_scene.json").read_text(encoding="utf-8"))


def gate_positions(scene: dict, venue: dict | None = None) -> list[tuple[str, str, float]]:
    """(route_id, gate_id, fraction along the route) where a route passes through a gate zone."""
    sim = simulation_module()
    assigner = ZoneAssigner(venue or load_venue())
    gates = {z["zone_id"] for z in assigner.venue["zones"] if z["zone_type"] == "exit"}
    out = []
    for r in scene["routes"]:
        p = sim.path(r, scene["scale"])
        n = max(2, int(p["length"] / 0.25))
        hits: dict[str, list[float]] = {}
        for k in range(n + 1):
            s = p["length"] * k / n
            pos = sim.position(r, p, s)
            z = assigner.assign(pos["x"], pos["y"]).zone_id
            if z in gates:
                hits.setdefault(z, []).append(s)
        for g, ss in sorted(hits.items()):
            out.append((r["id"], g, (min(ss) + max(ss)) / 2 / p["length"]))
    return out


def _rng(master_seed: int, regime: str, index: int) -> np.random.Generator:
    h = hashlib.sha256(f"{GENERATOR_VERSION}|{master_seed}|{regime}|{index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _split(index: int, n: int) -> str:
    cut_train = math.floor(n * SPLIT_FRACTIONS[0][1] + 1e-9)
    cut_val = cut_train + max(1, math.floor(n * SPLIT_FRACTIONS[1][1] + 1e-9))
    return "train" if index < cut_train else "val" if index < cut_val else "test"


def _route_sources(rng, routes: list[str], degraded: bool) -> list[dict]:
    m = (lambda lo, hi: float(rng.uniform(lo, hi)))
    miss = (lambda lo, hi: m(0.35, 0.6) if degraded else m(lo, hi))
    stretch = 2 if degraded else 1
    cam_cov = routes if not degraded or rng.random() < 0.5 else routes[: max(1, len(routes) - 1)]
    return [
        {"id": "camera_1", "kind": "camera", "route_ids": list(cam_cov), "interval_s": int(rng.choice([1, 2, 3, 5])),
         "delay_s": int(rng.integers(1, 7)) * stretch, "missing_probability": miss(0.02, 0.2), "relative_noise": m(0.03, 0.12)},
        {"id": "sensor_1", "kind": "sensor", "route_ids": list(routes), "interval_s": int(rng.choice([1, 2])),
         "delay_s": int(rng.integers(0, 4)) * stretch, "missing_probability": miss(0.01, 0.08), "relative_noise": m(0.01, 0.05)},
        {"id": "warden_1", "kind": "warden", "route_ids": list(routes), "interval_s": int(rng.choice([5, 10, 15])),
         "delay_s": int(rng.integers(3, 11)) * stretch, "missing_probability": miss(0.05, 0.25), "relative_noise": m(0.05, 0.2)},
    ]


def _zone_sources(rng, zone_ids: list[str], degraded: bool) -> tuple:
    m = (lambda lo, hi: float(rng.uniform(lo, hi)))
    miss = (lambda lo, hi: m(0.35, 0.6) if degraded else m(lo, hi))
    stretch = 2 if degraded else 1
    cam_zones = None
    if degraded:
        keep = rng.permutation(zone_ids)[: max(3, int(len(zone_ids) * 0.6))]
        cam_zones = tuple(sorted(keep.tolist()))
    sources = [
        ZoneSource("zcam_1", "zone_camera", int(rng.choice([5, 10])), int(rng.integers(1, 7)) * stretch, miss(0.02, 0.15),
                   relative_noise=m(0.04, 0.12), zone_ids=cam_zones),
        ZoneSource("zcount_1", "zone_counter", int(rng.choice([5, 10])), int(rng.integers(1, 5)) * stretch, miss(0.01, 0.08),
                   relative_noise=m(0.01, 0.05)),
        ZoneSource("gate_1", "gate_sensor", int(rng.choice([5, 10])), int(rng.integers(0, 4)) * stretch, miss(0.01, 0.08),
                   error_probability=m(0.0, 0.05)),
    ]
    if rng.random() < 0.35:                      # redundant partial coverage makes source conflicts possible
        extra = tuple(sorted(rng.permutation(zone_ids)[:4].tolist()))
        sources.append(ZoneSource("zcam_2", "zone_camera", 10, int(rng.integers(2, 9)), m(0.05, 0.2),
                                  relative_noise=m(0.08, 0.2), zone_ids=extra))
    return tuple(sources)


def sample_scenario(regime: str, index: int, n_in_regime: int, master_seed: int = 2026) -> Scenario:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    rng = _rng(master_seed, regime, index)
    scene = base_scene()
    scene.pop("background", None)
    duration = int(rng.choice([240, 270, 300, 330, 360]))
    seed = int(rng.integers(0, 2**31 - 1))
    routes = scene["routes"]
    for r in routes:
        r["count"] = int(rng.integers(60, 241))
        r["speed"] = round(float(rng.uniform(1.0, 1.6)), 3)
    route_ids = [r["id"] for r in routes]
    events = []
    if regime in ("surge", "compound"):
        for i in range(int(rng.integers(1, 3))):
            start = int(duration * rng.uniform(0.1, 0.6))
            end = min(duration, start + int(rng.integers(10, 61)))
            events.append({"id": f"surge{i}", "name": f"Surge {i + 1}", "type": "surge", "routeId": str(rng.choice(route_ids)),
                           "at": round(float(rng.uniform(0.1, 0.8)), 4), "start": start, "end": end,
                           "people": int(rng.integers(50, 301))})
    if regime in ("blockage", "compound"):
        gates = gate_positions(scene)
        for i in range(int(rng.integers(1, 3))):
            if gates and rng.random() < 0.5:
                route_id, gate_id, at = gates[int(rng.integers(len(gates)))]
                placement = f"gate:{gate_id}"
            else:
                route_id, at, placement = str(rng.choice(route_ids)), float(rng.uniform(0.15, 0.9)), "route"
            start = int(duration * rng.uniform(0.15, 0.6))
            end = min(duration, start + int(rng.integers(30, 151)))
            events.append({"id": f"collapse{i}", "name": f"Collapse {i + 1}", "type": "collapse", "routeId": route_id,
                           "at": round(at, 4), "start": start, "end": end, "people": 0,
                           "radiusM": round(float(rng.uniform(1.5, 6.0)), 2), "_placement": placement})
    degraded = regime == "observation_outage" or (regime == "compound" and rng.random() < 0.5)
    placements = [e.pop("_placement") for e in events if "_placement" in e]
    scene.update({"duration": duration, "seed": seed, "randomCount": int(rng.integers(1, 5)), "events": events,
                  "observationSources": _route_sources(rng, route_ids, degraded)})
    total = sum(r["count"] for r in routes) + sum(e.get("people", 0) for e in events)
    if total > 3000:
        raise ValueError("sampled population exceeds the engine limit")
    simulation_module().validate(scene)
    zone_ids = [z["zone_id"] for z in load_venue()["zones"]]
    params = {"duration_s": duration, "route_counts": {r["id"]: r["count"] for r in routes},
              "route_speeds": {r["id"]: r["speed"] for r in routes}, "events": events, "collapse_placements": placements,
              "degraded_reporting": degraded, "route_sources": scene["observationSources"]}
    sid = f"mc_{regime}_{index:03d}"
    return Scenario(sid, regime, _split(index, n_in_regime), seed, params, scene, _zone_sources(rng, zone_ids, degraded))


def sample_all(n_per_regime: int, master_seed: int = 2026) -> list[Scenario]:
    return [sample_scenario(reg, i, n_per_regime, master_seed) for reg in REGIMES for i in range(n_per_regime)]
