"""Twin Studio v3: A* exit recommendation on the canonical venue graph."""
import copy
import json
import random

import pytest

from services.twin_studio.v3 import engine
from services.twin_studio.v3.ingest import ZONE_SCHEMA, Measurement, SourceSpec
from services.twin_studio.v3.replay import AuthorizedSnapshot, topology_from_venue
from services.twin_studio.v3.routing import (P_GATE_UNVERIFIED, astar, evidence_from_snapshot, recommend, scene_alignment,
                                             validate_graph)
from services.twin_studio.zones import load_venue

VENUE = load_venue()
TOPO = topology_from_venue(VENUE)
GATES = set(TOPO.gate_ids)
SRC = (SourceSpec("zcam_1", "zone_camera", 5, 2, frozenset(TOPO.zone_ids), ZONE_SCHEMA),
       SourceSpec("gate_1", "gate_sensor", 5, 1, frozenset(GATES), ZONE_SCHEMA))


def _m(zone, var, value, observed, source="zcam_1"):
    delay = 2 if source == "zcam_1" else 1
    return Measurement(f"{source}:{zone}:{observed}", source, "zone_camera" if source == "zcam_1" else "gate_sensor",
                       "zone", zone, var, value, "x", observed, observed + delay, ZONE_SCHEMA)


def _snap(measurements, T=100, topo=TOPO):
    return AuthorizedSnapshot("t", T, tuple(measurements), (), (), topo, SRC)


def _fresh_world(T=100, gate_state=None, density=None, speed=None):
    ms = []
    for z in TOPO.zone_ids:
        ms.append(_m(z, "density", (density or {}).get(z, 0.3), T - 5))
        ms.append(_m(z, "speed", (speed or {}).get(z, 1.2), T - 5))
        ms.append(_m(z, "queue_length", 0.0, T - 5))
    for g, state in (gate_state if gate_state is not None else {g: 1.0 for g in GATES}).items():
        if state is not None:
            ms.append(_m(g, "exit_open", state, T - 5, source="gate_1"))
    return ms


def _valid_path(path):
    edges = {frozenset((a, b)) for _, a, b, *_ in TOPO.connections}
    return all(frozenset(p) in edges for p in zip(path, path[1:]))


def test_canonical_graph_matches_geometry():
    report = validate_graph(VENUE)
    assert report["valid"] and report["connections"] == 16 and len(report["gates"]) == 3


def test_tampered_graph_is_detected():
    bad = copy.deepcopy(VENUE)
    bad["connections"].append({**bad["connections"][0], "edge_id": "E_FAKE", "from_zone": "T_ENTRANCE", "to_zone": "T_ANNEX"})
    assert not validate_graph(bad)["valid"]


def test_route_found_uses_real_edges_and_verified_open_exit():
    rec = recommend(_snap(_fresh_world()), "T_COURTYARD", trigger="operator_request")
    assert rec["status"] == "ROUTE_FOUND" and rec["recommended_exit"] in GATES
    assert _valid_path(rec["path"]) and rec["path"][0] == "T_COURTYARD" and rec["path"][-1] == rec["recommended_exit"]
    assert rec["estimated_cost_s"] > 0 and rec["breakdown"] and rec["cost_units"].startswith("seconds")


def test_closed_exit_is_never_recommended():
    open_all = recommend(_snap(_fresh_world()), "T_ANNEX", trigger="t")
    assert open_all["recommended_exit"] == "T_GATE_E"
    rec = recommend(_snap(_fresh_world(gate_state={"T_GATE_E": 0.0, "T_GATE_SW": 1.0, "T_GATE_NW": 1.0})), "T_ANNEX", trigger="t")
    assert rec["status"] == "ROUTE_FOUND" and rec["recommended_exit"] != "T_GATE_E"
    assert "T_GATE_E" in rec["known_closures"] and all(o["exit"] != "T_GATE_E" for o in rec["alternatives"])


def test_missing_or_stale_exit_status_never_means_open():
    missing = recommend(_snap(_fresh_world(gate_state={})), "T_COURTYARD", trigger="t")
    assert missing["status"] == "NO_CONFIDENT_ROUTE" and missing["reason_codes"] == ["NO_EXIT_WITH_FRESH_OPEN_STATUS"]
    assert all(o["exit_status"] == "unverified" for o in missing["alternatives"])
    stale = [m for m in _fresh_world(gate_state={}) ] + [_m(g, "exit_open", 1.0, 10, "gate_1") for g in GATES]
    rec = recommend(_snap(stale, T=100), "T_COURTYARD", trigger="t")
    assert rec["status"] == "NO_CONFIDENT_ROUTE"
    assert any(o["breakdown"][-1].get("gate_unverified_s") == P_GATE_UNVERIFIED for o in rec["alternatives"])


def test_all_exits_closed_returns_no_confident_route():
    rec = recommend(_snap(_fresh_world(gate_state={g: 0.0 for g in GATES})), "T_COURTYARD", trigger="t")
    assert rec["status"] == "NO_CONFIDENT_ROUTE" and rec["reason_codes"] == ["ALL_EXITS_CLOSED_OR_UNREACHABLE"]


def test_congestion_and_missing_evidence_raise_costs():
    calm = recommend(_snap(_fresh_world()), "T_WALK_S", trigger="t")
    used = calm["path"][1]                                    # first zone the calm route enters
    jam = recommend(_snap(_fresh_world(speed={used: 0.05}, density={used: 6.0})), "T_WALK_S", trigger="t")
    assert jam["estimated_cost_s"] > calm["estimated_cost_s"]
    assert jam["path"] != calm["path"] or jam["breakdown"][0]["travel_s"] > calm["breakdown"][0]["travel_s"]
    via = calm["path"][1]                                     # costs are charged on zones the path enters
    thin = [m for m in _fresh_world() if m.entity_id != via]
    rec = recommend(_snap(thin), "T_WALK_S", trigger="t")
    legs = [leg for o in [rec] + rec.get("alternatives", []) for leg in o.get("breakdown", []) if leg["zone"] == via]
    assert legs and all(leg["uncertainty_s"] > 0 and leg["speed_basis"] == "free_walking_assumption" for leg in legs)


def test_disconnected_graph_has_no_route():
    cut = TOPO.__class__(**{**TOPO.__dict__, "connections": tuple(c for c in TOPO.connections
                                                                  if "T_COURTYARD" not in (c[1], c[2]))})
    rec = recommend(_snap(_fresh_world(), topo=cut), "T_COURTYARD", trigger="t")
    assert rec["status"] == "NO_CONFIDENT_ROUTE" and rec["reason_codes"] == ["ALL_EXITS_CLOSED_OR_UNREACHABLE"]


def test_heuristic_is_admissible_against_dijkstra():
    rng = random.Random(7)
    for _ in range(25):
        world = _fresh_world(density={z: rng.uniform(0, 5) for z in TOPO.zone_ids},
                             speed={z: rng.uniform(0.05, 2.5) for z in TOPO.zone_ids})
        ev = evidence_from_snapshot(_snap(world))
        start = rng.choice(TOPO.zone_ids)
        for g in GATES:
            with_h = astar(TOPO, ev, start, {g})
            no_h = astar(TOPO.__class__(**{**TOPO.__dict__, "zone_centroid_m": {z: (0.0, 0.0) for z in TOPO.zone_ids}}),
                         ev, start, {g})
            assert with_h and no_h and abs(with_h[1] - no_h[1]) < 1e-9


def test_unaligned_scene_disables_routing():
    scene = json.loads((engine.ARCHIVE_DIR / "examples/event_horizon_scene.json").read_text(encoding="utf-8"))
    assert scene_alignment(scene, VENUE)["aligned"]
    custom = copy.deepcopy(scene)
    custom["useOriginalVenue"] = False
    rec = recommend(_snap(_fresh_world()), "T_COURTYARD", trigger="t", alignment=scene_alignment(custom, VENUE))
    assert rec["status"] == "ROUTING_DISABLED"
    off_map = copy.deepcopy(scene)
    off_map["routes"][0]["points"] = [{"x": 5, "y": 5}, {"x": 60, "y": 5}, {"x": 60, "y": 60}]
    assert not scene_alignment(off_map, VENUE)["aligned"]
