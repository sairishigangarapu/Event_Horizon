"""Phase 1 (canonical data model) tests for services/twin_studio.

Every check recomputes from the raw venue.js rectangles or from real engine runs rather than
trusting the derived numbers stored in twin_venue_v1.json.
"""
import hashlib
import itertools
import json
from collections import deque
from pathlib import Path

import pytest

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION
from services.twin_studio import timebase
from services.twin_studio.engine import simulation
from services.twin_studio.zones import (OBSTACLE_KEY, OUTSIDE_KEY, STATUS_OBSTACLE, STATUS_OUTSIDE, STATUS_ZONE,
                                        ZoneAssigner, load_venue)

SUBSYSTEM = Path(__file__).resolve().parents[1] / "services" / "twin_studio"
EXPECTED_ZONES = {"T_ENTRANCE", "T_CONNECTOR", "T_WALK_N", "T_WALK_W", "T_WALK_S", "T_WALK_E",
                  "T_COURTYARD", "T_ANNEX", "T_GATE_NW", "T_GATE_SW", "T_GATE_E"}
GATES = {"T_GATE_NW", "T_GATE_SW", "T_GATE_E"}
# Raw rectangles exactly as in the archive's dist/venue.js `areas` (canvas units)
RAW_AREAS = {
    "T_ENTRANCE": (310, 55, 135, 145), "T_CONNECTOR": (445, 125, 80, 75), "T_WALK_N": (190, 200, 545, 90),
    "T_WALK_W": (190, 290, 90, 360), "T_WALK_S": (280, 575, 455, 75), "T_WALK_E": (645, 290, 90, 285),
    "T_COURTYARD": (280, 290, 365, 285), "T_ANNEX": (735, 435, 230, 215),
    "T_GATE_NW": (305, 55, 60, 12), "T_GATE_SW": (180, 590, 16, 50), "T_GATE_E": (958, 555, 14, 50),
}
RAW_POND = (415, 410, 78, 70)


@pytest.fixture(scope="module")
def venue():
    return load_venue()


@pytest.fixture(scope="module")
def assigner(venue):
    return ZoneAssigner(venue)


def _overlap(a, b):
    ax, ay, aw, ad = a
    bx, by, bw, bd = b
    return max(0, min(ax + aw, bx + bw) - max(ax, bx)), max(0, min(ay + ad, by + bd) - max(ay, by))


# ---------------------------------------------------------------- provenance

def test_original_engine_copies_are_byte_identical_to_archive_source():
    # Phase 2 modified engine/simulation.py; the archive originals are preserved under engine/original/.
    source = json.loads((SUBSYSTEM / "engine" / "SOURCE.json").read_text(encoding="utf-8"))
    py = hashlib.sha256((SUBSYSTEM / "engine" / "original" / "simulation.py").read_bytes()).hexdigest()
    js = hashlib.sha256((SUBSYSTEM / "engine" / "original" / "engine.js").read_bytes()).hexdigest()
    assert py == source["original"]["python"]["sha256"] == "ce39174c0881e687829f81afe07a6f769d04ed9ff0a06cf975d71cf263aa8c03"
    assert js == source["original"]["javascript"]["sha256"] == "866882f1836545cf3011d63cc6518710e2f412f9782a5d29e714fe9713e48b67"


def test_venue_identity(venue):
    assert venue["map_version"] == MAP_VERSION == "twin_venue_v1"
    assert venue["environment_id"] == ENVIRONMENT_ID == "eh_twin_studio"
    assert venue["coordinate_system"]["scale_m_per_unit"] == 0.1
    assert all(z["map_version"] == MAP_VERSION for z in venue["zones"])
    assert all(c["map_version"] == MAP_VERSION for c in venue["connections"])


# ---------------------------------------------------------------- zones and geometry

def test_exactly_the_eleven_approved_zones(venue):
    ids = [z["zone_id"] for z in venue["zones"]]
    assert len(ids) == len(set(ids)) == 11
    assert set(ids) == EXPECTED_ZONES
    for z in venue["zones"]:
        g = z["geometry"]
        assert (g["x"], g["y"], g["w"], g["d"]) == RAW_AREAS[z["zone_id"]]


def test_only_gates_are_exits_and_capacity_is_not_invented(venue):
    for z in venue["zones"]:
        assert z["has_exit"] == (z["zone_id"] in GATES) == (z["zone_type"] == "exit")
        assert z["capacity"] is None and "Not provided" in z["capacity_definition"]
        assert z["capacity_units"] == "persons"


def test_pond_is_the_only_obstacle_and_not_walkable(venue):
    assert [o["obstacle_id"] for o in venue["obstacles"]] == ["O_POND"]
    pond = venue["obstacles"][0]
    assert pond["walkable"] is False
    g = pond["geometry"]
    assert (g["x"], g["y"], g["w"], g["d"]) == RAW_POND


def test_walkable_areas_recomputed_from_raw_rectangles(venue):
    for z in venue["zones"]:
        x, y, w, d = RAW_AREAS[z["zone_id"]]
        units = w * d
        if z["zone_id"] not in GATES:
            for g in GATES:
                iw, ih = _overlap((x, y, w, d), RAW_AREAS[g])
                units -= iw * ih
            iw, ih = _overlap((x, y, w, d), RAW_POND)
            units -= iw * ih
        assert z["effective_walkable_area_m2"] == pytest.approx(units * 0.01)


def test_lattice_points_have_exactly_one_classification(assigner, venue):
    """Every integer point of the 1000 x 700 canvas is assigned exactly once, and the per-zone counts
    equal the walkable areas, so gate-over-host and pond exclusion are exact (half-open rectangles)."""
    counts = {zid: 0 for zid in EXPECTED_ZONES}
    counts[OBSTACLE_KEY] = 0
    counts[OUTSIDE_KEY] = 0
    for x in range(1000):
        for y in range(700):
            a = assigner.assign(x, y)
            if a.status == STATUS_ZONE:
                counts[a.zone_id] += 1
            elif a.status == STATUS_OBSTACLE:
                counts[OBSTACLE_KEY] += 1
            else:
                counts[OUTSIDE_KEY] += 1
    assert sum(counts.values()) == 1000 * 700
    for z in venue["zones"]:
        assert counts[z["zone_id"]] == round(z["effective_walkable_area_m2"] * 100), z["zone_id"]
    assert counts[OBSTACLE_KEY] == RAW_POND[2] * RAW_POND[3]


@pytest.mark.parametrize("point,expected", [
    ((340, 60), "T_GATE_NW"),     # inside the NW gate where it overlaps the main entrance
    ((306, 60), "T_GATE_NW"),     # gate part that sticks out beyond the entrance
    ((193, 600), "T_GATE_SW"),    # SW gate overlapping the west walkway
    ((960, 560), "T_GATE_E"),     # E gate overlapping the annex
    ((970, 560), "T_GATE_E"),     # E gate part beyond the annex
    ((340, 100), "T_ENTRANCE"),
    ((500, 150), "T_CONNECTOR"),
    ((450, 250), "T_WALK_N"),
    ((200, 400), "T_WALK_W"),
    ((500, 600), "T_WALK_S"),
    ((700, 400), "T_WALK_E"),
    ((350, 350), "T_COURTYARD"),
    ((800, 500), "T_ANNEX"),
])
def test_priority_and_interior_points(assigner, point, expected):
    a = assigner.assign(*point)
    assert (a.status, a.zone_id) == (STATUS_ZONE, expected)


def test_pond_and_outside_points(assigner):
    pond = assigner.assign(450, 440)
    assert (pond.status, pond.zone_id, pond.obstacle_id) == (STATUS_OBSTACLE, None, "O_POND")
    for p in [(50, 50), (990, 100), (100, 690), (600, 100)]:
        assert assigner.assign(*p).status == STATUS_OUTSIDE


def test_shared_boundary_points_belong_to_exactly_one_zone(assigner):
    # x = 280 is the west-walkway / courtyard boundary; y = 290 the north-walkway / courtyard boundary
    assert assigner.assign(280, 400).zone_id == "T_COURTYARD"
    assert assigner.assign(279.999, 400).zone_id == "T_WALK_W"
    assert assigner.assign(400, 290).zone_id == "T_COURTYARD"
    assert assigner.assign(400, 289.999).zone_id == "T_WALK_N"


# ---------------------------------------------------------------- connections

def _geometric_contacts():
    """All zone pairs that physically touch (shared boundary of positive length) or overlap (gate on host)."""
    contacts = {}
    for a, b in itertools.combinations(sorted(RAW_AREAS), 2):
        ra, rb = RAW_AREAS[a], RAW_AREAS[b]
        iw, ih = _overlap(ra, rb)
        if iw > 0 and ih > 0:  # overlap: only allowed between a gate and its host
            assert (a in GATES) != (b in GATES), f"non-gate overlap {a} {b}"
            gw, gd = (ra if a in GATES else rb)[2:]
            contacts[frozenset((a, b))] = iw if gw >= gd else ih
            continue
        ax, ay, aw, ad = ra
        bx, by, bw, bd = rb
        if (ax + aw == bx or bx + bw == ax) and ih > 0:
            contacts[frozenset((a, b))] = ih
        elif (ay + ad == by or by + bd == ay) and iw > 0:
            contacts[frozenset((a, b))] = iw
    return contacts


def test_sixteen_connections_match_geometry_exactly(venue):
    conns = venue["connections"]
    assert len(conns) == 16
    ids = [c["edge_id"] for c in conns]
    assert len(set(ids)) == 16
    pairs = [frozenset((c["from_zone"], c["to_zone"])) for c in conns]
    assert len(set(pairs)) == 16, "duplicate or reverse-duplicate physical connection"
    contacts = _geometric_contacts()
    assert set(pairs) == set(contacts), "connections must be exactly the geometric contacts"
    for c in conns:
        assert c["from_zone"] in EXPECTED_ZONES and c["to_zone"] in EXPECTED_ZONES
        assert c["from_zone"] != c["to_zone"]
        assert c["width_m"] == pytest.approx(contacts[frozenset((c["from_zone"], c["to_zone"]))] * 0.1)
        assert c["distance_m"] > 0


def test_connections_are_physical_not_message_passing(venue):
    for c in venue["connections"]:
        assert c["direction"] == "bidirectional"
        assert c["physically_traversable"] is True
        assert c["derivation"] in ("shared_boundary", "gate_attachment")
        assert not c["edge_id"].endswith("__rev")
    gate_links = [c for c in venue["connections"] if c["derivation"] == "gate_attachment"]
    assert {c["to_zone"] for c in gate_links} == GATES
    assert {(c["from_zone"], c["to_zone"]) for c in gate_links} == {
        ("T_ENTRANCE", "T_GATE_NW"), ("T_WALK_W", "T_GATE_SW"), ("T_ANNEX", "T_GATE_E")}


def test_every_zone_reachable_and_every_gate_reachable_from_entrance(venue):
    adj = {z: set() for z in EXPECTED_ZONES}
    for c in venue["connections"]:
        adj[c["from_zone"]].add(c["to_zone"])
        adj[c["to_zone"]].add(c["from_zone"])
    seen, queue = {"T_ENTRANCE"}, deque(["T_ENTRANCE"])
    while queue:
        for n in adj[queue.popleft()]:
            if n not in seen:
                seen.add(n)
                queue.append(n)
    assert seen == EXPECTED_ZONES
    for g in GATES:
        assert len(adj[g]) == 1  # each gate attaches to exactly its host area


# ---------------------------------------------------------------- occupancy accounting on real engine frames

SCENE = {
    "duration": 60, "scale": 0.1, "seed": 11, "randomCount": 1, "events": [],
    "routes": [
        # entrance -> north walkway -> through the courtyard and the pond -> south walkway
        {"id": "r1", "name": "Through pond", "points": [{"x": 380, "y": 100}, {"x": 450, "y": 250},
                                                       {"x": 450, "y": 440}, {"x": 450, "y": 620}],
         "count": 60, "speed": 1.4, "width": 3},
        # west walkway into the SW gate
        {"id": "r2", "name": "To SW gate", "points": [{"x": 230, "y": 320}, {"x": 230, "y": 615}, {"x": 185, "y": 615}],
         "count": 40, "speed": 1.2, "width": 3},
        # starts outside every zone
        {"id": "r3", "name": "Outside", "points": [{"x": 40, "y": 650}, {"x": 120, "y": 650}],
         "count": 20, "speed": 1.0, "width": 2},
    ],
}


def test_engine_frames_count_every_agent_exactly_once(assigner):
    result = simulation.simulate(SCENE, "ideal")
    seen_status = set()
    for frame in result["frames"]:
        counts = assigner.count((a["x"], a["y"]) for a in frame)
        assert sum(counts.values()) == len(frame)
        for a in frame:
            seen_status.add(assigner.assign(a["x"], a["y"]).status)
    assert seen_status == {STATUS_ZONE, STATUS_OBSTACLE, STATUS_OUTSIDE}


# ---------------------------------------------------------------- end-of-interval time mapping

def test_step_instant_mapping():
    assert timebase.step_interval(0) == (0, 1)
    assert timebase.step_instant(0) == 1
    assert timebase.instant_step(1) == 0
    assert timebase.instant_step(0) is None  # instant 0 = empty venue, no engine frame
    for k in range(50):
        assert timebase.instant_step(timebase.step_instant(k)) == k
    with pytest.raises(ValueError):
        timebase.step_interval(-1)


def test_engine_frames_are_end_of_interval_states():
    result = simulation.simulate(SCENE, "ideal")
    assert len(result["frames"]) == SCENE["duration"] + 1
    assert timebase.last_instant(SCENE["duration"]) == SCENE["duration"] + 1
    first = next(a for a in result["frames"][0] if a["ri"] == 0)
    start = SCENE["routes"][0]["points"][0]
    # The agent spawned at t=0 has already moved once in frames[0], so frames[0] is the state at instant 1 s.
    assert first["v"] > 0
    assert (first["x"], first["y"]) != (start["x"], start["y"])


def test_canonical_five_second_grid():
    grid = timebase.canonical_instants(60, 5)
    assert grid[0] == 0 and grid[-1] == 60 and all(b - a == 5 for a, b in zip(grid, grid[1:]))
    assert all(t <= timebase.last_instant(60) for t in grid)
    assert list(timebase.interval_steps(10, 5)) == [5, 6, 7, 8, 9]   # steps covering [5, 10)
    assert list(timebase.interval_steps(0, 5)) == []
    assert [timebase.instant_step(t) for t in (5, 10)] == [4, 9]


def test_utc_mapping_requires_aware_epoch():
    from datetime import datetime, timezone
    epoch = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)
    assert timebase.instant_to_utc(epoch, 5).isoformat() == "2026-10-15T09:00:05+00:00"
    with pytest.raises(ValueError):
        timebase.instant_to_utc(datetime(2026, 10, 15, 9, 0), 5)
