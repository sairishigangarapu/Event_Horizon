"""A* exit and mitigation recommendation on the canonical venue graph (Phase I, eh-v3-routing-1).

Decision support only: it recommends a zone path to an exit gate. It does not change how Twin
Studio agents move (they follow their configured polylines).

Graph: twin_venue_v1 zones and its 16 physical bidirectional connections, re-validated against
the rectangle geometry (validate_graph). Scenes whose routes do not align with this graph disable
recommendations (scene_alignment) instead of inventing connections.

Objective: lowest estimated traversal time (seconds) plus explicit penalties. Per edge u -> v:
  travel_s    = distance_m / v_est, where v_est is v's fresh observed speed clamped to
                [V_FLOOR, V_MAX], else V_FREE / slowdown(density) when only fresh density exists,
                else V_FREE (slowdown is the v3 engine's own rule: 1 + max(0, d - 0.5) * 1.8)
  queue_s     = fresh queue estimate in v * 0.65 m (engine single-file spacing) / v_est
  uncertain_s = P_UNCERTAIN when v has no fresh evidence (stale or missing)
Gates: a fresh "closed" report removes the gate; stale or missing status adds P_GATE_UNVERIFIED and
the route cannot be confident; only a fresh "open" report makes a gate verified. Capacity is
unknown in twin_venue_v1 and is not used (no capacity is invented).
Heuristic: straight-line centroid distance to the nearest candidate gate / V_MAX. Edge distances are
centroid distances, so path length >= straight-line distance; v_est <= V_MAX and penalties >= 0, so
the heuristic never overestimates (admissible).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from services.twin_studio.v3.replay import AuthorizedSnapshot, StaticTopology
from services.twin_studio.v3.staleness import StalenessPolicy
from services.twin_studio.zones import ZoneAssigner

VERSION = "eh-v3-routing-1"
V_FREE = 1.3            # m/s, the archive's default route speed
V_FLOOR = 0.2           # m/s, documented lower bound for a traversable zone
V_MAX = 3.45            # m/s, engine maximum (3.0 m/s route speed x 1.15 draw)
SPACING_M = 0.65        # engine single-file spacing
P_UNCERTAIN = 20.0      # s, zone without fresh evidence
P_GATE_UNVERIFIED = 60.0  # s, gate without a fresh status report


# ---------------------------------------------------------------- graph validation

def _rect(z):
    g = z["geometry"]
    return g["x"], g["y"], g["x"] + g["w"], g["y"] + g["d"]


def validate_graph(venue: dict) -> dict:
    """Recompute physical adjacency from rectangles and compare with the declared connections."""
    zones = {z["zone_id"]: z for z in venue["zones"]}
    gates = {z for z, d in zones.items() if d["zone_type"] == "exit"}
    declared = {frozenset((c["from_zone"], c["to_zone"])) for c in venue["connections"]}
    problems = []
    for c in venue["connections"]:
        if c["from_zone"] not in zones or c["to_zone"] not in zones:
            problems.append(f"connection {c['edge_id']} references an unknown zone")
        if c.get("direction") != "bidirectional":
            problems.append(f"connection {c['edge_id']} is not bidirectional")
    derived = set()
    ids = sorted(zones)
    for i, a in enumerate(ids):
        ax0, ay0, ax1, ay1 = _rect(zones[a])
        for b in ids[i + 1:]:
            bx0, by0, bx1, by1 = _rect(zones[b])
            ox, oy = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
            if (a in gates) != (b in gates) and ox > 0 and oy > 0:
                derived.add(frozenset((a, b)))                      # gate overlapping its host area
            elif a not in gates and b not in gates and ((ox == 0 and oy > 0) or (oy == 0 and ox > 0)):
                derived.add(frozenset((a, b)))                      # shared boundary segment
    if derived != declared:
        problems.append(f"declared connections differ from geometry: missing={sorted(map(sorted, derived - declared))} "
                        f"extra={sorted(map(sorted, declared - derived))}")
    for g in gates:
        if not any(g in e for e in declared):
            problems.append(f"gate {g} has no connection")
    return {"valid": not problems, "problems": problems, "zones": len(zones), "connections": len(declared),
            "gates": sorted(gates)}


def scene_alignment(scene: dict, venue: dict, max_outside_fraction: float = 0.05) -> dict:
    """Do the scene's routes stay on the canonical venue and move only along physical connections?"""
    from services.twin_studio.v3.engine import simulation_module
    sim = simulation_module()
    if not scene.get("useOriginalVenue", False):
        return {"aligned": False, "reason": "scene does not use the canonical venue geometry"}
    assigner = ZoneAssigner(venue)
    edges = {frozenset((c["from_zone"], c["to_zone"])) for c in venue["connections"]}
    details = {}
    aligned = True
    for r in scene["routes"]:
        p = sim.path(r, scene["scale"])
        n = max(2, int(p["length"] / 0.5))
        seq, outside = [], 0
        for k in range(n + 1):
            pos = sim.position(r, p, p["length"] * k / n)
            z = assigner.assign(pos["x"], pos["y"]).zone_id
            if z is None:
                outside += 1
                continue
            if not seq or seq[-1] != z:
                seq.append(z)
        bad = [(a, b) for a, b in zip(seq, seq[1:]) if frozenset((a, b)) not in edges]
        frac = outside / (n + 1)
        ok = frac <= max_outside_fraction and not bad
        aligned &= ok
        details[r["id"]] = {"zone_sequence": seq, "outside_fraction": round(frac, 4), "non_adjacent_transitions": bad,
                            "aligned": ok}
    return {"aligned": aligned, "routes": details}


# ---------------------------------------------------------------- evidence and A*

@dataclass
class ZoneEvidence:
    speed: float | None = None
    density: float | None = None
    queue: float | None = None
    exit_open: float | None = None
    fresh: bool = False
    observed_at: dict = field(default_factory=dict)
    stale_variables: list = field(default_factory=list)


def evidence_from_snapshot(snapshot: AuthorizedSnapshot, policy: StalenessPolicy = StalenessPolicy()) -> dict:
    """Latest fresh zone-level evidence per zone (stale evidence is recorded but not used as fresh)."""
    T = snapshot.as_of_s
    src = {s.source_id: s for s in snapshot.sources}
    ev = {z: ZoneEvidence() for z in snapshot.topology.zone_ids}
    for var in ("speed", "density", "queue_length", "exit_open"):
        for z in snapshot.topology.zone_ids:
            m = snapshot.latest("zone", z, var)
            if m is None:
                continue
            s = src[m.source_id]
            fresh = T - m.observed_at_s <= policy.max_age(s.delay_s, s.interval_s)
            e = ev[z]
            e.observed_at[var] = m.observed_at_s
            if not fresh:
                e.stale_variables.append(var)
                continue
            e.fresh = True
            setattr(e, "queue" if var == "queue_length" else var, float(m.value))
    return ev


def _edge_cost(dest: str, distance: float, ev: ZoneEvidence) -> tuple[float, dict]:
    if ev.speed is not None:
        v, basis = min(V_MAX, max(V_FLOOR, ev.speed)), "observed_speed"
    elif ev.density is not None:
        v, basis = min(V_MAX, max(V_FLOOR, V_FREE / (1 + max(0.0, ev.density - 0.5) * 1.8))), "density_slowdown"
    else:
        v, basis = V_FREE, "free_walking_assumption"
    travel = distance / v
    queue = (ev.queue or 0.0) * SPACING_M / v if ev.queue else 0.0
    uncertain = 0.0 if ev.fresh else P_UNCERTAIN
    return travel + queue + uncertain, {"zone": dest, "distance_m": distance, "speed_m_s": round(v, 3), "speed_basis": basis,
                                        "travel_s": round(travel, 2), "queue_s": round(queue, 2), "uncertainty_s": uncertain}


def _gate_status(ev: ZoneEvidence) -> str:
    if ev.exit_open is not None:
        return "open_verified" if ev.exit_open >= 0.5 else "closed_verified"
    return "unverified"


def astar(topology: StaticTopology, evidence: dict, start: str, goals: set) -> tuple[list, float, list] | None:
    adj = {z: [] for z in topology.zone_ids}
    for _, a, b, _, dist, ok in topology.connections:
        if ok:
            adj[a].append((b, dist))
            adj[b].append((a, dist))
    cent = topology.zone_centroid_m

    def h(z):
        return min(math.dist(cent[z], cent[g]) for g in goals) / V_MAX

    frontier = [(h(start), 0.0, start)]
    best = {start: 0.0}
    came: dict = {}
    while frontier:
        _, g, z = heapq.heappop(frontier)
        if z in goals:
            path, legs = [z], []
            while z in came:
                prev, leg = came[z]
                legs.append(leg)
                path.append(prev)
                z = prev
            return path[::-1], g, legs[::-1]
        if g > best.get(z, math.inf):
            continue
        for nxt, dist in adj[z]:
            cost, leg = _edge_cost(nxt, dist, evidence[nxt])
            if nxt in goals and _gate_status(evidence[nxt]) == "unverified":
                cost += P_GATE_UNVERIFIED
                leg["gate_unverified_s"] = P_GATE_UNVERIFIED
            ng = g + cost
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                came[nxt] = (z, leg)
                heapq.heappush(frontier, (ng + h(nxt), ng, nxt))
    return None


def recommend(snapshot: AuthorizedSnapshot, start_zone: str, *, trigger: str, alignment: dict | None = None,
              policy: StalenessPolicy = StalenessPolicy()) -> dict:
    topo = snapshot.topology
    base = {"version": VERSION, "as_of_s": snapshot.as_of_s, "start_zone": start_zone, "trigger": trigger,
            "cost_units": "seconds (estimated traversal time plus explicit penalties)"}
    if alignment is not None and not alignment.get("aligned", False):
        return base | {"status": "ROUTING_DISABLED", "reason": "scene routes do not align with the canonical venue graph",
                       "alignment": alignment}
    if start_zone not in topo.zone_ids:
        raise ValueError(f"unknown start zone {start_zone!r}")
    ev = evidence_from_snapshot(snapshot, policy)
    gate_status = {g: _gate_status(ev[g]) for g in topo.gate_ids}
    closed = sorted(g for g, s in gate_status.items() if s == "closed_verified")
    candidates = [g for g in topo.gate_ids if g not in closed]
    used_obs = {z: {"observed_at_s": e.observed_at, "stale_variables": e.stale_variables} for z, e in ev.items() if e.observed_at}
    base |= {"gate_status": gate_status, "known_closures": closed, "evidence_used": used_obs}
    options = []
    for g in candidates:
        found = astar(topo, ev, start_zone, {g}) if start_zone != g else ([g], 0.0, [])
        if found:
            path, cost, legs = found
            options.append({"exit": g, "exit_status": gate_status[g], "path": path, "estimated_cost_s": round(cost, 2),
                            "breakdown": legs,
                            "zones_without_fresh_evidence": [z for z in path if not ev[z].fresh]})
    options.sort(key=lambda o: o["estimated_cost_s"])
    if not options:
        return base | {"status": "NO_CONFIDENT_ROUTE", "reason_codes": ["ALL_EXITS_CLOSED_OR_UNREACHABLE"], "alternatives": []}
    verified = [o for o in options if o["exit_status"] == "open_verified"]
    if not verified:
        return base | {"status": "NO_CONFIDENT_ROUTE", "reason_codes": ["NO_EXIT_WITH_FRESH_OPEN_STATUS"],
                       "alternatives": options}
    best = verified[0]
    return base | {"status": "ROUTE_FOUND", "recommended_exit": best["exit"], "path": best["path"],
                   "estimated_cost_s": best["estimated_cost_s"], "breakdown": best["breakdown"],
                   "zones_without_fresh_evidence": best["zones_without_fresh_evidence"],
                   "alternatives": [o for o in options if o is not best]}
