"""Zone-level reference states from v3 per-agent frames (offline: hidden truth only).

v3 frame semantics (archive simulation.py): frame t is the state at instant t, recorded before the
step's movement; an agent's ``v`` in frame t is its displacement over [t-1, t]; an agent first
appears in the frame of the instant it is admitted (with v = 0, which is not a measurement);
agents absent from frame t but present in frame t-1 completed their route during [t-1, t].

Fields at instant T (sample grid, default 5 s), per zone of twin_venue_v1:
  occupancy      agents assigned to the zone at T (ZoneAssigner: gate > obstacle > host area)
  density        occupancy / effective walkable area (persons/m2)
  queue_length   agents present at T with v < 0.1 m/s that were already present at T-1
                 (the engine's own "queued" rule: speed < 0.1 and admitted before t)
  speed          mean over agents present at T of each agent's mean v over frames in (T-dt, T] in
                 which it moved from a previous frame (m/s); None when no agent qualifies
  inflow/outflow zone entries / exits over transitions in (T-dt, T], divided by dt (persons/s);
                 admissions count as entries, route completions as exits; None at T = 0
  exit_open      gate zones only: 0.0 if an active v3 barrier circle intersects the gate rectangle
                 at T (barrier active for start <= T < end), else 1.0. Non-gate zones: None,
                 exit_applicable False.

Route-level aggregates (rows / metrics.csv) are never divided among zones.
Known limit: a zone crossed entirely within one 1 s step is not registered as visited.
"""
from __future__ import annotations

import math
from collections import defaultdict

from services.twin_studio.zones import OBSTACLE_KEY, OUTSIDE_KEY, STATUS_ZONE, ZoneAssigner, load_venue

QUEUE_SPEED_M_S = 0.1
SCHEMA_VERSION = "eh-v3-zone-states-1"
LABEL_BASIS_EXIT = "barrier_circle_intersects_gate_rectangle"
VARIABLES = ("density", "speed", "inflow", "outflow", "queue_length", "exit_open")


class ZoneStateError(ValueError):
    pass


def _place(assigner: ZoneAssigner, x: float, y: float) -> str:
    a = assigner.assign(x, y)
    if a.status == STATUS_ZONE:
        return a.zone_id
    return OBSTACLE_KEY if a.obstacle_id else OUTSIDE_KEY


def _circle_hits_rect(cx_m: float, cy_m: float, r_m: float, rect_m: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = rect_m
    nx, ny = min(max(cx_m, x0), x1), min(max(cy_m, y0), y1)
    return (cx_m - nx) ** 2 + (cy_m - ny) ** 2 <= r_m * r_m


def gate_blocked(barriers: list[dict], gate_rect_m: tuple, scale: float, instant_s: int) -> bool:
    for b in barriers:
        if b["start"] <= instant_s < b["end"]:
            if _circle_hits_rect(b["center"]["x"] * scale, b["center"]["y"] * scale, float(b["radiusM"]), gate_rect_m):
                return True
    return False


def derive_zone_states(result: dict, scene: dict, venue: dict | None = None, sample_seconds: int = 5) -> list[dict]:
    """Zone states at instants 0, dt, 2dt, ... <= duration. One dict per instant."""
    venue = venue or load_venue()
    assigner = ZoneAssigner(venue)
    scale = float(scene["scale"])
    if abs(scale - assigner.scale_m_per_unit) > 1e-12:
        raise ZoneStateError(f"scene scale {scale} differs from venue scale {assigner.scale_m_per_unit}")
    zones = venue["zones"]
    area = {z["zone_id"]: float(z["effective_walkable_area_m2"]) for z in zones}
    gates = {z["zone_id"]: tuple(v * scale for v in (z["geometry"]["x"], z["geometry"]["y"],
                                                      z["geometry"]["x"] + z["geometry"]["w"],
                                                      z["geometry"]["y"] + z["geometry"]["d"]))
             for z in zones if z["zone_type"] == "exit"}
    frames = result["frames"]
    waiting = result["waitingFrames"]
    barriers = result.get("barriers", [])
    duration = len(frames) - 1

    placed = [{a["id"]: _place(assigner, a["x"], a["y"]) for a in frame} for frame in frames]
    speeds = [{a["id"]: a["v"] for a in frame} for frame in frames]

    # Per-second transitions t-1 -> t (t >= 1): entries/exits per zone.
    entries = [defaultdict(int) for _ in frames]
    exits = [defaultdict(int) for _ in frames]
    for t in range(1, len(frames)):
        prev, cur = placed[t - 1], placed[t]
        for aid, z in cur.items():
            if aid not in prev:
                entries[t][z] += 1                       # admitted during the step ending at t
            elif prev[aid] != z:
                exits[t][prev[aid]] += 1
                entries[t][z] += 1
        for aid, z in prev.items():
            if aid not in cur:
                exits[t][z] += 1                         # completed its route
    if placed:
        for aid, z in placed[0].items():
            entries[0][z] += 1                           # admitted at instant 0

    states = []
    for T in range(0, duration + 1, sample_seconds):
        cur = placed[T]
        prev = placed[T - 1] if T > 0 else {}
        window = range(max(1, T - sample_seconds + 1), T + 1) if T > 0 else range(0)
        zstate = {}
        for z in zones:
            zid = z["zone_id"]
            present = [aid for aid, zz in cur.items() if zz == zid]
            occ = len(present)
            queued = sum(1 for aid in present if aid in prev and speeds[T][aid] < QUEUE_SPEED_M_S)
            per_agent = []
            for aid in present:
                vs = [speeds[t][aid] for t in window if aid in speeds[t] and aid in speeds[t - 1]]
                if vs:
                    per_agent.append(sum(vs) / len(vs))
            if T > 0:
                in_count = sum(entries[t][zid] for t in window)
                out_count = sum(exits[t][zid] for t in window)
            else:
                in_count = out_count = None
            is_gate = zid in gates
            zstate[zid] = {
                "occupancy": occ,
                "density": occ / area[zid],
                "queue_length": queued,
                "speed": (sum(per_agent) / len(per_agent)) if per_agent else None,
                "inflow": in_count / sample_seconds if in_count is not None else None,
                "outflow": out_count / sample_seconds if out_count is not None else None,
                "inflow_count": in_count,
                "outflow_count": out_count,
                "exit_applicable": is_gate,
                "exit_open": (0.0 if gate_blocked(barriers, gates[zid], scale, T) else 1.0) if is_gate else None,
            }
        counts = defaultdict(int)
        for zz in cur.values():
            counts[zz] += 1
        states.append({
            "time_s": T,
            "schema_version": SCHEMA_VERSION,
            "zones": zstate,
            "accounting": {
                "active": len(cur),
                OUTSIDE_KEY: counts[OUTSIDE_KEY],
                OBSTACLE_KEY: counts[OBSTACLE_KEY],
                "waiting_entry": sum(g["count"] for g in waiting[T]),
            },
        })
    check_invariants(states, sample_seconds)
    return states


def check_invariants(states: list[dict], sample_seconds: int) -> None:
    for s in states:
        acc = s["accounting"]
        in_zones = sum(z["occupancy"] for z in s["zones"].values())
        if in_zones + acc[OUTSIDE_KEY] + acc[OBSTACLE_KEY] != acc["active"]:
            raise ZoneStateError(f"agent accounting broken at t={s['time_s']}")
        for z in s["zones"].values():
            if not (z["density"] >= 0 and math.isfinite(z["density"])):
                raise ZoneStateError("invalid density")
    for a, b in zip(states, states[1:]):
        if b["time_s"] - a["time_s"] != sample_seconds:
            continue
        for zid, zb in b["zones"].items():
            za = a["zones"][zid]
            if zb["occupancy"] - za["occupancy"] != zb["inflow_count"] - zb["outflow_count"]:
                raise ZoneStateError(f"flow conservation broken for {zid} at t={b['time_s']}")


def label(zone_state: dict, variable: str):
    """(value, valid) for one of the six target variables."""
    if variable == "exit_open":
        if not zone_state["exit_applicable"]:
            return None, False
        return zone_state["exit_open"], True
    v = zone_state[variable]
    return v, v is not None
