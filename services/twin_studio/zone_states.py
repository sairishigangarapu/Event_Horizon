"""Zone-level reference states derived from the engine's per-agent frames (Phase 2B).

Inputs: one engine result (engine/simulation.py `simulate`) and the frozen twin_venue_v1 model.
Positions are assigned to zones ONLY through zones.ZoneAssigner. Route-level rows (`net_flow`,
`mean_density_p_m2`, ...) are never used to build zone states.

Time (timebase.py): engine step k covers [k, k+1) s; frames[k] is the state at instant k+1.
A state at instant T (sample interval dt) carries:
  instantaneous fields at T   occupancy, density, queue_length, probability_exit_open, accounting
  interval fields over [T-dt, T)   speed, inflow, outflow, gate_exits, route_completed_not_exit,
                                   gate_open_fraction
Instant 0 is the empty venue; its interval fields are null.

Variable rules:
  occupancy       agents assigned to the zone at T (each agent once)
  density         occupancy / effective_walkable_area_m2 (gate overlaps and the pond excluded)
  speed           mean, over agents present in the zone at T, of each agent's mean engine speed
                  over the steps of [T-dt, T) in which it was active; null + speed_valid=false
                  when the zone is empty; a fully stopped occupied zone has valid speed 0
  queue_length    agents present at T whose last-step speed is below QUEUE_SPEED_THRESHOLD_M_S
                  (the engine's own "queued" rule)
  inflow/outflow  persons/s over [T-dt, T): actual zone changes between consecutive frames,
                  plus spawns (into the spawn-position zone) and completions (out of the
                  completion-position zone); a completion is a gate exit only inside a gate
  probability_exit_open   gate zones only: 1.0 if the gate is open at instant T, else 0.0;
                  null and exit_applicable=false for every other zone
Agents in no zone or inside the pond are tracked in pseudo-zones so accounting is exact.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION
from services.twin_studio.timebase import canonical_instants, interval_steps, last_instant
from services.twin_studio.zones import OBSTACLE_KEY, OUTSIDE_KEY, STATUS_OBSTACLE, STATUS_ZONE, ZoneAssigner

QUEUE_SPEED_THRESHOLD_M_S = 0.1
UNITS = {"occupancy": "persons", "density": "persons/m2", "speed": "m/s", "inflow": "persons/s",
         "outflow": "persons/s", "queue_length": "persons", "probability_exit_open": "probability",
         "gate_exits": "persons", "route_completed_not_exit": "persons"}
UNSUPPORTED = {
    "trailing_partial_interval": "The canonical grid ends at the last multiple of the sampling interval; engine "
                                 "steps after it (e.g. [90, 91) s for a 90 s scene) appear only in the 1 s derivation.",
    "intermediate_zone_visits": "Zone changes are counted between consecutive 1 s frames; a zone crossed "
                                "entirely within one engine step is not counted (possible only for zones "
                                "shorter than one step's movement, i.e. open gates crossed mid-route).",
}


def _place(assigner: ZoneAssigner, x: float, y: float) -> str:
    a = assigner.assign(x, y)
    if a.status == STATUS_ZONE:
        return a.zone_id
    return OBSTACLE_KEY if a.status == STATUS_OBSTACLE else OUTSIDE_KEY


def gate_open_at_instant(events: list[dict], gate_id: str, instant_s: int) -> bool:
    """A gate_closure with [start, end) closes the gate at every instant start <= T < end."""
    return not any(e["type"] == "gate_closure" and e["gateId"] == gate_id and e["start"] <= instant_s < e["end"]
                   for e in events)


def step_transitions(result: dict, assigner: ZoneAssigner) -> list[dict]:
    """Per engine step k: zone of every agent at instant k (or its spawn zone) and at instant k+1,
    spawns, completions and the resulting per-zone in/out counts (pseudo-zones included)."""
    frames = result["frames"]
    spawns_by_t, done_by_t = defaultdict(list), defaultdict(list)
    for s in result["spawns"]:
        spawns_by_t[s["t"]].append(s)
    for c in result["completions"]:
        done_by_t[c["t"]].append(c)
    steps, prev = [], {}
    for k, frame in enumerate(frames):
        now = {a["id"]: _place(assigner, a["x"], a["y"]) for a in frame}
        before = dict(prev)
        inflow, outflow = defaultdict(int), defaultdict(int)
        spawn_zone = {}
        for s in spawns_by_t.get(k, []):
            z = _place(assigner, s["x"], s["y"])
            spawn_zone[s["id"]] = z
            before[s["id"]] = z
            inflow[z] += 1                      # entered the venue model at its spawn position
        completions = []
        for c in done_by_t.get(k, []):
            zc = _place(assigner, c["x"], c["y"])
            zb = before[c["id"]]
            if zb != zc:                        # moved into the completion zone during this step
                outflow[zb] += 1
                inflow[zc] += 1
            outflow[zc] += 1                    # removed at completion
            completions.append({**c, "zone": zc, "is_gate_exit": c["gateId"] is not None})
        for aid, zn in now.items():
            zb = before[aid]
            if zb != zn:
                outflow[zb] += 1
                inflow[zn] += 1
        steps.append({"step": k, "before": before, "now": now, "inflow": dict(inflow), "outflow": dict(outflow),
                      "spawn_zone": spawn_zone, "completions": completions})
        prev = now
    return steps


def derive_zone_states(result: dict, venue: dict | None = None, sample_seconds: int = 5,
                       assigner: ZoneAssigner | None = None) -> list[dict]:
    """Fully informed zone states at instants 0, dt, 2dt, ... (dt = sample_seconds)."""
    assigner = assigner or ZoneAssigner(venue)
    venue = assigner.venue
    zones = venue["zones"]
    area = {z["zone_id"]: z["effective_walkable_area_m2"] for z in zones}
    gate_ids = {z["zone_id"] for z in zones if z["zone_type"] == "exit"}
    frames = result["frames"]
    duration = len(frames) - 1
    steps = step_transitions(result, assigner)
    speeds = [{a["id"]: a["v"] for a in frame} for frame in frames]

    states, cum = [], {"spawned": 0, "gate_exits": 0, "route_completed_not_exit": 0}
    cum_by_k = []
    for st in steps:
        cum["spawned"] += len(st["spawn_zone"])
        cum["gate_exits"] += sum(c["is_gate_exit"] for c in st["completions"])
        cum["route_completed_not_exit"] += sum(not c["is_gate_exit"] for c in st["completions"])
        cum_by_k.append(dict(cum))

    for T in canonical_instants(duration, sample_seconds):
        window = list(interval_steps(T, sample_seconds))
        has_interval = bool(window)
        current = steps[T - 1]["now"] if T > 0 else {}
        counts = defaultdict(int)
        for zone in current.values():
            counts[zone] += 1
        zstate = {}
        for z in zones:
            zid = z["zone_id"]
            present = [aid for aid, zone in current.items() if zone == zid]
            occ = len(present)
            if occ and has_interval:
                per_agent = []
                for aid in present:
                    vs = [speeds[k][aid] for k in window if aid in speeds[k]]
                    per_agent.append(sum(vs) / len(vs))
                speed = sum(per_agent) / len(per_agent)
            else:
                speed = None
            last_v = speeds[T - 1] if T > 0 else {}
            inflow = sum(steps[k]["inflow"].get(zid, 0) for k in window)
            outflow = sum(steps[k]["outflow"].get(zid, 0) for k in window)
            exits_here = [c for k in window for c in steps[k]["completions"] if c["zone"] == zid]
            is_gate = zid in gate_ids
            zstate[zid] = {
                "occupancy": occ,
                "density": occ / area[zid],
                "speed": speed,
                "speed_valid": speed is not None,
                "queue_length": sum(1 for aid in present if last_v[aid] < QUEUE_SPEED_THRESHOLD_M_S),
                "inflow": inflow / sample_seconds if has_interval else None,
                "outflow": outflow / sample_seconds if has_interval else None,
                "inflow_count": inflow if has_interval else None,
                "outflow_count": outflow if has_interval else None,
                "gate_exits": sum(c["is_gate_exit"] for c in exits_here) if has_interval else None,
                "route_completed_not_exit": sum(not c["is_gate_exit"] for c in exits_here) if has_interval else None,
                "exit_applicable": is_gate,
                "probability_exit_open": (1.0 if gate_open_at_instant(result["events"], zid, T) else 0.0) if is_gate else None,
                "gate_open_fraction": (sum(gate_open_at_instant(result["events"], zid, k) for k in window) / len(window)
                                       if is_gate and has_interval else None),
            }
        c = cum_by_k[T - 1] if T > 0 else {"spawned": 0, "gate_exits": 0, "route_completed_not_exit": 0}
        states.append({
            "environment_id": ENVIRONMENT_ID, "map_version": MAP_VERSION,
            "run_mode": result["mode"], "seed": result["rows"][0]["seed"],
            "simulation_second": T,
            "interval_start_second": T - sample_seconds if has_interval else T,
            "interval_end_second": T,
            "zones": zstate,
            "accounting": {
                "in_zones": sum(counts[z["zone_id"]] for z in zones),
                OBSTACLE_KEY: counts[OBSTACLE_KEY],
                OUTSIDE_KEY: counts[OUTSIDE_KEY],
                "spawned_cumulative": c["spawned"],
                "gate_exits_cumulative": c["gate_exits"],
                "route_completed_not_exit_cumulative": c["route_completed_not_exit"],
            },
        })
    assert states[-1]["simulation_second"] <= last_instant(duration)
    return states
