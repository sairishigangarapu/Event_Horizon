"""Topology-aware tree inputs from an authorized snapshot (Phase F, schema eh-v3-topo-features-1).

Inputs are ONLY an AuthorizedSnapshot from the trusted replay controller: zone-level measurements
with observed/available <= T, plus static venue topology and source settings. No hidden reference
values, full-run files, incident schedules or same-timestep GNN outputs are accepted.

Per zone z (variables v in density, speed, inflow, outflow, queue_length, exit_open):
  own_value_v       latest valid zone-level measurement of v (NaN if none: absence is not zero)
  own_age_v         T - observed_at of that measurement (NaN if none)
  own_present_v     1 if a measurement exists, else 0
  own_fresh_v       1 if age <= delay + 2 * interval of its source, else 0
  nb_mean_v         mean of the neighbours' own_value_v over physically connected zones (NaN if none)
  nb_frac_v         fraction of neighbours with a measurement of v
  nb_min_age_v      youngest neighbour measurement age (NaN if none)
  nb_gap_v          |own_value_v - nb_mean_v| / (|nb_mean_v| + scale_v) (NaN if either is missing)
static: degree, total connected width (m), mean connection distance (m), effective walkable area (m2),
zone-type one-hot (entrance, corridor, assembly, stairway, exit), exit_applicable.
Missing values stay NaN (scikit-learn trees route them natively) and are always paired with masks.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from services.twin_studio.v3.replay import AuthorizedSnapshot

SCHEMA_VERSION = "eh-v3-topo-features-1"
VARIABLES = ("density", "speed", "inflow", "outflow", "queue_length", "exit_open")
ZONE_TYPES = ("entrance", "corridor", "assembly", "stairway", "exit")
SCALE = {"density": 0.5, "speed": 0.3, "inflow": 0.5, "outflow": 0.5, "queue_length": 2.0, "exit_open": 0.5}


class InferenceInputError(TypeError):
    """Raised when anything other than an authorized replay snapshot is offered as model input."""


def feature_names() -> list[str]:
    names = []
    for v in VARIABLES:
        names += [f"own_value_{v}", f"own_age_{v}", f"own_present_{v}", f"own_fresh_{v}"]
    for v in VARIABLES:
        names += [f"nb_mean_{v}", f"nb_frac_{v}", f"nb_min_age_{v}", f"nb_gap_{v}"]
    names += ["degree", "connected_width_m", "mean_connection_distance_m", "area_m2"]
    names += [f"type_{t}" for t in ZONE_TYPES] + ["exit_applicable"]
    return names


def require_snapshot(obj: Any) -> AuthorizedSnapshot:
    if isinstance(obj, AuthorizedSnapshot):
        return obj
    cls = getattr(obj, "artifact_class", None)
    if cls is not None:
        raise InferenceInputError(f"artifact of class {cls} (truth_bearing={getattr(obj, 'truth_bearing', None)}) "
                                  "is not an inference input; only an AuthorizedSnapshot from the replay controller is")
    raise InferenceInputError(f"{type(obj).__name__} is not an inference input; only an AuthorizedSnapshot "
                              "from the replay controller is accepted")


def _neighbours(snapshot: AuthorizedSnapshot) -> dict[str, list[tuple[str, float, float]]]:
    nb: dict[str, list[tuple[str, float, float]]] = {z: [] for z in snapshot.topology.zone_ids}
    for _, a, b, width, dist, traversable in snapshot.topology.connections:
        if not traversable:
            continue
        nb[a].append((b, width, dist))
        nb[b].append((a, width, dist))
    return nb


def build_features(snapshot: Any) -> tuple[list[str], np.ndarray]:
    """(zone_ids, X[Z, F]) for the zones of the snapshot's topology, in topology order."""
    snap = require_snapshot(snapshot)
    T = snap.as_of_s
    topo = snap.topology
    sources = {s.source_id: s for s in snap.sources}
    latest: dict[tuple[str, str], Any] = {}
    for m in snap.measurements:
        if m.entity_type != "zone" or m.variable not in VARIABLES:
            continue
        k = (m.entity_id, m.variable)
        cur = latest.get(k)
        if cur is None or (m.observed_at_s, m.available_at_s, m.report_id) > (cur.observed_at_s, cur.available_at_s, cur.report_id):
            latest[k] = m

    own = {}
    for z in topo.zone_ids:
        for v in VARIABLES:
            m = latest.get((z, v))
            if m is None:
                own[(z, v)] = (math.nan, math.nan, 0.0, 0.0)
            else:
                src = sources[m.source_id]
                age = T - m.observed_at_s
                own[(z, v)] = (float(m.value), age, 1.0, 1.0 if age <= src.delay_s + 2 * src.interval_s else 0.0)

    nbs = _neighbours(snap)
    rows = []
    for z in topo.zone_ids:
        row = []
        for v in VARIABLES:
            row += list(own[(z, v)])
        for v in VARIABLES:
            vals = [own[(n, v)][0] for n, _, _ in nbs[z] if own[(n, v)][2] == 1.0]
            ages = [own[(n, v)][1] for n, _, _ in nbs[z] if own[(n, v)][2] == 1.0]
            mean = float(np.mean(vals)) if vals else math.nan
            frac = len(vals) / len(nbs[z]) if nbs[z] else 0.0
            gap = abs(own[(z, v)][0] - mean) / (abs(mean) + SCALE[v]) if vals and own[(z, v)][2] == 1.0 else math.nan
            row += [mean, frac, min(ages) if ages else math.nan, gap]
        widths = [w for _, w, _ in nbs[z]]
        dists = [d for _, _, d in nbs[z]]
        ztype = topo.zone_types[z]
        row += [float(len(nbs[z])), float(sum(widths)), float(np.mean(dists)) if dists else 0.0, topo.zone_area_m2[z]]
        row += [1.0 if ztype == t else 0.0 for t in ZONE_TYPES] + [1.0 if z in topo.gate_ids else 0.0]
        rows.append(row)
    X = np.array(rows, dtype=np.float64)
    if X.shape[1] != len(feature_names()):
        raise AssertionError("feature schema drift")
    return list(topo.zone_ids), X
