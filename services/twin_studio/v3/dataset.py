"""Offline Monte Carlo dataset construction for Twin Studio v3 (authorized offline use only).

  1. render every sampled scenario in the archive's run layout (runs.render_run, mode "custom"),
     plus scenario_params.json (hidden truth: regime, parameters, seed, split);
  2. replay each run through the same trusted controller the model uses, and at every 5 s instant
     T record exactly what inference would see (topology features and per-zone observation
     summaries), joined with the hidden zone-state labels at T (OfflineAccess required).

Inputs to features never include labels; labels are only joined afterwards, per sample.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from services.twin_studio.v3.artifacts import OfflineAccess, read_hidden_truth
from services.twin_studio.v3.features import VARIABLES, build_features, feature_names
from services.twin_studio.v3.operator_side import session_from_run_dir
from services.twin_studio.v3.runs import render_run
from services.twin_studio.v3.scenarios import Scenario
from services.twin_studio.v3.zone_states import label

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO / "data" / "environments" / "eh_twin_studio" / "v3"
SAMPLE_SECONDS = 5


def render_scenarios(scenarios: list[Scenario], out_root: Path) -> list[Path]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = []
    for sc in scenarios:
        d = out_root / sc.scenario_id
        if d.exists():
            raise FileExistsError(f"{d} exists; datasets are never overwritten")
        render_run(sc.scene, out_root, run_id=sc.scenario_id, modes=("custom",), zone_sources=sc.zone_sources)
        (d / "scenario_params.json").write_text(json.dumps(sc.record(), indent=2), encoding="utf-8")
        dirs.append(d)
    index = [sc.record() | {"run_dir": sc.scenario_id} for sc in scenarios]
    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return dirs


def observation_summary(snapshot) -> dict[str, np.ndarray]:
    """Per zone and variable: latest value, age (s), present, fresh; NaN/0 where absent."""
    zones = list(snapshot.topology.zone_ids)
    zi = {z: i for i, z in enumerate(zones)}
    sources = {s.source_id: s for s in snapshot.sources}
    out = {k: np.full((len(zones), len(VARIABLES)), math.nan if k in ("values", "ages") else 0.0)
           for k in ("values", "ages", "present", "fresh")}
    best = {}
    for m in snapshot.measurements:
        if m.entity_type == "zone" and m.variable in VARIABLES:
            k = (m.entity_id, m.variable)
            if k not in best or (m.observed_at_s, m.available_at_s, m.report_id) > (best[k].observed_at_s, best[k].available_at_s, best[k].report_id):
                best[k] = m
    for (z, v), m in best.items():
        i, j = zi[z], VARIABLES.index(v)
        age = snapshot.as_of_s - m.observed_at_s
        src = sources[m.source_id]
        out["values"][i, j] = float(m.value)
        out["ages"][i, j] = age
        out["present"][i, j] = 1.0
        out["fresh"][i, j] = 1.0 if age <= src.delay_s + 2 * src.interval_s else 0.0
    return out


def build_samples(run_dir: Path, access: OfflineAccess) -> dict:
    """Replay one run; return arrays over instants T = 0, 5, ..., duration."""
    (ctrl, key), meta, _ = session_from_run_dir(run_dir, "custom", step_s=SAMPLE_SECONDS)
    truth = read_hidden_truth(run_dir, "custom", access)
    states = {s["time_s"]: s for s in truth["zone_states"]}
    view = ctrl.model_view()
    times, X, obs, Y, M = [], [], {k: [] for k in ("values", "ages", "present", "fresh")}, [], []
    zones = None
    while True:
        T = int(ctrl.watermark_s)
        if T > meta["duration_s"]:
            break
        snap = view.snapshot()
        zone_ids, feats = build_features(snap)
        zones = zones or zone_ids
        summary = observation_summary(snap)
        y = np.full((len(zone_ids), len(VARIABLES)), math.nan)
        mask = np.zeros_like(y)
        for i, z in enumerate(zone_ids):
            for j, v in enumerate(VARIABLES):
                val, valid = label(states[T]["zones"][z], v)
                if valid:
                    y[i, j], mask[i, j] = float(val), 1.0
        times.append(T); X.append(feats); Y.append(y); M.append(mask)
        for k in obs:
            obs[k].append(summary[k])
        if ctrl.finished or T + SAMPLE_SECONDS > meta["duration_s"]:
            break
        ctrl.advance(key)
    return {"zone_ids": np.array(zones), "times": np.array(times), "X": np.stack(X), "Y": np.stack(Y), "M": np.stack(M),
            **{f"obs_{k}": np.stack(v) for k, v in obs.items()}}


def build_sample_store(runs_root: Path, samples_root: Path, access: OfflineAccess) -> dict:
    runs_root, samples_root = Path(runs_root), Path(samples_root)
    samples_root.mkdir(parents=True, exist_ok=True)
    index = json.loads((runs_root / "index.json").read_text(encoding="utf-8"))
    manifest = {"feature_schema": feature_names(), "variables": list(VARIABLES), "sample_seconds": SAMPLE_SECONDS,
                "scenarios": []}
    for rec in index:
        target = samples_root / f"{rec['scenario_id']}.npz"
        if target.exists():
            raise FileExistsError(f"{target} exists; datasets are never overwritten")
        arrays = build_samples(runs_root / rec["run_dir"], access)
        np.savez_compressed(target, **arrays)
        manifest["scenarios"].append({"scenario_id": rec["scenario_id"], "regime": rec["regime"], "split": rec["split"],
                                      "instants": int(len(arrays["times"])),
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (samples_root / "samples_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_samples(samples_root: Path) -> tuple[dict, dict]:
    samples_root = Path(samples_root)
    manifest = json.loads((samples_root / "samples_manifest.json").read_text(encoding="utf-8"))
    data = {}
    for s in manifest["scenarios"]:
        with np.load(samples_root / f"{s['scenario_id']}.npz") as z:
            data[s["scenario_id"]] = {k: z[k] for k in z.files} | {"regime": s["regime"], "split": s["split"]}
    return manifest, data
