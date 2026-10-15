"""Integrated Twin v3 partial-information pipeline (Phase J, eh-v3-pipeline-1).

  canonical v3 run -> separated artifacts -> trusted replay (operator-owned clock)
  -> strict ingestion -> topology features -> six-channel tree bank -> hypernetwork + GNN -> GRU
  -> persisted forecasts (Stream A, Stream B) + notices + staleness assessments
  -> triggered A* recommendations -> delayed reference evaluation (offline, after the replay)

Model side (inside the replay loop) touches only AuthorizedSnapshot objects. The operator side
creates the session, advances the clock and, after the replay, runs the reference evaluator,
which reads hidden truth through OfflineAccess and only for targets whose labels are eligible
(label_available_at = target + LABEL_DELAY_S <= final watermark).

Streams (independently labelled, never merged):
  stream_a_observation_ewma   observation-only exponentially weighted moving average of distinct
                              zone reports over the last 90 s, held across horizons. No truth.
                              (The Phase 6 Twin evaluator used latest-value persistence instead.)
  stream_b_architecture       trees + hypernetwork/GNN + GRU, with its training status attached.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from services.twin_studio.v3 import neural, trees
from services.twin_studio.v3.artifacts import OfflineAccess, read_hidden_truth
from services.twin_studio.v3.dataset import observation_summary
from services.twin_studio.v3.features import VARIABLES, build_features
from services.twin_studio.v3.operator_side import session_from_run_dir
from services.twin_studio.v3.routing import recommend, scene_alignment
from services.twin_studio.v3.staleness import StalenessPolicy, assess_snapshot
from services.twin_studio.zones import load_venue

VERSION = "eh-v3-pipeline-1"
LABEL_DELAY_S = 60
EWMA_ALPHA, EWMA_LOOKBACK_S = 0.35, 90.0
TRIGGER_NOTICES = {"possible_obstruction", "exit_reported_closed", "high_density"}
STREAM_A, STREAM_B = "stream_a_observation_ewma", "stream_b_architecture"


class _Log:
    """Append-only JSONL written and fsynced as records are produced."""

    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"{path} exists; evaluation outputs are never overwritten")
        self.f = open(path, "w", encoding="utf-8")

    def write(self, records):
        for r in records:
            self.f.write(json.dumps(r, allow_nan=False) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def stream_a(snapshot, horizons) -> list[dict]:
    """Observation-only EWMA per zone/variable over distinct reports observed in the last 90 s."""
    T = snapshot.as_of_s
    per = {}
    for m in snapshot.measurements:
        if m.entity_type == "zone" and m.variable in VARIABLES and m.observed_at_s >= T - EWMA_LOOKBACK_S:
            per.setdefault((m.entity_id, m.variable), {})[(m.observed_at_s, m.report_id)] = float(m.value)
    rows = []
    for (z, v), reports in sorted(per.items()):
        seq = [reports[k] for k in sorted(reports)]
        level = seq[0]
        for x in seq[1:]:
            level = EWMA_ALPHA * x + (1 - EWMA_ALPHA) * level
        for h in horizons:
            rows.append({"stream_id": STREAM_A, "zone_id": z, "variable": v, "issue_s": T, "horizon_s": h,
                         "target_s": T + h, "value": round(level, 5)})
    return rows


class StreamB:
    """Trees + hypernetwork/GNN + GRU over the replay history (model side)."""

    def __init__(self, bank: trees.TreeBankV3, model: neural.Model | None):
        self.bank, self.model = bank, model
        self.hist_feats, self.hist_valid, self.hist_u24, self.hist_H = [], [], [], []

    def step(self, snapshot) -> tuple[list[dict], dict]:
        zone_ids, X = build_features(snapshot)
        tvals, tmasks = self.bank.predict(X)
        estimates = {(z, VARIABLES[j]): float(tvals[i, j]) for i, z in enumerate(zone_ids)
                     for j in range(len(VARIABLES)) if tmasks[i, j] > 0}
        if self.model is None:
            return [], estimates
        summary = observation_summary(snapshot)
        d = {"obs_values": summary["values"][None], "obs_ages": summary["ages"][None], "obs_present": summary["present"][None],
             "obs_fresh": summary["fresh"][None], "Y": np.zeros((1,) + tvals.shape), "M": np.zeros((1,) + tvals.shape),
             "times": np.array([snapshot.as_of_s])}
        s = neural.scenario_tensors(d, tvals[None], tmasks[None])
        with torch.no_grad():
            H = self.model.spatial(torch.from_numpy(s["feats"]), torch.from_numpy(s["valid"]))[0]
        self.hist_u24.append(torch.from_numpy(s["u24"][0]))
        self.hist_H.append(H)
        window = list(range(max(0, len(self.hist_H) - neural.LOOKBACK), len(self.hist_H)))
        window = [window[0]] * (neural.LOOKBACK - len(window)) + window
        U = torch.stack([torch.cat([self.hist_u24[i], self.hist_H[i]], dim=1) for i in window])   # [L, Z, 40]
        with torch.no_grad():
            out = self.model.temporal(U.permute(1, 0, 2)).numpy()                                  # [Z, H, 6]
        rows = []
        for i, z in enumerate(zone_ids):
            for k, h in enumerate(neural.HORIZONS_S):
                for j, v in enumerate(VARIABLES):
                    if v == "exit_open" and tmasks[i, j] == 0:
                        continue
                    rows.append({"stream_id": STREAM_B, "zone_id": z, "variable": v, "issue_s": snapshot.as_of_s,
                                 "horizon_s": h, "target_s": snapshot.as_of_s + h, "value": round(float(out[i, k, j]), 5)})
        return rows, estimates


def run(run_dir: Path, out_root: Path, *, tree_dir: Path, neural_dir: Path | None, policy: StalenessPolicy,
        mode: str = "custom", evaluate: bool = True) -> Path:
    run_dir, out_root = Path(run_dir), Path(out_root)
    out = out_root / f"{run_dir.name}__{mode}"
    if out.exists():
        raise FileExistsError(f"{out} exists; evaluation outputs are never overwritten")
    out.mkdir(parents=True)

    # ---------------- operator side: session, alignment check, model loading
    (ctrl, key), meta, _ = session_from_run_dir(run_dir, mode)
    scene = json.loads((run_dir / "hidden_ground_truth" / "scenario.json").read_text(encoding="utf-8"))
    alignment = scene_alignment(scene, load_venue())
    bank = trees.load(tree_dir)
    tree_report = json.loads((Path(tree_dir) / "tree_report.json").read_text(encoding="utf-8"))
    validation_file = Path(tree_dir) / "tree_validation.json"
    validated_channels = set(json.loads(validation_file.read_text(encoding="utf-8"))["validated_channels"]
                             if validation_file.is_file() else [])
    model, neural_report = (neural.load(neural_dir) if neural_dir else (None, {"training_status": "not_loaded"}))
    forecast_validated = neural_report.get("training_status") == "trained_validated"
    horizons = list(neural.HORIZONS_S)
    del scene                                   # nothing below the model boundary may use the scene

    logs = {n: _Log(out / f"{n}.jsonl") for n in ("forecasts", "assessments", "notices", "recommendations")}
    b = StreamB(bank, model)
    view = ctrl.model_view()
    seen_notices, last_issue = set(), None
    try:
        while True:
            T = ctrl.watermark_s
            if T > meta["duration_s"]:
                break
            # ---------------- model side: only the authorized snapshot
            snap = view.snapshot()
            fa = stream_a(snap, horizons)
            fb, estimates = b.step(snap)
            logs["forecasts"].write([r | {"persisted_at_watermark_s": T} for r in fa + fb])
            fb30 = {("zone", r["zone_id"], r["variable"]): (r["value"], neural_report.get("version", "none"))
                    for r in fb if r["horizon_s"] == horizons[0]}
            # Tree estimates are not used as staleness evidence: the calibrated policy (eh-v3-staleness-1)
            # was evaluated with age, observed conflicts and newer reports only.
            assessments = assess_snapshot(snap, policy, forecasts=fb30, forecasts_validated=forecast_validated)
            logs["assessments"].write([a.to_dict() for a in assessments if a.alert])
            new_notices = [n for n in snap.notices if n.notice_id not in seen_notices]
            seen_notices.update(n.notice_id for n in new_notices)
            logs["notices"].write([n.to_dict() for n in new_notices])
            gate_alerts = [a for a in assessments if a.variable == "exit_open" and a.alert]
            triggers = sorted({n.notice_type for n in new_notices} & TRIGGER_NOTICES) + (["gate_reverification"] if gate_alerts else [])
            if triggers:
                occupied = sorted({a.entity_id for a in assess_snapshot(snap, policy) if a.entity_type == "zone"
                                   and a.variable == "density" and a.freshness_status == "FRESH" and a.value > 0})
                logs["recommendations"].write([recommend(snap, z, trigger=",".join(triggers), alignment=alignment, policy=policy)
                                               for z in occupied])
            last_issue = T
            if ctrl.finished or T + 5 > meta["duration_s"]:
                break
            ctrl.advance(key)                   # operator side
    finally:
        for log in logs.values():
            log.close()
    final_watermark = ctrl.watermark_s

    report = {"pipeline_version": VERSION, "run": run_dir.name, "mode": mode, "last_issue_s": last_issue,
              "final_watermark_s": final_watermark, "label_delay_s": LABEL_DELAY_S, "horizons_s": horizons,
              "streams": {STREAM_A: "observation-only EWMA of zone reports (no truth)",
                          STREAM_B: {"tree_bank": bank.version, "tree_validated_channels": sorted(validated_channels),
                                     "neural_status": neural_report.get("training_status")}},
              "staleness_policy": {"version": policy.version, "age_multiplier": policy.age_multiplier,
                                   "model_estimate_evidence": "disabled (not part of the evaluated policy)",
                                   "forecast_change_warning": "active only if neural status is trained_validated"},
              "routing_alignment": {"aligned": alignment.get("aligned")}}
    if evaluate:
        # After the replay the evaluator advances its own label clock until every issued forecast's
        # label is eligible; labels are still released only at target + LABEL_DELAY_S.
        label_watermark = (last_issue or 0) + max(horizons) + LABEL_DELAY_S
        report["evaluation"] = reference_evaluation(run_dir, mode, out, label_watermark, meta["duration_s"])
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def reference_evaluation(run_dir: Path, mode: str, out: Path, label_watermark: float, duration_s: int) -> dict:
    """Offline: compare persisted forecasts with hidden zone states whose labels are eligible."""
    truth = read_hidden_truth(run_dir, mode, OfflineAccess("evaluation"))
    states = {s["time_s"]: s for s in truth["zone_states"]}
    errs: dict = {}
    pending = beyond_run = 0
    with open(out / "forecasts.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["persisted_at_watermark_s"] >= r["target_s"] + LABEL_DELAY_S:
                raise AssertionError("a forecast was persisted after its label became eligible")
            if r["target_s"] > duration_s:
                beyond_run += 1                     # the simulated run ends before the target: no truth
                continue
            if r["target_s"] + LABEL_DELAY_S > label_watermark or r["target_s"] not in states:
                pending += 1
                continue
            z = states[r["target_s"]]["zones"][r["zone_id"]]
            tv = z["exit_open"] if r["variable"] == "exit_open" else z[r["variable"]]
            if tv is None:
                continue
            errs.setdefault((r["stream_id"], r["horizon_s"], r["variable"]), []).append(r["value"] - tv)
    metrics = {}
    for (stream, h, v), e in sorted(errs.items()):
        e = np.array(e)
        metrics.setdefault(stream, {}).setdefault(str(h), {})[v] = {
            "n": int(len(e)), "mae": round(float(np.mean(np.abs(e))), 4), "rmse": round(float(np.sqrt(np.mean(e ** 2))), 4)}
    return {"metrics": metrics, "labels_not_yet_eligible": pending, "targets_beyond_run_end": beyond_run,
            "label_watermark_s": label_watermark,
            "rule": f"label_available_at = target + {LABEL_DELAY_S}s; only eligible labels are scored"}
