"""Twin Studio evaluation runner (Phase 6): both streams, persisted forecasts, delayed labels.

Roles and data paths (no shortcut around the Phase 4 APIs):
  observation API  <- operator control token: replay created and advanced in lockstep with issue time;
                      reports read only up to the replay watermark.
  Stream B         <- pipeline_adapter.TwinArchitectureRunner (architecture pipeline, untrained).
  Stream A         <- pipeline_adapter.observation_persistence (observation-only baseline). The repo's
                      CentralSourceAdapter.generate_simulated_forecasts derives forecasts from
                      reference truth and is therefore NOT used for Twin Studio.
  reference API    <- evaluator token: one session per stream; predictions registered at the
                      session's current issue time; labels released only by the server watermark.

Per issue step t (every sample_seconds):
  1. read reports with received_at <= t (observation replay at t);
  2. compute both streams' forecasts;
  3. append them to forecasts.jsonl and fsync  (persisted BEFORE any related label can be eligible:
     the earliest availability is t + 30 + 60 s);
  4. register them with the reference API (server re-checks issue time, horizon and timeline);
  5. advance observation replay and both sessions by one step.
After the last issue step the sessions are advanced to their final watermark and the released
labels are fetched. Metrics are computed only from labels the reference API released.

This runner is the "offline evaluator": it holds the reference token and must run on the reference
network or through the opt-in loopback route. The dashboard and the main application never do.

Usage (evaluator host / container):
  python -m services.twin_studio.evaluation --run-id ehts_... --out data/twin_evaluations
     with EH_OBS_API_URL, EH_REF_API_URL, EH_OBS_CONTROL_TOKEN_FILE, EH_REF_TOKEN_FILE
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from services.twin_studio.pipeline_adapter import (STREAM_B_ID, ForecastRow, TwinArchitectureRunner,
                                                   environment_manifest, observation_persistence)

STREAM_A_ID = "stream_a_observation_persistence"
REPORT_SCHEMA = "twin_evaluation_report_v1"


class EvaluationError(RuntimeError):
    pass


def _ok(resp, expected=(200, 201)):
    if resp.status_code not in expected:
        raise EvaluationError(f"{resp.request.method} {resp.request.url.path} -> {resp.status_code}")
    return resp.json()


class ForecastLog:
    """Append-only, fsynced JSONL of every issued forecast (written before registration)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise EvaluationError(f"{path} exists; evaluation outputs are never overwritten")
        self._f = open(path, "x", encoding="utf-8")

    def append(self, stream: str, rows: list[ForecastRow], written_at_watermark: int) -> None:
        for r in rows:
            self._f.write(json.dumps({"stream_id": stream, **r.__dict__,
                                      "persisted_at_watermark_second": written_at_watermark}, sort_keys=True) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()


def _metrics(released: list[dict]) -> list[dict]:
    """Per variable x horizon. Inapplicable labels are excluded; invalid (missing) labels are counted."""
    groups: dict = defaultdict(lambda: {"pairs": [], "missing_label": 0})
    for item in released:
        lab, pred = item["label"], item["prediction"]
        if not lab["applicable"]:
            continue
        g = groups[(pred["variable"], pred["horizon_seconds"])]
        if not lab["value_valid"]:
            g["missing_label"] += 1
        else:
            g["pairs"].append((pred["value"], lab["value"]))
    out = []
    for (variable, h), g in sorted(groups.items()):
        n = len(g["pairs"])
        row = {"variable": variable, "horizon_seconds": h, "n": n, "missing_label": g["missing_label"]}
        if n:
            errs = [p - t for p, t in g["pairs"]]
            row["mae"] = sum(abs(e) for e in errs) / n
            row["rmse"] = math.sqrt(sum(e * e for e in errs) / n)
            if variable == "probability_exit_open":
                row["brier"] = sum(e * e for e in errs) / n
                row["accuracy"] = sum((p >= 0.5) == (t >= 0.5) for p, t in g["pairs"]) / n
        out.append(row)
    return out


def run_evaluation(obs, ref, *, run_id: str, control_token: str, evaluator_token: str, out_dir: str | Path,
                   seed: int = 0) -> dict:
    """obs/ref: httpx.Client-compatible clients for the observation and reference APIs."""
    ctl = {"Authorization": f"Bearer {control_token}"}
    ev = {"Authorization": f"Bearer {evaluator_token}"}
    runs = {r["run_id"]: r for r in _ok(obs.get("/runs"))["runs"]}
    if run_id not in runs:
        raise EvaluationError("run not available on the observation API")
    meta = runs[run_id]
    venue = _ok(obs.get(f"/runs/{run_id}/venue"))
    gate_ids = {z["zone_id"] for z in venue["zones"] if z["zone_type"] == "exit"}
    replay = _ok(obs.post("/replays", json={"run_id": run_id}, headers=ctl))
    sessions = {s: _ok(ref.post("/sessions", json={"run_id": run_id}, headers=ev)) for s in (STREAM_A_ID, STREAM_B_ID)}
    sess = sessions[STREAM_B_ID]
    dt, horizons, last = sess["sample_seconds"], sess["horizons_seconds"], sess["last_state_second"]
    targets = set(sess["target_variables"])
    runner = TwinArchitectureRunner(venue, run_id, meta["epoch_utc"], seed=seed, horizons_seconds=horizons)

    out_dir = Path(out_dir) / run_id / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = ForecastLog(out_dir / "forecasts.jsonl")
    observation_view: list[dict] = []
    issued = {STREAM_A_ID: 0, STREAM_B_ID: 0}
    reports: list[dict] = []
    t = 0
    try:
        while True:
            if replay["watermark_second"] != t or any(s["current_second"] != t for s in sessions.values()):
                raise EvaluationError("observation replay and reference sessions are out of lockstep")
            reports = _ok(obs.get(f"/runs/{run_id}/reports", params={"replay_id": replay["replay_id"]}))["reports"]
            observation_view.append({"issue_second": t, "reports_visible": len(reports)})
            streams = {
                STREAM_B_ID: runner.step(t, reports),
                STREAM_A_ID: observation_persistence(t, reports, meta["epoch_utc"], horizons_seconds=horizons,
                                                     gate_ids=gate_ids),
            }
            for stream, rows in streams.items():
                rows = [r for r in rows if r.target_second <= last and r.variable in targets]
                if not rows:
                    continue
                log.append(stream, rows, t)                       # persisted before registration
                _ok(ref.post(f"/sessions/{sessions[stream]['session_id']}/predictions", headers=ev,
                             json={"issue_second": t, "predictions": [
                                 {"zone_id": r.zone_id, "variable": r.variable, "horizon_seconds": r.horizon_seconds,
                                  "value": r.value} for r in rows]}))
                issued[stream] += len(rows)
            if t + min(horizons) > last:
                break
            for stream in sessions:
                sessions[stream] = _ok(ref.post(f"/sessions/{sessions[stream]['session_id']}/advance",
                                                json={"steps": 1}, headers=ev))
            adv = obs.post(f"/replays/{replay['replay_id']}/advance", json={"steps": 1}, headers=ctl)
            if adv.status_code == 409:     # replay already at its final watermark: hold it there
                raise EvaluationError("observation replay ended before the last issue time")
            replay = _ok(adv)
            t += dt
    finally:
        log.close()

    results = {}
    for stream, s in sessions.items():
        while s["current_second"] < s["final_second"]:
            s = _ok(ref.post(f"/sessions/{s['session_id']}/advance", json={"steps": 12}, headers=ev))
        labels = _ok(ref.get(f"/sessions/{s['session_id']}/labels", headers=ev))
        results[stream] = {"issued": issued[stream], "released": len(labels["released"]), "pending": labels["pending"],
                           "metrics": _metrics(labels["released"]), "labels": labels["released"]}

    manifest = environment_manifest()
    report = {
        "report_schema": REPORT_SCHEMA, "run_id": run_id, "scenario_id": meta["scenario_id"],
        "epoch_utc": meta["epoch_utc"], "sample_seconds": dt, "horizons_seconds": horizons,
        "last_issue_second": t, "observation_view": observation_view,
        # observation-side records the evaluator saw at its last issue time (allowlisted by the API)
        "observations": reports,
        "models": {
            STREAM_B_ID: {"pipeline_version": manifest["pipeline_version"], "training_status": manifest["training_status"],
                          "environment_status": "untrained; out-of-environment components not used",
                          "unknown_capacity_zones": runner.unknown_capacity_zones, "seed": seed},
            STREAM_A_ID: {"description": "latest received observation held constant (observation-only baseline)"},
        },
        "skill_claim": "none - untrained models; a replay demonstrates the data path, not forecast accuracy",
        "forecast_log": "forecasts.jsonl",
        "streams": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {**report, "out_dir": str(out_dir)}


def main(argv=None) -> int:
    import httpx
    ap = argparse.ArgumentParser(description="Evaluate both streams on a published Twin Studio run")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/twin_evaluations"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    read = lambda env: Path(os.environ[env]).read_text(encoding="utf-8").strip()
    with httpx.Client(base_url=os.environ["EH_OBS_API_URL"], timeout=60) as obs, \
            httpx.Client(base_url=os.environ["EH_REF_API_URL"], timeout=60) as ref:
        rep = run_evaluation(obs, ref, run_id=args.run_id, control_token=read("EH_OBS_CONTROL_TOKEN_FILE"),
                             evaluator_token=read("EH_REF_TOKEN_FILE"), out_dir=args.out, seed=args.seed)
    print(json.dumps({"out_dir": rep["out_dir"], **{s: {k: v for k, v in r.items() if k != "labels"}
                                                    for s, r in rep["streams"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
