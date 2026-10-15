"""Read-only view model over evaluator outputs for the dashboard (Phase 7).

The dashboard never talks to the reference API and never receives reference credentials. It only
reads files the offline evaluator wrote (services.twin_studio.evaluation):
  <root>/<run_id>/<stamp>/report.json      observations seen, metrics, labels RELEASED by the
                                           reference API's server-side watermark
  <root>/<run_id>/<stamp>/forecasts.jsonl  forecasts persisted before their labels were eligible
Labels are shown only after the user explicitly reveals them; they are already restricted to what
the reference service released.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_ROOT = Path("data/twin_evaluations")
REPORT_SCHEMA = "twin_evaluation_report_v1"
UNAVAILABLE_NOTES = (
    "Decision-tree channels: unavailable for eh_twin_studio (mask 0); the production tree bank was trained on another simulator.",
    "Venue capacity: unknown (null) for every twin_venue_v1 zone; no capacity values are shown or invented.",
    "Occupancy, gate_exits and route_completed_not_exit are reference-only and never forecast.",
)


def list_evaluations(root: str | Path = DEFAULT_ROOT) -> list[dict]:
    root = Path(root)
    out = []
    if not root.is_dir():
        return out
    for report in sorted(root.glob("ehts_*/*/report.json")):
        try:
            doc = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("report_schema") != REPORT_SCHEMA:
            continue
        out.append({"run_id": doc["run_id"], "scenario_id": doc["scenario_id"], "evaluation": report.parent.name,
                    "path": str(report.parent)})
    return out


def load_evaluation(path: str | Path) -> dict:
    path = Path(path)
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    forecasts = [json.loads(l) for l in (path / report["forecast_log"]).read_text(encoding="utf-8").splitlines() if l]
    return {"report": report, "forecasts": forecasts}


def observation_rows(report: dict) -> list[dict]:
    cols = ("zone_id", "variable", "value", "unit", "source_id", "observed_at", "published_at", "received_at", "report_id")
    return [{c: r[c] for c in cols} for r in report.get("observations", [])]


def forecast_rows(forecasts: list[dict], *, stream: str, horizon: int) -> list[dict]:
    cols = ("zone_id", "variable", "issue_second", "target_second", "horizon_seconds", "value", "unit")
    return [{c: f[c] for c in cols} for f in forecasts if f["stream_id"] == stream and f["horizon_seconds"] == horizon]


def revealed_label_rows(report: dict, *, stream: str, horizon: int) -> list[dict]:
    rows = []
    for item in report["streams"][stream]["labels"]:
        p, lab = item["prediction"], item["label"]
        if p["horizon_seconds"] != horizon:
            continue
        rows.append({"zone_id": p["zone_id"], "variable": p["variable"], "issue_second": p["issue_second"],
                     "target_at": lab["target_at"], "label_available_at": lab["label_available_at"],
                     "forecast": p["value"], "label": lab["value"] if lab["value_valid"] else None,
                     "label_status": ("not applicable" if not lab["applicable"] else
                                      "valid" if lab["value_valid"] else "missing (undefined at target)"),
                     "absolute_error": item["absolute_error"]})
    return rows


def metric_rows(report: dict) -> list[dict]:
    rows = []
    for stream, res in report["streams"].items():
        for m in res["metrics"]:
            rows.append({"stream": stream, **m})
    return rows


def model_warnings(report: dict) -> list[str]:
    warnings = [f"Skill claim: {report['skill_claim']}"]
    for stream, info in report["models"].items():
        status = info.get("training_status")
        if status:
            warnings.append(f"{stream}: " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())))
        if info.get("environment_status"):
            warnings.append(f"{stream}: {info['environment_status']}")
    return warnings
