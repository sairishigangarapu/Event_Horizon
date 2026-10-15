"""Read-only view over persisted Twin v3 pipeline outputs, for the main application API.

Reads only files written by pipeline.run under the evaluations root: forecasts, alerting
assessments, notices, recommendations and the report (whose metrics come from labels the
reference evaluator released after the replay). It never opens run outputs, hidden truth,
scenes, manifests or observation streams, and it never starts a replay.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from services.twin_studio.v3.dataset import DATA_ROOT

EVALUATIONS_ROOT = DATA_ROOT / "evaluations"
EVAL_ID_RE = re.compile(r"[A-Za-z0-9_]+__(ideal|custom|random)")
FILES = ("forecasts", "assessments", "notices", "recommendations")


class OutputsError(LookupError):
    pass


def _dir(eval_id: str, root: Path) -> Path:
    if not EVAL_ID_RE.fullmatch(eval_id):
        raise OutputsError("invalid evaluation id")
    d = Path(root) / eval_id
    if not (d / "report.json").is_file():
        raise OutputsError("unknown evaluation")
    return d


def list_evaluations(root: Path = EVALUATIONS_ROOT) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if EVAL_ID_RE.fullmatch(d.name) and (d / "report.json").is_file():
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
            out.append({"evaluation_id": d.name, "pipeline_version": rep["pipeline_version"], "streams": rep["streams"],
                        "final_watermark_s": rep["final_watermark_s"]})
    return out


def report(eval_id: str, root: Path = EVALUATIONS_ROOT) -> dict:
    return json.loads((_dir(eval_id, root) / "report.json").read_text(encoding="utf-8"))


def records(eval_id: str, kind: str, *, as_of_s: float | None = None, root: Path = EVALUATIONS_ROOT) -> list[dict]:
    """Persisted records of one kind; with as_of_s, only those produced at that replay instant
    (assessments, recommendations) or available by then (notices, forecasts issued by then)."""
    if kind not in FILES:
        raise OutputsError("unknown record kind")
    rows = []
    with open(_dir(eval_id, root) / f"{kind}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if as_of_s is not None:
                t = {"forecasts": r.get("issue_s"), "assessments": r.get("evaluated_at_s"),
                     "notices": r.get("available_at_s"), "recommendations": r.get("as_of_s")}[kind]
                if kind in ("assessments", "recommendations") and t != as_of_s:
                    continue
                if kind in ("forecasts", "notices") and t > as_of_s:
                    continue
            rows.append(r)
    return rows
