"""Artifact registry for Twin Studio v3 outputs (Phase B).

Every file produced by a v3 render (``outputs/<run_id>/`` in the archive's layout) or shipped as an
archive example is registered with its semantic role and permitted uses. Nothing is a model input
merely because it exists:

  class                 examples                                        permitted uses
  OBSERVATION_STREAM    observable/<mode>.jsonl,                         replay_source, training, evaluation
                        observable_ext/zone_sensing/<mode>.jsonl
  OBSERVATION_ARCHIVE   observable.zip (contains later reports)          training, evaluation
  HIDDEN_TRUTH          hidden_ground_truth/**, hidden_ground_truth.zip,  training, evaluation
                        hidden_ground_truth_ext/** (derived zone states), scenario_params.json
  ANALYSIS_CSV          01_ideal_flow.csv ... (exact simulator output)   training, evaluation, operator_reporting
  OPERATOR_REPORT       master_report.html, report_data.json,            operator_reporting, provenance
                        run_manifest.json, decisions_*.jsonl
  SCENE_CONFIG          examples/event_horizon_scene.json                simulation_input, operator_reporting
  SEPARATION_NOTE       DATA_SEPARATION.txt                              provenance
  UNCLASSIFIED          anything else                                    (none: fail closed)

No file is inference-safe. Inference receives only an AuthorizedSnapshot from the trusted replay
controller (replay.py), which reads OBSERVATION_STREAM files on the operator's side.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from services.twin_studio.v3 import MODES

CSV_NAMES = {"ideal": "01_ideal_flow.csv", "custom": "02_custom_disasters.csv", "random": "03_random_collapses.csv"}
CSV_COLUMNS = ["run", "seed", "time_s", "route_id", "route_name", "scheduled", "spawned", "exited", "active",
               "waiting_entry", "completed", "mean_speed_m_s", "mean_density_p_m2", "peak_local_density_p_m2",
               "queued", "blocked", "net_flow"]
CSV_INT = {"seed", "time_s", "scheduled", "spawned", "exited", "active", "waiting_entry", "completed", "queued",
           "blocked", "net_flow"}
CSV_FLOAT = {"mean_speed_m_s", "mean_density_p_m2", "peak_local_density_p_m2"}
RUN_ID_RE = re.compile(r"[a-f0-9]{32}|mc_[a-z0-9_]+")

OBSERVATION_STREAM = "OBSERVATION_STREAM"
OBSERVATION_ARCHIVE = "OBSERVATION_ARCHIVE"
HIDDEN_TRUTH = "HIDDEN_TRUTH"
ANALYSIS_CSV = "ANALYSIS_CSV"
OPERATOR_REPORT = "OPERATOR_REPORT"
SCENE_CONFIG = "SCENE_CONFIG"
SEPARATION_NOTE = "SEPARATION_NOTE"
UNCLASSIFIED = "UNCLASSIFIED"

PERMITTED = {
    OBSERVATION_STREAM: frozenset({"replay_source", "training", "evaluation"}),
    OBSERVATION_ARCHIVE: frozenset({"training", "evaluation"}),
    HIDDEN_TRUTH: frozenset({"training", "evaluation"}),
    ANALYSIS_CSV: frozenset({"training", "evaluation", "operator_reporting"}),
    OPERATOR_REPORT: frozenset({"operator_reporting", "provenance"}),
    SCENE_CONFIG: frozenset({"simulation_input", "operator_reporting"}),
    SEPARATION_NOTE: frozenset({"provenance"}),
    UNCLASSIFIED: frozenset(),
}
TRUTH_BEARING = {HIDDEN_TRUTH, ANALYSIS_CSV, OPERATOR_REPORT, SCENE_CONFIG, OBSERVATION_ARCHIVE, UNCLASSIFIED}
TIME_SEMANTICS = {
    OBSERVATION_STREAM: "observed_at_s / available_at_s, simulation-relative seconds",
    OBSERVATION_ARCHIVE: "zip of observation streams; includes reports available after any cutoff",
    HIDDEN_TRUTH: "frame/row time_s = state at instant t before movement (kinematic-v3)",
    ANALYSIS_CSV: "one row per route per instant t (kinematic-v3); exited counts the previous step",
    OPERATOR_REPORT: "render time; content covers the whole run including future events",
    SCENE_CONFIG: "configuration; incident start/end in simulation seconds",
    SEPARATION_NOTE: "none",
    UNCLASSIFIED: "unknown",
}


class ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    artifact_class: str
    permitted_uses: frozenset
    truth_bearing: bool
    run_id: str | None
    mode: str | None
    origin: str               # "fresh_run" | "archive_example" | "mc_dataset"
    sha256: str
    bytes: int
    schema_version: str
    timestamp_semantics: str

    def allows(self, use: str) -> bool:
        return use in self.permitted_uses

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["permitted_uses"] = sorted(self.permitted_uses)
        return d


def classify(rel: str) -> tuple[str, str | None, str]:
    """(artifact_class, mode, schema_version) for a path relative to a run directory."""
    parts = rel.split("/")
    name = parts[-1]
    if rel in {f"observable/{m}.jsonl" for m in MODES}:
        return OBSERVATION_STREAM, name[:-6], "eh-v3-route-reports"
    if len(parts) == 3 and parts[:2] == ["observable_ext", "zone_sensing"] and name[:-6] in MODES and name.endswith(".jsonl"):
        return OBSERVATION_STREAM, name[:-6], "eh-v3-zone-sensing-1"
    if rel in ("observable.zip",):
        return OBSERVATION_ARCHIVE, None, "zip"
    if rel in ("hidden_ground_truth.zip", "scenario_params.json") or parts[0] in ("hidden_ground_truth", "hidden_ground_truth_ext"):
        mode = parts[1] if len(parts) > 2 and parts[1] in MODES else None
        return HIDDEN_TRUTH, mode, "kinematic-v3"
    for mode, csv_name in CSV_NAMES.items():
        if rel in (csv_name, f"analysis/{csv_name}"):
            return ANALYSIS_CSV, mode, "kinematic-v3-route-rows"
    if name in ("master_report.html", "report_data.json", "run_manifest.json") or re.fullmatch(r"decisions_[\w-]+\.jsonl", name):
        return OPERATOR_REPORT, None, "report-v3"
    if rel == "examples/event_horizon_scene.json":
        return SCENE_CONFIG, None, "studio-scene-v3"
    if name == "DATA_SEPARATION.txt":
        return SEPARATION_NOTE, None, "text"
    return UNCLASSIFIED, None, "unknown"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_directory(root: Path, *, origin: str, run_id: str | None = None,
                       include: tuple[str, ...] | None = None) -> list[ArtifactRecord]:
    """Register every file under ``root`` (or only ``include`` relative paths)."""
    root = Path(root)
    if run_id is not None and not RUN_ID_RE.fullmatch(run_id):
        raise ArtifactError(f"invalid run id {run_id!r}")
    records = []
    paths = [root / p for p in include] if include else sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        rel = path.relative_to(root).as_posix()
        cls, mode, schema = classify(rel)
        records.append(ArtifactRecord(rel, cls, PERMITTED[cls], cls in TRUTH_BEARING, run_id, mode, origin,
                                      _sha(path), path.stat().st_size, schema, TIME_SEMANTICS[cls]))
    return records


def register_archive_examples(archive_dir: Path) -> list[ArtifactRecord]:
    include = tuple(f"analysis/{n}" for n in CSV_NAMES.values()) + (
        "analysis/report_data.json", "analysis/master_report.html", "examples/event_horizon_scene.json")
    return register_directory(archive_dir, origin="archive_example", include=include)


def write_registry(records: list[ArtifactRecord], path: Path) -> None:
    Path(path).write_text(json.dumps([r.to_dict() for r in records], indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- route-level CSV validation

def _parse_int(name: str, s: str) -> int:
    if not re.fullmatch(r"-?\d+", s):
        raise ArtifactError(f"{name}: expected an integer, got {s!r}")
    return int(s)


def _parse_float(name: str, s: str) -> float:
    try:
        v = float(s)
    except ValueError as exc:
        raise ArtifactError(f"{name}: expected a number, got {s!r}") from exc
    if not math.isfinite(v) or v < 0:
        raise ArtifactError(f"{name}: expected a finite nonnegative number, got {s!r}")
    return v


def validate_route_csv(path: Path, expected_mode: str | None = None) -> dict:
    """Validates one v3 route-level CSV. Returns its summary; raises ArtifactError on any defect."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_COLUMNS:
            raise ArtifactError(f"{Path(path).name}: header {reader.fieldnames} != {CSV_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ArtifactError("CSV has no rows")
    parsed, seen = [], set()
    for i, raw in enumerate(rows):
        r = {}
        for k, s in raw.items():
            if k in CSV_INT:
                r[k] = _parse_int(k, s)
                if k not in ("net_flow",) and r[k] < 0:
                    raise ArtifactError(f"row {i}: {k} must be nonnegative")
            elif k in CSV_FLOAT:
                r[k] = _parse_float(k, s)
            else:
                r[k] = s
        if r["run"] not in MODES:
            raise ArtifactError(f"row {i}: unknown run mode {r['run']!r}")
        key = (r["time_s"], r["route_id"])
        if key in seen:
            raise ArtifactError(f"duplicate row for time {key[0]} route {key[1]}")
        seen.add(key)
        if r["net_flow"] != r["spawned"] - r["exited"]:
            raise ArtifactError(f"row {i}: net_flow != spawned - exited")
        parsed.append(r)
    modes = {r["run"] for r in parsed}
    seeds = {r["seed"] for r in parsed}
    if len(modes) != 1 or len(seeds) != 1:
        raise ArtifactError("a CSV must contain exactly one run mode and one seed")
    mode = modes.pop()
    if expected_mode is not None and mode != expected_mode:
        raise ArtifactError(f"file is for mode {mode!r}, expected {expected_mode!r}")
    routes = sorted({r["route_id"] for r in parsed})
    names = {r["route_id"]: r["route_name"] for r in parsed}
    duration = max(r["time_s"] for r in parsed)
    if len(parsed) != (duration + 1) * len(routes) or {r["time_s"] for r in parsed} != set(range(duration + 1)):
        raise ArtifactError("timestamp grid is not 0..duration for every route")
    for rid in routes:
        series = [r for r in parsed if r["route_id"] == rid]
        if any(b["completed"] < a["completed"] for a, b in zip(series, series[1:])):
            raise ArtifactError(f"route {rid}: cumulative completed decreases")
    return {"mode": mode, "seed": seeds.pop(), "duration_s": duration, "routes": routes, "route_names": names,
            "rows": len(parsed), "parsed": parsed}


def validate_csv_set(paths: dict[str, Path]) -> dict:
    """Cross-run consistency of the three CSVs of one render."""
    summaries = {m: validate_route_csv(p, m) for m, p in paths.items()}
    keys = ("seed", "duration_s", "routes", "route_names")
    first = next(iter(summaries.values()))
    for m, s in summaries.items():
        for k in keys:
            if s[k] != first[k]:
                raise ArtifactError(f"{m}: {k} differs from the other runs")
    if "ideal" in summaries and "random" in summaries:
        a = {(r["time_s"], r["route_id"]): r["scheduled"] for r in summaries["ideal"]["parsed"]}
        b = {(r["time_s"], r["route_id"]): r["scheduled"] for r in summaries["random"]["parsed"]}
        if a != b:
            raise ArtifactError("ideal and random base arrivals differ (they must be identical)")
    if "ideal" in summaries and "custom" in summaries:
        a = {(r["time_s"], r["route_id"]): r["scheduled"] for r in summaries["ideal"]["parsed"]}
        for r in summaries["custom"]["parsed"]:
            if r["scheduled"] < a[(r["time_s"], r["route_id"])]:
                raise ArtifactError("custom schedules fewer arrivals than the shared base population")
    return {m: {k: v for k, v in s.items() if k != "parsed"} for m, s in summaries.items()}


# ---------------------------------------------------------------- hidden truth (offline only)

@dataclass(frozen=True)
class OfflineAccess:
    """Capability for authorized offline dataset construction and retrospective evaluation.
    The inference path never constructs one; readers below refuse to run without it."""
    purpose: str
    holder: str = field(default="offline")

    def __post_init__(self):
        if self.purpose not in ("training", "evaluation", "dataset_construction"):
            raise ArtifactError(f"offline access purpose {self.purpose!r} is not permitted")


def read_hidden_truth(run_dir: Path, mode: str, access: OfflineAccess) -> dict:
    if not isinstance(access, OfflineAccess):
        raise ArtifactError("hidden ground truth requires an OfflineAccess capability")
    base = Path(run_dir) / "hidden_ground_truth"
    folder = base / mode
    def jsonl(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    out = {"scenario": json.loads((base / "scenario.json").read_text(encoding="utf-8")),
           "frames": jsonl(folder / "frames.jsonl"),
           "incident_labels": jsonl(folder / "incident_labels.jsonl"),
           "audit": json.loads((folder / "audit.json").read_text(encoding="utf-8"))}
    with open(folder / "metrics.csv", encoding="utf-8", newline="") as f:
        out["metrics"] = list(csv.DictReader(f))
    zs = Path(run_dir) / "hidden_ground_truth_ext" / mode / "zone_states.jsonl"
    if zs.is_file():
        out["zone_states"] = jsonl(zs)
    return out
