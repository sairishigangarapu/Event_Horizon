"""Write a v3 run directory exactly as the archive's POST /api/render does, plus the extensions.

Canonical outputs (identical procedure to archive lab_server.py):
  outputs/<run_id>/observable/<mode>.jsonl, observable.zip
  outputs/<run_id>/hidden_ground_truth/{scenario.json, <mode>/frames.jsonl|incident_labels.jsonl|metrics.csv|audit.json}
  outputs/<run_id>/hidden_ground_truth.zip, DATA_SEPARATION.txt
  outputs/<run_id>/01_ideal_flow.csv, 02_custom_disasters.csv, 03_random_collapses.csv, run_manifest.json
  outputs/<run_id>/master_report.html (optional)
Extensions (sibling folders, so the archive's own zips stay consistent with its folders):
  outputs/<run_id>/observable_ext/zone_sensing/<mode>.jsonl        eh-v3-zone-sensing-1 reports
  outputs/<run_id>/hidden_ground_truth_ext/<mode>/zone_states.jsonl  eh-v3-zone-states-1 labels
  outputs/<run_id>/hidden_ground_truth_ext/zone_sources.json        extension source settings
Existing run directories are never overwritten.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import uuid
from pathlib import Path

from services.twin_studio.v3 import MODES
from services.twin_studio.v3.artifacts import CSV_NAMES
from services.twin_studio.v3.engine import ARCHIVE_DIR, observations_module, simulate, simulation_module
from services.twin_studio.v3.zone_sensing import DEFAULT_ZONE_SOURCES, ZoneSource, generate_zone_observations
from services.twin_studio.v3.zone_states import derive_zone_states

SAMPLE_SECONDS = 5


def _jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n")


def _master_report(scene_path: Path, out_path: Path) -> None:
    """The archive's own report CLI, in UTF-8 mode (its file reads assume UTF-8; see SOURCE.json)."""
    subprocess.run([sys.executable, "-X", "utf8", str(ARCHIVE_DIR / "reporting.py"), str(scene_path), "--out", str(out_path)],
                   check=True, cwd=str(ARCHIVE_DIR), capture_output=True)


def render_run(scene: dict, out_root: Path, *, run_id: str | None = None, modes=MODES, include_report: bool = False,
               zone_sources=DEFAULT_ZONE_SOURCES) -> Path:
    simulation_module().validate(scene)
    observations_module().sources_for(scene)
    run_id = run_id or uuid.uuid4().hex
    directory = Path(out_root) / run_id
    if directory.exists():
        raise FileExistsError(f"run directory {directory} exists; outputs are never overwritten")
    results = {m: simulate(scene, m) for m in modes}
    directory.mkdir(parents=True)
    observations_module().export_separated(directory, scene, results)
    for mode in modes:
        rows = results[mode]["rows"]
        with open(directory / CSV_NAMES[mode], "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {"model": "event-horizon-kinematic-v3", "config": scene, "events": {m: r["events"] for m, r in results.items()},
                "note": "Synthetic movement. Incident log is scenario truth; keep it out of prediction inputs."}
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if include_report:
        scene_path = directory / "hidden_ground_truth" / "scenario.json"
        _master_report(scene_path, directory / "master_report.html")

    sources = [s if isinstance(s, ZoneSource) else ZoneSource(**s) for s in zone_sources]
    ext_truth = directory / "hidden_ground_truth_ext"
    ext_truth.mkdir()
    (ext_truth / "zone_sources.json").write_text(json.dumps([s.to_dict() for s in sources], indent=2), encoding="utf-8")
    for mode in modes:
        states = derive_zone_states(results[mode], scene, sample_seconds=SAMPLE_SECONDS)
        _jsonl(ext_truth / mode / "zone_states.jsonl", states)
        _jsonl(directory / "observable_ext" / "zone_sensing" / f"{mode}.jsonl",
               generate_zone_observations(states, sources, scene["seed"], SAMPLE_SECONDS))
    return directory
