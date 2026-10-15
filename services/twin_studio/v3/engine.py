"""Loads the canonical v3 engine and report generator from the byte-identical archive copy.

The archive files use top-level imports (``from simulation import ...``). The parent Twin Studio
subsystem and the archive's own tests may also put a different ``simulation`` module on sys.path,
so the v3 modules are loaded under private names and the ``simulation`` alias is only present while
``observations.py`` executes. Nothing in archive/ is modified.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

from services.twin_studio.v3 import CANONICAL_ARCHIVE_SHA256, ENGINE_MODEL_VERSION, MODES

V3_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = V3_DIR / "archive" / "event_horizon_twin_studio"
SOURCE_PATH = V3_DIR / "SOURCE.json"

_LOCK = threading.Lock()
_CACHE: dict[str, ModuleType] = {}


class ProvenanceError(RuntimeError):
    """The archive copy does not match its recorded provenance."""


def source_record() -> dict:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def verify_archive_copy(files: tuple[str, ...] | None = None) -> None:
    """Check archive files against SOURCE.json (all files when ``files`` is None)."""
    record = source_record()
    if record["archive_sha256"] != CANONICAL_ARCHIVE_SHA256:
        raise ProvenanceError("SOURCE.json does not name the canonical v3 archive")
    for name, meta in record["files"].items():
        if files is not None and name not in files:
            continue
        path = V3_DIR / "archive" / name
        if not path.is_file():
            raise ProvenanceError(f"archive file missing: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            raise ProvenanceError(f"archive file modified: {name}")


def _load(private_name: str, filename: str, alias_simulation: ModuleType | None = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(private_name, ARCHIVE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("simulation")
    try:
        if alias_simulation is not None:
            sys.modules["simulation"] = alias_simulation
        spec.loader.exec_module(module)
    finally:
        if alias_simulation is not None:
            if previous is None:
                sys.modules.pop("simulation", None)
            else:
                sys.modules["simulation"] = previous
    return module


def simulation_module() -> ModuleType:
    with _LOCK:
        if "simulation" not in _CACHE:
            verify_archive_copy(("event_horizon_twin_studio/simulation.py",))
            _CACHE["simulation"] = _load("eh_twin_v3_simulation", "simulation.py")
        return _CACHE["simulation"]


def observations_module() -> ModuleType:
    sim = simulation_module()
    with _LOCK:
        if "observations" not in _CACHE:
            verify_archive_copy(("event_horizon_twin_studio/observations.py",))
            _CACHE["observations"] = _load("eh_twin_v3_observations", "observations.py", alias_simulation=sim)
        return _CACHE["observations"]


def simulate(scene: dict, mode: str) -> dict:
    """Run the canonical v3 engine (validates the scene itself)."""
    if mode not in MODES:
        raise ValueError(f"Unknown run mode: {mode}")
    result = simulation_module().simulate(scene, mode)
    if result.get("modelVersion") != ENGINE_MODEL_VERSION:
        raise ProvenanceError(f"unexpected engine model version {result.get('modelVersion')}")
    return result


def route_observations(scene: dict, result: dict) -> list[dict]:
    """Canonical v3 route-level reports (camera / sensor / warden) for one simulated mode."""
    return observations_module().generate_observations(scene, result)


def observation_sources(scene: dict) -> list[dict]:
    """Validated canonical source settings (defaults when the scene does not configure them)."""
    return [dict(s) for s in observations_module().sources_for(scene)]
