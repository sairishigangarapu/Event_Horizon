"""Deterministic run identity for Twin Studio runs (Phase 3).

run_id = "ehts_" + first 32 hex digits (128 bits) of SHA-256(canonical_json(run_inputs)).
The full digest is kept in the reference manifest as `run_input_sha256`.

Canonical JSON: UTF-8, sorted keys, separators (",", ":"), ASCII escapes, NaN/Infinity rejected.
Numbers are serialised exactly as given (1 and 1.0 are different spellings and give different IDs,
which can only split identities, never merge different inputs).

Identity inputs are only things that determine the generated artifacts: environment, code
fingerprint, venue hash, scene (with gates and incidents), mode, seed, scenario id, simulated
epoch, sampling, observation and label configuration, schema versions. Wall-clock time, UUIDs,
file times and output hashes are never inputs.

Phase 4: the PUBLIC run ID of published runs is keyed (keyed_ids.RUN_ID_SCHEME); this unkeyed
digest then stays private to the reference bundle (`run_input_sha256`). The unkeyed
`run_id_from_digest` remains only for local Phase 3 bundles (scheme sha256_unkeyed_v1).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION

SUBSYSTEM = Path(__file__).resolve().parent
RUN_ID_PREFIX = "ehts_"
RUN_ID_HEX_DIGITS = 32
SCHEMA_VERSIONS = {
    "timebase": "end_of_interval_v1",
    "zone_states": "zone_states_v1",
    "observations": "eh-twin-schema-1.0",
    "reference_records": "reference_records_v1",
    "future_labels": "future_labels_v1",
    "bundle": "twin_bundle_v1",
    "run_identity": "run_identity_v1",
}
# Modules whose code shapes the artifacts. Line endings are normalised so CRLF/LF checkouts agree.
FINGERPRINT_FILES = ("engine/simulation.py", "zones.py", "timebase.py", "zone_states.py", "observations.py",
                     "labels.py", "bundle.py", "run_identity.py", "keyed_ids.py", "__init__.py")
VENUE_FILE = "venue/twin_venue_v1.json"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _normalised_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def file_sha256(path: Path, normalise_newlines: bool = False) -> str:
    data = _normalised_bytes(path) if normalise_newlines else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def code_fingerprint(root: Path = SUBSYSTEM) -> dict[str, Any]:
    files = {name: file_sha256(root / name, normalise_newlines=True) for name in FINGERPRINT_FILES}
    combined = hashlib.sha256(canonical_json(files).encode()).hexdigest()
    return {"files": files, "sha256": combined}


def venue_sha256(root: Path = SUBSYSTEM) -> str:
    return file_sha256(root / VENUE_FILE, normalise_newlines=True)


def normalise_epoch(epoch: str | datetime) -> str:
    dt = datetime.fromisoformat(epoch.replace("Z", "+00:00")) if isinstance(epoch, str) else epoch
    if dt.tzinfo is None:
        raise ValueError("epoch_utc must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_inputs(spec, *, fingerprint: dict | None = None, venue_hash: str | None = None) -> dict[str, Any]:
    """The complete, canonical identity inputs of a run specification (see bundle.RunSpec)."""
    fp = fingerprint if fingerprint is not None else code_fingerprint()
    inputs = {
        "environment_id": ENVIRONMENT_ID,
        "map_version": MAP_VERSION,
        "venue_sha256": venue_hash if venue_hash is not None else venue_sha256(),
        "code_fingerprint_sha256": fp["sha256"],
        "schema_versions": SCHEMA_VERSIONS,
        "scenario_id": spec.scenario_id,
        "mode": spec.mode,
        "seed": spec.scene["seed"],
        "scene": spec.scene,
        "epoch_utc": normalise_epoch(spec.epoch_utc),
        "sample_seconds": spec.sample_seconds,
        "observation_config": dataclasses.asdict(spec.observation_config),
        "label_config": {"horizons_seconds": list(spec.label_horizons_seconds),
                         "label_delay_seconds": spec.label_delay_seconds,
                         "target_variables": list(spec.label_variables)},
    }
    if getattr(spec, "public_scenario_id", None) is not None:   # Phase 4: explicit public alias shapes outputs
        inputs["public_scenario_id"] = spec.public_scenario_id
    return inputs


def run_input_digest(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(inputs).encode()).hexdigest()


def run_id_from_digest(digest: str) -> str:
    return RUN_ID_PREFIX + digest[:RUN_ID_HEX_DIGITS]
