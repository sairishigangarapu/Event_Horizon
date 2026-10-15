"""Publication markers and object layout (Phase 4).

Both markers are DETERMINISTIC (no wall-clock time) so an interrupted publication can be retried and
an identical republish compares byte-for-byte. Each pins every artifact by version ID, SHA-256 and size.

reference marker  eh-reference/published/<run_id>.json   (reference-side; full object pins)
catalog marker    eh-catalog/runs/<run_id>.json          (sanitized; observation pins + reference
                                                          marker SHA-256; written LAST = visibility)
"""
from __future__ import annotations

import json

from services.twin_studio.bundle import OBS_FILES, REF_FILES

MARKER_SCHEMA_VERSION = "twin_marker_v1"
CATALOG_FIELDS = ("marker_kind", "marker_schema_version", "run_id", "scenario_id", "environment_id", "map_version",
                  "epoch_utc", "sample_seconds", "bundle_schema_version", "observation_schema_version",
                  "observation_manifest_sha256", "observation_objects", "reference_marker_sha256")
PIN_FIELDS = ("key", "version_id", "sha256", "bytes")
REFERENCE_MARKER_FIELDS = ("marker_kind", "marker_schema_version", "run_id", "run_id_scheme", "id_key_version",
                           "observation_manifest_sha256", "reference_manifest_sha256", "observation_objects",
                           "reference_objects")


def obs_key(run_id: str, name: str) -> str:
    return f"runs/{run_id}/{name}"


def ref_key(run_id: str, name: str) -> str:
    return f"runs/{run_id}/{name}"


def reference_marker_key(run_id: str) -> str:
    return f"published/{run_id}.json"


def catalog_key(run_id: str) -> str:
    return f"runs/{run_id}.json"


def marker_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def build_reference_marker(run_id: str, ref_manifest: dict, obs_sha: str, ref_sha: str,
                           obs_pins: dict, ref_pins: dict) -> dict:
    return {"marker_kind": "reference_publication", "marker_schema_version": MARKER_SCHEMA_VERSION,
            "run_id": run_id, "run_id_scheme": ref_manifest["run_id_scheme"],
            "id_key_version": ref_manifest["id_key_version"],
            "observation_manifest_sha256": obs_sha, "reference_manifest_sha256": ref_sha,
            "observation_objects": obs_pins, "reference_objects": ref_pins}


def build_catalog_marker(run_id: str, obs_manifest: dict, obs_sha: str, obs_pins: dict, ref_marker_sha: str) -> dict:
    """Only approved, observation-side metadata. Never: digests of inputs, mode, seed, scenario name,
    reference manifest content, reference paths or key versions."""
    return {"marker_kind": "run_publication", "marker_schema_version": MARKER_SCHEMA_VERSION,
            "run_id": run_id, "scenario_id": obs_manifest["scenario_id"],
            "environment_id": obs_manifest["environment_id"], "map_version": obs_manifest["map_version"],
            "epoch_utc": obs_manifest["epoch_utc"], "sample_seconds": obs_manifest["sample_seconds"],
            "bundle_schema_version": obs_manifest["bundle_schema_version"],
            "observation_schema_version": obs_manifest["observation_schema_version"],
            "observation_manifest_sha256": obs_sha, "observation_objects": obs_pins,
            "reference_marker_sha256": ref_marker_sha}


def check_catalog_marker(doc: dict, run_id: str) -> None:
    if set(doc) != set(CATALOG_FIELDS):
        raise ValueError("catalog marker fields are not the approved set")
    if doc["marker_kind"] != "run_publication" or doc["marker_schema_version"] != MARKER_SCHEMA_VERSION:
        raise ValueError("catalog marker kind/schema incompatible")
    if doc["run_id"] != run_id or set(doc["observation_objects"]) != set(OBS_FILES):
        raise ValueError("catalog marker identity or object set mismatch")
    for name, pin in doc["observation_objects"].items():
        if set(pin) != set(PIN_FIELDS) or pin["key"] != obs_key(run_id, name):
            raise ValueError("catalog marker pin malformed")


def check_reference_marker(doc: dict, run_id: str) -> None:
    if set(doc) != set(REFERENCE_MARKER_FIELDS):
        raise ValueError("reference marker fields are not the approved set")
    if doc["marker_kind"] != "reference_publication" or doc["marker_schema_version"] != MARKER_SCHEMA_VERSION:
        raise ValueError("reference marker kind/schema incompatible")
    if (doc["run_id"] != run_id or set(doc["observation_objects"]) != set(OBS_FILES)
            or set(doc["reference_objects"]) != set(REF_FILES)):
        raise ValueError("reference marker identity or object set mismatch")
    for name, pin in doc["reference_objects"].items():
        if set(pin) != set(PIN_FIELDS) or pin["key"] != ref_key(run_id, name):
            raise ValueError("reference marker pin malformed")
