"""Verified, read-only access to published runs (Phase 4).

A run is visible ONLY if all of these hold, re-checked on first access by each reader:
  1. the catalog marker exists and has exactly ONE stored version (first write wins; any later
     version - e.g. an unconditional overwrite - makes the run fail closed);
  2. the marker has exactly the approved fields and pins every bundle object by version ID;
  3. every pinned object is fetched BY VERSION ID and its size and SHA-256 match the pin and the
     bundle manifest;
  4. manifests agree on the run identity and schema versions are the supported ones;
  5. (reference reader) the reference marker also has one version, matches the SHA-256 recorded in
     the catalog marker, and cross-links the same observation manifest.
Failures raise RunUnavailable with a generic message; details go to the server log without
values, digests or credentials.
"""
from __future__ import annotations

import json
import logging
import threading

from services.twin_studio.api.timeutil import seconds_since
from services.twin_studio.bundle import OBS_RECORD_FIELDS, RUN_ID_RE
from services.twin_studio.run_identity import SCHEMA_VERSIONS
from services.twin_studio.storage.markers import (catalog_key, check_catalog_marker, check_reference_marker,
                                                  reference_marker_key)
from services.twin_studio.storage.objects import (CATALOG_BUCKET, OBSERVATION_BUCKET, REFERENCE_BUCKET,
                                                  ObjectNotFound, ObjectStore, StorageError, sha256_hex)

log = logging.getLogger("eh_twin.storage")


class RunUnavailable(LookupError):
    """Unknown, unpublished, incompletely published or integrity-failed run (deliberately generic)."""


class IncompatibleSchema(RuntimeError):
    pass


def _single_version_json(store: ObjectStore, bucket: str, key: str) -> tuple[dict, bytes]:
    versions = store.versions(bucket, key)
    if not versions:
        raise ObjectNotFound(key)
    if len(versions) != 1:
        raise StorageError(f"{bucket}/{key} has {len(versions)} versions; a published marker must have exactly one")
    data, _ = store.get(bucket, key, versions[0])
    return json.loads(data), data


def _fetch_pinned(store: ObjectStore, bucket: str, pins: dict) -> dict[str, bytes]:
    files = {}
    for name, pin in pins.items():
        data, _ = store.get(bucket, pin["key"], pin["version_id"])
        if len(data) != pin["bytes"] or sha256_hex(data) != pin["sha256"]:
            raise StorageError(f"{bucket}/{pin['key']} does not match its pin")
        files[name] = data
    return files


def _check_manifest_files(manifest: dict, files: dict[str, bytes]) -> None:
    for name, entry in manifest["files"].items():
        if name not in files or sha256_hex(files[name]) != entry["sha256"] or len(files[name]) != entry["bytes"]:
            raise StorageError(f"{name} does not match its bundle manifest")


def _jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.splitlines() if line]


class ObservationRun:
    def __init__(self, catalog: dict, manifest: dict, venue: dict, reports: list[dict]):
        self.catalog, self.manifest, self.venue, self.reports = catalog, manifest, venue, reports
        self.run_id = catalog["run_id"]
        self.max_received_second = max((r["_received_second"] for r in reports), default=0)


class ObservationStore:
    """Reads ONLY eh-catalog and eh-observations."""

    def __init__(self, store: ObjectStore):
        self.store = store
        self._cache: dict[str, ObservationRun] = {}
        self._lock = threading.Lock()

    def list_run_ids(self) -> list[str]:
        ids = []
        for key in self.store.list_keys(CATALOG_BUCKET, "runs/"):
            rid = key[len("runs/"):-len(".json")] if key.endswith(".json") else ""
            if RUN_ID_RE.match(rid):
                ids.append(rid)
        return sorted(ids)

    def load(self, run_id: str) -> ObservationRun:
        if not RUN_ID_RE.match(run_id or ""):
            raise RunUnavailable("run not available")
        with self._lock:
            if run_id in self._cache:
                return self._cache[run_id]
        try:
            catalog, _ = _single_version_json(self.store, CATALOG_BUCKET, catalog_key(run_id))
            check_catalog_marker(catalog, run_id)
            if (catalog["bundle_schema_version"] != SCHEMA_VERSIONS["bundle"]
                    or catalog["observation_schema_version"] != SCHEMA_VERSIONS["observations"]):
                raise IncompatibleSchema("incompatible schema version")
            files = _fetch_pinned(self.store, OBSERVATION_BUCKET, catalog["observation_objects"])
            if sha256_hex(files["manifest.json"]) != catalog["observation_manifest_sha256"]:
                raise StorageError("observation manifest does not match the catalog")
            manifest = json.loads(files["manifest.json"])
            _check_manifest_files(manifest, files)
            if (manifest["run_id"] != run_id or manifest["scenario_id"] != catalog["scenario_id"]
                    or manifest["bundle_schema_version"] != SCHEMA_VERSIONS["bundle"]):
                raise StorageError("observation manifest identity mismatch")
            reports = _jsonl(files["observations.jsonl"])
            if any(set(r) != set(OBS_RECORD_FIELDS) or r["run_id"] != run_id for r in reports):
                raise StorageError("observation records violate the allowlist")
        except IncompatibleSchema:
            raise
        except (StorageError, ValueError, KeyError, json.JSONDecodeError) as exc:
            log.warning("observation run %s unavailable: %s", run_id, type(exc).__name__)
            raise RunUnavailable("run not available") from None
        for r in reports:
            r["_received_second"] = seconds_since(manifest["epoch_utc"], r["received_at"])
            r["_observed_second"] = seconds_since(manifest["epoch_utc"], r["observed_at"])
        run = ObservationRun(catalog, manifest, json.loads(files["venue.json"]), reports)
        with self._lock:
            self._cache[run_id] = run
        return run


class ReferenceRun:
    def __init__(self, catalog: dict, marker: dict, manifest: dict, states: list[dict], labels: list[dict]):
        self.catalog, self.marker, self.manifest = catalog, marker, manifest
        self.run_id = catalog["run_id"]
        rule = manifest["label_rule"]
        self.sample_seconds = manifest["sample_seconds"]
        self.label_delay_seconds = rule["label_delay_seconds"]
        self.horizons_seconds = list(rule["horizons_seconds"])
        self.target_variables = list(rule["target_variables"])
        self.last_second = max(s["simulation_second"] for s in states)
        self.zone_ids = sorted({s["zone_id"] for s in states})
        self.states = {(s["simulation_second"], s["zone_id"], s["variable"]): s for s in states}
        self.labels = {(l["issue_second"], l["horizon_seconds"], l["zone_id"], l["variable"]): l for l in labels}


class ReferenceStore:
    """Reads ONLY eh-catalog (approved marker) and eh-reference."""

    def __init__(self, store: ObjectStore):
        self.store = store
        self._cache: dict[str, ReferenceRun] = {}
        self._lock = threading.Lock()

    def verify(self, run_id: str) -> tuple[list[tuple[str, bool]], ReferenceRun | None]:
        """Full re-verification from storage (uncached). Returns scoped check results."""
        checks: list[tuple[str, bool]] = []

        def step(name, fn):
            try:
                result = fn()
                checks.append((name, True))
                return result
            except Exception as exc:  # noqa: BLE001 - every failure is reported as a failed check
                log.warning("reference verification of %s failed at %r: %s", run_id, name, type(exc).__name__)
                checks.append((name, False))
                raise RunUnavailable("run not available") from None

        if not RUN_ID_RE.match(run_id or ""):
            return [("run id format", False)], None
        try:
            catalog, _ = step("catalog marker present with a single version",
                              lambda: _single_version_json(self.store, CATALOG_BUCKET, catalog_key(run_id)))
            step("catalog marker fields approved", lambda: check_catalog_marker(catalog, run_id))
            marker, marker_bytes = step("reference marker present with a single version",
                                        lambda: _single_version_json(self.store, REFERENCE_BUCKET,
                                                                     reference_marker_key(run_id)))
            step("reference marker fields approved", lambda: check_reference_marker(marker, run_id))

            def bound():
                if sha256_hex(marker_bytes) != catalog["reference_marker_sha256"]:
                    raise StorageError("reference marker not bound to catalog")
                if marker["observation_manifest_sha256"] != catalog["observation_manifest_sha256"]:
                    raise StorageError("observation manifest cross-link mismatch")
                if marker["observation_objects"] != catalog["observation_objects"]:
                    raise StorageError("observation pins differ")
            step("markers bound to each other", bound)
            files = step("reference objects match their version pins",
                         lambda: _fetch_pinned(self.store, REFERENCE_BUCKET, marker["reference_objects"]))

            def manifest_ok():
                if sha256_hex(files["manifest.json"]) != marker["reference_manifest_sha256"]:
                    raise StorageError("reference manifest mismatch")
                m = json.loads(files["manifest.json"])
                _check_manifest_files(m, files)
                if m["run_id"] != run_id or m["observation_manifest_sha256"] != catalog["observation_manifest_sha256"]:
                    raise StorageError("reference manifest identity mismatch")
                return m
            manifest = step("reference manifest matches its files and run identity", manifest_ok)
            def schema_ok():
                if manifest["schema_versions"] != SCHEMA_VERSIONS:
                    raise IncompatibleSchema("incompatible schema version")
            step("schema versions supported", schema_ok)
            run = step("reference records parse",
                       lambda: ReferenceRun(catalog, marker, manifest, _jsonl(files["reference_states.jsonl"]),
                                            _jsonl(files["future_labels.jsonl"])))
            return checks, run
        except RunUnavailable:
            return checks, None

    def load(self, run_id: str) -> ReferenceRun:
        with self._lock:
            if run_id in self._cache:
                return self._cache[run_id]
        _, run = self.verify(run_id)
        if run is None:
            raise RunUnavailable("run not available")
        with self._lock:
            self._cache[run_id] = run
        return run
