"""MinIO publication adapter for Twin Studio runs (Phase 4).

Reuses the Phase 3 path unchanged as the single serializer:
  build_run (one simulation) -> determinism re-check -> validate_outputs -> local staged publication
  (bundle.publish_run, non-overwriting, marker last) -> verify_published.
Then uploads EXACTLY those verified bytes:
  1. every observation and reference object with a conditional create (If-None-Match: *); if an
     object already exists (earlier interrupted attempt) it is reused only if it has a single version
     whose bytes are identical, otherwise BundleIntegrityError;
  2. read-back verification of every object BY VERSION ID (size + SHA-256);
  3. the reference marker (conditional create, verified);
  4. the sanitized catalog marker LAST (conditional create, verified). Both APIs expose a run only
     when this marker exists, has one version and verifies (storage.readers).
Markers are deterministic, so a retry after any interruption converges to the same published state.

Usage (inside the publisher container):
  python -m services.twin_studio.storage.publisher --spec /specs/run.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from services.twin_studio.bundle import (OBS_FILES, REF_FILES, BundleIntegrityError, RunSpec, prepare_spec,
                                         publish_run, spec_from_json, verify_published)
from services.twin_studio.keyed_ids import IdentityKeyring, KeyringError, load_keyring
from services.twin_studio.run_identity import canonical_json, run_inputs
from services.twin_studio.storage.markers import (build_catalog_marker, build_reference_marker, catalog_key,
                                                  marker_bytes, obs_key, ref_key, reference_marker_key)
from services.twin_studio.storage.objects import (CATALOG_BUCKET, OBSERVATION_BUCKET, REFERENCE_BUCKET,
                                                  ObjectExists, ObjectNotFound, ObjectStore, sha256_hex)
from services.twin_studio.storage.readers import ReferenceStore
from services.twin_studio.zones import load_venue

log = logging.getLogger("eh_twin.publisher")


@dataclass
class RemotePublishResult:
    run_id: str
    status: str          # published | recovered | already_published | already_published_under_key_version
    key_version: str
    summary: dict


def _put_verified(store: ObjectStore, bucket: str, key: str, data: bytes) -> tuple[str, bool]:
    """Returns (version_id, created). Never replaces an existing object."""
    try:
        version = store.put_if_absent(bucket, key, data)
        created = True
    except ObjectExists:
        versions = store.versions(bucket, key)
        if len(versions) != 1:
            raise BundleIntegrityError(f"{bucket}/{key} exists with {len(versions)} versions") from None
        version, created = versions[0], False
    stored, _ = store.get(bucket, key, version)
    if stored != data:
        raise BundleIntegrityError(f"{bucket}/{key}: stored object differs from the verified bundle"
                                   + ("" if created else " (conflicting earlier publication)"))
    return version, created


def _pin(key: str, version: str, data: bytes) -> dict:
    return {"key": key, "version_id": version, "sha256": sha256_hex(data), "bytes": len(data)}


def _exists(store: ObjectStore, bucket: str, key: str) -> bool:
    try:
        return bool(store.versions(bucket, key))
    except ObjectNotFound:
        return False


def published_under_other_version(store: ObjectStore, spec: RunSpec, keyring: IdentityKeyring) -> tuple[str, str] | None:
    """(run_id, key_version) if these exact inputs are already in the catalog under a NON-active key."""
    canonical = canonical_json(run_inputs(prepare_spec(spec, load_venue())))
    for version in keyring.versions:
        if version == keyring.active_version:
            continue
        rid = keyring.run_id(canonical, version)
        if _exists(store, CATALOG_BUCKET, catalog_key(rid)):
            return rid, version
    return None


def publish_to_store(spec: RunSpec, store: ObjectStore, keyring: IdentityKeyring, local_root: str | Path, *,
                     allow_new_identity_after_rotation: bool = False,
                     verifier: ReferenceStore | None = None) -> RemotePublishResult:
    if not isinstance(keyring, IdentityKeyring):
        raise KeyringError("an identity keyring is required to publish")
    previous = published_under_other_version(store, spec, keyring)
    if previous and not allow_new_identity_after_rotation:
        return RemotePublishResult(previous[0], "already_published_under_key_version", previous[1], {})

    local = publish_run(spec, local_root, identity=keyring)        # Phase 3 path: validated + staged + marked
    verify_published(local_root, local.run_id)
    run_id = local.run_id
    obs_dir, ref_dir = Path(local_root) / "observations" / run_id, Path(local_root) / "reference" / run_id
    obs_files = {n: (obs_dir / n).read_bytes() for n in OBS_FILES}
    ref_files = {n: (ref_dir / n).read_bytes() for n in REF_FILES}
    obs_manifest, ref_manifest = json.loads(obs_files["manifest.json"]), json.loads(ref_files["manifest.json"])
    if ref_manifest["id_key_version"] != keyring.active_version:
        raise BundleIntegrityError("local bundle was not generated with the active key version")
    obs_sha, ref_sha = sha256_hex(obs_files["manifest.json"]), sha256_hex(ref_files["manifest.json"])

    if _exists(store, CATALOG_BUCKET, catalog_key(run_id)):
        _verify_existing(store, run_id, obs_sha, ref_sha, verifier)
        return RemotePublishResult(run_id, "already_published", keyring.active_version, local.summary)

    reused_any = False      # True if an earlier, interrupted attempt already stored some objects
    obs_pins, ref_pins = {}, {}
    for name in OBS_FILES:
        version, created = _put_verified(store, OBSERVATION_BUCKET, obs_key(run_id, name), obs_files[name])
        obs_pins[name] = _pin(obs_key(run_id, name), version, obs_files[name])
        reused_any |= not created
    for name in REF_FILES:
        version, created = _put_verified(store, REFERENCE_BUCKET, ref_key(run_id, name), ref_files[name])
        ref_pins[name] = _pin(ref_key(run_id, name), version, ref_files[name])
        reused_any |= not created

    ref_marker = marker_bytes(build_reference_marker(run_id, ref_manifest, obs_sha, ref_sha, obs_pins, ref_pins))
    reused_any |= not _put_verified(store, REFERENCE_BUCKET, reference_marker_key(run_id), ref_marker)[1]
    catalog = marker_bytes(build_catalog_marker(run_id, obs_manifest, obs_sha, obs_pins, sha256_hex(ref_marker)))
    _put_verified(store, CATALOG_BUCKET, catalog_key(run_id), catalog)          # visibility switch, LAST
    if verifier is not None:
        _verify_existing(store, run_id, obs_sha, ref_sha, verifier)
    return RemotePublishResult(run_id, "recovered" if reused_any else "published", keyring.active_version,
                               local.summary)


def _verify_existing(store: ObjectStore, run_id: str, obs_sha: str, ref_sha: str, verifier: ReferenceStore | None):
    checker = verifier or ReferenceStore(store)
    checks, run = checker.verify(run_id)
    if run is None:
        failed = [name for name, ok in checks if not ok]
        raise BundleIntegrityError(f"{run_id}: published run fails verification ({failed})")
    if run.marker["observation_manifest_sha256"] != obs_sha or run.marker["reference_manifest_sha256"] != ref_sha:
        raise BundleIntegrityError(f"{run_id}: published content differs from the deterministic result")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Publish one Twin Studio run to MinIO (keyed identity)")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--local-root", type=Path, default=Path(os.environ.get("EH_PUBLISHER_LOCAL_ROOT",
                                                                               "/var/lib/eh-publisher/bundles")))
    parser.add_argument("--allow-new-identity-after-rotation", action="store_true",
                        help="Explicitly publish inputs again under the new active key version")
    args = parser.parse_args(argv)
    from services.twin_studio.storage.objects import s3_store_from_env
    try:
        keyring = load_keyring()
    except KeyringError as exc:
        log.error("publication refused: %s", exc)       # KeyringError never contains key material
        return 2
    store = s3_store_from_env("EH_PUBLISHER")
    spec = spec_from_json(json.loads(args.spec.read_text(encoding="utf-8")))
    try:
        res = publish_to_store(spec, store, keyring, args.local_root,
                               allow_new_identity_after_rotation=args.allow_new_identity_after_rotation)
    except BundleIntegrityError as exc:
        log.error("publication failed integrity check: %s", exc)
        return 3
    print(json.dumps({"run_id": res.run_id, "status": res.status, "id_key_version": res.key_version,
                      **{k: v for k, v in res.summary.items() if k in ("states", "reports", "future_labels")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
