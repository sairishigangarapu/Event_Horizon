"""Real-MinIO tests: bucket protection, least-privilege identities, conditional publication, integrity."""
import json

import pytest
from botocore.exceptions import ClientError

from conftest import make_spec, secret
from services.twin_studio.bundle import BundleIntegrityError, build_run
from services.twin_studio.storage.markers import catalog_key, obs_key, ref_key, reference_marker_key
from services.twin_studio.storage.objects import (CATALOG_BUCKET, OBSERVATION_BUCKET, REFERENCE_BUCKET,
                                                  AccessDenied, ObjectExists, StorageError)
from services.twin_studio.storage.publisher import publish_to_store
from services.twin_studio.storage.readers import ObservationStore, ReferenceStore, RunUnavailable

BUCKETS = (OBSERVATION_BUCKET, REFERENCE_BUCKET, CATALOG_BUCKET)


def code(exc):
    return exc.response["Error"]["Code"]


# ---------------------------------------------------------------- bucket configuration

@pytest.mark.parametrize("bucket", BUCKETS)
def test_buckets_have_object_lock_and_versioning(admin, bucket):
    lock = admin.client.get_object_lock_configuration(Bucket=bucket)["ObjectLockConfiguration"]
    assert lock["ObjectLockEnabled"] == "Enabled"
    assert lock["Rule"]["DefaultRetention"]["Mode"] in ("GOVERNANCE", "COMPLIANCE")
    assert admin.client.get_bucket_versioning(Bucket=bucket)["Status"] == "Enabled"


def test_published_objects_carry_retention(admin, published_run):
    _, run_id = published_run
    r = admin.client.get_object_retention(Bucket=CATALOG_BUCKET, Key=catalog_key(run_id))["Retention"]
    assert r["Mode"] in ("GOVERNANCE", "COMPLIANCE")


# ---------------------------------------------------------------- immutability mechanisms (experimental validation)

def test_conditional_create_refuses_existing_key(publisher, published_run):
    _, run_id = published_run
    with pytest.raises(ObjectExists):
        publisher.put_if_absent(CATALOG_BUCKET, catalog_key(run_id), b"{}")
    with pytest.raises(ObjectExists):
        publisher.put_if_absent(OBSERVATION_BUCKET, obs_key(run_id, "observations.jsonl"), b"x")
    assert len(publisher.versions(CATALOG_BUCKET, catalog_key(run_id))) == 1


@pytest.mark.parametrize("bucket,key", [(CATALOG_BUCKET, "catalog"), (OBSERVATION_BUCKET, "obs"),
                                        (REFERENCE_BUCKET, "ref")])
def test_publisher_cannot_delete_anything(publisher, published_run, bucket, key):
    _, run_id = published_run
    k = {"catalog": catalog_key(run_id), "obs": obs_key(run_id, "manifest.json"), "ref": ref_key(run_id, "manifest.json")}[key]
    version = publisher.versions(bucket, k)[0]
    with pytest.raises(ClientError) as e:
        publisher.client.delete_object(Bucket=bucket, Key=k)
    assert code(e.value) == "AccessDenied"
    with pytest.raises(ClientError) as e:
        publisher.client.delete_object(Bucket=bucket, Key=k, VersionId=version, BypassGovernanceRetention=True)
    assert code(e.value) == "AccessDenied"
    assert publisher.versions(bucket, k) == [version]


def test_publisher_cannot_weaken_protection(publisher, published_run):
    _, run_id = published_run
    with pytest.raises(ClientError):
        publisher.client.put_bucket_versioning(Bucket=CATALOG_BUCKET, VersioningConfiguration={"Status": "Suspended"})
    with pytest.raises(ClientError):
        publisher.client.put_object_retention(Bucket=CATALOG_BUCKET, Key=catalog_key(run_id),
                                              Retention={"Mode": "GOVERNANCE", "RetainUntilDate": "2026-10-16T00:00:00Z"},
                                              BypassGovernanceRetention=True)
    with pytest.raises(ClientError):
        publisher.client.put_bucket_policy(Bucket=CATALOG_BUCKET, Policy=json.dumps({"Version": "2012-10-17", "Statement": []}))


def test_retention_blocks_version_deletion_even_for_admin_without_bypass(admin, published_run):
    _, run_id = published_run
    version = admin.versions(CATALOG_BUCKET, catalog_key(run_id))[0]
    with pytest.raises(ClientError) as e:
        admin.client.delete_object(Bucket=CATALOG_BUCKET, Key=catalog_key(run_id), VersionId=version)
    assert code(e.value) in ("AccessDenied", "InvalidRequest", "ObjectLocked")
    assert admin.versions(CATALOG_BUCKET, catalog_key(run_id)) == [version]


def test_unconditional_overwrite_cannot_change_a_published_run(publisher, keyring, session_tag, tmp_path, obs_reader,
                                                               ref_reader):
    """Documents the limit of conditional writes: a holder of the publisher credential CAN add a new
    version with a plain PUT. Pinned artifacts are unaffected; a second marker version fails closed."""
    run_id = publish_to_store(make_spec(session_tag, "overwrite"), publisher, keyring, tmp_path).run_id
    publisher.client.put_object(Bucket=OBSERVATION_BUCKET, Key=obs_key(run_id, "observations.jsonl"), Body=b"{}\n")
    assert len(publisher.versions(OBSERVATION_BUCKET, obs_key(run_id, "observations.jsonl"))) == 2
    assert ObservationStore(obs_reader).load(run_id).run_id == run_id          # pinned version still served
    publisher.client.put_object(Bucket=CATALOG_BUCKET, Key=catalog_key(run_id), Body=b"{}")
    with pytest.raises(RunUnavailable):
        ObservationStore(obs_reader).load(run_id)
    checks, run = ReferenceStore(ref_reader).verify(run_id)
    assert run is None and ("catalog marker present with a single version", False) in checks
    with pytest.raises(BundleIntegrityError):
        publish_to_store(make_spec(session_tag, "overwrite"), publisher, keyring, tmp_path)


# ---------------------------------------------------------------- least-privilege identities

def test_observation_reader_permissions(obs_reader, published_run):
    _, run_id = published_run
    obs_reader.get(OBSERVATION_BUCKET, obs_key(run_id, "observations.jsonl"))
    obs_reader.get(CATALOG_BUCKET, catalog_key(run_id))
    for call in (lambda: obs_reader.get(REFERENCE_BUCKET, ref_key(run_id, "future_labels.jsonl")),
                 lambda: obs_reader.get(REFERENCE_BUCKET, reference_marker_key(run_id)),
                 lambda: obs_reader.list_keys(REFERENCE_BUCKET, ""),
                 lambda: obs_reader.versions(REFERENCE_BUCKET, ref_key(run_id, "manifest.json")),
                 lambda: obs_reader.put_if_absent(OBSERVATION_BUCKET, "runs/x/evil", b"x"),
                 lambda: obs_reader.put_if_absent(CATALOG_BUCKET, "runs/evil.json", b"x"),
                 lambda: obs_reader.list_keys(OBSERVATION_BUCKET, "")):
        with pytest.raises(AccessDenied):
            call()


def test_reference_reader_permissions(ref_reader, published_run):
    _, run_id = published_run
    ref_reader.get(REFERENCE_BUCKET, ref_key(run_id, "future_labels.jsonl"))
    ref_reader.get(CATALOG_BUCKET, catalog_key(run_id))
    for call in (lambda: ref_reader.get(OBSERVATION_BUCKET, obs_key(run_id, "observations.jsonl")),
                 lambda: ref_reader.list_keys(OBSERVATION_BUCKET, ""),
                 lambda: ref_reader.put_if_absent(REFERENCE_BUCKET, "runs/x/evil", b"x"),
                 lambda: ref_reader.put_if_absent(CATALOG_BUCKET, "runs/evil.json", b"x")):
        with pytest.raises(AccessDenied):
            call()
    with pytest.raises(ClientError) as e:
        ref_reader.client.delete_object(Bucket=REFERENCE_BUCKET, Key=ref_key(run_id, "manifest.json"))
    assert code(e.value) == "AccessDenied"


def test_wrong_credentials_are_rejected(published_run):
    from conftest import ENDPOINT
    from services.twin_studio.storage.objects import s3_store
    _, run_id = published_run
    bad = s3_store(ENDPOINT, secret("minio_obs_reader_access_key"), "wrong-secret-0000000000")
    with pytest.raises(AccessDenied):
        bad.get(CATALOG_BUCKET, catalog_key(run_id))


def test_service_credentials_and_keys_are_all_distinct(keyring):
    root_pw = secret("minio_root_password")
    values = [secret(f"minio_{u}_secret_key") for u in ("publisher", "obs_reader", "ref_reader")]
    values += [secret("reference_api_token"), secret("observation_control_token")]
    assert root_pw not in values and len(set(values)) == len(values)
    assert not keyring.contains_secret(root_pw.encode())
    assert not any(keyring.contains_secret(v.encode()) for v in values)


# ---------------------------------------------------------------- publication integrity

def test_identical_republish_is_idempotent(publisher, keyring, published_run, tmp_path):
    spec, run_id = published_run
    res = publish_to_store(spec, publisher, keyring, tmp_path)
    assert (res.status, res.run_id) == ("already_published", run_id)
    assert len(publisher.versions(CATALOG_BUCKET, catalog_key(run_id))) == 1


class Interrupting:
    """Wraps the real S3 store and fails after N successful object creations."""

    def __init__(self, inner, limit):
        self.inner, self.limit, self.n = inner, limit, 0

    def put_if_absent(self, bucket, key, data):
        if self.n >= self.limit:
            raise StorageError("simulated crash")
        self.n += 1
        return self.inner.put_if_absent(bucket, key, data)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.mark.parametrize("limit", [4, 9, 10])   # mid-observations, all objects but no markers, ref marker only
def test_interrupted_publication_is_invisible_and_retry_recovers(publisher, keyring, session_tag, tmp_path,
                                                                 obs_reader, ref_reader, limit, obs_api):
    spec = make_spec(session_tag, f"interrupt{limit}")
    with pytest.raises(StorageError):
        publish_to_store(spec, Interrupting(publisher, limit), keyring, tmp_path)
    run_id = build_run(spec, keyring).run_id
    assert run_id not in ObservationStore(obs_reader).list_run_ids()
    with pytest.raises(RunUnavailable):
        ObservationStore(obs_reader).load(run_id)
    with pytest.raises(RunUnavailable):
        ReferenceStore(ref_reader).load(run_id)
    assert run_id not in [r["run_id"] for r in obs_api.get("/runs").json()["runs"]]
    assert obs_api.get(f"/runs/{run_id}/venue").status_code == 404
    res = publish_to_store(spec, publisher, keyring, tmp_path)
    assert res.status == "recovered" and res.run_id == run_id
    ObservationStore(obs_reader).load(run_id)
    ReferenceStore(ref_reader).load(run_id)


def test_conflicting_planted_object_fails_with_integrity_error(publisher, keyring, session_tag, tmp_path, obs_reader):
    spec = make_spec(session_tag, "conflict")
    run_id = build_run(spec, keyring).run_id
    publisher.put_if_absent(REFERENCE_BUCKET, ref_key(run_id, "future_labels.jsonl"), b'{"value": 0}\n')
    with pytest.raises(BundleIntegrityError, match="differs"):
        publish_to_store(spec, publisher, keyring, tmp_path)
    with pytest.raises(ObjectExists):    # and nothing was replaced
        publisher.put_if_absent(REFERENCE_BUCKET, ref_key(run_id, "future_labels.jsonl"), b"")
    with pytest.raises(RunUnavailable):
        ObservationStore(obs_reader).load(run_id)


def test_catalog_contains_only_approved_metadata(obs_reader, published_run, keyring):
    spec, run_id = published_run
    data, _ = obs_reader.get(CATALOG_BUCKET, catalog_key(run_id))
    text = data.decode()
    for hidden in ("run_input_sha256", spec.scenario_id, "east_gate", "custom", '"seed"', "gate_closure", "inc_e",
                   "eh-reference", "id_key_version", "reference_objects"):
        assert hidden not in text
    assert not keyring.contains_secret(data)


def test_hidden_digest_and_keys_absent_from_observation_bucket(obs_reader, ref_reader, published_run, keyring):
    _, run_id = published_run
    ref_manifest = json.loads(ref_reader.get(REFERENCE_BUCKET, ref_key(run_id, "manifest.json"))[0])
    digest = ref_manifest["run_input_sha256"]
    for name in ("observations.jsonl", "venue.json", "manifest.json"):
        data, _ = obs_reader.get(OBSERVATION_BUCKET, obs_key(run_id, name))
        assert digest.encode() not in data and not keyring.contains_secret(data)
