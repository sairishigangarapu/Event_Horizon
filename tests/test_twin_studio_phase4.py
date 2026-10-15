"""Phase 4 host tests: keyed identifiers, keyed bundle validation, publication/reader semantics,
both APIs (in-process), secret initialization and the Compose security configuration.

The storage guarantees (object lock, conditional writes, MinIO policies, network isolation) are tested
against the REAL MinIO deployment in tests/twin_phase4_integration (run inside Compose). The in-memory
store used here only exercises the adapter/reader logic.
"""
import dataclasses
import hashlib
import json
import secrets
import subprocess
import sys
from pathlib import Path

import pytest

from services.twin_studio.bundle import (OBS_FORBIDDEN_TOKENS, OBS_RECORD_FIELDS, BundleIntegrityError,
                                         BundleValidationError, RunSpec, build_run, publish_run, validate_outputs)
from services.twin_studio.keyed_ids import (IdentityKeyring, KeyringError, load_keyring, parse_keyring,
                                            validate_public_alias)
from services.twin_studio.observations import ObservationConfig
from services.twin_studio.run_identity import canonical_json, run_inputs
from services.twin_studio.storage.markers import CATALOG_FIELDS, catalog_key, obs_key
from services.twin_studio.storage.objects import (CATALOG_BUCKET, OBSERVATION_BUCKET, REFERENCE_BUCKET,
                                                  MemoryObjectStore, ObjectExists)
from services.twin_studio.storage.publisher import publish_to_store
from services.twin_studio.storage.readers import ObservationStore, ReferenceStore, RunUnavailable

ROOT = Path(__file__).resolve().parents[1]
SCENE = {
    "duration": 60, "scale": .1, "seed": 11, "randomCount": 1,
    "routes": [{"id": "r_e", "name": "Annex to E gate", "points": [{"x": 800, "y": 580}, {"x": 968, "y": 580}],
                "count": 40, "speed": 1.4, "width": 3},
               {"id": "r_s", "name": "Courtyard to south", "points": [{"x": 450, "y": 450}, {"x": 450, "y": 640}],
                "count": 30, "speed": 1.2, "width": 3}],
    "events": [{"id": "inc_e", "name": "East gate shut", "type": "gate_closure", "gateId": "T_GATE_E", "start": 15, "end": 40}],
}
SPEC = RunSpec("east_gate_closure_seed11", SCENE, "custom", "2026-10-15T09:00:00Z")
K1 = bytes(range(32))
K2 = bytes(range(100, 132))


def ring(active="k1", **extra):
    keys = {"k1": K1, **extra}
    return IdentityKeyring(keys, active)


KR = ring()


@pytest.fixture(scope="module")
def keyed():
    return build_run(SPEC, KR)


@pytest.fixture(scope="module")
def unkeyed():
    return build_run(SPEC)


def obs_records(out):
    return [json.loads(l) for l in out.observation_files["observations.jsonl"].splitlines()]


# ---------------------------------------------------------------- keyring

def test_keyring_parse_and_fail_closed(tmp_path, monkeypatch):
    good = {"format": "eh-twin-id-keyring-v1", "active": "k1", "keys": {"k1": K1.hex()}}
    assert parse_keyring(json.dumps(good)).active_version == "k1"
    bad_docs = ["not json", json.dumps({**good, "format": "x"}), json.dumps({**good, "active": "k2"}),
                json.dumps({**good, "keys": {"k1": "ab" * 8}}), json.dumps({**good, "keys": {"key1": K1.hex()}}),
                json.dumps({**good, "keys": {"k1": "00" * 32}}), json.dumps({**good, "keys": {"k1": K1.hex().upper()}}),
                json.dumps({**good, "keys": {"k1": K1.hex(), "k2": K1.hex()}})]
    for doc in bad_docs:
        with pytest.raises(KeyringError) as e:
            parse_keyring(doc)
        assert K1.hex() not in str(e.value)
    monkeypatch.delenv("EH_TWIN_ID_KEYRING_FILE", raising=False)
    with pytest.raises(KeyringError):
        load_keyring()
    with pytest.raises(KeyringError):
        load_keyring(tmp_path / "missing.json")


def test_keyring_repr_never_shows_key_material():
    text = repr(KR) + str(KR)
    assert K1.hex() not in text and "k1" in text


def test_missing_keyring_makes_publication_fail_safely(tmp_path):
    with pytest.raises(KeyringError):
        publish_to_store(SPEC, MemoryObjectStore(), None, tmp_path)
    assert not (tmp_path / "published").exists()


# ---------------------------------------------------------------- keyed identifiers

def test_same_input_same_key_same_ids(keyed):
    again = build_run(SPEC, ring())
    assert again.run_id == keyed.run_id
    assert again.observation_files == keyed.observation_files and again.reference_files == keyed.reference_files


@pytest.mark.parametrize("changed", [
    dataclasses.replace(SPEC, scene={**SCENE, "seed": 12}),
    dataclasses.replace(SPEC, scene={**SCENE, "events": [{**SCENE["events"][0], "end": 41}]}),
    dataclasses.replace(SPEC, mode="ideal"),
    dataclasses.replace(SPEC, observation_config=ObservationConfig(dropout_probability=0.1)),
    dataclasses.replace(SPEC, sample_seconds=5, label_delay_seconds=90),
    dataclasses.replace(SPEC, public_scenario_id="morning_crowd"),
])
def test_relevant_input_change_changes_keyed_run_id(keyed, changed):
    assert build_run(changed, KR).run_id != keyed.run_id


def test_key_versions_are_explicit_and_distinct(keyed):
    rotated = ring(active="k2", k2=K2)
    other = build_run(SPEC, rotated)
    assert other.run_id != keyed.run_id
    m = json.loads(other.reference_files["manifest.json"])
    assert m["id_key_version"] == "k2" and m["run_id_scheme"] == "hmac_sha256_v1"
    # the retired version still reproduces the old ID explicitly
    inputs_text = keyed.reference_files["run_inputs.json"].decode().rstrip("\n")
    assert rotated.run_id(inputs_text, "k1") == keyed.run_id
    validate_outputs(keyed, rotated)          # old run verifiable with the rotated keyring (k1 retained)


def test_keyed_ids_differ_from_unkeyed_and_report_ids_not_recomputable(keyed, unkeyed):
    assert keyed.run_id != unkeyed.run_id and keyed.run_id.startswith("ehts_") and len(keyed.run_id) == 37
    k_obs, u_obs = obs_records(keyed), obs_records(unkeyed)
    k_ids, u_ids = {r["report_id"] for r in k_obs}, {r["report_id"] for r in u_obs}
    assert not (k_ids & u_ids)
    # Offline guessing attack: an attacker enumerating seeds/modes with the unkeyed construction
    for seed in range(0, 50):
        for mode in ("ideal", "custom", "random"):
            key = f"{SPEC.scenario_id}|{mode}|{seed}|sim_gate_sensor|T_GATE_E|probability_exit_open|15".encode()
            assert "R-" + hashlib.sha256(b"id|" + key).hexdigest()[:20] not in k_ids


def test_keyed_values_identical_to_unkeyed_except_identifiers(keyed, unkeyed):
    strip = lambda rs: sorted((r["source_id"], r["zone_id"], r["variable"], r["simulation_second"], r["value"],
                               r["observed_at"], r["published_at"], r["received_at"]) for r in rs)
    assert strip(obs_records(keyed)) == strip(obs_records(unkeyed))


def test_keyed_bundle_validates_only_with_its_key(keyed):
    assert "run_id is the keyed HMAC of run_inputs" in validate_outputs(keyed, KR)
    with pytest.raises(BundleValidationError, match="keyring lacks key version"):
        validate_outputs(keyed)
    with pytest.raises(BundleValidationError, match="keyring lacks key version"):
        validate_outputs(keyed, IdentityKeyring({"k2": K2}, "k2"))
    with pytest.raises(BundleValidationError):
        validate_outputs(keyed, IdentityKeyring({"k1": K2}, "k1"))   # wrong material under the same name


def test_digest_alias_and_secrets_stay_out_of_observation_side(keyed):
    ref_m = json.loads(keyed.reference_files["manifest.json"])
    obs_m = json.loads(keyed.observation_files["manifest.json"])
    assert ref_m["scenario_id"] == SPEC.scenario_id and ref_m["public_scenario_id"].startswith("scn_")
    assert obs_m["scenario_id"] == ref_m["public_scenario_id"]
    for name, data in keyed.observation_files.items():
        text = data.decode()
        assert ref_m["run_input_sha256"] not in text, name
        assert SPEC.scenario_id not in text and "closure" not in text and "seed" not in text, name
        assert not any(t in text for t in OBS_FORBIDDEN_TOKENS), name
    for data in list(keyed.observation_files.values()) + list(keyed.reference_files.values()):
        assert not KR.contains_secret(data)
    assert all(set(r) == set(OBS_RECORD_FIELDS) for r in obs_records(keyed))


@pytest.mark.parametrize("alias", ["east_gate_closure", "surge_day", "seed_eleven", "run7", "Morning", "random_walk",
                                   "x", "t_gate_e"])
def test_public_alias_rejects_revealing_names(alias):
    with pytest.raises(ValueError):
        validate_public_alias(alias)
    with pytest.raises(ValueError):
        build_run(dataclasses.replace(SPEC, public_scenario_id=alias), KR)


def test_approved_neutral_alias_is_used():
    out = build_run(dataclasses.replace(SPEC, public_scenario_id="morning_crowd"), KR)
    assert json.loads(out.observation_files["manifest.json"])["scenario_id"] == "morning_crowd"
    assert {r["scenario_id"] for r in obs_records(out)} == {"morning_crowd"}
    validate_outputs(out, KR)


# ---------------------------------------------------------------- publication adapter (logic)

def test_publish_idempotent_and_visible(tmp_path):
    store = MemoryObjectStore()
    res = publish_to_store(SPEC, store, KR, tmp_path)
    assert res.status == "published" and res.key_version == "k1"
    assert publish_to_store(SPEC, store, KR, tmp_path).status == "already_published"
    assert ObservationStore(store).list_run_ids() == [res.run_id]
    assert ReferenceStore(store).load(res.run_id).run_id == res.run_id
    catalog = json.loads(store.get(CATALOG_BUCKET, catalog_key(res.run_id))[0])
    assert set(catalog) == set(CATALOG_FIELDS)
    text = json.dumps(catalog)
    for hidden in ("run_input_sha256", "seed", "custom", SPEC.scenario_id, "id_key_version", "eh-reference", "k1"):
        assert hidden not in text


def test_interrupted_publication_invisible_then_recovers(tmp_path):
    store = MemoryObjectStore()
    store.fail_after_puts = 5
    with pytest.raises(Exception):
        publish_to_store(SPEC, store, KR, tmp_path)
    run_id = build_run(SPEC, KR).run_id
    assert ObservationStore(store).list_run_ids() == []
    with pytest.raises(RunUnavailable):
        ObservationStore(store).load(run_id)
    with pytest.raises(RunUnavailable):
        ReferenceStore(store).load(run_id)
    store.fail_after_puts = None
    assert publish_to_store(SPEC, store, KR, tmp_path).status == "recovered"
    ObservationStore(store).load(run_id)


def test_conflicting_existing_object_is_an_integrity_error(tmp_path):
    store = MemoryObjectStore()
    run_id = build_run(SPEC, KR).run_id
    store.put_if_absent(OBSERVATION_BUCKET, obs_key(run_id, "observations.jsonl"), b"planted\n")
    with pytest.raises(BundleIntegrityError, match="differs"):
        publish_to_store(SPEC, store, KR, tmp_path)
    with pytest.raises(RunUnavailable):
        ObservationStore(store).load(run_id)


def test_overwritten_marker_or_object_fails_closed_or_is_ignored(tmp_path):
    store = MemoryObjectStore()
    run_id = publish_to_store(SPEC, store, KR, tmp_path).run_id
    # an extra version of a pinned artifact is ignored: readers fetch the pinned version and verify it
    store.force_put(OBSERVATION_BUCKET, obs_key(run_id, "observations.jsonl"), b"{}\n")
    assert ObservationStore(store).load(run_id).run_id == run_id
    # an extra version of the catalog marker makes the run unavailable (first write wins, fail closed)
    store.force_put(CATALOG_BUCKET, catalog_key(run_id), b"{}")
    with pytest.raises(RunUnavailable):
        ObservationStore(store).load(run_id)
    with pytest.raises(BundleIntegrityError):
        publish_to_store(SPEC, store, KR, tmp_path)


def test_rotation_does_not_silently_create_a_new_identity(tmp_path):
    store = MemoryObjectStore()
    old = publish_to_store(SPEC, store, KR, tmp_path)
    rotated = ring(active="k2", k2=K2)
    res = publish_to_store(SPEC, store, rotated, tmp_path)
    assert (res.status, res.run_id, res.key_version) == ("already_published_under_key_version", old.run_id, "k1")
    new = publish_to_store(SPEC, store, rotated, tmp_path, allow_new_identity_after_rotation=True)
    assert new.status == "published" and new.run_id != old.run_id
    assert sorted(ObservationStore(store).list_run_ids()) == sorted([old.run_id, new.run_id])
    ReferenceStore(store).load(old.run_id)          # old run still readable (readers need no keys)


def test_memory_store_is_create_only():
    store = MemoryObjectStore()
    store.put_if_absent("b", "k", b"1")
    with pytest.raises(ObjectExists):
        store.put_if_absent("b", "k", b"2")


# ---------------------------------------------------------------- APIs (in-process)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from services.twin_studio.api.common import FileSessionStore, TokenVerifier  # noqa: E402
from services.twin_studio.api import observation_app, reference_app  # noqa: E402

CONTROL, EVAL = "control-" + "a" * 30, "evaluator-" + "b" * 30
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    store = MemoryObjectStore()
    res = publish_to_store(SPEC, store, KR, tmp_path_factory.mktemp("pub"))
    return store, res.run_id


@pytest.fixture()
def obs_client(published, tmp_path):
    store, run_id = published
    app = observation_app.create_app(ObservationStore(store), FileSessionStore(tmp_path / "replays"),
                                     TokenVerifier([sha(CONTROL)]))
    return TestClient(app), run_id


@pytest.fixture()
def ref_client(published, tmp_path):
    store, run_id = published
    app = reference_app.create_app(ReferenceStore(store), FileSessionStore(tmp_path / "sessions"),
                                   TokenVerifier([sha(EVAL)]))
    return TestClient(app), run_id, tmp_path / "sessions", store


H_CONTROL = {"Authorization": f"Bearer {CONTROL}"}
H_EVAL = {"Authorization": f"Bearer {EVAL}"}


def assert_no_hidden(text: str):
    for token in ("run_mode", '"seed"', "run_input_sha256", "incidents", "gate_closure", "inc_e", "East gate shut",
                  "eh-reference", "future_labels", "reference_states", "label_available_at", "state_at",
                  SPEC.scenario_id, "id_key_version", CONTROL, EVAL, K1.hex()):
        assert token not in text, token


def test_observation_api_watermark_and_allowlist(obs_client):
    c, run_id = obs_client
    assert c.get("/health").json()["status"] == "ok"
    runs = c.get("/runs").json()["runs"]
    assert [r["run_id"] for r in runs] == [run_id] and set(runs[0]) == set(observation_app.RUN_LIST_FIELDS)
    assert c.post("/replays", json={"run_id": run_id}).status_code == 401
    assert c.post("/replays", json={"run_id": run_id}, headers={"Authorization": "Bearer nope"}).status_code == 401
    rp = c.post("/replays", json={"run_id": run_id}, headers=H_CONTROL).json()
    rid = rp["replay_id"]
    assert rp["watermark_second"] == 0
    assert c.get(f"/runs/{run_id}/reports", params={"replay_id": rid}).json()["count"] == 0
    assert c.post(f"/replays/{rid}/advance", json={"steps": 13}, headers=H_CONTROL).status_code == 422
    assert c.post(f"/replays/{rid}/advance", json={"steps": 1}).status_code == 401
    counts = []
    for _ in range(8):
        w = c.post(f"/replays/{rid}/advance", json={"steps": 2}, headers=H_CONTROL).json()["watermark_second"]
        body = c.get(f"/runs/{run_id}/reports", params={"replay_id": rid}).json()
        epoch = "2026-10-15T09:00:00Z"
        from services.twin_studio.api.timeutil import seconds_since
        assert all(seconds_since(epoch, r["received_at"]) <= w for r in body["reports"])
        assert all(set(r) == set(OBS_RECORD_FIELDS) for r in body["reports"])
        counts.append(body["count"])
        assert_no_hidden(json.dumps(body))
    assert counts == sorted(counts) and counts[-1] > 0
    # as_of may look back, never forward
    future = "2026-10-15T09:59:00Z"
    assert c.get(f"/runs/{run_id}/reports", params={"replay_id": rid, "as_of": future}).status_code == 409
    assert c.get(f"/runs/{run_id}/reports", params={"replay_id": rid, "as_of": "garbage"}).status_code == 422
    past = c.get(f"/runs/{run_id}/reports", params={"replay_id": rid, "as_of": "2026-10-15T09:00:20Z"}).json()
    assert past["as_of_second"] == 20 and past["count"] <= counts[-1]


def test_observation_snapshot_missingness_and_applicability(obs_client):
    c, run_id = obs_client
    rid = c.post("/replays", json={"run_id": run_id}, headers=H_CONTROL).json()["replay_id"]
    snap0 = c.get(f"/runs/{run_id}/snapshot", params={"replay_id": rid}).json()
    assert all(not e["present"] and e["value"] is None and e["age_seconds"] is None for e in snap0["entries"])
    c.post(f"/replays/{rid}/advance", json={"steps": 12}, headers=H_CONTROL)
    snap = c.get(f"/runs/{run_id}/snapshot", params={"replay_id": rid}).json()
    exits = [e for e in snap["entries"] if e["variable"] == "probability_exit_open"]
    assert all(e["applicable"] == e["zone_id"].startswith("T_GATE_") for e in exits)
    assert all(e["value"] is None for e in exits if not e["applicable"])
    present = [e for e in snap["entries"] if e["present"]]
    assert present and all(e["age_seconds"] >= 0 for e in present)
    assert_no_hidden(json.dumps(snap))


def test_observation_api_safe_errors(obs_client):
    c, run_id = obs_client
    unknown = "ehts_" + "0" * 32
    r = c.get(f"/runs/{unknown}/venue")
    assert r.status_code == 404 and r.json() == {"detail": "run not available"}
    assert c.get("/runs/../etc/venue").status_code == 404
    assert c.get(f"/runs/{run_id}/reports", params={"replay_id": "x" * 30}).status_code == 404
    venue = c.get(f"/runs/{run_id}/venue").json()
    assert set(venue) == set(observation_app.VENUE_FIELDS)


def test_reference_api_requires_auth(ref_client):
    c, run_id, *_ = ref_client
    assert c.get("/health").status_code == 200
    for method, path, body in [("post", "/sessions", {"run_id": run_id}), ("get", "/sessions/" + "a" * 32 + "/labels", None),
                               ("get", f"/runs/{run_id}/validation", None)]:
        assert c.request(method, path, json=body).status_code == 401
        assert c.request(method, path, json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.request(method, path, json=body, headers={"Authorization": f"Basic {EVAL}"}).status_code == 401


def test_reference_watermark_registration_and_release(ref_client):
    c, run_id, session_dir, store = ref_client
    s = c.post("/sessions", json={"run_id": run_id}, headers=H_EVAL).json()
    sid = s["session_id"]
    assert (s["current_second"], s["horizons_seconds"], s["label_delay_seconds"]) == (0, [30, 60, 120], 60)
    pred = lambda h, issue=0, **kw: {"issue_second": issue, "predictions": [
        {"zone_id": "T_ANNEX", "variable": "density", "horizon_seconds": h, "value": 0.1, **kw}]}
    assert c.post(f"/sessions/{sid}/predictions", json=pred(30), headers=H_EVAL).status_code == 201
    assert c.post(f"/sessions/{sid}/predictions", json=pred(30), headers=H_EVAL).status_code == 409   # duplicate
    assert c.post(f"/sessions/{sid}/predictions", json=pred(45), headers=H_EVAL).status_code == 422   # not a horizon
    assert c.post(f"/sessions/{sid}/predictions", json=pred(120), headers=H_EVAL).status_code == 422  # beyond run
    assert c.post(f"/sessions/{sid}/predictions", json=pred(30, issue=5), headers=H_EVAL).status_code == 409  # future
    # client timestamps/claims are ignored: label not released before the server watermark reaches 90 s
    lab = c.get(f"/sessions/{sid}/labels", headers=H_EVAL, params={"as_of": "2026-10-15T23:00:00Z"}).json()
    assert lab["released"] == [] and lab["pending"] == 1
    assert c.post(f"/sessions/{sid}/advance", json={"steps": 13}, headers=H_EVAL).status_code == 422
    for _ in range(17):
        c.post(f"/sessions/{sid}/advance", json={"steps": 1}, headers=H_EVAL)
    assert c.get(f"/sessions/{sid}", headers=H_EVAL).json()["current_second"] == 85
    assert c.get(f"/sessions/{sid}/labels", headers=H_EVAL).json()["pending"] == 1
    c.post(f"/sessions/{sid}/advance", json={"steps": 1}, headers=H_EVAL)     # watermark 90 = 30 + 60
    lab = c.get(f"/sessions/{sid}/labels", headers=H_EVAL).json()
    assert len(lab["released"]) == 1 and lab["pending"] == 0
    item = lab["released"][0]
    assert item["label"]["target_second"] == 30 and item["label"]["label_available_at"] == "2026-10-15T09:01:30Z"
    assert set(item["label"]) <= set(reference_app.LABEL_FIELDS)
    # the label equals the stored reference state at the target instant
    ref = ReferenceStore(store).load(run_id)
    assert item["label"]["value"] == ref.states[(30, "T_ANNEX", "density")]["value"]
    # backdating at watermark 90: issue_second 0 is rejected
    assert c.post(f"/sessions/{sid}/predictions", json=pred(60), headers=H_EVAL).status_code == 409
    assert "run_mode" not in json.dumps(lab) and '"seed"' not in json.dumps(lab)
    # registrations survive a restart (durable session file)
    app2 = reference_app.create_app(ReferenceStore(store), FileSessionStore(session_dir), TokenVerifier([sha(EVAL)]))
    assert TestClient(app2).get(f"/sessions/{sid}", headers=H_EVAL).json()["registered_predictions"] == 1


def test_reference_state_query_is_gated(ref_client):
    c, run_id, *_ = ref_client
    sid = c.post("/sessions", json={"run_id": run_id}, headers=H_EVAL).json()["session_id"]
    q = {"simulation_second": 20, "zone_id": "T_ANNEX", "variable": "occupancy"}
    assert c.get(f"/sessions/{sid}/reference-states", params=q, headers=H_EVAL).status_code == 403
    c.post(f"/sessions/{sid}/advance", json={"steps": 12}, headers=H_EVAL)      # 60 s < 80 s
    assert c.get(f"/sessions/{sid}/reference-states", params=q, headers=H_EVAL).status_code == 403
    c.post(f"/sessions/{sid}/advance", json={"steps": 4}, headers=H_EVAL)       # 80 s
    r = c.get(f"/sessions/{sid}/reference-states", params=q, headers=H_EVAL)
    assert r.status_code == 200 and r.json()["state"]["simulation_second"] == 20


def test_reference_validation_endpoint_is_scoped(ref_client):
    c, run_id, *_ = ref_client
    body = c.get(f"/runs/{run_id}/validation", headers=H_EVAL).json()
    assert body["verified"] and all(ch["passed"] for ch in body["checks"])
    text = json.dumps(body)
    assert "sha256" not in text and "version_id" not in text and "run_input" not in text


# ---------------------------------------------------------------- secrets and Compose configuration

def test_init_secrets_generates_distinct_values_and_rotates(tmp_path):
    script = ROOT / "deploy" / "twin_studio" / "init_secrets.py"
    out = subprocess.run([sys.executable, str(script), "--dir", str(tmp_path)], capture_output=True, text=True, check=True)
    files = {p.name: p.read_text().strip() for p in tmp_path.iterdir()}
    keyring = json.loads(files["id_keyring.json"])
    secrets_values = [v for n, v in files.items() if not n.endswith(".sha256") and n != "id_keyring.json"]
    secrets_values += list(keyring["keys"].values())
    assert len(set(secrets_values)) == len(secrets_values)
    assert files["minio_root_password"] not in keyring["keys"].values()
    assert all(v not in out.stdout for v in secrets_values)
    assert files["reference_api_token.sha256"] == hashlib.sha256(files["reference_api_token"].encode()).hexdigest()
    again = subprocess.run([sys.executable, str(script), "--dir", str(tmp_path)], capture_output=True, text=True, check=True)
    assert "nothing" in again.stdout and json.loads((tmp_path / "id_keyring.json").read_text()) == keyring
    subprocess.run([sys.executable, str(script), "--dir", str(tmp_path), "--rotate-id-key"], check=True, capture_output=True)
    rotated = json.loads((tmp_path / "id_keyring.json").read_text())
    assert rotated["active"] == "k2" and rotated["keys"]["k1"] == keyring["keys"]["k1"]
    load_keyring(tmp_path / "id_keyring.json")


yaml = pytest.importorskip("yaml")
COMPOSE = yaml.safe_load((ROOT / "deploy" / "twin_studio" / "compose.yaml").read_text())
HOST = yaml.safe_load((ROOT / "deploy" / "twin_studio" / "compose.reference-host.yaml").read_text())


def test_compose_network_isolation():
    svcs = COMPOSE["services"]
    on = lambda net: sorted(n for n, s in svcs.items() if net in s.get("networks", []))
    assert on("eh_reference") == ["reference-api", "tests"]
    assert on("eh_observation") == ["main-app-probe", "observation-api", "tests"]
    assert svcs["main-app-probe"]["networks"] == ["eh_observation"]
    assert not (set(svcs["observation-api"]["networks"]) & set(svcs["reference-api"]["networks"]))
    for net in ("eh_reference", "eh_obs_storage", "eh_ref_storage", "eh_publish_storage"):
        assert COMPOSE["networks"][net] == {"internal": True}
    assert "ports" not in svcs["reference-api"] and "ports" not in svcs["minio"]
    assert all(p.startswith("127.0.0.1:") for p in svcs["observation-api"]["ports"])
    assert all(p.startswith("127.0.0.1:") for p in HOST["services"]["reference-api"]["ports"])


def test_compose_credentials_are_scoped():
    svcs = COMPOSE["services"]
    secrets_of = lambda n: set(svcs[n].get("secrets", []))
    root = {"minio_root_user", "minio_root_password"}
    assert sorted(n for n in svcs if secrets_of(n) & root) == ["minio", "minio-init", "tests"]
    assert secrets_of("observation-api") == {"minio_obs_reader_access_key", "minio_obs_reader_secret_key",
                                             "observation_control_token_sha256"}
    assert secrets_of("reference-api") == {"minio_ref_reader_access_key", "minio_ref_reader_secret_key",
                                           "reference_api_token_sha256"}
    assert secrets_of("publisher") == {"minio_publisher_access_key", "minio_publisher_secret_key", "id_keyring.json"}
    assert not secrets_of("main-app-probe")
    assert "id_keyring.json" not in secrets_of("observation-api") | secrets_of("reference-api")


def test_images_pinned_and_secrets_ignored():
    text = (ROOT / "deploy" / "twin_studio" / "compose.yaml").read_text()
    assert ":latest" not in text
    for f in ("deploy/twin_studio/Dockerfile", "deploy/twin_studio/minio/Dockerfile"):
        for line in (ROOT / f).read_text().splitlines():
            if line.startswith("FROM "):
                image = line.split()[1]
                assert ":" in image and not image.endswith(":latest"), image
    reqs = (ROOT / "deploy" / "twin_studio" / "requirements.txt").read_text().splitlines()
    assert all("==" in r for r in reqs if r and not r.startswith("#"))
    assert subprocess.run(["git", "check-ignore", "-q", "deploy/twin_studio/secrets/id_keyring.json"], cwd=ROOT).returncode == 0
    assert "**/secrets/" in (ROOT / ".dockerignore").read_text()
