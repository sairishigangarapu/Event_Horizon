"""Live HTTP tests against the running observation and reference API containers."""
import json
import os
from pathlib import Path

from conftest import secret
from services.twin_studio.bundle import OBS_RECORD_FIELDS


def hidden_tokens(spec, keyring_secret_probe):
    return ["run_mode", '"seed"', "run_input_sha256", "incidents", "gate_closure", "inc_e", "East gate shut",
            "eh-reference", "future_labels", "reference_states", "label_available_at", spec.scenario_id,
            "id_key_version", secret("reference_api_token"), secret("observation_control_token"),
            secret("minio_root_password"), secret("minio_obs_reader_secret_key")]


def test_health(obs_api, ref_api):
    assert obs_api.get("/health").json()["status"] == "ok"
    assert ref_api.get("/health").json()["status"] == "ok"


def test_observation_api_end_to_end(obs_api, published_run, control_headers, keyring):
    spec, run_id = published_run
    runs = obs_api.get("/runs").json()["runs"]
    mine = [r for r in runs if r["run_id"] == run_id]
    assert mine and mine[0]["scenario_id"].startswith("scn_")
    assert obs_api.post("/replays", json={"run_id": run_id}).status_code == 401
    rid = obs_api.post("/replays", json={"run_id": run_id}, headers=control_headers).json()["replay_id"]
    last = -1
    for _ in range(10):
        w = obs_api.post(f"/replays/{rid}/advance", json={"steps": 2}, headers=control_headers)
        if w.status_code == 409:
            break
        body = obs_api.get(f"/runs/{run_id}/reports", params={"replay_id": rid})
        assert body.status_code == 200
        text = body.text
        assert all(t not in text for t in hidden_tokens(spec, keyring))
        assert not keyring.contains_secret(body.content)
        data = body.json()
        assert data["count"] >= last and all(set(r) == set(OBS_RECORD_FIELDS) for r in data["reports"])
        last = data["count"]
    assert obs_api.get(f"/runs/{run_id}/reports",
                       params={"replay_id": rid, "as_of": "2030-01-01T00:00:00Z"}).status_code == 409
    snap = obs_api.get(f"/runs/{run_id}/snapshot", params={"replay_id": rid})
    assert snap.status_code == 200 and all(t not in snap.text for t in hidden_tokens(spec, keyring))
    os.makedirs("/tmp/examples", exist_ok=True)
    Path("/tmp/examples/observation_reports.json").write_text(json.dumps(
        {**data, "reports": data["reports"][:2]}, indent=2))


def test_reference_api_auth_and_watermark(ref_api, published_run, eval_headers, control_headers):
    _, run_id = published_run
    assert ref_api.post("/sessions", json={"run_id": run_id}).status_code == 401
    assert ref_api.post("/sessions", json={"run_id": run_id},
                        headers={"Authorization": "Bearer " + secret("observation_control_token")}).status_code == 401
    assert ref_api.post("/sessions", json={"run_id": run_id}, headers=control_headers).status_code == 401
    s = ref_api.post("/sessions", json={"run_id": run_id}, headers=eval_headers).json()
    sid = s["session_id"]
    assert ref_api.get(f"/sessions/{sid}/labels").status_code == 401
    batch = {"issue_second": 0, "predictions": [{"zone_id": "T_GATE_E", "variable": "probability_exit_open",
                                                 "horizon_seconds": 30, "value": 1.0}]}
    assert ref_api.post(f"/sessions/{sid}/predictions", json=batch, headers=eval_headers).status_code == 201
    # a client-supplied timestamp / header cannot unlock the label
    r = ref_api.get(f"/sessions/{sid}/labels", headers={**eval_headers, "Date": "Wed, 01 Jan 2031 00:00:00 GMT"},
                    params={"as_of": "2031-01-01T00:00:00Z", "current_second": 9999})
    assert r.json()["released"] == [] and r.json()["pending"] == 1
    for _ in range(7):
        ref_api.post(f"/sessions/{sid}/advance", json={"steps": 12}, headers=eval_headers)
    lab = ref_api.get(f"/sessions/{sid}/labels", headers=eval_headers).json()
    assert len(lab["released"]) == 1 and lab["released"][0]["label"]["target_at"] == "2026-10-15T09:00:30Z"
    assert lab["released"][0]["label"]["value"] == 0.0          # T_GATE_E closed over [15, 40)
    late = {"issue_second": 0, "predictions": [{"zone_id": "T_GATE_E", "variable": "probability_exit_open",
                                                "horizon_seconds": 60, "value": 0.0}]}
    assert ref_api.post(f"/sessions/{sid}/predictions", json=late, headers=eval_headers).status_code == 409
    Path("/tmp/examples").mkdir(exist_ok=True)
    Path("/tmp/examples/reference_labels.json").write_text(json.dumps(lab, indent=2))


def test_reference_validation_endpoint(ref_api, published_run, eval_headers):
    _, run_id = published_run
    body = ref_api.get(f"/runs/{run_id}/validation", headers=eval_headers).json()
    assert body["verified"] is True and all(c["passed"] for c in body["checks"])
    assert ref_api.get(f"/runs/{run_id}/validation").status_code == 401


def test_no_secret_material_in_image_or_environment(keyring):
    for k, v in os.environ.items():
        assert not keyring.contains_secret(v.encode()), k
    for path in Path("/app").rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert not keyring.contains_secret(data), path
            assert secret("minio_root_password").encode() not in data, path
