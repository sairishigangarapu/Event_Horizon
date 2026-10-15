"""Twin Studio v3: inference-input boundary, topology-aware features, integrated pipeline, API."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.twin_studio.v3 import engine
from services.twin_studio.v3.artifacts import OfflineAccess, register_directory
from services.twin_studio.v3.dataset import build_samples
from services.twin_studio.v3.features import InferenceInputError, build_features, feature_names
from services.twin_studio.v3.ingest import Measurement
from services.twin_studio.v3.operator_side import safe_metadata, session_from_reports
from services.twin_studio.v3.replay import AuthorizedSnapshot
from services.twin_studio.v3.runs import render_run
from services.twin_studio.v3.staleness import StalenessPolicy
from services.twin_studio.v3.zone_sensing import DEFAULT_ZONE_SOURCES, generate_zone_observations
from services.twin_studio.v3.zone_states import derive_zone_states

REPO = Path(__file__).resolve().parents[1]
TREE_DIR = REPO / "models/environments/eh_twin_studio/v3/trees/dt_bank_v3_topo_1__mc_v1_n16_seed2026"
NEURAL_ROOT = REPO / "models/environments/eh_twin_studio/v3/neural"
NAMES = feature_names()


def _scene():
    return json.loads((engine.ARCHIVE_DIR / "examples/event_horizon_scene.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return render_run(_scene(), tmp_path_factory.mktemp("runs"), run_id="b" * 32, modes=("custom",))


@pytest.fixture(scope="module")
def reports():
    scene = _scene()
    res = engine.simulate(scene, "custom")
    states = derive_zone_states(res, scene)
    return scene, engine.route_observations(scene, res) + generate_zone_observations(states, DEFAULT_ZONE_SOURCES, scene["seed"], 5)


def _snapshot_at(scene, reports, t):
    ctrl, key = session_from_reports("f", reports, safe_metadata(scene), DEFAULT_ZONE_SOURCES)
    while ctrl.watermark_s < t:
        ctrl.advance(key)
    return ctrl.model_view().snapshot()


# ---------------------------------------------------------------- inference-input boundary

def test_feature_builder_rejects_everything_but_an_authorized_snapshot(run_dir, reports):
    records = {r.relative_path: r for r in register_directory(run_dir, origin="fresh_run", run_id="b" * 32)}
    for rel in ("hidden_ground_truth/custom/frames.jsonl", "hidden_ground_truth/custom/incident_labels.jsonl",
                "02_custom_disasters.csv", "run_manifest.json", "observable/custom.jsonl"):
        with pytest.raises(InferenceInputError, match="not an inference input"):
            build_features(records[rel])
    for bad in (str(run_dir / "observable" / "custom.jsonl"), run_dir, reports[1], {"zones": {}}, None):
        with pytest.raises(InferenceInputError):
            build_features(bad)


def test_neighbour_reports_change_a_zones_features_but_own_features_do_not(reports):
    scene, reps = reports
    base = _snapshot_at(scene, reps, 120)
    zones, X = build_features(base)
    i = zones.index("T_ANNEX")
    # Raise the density reported in a neighbour of the annex only.
    bumped = tuple(Measurement(**{**m.__dict__, "value": m.value + 3.0}) if (m.entity_id == "T_WALK_E" and m.variable == "density")
                   else m for m in base.measurements)
    _, X2 = build_features(AuthorizedSnapshot(**{**base.__dict__, "measurements": bumped}))
    nb = NAMES.index("nb_mean_density")
    assert X2[i, nb] != X[i, nb]
    own = [NAMES.index(n) for n in NAMES if n.startswith("own_")]
    assert np.array_equal(np.nan_to_num(X2[i, own], nan=-1), np.nan_to_num(X[i, own], nan=-1))


def test_future_reports_do_not_change_earlier_features(reports):
    scene, reps = reports
    tampered = copy.deepcopy(reps)
    for r in tampered:
        if r["available_at_s"] > 120 and "density_estimate_p_m2" in r["values"]:
            r["values"]["density_estimate_p_m2"] += 4.0
    a = build_features(_snapshot_at(scene, reps, 120))[1]
    b = build_features(_snapshot_at(scene, tampered, 120))[1]
    assert np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))


def test_hidden_reference_changes_labels_but_never_features(run_dir, tmp_path):
    copy_dir = tmp_path / run_dir.name
    shutil.copytree(run_dir, copy_dir)
    zs = copy_dir / "hidden_ground_truth_ext" / "custom" / "zone_states.jsonl"
    states = [json.loads(l) for l in zs.read_text(encoding="utf-8").splitlines()]
    for s in states:
        for z in s["zones"].values():
            z["density"] += 7.0
    zs.write_text("\n".join(json.dumps(s) for s in states) + "\n", encoding="utf-8")
    a = build_samples(run_dir, OfflineAccess("dataset_construction"))
    b = build_samples(copy_dir, OfflineAccess("dataset_construction"))
    assert np.array_equal(np.nan_to_num(a["X"], nan=-1), np.nan_to_num(b["X"], nan=-1))
    assert not np.allclose(np.nan_to_num(a["Y"]), np.nan_to_num(b["Y"]))


# ---------------------------------------------------------------- integrated pipeline

def _neural_dir():
    dirs = sorted(NEURAL_ROOT.glob("*")) if NEURAL_ROOT.is_dir() else []
    return dirs[-1] if dirs else None


@pytest.fixture(scope="module")
def pipeline_out(run_dir, tmp_path_factory):
    from services.twin_studio.v3.pipeline import run
    if not TREE_DIR.is_dir():
        pytest.skip("trained v3 tree bank not present")
    root = tmp_path_factory.mktemp("evals")
    return run(run_dir, root, tree_dir=TREE_DIR, neural_dir=_neural_dir(), policy=StalenessPolicy(age_multiplier=1.0))


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def test_pipeline_persists_forecasts_before_labels_and_keeps_streams_separate(pipeline_out):
    fc = _read(pipeline_out / "forecasts.jsonl")
    assert fc and {r["stream_id"] for r in fc} <= {"stream_a_observation_ewma", "stream_b_architecture"}
    assert all(r["persisted_at_watermark_s"] == r["issue_s"] < r["target_s"] + 60 for r in fc)
    gates = {"T_GATE_NW", "T_GATE_SW", "T_GATE_E"}
    assert all(r["zone_id"] in gates for r in fc if r["variable"] == "exit_open")
    report = json.loads((pipeline_out / "report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["metrics"]["stream_a_observation_ewma"]
    assert report["streams"]["stream_b_architecture"]["tree_bank"] == "dt_bank_v3_topo_1"


def test_pipeline_emits_notices_assessments_and_route_recommendations(pipeline_out):
    notices = _read(pipeline_out / "notices.jsonl")
    assert any(n["notice_type"] == "exit_reported_closed" and n["entity_id"] == "T_GATE_E" for n in notices)
    recs = _read(pipeline_out / "recommendations.jsonl")
    assert recs and {r["status"] for r in recs} <= {"ROUTE_FOUND", "NO_CONFIDENT_ROUTE", "ROUTING_DISABLED"}
    after_closure = [r for r in recs if r["as_of_s"] >= 200 and r["status"] == "ROUTE_FOUND"]
    assert after_closure and all(r["recommended_exit"] != "T_GATE_E" for r in after_closure
                                 if "T_GATE_E" in r["known_closures"])
    assessments = _read(pipeline_out / "assessments.jsonl")
    assert assessments and all(a["recommended_action"] in ("VERIFY", "REVERIFY") and a["reason_codes"] for a in assessments)


def test_editing_hidden_truth_after_render_changes_no_pipeline_output(run_dir, pipeline_out, tmp_path):
    from services.twin_studio.v3.pipeline import run
    tampered = tmp_path / run_dir.name
    shutil.copytree(run_dir, tampered)
    labels = tampered / "hidden_ground_truth" / "custom" / "incident_labels.jsonl"
    events = [json.loads(l) for l in labels.read_text(encoding="utf-8").splitlines()]
    for e in events:
        e["start"], e["end"] = 1, 2
    labels.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    scen = tampered / "hidden_ground_truth" / "scenario.json"
    scene = json.loads(scen.read_text(encoding="utf-8"))
    for e in scene["events"]:
        e["start"], e["end"] = 1, 2
    scen.write_text(json.dumps(scene), encoding="utf-8")
    (tampered / "run_manifest.json").write_text("{}", encoding="utf-8")
    out = run(tampered, tmp_path / "evals", tree_dir=TREE_DIR, neural_dir=_neural_dir(),
              policy=StalenessPolicy(age_multiplier=1.0), evaluate=False)
    for kind in ("forecasts", "assessments", "notices", "recommendations"):
        assert (out / f"{kind}.jsonl").read_bytes() == (pipeline_out / f"{kind}.jsonl").read_bytes(), kind


def test_api_serves_persisted_outputs_only(pipeline_out, monkeypatch):
    from crowd_reliability import api as api_module
    from services.twin_studio.v3 import outputs_view
    monkeypatch.setattr(outputs_view, "EVALUATIONS_ROOT", pipeline_out.parent)
    for fn in ("list_evaluations", "report", "records"):
        orig = getattr(outputs_view, fn)
        monkeypatch.setattr(outputs_view, fn, (lambda o: (lambda *a, **k: o(*a, **{**k, "root": pipeline_out.parent})))(orig))
    client = TestClient(api_module.app)
    listing = client.get("/api/twin-v3/evaluations").json()["evaluations"]
    assert [e["evaluation_id"] for e in listing] == [pipeline_out.name]
    assert client.get(f"/api/twin-v3/evaluations/{pipeline_out.name}").status_code == 200
    a = client.get(f"/api/twin-v3/evaluations/{pipeline_out.name}/assessments?as_of_s=200").json()
    assert all(r["evaluated_at_s"] == 200 for r in a["records"])
    assert client.get(f"/api/twin-v3/evaluations/{pipeline_out.name}/frames").status_code == 404
    assert client.get("/api/twin-v3/evaluations/..%2F..%2Fsecrets").status_code == 404
