"""Phase 5-6 host tests: Twin Studio -> main pipeline adapter and the delayed-label evaluator.

The APIs run in-process over the in-memory object store (logic only); the real MinIO/API stack is
exercised by tests/twin_phase4_integration inside Compose.
"""
import copy
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from fastapi.testclient import TestClient  # noqa: E402

from crowd_reliability.ml.graph import SpatialGraphBuilder  # noqa: E402
from crowd_reliability.ml.tree_bank import TREE_CHANNELS  # noqa: E402
from crowd_reliability.simulation.scenarios import get_default_venue  # noqa: E402
from services.twin_studio import pipeline_adapter as PA  # noqa: E402
from services.twin_studio.api import observation_app, reference_app  # noqa: E402
from services.twin_studio.api.common import FileSessionStore, TokenVerifier  # noqa: E402
from services.twin_studio.bundle import build_run  # noqa: E402
from services.twin_studio.evaluation import STREAM_A_ID, STREAM_B_ID, run_evaluation  # noqa: E402
from services.twin_studio.storage.objects import MemoryObjectStore  # noqa: E402
from services.twin_studio.storage.publisher import publish_to_store  # noqa: E402
from services.twin_studio.storage.readers import ObservationStore, ReferenceStore  # noqa: E402
from tests.test_twin_studio_phase4 import KR, SPEC  # noqa: E402

EPOCH = "2026-10-15T09:00:00Z"


@pytest.fixture(scope="module")
def out():
    return build_run(SPEC, KR)


@pytest.fixture(scope="module")
def venue(out):
    return json.loads(out.observation_files["venue.json"])


@pytest.fixture(scope="module")
def reports(out):
    return [json.loads(l) for l in out.observation_files["observations.jsonl"].splitlines()]


def runner(out, venue, seed=0):
    return PA.TwinArchitectureRunner(venue, out.run_id, EPOCH, seed=seed)


def run_all(r, reports, times=range(0, 61, 5)):
    rows = []
    for t in times:
        rows += r.step(t, reports)
    return rows


# ---------------------------------------------------------------- venue and capacity

def test_venue_layout_keeps_capacity_unknown(venue):
    layout = PA.venue_layout(venue)
    assert len(layout.zones) == 11 and len(layout.connections) == 16
    assert all(z["capacity_definition"] is None and z["capacity_status"] == "unknown" for z in layout.zones)
    g = SpatialGraphBuilder(layout.zones, layout.connections)
    assert sorted(g.unknown_capacity_zones) == sorted(z["zone_id"] for z in layout.zones)
    assert torch.all(g.node_static[:, 5] == 0.0)


def test_missing_capacity_without_declaration_fails():
    zones = copy.deepcopy(get_default_venue().zones)
    zones[0]["capacity_definition"] = None
    with pytest.raises(ValueError, match="capacity_status"):
        SpatialGraphBuilder(zones, [])


def test_main_venue_capacity_features_unchanged():
    v = get_default_venue()
    g = SpatialGraphBuilder(v.zones, v.connections)
    assert g.unknown_capacity_zones == []
    assert [round(float(x), 6) for x in g.node_static[:, 5]] == [round(z["capacity_definition"] / 400.0, 6) for z in v.zones]


def test_venue_from_other_environment_or_with_capacity_is_rejected(venue):
    with pytest.raises(PA.AdapterError):
        PA.venue_layout({**venue, "map_version": "v1.0"})
    bad = copy.deepcopy(venue)
    bad["zones"][0]["capacity"] = 120
    with pytest.raises(PA.AdapterError, match="capacity"):
        PA.venue_layout(bad)


# ---------------------------------------------------------------- observation schema

def test_six_channels_units_zones_and_timestamps_preserved(out, venue, reports):
    zone_ids = {z["zone_id"] for z in venue["zones"]}
    recs = PA.to_observation_records(reports, run_id=out.run_id, zone_ids=zone_ids)
    assert len(recs) == len(reports)
    assert {r.variable for r in recs} == {"density", "speed", "inflow", "outflow", "queue_length", "exit_open"}
    for src, rec in zip(reports, recs):
        assert rec.zone_id == src["zone_id"] and rec.report_id == src["report_id"] and rec.value == float(src["value"])
        assert rec.unit == PA.TWIN_UNIT_TO_PIPELINE[src["unit"]]
        assert rec.observed_at.isoformat().replace("+00:00", "Z") == src["observed_at"]
        assert rec.received_at.isoformat().replace("+00:00", "Z") == src["received_at"]
    assert {r.variable for r in recs if r.zone_id.startswith("T_GATE_")} >= {"exit_open"}
    assert not any(r.variable == "exit_open" for r in recs if not r.zone_id.startswith("T_GATE_"))


@pytest.mark.parametrize("mutate,match", [
    (lambda r: {**r, "label_available_at": r["observed_at"]}, "extra"),
    (lambda r: {**r, "seed": 11}, "extra"),
    (lambda r: {k: v for k, v in r.items() if k != "received_at"}, "missing"),
    (lambda r: {**r, "environment_id": "eh_main"}, "environment"),
    (lambda r: {**r, "schema_version": "eh-twin-schema-0.9"}, "schema"),
    (lambda r: {**r, "variable": "occupancy"}, "unsupported"),
    (lambda r: {**r, "unit": "p/m2"}, "unit"),
    (lambda r: {**r, "zone_id": "concourse"}, "zone"),
    (lambda r: {**r, "run_id": "ehts_" + "0" * 32}, "another run"),
    (lambda r: {**r, "received_at": r["observed_at"]}, "timestamps"),
    (lambda r: {**r, "value": None}, "numeric"),
])
def test_unsupported_or_reference_fields_fail_clearly(out, venue, reports, mutate, match):
    with pytest.raises(PA.AdapterError, match=match):
        PA.to_observation_records([mutate(reports[0])], run_id=out.run_id,
                                  zone_ids={z["zone_id"] for z in venue["zones"]})


# ---------------------------------------------------------------- environment separation and inference

def test_tree_channels_unavailable_and_no_production_artifact_used(out, venue, reports):
    r = runner(out, venue)
    trees, _ = r.pipeline.step("probe", r.epoch + timedelta(seconds=30),
                               PA.to_observation_records(reports, run_id=out.run_id, zone_ids=r.zone_ids))
    assert all(t.channel_names == TREE_CHANNELS and t.channel_masks == [0] * 6 for t in trees)
    assert all(t.tree_version == PA.UNAVAILABLE_TREE_VERSION for t in trees)
    assert r.pipeline.manifest.is_active is False and r.pipeline.manifest.model_hashes == {}
    assert r.manifest["training_status"]["tree_bank"] == "unavailable_for_environment"


def test_missing_environment_manifest_fails(out, venue, monkeypatch, tmp_path):
    monkeypatch.setattr(PA, "ENV_MODEL_DIR", tmp_path)
    with pytest.raises(PA.AdapterError, match="environment_manifest"):
        runner(out, venue)


def test_deterministic_inference(out, venue, reports):
    a, b = run_all(runner(out, venue), reports), run_all(runner(out, venue), reports)
    assert PA.forecast_digest(a) == PA.forecast_digest(b) and a
    assert all(r.unit == PA.EXPECTED_TWIN_UNIT[r.variable] for r in a)
    assert not any(r.variable == "probability_exit_open" and not r.zone_id.startswith("T_GATE_") for r in a)


def test_upstream_gru_train_mode_is_nondeterministic_evidence(out, venue, reports):
    """Evidence for the Person 4 issue: without eval() the GRU's inter-layer dropout is active."""
    r = runner(out, venue)
    x = torch.randn(11, 12, r.pipeline.u_dim)
    r.pipeline.temporal_model.train()
    with torch.no_grad():
        assert not torch.equal(r.pipeline.temporal_model(x), r.pipeline.temporal_model(x))
    r.pipeline.temporal_model.eval()
    with torch.no_grad():
        assert torch.equal(r.pipeline.temporal_model(x), r.pipeline.temporal_model(x))


def test_time_filtering_ignores_reports_received_later(out, venue, reports):
    t = 25
    early = [r for r in reports if r["received_at"] <= f"2026-10-15T09:00:{t:02d}Z"]
    assert len(early) < len(reports)
    assert PA.forecast_digest(run_all(runner(out, venue), reports, [t])) == \
        PA.forecast_digest(run_all(runner(out, venue), early, [t]))


def test_reference_truth_cannot_affect_observations_or_forecasts(out, venue, reports):
    base = PA.forecast_digest(run_all(runner(out, venue), reports))
    tampered = copy.deepcopy(out)
    tampered.reference_files["future_labels.jsonl"] = b""
    tampered.reference_files["reference_states.jsonl"] = b'{"value": 999}\n'
    assert tampered.observation_files == out.observation_files
    again = [json.loads(l) for l in tampered.observation_files["observations.jsonl"].splitlines()]
    assert PA.forecast_digest(run_all(runner(tampered, venue), again)) == base


def test_stream_a_uses_only_received_observations(reports):
    rows = PA.observation_persistence(10, reports, EPOCH, horizons_seconds=(30, 60), gate_ids={"T_GATE_E"})
    later = rows == PA.observation_persistence(10, reports + [{**reports[0], "received_at": "2026-10-15T09:00:59Z",
                                                                "observed_at": "2026-10-15T09:00:55Z",
                                                                "report_id": "R-zzz", "value": 99.0}],
                                                EPOCH, horizons_seconds=(30, 60), gate_ids={"T_GATE_E"})
    assert later and all(r.issue_second == 10 for r in rows)


# ---------------------------------------------------------------- Phase 6: evaluation through the APIs

CONTROL, EVAL = "control-" + "c" * 30, "evaluator-" + "e" * 30
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()


def clients(store, tmp_path, ref_store=None):
    obs = TestClient(observation_app.create_app(ObservationStore(store), FileSessionStore(tmp_path / "replays"),
                                                TokenVerifier([sha(CONTROL)])))
    ref = TestClient(reference_app.create_app(ref_store or ReferenceStore(store), FileSessionStore(tmp_path / "sess"),
                                              TokenVerifier([sha(EVAL)])))
    return obs, ref


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = MemoryObjectStore()
    publish_to_store(SPEC, s, KR, tmp_path_factory.mktemp("bundles"))
    return s


def evaluate(store, tmp_path, ref_store=None):
    obs, ref = clients(store, tmp_path, ref_store)
    run_id = ObservationStore(store).list_run_ids()[0]
    return run_evaluation(obs, ref, run_id=run_id, control_token=CONTROL, evaluator_token=EVAL,
                          out_dir=tmp_path / "eval")


def read_log(rep):
    return [json.loads(l) for l in (Path(rep["out_dir"]) / "forecasts.jsonl").read_text().splitlines()]


def test_evaluation_end_to_end(store, tmp_path):
    rep = evaluate(store, tmp_path)
    assert rep["skill_claim"].startswith("none")
    assert rep["models"][STREAM_B_ID]["training_status"]["temporal"] == "untrained_initial"
    log = read_log(rep)
    for stream in (STREAM_A_ID, STREAM_B_ID):
        res = rep["streams"][stream]
        assert res["issued"] == sum(1 for r in log if r["stream_id"] == stream) > 0
        assert res["released"] == res["issued"] and res["pending"] == 0
        hs = {m["horizon_seconds"] for m in res["metrics"]}
        assert hs <= {30, 60} and {m["variable"] for m in res["metrics"]} <= set(PA.EXPECTED_TWIN_UNIT)
    # every forecast was persisted at its issue watermark, strictly before its label could be eligible
    assert all(r["persisted_at_watermark_second"] == r["issue_second"] < r["target_second"] + 60 for r in log)
    # released labels equal the reference state at the target
    run = ReferenceStore(store).load(rep["run_id"])
    for item in rep["streams"][STREAM_B_ID]["labels"]:
        lab = item["label"]
        assert lab["value"] == run.states[(lab["target_second"], lab["zone_id"], lab["variable"])]["value"]
    assert Path(rep["out_dir"], "report.json").is_file()
    report_text = Path(rep["out_dir"], "report.json").read_text()
    assert SPEC.scenario_id not in report_text and "run_input_sha256" not in report_text


def test_mutating_future_truth_does_not_change_issued_forecasts(store, tmp_path):
    clean = evaluate(store, tmp_path / "a")
    ref_store = ReferenceStore(store)
    run = ref_store.load(clean["run_id"])
    for key, lab in run.labels.items():                   # future truth changed after publication
        if lab["value_valid"]:
            run.labels[key] = {**lab, "value": lab["value"] + 100.0}
    mutated = evaluate(store, tmp_path / "b", ref_store)
    strip = lambda log: [{k: v for k, v in r.items()} for r in log]
    assert strip(read_log(clean)) == strip(read_log(mutated))           # predictions unchanged
    assert clean["observation_view"] == mutated["observation_view"]     # observation access unchanged
    vals = lambda rep: [i["label"]["value"] for i in rep["streams"][STREAM_B_ID]["labels"] if i["label"]["value_valid"]]
    assert all(m == c + 100.0 for c, m in zip(vals(clean), vals(mutated)))   # only the revealed truth moved


def test_evaluation_outputs_are_never_overwritten(store, tmp_path):
    from services.twin_studio.evaluation import EvaluationError, ForecastLog
    p = tmp_path / "f.jsonl"
    ForecastLog(p).close()
    with pytest.raises(EvaluationError):
        ForecastLog(p)
