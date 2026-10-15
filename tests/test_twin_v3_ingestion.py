"""Twin Studio v3: provenance, artifact roles, CSV validation, strict ingestion, zone labels."""
import copy
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.twin_studio.v3 import CANONICAL_ARCHIVE_SHA256, engine
from services.twin_studio.v3.artifacts import (
    ANALYSIS_CSV, HIDDEN_TRUTH, OBSERVATION_ARCHIVE, OBSERVATION_STREAM, OPERATOR_REPORT, UNCLASSIFIED, ArtifactError,
    OfflineAccess, read_hidden_truth, register_archive_examples, register_directory, validate_csv_set,
    validate_route_csv)
from services.twin_studio.v3.ingest import (IngestContext, IngestError, ReportNormalizer, route_source_specs, to_utc)
from services.twin_studio.v3.runs import render_run
from services.twin_studio.v3.zone_sensing import DEFAULT_ZONE_SOURCES, generate_zone_observations, source_specs
from services.twin_studio.v3.zone_states import derive_zone_states
from services.twin_studio.zones import load_venue

WORKSPACE_ZIP = Path(__file__).resolve().parents[2] / "Event_Horizon_Twin_Studio.zip"
VENUE = load_venue()
ZONES = [z["zone_id"] for z in VENUE["zones"]]
GATES = [z["zone_id"] for z in VENUE["zones"] if z["zone_type"] == "exit"]


@pytest.fixture(scope="module")
def scene():
    return json.loads((engine.ARCHIVE_DIR / "examples/event_horizon_scene.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def custom(scene):
    return engine.simulate(scene, "custom")


@pytest.fixture(scope="module")
def states(scene, custom):
    return derive_zone_states(custom, scene)


@pytest.fixture(scope="module")
def ctx(scene):
    routes = [r["id"] for r in scene["routes"]]
    sources = {**route_source_specs(engine.observation_sources(scene), routes), **source_specs(DEFAULT_ZONE_SOURCES, ZONES, GATES)}
    return IngestContext("run", frozenset(routes), frozenset(ZONES), frozenset(GATES), sources)


@pytest.fixture(scope="module")
def reports(scene, custom, states):
    return engine.route_observations(scene, custom) + generate_zone_observations(states, DEFAULT_ZONE_SOURCES, scene["seed"], 5)


# ---------------------------------------------------------------- provenance

def test_archive_copy_matches_canonical_hash_and_every_file():
    record = engine.source_record()
    assert record["archive_sha256"] == CANONICAL_ARCHIVE_SHA256 != "e3d2e8df8bae997fee741d54923a50999ada261e1fcfff6320cd71649213f97d"
    engine.verify_archive_copy()
    assert len(record["files"]) == 31
    if WORKSPACE_ZIP.is_file():
        assert hashlib.sha256(WORKSPACE_ZIP.read_bytes()).hexdigest() == CANONICAL_ARCHIVE_SHA256


def test_engine_is_kinematic_v3_and_venue_geometry_unchanged(custom):
    assert custom["modelVersion"] == "kinematic-v3"
    venue_js = engine.ARCHIVE_DIR / "dist" / "venue.js"
    assert hashlib.sha256(venue_js.read_bytes()).hexdigest() == VENUE["source"]["venue_js_sha256"]


def test_archive_example_csvs_reproduce_from_scene(scene):
    for mode, name in (("ideal", "01_ideal_flow.csv"), ("custom", "02_custom_disasters.csv"), ("random", "03_random_collapses.csv")):
        rows = engine.simulate(scene, mode)["rows"]
        shipped = list(csv.DictReader(open(engine.ARCHIVE_DIR / "analysis" / name, encoding="utf-8-sig")))
        assert len(rows) == len(shipped)
        assert all(str(r[k]) == s[k] for r, s in zip(rows, shipped) for k in r)


# ---------------------------------------------------------------- artifact roles

@pytest.fixture(scope="module")
def run_dir(scene, tmp_path_factory):
    return render_run(scene, tmp_path_factory.mktemp("outputs"), run_id="a" * 32)


def test_every_render_artifact_is_classified(run_dir):
    records = register_directory(run_dir, origin="fresh_run", run_id="a" * 32)
    by_path = {r.relative_path: r for r in records}
    assert not [r for r in records if r.artifact_class == UNCLASSIFIED]
    assert by_path["observable/custom.jsonl"].artifact_class == OBSERVATION_STREAM
    assert by_path["observable_ext/zone_sensing/custom.jsonl"].artifact_class == OBSERVATION_STREAM
    assert by_path["observable.zip"].artifact_class == OBSERVATION_ARCHIVE
    assert by_path["hidden_ground_truth/custom/frames.jsonl"].artifact_class == HIDDEN_TRUTH
    assert by_path["hidden_ground_truth_ext/custom/zone_states.jsonl"].artifact_class == HIDDEN_TRUTH
    assert by_path["02_custom_disasters.csv"].artifact_class == ANALYSIS_CSV
    assert by_path["run_manifest.json"].artifact_class == OPERATOR_REPORT
    for r in records:
        assert not r.allows("inference")                      # no file is ever a direct inference input
        if r.artifact_class in (HIDDEN_TRUTH, ANALYSIS_CSV, OPERATOR_REPORT, OBSERVATION_ARCHIVE):
            assert r.truth_bearing and not r.allows("replay_source")


def test_archive_examples_are_distinguished_from_fresh_runs():
    records = register_archive_examples(engine.ARCHIVE_DIR)
    assert {r.origin for r in records} == {"archive_example"}
    assert all(r.truth_bearing for r in records)


def test_unknown_files_fail_closed(tmp_path):
    (tmp_path / "surprise.bin").write_bytes(b"x")
    (rec,) = register_directory(tmp_path, origin="fresh_run")
    assert rec.artifact_class == UNCLASSIFIED and not rec.permitted_uses


def test_csv_set_validates_and_is_consistent(run_dir):
    summary = validate_csv_set({m: run_dir / n for m, n in
                                (("ideal", "01_ideal_flow.csv"), ("custom", "02_custom_disasters.csv"), ("random", "03_random_collapses.csv"))})
    assert {s["rows"] for s in summary.values()} == {602}


@pytest.mark.parametrize("mutate, match", [
    (lambda rows: rows.__setitem__(0, rows[0].replace("time_s", "t")), "header"),
    (lambda rows: rows.insert(2, rows[1]), "duplicate"),
    (lambda rows: rows.__setitem__(1, rows[1].replace("custom,", "ideal,", 1)), "one run mode"),
    (lambda rows: rows.__setitem__(1, rows[1].rsplit(",", 1)[0] + ",999"), "net_flow"),
])
def test_csv_defects_are_rejected(run_dir, tmp_path, mutate, match):
    lines = (run_dir / "02_custom_disasters.csv").read_text(encoding="utf-8-sig").splitlines()
    mutate(lines)
    bad = tmp_path / "02_custom_disasters.csv"
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match=match):
        validate_route_csv(bad, "custom")


def test_hidden_truth_requires_offline_capability(run_dir):
    with pytest.raises(ArtifactError):
        read_hidden_truth(run_dir, "custom", access=None)
    with pytest.raises(ArtifactError):
        OfflineAccess("inference")
    truth = read_hidden_truth(run_dir, "custom", OfflineAccess("evaluation"))
    assert truth["zone_states"] and truth["frames"]


# ---------------------------------------------------------------- strict ingestion

def test_all_generated_reports_pass_and_units_are_normalised_once(ctx, reports):
    n = ReportNormalizer(ctx)
    ms = n.normalize_many(reports)
    units = {(m.variable, m.unit) for m in ms}
    assert ("entry_rate", "persons/s") in units and ("density", "persons/m2") in units and ("exit_open", "probability") in units
    assert n.normalize(reports[0]) == []                      # exact duplicate ignored


def _first(reports, kind):
    return copy.deepcopy(next(r for r in reports if r["source_kind"] == kind and r["observed_at_s"] >= 20))


@pytest.mark.parametrize("kind, mutate, match", [
    ("camera", lambda r: r.update(route_id="nope"), "unknown route_id|observation_id"),
    ("camera", lambda r: r.update(source_id="ghost"), "unknown source_id"),
    ("camera", lambda r: r.update(extra=1), "fields not in the contract"),
    ("camera", lambda r: r.pop("values"), "fields not in the contract"),
    ("camera", lambda r: r["values"].update(density_estimate_p_m2=float("nan")), "finite"),
    ("camera", lambda r: r["values"].update(density_estimate_p_m2=-1.0), "finite number"),
    ("camera", lambda r: r["values"].update(density_estimate_p_m2=150.0), "finite number"),   # e.g. wrong unit
    ("camera", lambda r: r["values"].update(visible_people_estimate=True), "whole number"),
    ("camera", lambda r: r.update(available_at_s=r["observed_at_s"] - 1), "observed_at_s must be <="),
    ("camera", lambda r: r.update(available_at_s=r["available_at_s"] + 7), "delivery latency"),
    ("camera", lambda r: r.update(observed_at_s=r["observed_at_s"] + 1, available_at_s=r["available_at_s"] + 1), "grid"),
    ("camera", lambda r: r.update(observed_at_s=float("inf")), "simulation time"),
    ("camera", lambda r: r.update(source_kind="warden"), "is a camera"),
    ("sensor", lambda r: r["values"].update(count_window_s=2), "count_window_s"),
    ("warden", lambda r: r["values"].update(movement_report="running"), "movement_report"),
    ("gate_sensor", lambda r: r["values"].update(exit_state="ajar"), "exit_state"),
    ("gate_sensor", lambda r: r.update(schema_version="other"), "schema_version"),
    ("zone_counter", lambda r: r["values"].update(count_window_s=1), "count_window_s"),
])
def test_malformed_reports_are_rejected(ctx, reports, kind, mutate, match):
    r = _first(reports, kind)
    mutate(r)
    with pytest.raises(IngestError, match=match):
        ReportNormalizer(ctx).normalize(r)


def test_gate_sensor_on_non_gate_zone_and_conflicting_duplicate_rejected(ctx, reports):
    g = _first(reports, "gate_sensor")
    g["zone_id"] = "T_COURTYARD"
    g["observation_id"] = f"{g['source_id']}:T_COURTYARD:{g['observed_at_s']}"
    with pytest.raises(IngestError):
        ReportNormalizer(ctx).normalize(g)
    r = _first(reports, "camera")
    n = ReportNormalizer(ctx)
    n.normalize(r)
    other = copy.deepcopy(r)
    other["values"]["density_estimate_p_m2"] += 1
    with pytest.raises(IngestError, match="duplicate observation_id"):
        n.normalize(other)


def test_utc_mapping_requires_aware_epoch():
    with pytest.raises(IngestError):
        to_utc(datetime(2026, 10, 15, 9), 5)
    assert to_utc(datetime(2026, 10, 15, 9, tzinfo=timezone.utc), 5).tzinfo is not None


# ---------------------------------------------------------------- zone labels and zone sensing

def test_zone_states_respect_v3_semantics(custom, states):
    assert [s["time_s"] for s in states] == list(range(0, 301, 5))
    first = states[0]["zones"]
    assert all(z["inflow"] is None and z["outflow"] is None for z in first.values())
    for s in states:
        for zid, z in s["zones"].items():
            assert (z["exit_open"] is not None) == (zid in GATES)
            if z["occupancy"] == 0:
                assert z["speed"] is None                 # no value, not zero
    # the scene's 'exit collaps' barrier (150-300 s, end of Route 2) closes T_GATE_E
    gate_e = {s["time_s"]: s["zones"]["T_GATE_E"]["exit_open"] for s in states}
    assert gate_e[145] == 1.0 and gate_e[150] == 0.0 and gate_e[295] == 0.0


def test_zone_reports_are_causal_and_never_report_absent_values(scene, states):
    reports = generate_zone_observations(states, DEFAULT_ZONE_SOURCES, scene["seed"], 5)
    cut = 120
    future_changed = copy.deepcopy(states)
    for s in future_changed:
        if s["time_s"] > cut:
            for z in s["zones"].values():
                z["density"] += 5.0
                z["queue_length"] += 50
    again = generate_zone_observations(future_changed, DEFAULT_ZONE_SOURCES, scene["seed"], 5)
    early = lambda rs: [r for r in rs if r["observed_at_s"] <= cut]
    assert early(reports) == early(again)
    for r in reports:
        if r["source_kind"] == "zone_camera":
            z = states[r["observed_at_s"] // 5]["zones"][r["zone_id"]]
            assert ("mean_speed_estimate_m_s" in r["values"]) == (z["speed"] is not None)
