"""Phase 3 tests: deterministic run identity, future labels, bundle separation, staged immutable publication."""
import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.twin_studio import bundle as B
from services.twin_studio.bundle import (BundleIntegrityError, BundleValidationError, OBS_FORBIDDEN_TOKENS,
                                         OBS_MANIFEST_FIELDS, OBS_RECORD_FIELDS, RunSpec, build_run, is_published,
                                         list_published, publish_run, validate_outputs, verify_published)
from services.twin_studio.engine import simulation as S
from services.twin_studio.labels import future_labels
from services.twin_studio.observations import ObservationConfig
from services.twin_studio.run_identity import code_fingerprint, run_id_from_digest, run_input_digest, run_inputs
from services.twin_studio.zone_states import derive_zone_states
from services.twin_studio.zones import load_venue

VENUE = load_venue()
SCENE = {
    "duration": 60, "scale": .1, "seed": 11, "randomCount": 1,
    "routes": [{"id": "r_e", "name": "Annex to E gate", "points": [{"x": 800, "y": 580}, {"x": 968, "y": 580}],
                "count": 40, "speed": 1.4, "width": 3},
               {"id": "r_s", "name": "Courtyard to south", "points": [{"x": 450, "y": 450}, {"x": 450, "y": 640}],
                "count": 30, "speed": 1.2, "width": 3}],
    "events": [{"id": "inc_e", "name": "East gate shut", "type": "gate_closure", "gateId": "T_GATE_E", "start": 15, "end": 40}],
}
SPEC = RunSpec("twin_fixture_a", SCENE, "custom", "2026-10-15T09:00:00Z")
SPEC_B = RunSpec("twin_fixture_b", {**SCENE, "seed": 99}, "random", "2026-10-15T10:00:00Z")


def with_scene(**kw):
    return dataclasses.replace(SPEC, scene={**SCENE, **kw})


@pytest.fixture(scope="module")
def built():
    return build_run(SPEC)


# ---------------------------------------------------------------- run identity

def test_identical_inputs_give_identical_run_id_and_bytes(built):
    again = build_run(SPEC)
    assert again.run_id == built.run_id and again.run_input_sha256 == built.run_input_sha256
    assert again.observation_files == built.observation_files and again.reference_files == built.reference_files
    assert built.run_id == run_id_from_digest(built.run_input_sha256) and len(built.run_input_sha256) == 64


@pytest.mark.parametrize("changed", [
    lambda: with_scene(seed=12),
    lambda: with_scene(routes=[{**SCENE["routes"][0], "count": 41}] + SCENE["routes"][1:]),
    lambda: with_scene(events=[{**SCENE["events"][0], "end": 41}]),
    lambda: dataclasses.replace(SPEC, mode="ideal"),
    lambda: dataclasses.replace(SPEC, scenario_id="twin_fixture_other"),
    lambda: dataclasses.replace(SPEC, epoch_utc="2026-10-15T09:00:01Z"),
    lambda: dataclasses.replace(SPEC, observation_config=ObservationConfig(dropout_probability=0.1)),
    lambda: dataclasses.replace(SPEC, label_delay_seconds=90),
    lambda: dataclasses.replace(SPEC, label_horizons_seconds=(30, 60)),
])
def test_relevant_input_changes_change_the_run_id(built, changed):
    assert build_run(changed()).run_id != built.run_id


def test_code_and_venue_fingerprints_are_identity_inputs():
    spec = B.prepare_spec(SPEC, VENUE)
    base = run_input_digest(run_inputs(spec))
    fp = code_fingerprint()
    other_fp = {**fp, "sha256": "0" * 64}
    assert run_input_digest(run_inputs(spec, fingerprint=fp)) == base
    assert run_input_digest(run_inputs(spec, fingerprint=other_fp)) != base
    assert run_input_digest(run_inputs(spec, venue_hash="f" * 64)) != base


def test_identity_ignores_wall_clock_and_epoch_spelling(monkeypatch, built):
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2031, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(B, "datetime", Later)
    assert build_run(SPEC).run_id == built.run_id
    assert build_run(dataclasses.replace(SPEC, epoch_utc="2026-10-15T09:00:00+00:00")).run_id == built.run_id
    inputs = json.loads(built.reference_files["run_inputs.json"])
    assert not {"generated_at", "published_at", "uuid", "files", "run_id"} & set(inputs)


# ---------------------------------------------------------------- bundle separation

def test_both_bundles_identify_the_same_run(built):
    obs_m = json.loads(built.observation_files["manifest.json"])
    ref_m = json.loads(built.reference_files["manifest.json"])
    assert obs_m["run_id"] == ref_m["run_id"] == built.run_id
    import hashlib
    assert ref_m["observation_manifest_sha256"] == hashlib.sha256(built.observation_files["manifest.json"]).hexdigest()
    for line in built.observation_files["observations.jsonl"].splitlines():
        assert json.loads(line)["run_id"] == built.run_id


def test_observation_bundle_contains_only_safe_content(built):
    assert set(built.observation_files) == {"observations.jsonl", "venue.json", "manifest.json"}
    assert set(json.loads(built.observation_files["manifest.json"])) == set(OBS_MANIFEST_FIELDS)
    for line in built.observation_files["observations.jsonl"].splitlines():
        r = json.loads(line)
        assert set(r) == set(OBS_RECORD_FIELDS)
        assert "run_mode" not in r and "seed" not in r
    incidents = json.loads(built.reference_files["incidents.json"])["events"]
    assert incidents  # the fixture has a hidden incident...
    for name, data in built.observation_files.items():
        text = data.decode()
        for token in OBS_FORBIDDEN_TOKENS:
            assert token not in text, (name, token)
        for e in incidents:          # ...that never reaches the observation bundle
            assert f'"{e["name"]}"' not in text and f'"{e["id"]}"' not in text


def test_reference_bundle_preserves_full_provenance(built):
    ref_m = json.loads(built.reference_files["manifest.json"])
    assert ref_m["mode"] == "custom" and ref_m["seed"] == 11
    assert ref_m["code_fingerprint"]["sha256"] == code_fingerprint()["sha256"]
    assert ref_m["engine_source"]["original"]["python"]["sha256"].startswith("ce39174c")
    assert json.loads(built.reference_files["incidents.json"])["events"][0]["type"] == "gate_closure"
    inputs = json.loads(built.reference_files["run_inputs.json"])
    assert run_input_digest(inputs) == ref_m["run_input_sha256"]
    assert inputs["scene"]["events"] == SCENE["events"]


def test_reference_only_changes_cannot_change_observation_artifacts(monkeypatch, built):
    monkeypatch.setattr(B, "reference_records", lambda states, **kw: [{"bogus": True}])
    monkeypatch.setattr(B, "future_labels", lambda states, **kw: ([{"bogus": True}], {}))
    altered = build_run(SPEC)
    assert altered.observation_files == built.observation_files
    assert altered.reference_files["reference_states.jsonl"] != built.reference_files["reference_states.jsonl"]


# ---------------------------------------------------------------- future labels

@pytest.fixture(scope="module")
def states():
    spec = B.prepare_spec(SPEC, VENUE)
    return derive_zone_states(S.simulate(spec.scene, spec.mode), VENUE, 5)


def _labels(built):
    return [json.loads(l) for l in built.reference_files["future_labels.jsonl"].splitlines()]


def test_every_label_equals_the_reference_state_at_its_target(built, states):
    by_t = {st["simulation_second"]: st for st in states}
    labels = _labels(built)
    assert labels
    for lab in labels:
        assert lab["target_second"] == lab["issue_second"] + lab["horizon_seconds"]
        zs = by_t[lab["target_second"]]["zones"][lab["zone_id"]]
        expected = zs[lab["variable"]] if (lab["variable"] != "probability_exit_open" or zs["exit_applicable"]) else None
        assert lab["value"] == expected


def test_out_of_range_targets_are_excluded_explicitly(built, states):
    labels = _labels(built)
    last = states[-1]["simulation_second"]
    assert max(l["target_second"] for l in labels) == last
    excluded = json.loads(built.reference_files["manifest.json"])["label_rule"]["excluded_issue_instants_by_horizon"]
    assert excluded == {"30": 6, "60": 12, "120": len(states)}   # 60 s run: no 120 s target exists
    assert not [l for l in labels if l["horizon_seconds"] == 120]


def test_missing_values_and_exit_applicability_in_labels(built):
    labels = _labels(built)
    assert all(l["value_valid"] == (l["value"] is not None) for l in labels)
    exit_labels = [l for l in labels if l["variable"] == "probability_exit_open"]
    assert all(l["applicable"] == l["zone_id"].startswith("T_GATE_") for l in exit_labels)
    assert all(l["value"] is None for l in exit_labels if not l["applicable"])
    assert any(l["variable"] == "speed" and l["value"] is None and l["applicable"] for l in labels)


def test_label_availability_follows_the_approved_rule(built):
    epoch = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)
    for lab in _labels(built):
        avail = (datetime.fromisoformat(lab["label_available_at"].replace("Z", "+00:00")) - epoch).total_seconds()
        assert avail == lab["target_second"] + 60 > lab["issue_second"]


def test_horizons_must_align_with_the_reference_grid(states):
    with pytest.raises(ValueError):
        future_labels(states, run_id="x", scenario_id="x", epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
                      sample_seconds=5, horizons_seconds=(32,))


# ---------------------------------------------------------------- publication

def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_publish_is_complete_verified_and_idempotent(tmp_path):
    res = publish_run(SPEC, tmp_path)
    assert res.status == "published" and is_published(tmp_path, res.run_id)
    assert list_published(tmp_path) == [res.run_id]
    assert not list((tmp_path / ".staging").iterdir())
    before = _tree(tmp_path)
    again = publish_run(SPEC, tmp_path)
    assert again.status == "already_published" and _tree(tmp_path) == before


def test_different_seeds_and_scenarios_coexist(tmp_path):
    a, b = publish_run(SPEC, tmp_path), publish_run(SPEC_B, tmp_path)
    assert a.run_id != b.run_id and list_published(tmp_path) == sorted([a.run_id, b.run_id])
    verify_published(tmp_path, a.run_id)
    verify_published(tmp_path, b.run_id)


def test_tampered_artifact_is_detected_and_never_overwritten(tmp_path):
    res = publish_run(SPEC, tmp_path)
    target = tmp_path / "observations" / res.run_id / "observations.jsonl"
    target.write_bytes(target.read_bytes() + b"{}\n")
    tampered = target.read_bytes()
    assert not is_published(tmp_path, res.run_id)
    with pytest.raises(BundleIntegrityError):
        publish_run(SPEC, tmp_path)
    assert target.read_bytes() == tampered          # the conflicting content was not overwritten


def test_invalid_data_prevents_publication(tmp_path, monkeypatch):
    real = B.generate_observations

    def bad(*a, **kw):
        reports = real(*a, **kw)
        reports[0] = {**reports[0], "received_at": reports[0]["observed_at"]}   # receipt before publication
        return reports
    monkeypatch.setattr(B, "generate_observations", bad)
    with pytest.raises(BundleValidationError):
        publish_run(SPEC, tmp_path)
    assert list_published(tmp_path) == []
    assert not (tmp_path / "observations").exists() and not (tmp_path / "reference").exists()


def test_interrupted_publication_is_not_complete_and_recovers(tmp_path, monkeypatch):
    real_replace = os.replace

    def crash(src, dst):
        if str(dst).endswith(".json") and "published" in str(dst):
            raise OSError("simulated crash before the completion marker")
        return real_replace(src, dst)
    monkeypatch.setattr(B.os, "replace", crash)
    with pytest.raises(OSError):
        publish_run(SPEC, tmp_path)
    run_id = build_run(SPEC).run_id
    assert (tmp_path / "observations" / run_id).is_dir() and (tmp_path / "reference" / run_id).is_dir()
    assert not is_published(tmp_path, run_id) and list_published(tmp_path) == []
    assert not list((tmp_path / ".staging").iterdir())
    monkeypatch.setattr(B.os, "replace", real_replace)
    res = publish_run(SPEC, tmp_path)
    assert res.status == "recovered" and is_published(tmp_path, run_id)


def test_conflicting_unmarked_bundle_is_never_replaced(tmp_path):
    run_id = build_run(SPEC).run_id
    squatter = tmp_path / "observations" / run_id
    squatter.mkdir(parents=True)
    (squatter / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleIntegrityError):
        publish_run(SPEC, tmp_path)
    assert (squatter / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert list_published(tmp_path) == []


def test_nondeterministic_generation_is_rejected(tmp_path, monkeypatch):
    real = B._jsonl
    counter = {"n": 0}

    def drifting(records):
        counter["n"] += 1
        return real(records) + (b"\n" if counter["n"] > 5 else b"")
    monkeypatch.setattr(B, "_jsonl", drifting)
    with pytest.raises(BundleIntegrityError, match="nondeterminism"):
        publish_run(SPEC, tmp_path)
    assert list_published(tmp_path) == []


def test_validation_reports_the_checks_it_ran(built):
    checks = validate_outputs(built)
    assert len(checks) >= 30 and "label availability" in checks and "observation record allowlist" in checks
