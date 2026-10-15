import copy
import inspect
import os
from datetime import timedelta

import numpy as np
import pytest

from crowd_reliability.config import settings
from crowd_reliability.ml.pipeline import ArchitecturePipeline, _calc_sha256
from crowd_reliability.ml.spatial import SpatialEncoder
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES
from crowd_reliability.models.manifest import PipelineManifest
from crowd_reliability.models.schema import ObservationRecord
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import SCENARIO_PRESETS, get_default_venue
from crowd_reliability.simulation.simulator import CrowdSimulator

@pytest.fixture(scope="module")
def scenario_data():
    venue = get_default_venue()
    sc = SCENARIO_PRESETS["bottleneck_exit_closure"]
    refs, _ = CrowdSimulator(venue=venue, seed=42).run_simulation(sc)
    obs = ObservationGenerator(seed=42).generate_observations(refs, sc)
    as_of = sorted(o.received_at for o in obs)[len(obs) // 2]
    return venue, sc, refs, obs, as_of

def _step_snapshot(pipeline, sc, as_of, obs):
    pipeline.history_buffer.clear()  # no rolling state shared between compared calls
    trees, forecasts = pipeline.step(sc.scenario_id, as_of, obs)
    return (
        [t.channel_values for t in trees],
        [(f.zone_id, f.variable, f.target_at, f.value) for f in forecasts],
        pipeline.last_generated["node_values"].copy(),
        pipeline.last_generated["edge_values"].copy(),
    )

def _assert_same(a, b):
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert np.array_equal(a[2], b[2])
    assert np.array_equal(a[3], b[3])

def test_future_received_observation_changes_nothing(scenario_data):
    venue, sc, _, obs, as_of = scenario_data
    pipeline = ArchitecturePipeline(venue=venue)
    base = _step_snapshot(pipeline, sc, as_of, obs)

    future = []
    for z in venue.zones:
        for var in TARGET_VAR_NAMES:
            future.append(ObservationRecord(
                report_id=f"future_{z['zone_id']}_{var}", scenario_id=sc.scenario_id, zone_id=z["zone_id"],
                source_id="future_sensor", variable=var, value=777.0, unit="x",
                observed_at=as_of,  # observed in time, but received after the cutoff
                published_at=as_of,
                received_at=as_of + timedelta(seconds=1),
            ))
    _assert_same(base, _step_snapshot(pipeline, sc, as_of, obs + future))

    # Spatial input tensors themselves are unchanged
    for z in venue.zones:
        a = pipeline.extract_zone_features_detailed(z["zone_id"], as_of, obs)
        b = pipeline.extract_zone_features_detailed(z["zone_id"], as_of, obs + future)
        for key in a:
            assert np.array_equal(a[key], b[key])

def test_reference_archive_mutation_changes_nothing(scenario_data):
    venue, sc, refs, obs, as_of = scenario_data
    pipeline = ArchitecturePipeline(venue=venue)
    base = _step_snapshot(pipeline, sc, as_of, obs)
    mutated = copy.deepcopy(refs)
    for r in mutated:
        r.value = -123.0
    _assert_same(base, _step_snapshot(pipeline, sc, as_of, obs))

def test_encode_spatial_accepts_no_targets_or_references():
    params = set(inspect.signature(SpatialEncoder.encode_spatial).parameters)
    assert params == {"self", "zone_features", "tree_values", "tree_masks", "tree_version"}

def _original_extract_zone_features(zone_id, as_of_time, recent_observations):
    """Verbatim copy of the pre-change ArchitecturePipeline.extract_zone_features body."""
    var_indices = {v: i for i, v in enumerate(TARGET_VAR_NAMES)}
    raw_values = np.zeros(6, dtype=np.float32)
    ages_sec = np.full(6, 999.0, dtype=np.float32)
    obs_masks = np.zeros(6, dtype=np.float32)
    zone_obs = [o for o in recent_observations if o.zone_id == zone_id and o.received_at <= as_of_time]
    for obs in zone_obs:
        if obs.variable in var_indices:
            idx = var_indices[obs.variable]
            age = max(0.0, (as_of_time - obs.observed_at).total_seconds())
            if age < ages_sec[idx]:
                ages_sec[idx] = age
                raw_values[idx] = obs.value
                obs_masks[idx] = 1.0 if age <= 90.0 else 0.0
    if obs_masks[0] == 0: raw_values[0] = 0.8
    if obs_masks[1] == 0: raw_values[1] = 1.2
    if obs_masks[2] == 0: raw_values[2] = 1.0
    if obs_masks[3] == 0: raw_values[3] = 1.0
    if obs_masks[4] == 0: raw_values[4] = 0.0
    if obs_masks[5] == 0: raw_values[5] = 1.0
    norm_ages = np.clip(ages_sec / 120.0, 0.0, 1.0)
    return raw_values, norm_ages, obs_masks

def test_tree_and_u_features_are_byte_identical_to_before(scenario_data):
    venue, _, _, obs, _ = scenario_data
    pipeline = ArchitecturePipeline(venue=venue)
    times = sorted({o.received_at for o in obs})
    for t in [times[0] - timedelta(seconds=1), times[5], times[len(times) // 2], times[-1] + timedelta(seconds=200)]:
        for z in venue.zones:
            new = pipeline.extract_zone_features(z["zone_id"], t, obs)
            old = _original_extract_zone_features(z["zone_id"], t, obs)
            for a, b in zip(new, old):
                assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert pipeline.u_dim == 40

# ---------- smoke experiment data rules ----------

def test_experiment_inputs_never_receive_references_and_exclude_stale(scenario_data):
    from crowd_reliability.ml import spatial_train
    venue, sc, refs, obs, _ = scenario_data
    assert set(inspect.signature(spatial_train.build_inputs).parameters) == {"pipeline", "observations", "times"}

    kept = spatial_train.filter_reports(obs, include_stale=False)
    assert any(o.report_id.startswith("rep_stale_") for o in obs)
    assert not any(o.report_id.startswith("rep_stale_") for o in kept)
    assert len(spatial_train.filter_reports(obs, include_stale=True)) == len(obs)

    # Inputs depend on observations only: mutating references afterwards changes targets, not inputs
    pipeline = ArchitecturePipeline(venue=venue)
    times = sorted({r.state_at for r in refs})[:6]
    a = spatial_train.build_inputs(pipeline, kept, times)
    mutated = copy.deepcopy(refs)
    for r in mutated:
        r.value = r.value + 50.0
    b = spatial_train.build_inputs(pipeline, kept, times)
    for key in a:
        assert np.array_equal(a[key], b[key])
    grid_a, _ = spatial_train.build_reference_grid([r for r in refs if r.state_at in times], pipeline.graph_builder.zone_ids, times)
    grid_b, _ = spatial_train.build_reference_grid([r for r in mutated if r.state_at in times], pipeline.graph_builder.zone_ids, times)
    assert not np.array_equal(grid_a, grid_b)

def test_smoke_trained_pair_cannot_become_active(tmp_path):
    from crowd_reliability.ml.spatial import write_spatial_artifact
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    pipeline.spatial.stamp(training_status="smoke_trained_auxiliary")
    with pytest.raises(ValueError, match="experimental"):
        write_spatial_artifact(pipeline.spatial, str(tmp_path))

# ---------- spatial-init: atomic pair switch ----------

@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    from crowd_reliability.ml.pipeline import load_verified_tree_bank
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    # save_pipeline refuses an unfitted tree bank; reuse the fitted production bank.
    pipeline.tree_bank = load_verified_tree_bank(os.path.join("models", "production"))
    pipeline.save_pipeline(str(tmp_path))
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    return tmp_path

def _snapshot_dir(path):
    return {
        os.path.relpath(os.path.join(root, f), path): _calc_sha256(os.path.join(root, f))
        for root, _, files in os.walk(path) for f in files
    }

def test_spatial_init_dry_run_writes_nothing(model_dir):
    from crowd_reliability.cli import cmd_spatial_init
    before = _snapshot_dir(model_dir)
    cmd_spatial_init(dry_run=True)
    assert _snapshot_dir(model_dir) == before

def test_spatial_init_switches_pair_and_keeps_other_artifacts(model_dir):
    from crowd_reliability.cli import cmd_spatial_init
    old = PipelineManifest.load(os.path.join(model_dir, "manifest.json"))
    cmd_spatial_init(seed=11)
    new = PipelineManifest.load(os.path.join(model_dir, "manifest.json"))
    assert new.spatial_artifact_dir != old.spatial_artifact_dir
    assert new.training_status["hypernet"] == new.training_status["gnn"] == "untrained_initial"
    for fname in ("tree_bank.joblib", "temporal_model.pt"):
        assert new.model_hashes[fname] == old.model_hashes[fname]
    assert os.path.isdir(os.path.join(model_dir, old.spatial_artifact_dir))  # old pair kept for rollback
    ArchitecturePipeline(venue=get_default_venue()).load_pipeline(str(model_dir))

def test_spatial_init_backup_is_byte_exact_and_preserves_existing(model_dir, monkeypatch):
    from crowd_reliability.cli import cmd_spatial_init
    manifest_path = os.path.join(model_dir, "manifest.json")
    backup_path = os.path.join(model_dir, "manifest.prev.json")
    with open(backup_path, "w") as f:
        f.write("older recovery point")
    original = _calc_sha256(manifest_path)

    cmd_spatial_init(seed=21)
    assert _calc_sha256(backup_path) == original
    preserved = [f for f in os.listdir(model_dir) if f.startswith("manifest.prev.") and f != "manifest.prev.json"]
    assert len(preserved) == 1
    with open(os.path.join(model_dir, preserved[0])) as f:
        assert f.read() == "older recovery point"

    # A failed post-switch load restores the active manifest byte-for-byte
    switched = _calc_sha256(manifest_path)
    def failing_load(self, directory):
        raise RuntimeError("simulated post-switch load failure")
    monkeypatch.setattr(ArchitecturePipeline, "load_pipeline", failing_load)
    with pytest.raises(RuntimeError):
        cmd_spatial_init(seed=22)
    assert _calc_sha256(manifest_path) == switched

def test_failed_validation_never_changes_active_pair(model_dir, monkeypatch):
    from crowd_reliability.cli import cmd_spatial_init
    before = _snapshot_dir(model_dir)

    def broken_load(self, directory):
        raise RuntimeError("simulated crash while validating staged pair")
    monkeypatch.setattr(SpatialEncoder, "load", broken_load)
    with pytest.raises(RuntimeError):
        cmd_spatial_init(seed=12)
    assert _snapshot_dir(model_dir) == before  # no new dir, no staging leftovers, manifest unchanged

def test_legacy_root_checkpoint_gives_clear_error(model_dir):
    manifest_path = os.path.join(model_dir, "manifest.json")
    m = PipelineManifest.load(manifest_path)
    m.spatial_artifact_dir = None
    m.save(manifest_path)
    with open(os.path.join(model_dir, "hypernet.pt"), "wb") as f:
        f.write(b"legacy")
    with pytest.raises(ValueError, match="Legacy spatial checkpoint"):
        ArchitecturePipeline(venue=get_default_venue()).load_pipeline(str(model_dir))
