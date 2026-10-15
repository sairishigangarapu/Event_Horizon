import os
import shutil
import tempfile
from datetime import datetime, timezone
import pytest

from crowd_reliability.simulation.scenarios import get_default_venue, SCENARIO_PRESETS
from crowd_reliability.simulation.simulator import CrowdSimulator
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.ml.pipeline import ArchitecturePipeline, load_verified_tree_bank
from crowd_reliability.ml.tree_bank import TREE_CHANNELS

def test_pipeline_step_and_serialization():
    venue = get_default_venue()
    pipeline = ArchitecturePipeline(venue=venue, horizons_sec=[30, 60, 120])

    sim = CrowdSimulator(venue=venue, seed=42)
    sc = SCENARIO_PRESETS["crowd_surge"]
    refs, _ = sim.run_simulation(sc)

    gen = ObservationGenerator(seed=42)
    obs = gen.generate_observations(refs, sc)

    t_eval = obs[50].received_at
    tree_records, forecast_records = pipeline.step(sc.scenario_id, t_eval, obs)

    # 1. Check tree output records
    assert len(tree_records) == len(venue.zones)
    for tr in tree_records:
        assert len(tr.channel_names) == 6
        assert len(tr.channel_values) == 6
        assert len(tr.channel_masks) == 6
        assert tr.channel_names == TREE_CHANNELS

    # 2. Check forecast records
    assert len(forecast_records) > 0
    for fr in forecast_records:
        assert fr.stream_id == "stream_b_architecture"
        assert fr.target_at > fr.issued_at
        assert fr.variable in ["density", "speed", "inflow", "outflow", "queue_length", "exit_open"]

    # 3. Test serialization. Only a fitted tree bank may be saved, so use the production bank.
    pipeline.tree_bank = load_verified_tree_bank(os.path.join("models", "production"))
    tmp_dir = tempfile.mkdtemp()
    try:
        pipeline.save_pipeline(tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))
        assert os.path.exists(os.path.join(tmp_dir, "tree_bank.joblib"))
        assert os.path.exists(os.path.join(tmp_dir, "temporal_model.pt"))
        spatial_dir = os.path.join(tmp_dir, pipeline.manifest.spatial_artifact_dir)
        assert os.path.exists(os.path.join(spatial_dir, "hypernet.pt"))
        assert os.path.exists(os.path.join(spatial_dir, "gnn.pt"))

        new_pipeline = ArchitecturePipeline(venue=venue)
        new_pipeline.load_pipeline(tmp_dir)
        assert new_pipeline.manifest.pipeline_version == pipeline.manifest.pipeline_version

        # 4. Test integrity verification: tampering with a model file raises ValueError
        with open(os.path.join(spatial_dir, "hypernet.pt"), "ab") as f:
            f.write(b"corrupted_tampered_bytes")

        corrupt_pipeline = ArchitecturePipeline(venue=venue)
        with pytest.raises(ValueError, match="Model integrity verification failed"):
            corrupt_pipeline.load_pipeline(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir)



def _forecast_values(pipeline, scenario_id, t_eval, obs):
    pipeline.history_buffer.clear()  # compare single calls, not rolling state
    _, forecasts = pipeline.step(scenario_id, t_eval, obs)
    return [(f.zone_id, f.variable, f.target_at, f.value) for f in forecasts]


def test_step_is_deterministic_for_loaded_production_pipeline():
    # Regression: the GRU stayed in train mode, so its inter-layer dropout made
    # identical inputs produce different Stream B forecasts.
    venue = get_default_venue()
    sc = SCENARIO_PRESETS["crowd_surge"]
    refs, _ = CrowdSimulator(venue=venue, seed=42).run_simulation(sc)
    obs = ObservationGenerator(seed=42).generate_observations(refs, sc)
    t_eval = sorted(o.received_at for o in obs)[len(obs) // 2]

    pipeline = ArchitecturePipeline(venue=venue)
    pipeline.load_pipeline(os.path.join("models", "production"))
    assert not pipeline.temporal_model.training

    first = _forecast_values(pipeline, sc.scenario_id, t_eval, obs)
    assert first
    assert _forecast_values(pipeline, sc.scenario_id, t_eval, obs) == first

    # Even if a caller flips the model to train(), inference must stay deterministic.
    pipeline.temporal_model.train()
    assert _forecast_values(pipeline, sc.scenario_id, t_eval, obs) == first


def test_temporal_checkpoint_loads_in_eval_mode(tmp_path):
    from crowd_reliability.ml.temporal import MultivariateTemporalForecaster
    model = MultivariateTemporalForecaster(input_dim=40)
    path = str(tmp_path / "temporal.pt")
    model.save(path)
    assert MultivariateTemporalForecaster.load(path).training is False
    assert ArchitecturePipeline().temporal_model.training is False
