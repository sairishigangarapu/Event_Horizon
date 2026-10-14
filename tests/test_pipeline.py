import os
import shutil
import tempfile
from datetime import datetime, timezone
import pytest

from crowd_reliability.simulation.scenarios import get_default_venue, SCENARIO_PRESETS
from crowd_reliability.simulation.simulator import CrowdSimulator
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.ml.pipeline import ArchitecturePipeline
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

    # 3. Test serialization
    tmp_dir = tempfile.mkdtemp()
    try:
        pipeline.save_pipeline(tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))
        assert os.path.exists(os.path.join(tmp_dir, "tree_bank.joblib"))
        assert os.path.exists(os.path.join(tmp_dir, "hypernet.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "gnn.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "temporal_model.pt"))

        new_pipeline = ArchitecturePipeline(venue=venue)
        new_pipeline.load_pipeline(tmp_dir)
        assert new_pipeline.manifest.pipeline_version == pipeline.manifest.pipeline_version

        # 4. Test integrity verification: tampering with a model file raises ValueError
        with open(os.path.join(tmp_dir, "hypernet.pt"), "ab") as f:
            f.write(b"corrupted_tampered_bytes")

        corrupt_pipeline = ArchitecturePipeline(venue=venue)
        with pytest.raises(ValueError, match="Model integrity verification failed"):
            corrupt_pipeline.load_pipeline(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir)

