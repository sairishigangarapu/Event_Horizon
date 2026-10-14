"""
Contract tests for the spatial inference deliverable (Person 3 -> Person 4):
the active checkpoint matches the architecture that loads it, H is computed from the
hypernetwork-generated Vn/Ve, all six tree channels reach the hypernetwork input, and
inference is independent of reference truth.
"""
import copy
import inspect
import json
import os
from datetime import timedelta

import numpy as np
import pytest
import torch

from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.ml.spatial import SPATIAL_FEATURE_NAMES, spatial_contract
from crowd_reliability.ml.tree_bank import TREE_CHANNELS
from crowd_reliability.models.manifest import PipelineManifest
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import SCENARIO_PRESETS, get_default_venue
from crowd_reliability.simulation.simulator import CrowdSimulator

PRODUCTION_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "production")

@pytest.fixture(scope="module")
def production_pipeline():
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    pipeline.load_pipeline(PRODUCTION_DIR)
    return pipeline

@pytest.fixture(scope="module")
def closure_data():
    venue = get_default_venue()
    sc = SCENARIO_PRESETS["bottleneck_exit_closure"]
    refs, _ = CrowdSimulator(venue=venue, seed=42).run_simulation(sc)
    obs = ObservationGenerator(seed=42).generate_observations(refs, sc)
    as_of = min(r.state_at for r in refs) + timedelta(seconds=150)
    return refs, obs, as_of

def test_active_checkpoint_matches_loaded_architecture(production_pipeline):
    p = production_pipeline
    manifest = PipelineManifest.load(os.path.join(PRODUCTION_DIR, "manifest.json"))
    spatial_dir = os.path.join(PRODUCTION_DIR, manifest.spatial_artifact_dir)
    for name, module in (("hypernet.pt", p.hypernet), ("gnn.pt", p.gnn)):
        ckpt = torch.load(os.path.join(spatial_dir, name), map_location="cpu", weights_only=True)
        assert ckpt["config"] == module.config
        assert ckpt["version"] == module.version
    assert manifest.hypernet_version == p.hypernet.version
    assert manifest.gnn_version == p.gnn.version
    assert manifest.training_status["hypernet"] == p.hypernet.metadata["training_status"]

def test_contract_dimensions_come_from_loaded_modules(production_pipeline):
    p = production_pipeline
    c = spatial_contract(p.spatial)
    assert c["h_dim"] == p.gnn.config["embed_dim"] == p.gnn.layer2.out_norm.normalized_shape[0]
    assert c["vn_dim"] == 16 and c["ve_dim"] == 4
    assert c["num_edges"] == p.graph_builder.num_edges and len(c["edge_ids"]) == c["num_edges"]
    # The temporal forecaster that is loaded alongside must be sized for this H
    assert p.temporal_model.input_dim == p.node_in_dim + c["h_dim"] == p.u_dim

def test_spatial_snapshot_shapes_and_nonnegative_edges(production_pipeline, closure_data):
    _, obs, as_of = closure_data
    out, zone_inputs = production_pipeline.spatial_snapshot(as_of, obs)
    c = spatial_contract(production_pipeline.spatial)
    assert out.H.shape == (c["num_zones"], c["h_dim"])
    assert out.Vn.shape == (c["num_zones"], c["vn_dim"])
    assert out.Ve.shape == (c["num_edges"], c["ve_dim"])
    assert np.all(out.Ve >= 0)
    assert np.isfinite(out.H).all() and np.isfinite(out.Vn).all() and np.isfinite(out.Ve).all()
    assert all(len(v) == len(TREE_CHANNELS) for v in zone_inputs["tree_values"])

def test_h_is_computed_from_generated_node_and_edge_values(production_pipeline, closure_data):
    _, obs, as_of = closure_data
    out, _ = production_pipeline.spatial_snapshot(as_of, obs)
    gnn, ei = production_pipeline.gnn, production_pipeline.graph_builder.edge_index
    with torch.no_grad():
        vn, ve = torch.from_numpy(out.Vn), torch.from_numpy(out.Ve)
        assert np.allclose(gnn(vn, ei, ve).numpy(), out.H, atol=1e-6)
        assert not np.allclose(gnn(vn, ei, torch.zeros_like(ve)).numpy(), out.H)       # Ve is consumed
        assert not np.allclose(gnn(vn * 0.5, ei, ve).numpy(), out.H)                    # Vn is consumed

@pytest.mark.parametrize("channel", range(len(TREE_CHANNELS)))
def test_each_tree_channel_reaches_hypernetwork_input(production_pipeline, channel):
    """Each of the six D channels has its own slot in the dynamic input and changes Vn."""
    name = f"tree_{TREE_CHANNELS[channel]}"
    assert name in SPATIAL_FEATURE_NAMES
    enc = production_pipeline.spatial
    z = enc.graph.num_nodes
    features = torch.zeros(z, len(SPATIAL_FEATURE_NAMES))
    valid = torch.ones_like(features)
    perturbed = features.clone()
    perturbed[:, SPATIAL_FEATURE_NAMES.index(name)] = 0.9
    with torch.no_grad():
        _, vn_a, _ = enc.forward_tensors(features, valid)
        _, vn_b, _ = enc.forward_tensors(perturbed, valid)
    assert not torch.allclose(vn_a, vn_b)

def test_spatial_inference_is_independent_of_reference_truth(production_pipeline, closure_data):
    refs, obs, as_of = closure_data
    assert set(inspect.signature(ArchitecturePipeline.spatial_snapshot).parameters) == {
        "self", "as_of_time", "recent_observations"}
    a, _ = production_pipeline.spatial_snapshot(as_of, obs)
    mutated = copy.deepcopy(refs)
    for r in mutated:
        r.value = -999.0
    b, _ = production_pipeline.spatial_snapshot(as_of, obs)
    assert np.array_equal(a.H, b.H) and np.array_equal(a.Ve, b.Ve)

def test_spatial_inference_ignores_reports_received_after_as_of(production_pipeline, closure_data):
    _, obs, as_of = closure_data
    a, _ = production_pipeline.spatial_snapshot(as_of, obs)
    late = copy.deepcopy([o for o in obs if o.received_at <= as_of][:20])
    for o in late:
        o.value = 777.0
        o.received_at = as_of + timedelta(seconds=1)
    b, _ = production_pipeline.spatial_snapshot(as_of, obs + late)
    assert np.array_equal(a.H, b.H)

def test_export_spatial_sample_writes_contract_and_evidence(tmp_path, monkeypatch):
    from crowd_reliability import cli
    monkeypatch.setattr(cli.settings, "model_dir", PRODUCTION_DIR)
    out = tmp_path / "sample.json"
    sample = cli.cmd_export_spatial_sample(out=str(out))
    on_disk = json.loads(out.read_text())
    assert on_disk["contract"]["h_dim"] == sample["contract"]["h_dim"]
    ev = on_disk["evidence"]
    assert ev["max_abs_diff_H_vs_gnn(Vn,Ve)"] < 1e-6
    assert ev["max_abs_change_H_when_Ve_zeroed"] > 0
    assert ev["min_Ve"] >= 0
    assert ev["temporal_input_dim"] == ev["temporal_input_dim_expected"]
    assert on_disk["contract"]["training_status"] == "untrained_initial"
