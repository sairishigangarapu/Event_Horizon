import json
import os

import numpy as np
import pytest
import torch

from crowd_reliability.ml.graph import SpatialGraphBuilder, SpatialGNN, EDGE_STATIC_NAMES
from crowd_reliability.ml.spatial import (
    SPATIAL_DYN_DIM, SPATIAL_FEATURE_SCHEMA, SpatialEncoder, build_spatial_features,
    check_spatial_compat, default_normalization,
)
from crowd_reliability.ml.tree_bank import TREE_CHANNELS
from crowd_reliability.simulation.scenarios import get_default_venue

def _encoder(seed=0):
    venue = get_default_venue()
    return SpatialEncoder(SpatialGraphBuilder(venue.zones, venue.connections), seed=seed)

def _random_inputs(z, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 3.0, (z, 6)).astype(np.float32)
    values[:, 5] = 1.0
    ages = rng.uniform(0.0, 60.0, (z, 6)).astype(np.float32)
    present = np.ones((z, 6), dtype=np.float32)
    fresh = np.ones((z, 6), dtype=np.float32)
    tree_values = rng.uniform(0.0, 3.0, (z, 6)).astype(np.float32)
    tree_values[:, 5] = 0.7
    tree_masks = np.ones((z, 6), dtype=np.float32)
    return values, ages, present, fresh, tree_values, tree_masks

def _vn(enc, *arrays):
    f, v = build_spatial_features(*arrays)
    with torch.no_grad():
        _, vn, _ = enc.forward_tensors(torch.from_numpy(f), torch.from_numpy(v))
    return vn

# ---------- structure and edges ----------

def test_shapes_derived_from_data_and_nonnegative_edges():
    enc = _encoder()
    g = enc.graph
    venue = get_default_venue()
    assert g.num_edges == 2 * len(venue.connections) == len(g.edge_ids)
    f, v = build_spatial_features(*_random_inputs(g.num_nodes))
    h, vn, ve = enc.forward_tensors(torch.from_numpy(f), torch.from_numpy(v))
    assert vn.shape == (g.num_nodes, 16)
    assert ve.shape == (g.num_edges, 4)
    assert h.shape == (g.num_nodes, enc.embed_dim)
    assert torch.all(ve >= 0.0)
    assert SPATIAL_DYN_DIM == len(SPATIAL_FEATURE_SCHEMA) == 36

def test_edge_ids_align_and_direction_flags():
    zones = [{"zone_id": z, "zone_type": "corridor"} for z in ("a", "b", "c")]
    conns = [
        {"edge_id": "ab", "from_zone": "a", "to_zone": "b", "direction": "unidirectional"},
        {"edge_id": "bc", "from_zone": "b", "to_zone": "c", "direction": "bidirectional"},
    ]
    g = SpatialGraphBuilder(zones, conns)
    rev_i, trav_i = EDGE_STATIC_NAMES.index("is_reverse_copy"), EDGE_STATIC_NAMES.index("is_traversable")
    expected = {
        "ab": ("a", "b", 0.0, 1.0), "ab__rev": ("b", "a", 1.0, 0.0),
        "bc": ("b", "c", 0.0, 1.0), "bc__rev": ("c", "b", 1.0, 1.0),
    }
    assert g.edge_ids == ["ab", "ab__rev", "bc", "bc__rev"]
    for i, eid in enumerate(g.edge_ids):
        src, dst, is_rev, trav = expected[eid]
        assert g.zone_ids[g.edge_index[0, i]] == src
        assert g.zone_ids[g.edge_index[1, i]] == dst
        assert g.edge_attr[i, rev_i].item() == is_rev
        assert g.edge_attr[i, trav_i].item() == trav

def test_unknown_direction_or_zone_raises():
    zones = [{"zone_id": "a"}, {"zone_id": "b"}]
    with pytest.raises(ValueError):
        SpatialGraphBuilder(zones, [{"edge_id": "x", "from_zone": "a", "to_zone": "b", "direction": "sideways"}])
    with pytest.raises(ValueError):
        SpatialGraphBuilder(zones, [{"edge_id": "x", "from_zone": "a", "to_zone": "zz", "direction": "unidirectional"}])

# ---------- masks (one test per mask) ----------

def test_absent_report_value_and_age_are_irrelevant():
    enc = _encoder()
    z = enc.graph.num_nodes
    values, ages, present, fresh, tv, tm = _random_inputs(z)
    present[:, 0] = 0.0
    fresh[:, 0] = 0.0
    base = _vn(enc, values, ages, present, fresh, tv, tm)
    values2, ages2 = values.copy(), ages.copy()
    values2[:, 0] = 999.0
    ages2[:, 0] = 12345.0
    assert torch.equal(base, _vn(enc, values2, ages2, present, fresh, tv, tm))

    # Same property at the hypernetwork input, before normalization: invalid raw entries never leak
    f, v = build_spatial_features(values, ages, present, fresh, tv, tm)
    f_bad = f.copy()
    f_bad[v == 0] = 1e6
    with torch.no_grad():
        _, a, _ = enc.forward_tensors(torch.from_numpy(f), torch.from_numpy(v))
        _, b, _ = enc.forward_tensors(torch.from_numpy(f_bad), torch.from_numpy(v))
    assert torch.equal(a, b)

def test_obs_fresh_alone_changes_output():
    enc = _encoder()
    arrays = list(_random_inputs(enc.graph.num_nodes))
    base = _vn(enc, *arrays)
    arrays[3] = arrays[3].copy()
    arrays[3][:, 2] = 0.0
    assert not torch.allclose(base, _vn(enc, *arrays))

def test_masked_tree_channel_value_is_irrelevant():
    enc = _encoder()
    values, ages, present, fresh, tv, tm = _random_inputs(enc.graph.num_nodes)
    tm[:, 5] = 0.0
    base = _vn(enc, values, ages, present, fresh, tv, tm)
    tv2 = tv.copy()
    tv2[:, 5] = 0.0
    assert torch.equal(base, _vn(enc, values, ages, present, fresh, tv2, tm))

def test_tree_mask_itself_changes_output():
    enc = _encoder()
    values, ages, present, fresh, tv, tm = _random_inputs(enc.graph.num_nodes)
    tv[:, 5] = 0.0  # same value either way, only the mask differs
    base = _vn(enc, values, ages, present, fresh, tv, tm)
    tm2 = tm.copy()
    tm2[:, 5] = 0.0
    assert not torch.allclose(base, _vn(enc, values, ages, present, fresh, tv, tm2))

def test_observed_zero_differs_from_missing():
    values, ages, present, fresh, tv, tm = _random_inputs(2)
    values[:, 4] = 0.0
    observed, _ = build_spatial_features(values, ages, present, fresh, tv, tm)
    present2, fresh2 = present.copy(), fresh.copy()
    present2[:, 4] = 0.0
    fresh2[:, 4] = 0.0
    missing, _ = build_spatial_features(values, ages, present2, fresh2, tv, tm)
    assert not np.array_equal(observed, missing)

def test_normalization_leaves_binary_and_probability_features_unchanged():
    enc = _encoder()
    f, v = build_spatial_features(*_random_inputs(enc.graph.num_nodes))
    x = enc.hypernet.normalize_inputs(torch.from_numpy(f), torch.from_numpy(v))
    for i, spec in enumerate(SPATIAL_FEATURE_SCHEMA):
        if spec.kind != "continuous":
            assert torch.equal(x[:, i], torch.from_numpy(f)[:, i]), spec.name
    norm = default_normalization()
    i = [s.name for s in SPATIAL_FEATURE_SCHEMA].index("obs_value_queue_length")
    assert torch.allclose(x[:, i], (torch.from_numpy(f)[:, i] - norm["mean"][i]) / norm["std"][i])

# ---------- sensitivity and gradients ----------

@pytest.mark.parametrize("channel", range(len(TREE_CHANNELS)))
def test_each_tree_channel_individually_changes_node_values(channel):
    enc = _encoder()
    values, ages, present, fresh, tv, tm = _random_inputs(enc.graph.num_nodes)
    base = _vn(enc, values, ages, present, fresh, tv, tm)
    tv2 = tv.copy()
    tv2[:, channel] += 0.5 if channel != 5 else -0.5
    assert not torch.allclose(base, _vn(enc, values, ages, present, fresh, tv2, tm))

def test_gradients_reach_every_parameter():
    enc = _encoder()
    f, v = build_spatial_features(*_random_inputs(enc.graph.num_nodes))
    h, vn, ve = enc.forward_tensors(torch.from_numpy(f), torch.from_numpy(v))
    (h.sum() + vn.sum() + ve.sum()).backward()
    for module in (enc.hypernet, enc.gnn):
        for name, p in module.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, name

# ---------- graph effects ----------

def _h_with_edges(enc, f, v, keep):
    ei, ea = enc.graph.edge_index[:, keep], enc.graph.edge_attr[keep]
    with torch.no_grad():
        h, _, _ = enc.forward_tensors(f, v, ei, ea)
    return h

@pytest.mark.parametrize("removed", ["e_conc_exit1", "e_conc_exit1__rev", "e_ent_a_corr1"])
def test_edge_removal_affects_only_two_hop_reachable_nodes(removed):
    enc = _encoder()
    g = enc.graph
    f, v = (torch.from_numpy(a) for a in build_spatial_features(*_random_inputs(g.num_nodes)))
    all_edges = torch.ones(g.num_edges, dtype=torch.bool)
    keep = all_edges.clone()
    r = g.edge_ids.index(removed)
    keep[r] = False

    h_full = _h_with_edges(enc, f, v, all_edges)
    h_cut = _h_with_edges(enc, f, v, keep)

    dst = g.edge_index[1, r].item()
    remaining = g.edge_index[:, keep]
    affected = {dst} | {remaining[1, i].item() for i in range(remaining.size(1)) if remaining[0, i].item() == dst}
    changed = (h_full - h_cut).abs().max(dim=1).values > 1e-6
    assert changed[dst]
    for n in range(g.num_nodes):
        if n not in affected:
            assert not changed[n], f"{g.zone_ids[n]} changed but is not 2-hop reachable from {g.zone_ids[dst]}"

def test_reverse_copies_let_concourse_see_exit_state():
    """Message propagation only: the reverse edge is not physically traversable."""
    enc = _encoder()
    g = enc.graph
    values, ages, present, fresh, tv, tm = _random_inputs(g.num_nodes)
    exit_i, conc_i = g.zone_ids.index("exit_1"), g.zone_ids.index("concourse")
    f1, v1 = build_spatial_features(values, ages, present, fresh, tv, tm)
    values2 = values.copy()
    values2[exit_i, 4] = 80.0  # queue builds at the exit
    f2, v2 = build_spatial_features(values2, ages, present, fresh, tv, tm)
    f1, v1, f2, v2 = (torch.from_numpy(a) for a in (f1, v1, f2, v2))

    with_rev = torch.ones(g.num_edges, dtype=torch.bool)
    forward_only = torch.tensor([not e.endswith("__rev") for e in g.edge_ids])
    delta_rev = (_h_with_edges(enc, f1, v1, with_rev) - _h_with_edges(enc, f2, v2, with_rev))[conc_i].abs().max()
    delta_fwd = (_h_with_edges(enc, f1, v1, forward_only) - _h_with_edges(enc, f2, v2, forward_only))[conc_i].abs().max()
    assert delta_rev > 1e-6
    assert delta_fwd <= 1e-6

def test_empty_edge_set_equals_message_free_full_path():
    gnn = SpatialGNN(seed=3)
    x = torch.randn(5, 16)
    empty = gnn(x, torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 4)))
    with torch.no_grad():
        h1 = gnn.layer1.relu(gnn.layer1.out_norm(gnn.layer1.self_linear(x)))
        h2 = gnn.layer2.relu(gnn.layer2.out_norm(gnn.layer2.self_linear(h1)))
    assert torch.allclose(empty, h2)

# ---------- checkpoints ----------

def test_checkpoint_roundtrip_is_identical(tmp_path):
    enc = _encoder(seed=5)
    f, v = (torch.from_numpy(a) for a in build_spatial_features(*_random_inputs(enc.graph.num_nodes)))
    enc.save(str(tmp_path))
    other = _encoder(seed=99)
    other.load(str(tmp_path))
    enc.hypernet.eval(); other.hypernet.eval()
    with torch.no_grad():
        a = enc.forward_tensors(f, v)
        b = other.forward_tensors(f, v)
    for x, y in zip(a, b):
        assert torch.equal(x, y)

def test_checkpoint_rejects_reordered_zones_and_mismatched_pair(tmp_path):
    enc = _encoder()
    enc.save(str(tmp_path / "a"))

    venue = get_default_venue()
    reordered = SpatialEncoder(SpatialGraphBuilder(list(reversed(venue.zones)), venue.connections), seed=0)
    with pytest.raises(ValueError, match="zone_ids"):
        reordered.load(str(tmp_path / "a"))

    other = _encoder(seed=1)
    other.save(str(tmp_path / "b"))
    os.replace(tmp_path / "b" / "gnn.pt", tmp_path / "a" / "gnn.pt")
    with pytest.raises(ValueError, match="same pair"):
        _encoder().load(str(tmp_path / "a"))

def test_compat_contract_names_schema_and_scaler():
    enc = _encoder()
    meta = dict(enc.hypernet.metadata)
    check_spatial_compat(meta, enc.graph.metadata())
    meta["schema_hash"] = "x"
    meta["scaler_id"] = None
    with pytest.raises(ValueError, match="schema hash.*scaler"):
        check_spatial_compat(meta, enc.graph.metadata())

def test_same_seed_same_init():
    a, b, c = _encoder(seed=7), _encoder(seed=7), _encoder(seed=8)
    for (n, p), (_, q) in zip(a.hypernet.named_parameters(), b.hypernet.named_parameters()):
        assert torch.equal(p, q), n
    assert not torch.equal(a.hypernet.node_head.weight, c.hypernet.node_head.weight)
