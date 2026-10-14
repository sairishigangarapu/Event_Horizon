import numpy as np
import torch
from crowd_reliability.ml.graph import GraphHyperNetwork, SpatialGraphBuilder
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.simulation.scenarios import get_default_venue


def _build():
    venue = get_default_venue()
    builder = SpatialGraphBuilder(venue.zones, venue.connections)
    hn = GraphHyperNetwork(
        dyn_dim=24,
        static_node_dim=builder.node_static.size(1),
        static_edge_dim=builder.edge_attr.size(1),
        node_out_dim=16,
        edge_out_dim=4,
    )
    return venue, builder, hn


def test_hypernet_output_shapes_and_edge_positivity():
    venue, builder, hn = _build()
    dyn = torch.randn(builder.num_nodes, 24)
    node_vals, edge_vals = hn(dyn, builder.node_static, builder.edge_index, builder.edge_attr)

    assert node_vals.shape == (builder.num_nodes, 16)
    assert edge_vals.shape == (builder.edge_index.size(1), 4)
    assert torch.all(edge_vals >= 0.0), "Generated edge values must be non-negative"


def test_hypernet_values_depend_on_dynamic_context():
    """Generated values must change when the observed state (incl. tree vector) changes."""
    venue, builder, hn = _build()
    dyn_a = torch.zeros(builder.num_nodes, 24)
    dyn_b = torch.zeros(builder.num_nodes, 24)
    dyn_b[:, 18:24] = 3.0  # perturb the six decision-tree channels only

    nv_a, ev_a = hn(dyn_a, builder.node_static, builder.edge_index, builder.edge_attr)
    nv_b, ev_b = hn(dyn_b, builder.node_static, builder.edge_index, builder.edge_attr)

    assert not torch.allclose(nv_a, nv_b)
    assert not torch.allclose(ev_a, ev_b)


def test_pipeline_uses_hypernet_generated_values():
    venue = get_default_venue()
    pipeline = ArchitecturePipeline(venue=venue)
    assert pipeline.gnn.layer1.self_linear.in_features == pipeline.hypernet.node_out_dim
    # Temporal input contract unchanged: X(18) + D(6) + H(16)
    assert pipeline.u_dim == 40
