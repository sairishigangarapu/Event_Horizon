import numpy as np
import pytest
from crowd_reliability.ml.tree_bank import DecisionTreeBank, TREE_CHANNELS
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.simulation.scenarios import get_default_venue

def test_tree_bank_preserves_all_named_channels():
    """
    Validates non-negotiable representation rule (Section 2 of ARCHITECTURE.md):
    Decision tree bank must emit all 6 separate named channels and must NOT collapse them.
    """
    bank = DecisionTreeBank()
    dummy_features = np.array([1.5, 0.9, 2.0, 1.8, 5.0, 1.0] + [10.0]*6 + [1.0]*6)

    # Test exit zone
    vals, masks = bank.predict_zone(dummy_features, has_exit=True)
    assert len(vals) == 6, f"Expected 6 channels, got {len(vals)}"
    assert len(masks) == 6, f"Expected 6 masks, got {len(masks)}"
    assert masks[5] == 1, "Exit channel mask should be 1 for exit zones"

    # Test non-exit zone
    vals_non_exit, masks_non_exit = bank.predict_zone(dummy_features, has_exit=False)
    assert len(vals_non_exit) == 6
    assert masks_non_exit[5] == 0, "Exit channel mask should be 0 for non-exit zones"

    # Confirm all channel names match specification
    expected_names = [
        "estimated_density",
        "estimated_speed",
        "estimated_inflow",
        "estimated_outflow",
        "estimated_queue_length",
        "probability_exit_open"
    ]
    assert TREE_CHANNELS == expected_names

def test_pipeline_temporal_input_channel_preservation():
    """
    Validates that tree vector D[z,t] enters the temporal forecaster as distinct channels.
    """
    venue = get_default_venue()
    pipeline = ArchitecturePipeline(venue=venue)

    # Node input dim should be raw(6) + ages(6) + masks(6) + tree_vals(6) = 24
    assert pipeline.node_in_dim == 24

    # Temporal forecaster input dim U should be node_feat(24) + spatial_embed(16) = 40
    assert pipeline.u_dim == 40
