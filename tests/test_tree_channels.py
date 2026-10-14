"""
Acceptance tests for the named current-state tree vector D[z,t]:
- changing only one channel's targets changes only that exported channel
- all six values survive serialization; masks roundtrip
- exit applicability mask is 0 outside exit zones
- inference is deterministic
"""
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from crowd_reliability.ml.tree_bank import CHANNEL_METADATA, TREE_CHANNELS, DecisionTreeBank
from crowd_reliability.ml.trees import TreeSnapshot, TreeVectorResult, infer_tree_vector

T_BASE = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
ZONE_FLAGS = {"zone_a": False, "zone_b": True, "zone_c": False}
INFLOW_IDX = TREE_CHANNELS.index("estimated_inflow")
EXIT_IDX = TREE_CHANNELS.index("probability_exit_open")


def _make_training_data(n: int = 60):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 3, size=(n, 18)).astype(np.float32)
    Y = {}
    for i, channel in enumerate(TREE_CHANNELS):
        if channel == "probability_exit_open":
            Y[channel] = (X[:, 0] > 1.5).astype(float)
        else:
            Y[channel] = X[:, i % 6] * 2.0 + 0.1
    return X, Y


def _fitted_bank(seed: int = 42, inflow_scale: float = 1.0) -> DecisionTreeBank:
    X, Y = _make_training_data()
    if inflow_scale != 1.0:
        Y = dict(Y)
        Y["estimated_inflow"] = Y["estimated_inflow"] * inflow_scale
    bank = DecisionTreeBank(random_state=seed)
    bank.fit(X, Y)
    return bank


def _snapshot() -> TreeSnapshot:
    obs = []
    for zone in ZONE_FLAGS:
        for var in ["density", "speed", "inflow", "outflow", "queue_length", "exit_open"]:
            obs.append({
                "report_id": f"r_{zone}_{var}",
                "scenario_id": "s_test",
                "zone_id": zone,
                "source_id": "cam_1",
                "variable": var,
                "value": 1.5,
                "observed_at": T_BASE - timedelta(seconds=10),
                "received_at": T_BASE - timedelta(seconds=8),
            })
    return TreeSnapshot(
        scenario_id="s_test",
        as_of_time=T_BASE,
        observations=obs,
        zone_ids=list(ZONE_FLAGS.keys()),
        zone_exit_flags=ZONE_FLAGS,
    )


def test_changing_only_channel_3_changes_only_that_channel():
    """Acceptance: retrain with modified channel-3 targets; other channels unchanged."""
    bank_a = _fitted_bank()
    bank_b = _fitted_bank(inflow_scale=3.0)
    snapshot = _snapshot()

    res_a = infer_tree_vector(snapshot, bank_a)
    res_b = infer_tree_vector(snapshot, bank_b)

    assert not np.allclose(res_a.values[:, INFLOW_IDX], res_b.values[:, INFLOW_IDX]), \
        "Channel 3 (estimated_inflow) must change when its targets change"
    for idx in range(len(TREE_CHANNELS)):
        if idx == INFLOW_IDX:
            continue
        assert np.array_equal(res_a.values[:, idx], res_b.values[:, idx]), \
            f"Channel {TREE_CHANNELS[idx]} changed despite unchanged targets (aggregation leak)"
    assert np.array_equal(res_a.masks, res_b.masks)


def test_all_six_channels_survive_serialization_roundtrip():
    result = infer_tree_vector(_snapshot(), _fitted_bank())
    restored = TreeVectorResult.from_json(result.to_json())

    assert restored.zone_ids == result.zone_ids
    assert np.array_equal(restored.values, result.values)
    assert restored.values.shape == (3, 6)
    assert np.array_equal(restored.masks, result.masks)
    assert restored.tree_version == result.tree_version
    assert [m["name"] for m in restored.channel_metadata] == TREE_CHANNELS
    assert [m["unit"] for m in restored.channel_metadata] == [
        "p/m2", "m/s", "p/s", "p/s", "persons", "probability"
    ]


def test_exit_channel_applicability_mask():
    result = infer_tree_vector(_snapshot(), _fitted_bank())
    for i, z_id in enumerate(result.zone_ids):
        expected = 1 if ZONE_FLAGS[z_id] else 0
        assert result.masks[i, EXIT_IDX] == expected, \
            f"Exit applicability mask wrong for {z_id}"
        # Non-exit channels are always applicable
        assert result.masks[i, :EXIT_IDX].tolist() == [1] * 5


def test_inference_is_deterministic():
    bank = _fitted_bank()
    res_a = infer_tree_vector(_snapshot(), bank)
    res_b = infer_tree_vector(_snapshot(), bank)
    assert np.array_equal(res_a.values, res_b.values)
    assert np.array_equal(res_a.masks, res_b.masks)
    assert res_a.zone_ids == sorted(res_a.zone_ids)


def test_channel_metadata_complete():
    assert len(CHANNEL_METADATA) == 6
    for meta in CHANNEL_METADATA:
        assert {"name", "unit", "kind", "source_variable", "applicability"} <= set(meta)
    assert CHANNEL_METADATA[EXIT_IDX]["applicability"] == "exit_zones_only"
