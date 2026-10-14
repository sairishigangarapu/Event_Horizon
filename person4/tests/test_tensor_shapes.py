"""
test_tensor_shapes.py – Validates output tensor shapes for model, sequence, and inference.

Checks:
  - U_DIM = 39 for assembled U vectors.
  - WindowBatch u_history shape: (num_zones, HISTORY_STEPS, U_DIM).
  - Model raw output shape: (num_zones, num_horizons, 6).
  - ForecastRecord list length matches (zones × horizons × applicable variables).
"""

from __future__ import annotations

import pytest
import numpy as np
import torch
from datetime import datetime, timezone

from person4.event_horizon_p4.config import (
    HISTORY_STEPS,
    HORIZONS_SEC,
    NUM_MOCK_ZONES,
    U_DIM,
    VAR_NAMES,
    NUM_CHANNELS,
)
from person4.event_horizon_p4.model import TemporalGRUForecaster
from person4.event_horizon_p4.sequence import (
    SequenceWindowAssembler,
    assemble_u_vec,
)
from person4.event_horizon_p4.synthetic import (
    MOCK_ZONE_IDS,
    EXIT_ZONE_IDS,
    generate_scenario,
    get_tree_output,
    get_h_embedding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scenario():
    return generate_scenario(
        scenario_id="test_shape_sc",
        split="test",
        num_steps=HISTORY_STEPS + 5,
        seed=42,
    )


@pytest.fixture(scope="module")
def assembler():
    return SequenceWindowAssembler()


@pytest.fixture(scope="module")
def first_batch(scenario, assembler):
    assembler.reset()
    from person4.event_horizon_p4.synthetic import generate_scenario
    sc = generate_scenario(
        scenario_id="test_shape_sc_fresh",
        split="test",
        num_steps=HISTORY_STEPS + 5,
        seed=42,
    )
    step_times = sorted({t.evaluation_time for t in sc.tree_outputs})
    return assembler.step(sc, step_times[0], 0), sc


# ---------------------------------------------------------------------------
# U vector dimensionality
# ---------------------------------------------------------------------------

class TestUVectorShape:
    def test_u_dim_constant_matches_39(self):
        assert U_DIM == 39

    def test_assembled_u_vec_has_correct_dim(self, scenario):
        zone_id = MOCK_ZONE_IDS[0]
        step_times = sorted({t.evaluation_time for t in scenario.tree_outputs})
        issue = step_times[HISTORY_STEPS]  # use a step with full history

        tree_out  = get_tree_output(scenario, zone_id, issue)
        h_emb     = get_h_embedding(scenario, zone_id, HISTORY_STEPS)
        avail_obs = [o for o in scenario.observations
                     if o.zone_id == zone_id and o.received_at <= issue]

        u_vec = assemble_u_vec(zone_id, issue, avail_obs, tree_out, h_emb)
        assert u_vec.shape == (U_DIM,), f"Expected ({U_DIM},), got {u_vec.shape}"

    def test_u_vec_dtype_float32(self, scenario):
        zone_id   = MOCK_ZONE_IDS[0]
        step_times = sorted({t.evaluation_time for t in scenario.tree_outputs})
        issue     = step_times[0]
        tree_out  = get_tree_output(scenario, zone_id, issue)
        h_emb     = get_h_embedding(scenario, zone_id, 0)
        avail_obs = [o for o in scenario.observations
                     if o.zone_id == zone_id and o.received_at <= issue]
        u_vec = assemble_u_vec(zone_id, issue, avail_obs, tree_out, h_emb)
        assert u_vec.dtype == np.float32


# ---------------------------------------------------------------------------
# WindowBatch shapes
# ---------------------------------------------------------------------------

class TestWindowBatchShape:
    def test_u_history_shape(self, first_batch):
        batch, _ = first_batch
        assert batch.u_history.shape == (
            NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM
        ), f"Unexpected u_history shape: {batch.u_history.shape}"

    def test_zone_ids_count(self, first_batch):
        batch, _ = first_batch
        assert len(batch.zone_ids) == NUM_MOCK_ZONES

    def test_target_times_count(self, first_batch):
        batch, _ = first_batch
        assert len(batch.target_times) == len(HORIZONS_SEC)
        for h in HORIZONS_SEC:
            assert h in batch.target_times


# ---------------------------------------------------------------------------
# Model output shape
# ---------------------------------------------------------------------------

class TestModelOutputShape:
    def test_model_forward_shape(self):
        model = TemporalGRUForecaster()
        model.eval()
        dummy_input = torch.zeros(NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM)
        with torch.no_grad():
            out = model(dummy_input)
        expected = (NUM_MOCK_ZONES, len(HORIZONS_SEC), NUM_CHANNELS)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

    def test_numeric_channels_non_negative(self):
        model = TemporalGRUForecaster()
        model.eval()
        dummy_input = torch.randn(NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM)
        with torch.no_grad():
            out = model(dummy_input)
        numeric = out[..., :5].numpy()
        assert np.all(numeric >= 0), "Numeric outputs should be non-negative (ReLU)"

    def test_model_variable_names_count(self):
        assert len(VAR_NAMES) == NUM_CHANNELS == 6


# ---------------------------------------------------------------------------
# ForecastRecord count
# ---------------------------------------------------------------------------

class TestForecastRecordCount:
    def test_forecast_record_exit_omitted_for_non_exit_zones(self, scenario):
        from person4.event_horizon_p4.inference import TemporalForecaster
        from person4.event_horizon_p4.sequence import SequenceWindowAssembler

        model = TemporalGRUForecaster()
        forecaster = TemporalForecaster(model=model)

        asm   = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in scenario.tree_outputs})
        batch = asm.step(scenario, step_times[HISTORY_STEPS], HISTORY_STEPS)
        records = forecaster.forecast(batch, scenario_id=scenario.scenario_id)

        exit_zone_recs = [r for r in records
                          if r.variable == "exit_open" and r.zone_id not in EXIT_ZONE_IDS]
        assert len(exit_zone_recs) == 0, (
            "exit_open should be omitted for non-exit zones, not set to zero"
        )

    def test_forecast_record_exit_present_for_exit_zones(self, scenario):
        from person4.event_horizon_p4.inference import TemporalForecaster
        from person4.event_horizon_p4.sequence import SequenceWindowAssembler

        model = TemporalGRUForecaster()
        forecaster = TemporalForecaster(model=model)

        asm   = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in scenario.tree_outputs})
        batch = asm.step(scenario, step_times[HISTORY_STEPS], HISTORY_STEPS)
        records = forecaster.forecast(batch, scenario_id=scenario.scenario_id)

        for zone_id in EXIT_ZONE_IDS:
            exit_recs = [r for r in records
                         if r.variable == "exit_open" and r.zone_id == zone_id]
            assert len(exit_recs) == len(HORIZONS_SEC), (
                f"Expected {len(HORIZONS_SEC)} exit_open records for {zone_id}"
            )
