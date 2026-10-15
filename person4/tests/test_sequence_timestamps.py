"""
test_sequence_timestamps.py – Tests for timestamp ordering and out-of-order detection.

Validates:
  - Steps must arrive in strict monotonic order.
  - Out-of-order steps raise ValueError.
  - Only observations with received_at <= issue_time enter the model input.
  - Future observations are excluded (no data leakage).
  - Rolling window is correctly padded for early steps.
"""

from __future__ import annotations

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

from person4.event_horizon_p4.config import HISTORY_STEPS, U_DIM, NUM_MOCK_ZONES, HORIZONS_SEC
from person4.event_horizon_p4.sequence import (
    SequenceWindowAssembler,
    assemble_u_vec,
    build_all_windows,
)
from person4.event_horizon_p4.synthetic import (
    MOCK_ZONE_IDS,
    MockObservation,
    generate_scenario,
    is_exit_zone,
)
from person4.event_horizon_p4.config import VAR_UNITS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)


def _make_obs(zone_id: str, var: str, value: float,
              observed: datetime, received: datetime,
              scenario_id: str = "sc_ts_test") -> MockObservation:
    return MockObservation(
        report_id   = f"obs_{zone_id}_{var}",
        scenario_id = scenario_id,
        zone_id     = zone_id,
        source_id   = "test_sensor",
        variable    = var,
        value       = value,
        unit        = VAR_UNITS.get(var, "unknown"),
        observed_at = observed,
        published_at = received,
        received_at  = received,
    )


# ---------------------------------------------------------------------------
# Out-of-order detection
# ---------------------------------------------------------------------------

class TestOutOfOrderDetection:
    def test_reversed_steps_raise_value_error(self):
        sc = generate_scenario("sc_oor", "test", num_steps=5, seed=0)
        asm = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in sc.tree_outputs})

        # Process step 3 first — should succeed
        asm.step(sc, step_times[3], step=3)

        # Then try step 2 — should raise
        with pytest.raises(ValueError, match="Out-of-order"):
            asm.step(sc, step_times[2], step=2)

    def test_same_step_index_raises_value_error(self):
        sc = generate_scenario("sc_dup", "test", num_steps=5, seed=1)
        asm = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in sc.tree_outputs})

        asm.step(sc, step_times[0], step=0)
        with pytest.raises(ValueError, match="Out-of-order"):
            asm.step(sc, step_times[1], step=0)  # same step=0

    def test_sequential_steps_succeed(self):
        sc = generate_scenario("sc_seq", "test", num_steps=5, seed=2)
        asm = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in sc.tree_outputs})

        for i, ts in enumerate(step_times[:5]):
            batch = asm.step(sc, ts, step=i)
            assert batch.u_history.shape == (NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM)


# ---------------------------------------------------------------------------
# Future-time input exclusion
# ---------------------------------------------------------------------------

class TestFutureTimeExclusion:
    def test_future_received_observation_excluded(self):
        """
        Observation with received_at > issue_time must NOT enter the U vector.
        This verifies the core data-leakage guard.
        """
        from person4.event_horizon_p4.config import H_DIM
        zone_id   = MOCK_ZONE_IDS[0]
        issue_t   = _BASE
        future_t  = _BASE + timedelta(seconds=100)

        # Future observation (should be excluded)
        future_obs = _make_obs(zone_id, "density", 99.0,
                               observed=future_t, received=future_t)
        # No tree out; zero H embedding
        h_emb = np.zeros(H_DIM, dtype=np.float32)

        u_vec = assemble_u_vec(zone_id, issue_t, [future_obs], None, h_emb)
        # X[0] (density) should be the default fallback (0.8), not 99.0
        assert u_vec[0] != 99.0, "Future observation must not enter U vector"

    def test_past_received_observation_included(self):
        """
        Observation with received_at <= issue_time should populate the U vector.
        """
        from person4.event_horizon_p4.config import H_DIM
        zone_id = MOCK_ZONE_IDS[0]
        issue_t = _BASE + timedelta(seconds=50)
        past_t  = _BASE + timedelta(seconds=10)

        past_obs = _make_obs(zone_id, "density", 3.5,
                             observed=past_t, received=past_t)
        h_emb = np.zeros(H_DIM, dtype=np.float32)

        u_vec = assemble_u_vec(zone_id, issue_t, [past_obs], None, h_emb)
        # density value (X[0]) should be 3.5
        assert abs(u_vec[0] - 3.5) < 1e-4, (
            f"Past observation should populate X[0]. Got {u_vec[0]}"
        )

    def test_stale_observation_sets_mask_to_zero(self):
        """
        Observation older than OBS_STALE_THRESHOLD_SEC (90 s) gets obs_mask = 0.
        """
        from person4.event_horizon_p4.config import OBS_STALE_THRESHOLD_SEC, X_DIM, H_DIM, AGE_DIM
        zone_id = MOCK_ZONE_IDS[0]
        issue_t = _BASE + timedelta(seconds=200)
        old_t   = _BASE   # 200 s ago — stale

        old_obs = _make_obs(zone_id, "density", 5.0,
                            observed=old_t, received=old_t)
        h_emb = np.zeros(H_DIM, dtype=np.float32)

        u_vec = assemble_u_vec(zone_id, issue_t, [old_obs], None, h_emb)
        # In layout [X(6), AGE(6), OBS_MASK(6), D(6), H(16)], obs_mask starts at X_DIM + AGE_DIM
        obs_mask_start = X_DIM + AGE_DIM
        obs_mask_density = u_vec[obs_mask_start]
        assert obs_mask_density == 0.0, (
            f"Stale observation should set obs_mask=0, got {obs_mask_density}"
        )


# ---------------------------------------------------------------------------
# Rolling window padding
# ---------------------------------------------------------------------------

class TestRollingWindowPadding:
    def test_early_steps_padded_to_history_length(self):
        sc = generate_scenario("sc_pad", "test", num_steps=3, seed=3)
        asm = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in sc.tree_outputs})

        # Step 0 — only 1 real step; rest should be padded
        batch = asm.step(sc, step_times[0], step=0)
        assert batch.u_history.shape[1] == HISTORY_STEPS, (
            "History should always be padded to HISTORY_STEPS"
        )

    def test_padded_rows_equal_first_real_step(self):
        sc = generate_scenario("sc_pad2", "test", num_steps=3, seed=4)
        asm = SequenceWindowAssembler()
        asm.reset()
        step_times = sorted({t.evaluation_time for t in sc.tree_outputs})

        batch = asm.step(sc, step_times[0], step=0)
        zone_hist = batch.u_history[0]  # (HISTORY_STEPS, U_DIM)

        # All padded rows should be equal (left-pad with the first real vector)
        first_real = zone_hist[-1]
        for row in zone_hist[:-1]:  # all but last should be the padding
            np.testing.assert_array_almost_equal(row, first_real)

    def test_reset_clears_buffer(self):
        sc = generate_scenario("sc_reset", "test", num_steps=HISTORY_STEPS + 2, seed=5)
        asm = SequenceWindowAssembler()

        # Build a full buffer
        asm.reset()
        step_times = sorted({t.evaluation_time for t in sc.tree_outputs})
        for i, ts in enumerate(step_times[:HISTORY_STEPS]):
            asm.step(sc, ts, step=i)

        # Reset and verify step ordering starts fresh
        asm.reset()
        batch = asm.step(sc, step_times[0], step=0)
        # After reset, first step should produce padded window
        assert batch.u_history.shape == (NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM)


# ---------------------------------------------------------------------------
# Convenience helper build_all_windows
# ---------------------------------------------------------------------------

class TestBuildAllWindows:
    def test_build_all_windows_generates_batch_per_step(self):
        """build_all_windows returns one WindowBatch per distinct tree_output timestamp."""
        num_steps = 5
        sc = generate_scenario("sc_all_win", "test", num_steps=num_steps, seed=10)
        batches = build_all_windows(sc)
        assert len(batches) == num_steps

    def test_build_all_windows_tensor_shapes(self):
        """Every WindowBatch must conform to (NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM)."""
        sc = generate_scenario("sc_all_shapes", "test", num_steps=4, seed=11)
        batches = build_all_windows(sc)
        for batch in batches:
            assert batch.u_history.shape == (NUM_MOCK_ZONES, HISTORY_STEPS, U_DIM)
            assert batch.zone_ids == MOCK_ZONE_IDS
            assert len(batch.target_times) == len(HORIZONS_SEC)

    def test_build_all_windows_monotonic_target_times(self):
        """Target times across consecutive batches must advance chronologically."""
        sc = generate_scenario("sc_all_time", "test", num_steps=4, seed=12)
        batches = build_all_windows(sc)
        for i in range(1, len(batches)):
            for h in HORIZONS_SEC:
                assert batches[i].target_times[h] > batches[i - 1].target_times[h]
