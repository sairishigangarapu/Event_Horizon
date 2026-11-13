"""Regression test for F-005: pipeline step-cache staleness.

ArchitecturePipeline.step() caches forecasts keyed only on (scenario_id,
as_of_time).  If the same scenario and timestamp are reused with different
observation inputs (e.g. a retried tick after new evidence arrived), the
cache should detect the changed input and recompute.

These tests verify the intended contract.
"""
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.models.schema import ObservationRecord
from crowd_reliability.contracts import utc


def _make_obs(scenario_id, zone_id, variable, value, received_at, report_id="r1"):
    return ObservationRecord(
        report_id=report_id,
        scenario_id=scenario_id,
        zone_id=zone_id,
        source_id="sensor_1",
        variable=variable,
        value=value,
        unit="p/m2" if variable == "density" else "m/s",
        observed_at=received_at,
        published_at=received_at,
        received_at=received_at,
    )


@pytest.fixture(scope="module")
def pipeline():
    p = ArchitecturePipeline()
    p.load_pipeline(os.path.join("models", "candidates", "mc_v2"))
    yield p


def test_cache_is_invalidated_when_observations_change(pipeline):
    """F-005 regression: same timestamp + different observations => recompute.

    We spy on the temporal model's forward pass to establish whether the
    model was actually invoked on the second call. If the cache returns a
    stale result, the model is never called and outputs are identical.
    If the cache is invalidated, the model is called and outputs reflect
    the new input.
    """
    scenario_id = "cache_invalidation_test"
    T = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Observation set A: low density (0.5 p/m2)
    obs_A = [
        _make_obs(scenario_id, "ent_a", "density", 0.5, T, report_id="rA"),
    ]

    # First call — computation runs, cache is populated
    tree_A, forecast_A = pipeline.step(scenario_id, T, obs_A)

    # Observation set B: high density (5.0 p/m2) — semantically different
    obs_B = [
        _make_obs(scenario_id, "ent_a", "density", 5.0, T, report_id="rB"),
    ]

    # Spy on tempor_model.forward to count invocations during the second call
    call_count = [0]
    original_forward = pipeline.temporal_model.forward

    def counting_forward(*args, **kwargs):
        call_count[0] += 1
        return original_forward(*args, **kwargs)

    # Second call at the same timestamp with different observations
    with patch.object(pipeline.temporal_model, "forward", counting_forward):
        tree_B, forecast_B = pipeline.step(scenario_id, T, obs_B)

    # The temporal model must have been invoked — if it was not, the cache
    # returned the stale result from the first call.
    assert call_count[0] > 0, (
        "temporal_model.forward was never called on the second step() — "
        "cache returned stale results for different observation inputs."
    )

    # As a secondary check, verify the cache was updated with new results
    # (the cache tuple should differ from before the second call)
    cache_entry = pipeline._step_cache.get(scenario_id)
    assert cache_entry is not None, "Cache should have an entry after step()"
    cached_time, (cached_tree, cached_forecast), cached_fp = cache_entry
    assert cached_time == utc(T), "Cache timestamp should match the call time"
    assert cached_tree is not tree_A, (
        "Cache returned the same TreeOutputRecord list from the first call — "
        "cache was not invalidated despite different observations."
    )


def test_cache_hit_returns_same_result_for_identical_inputs(pipeline):
    """Control case: same timestamp + same observations => cache hit, same result.

    This verifies the cache still works for the intended case (retries with
    identical data) and does not over-invalidate.
    """
    scenario_id = "cache_hit_test"
    T = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

    obs = [
        _make_obs(scenario_id, "ent_a", "density", 0.5, T, report_id="r1"),
    ]

    # First call
    _, forecast_1 = pipeline.step(scenario_id, T, obs)

    # Spy to verify the model is NOT called on cache hit
    call_count = [0]
    original_forward = pipeline.temporal_model.forward

    def counting_forward(*args, **kwargs):
        call_count[0] += 1
        return original_forward(*args, **kwargs)

    # Second call with identical inputs — should hit cache
    with patch.object(pipeline.temporal_model, "forward", counting_forward):
        _, forecast_2 = pipeline.step(scenario_id, T, obs)

    # For identical inputs, cache SHOULD hit (model not called)
    assert call_count[0] == 0, (
        "temporal_model.forward was called on second step() with identical inputs — "
        "cache should have served the result without recomputation."
    )

    # Outputs should be the same (deep equality check on values)
    assert len(forecast_1) == len(forecast_2)
    for f1, f2 in zip(forecast_1, forecast_2):
        assert f1.value == pytest.approx(f2.value, abs=1e-6), (
            f"Forecast values differ on cache hit: {f1.value} vs {f2.value}"
        )
