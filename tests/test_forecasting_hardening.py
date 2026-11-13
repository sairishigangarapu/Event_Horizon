import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from starlette.requests import Request

from crowd_reliability.api import _rate_limit_client_key
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.ml.temporal import MultivariateTemporalForecaster
from crowd_reliability.ml.train_system import (
    conformal_intervals,
    ewma_baseline_metrics,
    fit_neural,
    monte_carlo_scenarios,
    run as run_train_system,
)
from crowd_reliability.models.schema import ForecastRecord, ReferenceStateRecord
from crowd_reliability.runtime import EVALUATION, SESSIONS, SESSION_TTL_SECONDS, _evict_expired_sessions
from crowd_reliability.services.evaluator import EvaluationService
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import SCENARIO_PRESETS, get_default_venue
from crowd_reliability.simulation.simulator import CrowdSimulator


def test_temporal_forecaster_numeric_activation_roundtrip(tmp_path):
    model = MultivariateTemporalForecaster(input_dim=40, hidden_dim=32, numeric_activation="softplus")
    assert model.hidden_dim == 32
    assert model.numeric_activation == "softplus"
    assert isinstance(model.numeric_heads[0][-1], torch.nn.Softplus)

    ckpt = str(tmp_path / "temporal_softplus.pt")
    model.save(ckpt)
    loaded = MultivariateTemporalForecaster.load(ckpt)
    assert loaded.hidden_dim == 32
    assert loaded.numeric_activation == "softplus"
    assert isinstance(loaded.numeric_heads[0][-1], torch.nn.Softplus)

    x = torch.zeros((2, 12, 40), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), loaded(x))


def test_train_system_wires_hidden_dim_and_reports_conformal_and_ewma(tmp_path):
    out_dir = tmp_path / "cand"
    summary = run_train_system(str(out_dir), seed=42, runs=2, epochs=2)
    assert summary["selected"]["hidden"] == 32
    assert "conformal_q90_intervals" in summary
    assert "density/30s" in summary["conformal_q90_intervals"]
    assert summary["conformal_q90_intervals"]["density/30s"]["q90_half_width"] >= 0.0
    assert "validation_ewma_metrics" in summary
    assert "heldout_ewma_metrics" in summary
    assert "wins_by_horizon" in summary["neural_validation_gate"]

    loaded = MultivariateTemporalForecaster.load(str(out_dir / "temporal_model.pt"))
    assert loaded.hidden_dim == 32
    assert loaded.numeric_activation == "softplus"


def test_pipeline_step_thread_safe_and_lru_bounded():
    venue = get_default_venue()
    pipeline = ArchitecturePipeline(venue=venue, horizons_sec=[30, 60, 120])
    pipeline.load_pipeline(os.path.join("models", "production"))
    pipeline.max_scenarios = 3

    sc = SCENARIO_PRESETS["crowd_surge"]
    refs, _ = CrowdSimulator(venue=venue, seed=42).run_simulation(sc)
    obs = ObservationGenerator(seed=42).generate_observations(refs, sc)
    t_eval = sorted(o.received_at for o in obs)[len(obs) // 2]

    def _call(idx: int):
        sid = f"sc_{idx}"
        sc_obs = [
            type(o)(
                report_id=f"{o.report_id}_{idx}",
                scenario_id=sid,
                zone_id=o.zone_id,
                source_id=o.source_id,
                variable=o.variable,
                value=o.value,
                unit=o.unit,
                observed_at=o.observed_at,
                published_at=o.published_at,
                received_at=o.received_at,
            )
            for o in obs[:40]
        ]
        return pipeline.step(sid, t_eval, sc_obs)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_call, range(6)))

    assert all(len(trees) == len(venue.zones) and len(fcs) > 0 for trees, fcs in results)
    assert len(pipeline.history_buffer) <= 3


def test_runtime_evicts_expired_sessions():
    now = time.monotonic()
    SESSIONS["__test_expired__"] = {"last_active_at": now - SESSION_TTL_SECONDS - 10.0}
    EVALUATION["__test_expired__"] = {("z1", "density", 0): 1.0}
    SESSIONS["__test_active__"] = {"last_active_at": now}
    try:
        evicted = _evict_expired_sessions(now=now)
        assert evicted >= 1
        assert "__test_expired__" not in SESSIONS
        assert "__test_expired__" not in EVALUATION
        assert "__test_active__" in SESSIONS
    finally:
        SESSIONS.pop("__test_expired__", None)
        SESSIONS.pop("__test_active__", None)
        EVALUATION.pop("__test_expired__", None)


def test_rate_limit_client_key_uses_forwarded_for():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/scenarios",
        "headers": [(b"x-forwarded-for", b"203.0.113.10, 10.0.0.1")],
        "client": ("10.0.0.1", 12345),
    }
    req = Request(scope)
    assert _rate_limit_client_key(req) == "203.0.113.10"


def test_evaluator_includes_bias_metric():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=30)
    f = ForecastRecord(
        forecast_id="f1",
        stream_id="stream_b_architecture",
        scenario_id="s1",
        zone_id="z1",
        variable="density",
        unit="p/m2",
        issued_at=t0,
        received_at=t0,
        target_at=t1,
        value=1.5,
        model_or_provider_version="v1",
    )
    r = ReferenceStateRecord(
        scenario_id="s1",
        zone_id="z1",
        variable="density",
        value=1.0,
        unit="p/m2",
        state_at=t1,
        label_available_at=t1,
    )
    res = EvaluationService().score_predictions([f], [r], horizons_sec=[30], evaluation_time=t1)
    assert res["stream_b_architecture"][30]["density"]["bias"] == 0.5
