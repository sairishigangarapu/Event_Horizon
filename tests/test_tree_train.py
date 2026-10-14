"""
Tests for tree training and Monte Carlo refitting:
- leakage guard: forecasts can never reach tree fitting
- seeded determinism of MC refits and versioned artifacts
- held-out scenario excluded from fitting (OOS predictions)
- per-channel benchmark metrics
"""
import json
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from crowd_reliability.ml.tree_bank import TREE_CHANNELS
from crowd_reliability.ml.tree_train import (
    LeakageError,
    benchmark_channels,
    build_training_set,
    monte_carlo_refit,
    train_tree_bank,
)
from crowd_reliability.simulation.scenarios import ScenarioConfig
import crowd_reliability.ml.tree_train as tree_train_module

T_BASE = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
ZONE_FLAGS = {"exit_1": True, "concourse": False}


def _ref(scenario, zone, var, value, t):
    return {
        "scenario_id": scenario, "zone_id": zone, "variable": var,
        "value": value, "unit": "u", "state_at": t, "label_available_at": t,
    }


def _obs(scenario, zone, var, value, t):
    return {
        "report_id": f"r_{scenario}_{zone}_{var}",
        "scenario_id": scenario, "zone_id": zone, "source_id": "cam_1",
        "variable": var, "value": value,
        "observed_at": t, "received_at": t,
    }


def test_training_rejects_forecast_labels():
    refs = [_ref("s1", "concourse", "density", 1.0, T_BASE)]
    with pytest.raises(LeakageError):
        build_training_set([], refs, ZONE_FLAGS, forecasts=[{"anything": "goes"}])


def test_training_cutoff_excludes_future_labels():
    refs = [
        _ref("s1", "concourse", "density", 1.0, T_BASE),
        _ref("s1", "concourse", "density", 2.0, T_BASE + timedelta(seconds=300)),
    ]
    obs = [_obs("s1", "concourse", "density", 1.0, T_BASE)]
    X_all, _, meta_all = build_training_set(obs, refs, ZONE_FLAGS)
    X_cut, _, meta_cut = build_training_set(
        obs, refs, ZONE_FLAGS, training_cutoff=T_BASE + timedelta(seconds=60)
    )
    assert meta_cut["n_samples"] < meta_all["n_samples"]
    assert X_cut.shape[0] == meta_cut["n_samples"]


def test_train_tree_bank_deterministic_same_seed():
    refs = []
    obs = []
    for i in range(30):
        t = T_BASE + timedelta(seconds=5 * i)
        refs.append(_ref("s1", "concourse", "density", 1.0 + 0.1 * i, t))
        obs.append(_obs("s1", "concourse", "density", 1.0 + 0.1 * i, t))

    bank_a, scaler_a, meta_a = train_tree_bank(obs, refs, ZONE_FLAGS, seed=42)
    bank_b, _, _ = train_tree_bank(obs, refs, ZONE_FLAGS, seed=42)

    assert meta_a["tree_version"] == "dt_bank_v2"
    assert scaler_a.version == "scaler_obs_only_v1"

    from crowd_reliability.ml.features import build_feature_row
    row = build_feature_row("concourse", T_BASE + timedelta(seconds=50), obs, scenario_id="s1")
    vals_a, _ = bank_a.predict_zone(row, has_exit=False)
    vals_b, _ = bank_b.predict_zone(row, has_exit=False)
    assert vals_a == vals_b


def _tiny_presets():
    return {
        f"mc_{name}": ScenarioConfig(
            scenario_id=f"mc_{name}", title=name, description="tiny test scenario",
            duration_seconds=30, step_seconds=5, seed=seed,
            surge_start_seconds=9999, surge_end_seconds=9999, surge_multiplier=1.0,
            exit_restriction_start=9999,
            sensor_dropout_rate=0.05, stale_report_probability=0.0, conflict_probability=0.0,
        )
        for name, seed in [("a", 101), ("b", 202), ("c", 303)]
    }


@pytest.fixture
def tiny_scenarios(monkeypatch):
    presets = _tiny_presets()
    for key, cfg in presets.items():
        monkeypatch.setitem(tree_train_module.SCENARIO_PRESETS, key, cfg)
    return sorted(presets.keys())


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_monte_carlo_refit_seeded_deterministic_and_versioned(tiny_scenarios, tmp_path):
    out_a = str(tmp_path / "run_a")
    out_b = str(tmp_path / "run_b")

    summary_a = monte_carlo_refit(n_runs=2, base_scenario_ids=tiny_scenarios, seed=7, out_dir=out_a)
    summary_b = monte_carlo_refit(n_runs=2, base_scenario_ids=tiny_scenarios, seed=7, out_dir=out_b)

    with open(os.path.join(out_a, "oos_predictions.jsonl")) as fa, \
         open(os.path.join(out_b, "oos_predictions.jsonl")) as fb:
        assert fa.read() == fb.read(), "Same seed must reproduce identical OOS predictions"

    # Versioned artifacts + seeding metadata per run
    meta_path = os.path.join(out_a, "run_000", "mc_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["seed"] == 7
    assert meta["tree_version"] == "dt_bank_v2"
    assert meta["scaler_version"] == "scaler_obs_only_v1"
    assert os.path.exists(meta["artifacts"]["tree_bank"])
    assert os.path.exists(meta["artifacts"]["scaler"])

    # Holdout is never in the training split
    holdout = summary_a["holdout_scenario"]
    assert holdout == tiny_scenarios[-1]
    assert all(holdout not in tid for tid in meta["train_scenario_ids"])
    assert holdout in meta["holdout_scenario_id"]

    # OOS predictions only reference the held-out scenario
    oos_rows = _read_jsonl(summary_a["oos_predictions"])
    assert oos_rows
    assert {r["scenario_id"] for r in oos_rows} == {f"{holdout}__mc000", f"{holdout}__mc001"}
    assert {r["channel"] for r in oos_rows} == set(TREE_CHANNELS)

    # Different seed -> different predictions
    out_c = str(tmp_path / "run_c")
    summary_c = monte_carlo_refit(n_runs=2, base_scenario_ids=tiny_scenarios, seed=8, out_dir=out_c)
    with open(os.path.join(out_a, "oos_predictions.jsonl")) as fa, \
         open(os.path.join(out_c, "oos_predictions.jsonl")) as fc:
        assert fa.read() != fc.read()


def test_benchmark_channels_reports_all_six(tiny_scenarios, tmp_path):
    out_dir = str(tmp_path / "mc")
    summary = monte_carlo_refit(n_runs=1, base_scenario_ids=tiny_scenarios, seed=7, out_dir=out_dir)

    metrics = benchmark_channels(
        _read_jsonl(summary["oos_predictions"]),
        _read_jsonl(summary["mc_references"]),
    )
    assert set(metrics.keys()) == set(TREE_CHANNELS)
    assert metrics["estimated_density"]["count"] > 0
    assert metrics["estimated_density"]["mae"] is not None
    assert "accuracy" in metrics["probability_exit_open"]
    assert "brier_score" in metrics["probability_exit_open"]
