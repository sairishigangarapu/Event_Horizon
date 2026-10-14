"""
Training, seeded Monte Carlo refitting, and benchmarking for the decision-tree bank.

Leakage contract (acceptance criterion):
    Tree training and fine-tuning read ONLY observations (Person 1) and
    reference current-state labels. Forecasts — Stream A or Stream B — are
    never used as labels or features. build_training_set enforces this by
    construction (it has no forecasts parameter that is used) and raises
    LeakageError if forecast rows are passed.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crowd_reliability.ml.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureScaler,
    build_feature_row,
)
from crowd_reliability.ml.tree_bank import CHANNEL_METADATA, TREE_CHANNELS, DecisionTreeBank
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES
from crowd_reliability.ml.trees import TreeSnapshot, infer_tree_vector
from crowd_reliability.reliability.conflicts import norm_ts_5s
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import (
    SCENARIO_PRESETS,
    ScenarioConfig,
    VenueLayout,
    get_default_venue,
)
from crowd_reliability.simulation.simulator import CrowdSimulator

ALLOWED_TRAINING_INPUTS = ("observations", "reference_states", "zones")

CHANNEL_TO_VARIABLE = {m["name"]: m["source_variable"] for m in CHANNEL_METADATA}
EXIT_CHANNEL = "probability_exit_open"


class LeakageError(RuntimeError):
    """Raised when a forbidden input (e.g. forecasts) reaches tree training."""


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _as_naive_utc(dt: Any) -> datetime:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def build_training_set(
    observations: List[Any],
    reference_states: List[Any],
    zone_exit_flags: Dict[str, bool],
    training_cutoff: Optional[datetime] = None,
    max_samples: Optional[int] = None,
    forecasts: Optional[List[Any]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Builds (X, Y, meta) for tree fitting.

    X: (n, 18) observed-only feature rows (fixed OBSERVED_ONLY_FEATURES order).
    Y: per-channel label arrays taken from reference current state at the same
       (scenario_id, zone_id, state_at) — 5s-normalized matching. Non-exit zones
       get -1 for probability_exit_open (filtered at fit time).

    Labels with label_available_at after training_cutoff are excluded
    (chronological hygiene: no future labels in fitting).
    """
    if forecasts is not None:
        raise LeakageError(
            "Forecast rows were passed to tree training. Tree targets must come "
            "from simulator reference current state only — never Stream A/B forecasts."
        )

    cutoff_naive = _as_naive_utc(training_cutoff) if training_cutoff else None

    eligible_refs = []
    for ref in reference_states:
        label_at = _field(ref, "label_available_at")
        if cutoff_naive and label_at and _as_naive_utc(label_at) > cutoff_naive:
            continue
        eligible_refs.append(ref)

    ref_lookup: Dict[Tuple[str, str, str, int], float] = {}
    for ref in eligible_refs:
        key = (
            _field(ref, "scenario_id"),
            _field(ref, "zone_id"),
            _field(ref, "variable"),
            norm_ts_5s(_field(ref, "state_at")),
        )
        ref_lookup[key] = float(_field(ref, "value"))

    sample_keys = sorted({
        (_field(r, "scenario_id"), _field(r, "zone_id"), norm_ts_5s(_field(r, "state_at")))
        for r in eligible_refs
    })
    if max_samples:
        sample_keys = sample_keys[:max_samples]

    X_rows: List[np.ndarray] = []
    Y: Dict[str, List[float]] = {c: [] for c in TREE_CHANNELS}

    for scenario_id, zone_id, t_norm in sample_keys:
        # Reconstruct an as_of time for feature extraction; reference state_at is
        # on the 5s grid so the norm round-trips exactly.
        as_of = datetime(2026, 1, 1) + timedelta(seconds=t_norm)
        X_rows.append(build_feature_row(zone_id, as_of, observations, scenario_id=scenario_id))

        has_exit = zone_exit_flags.get(zone_id, False)
        for channel in TREE_CHANNELS:
            var = CHANNEL_TO_VARIABLE[channel]
            ref_val = ref_lookup.get((scenario_id, zone_id, var, t_norm))
            if channel == EXIT_CHANNEL:
                if not has_exit or ref_val is None or ref_val < 0:
                    Y[channel].append(-1.0)
                else:
                    Y[channel].append(ref_val)
            else:
                # Missing reference for a variable falls back to the observed
                # feature value for that variable (keeps sample aligned).
                feat_idx = TARGET_VAR_NAMES.index(var)
                Y[channel].append(ref_val if ref_val is not None else float(X_rows[-1][feat_idx]))

    meta = {
        "n_samples": len(X_rows),
        "scenario_ids": sorted({k[0] for k in sample_keys}),
        "training_cutoff": training_cutoff.isoformat() if training_cutoff else None,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "allowed_inputs": list(ALLOWED_TRAINING_INPUTS),
    }
    X = np.array(X_rows, dtype=np.float32) if X_rows else np.zeros((0, len(TREE_CHANNELS) * 3), dtype=np.float32)
    return X, {k: np.array(v, dtype=np.float64) for k, v in Y.items()}, meta


def train_tree_bank(
    observations: List[Any],
    reference_states: List[Any],
    zone_exit_flags: Dict[str, bool],
    seed: int = 42,
    training_cutoff: Optional[datetime] = None,
    max_depth: int = 6,
    max_samples: Optional[int] = None,
) -> Tuple[DecisionTreeBank, FeatureScaler, Dict[str, Any]]:
    """
    Fits the versioned scaler on training features only, then fits the bank.
    Both artifacts carry the training seed and scenario metadata.
    """
    X, Y, meta = build_training_set(
        observations, reference_states, zone_exit_flags,
        training_cutoff=training_cutoff, max_samples=max_samples,
    )
    if X.shape[0] == 0:
        raise ValueError("No training samples: check scenario ids and reference states.")

    scaler = FeatureScaler().fit(
        X,
        fit_scenario_ids=meta["scenario_ids"],
        fit_cutoff=training_cutoff,
    )
    bank = DecisionTreeBank(max_depth=max_depth, random_state=seed)
    bank.fit(X, Y, scaler=scaler)

    meta.update({
        "tree_version": bank.version,
        "scaler_version": scaler.version,
        "seed": seed,
        "max_depth": max_depth,
    })
    return bank, scaler, meta


def save_training_artifacts(
    bank: DecisionTreeBank,
    scaler: FeatureScaler,
    meta: Dict[str, Any],
    out_dir: str,
) -> Dict[str, str]:
    """Writes tree bank, scaler (+ sidecar metadata), and train_metadata.json."""
    os.makedirs(out_dir, exist_ok=True)
    bank_path = os.path.join(out_dir, f"tree_bank_{bank.version}.joblib")
    scaler_path = os.path.join(out_dir, f"scaler_{scaler.version}.joblib")
    meta_path = os.path.join(out_dir, "train_metadata.json")

    bank.save(bank_path)
    scaler.save(scaler_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"tree_bank": bank_path, "scaler": scaler_path, "metadata": meta_path}


def split_scenarios(
    scenario_ids: List[str],
    holdout_scenario: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Deterministic scenario split; holdout defaults to the last sorted id."""
    ordered = sorted(scenario_ids)
    if not ordered:
        raise ValueError("No scenarios to split.")
    holdout = holdout_scenario or ordered[-1]
    train = [s for s in ordered if s != holdout]
    return train, [holdout]


def predict_scenario_vectors(
    bank: DecisionTreeBank,
    observations: List[Any],
    zone_exit_flags: Dict[str, bool],
    scenario_id: str,
    eval_times: List[datetime],
    zone_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs infer_tree_vector at each evaluation time and returns flat prediction
    rows: {scenario_id, zone_id, evaluation_time, channel, value, mask, tree_version}.
    """
    if zone_ids is None:
        zone_ids = sorted({_field(o, "zone_id") for o in observations} | set(zone_exit_flags.keys()))

    rows: List[Dict[str, Any]] = []
    for t in sorted(eval_times):
        snapshot = TreeSnapshot(
            scenario_id=scenario_id,
            as_of_time=t,
            observations=observations,
            zone_ids=zone_ids,
            zone_exit_flags=zone_exit_flags,
        )
        result = infer_tree_vector(snapshot, bank)
        for i, z_id in enumerate(result.zone_ids):
            for c_idx, channel in enumerate(TREE_CHANNELS):
                rows.append({
                    "scenario_id": scenario_id,
                    "zone_id": z_id,
                    "evaluation_time": t.isoformat(),
                    "channel": channel,
                    "value": float(result.values[i, c_idx]),
                    "mask": int(result.masks[i, c_idx]),
                    "tree_version": bank.version,
                })
    return rows


def benchmark_channels(
    prediction_rows: List[Dict[str, Any]],
    reference_states: List[Any],
) -> Dict[str, Dict[str, float]]:
    """
    Per-channel metrics against reference current state:
    MAE/RMSE for regression channels; accuracy/Brier for probability_exit_open.
    Masked-out (inapplicable) predictions are excluded.
    """
    ref_lookup: Dict[Tuple[str, str, str, int], float] = {}
    for ref in reference_states:
        key = (
            _field(ref, "scenario_id"),
            _field(ref, "zone_id"),
            _field(ref, "variable"),
            norm_ts_5s(_as_naive_utc(_field(ref, "state_at"))),
        )
        ref_lookup[key] = float(_field(ref, "value"))

    errors: Dict[str, List[float]] = {c: [] for c in TREE_CHANNELS}
    exit_pairs: List[Tuple[float, float]] = []

    for row in prediction_rows:
        if not row.get("mask", 1):
            continue
        channel = row["channel"]
        var = CHANNEL_TO_VARIABLE[channel]
        t_norm = norm_ts_5s(datetime.fromisoformat(row["evaluation_time"]))
        ref_val = ref_lookup.get((row["scenario_id"], row["zone_id"], var, t_norm))
        if ref_val is None:
            continue
        pred = float(row["value"])
        if channel == EXIT_CHANNEL:
            if ref_val >= 0:
                exit_pairs.append((pred, ref_val))
        else:
            errors[channel].append(pred - ref_val)

    metrics: Dict[str, Dict[str, float]] = {}
    for channel in TREE_CHANNELS:
        if channel == EXIT_CHANNEL:
            if exit_pairs:
                correct = sum(1 for p, r in exit_pairs if (p >= 0.5) == (r >= 0.5))
                brier = sum((p - r) ** 2 for p, r in exit_pairs) / len(exit_pairs)
                metrics[channel] = {
                    "accuracy": round(correct / len(exit_pairs), 4),
                    "brier_score": round(brier, 4),
                    "count": len(exit_pairs),
                }
            else:
                metrics[channel] = {"accuracy": None, "brier_score": None, "count": 0}
        else:
            errs = errors[channel]
            if errs:
                arr = np.array(errs)
                metrics[channel] = {
                    "mae": round(float(np.mean(np.abs(arr))), 4),
                    "rmse": round(float(np.sqrt(np.mean(arr ** 2))), 4),
                    "count": len(errs),
                }
            else:
                metrics[channel] = {"mae": None, "rmse": None, "count": 0}
    return metrics


def monte_carlo_refit(
    n_runs: int,
    base_scenario_ids: Optional[List[str]] = None,
    seed: int = 42,
    out_dir: str = "models/trees/mc",
    venue: Optional[VenueLayout] = None,
    holdout_scenario: Optional[str] = None,
    max_eval_times: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Seeded Monte Carlo refitting from Person 1 observations + reference labels only.

    Each run re-simulates the base scenarios with seed offsets, refits the tree
    bank on the training scenarios, and records out-of-sample predictions on the
    held-out scenario (never fitted on it). Artifacts are versioned per run with
    full parameter/seeding metadata.
    """
    venue = venue or get_default_venue()
    zone_exit_flags = {z["zone_id"]: bool(z.get("has_exit", False)) for z in venue.zones}
    zone_ids = sorted(zone_exit_flags.keys())

    base_ids = sorted(base_scenario_ids or SCENARIO_PRESETS.keys())
    train_ids, holdout_ids = split_scenarios(base_ids, holdout_scenario)
    holdout_id = holdout_ids[0]

    os.makedirs(out_dir, exist_ok=True)
    oos_path = os.path.join(out_dir, "oos_predictions.jsonl")
    refs_path = os.path.join(out_dir, "mc_references.jsonl")

    run_summaries: List[Dict[str, Any]] = []
    with open(oos_path, "w") as oos_file, open(refs_path, "w") as refs_file:
        for run in range(n_runs):
            run_seed = seed + run
            sim = CrowdSimulator(venue=venue, seed=run_seed)
            gen = ObservationGenerator(seed=run_seed)

            train_obs: List[Any] = []
            train_refs: List[Any] = []
            holdout_obs: List[Any] = []
            holdout_refs: List[Any] = []

            for base_id in base_ids:
                preset = SCENARIO_PRESETS[base_id]
                mc_id = f"{base_id}__mc{run:03d}"
                cfg = ScenarioConfig(
                    **{**preset.model_dump(), "scenario_id": mc_id, "seed": preset.seed + run_seed}
                )
                refs, _ = sim.run_simulation(cfg)
                obs = gen.generate_observations(refs, cfg)
                if base_id == holdout_id:
                    holdout_obs.extend(obs)
                    holdout_refs.extend(refs)
                else:
                    train_obs.extend(obs)
                    train_refs.extend(refs)

            # Only Monte Carlo observations + current-state labels reach fitting.
            train_cutoff = max(_as_naive_utc(_field(r, "label_available_at")) for r in train_refs)
            bank, scaler, meta = train_tree_bank(
                train_obs, train_refs, zone_exit_flags,
                seed=run_seed, training_cutoff=train_cutoff,
            )

            run_dir = os.path.join(out_dir, f"run_{run:03d}")
            paths = save_training_artifacts(bank, scaler, meta, run_dir)
            mc_meta = {
                "run": run,
                "seed": run_seed,
                "base_scenario_ids": base_ids,
                "train_scenario_ids": [f"{s}__mc{run:03d}" for s in train_ids],
                "holdout_scenario_id": f"{holdout_id}__mc{run:03d}",
                "tree_version": bank.version,
                "scaler_version": scaler.version,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "params": {"max_depth": bank.max_depth},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "artifacts": paths,
            }
            with open(os.path.join(run_dir, "mc_metadata.json"), "w") as f:
                json.dump(mc_meta, f, indent=2)

            # Chronological OOS predictions on the held-out scenario.
            eval_times = sorted({_as_naive_utc(_field(r, "state_at")) for r in holdout_refs})[::3]
            if max_eval_times:
                eval_times = eval_times[:max_eval_times]
            oos_rows = predict_scenario_vectors(
                bank, holdout_obs, zone_exit_flags,
                scenario_id=f"{holdout_id}__mc{run:03d}",
                eval_times=eval_times, zone_ids=zone_ids,
            )
            for row in oos_rows:
                row["run"] = run
                oos_file.write(json.dumps(row) + "\n")

            for ref in holdout_refs:
                refs_file.write(json.dumps({
                    "run": run,
                    "scenario_id": _field(ref, "scenario_id"),
                    "zone_id": _field(ref, "zone_id"),
                    "variable": _field(ref, "variable"),
                    "value": float(_field(ref, "value")),
                    "state_at": _as_naive_utc(_field(ref, "state_at")).isoformat(),
                }) + "\n")

            run_summaries.append({
                "run": run,
                "seed": run_seed,
                "n_train_samples": meta["n_samples"],
                "n_oos_rows": len(oos_rows),
            })

    summary = {
        "n_runs": n_runs,
        "seed": seed,
        "base_scenario_ids": base_ids,
        "holdout_scenario": holdout_id,
        "oos_predictions": oos_path,
        "mc_references": refs_path,
        "runs": run_summaries,
    }
    with open(os.path.join(out_dir, "mc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
