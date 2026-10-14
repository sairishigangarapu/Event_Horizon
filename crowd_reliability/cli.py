import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from crowd_reliability.config import settings
from crowd_reliability.database import init_db, SessionLocal, engine
from crowd_reliability.models.schema import (
    Base, ZoneRecord, ConnectionRecord, ObservationRecord,
    TreeOutputRecord, ForecastRecord, ReferenceStateRecord,
    AssessmentRecord, ModelManifestRecord
)
from crowd_reliability.models.manifest import PipelineManifest
from crowd_reliability.simulation.scenarios import (
    get_default_venue, SCENARIO_PRESETS, ScenarioConfig
)
from crowd_reliability.simulation.simulator import CrowdSimulator, TARGET_VARIABLES
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.ml.tree_bank import TREE_CHANNELS, DecisionTreeBank
from crowd_reliability.ml.tree_train import (
    benchmark_channels,
    monte_carlo_refit,
    predict_scenario_vectors,
    save_training_artifacts,
    split_scenarios,
    train_tree_bank,
)
from crowd_reliability.ml.trees import TreeSnapshot, TreeVectorResult, infer_tree_vector
from crowd_reliability.dataset.adapter import ingest_dataset, load_split_scenarios, load_zone_flags
from crowd_reliability.services.central_source import CentralSourceAdapter
from crowd_reliability.services.comparator import ForecastComparator
from crowd_reliability.services.evaluator import EvaluationService

DEFAULT_TREE_ARTIFACT_DIR = "./models/trees"
DATASET_DIR = "./data/dataset"

def cmd_db_init():
    """Initializes SQLite database and populates venue layout."""
    print("[db-init] Creating database tables...")
    init_db()

    venue = get_default_venue()
    db = SessionLocal()
    try:
        # Populate zones
        for z in venue.zones:
            existing = db.query(ZoneRecord).filter(ZoneRecord.zone_id == z["zone_id"]).first()
            if not existing:
                rec = ZoneRecord(
                    zone_id=z["zone_id"],
                    name=z["name"],
                    geometry={"x": z.get("x", 0), "y": z.get("y", 0), "area_m2": z["area_m2"]},
                    zone_type=z["zone_type"],
                    capacity_definition=z["capacity_definition"],
                    map_version="v1.0"
                )
                db.add(rec)

        # Populate connections
        for c in venue.connections:
            existing = db.query(ConnectionRecord).filter(ConnectionRecord.edge_id == c["edge_id"]).first()
            if not existing:
                rec = ConnectionRecord(
                    edge_id=c["edge_id"],
                    from_zone=c["from_zone"],
                    to_zone=c["to_zone"],
                    width_m=c["width_m"],
                    direction=c.get("direction", "unidirectional"),
                    map_version="v1.0"
                )
                db.add(rec)

        db.commit()
        print(f"[db-init] Successfully initialized {len(venue.zones)} zones and {len(venue.connections)} connections in {settings.database_url}")
    finally:
        db.close()

def cmd_seed_demo(seed: int = 42):
    """Generates demo scenarios, trains baseline models, and creates replay fixtures."""
    print(f"[seed-demo] Initializing demo data with seed={seed}...")
    cmd_db_init()

    venue = get_default_venue()
    simulator = CrowdSimulator(venue=venue, seed=seed)
    obs_generator = ObservationGenerator(seed=seed)
    central_adapter = CentralSourceAdapter()

    db = SessionLocal()
    try:
        all_ref_records: List[ReferenceStateRecord] = []
        all_obs_records: List[ObservationRecord] = []

        # 1. Run simulation for preset scenarios
        for sc_name, sc_config in SCENARIO_PRESETS.items():
            print(f"[seed-demo] Simulating scenario: {sc_name}...")
            ref_records, telemetry = simulator.run_simulation(sc_config)
            obs_records = obs_generator.generate_observations(ref_records, sc_config)

            # Store in DB
            db.bulk_save_objects(ref_records)
            db.bulk_save_objects(obs_records)
            db.commit()

            all_ref_records.extend(ref_records)
            all_obs_records.extend(obs_records)

        # 2. Build and export central forecast replay fixture for 'crowd_surge'
        surge_sc = SCENARIO_PRESETS["crowd_surge"]
        surge_refs = [r for r in all_ref_records if r.scenario_id == "crowd_surge"]
        print("[seed-demo] Generating simulated central forecast fixture...")
        cent_forecasts = central_adapter.generate_simulated_forecasts(
            scenario_id="crowd_surge",
            ref_records=surge_refs,
            venue=venue,
            horizons_sec=[30, 60, 120]
        )
        central_adapter.export_to_csv(cent_forecasts, settings.central_forecast_path)
        db.bulk_save_objects(cent_forecasts)
        db.commit()
        print(f"[seed-demo] Saved {len(cent_forecasts)} central forecasts to {settings.central_forecast_path}")

        # 3. Train DecisionTreeBank on observation snapshot features -> reference targets
        print("[seed-demo] Training decision-tree bank and initializing neural pipeline...")
        pipeline = ArchitecturePipeline(venue=venue, horizons_sec=[30, 60, 120])

        # Prepare simple training data for tree bank
        # Match observations to reference states
        X_train = []
        Y_train = {
            "estimated_density": [],
            "estimated_speed": [],
            "estimated_inflow": [],
            "estimated_outflow": [],
            "estimated_queue_length": [],
            "probability_exit_open": []
        }

        # Build feature snapshots
        ref_lookup = {(r.scenario_id, r.zone_id, r.variable, r.state_at.isoformat()): r.value for r in all_ref_records}
        unique_samples = set((r.scenario_id, r.zone_id, r.state_at) for r in all_ref_records)

        for sc_id, z_id, t_state in list(unique_samples)[:2000]:  # limit training set size for rapid seed
            t_iso = t_state.isoformat()
            z_data = next(z for z in venue.zones if z["zone_id"] == z_id)
            raw_vals, norm_ages, obs_masks = pipeline.extract_zone_features(z_id, t_state, all_obs_records)
            dt_feat = list(raw_vals) + list(norm_ages) + list(obs_masks)

            X_train.append(dt_feat)
            Y_train["estimated_density"].append(ref_lookup.get((sc_id, z_id, "density", t_iso), raw_vals[0]))
            Y_train["estimated_speed"].append(ref_lookup.get((sc_id, z_id, "speed", t_iso), raw_vals[1]))
            Y_train["estimated_inflow"].append(ref_lookup.get((sc_id, z_id, "inflow", t_iso), raw_vals[2]))
            Y_train["estimated_outflow"].append(ref_lookup.get((sc_id, z_id, "outflow", t_iso), raw_vals[3]))
            Y_train["estimated_queue_length"].append(ref_lookup.get((sc_id, z_id, "queue_length", t_iso), raw_vals[4]))
            Y_train["probability_exit_open"].append(ref_lookup.get((sc_id, z_id, "exit_open", t_iso), 1.0 if z_data.get("has_exit") else -1.0))

        import numpy as np
        pipeline.tree_bank.fit(np.array(X_train), {k: np.array(v) for k, v in Y_train.items()})

        # Save production pipeline artifacts
        pipeline.save_pipeline(settings.model_dir)

        manifest_rec = ModelManifestRecord(
            pipeline_version=pipeline.manifest.pipeline_version,
            tree_version=pipeline.manifest.tree_version,
            hypernet_version=pipeline.manifest.hypernet_version,
            gnn_version=pipeline.manifest.gnn_version,
            temporal_version=pipeline.manifest.temporal_version,
            scaler_version=pipeline.manifest.scaler_version,
            schema_version=pipeline.manifest.schema_version,
            training_cutoff=datetime.now(timezone.utc),
            scenario_split="production",
            is_active=True,
            model_hashes=pipeline.manifest.model_hashes
        )
        db.merge(manifest_rec)
        db.commit()

        print(f"[seed-demo] Pipeline saved to {settings.model_dir} and registered in database.")

        # 4. Generate initial architecture forecasts for crowd_surge
        print("[seed-demo] Generating initial Architecture forecasts for crowd_surge...")
        surge_obs = [o for o in all_obs_records if o.scenario_id == "crowd_surge"]
        unique_surge_times = sorted(list(set(o.received_at for o in surge_obs)))

        arch_forecasts: List[ForecastRecord] = []
        tree_records: List[TreeOutputRecord] = []

        # Run pipeline every 15 seconds
        for t_step in unique_surge_times[::3]:
            trees, f_recs = pipeline.step("crowd_surge", t_step, surge_obs)
            tree_records.extend(trees)
            arch_forecasts.extend(f_recs)

        db.bulk_save_objects(tree_records)
        db.bulk_save_objects(arch_forecasts)
        db.commit()

        # 5. Quick initial evaluation
        eval_service = EvaluationService()
        metrics = eval_service.score_predictions(arch_forecasts + cent_forecasts, surge_refs)
        print("[seed-demo] Demo seeding complete! Summary evaluation available.")
    finally:
        db.close()

def cmd_replay(scenario: str = "crowd_surge", seed: int = 42):
    """Replays a scenario and runs the architecture prediction pipeline."""
    print(f"[replay] Starting replay for scenario '{scenario}' (seed={seed})...")
    venue = get_default_venue()
    pipeline = ArchitecturePipeline(venue=venue)
    if os.path.exists(os.path.join(settings.model_dir, "manifest.json")):
        pipeline.load_pipeline(settings.model_dir)

    db = SessionLocal()
    try:
        obs_records = db.query(ObservationRecord).filter(ObservationRecord.scenario_id == scenario).all()
        ref_records = db.query(ReferenceStateRecord).filter(ReferenceStateRecord.scenario_id == scenario).all()

        if not obs_records:
            print(f"[replay] No observations found for '{scenario}'. Running simulation on the fly...")
            cfg = SCENARIO_PRESETS.get(scenario, SCENARIO_PRESETS["crowd_surge"])
            cfg.seed = seed
            sim = CrowdSimulator(venue=venue, seed=seed)
            gen = ObservationGenerator(seed=seed)
            ref_records, _ = sim.run_simulation(cfg)
            obs_records = gen.generate_observations(ref_records, cfg)
            db.bulk_save_objects(ref_records)
            db.bulk_save_objects(obs_records)
            db.commit()

        print(f"[replay] Running inference on {len(obs_records)} observations...")
        times = sorted(list(set(o.received_at for o in obs_records)))[::3]

        arch_forecasts = []
        for t in times:
            _, f_recs = pipeline.step(scenario, t, obs_records)
            arch_forecasts.extend(f_recs)

        print(f"[replay] Generated {len(arch_forecasts)} Architecture forecasts across {len(times)} time steps.")

        # Score predictions
        eval_service = EvaluationService()
        metrics = eval_service.score_predictions(arch_forecasts, ref_records)
        arch_metrics = metrics.get("stream_b_architecture", {})

        print("\n=== Replay Performance (Stream B: Architecture) ===")
        print(f"{'Horizon':<10} | {'Variable':<15} | {'MAE':<10} | {'RMSE':<10}")
        print("-" * 52)
        for h, v_dict in arch_metrics.items():
            for v_name, m in v_dict.items():
                print(f"{str(h)+'s':<10} | {v_name:<15} | {m.get('mae', 0.0):<10.4f} | {m.get('rmse', 0.0):<10.4f}")

    finally:
        db.close()

def cmd_evaluate(split: str = "test"):
    """Runs reference-based evaluation across scenarios."""
    print(f"[evaluate] Evaluating split: {split}...")
    venue = get_default_venue()
    db = SessionLocal()
    try:
        forecasts = db.query(ForecastRecord).all()
        reference_states = db.query(ReferenceStateRecord).all()

        if not forecasts or not reference_states:
            print("[evaluate] Database is empty. Please run 'python -m crowd_reliability.cli seed-demo' first.")
            return

        eval_service = EvaluationService()
        metrics = eval_service.score_predictions(forecasts, reference_states)

        print("\n=== Comparative Evaluation Summary ===")
        for stream_id, h_data in metrics.items():
            label = "Stream A (Central Source)" if "central" in stream_id else "Stream B (Architecture)"
            print(f"\n--- {label} ---")
            print(f"{'Horizon':<10} | {'Variable':<15} | {'MAE':<10} | {'RMSE':<10} | {'Extra':<15}")
            print("-" * 68)
            for h, v_dict in sorted(h_data.items()):
                for v_name, m in sorted(v_dict.items()):
                    extra = f"Acc: {m['accuracy']:.2f}" if "accuracy" in m else f"N={m['count']}"
                    print(f"{str(h)+'s':<10} | {v_name:<15} | {m.get('mae', 0.0):<10.4f} | {m.get('rmse', 0.0):<10.4f} | {extra:<15}")

    finally:
        db.close()

def _print_channel_metrics(metrics: Dict[str, Dict[str, Any]]):
    print(f"\n{'Channel':<26} | {'MAE':<10} | {'RMSE':<10} | {'Acc':<8} | {'Brier':<8} | {'N':<6}")
    print("-" * 82)
    for channel in TREE_CHANNELS:
        m = metrics.get(channel, {})
        mae = f"{m['mae']:.4f}" if m.get("mae") is not None else "-"
        rmse = f"{m['rmse']:.4f}" if m.get("rmse") is not None else "-"
        acc = f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "-"
        brier = f"{m['brier_score']:.4f}" if m.get("brier_score") is not None else "-"
        print(f"{channel:<26} | {mae:<10} | {rmse:<10} | {acc:<8} | {brier:<8} | {m.get('count', 0):<6}")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    import json
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_train_trees(train: str = None, test: str = None, seed: int = 42,
                    out_dir: str = DEFAULT_TREE_ARTIFACT_DIR, max_samples: int = None):
    """Fits the versioned scaler + decision-tree bank on training scenarios only.

    Labels come from reference current state (reference_states) — never forecasts.
    """
    print(f"[train-trees] Training tree bank (seed={seed}) -> {out_dir}")
    venue = get_default_venue()
    zone_exit_flags = {z["zone_id"]: bool(z.get("has_exit", False)) for z in venue.zones}

    db = SessionLocal()
    try:
        all_obs = db.query(ObservationRecord).all()
        all_refs = db.query(ReferenceStateRecord).all()
    finally:
        db.close()

    if not all_refs:
        print("[train-trees] No reference states found. Run 'seed-demo' first.")
        return

    available = sorted({r.scenario_id for r in all_refs})
    if train and test:
        train_ids = [s.strip() for s in train.split(",") if s.strip()]
        test_ids = [s.strip() for s in test.split(",") if s.strip()]
    else:
        train_ids, test_ids = split_scenarios(available)
    print(f"[train-trees] Train scenarios: {train_ids} | Held-out: {test_ids}")

    train_obs = [o for o in all_obs if o.scenario_id in train_ids]
    train_refs = [r for r in all_refs if r.scenario_id in train_ids]
    test_obs = [o for o in all_obs if o.scenario_id in test_ids]
    test_refs = [r for r in all_refs if r.scenario_id in test_ids]

    train_cutoff = max(r.label_available_at for r in train_refs)
    bank, scaler, meta = train_tree_bank(
        train_obs, train_refs, zone_exit_flags,
        seed=seed, training_cutoff=train_cutoff, max_samples=max_samples,
    )
    meta["train_scenario_ids"] = train_ids
    meta["test_scenario_ids"] = test_ids
    paths = save_training_artifacts(bank, scaler, meta, out_dir)
    print(f"[train-trees] Saved: {paths['tree_bank']}, {paths['scaler']}, {paths['metadata']}")

    # Out-of-sample benchmark on the held-out scenario
    if test_refs:
        eval_times = sorted({r.state_at for r in test_refs})[::3]
        rows = predict_scenario_vectors(
            bank, test_obs, zone_exit_flags, test_ids[0], eval_times,
            zone_ids=sorted(zone_exit_flags.keys()),
        )
        metrics = benchmark_channels(rows, test_refs)
        print(f"\n=== Held-out benchmark ({test_ids[0]}) ===")
        _print_channel_metrics(metrics)
        import json
        with open(os.path.join(out_dir, "holdout_benchmark.json"), "w") as f:
            json.dump(metrics, f, indent=2)


def cmd_export_tree_vector(scenario: str = "crowd_surge", as_of: str = None,
                           out: str = "./data/demo/tree_vector_sample.json"):
    """Exports one D[z,t] sample vector (all six channels) for Persons 3 and 4."""
    venue = get_default_venue()
    zone_exit_flags = {z["zone_id"]: bool(z.get("has_exit", False)) for z in venue.zones}
    zone_ids = sorted(zone_exit_flags.keys())

    bank = DecisionTreeBank()
    versioned_path = os.path.join(DEFAULT_TREE_ARTIFACT_DIR, f"tree_bank_{bank.version}.joblib")
    prod_path = os.path.join(settings.model_dir, "tree_bank.joblib")
    if os.path.exists(versioned_path):
        bank.load(versioned_path)
        print(f"[export-tree-vector] Loaded {versioned_path} (version={bank.version})")
    elif os.path.exists(prod_path):
        bank.load(prod_path)
        print(f"[export-tree-vector] Loaded {prod_path} (version={bank.version})")
    else:
        print("[export-tree-vector] WARNING: no fitted tree bank found; using heuristic fallback.")

    db = SessionLocal()
    try:
        observations = db.query(ObservationRecord).filter(
            ObservationRecord.scenario_id == scenario
        ).all()
    finally:
        db.close()
    if not observations:
        print(f"[export-tree-vector] No observations for scenario '{scenario}'. Run 'seed-demo' first.")
        return

    eval_time = datetime.fromisoformat(as_of) if as_of else max(o.received_at for o in observations)
    snapshot = TreeSnapshot(
        scenario_id=scenario,
        as_of_time=eval_time,
        observations=observations,
        zone_ids=zone_ids,
        zone_exit_flags=zone_exit_flags,
    )
    result = infer_tree_vector(snapshot, bank)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(result.to_json())
    print(f"[export-tree-vector] Wrote {len(result.zone_ids)} zones x {len(TREE_CHANNELS)} channels to {out}")


def cmd_monte_carlo_refit(runs: int = 5, seed: int = 42,
                          out_dir: str = os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "mc"),
                          max_eval_times: int = None):
    """Seeded Monte Carlo refitting from observations + reference labels only."""
    print(f"[mc-refit] {runs} runs, base seed={seed} -> {out_dir}")
    summary = monte_carlo_refit(
        n_runs=runs, seed=seed, out_dir=out_dir, max_eval_times=max_eval_times,
    )
    print(f"[mc-refit] Holdout scenario: {summary['holdout_scenario']}")
    for r in summary["runs"]:
        print(f"  run {r['run']:03d} (seed={r['seed']}): {r['n_train_samples']} train samples, {r['n_oos_rows']} OOS rows")

    preds = _load_jsonl(summary["oos_predictions"])
    refs = _load_jsonl(summary["mc_references"])
    metrics = benchmark_channels(preds, refs)
    print("\n=== Monte Carlo OOS benchmark (held-out scenario) ===")
    _print_channel_metrics(metrics)
    import json
    with open(os.path.join(out_dir, "mc_benchmarks.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def cmd_benchmark_trees(preds: str = os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "mc", "oos_predictions.jsonl"),
                        refs: str = os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "mc", "mc_references.jsonl")):
    """Recomputes per-channel benchmark metrics from stored MC predictions."""
    if not os.path.exists(preds) or not os.path.exists(refs):
        print(f"[benchmark-trees] Missing files. Run 'mc-refit' first (expects {preds}, {refs}).")
        return
    metrics = benchmark_channels(_load_jsonl(preds), _load_jsonl(refs))
    print("\n=== Per-channel benchmark vs reference current state ===")
    _print_channel_metrics(metrics)


def cmd_dataset_ingest(dataset_dir: str = DATASET_DIR, grid_cols: int = 8, grid_rows: int = 4,
                       exit_boundary: str = "bottom", force: bool = False):
    """Ingests the Event_Horizon dataset into the DB as zones + observations + reference states."""
    print(f"[dataset-ingest] Ingesting {dataset_dir} (grid {grid_cols}x{grid_rows}, exits={exit_boundary})")
    cmd_db_init()
    db = SessionLocal()
    try:
        summary = ingest_dataset(
            dataset_dir, db, grid_cols=grid_cols, grid_rows=grid_rows,
            exit_boundary=exit_boundary, clear_existing=force,
        )
    finally:
        db.close()
    print(f"[dataset-ingest] Zones: {summary['n_zones']} | "
          f"scenarios train/val/test = "
          f"{len(summary['scenarios']['train'])}/{len(summary['scenarios']['val'])}/{len(summary['scenarios']['test'])}")
    print(f"[dataset-ingest] Wrote {summary['zones_json']} and {summary['split_manifest']}")


def cmd_dataset_train(zones_json: str, split_manifest: str, seed: int = 42,
                      out_dir: str = os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "dataset"),
                      export_sample: str = "./data/demo/dataset_tree_vector_sample.json"):
    """Trains the tree bank on dataset train scenarios; benchmarks a test scenario and exports a D vector."""
    if not os.path.exists(zones_json) or not os.path.exists(split_manifest):
        print(f"[dataset-train] Missing {zones_json} or {split_manifest}. Run 'dataset-ingest' first.")
        return
    zone_flags = load_zone_flags(zones_json)
    with open(split_manifest) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    test_ids = splits["test"]

    print(f"[dataset-train] Train scenarios: {len(train_ids)} | Test scenarios: {len(test_ids)} (seed={seed})")
    db = SessionLocal()
    try:
        all_obs = db.query(ObservationRecord).filter(
            ObservationRecord.scenario_id.in_(train_ids + test_ids)
        ).all()
        all_refs = db.query(ReferenceStateRecord).filter(
            ReferenceStateRecord.scenario_id.in_(train_ids + test_ids)
        ).all()
    finally:
        db.close()
    if not all_refs:
        print("[dataset-train] No reference states found for those scenarios. Run 'dataset-ingest' first.")
        return

    train_obs = [o for o in all_obs if o.scenario_id in train_ids]
    train_refs = [r for r in all_refs if r.scenario_id in train_ids]
    test_obs = [o for o in all_obs if o.scenario_id in test_ids]
    test_refs = [r for r in all_refs if r.scenario_id in test_ids]

    train_cutoff = max(r.label_available_at for r in train_refs)
    bank, scaler, meta = train_tree_bank(
        train_obs, train_refs, zone_flags, seed=seed, training_cutoff=train_cutoff,
    )
    meta["train_scenario_ids"] = train_ids
    meta["test_scenario_ids"] = test_ids
    paths = save_training_artifacts(bank, scaler, meta, out_dir)
    print(f"[dataset-train] Artifacts: {paths['tree_bank']}, {paths['scaler']}")

    eval_times = sorted({r.state_at for r in test_refs})[::3]
    rows = predict_scenario_vectors(bank, test_obs, zone_flags, test_ids[0], eval_times,
                                    zone_ids=sorted(zone_flags.keys()))
    metrics = benchmark_channels(rows, test_refs)
    print("\n=== Dataset per-channel benchmark (test scenario: {}) ===".format(test_ids[0]))
    _print_channel_metrics(metrics)
    with open(os.path.join(out_dir, "dataset_benchmark.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Export one D[z,t] sample vector (Persons 3 & 4)
    sample_obs = [o for o in test_obs if o.scenario_id == test_ids[0]]
    if sample_obs:
        eval_time = max(o.received_at for o in sample_obs)
        result = infer_tree_vector(TreeSnapshot(
            scenario_id=test_ids[0], as_of_time=eval_time,
            observations=sample_obs, zone_ids=sorted(zone_flags.keys()),
            zone_exit_flags=zone_flags,
        ), bank)
        os.makedirs(os.path.dirname(export_sample) or ".", exist_ok=True)
        with open(export_sample, "w") as f:
            f.write(result.to_json())
        print(f"[dataset-train] Export sample D vector -> {export_sample}")


def main():
    parser = argparse.ArgumentParser(description="Crowd Reliability CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # db-init
    subparsers.add_parser("db-init", help="Initialize local SQLite schema and venue map")

    # seed-demo
    p_seed = subparsers.add_parser("seed-demo", help="Generate synthetic scenario fixtures and seed database")
    p_seed.add_argument("--seed", type=int, default=42, help="Random seed")

    # replay
    p_rep = subparsers.add_parser("replay", help="Replay scenario and run inference")
    p_rep.add_argument("--scenario", type=str, default="crowd_surge", help="Scenario identifier")
    p_rep.add_argument("--seed", type=int, default=42, help="Simulation seed")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Score forecasts against reference states")
    p_eval.add_argument("--split", type=str, default="test", help="Dataset split")

    # train-trees
    p_tt = subparsers.add_parser("train-trees", help="Fit versioned scaler + tree bank (labels: reference state only)")
    p_tt.add_argument("--train", type=str, default=None, help="Comma-separated training scenario ids")
    p_tt.add_argument("--test", type=str, default=None, help="Comma-separated held-out scenario ids")
    p_tt.add_argument("--seed", type=int, default=42, help="Random seed")
    p_tt.add_argument("--out", type=str, default=DEFAULT_TREE_ARTIFACT_DIR, help="Artifact output dir")
    p_tt.add_argument("--max-samples", type=int, default=None, help="Cap training samples")

    # export-tree-vector
    p_ev = subparsers.add_parser("export-tree-vector", help="Export sample D[z,t] vector JSON (Persons 3 & 4)")
    p_ev.add_argument("--scenario", type=str, default="crowd_surge", help="Scenario identifier")
    p_ev.add_argument("--as-of", type=str, default=None, help="ISO evaluation time (default: latest observation)")
    p_ev.add_argument("--out", type=str, default="./data/demo/tree_vector_sample.json", help="Output JSON path")

    # mc-refit
    p_mc = subparsers.add_parser("mc-refit", help="Seeded Monte Carlo tree refitting + OOS predictions")
    p_mc.add_argument("--runs", type=int, default=5, help="Number of Monte Carlo runs")
    p_mc.add_argument("--seed", type=int, default=42, help="Base random seed")
    p_mc.add_argument("--out", type=str, default=os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "mc"), help="Output dir")
    p_mc.add_argument("--max-eval-times", type=int, default=None, help="Cap OOS evaluation times per run")

    # benchmark-trees
    p_bt = subparsers.add_parser("benchmark-trees", help="Per-channel benchmark of stored MC predictions")
    p_bt.add_argument("--preds", type=str, default=os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "mc", "oos_predictions.jsonl"))
    p_bt.add_argument("--refs", type=str, default=os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "mc", "mc_references.jsonl"))

    # dataset-ingest
    p_ing = subparsers.add_parser("dataset-ingest", help="Ingest the Event_Horizon dataset into the DB")
    p_ing.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Dataset root dir")
    p_ing.add_argument("--grid-cols", type=int, default=8, help="Zone grid columns (longitudinal)")
    p_ing.add_argument("--grid-rows", type=int, default=1, help="Zone grid rows (lateral; 1 = full cross-section)")
    p_ing.add_argument("--exit-boundary", type=str, default="last_col",
                       choices=["last_col", "last_row", "bottom", "top", "left", "right"],
                       help="Which cells are exit zones")
    p_ing.add_argument("--force", action="store_true", help="Re-ingest even if grid zones already exist")

    # dataset-train
    p_dt = subparsers.add_parser("dataset-train", help="Train + benchmark trees on the ingested dataset")
    p_dt.add_argument("--zones-json", type=str, default=os.path.join(DATASET_DIR, "grid_zones.json"))
    p_dt.add_argument("--splits", type=str, default=os.path.join(DATASET_DIR, "splits.json"))
    p_dt.add_argument("--seed", type=int, default=42, help="Random seed")
    p_dt.add_argument("--out", type=str, default=os.path.join(DEFAULT_TREE_ARTIFACT_DIR, "dataset"),
                      help="Artifact output dir")
    p_dt.add_argument("--export-sample", type=str, default="./data/demo/dataset_tree_vector_sample.json",
                      help="Path for the sample D[z,t] export")

    args = parser.parse_args()

    if args.command == "db-init":
        cmd_db_init()
    elif args.command == "seed-demo":
        cmd_seed_demo(seed=args.seed)
    elif args.command == "replay":
        cmd_replay(scenario=args.scenario, seed=args.seed)
    elif args.command == "evaluate":
        cmd_evaluate(split=args.split)
    elif args.command == "train-trees":
        cmd_train_trees(train=args.train, test=args.test, seed=args.seed,
                        out_dir=args.out, max_samples=args.max_samples)
    elif args.command == "export-tree-vector":
        cmd_export_tree_vector(scenario=args.scenario, as_of=args.as_of, out=args.out)
    elif args.command == "mc-refit":
        cmd_monte_carlo_refit(runs=args.runs, seed=args.seed, out_dir=args.out,
                              max_eval_times=args.max_eval_times)
    elif args.command == "benchmark-trees":
        cmd_benchmark_trees(preds=args.preds, refs=args.refs)
    elif args.command == "dataset-ingest":
        cmd_dataset_ingest(
            dataset_dir=args.dataset_dir, grid_cols=args.grid_cols,
            grid_rows=args.grid_rows, exit_boundary=args.exit_boundary,
            force=args.force,
        )
    elif args.command == "dataset-train":
        cmd_dataset_train(
            zones_json=args.zones_json, split_manifest=args.splits,
            seed=args.seed, out_dir=args.out, export_sample=args.export_sample,
        )

if __name__ == "__main__":
    main()
