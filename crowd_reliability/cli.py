import os
import sys
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
from crowd_reliability.services.central_source import CentralSourceAdapter
from crowd_reliability.services.comparator import ForecastComparator
from crowd_reliability.services.evaluator import EvaluationService

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
            is_active=True
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

    args = parser.parse_args()

    if args.command == "db-init":
        cmd_db_init()
    elif args.command == "seed-demo":
        cmd_seed_demo(seed=args.seed)
    elif args.command == "replay":
        cmd_replay(scenario=args.scenario, seed=args.seed)
    elif args.command == "evaluate":
        cmd_evaluate(split=args.split)

if __name__ == "__main__":
    main()
