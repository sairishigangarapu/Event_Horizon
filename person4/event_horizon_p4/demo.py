"""
demo.py – End-to-end smoke test.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

Usage:
    python -m event_horizon_p4.demo --checkpoint artifacts/temporal_mock.pt --epochs 2

Workflow:
  1. (Optional) train a fresh model if requested via --epochs.
  2. Load checkpoint.
  3. Run inference on a test scenario.
  4. Run Stream A demo fixture on the same batch.
  5. Compare Stream A and B.
  6. Evaluate against reference labels.
  7. Write labelled demo artifacts to data/fixtures/person4/.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from .baseline import MovingAverageBaseline, PersistenceBaseline
from .central_adapter import CentralDemoFixtureAdapter
from .comparison import ForecastComparator, summarise_comparison
from .config import HORIZONS_SEC
from .evaluation import ReferenceEvaluator, export_metrics_csv
from .inference import TemporalForecaster, forecast_scenario
from .sequence import build_all_windows
from .synthetic import generate_all_scenarios

_FIXTURE_DIR = "data/fixtures/person4"
_WARNING     = (
    "SYNTHETIC FIXTURE ONLY — NOT VALIDATED. "
    "Do NOT use for real emergency decisions."
)


def _write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"[demo] Written: {path}")


def run_demo(checkpoint_path: str, epochs: int = 0) -> None:
    """
    Runs the full end-to-end demo.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        epochs:          If > 0, train a fresh model before demo.
    """
    # 1. Optional training
    if epochs > 0:
        print(f"[demo] Training for {epochs} epoch(s) before demo...")
        from .train import train
        train(output_path=checkpoint_path, epochs=epochs)

    # 2. Load model
    print(f"[demo] Loading checkpoint: {checkpoint_path}")
    forecaster = TemporalForecaster.from_checkpoint(checkpoint_path)

    # 3. Select a test scenario
    scenarios   = generate_all_scenarios()
    test_sc     = next(s for s in scenarios if s.split == "test")
    scenario_id = test_sc.scenario_id
    print(f"[demo] Running on test scenario: {scenario_id}")

    # 4. Architecture (Stream B) forecasts
    print("[demo] Running Stream B (architecture) inference ...")
    b_records = forecast_scenario(forecaster, test_sc, scenario_id)
    print(f"[demo] Stream B produced {len(b_records)} forecast records.")

    # 5. Persistence baseline
    pb = PersistenceBaseline()
    ma = MovingAverageBaseline(window=12)
    pb_records, ma_records = [], []
    for batch in build_all_windows(test_sc):
        pb_records.extend(pb.forecast(batch, scenario_id))
        ma_records.extend(ma.forecast(batch, scenario_id))

    # 6. Stream A (central demo fixture)
    print("[demo] Running Stream A (central demo fixture) ...")
    central = CentralDemoFixtureAdapter()
    a_records = []
    for batch in build_all_windows(test_sc):
        a_records.extend(central.forecast(batch, scenario_id))
    print(f"[demo] Stream A produced {len(a_records)} forecast records.")

    # 7. Comparison
    print("[demo] Comparing Stream A vs B ...")
    comparator = ForecastComparator()
    eval_time  = datetime.now(tz=timezone.utc)
    report     = comparator.compare(b_records, a_records, evaluation_time=eval_time)
    summary    = summarise_comparison(report)
    print(f"[demo] Paired: {len(report.paired)}, "
          f"Unavailable: {len(report.unavailable)}, "
          f"Unit-skipped: {len(report.unit_skipped)}")

    # 8. Evaluation
    print("[demo] Scoring against reference labels ...")
    evaluator    = ReferenceEvaluator()
    eval_report  = evaluator.score(b_records, test_sc.reference_states, evaluation_time=eval_time)
    print(f"[demo] Matched: {eval_report.n_matched}, "
          f"Awaiting labels: {eval_report.n_label_wait}, "
          f"No label: {eval_report.n_no_label}")

    # 9. Write artifacts
    print(f"[demo] Writing demo artifacts to {_FIXTURE_DIR}/ ...")

    _write_json(
        {
            "WARNING":     _WARNING,
            "scenario_id": scenario_id,
            "stream":      "stream_b_architecture",
            "n_records":   len(b_records),
            "records":     [r.to_dict() for r in b_records[:50]],  # sample
        },
        os.path.join(_FIXTURE_DIR, "mock_architecture_forecasts.json"),
    )

    _write_json(
        {
            "WARNING":            central.fixture_warning,
            "scenario_id":        scenario_id,
            "stream":             "stream_a_central_DEMO_FIXTURE",
            "available_vars":     central.available_variables,
            "unavailable_vars":   central.unavailable_variables,
            "n_records":          len(a_records),
            "records":            [r.to_dict() for r in a_records[:50]],  # sample
        },
        os.path.join(_FIXTURE_DIR, "mock_central_demo_fixture.json"),
    )

    _write_json(
        {
            "WARNING":              _WARNING,
            "scenario_id":          scenario_id,
            "evaluation_time":      eval_time.isoformat(),
            "summary_per_variable": summary,
            "paired_sample":        [p.to_dict() for p in report.paired[:20]],
            "unavailable":          {f"{k[0]}::{k[1]}": v
                                     for k, v in report.unavailable.items()},
            "unit_skipped":         report.unit_skipped,
        },
        os.path.join(_FIXTURE_DIR, "mock_comparison_differences.json"),
    )

    eval_dict = eval_report.to_dict()
    eval_dict["WARNING"]     = _WARNING
    eval_dict["scenario_id"] = scenario_id
    _write_json(eval_dict, os.path.join(_FIXTURE_DIR, "mock_evaluation_metrics.json"))

    # Export CSV
    export_metrics_csv(
        eval_report,
        os.path.join(_FIXTURE_DIR, "mock_evaluation_metrics.csv"),
    )

    # Baseline comparison summary
    pb_report = comparator.compare(pb_records, a_records, evaluation_time=eval_time)
    _write_json(
        {
            "WARNING":     _WARNING,
            "baseline":    "persistence",
            "n_paired":    len(pb_report.paired),
            "summary":     summarise_comparison(pb_report),
        },
        os.path.join(_FIXTURE_DIR, "mock_persistence_baseline_comparison.json"),
    )

    print("[demo] Done! All fixture files written.")
    print(f"[demo] Stream B numeric MAE (sample): "
          + ", ".join(
              f"{m.variable}@{m.horizon_sec}s={m.mae:.3f}"
              for m in eval_report.numeric[:3]
          ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Person 4: end-to-end smoke demo."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="artifacts/temporal_mock.pt",
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--epochs", type=int, default=0,
        help="If > 0, retrain before demo."
    )
    args = parser.parse_args()
    run_demo(checkpoint_path=args.checkpoint, epochs=args.epochs)


if __name__ == "__main__":
    _cli()
