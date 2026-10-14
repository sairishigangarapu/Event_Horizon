import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from crowd_reliability.models.schema import ForecastRecord, ReferenceStateRecord
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES

class EvaluationService:
    """
    Computes per-variable, per-horizon accuracy metrics against ground truth reference states.
    Compares Stream A (Central) vs Stream B (Architecture).
    """
    def score_predictions(
        self,
        forecasts: List[ForecastRecord],
        reference_states: List[ReferenceStateRecord],
        horizons_sec: List[int] = None
    ) -> Dict[str, Any]:
        if horizons_sec is None:
            horizons_sec = [30, 60, 120]

        def normalize_ts(dt: datetime) -> int:
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            epoch = datetime(2026, 1, 1)
            # Round to nearest 5-second step
            return int(round((dt - epoch).total_seconds() / 5.0) * 5)

        # Map reference states by (zone_id, variable, normalized_time_sec)
        ref_map: Dict[Tuple[str, str, int], float] = {
            (r.zone_id, r.variable, normalize_ts(r.state_at)): r.value
            for r in reference_states
        }

        # Metrics structure:
        # { stream_id: { horizon_sec: { variable: { "mae": ..., "rmse": ..., "count": ... } } } }
        metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}

        # Group forecasts by stream_id, horizon, variable
        grouped: Dict[Tuple[str, int, str], List[Tuple[float, float]]] = {}

        for f in forecasts:
            stream = f.stream_id
            lead_sec = int(round((f.target_at - f.issued_at).total_seconds()))

            # Match to nearest horizon
            matched_h = min(horizons_sec, key=lambda h: abs(h - lead_sec))
            if abs(matched_h - lead_sec) > 10:
                continue

            ref_val = ref_map.get((f.zone_id, f.variable, normalize_ts(f.target_at)))
            if ref_val is None:
                continue

            # Skip non-applicable exit_open
            if f.variable == "exit_open" and ref_val < 0:
                continue

            key = (stream, matched_h, f.variable)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((f.value, ref_val))

        for (stream, h_sec, var_name), pairs in grouped.items():
            if stream not in metrics:
                metrics[stream] = {}
            if h_sec not in metrics[stream]:
                metrics[stream][h_sec] = {}

            preds = np.array([p[0] for p in pairs])
            trues = np.array([p[1] for p in pairs])

            if len(preds) == 0:
                continue

            errors = preds - trues
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            var_metric: Dict[str, float] = {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "count": len(preds)
            }

            if var_name == "exit_open":
                # Classification accuracy and Brier score
                pred_binary = (preds >= 0.5).astype(float)
                acc = float(np.mean(pred_binary == trues))
                brier = float(np.mean((preds - trues) ** 2))
                var_metric["accuracy"] = round(acc, 4)
                var_metric["brier_score"] = round(brier, 4)

            metrics[stream][h_sec][var_name] = var_metric

        return metrics
