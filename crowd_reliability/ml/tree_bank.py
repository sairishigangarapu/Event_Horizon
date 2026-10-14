import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

TREE_CHANNELS = [
    "estimated_density",
    "estimated_speed",
    "estimated_inflow",
    "estimated_outflow",
    "estimated_queue_length",
    "probability_exit_open",
]

# Fixed per-channel metadata. Channel order in TREE_CHANNELS is the export order.
CHANNEL_METADATA: List[Dict[str, Any]] = [
    {"name": "estimated_density", "unit": "p/m2", "kind": "regression",
     "source_variable": "density", "applicability": "all_zones"},
    {"name": "estimated_speed", "unit": "m/s", "kind": "regression",
     "source_variable": "speed", "applicability": "all_zones"},
    {"name": "estimated_inflow", "unit": "p/s", "kind": "regression",
     "source_variable": "inflow", "applicability": "all_zones"},
    {"name": "estimated_outflow", "unit": "p/s", "kind": "regression",
     "source_variable": "outflow", "applicability": "all_zones"},
    {"name": "estimated_queue_length", "unit": "persons", "kind": "regression",
     "source_variable": "queue_length", "applicability": "all_zones"},
    {"name": "probability_exit_open", "unit": "probability", "kind": "classification",
     "source_variable": "exit_open", "applicability": "exit_zones_only"},
]

class DecisionTreeBank:
    """
    Bank of independent decision trees estimating current-state variables from observed context.
    NEVER collapses or sums channels into a single trust signal.
    Labels must come from simulator reference current state at time t — never forecasts.
    """
    def __init__(self, max_depth: int = 6, random_state: int = 42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.version = "dt_bank_v2"
        self.scaler: Any = None  # Optional fitted FeatureScaler applied before fit/predict
        self.trees: Dict[str, Any] = {
            "estimated_density": DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
            "estimated_speed": DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
            "estimated_inflow": DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
            "estimated_outflow": DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
            "estimated_queue_length": DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
            "probability_exit_open": DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        }
        self.is_fitted = False

    def channel_metadata(self) -> List[Dict[str, Any]]:
        return [dict(m) for m in CHANNEL_METADATA]

    def fit(self, features: np.ndarray, targets: Dict[str, np.ndarray], scaler: Any = None):
        """
        Fits each channel's tree on the training data.
        targets dict keys match TREE_CHANNELS.
        If a fitted FeatureScaler is provided, features are transformed with it and the
        scaler is stored so predict_zone applies the identical transform at inference.
        """
        self.scaler = scaler
        if self.scaler is not None:
            features = self.scaler.transform(features)
        for channel in TREE_CHANNELS:
            y = targets[channel]
            if channel == "probability_exit_open":
                # Only train on valid exit samples (not masked out with -1)
                valid_mask = (y >= 0)
                if np.sum(valid_mask) > 10 and len(np.unique(y[valid_mask])) > 1:
                    self.trees[channel].fit(features[valid_mask], y[valid_mask].astype(int))
                else:
                    # Fallback single class
                    self.trees[channel].fit(features[:2], np.array([1, 1]))
            else:
                self.trees[channel].fit(features, y)
        self.is_fitted = True

    def predict_zone(self, feature_row: np.ndarray, has_exit: bool = False) -> Tuple[List[float], List[int]]:
        """
        Generates D[z,t] vector preserving every individual channel.
        Returns:
            channel_values: [estimated_density, estimated_speed, ..., probability_exit_open]
            channel_masks: [1, 1, 1, 1, 1, 1 if has_exit else 0]
        """
        x = feature_row.reshape(1, -1)
        if self.scaler is not None and self.is_fitted:
            x = self.scaler.transform(x)
        values = []
        masks = []

        for channel in TREE_CHANNELS:
            if not self.is_fitted:
                # Heuristic fallback before fitting
                if channel == "estimated_density":
                    val = float(feature_row[0]) if len(feature_row) > 0 else 1.0
                elif channel == "estimated_speed":
                    val = float(feature_row[1]) if len(feature_row) > 1 else 1.2
                elif channel == "probability_exit_open":
                    val = 1.0 if has_exit else 0.0
                else:
                    val = float(feature_row[2]) if len(feature_row) > 2 else 0.5
            else:
                tree = self.trees[channel]
                if channel == "probability_exit_open":
                    if not has_exit:
                        val = 0.0
                    else:
                        if hasattr(tree, "predict_proba"):
                            proba = tree.predict_proba(x)[0]
                            val = float(proba[1]) if len(proba) > 1 else float(proba[0])
                        else:
                            val = float(tree.predict(x)[0])
                else:
                    val = float(tree.predict(x)[0])

            mask = 1 if (channel != "probability_exit_open" or has_exit) else 0
            values.append(round(val, 4))
            masks.append(mask)

        return values, masks

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump({
            "trees": self.trees,
            "is_fitted": self.is_fitted,
            "version": self.version,
            "scaler": self.scaler,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.trees = data["trees"]
        self.is_fitted = data["is_fitted"]
        self.version = data.get("version", self.version)
        self.scaler = data.get("scaler")
        self.max_depth = data.get("max_depth", self.max_depth)
        self.random_state = data.get("random_state", self.random_state)
