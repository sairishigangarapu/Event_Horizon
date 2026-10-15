import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from crowd_reliability.ml.temporal import TARGET_VAR_NAMES
from crowd_reliability.contracts import select_reports

FEATURE_SCHEMA_VERSION = "obs_only_v1"

RAW_VALUE_FEATURES = [f"{v}_value" for v in TARGET_VAR_NAMES]
NORM_AGE_FEATURES = [f"{v}_age_norm" for v in TARGET_VAR_NAMES]
OBS_MASK_FEATURES = [f"{v}_observed" for v in TARGET_VAR_NAMES]

# Fixed, observed-only feature order. Never reorder; bump FEATURE_SCHEMA_VERSION instead.
OBSERVED_ONLY_FEATURES: List[str] = RAW_VALUE_FEATURES + NORM_AGE_FEATURES + OBS_MASK_FEATURES

AGE_NORMALIZATION_SECONDS = 120.0
OBSERVED_MASK_MAX_AGE_SECONDS = 90.0
MISSING_AGE_SENTINEL = 999.0

# Documented defaults for unobserved variables (agreed observed-only schema, Person 1).
DEFAULT_VALUES = {
    "density": 0.8,
    "speed": 1.2,
    "inflow": 1.0,
    "outflow": 1.0,
    "queue_length": 0.0,
    "exit_open": 1.0,
}


def _field(obs: Any, name: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _as_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def build_feature_row(
    zone_id: str,
    as_of_time: datetime,
    observations: List[Any],
    scenario_id: Optional[str] = None,
) -> np.ndarray:
    """
    Builds the fixed-order, observed-only 18-dim feature row for one zone:
    [raw_values(6), norm_ages(6), obs_masks(6)] in TARGET_VAR_NAMES order,
    which corresponds 1:1 to TREE_CHANNELS order.

    Deterministic: latest observation per variable received at or before as_of_time wins.
    No forecasts and no future outcomes are ever read here. When scenario_id is
    given, observations from other scenarios are excluded (no cross-scenario
    leakage — scenarios share wall-clock time ranges).
    """
    as_of = _as_naive_utc(as_of_time)
    var_indices = {v: i for i, v in enumerate(TARGET_VAR_NAMES)}
    raw_values = np.zeros(6, dtype=np.float32)
    ages_sec = np.full(6, MISSING_AGE_SENTINEL, dtype=np.float32)
    obs_masks = np.zeros(6, dtype=np.float32)

    for obs in select_reports(observations, as_of, scenario_id=scenario_id, zone_id=zone_id):
        if _field(obs, "zone_id") != zone_id:
            continue
        if scenario_id is not None and _field(obs, "scenario_id") not in (None, scenario_id):
            continue
        received_at = _field(obs, "received_at")
        if received_at is None or _as_naive_utc(received_at) > as_of:
            continue
        variable = _field(obs, "variable")
        if variable not in var_indices:
            continue
        idx = var_indices[variable]
        age = max(0.0, (as_of - _as_naive_utc(_field(obs, "observed_at"))).total_seconds())
        if age < ages_sec[idx]:
            ages_sec[idx] = age
            raw_values[idx] = float(_field(obs, "value"))
            obs_masks[idx] = 1.0 if age <= OBSERVED_MASK_MAX_AGE_SECONDS else 0.0

    for i, var in enumerate(TARGET_VAR_NAMES):
        if obs_masks[i] == 0:
            raw_values[i] = DEFAULT_VALUES[var]

    norm_ages = np.clip(ages_sec / AGE_NORMALIZATION_SECONDS, 0.0, 1.0)
    return np.concatenate([raw_values, norm_ages, obs_masks]).astype(np.float32)


class FeatureScaler:
    """
    Versioned StandardScaler wrapper for the observed-only feature schema.
    Fit ONLY on training-scenario feature matrices — never on test scenarios.
    """

    VERSION = "scaler_obs_only_v1"

    def __init__(self):
        self.version = self.VERSION
        self.scaler = StandardScaler()
        self.feature_names: List[str] = list(OBSERVED_ONLY_FEATURES)
        self.is_fitted = False
        self.fit_scenario_ids: List[str] = []
        self.fit_cutoff: Optional[str] = None

    def fit(
        self,
        features: np.ndarray,
        fit_scenario_ids: Optional[List[str]] = None,
        fit_cutoff: Optional[datetime] = None,
    ) -> "FeatureScaler":
        if features.ndim != 2 or features.shape[1] != len(OBSERVED_ONLY_FEATURES):
            raise ValueError(
                f"Expected (n, {len(OBSERVED_ONLY_FEATURES)}) features, got {features.shape}"
            )
        self.scaler.fit(features)
        self.is_fitted = True
        self.fit_scenario_ids = sorted(fit_scenario_ids or [])
        self.fit_cutoff = fit_cutoff.isoformat() if fit_cutoff else None
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform()")
        return self.scaler.transform(features)

    def metadata(self) -> Dict[str, Any]:
        return {
            "scaler_version": self.version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": self.feature_names,
            "fit_scenario_ids": self.fit_scenario_ids,
            "fit_cutoff": self.fit_cutoff,
            "fitted": self.is_fitted,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump({"scaler": self.scaler, "metadata": self.metadata()}, filepath)
        meta_path = os.path.splitext(filepath)[0] + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "FeatureScaler":
        data = joblib.load(filepath)
        obj = cls()
        obj.scaler = data["scaler"]
        meta = data.get("metadata", {})
        obj.version = meta.get("scaler_version", cls.VERSION)
        obj.feature_names = meta.get("feature_names", list(OBSERVED_ONLY_FEATURES))
        obj.fit_scenario_ids = meta.get("fit_scenario_ids", [])
        obj.fit_cutoff = meta.get("fit_cutoff")
        obj.is_fitted = meta.get("fitted", True)
        return obj
