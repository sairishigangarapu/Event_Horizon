"""
config.py – Provisional model and schema configuration for Person 4.

NOTE: This module uses synthetic/mock constants for local testing but its
architectural dimensions (U_DIM, H_DIM, HISTORY_STEPS, HORIZONS_SEC) are
actively reconciled with the production ArchitecturePipeline. Any changes
to these values must be coordinated with Persons 1–3 and will require
retraining model checkpoints.

All architectural constants defined here are reconciled with Persons 1-3
for pipeline compatibility.

Key design decisions:
- GRU input U[z,t] = concat(X, age, obs_mask, D, H).
  D (tree vector) is passed directly even though D already enters the
  hypernetwork dynamic context (workplan rule).
- Sample interval: 5 seconds.
- History window: 12 steps (= 60 seconds look-back, matching ArchitecturePipeline.lookback_steps).
- Prediction horizons: 30 s (offset 6), 60 s (offset 12), 120 s (offset 24).
- Mock zones: 8 (matches VENUE_ZONE_IDS).
- Observation channels (X): 6  — matches TARGET_VAR_NAMES order.
- Tree vector (D): 6  — matches Person 2 DecisionTreeBank TREE_CHANNELS.
- Spatial embedding (H): 16 — matches Person 3 EdgeAwareGNN embed_dim.
- Age features: 6  — one per observation channel.
- Observation masks: 6  — one per channel (1 = valid/recent, 0 = stale/missing).
- Total GRU input dim (U_DIM): 6+6+6+6+16 = 40.
  Layout: [raw_values(6) | norm_ages(6) | obs_masks(6) | tree_values D(6) | spatial H(16)].
  tree_mask(6) and exit_mask(1) are intentionally excluded from the GRU input vector (see sequence.py).
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Output variable metadata
# ---------------------------------------------------------------------------

#: Canonical variable names, output order, and physical units.
#: The exit_open variable is only produced for exit-applicable zones.
VAR_NAMES: List[str] = [
    "density",       # persons / m²
    "speed",         # m / s
    "inflow",        # persons / s
    "outflow",       # persons / s
    "queue_length",  # persons
    "exit_open",     # binary probability  (0 = closed, 1 = open)
]

VAR_UNITS: dict = {
    "density":      "p/m2",
    "speed":        "m/s",
    "inflow":       "p/s",
    "outflow":      "p/s",
    "queue_length": "persons",
    "exit_open":    "binary",
}

NUM_CHANNELS: int = len(VAR_NAMES)         # 6
NUM_NUMERIC_CHANNELS: int = NUM_CHANNELS - 1   # 5  (excludes exit_open)


# ---------------------------------------------------------------------------
# Temporal contract (aligned with ArchitecturePipeline)
# ---------------------------------------------------------------------------

#: Sample interval in seconds. Changing this invalidates stored history.
SAMPLE_INTERVAL_SEC: int = 5

#: Number of historical steps fed to the GRU. Aligned with ArchitecturePipeline.lookback_steps (12 * 5 s = 60 s).
HISTORY_STEPS: int = 12

#: Prediction horizons in seconds. Corresponding step offsets:
#:   30 s → offset 6,  60 s → offset 12,  120 s → offset 24.
HORIZONS_SEC: List[int] = [30, 60, 120]

#: Derived step offsets (informational only).
HORIZON_OFFSETS: List[int] = [h // SAMPLE_INTERVAL_SEC for h in HORIZONS_SEC]


# ---------------------------------------------------------------------------
# Feature-vector dimensions (reconciled with Persons 1–3 on main)
# ---------------------------------------------------------------------------

#: Observation vector X: one value per channel.
X_DIM: int = NUM_CHANNELS        # 6

#: Decision-tree vector D: passed directly even though also in hypernetwork.
D_DIM: int = 6                   # 6  (matches Person 2 DecisionTreeBank TREE_CHANNELS)

#: Spatial GNN embedding H: 16-dim representation from Person 3's EdgeAwareGNN embed_dim.
H_DIM: int = 16                  # 16 (verified from models/production/spatial config)

#: Age features: one normalised age per observation channel.
AGE_DIM: int = NUM_CHANNELS      # 6

#: Observation missing/stale masks: one per observation channel.
OBS_MASK_DIM: int = NUM_CHANNELS # 6

#: Tree applicability masks: one per tree channel (carried in dynamic spatial context).
TREE_MASK_DIM: int = D_DIM       # 6

#: Exit applicability flag (scalar mask): 1 for exit zones, 0 otherwise.
EXIT_MASK_DIM: int = 1           # 1

#: Total GRU input dimension U[z,t] in ArchitecturePipeline:
#: [raw_values(6), norm_ages(6), obs_masks(6), tree_values(6), h_embeddings(16)] = 40.
U_DIM: int = X_DIM + AGE_DIM + OBS_MASK_DIM + D_DIM + H_DIM  # 40

# Sanity assertion baked in at import time.
assert U_DIM == 40, f"U_DIM mismatch: expected 40, got {U_DIM}"


# ---------------------------------------------------------------------------
# Venue zone metadata (reconciled with Person 1 get_default_venue)
# ---------------------------------------------------------------------------

VENUE_ZONE_IDS: List[str] = [
    "ent_a", "ent_b", "corridor_1", "corridor_2",
    "concourse", "stair_1", "exit_1", "exit_2"
]
EXIT_ZONE_IDS: List[str] = ["exit_1", "exit_2"]
NUM_MOCK_ZONES: int = len(VENUE_ZONE_IDS)  # 8 zones


# ---------------------------------------------------------------------------
# Observation staleness threshold
# ---------------------------------------------------------------------------

#: An observation older than this (in seconds) is flagged as missing/stale.
OBS_STALE_THRESHOLD_SEC: float = 90.0

#: Normalisation denominator for age features.
AGE_NORM_SEC: float = 120.0

#: Sentinel value stored for missing-observation age (pre-normalisation).
MISSING_AGE_SENTINEL: float = 999.0


# ---------------------------------------------------------------------------
# Model hyper-parameters (GRU)
# ---------------------------------------------------------------------------

@dataclass
class GRUConfig:
    """Hyper-parameters for the GRU temporal forecaster."""
    input_dim: int = U_DIM      # 40
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1        # only active when num_layers > 1 and in training mode
    num_horizons: int = len(HORIZONS_SEC)
    horizons_sec: List[int] = field(default_factory=lambda: list(HORIZONS_SEC))
    version: str = "gru_forecaster_v1_mock"


# ---------------------------------------------------------------------------
# Training configuration (smoke-test defaults)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    batch_size: int = 4
    epochs: int = 5
    seed: int = 42
    scenario_split_ratios: dict = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    checkpoint_dir: str = "artifacts"
    fixture_label: str = "synthetic_fixture_only_NOT_VALIDATED"


# ---------------------------------------------------------------------------
# Stream / version identifiers
# ---------------------------------------------------------------------------

STREAM_B_ID: str = "stream_b_architecture"
STREAM_A_ID: str = "stream_a_central"

MODEL_VERSION: str = "gru_forecaster_v1"


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

@dataclass
class P4Config:
    """Single object exposing all Person 4 configuration."""
    sample_interval_sec: int = SAMPLE_INTERVAL_SEC
    history_steps: int = HISTORY_STEPS
    horizons_sec: List[int] = field(default_factory=lambda: list(HORIZONS_SEC))
    u_dim: int = U_DIM
    num_mock_zones: int = NUM_MOCK_ZONES
    gru: GRUConfig = field(default_factory=GRUConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    stream_b_id: str = STREAM_B_ID
    stream_a_id: str = STREAM_A_ID
    model_version: str = MODEL_VERSION
