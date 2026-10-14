"""
config.py – Provisional model and schema configuration for Person 4.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

All architectural constants defined here must be reconciled with Persons 1-3
before real-data integration. The dimensions here are MOCK values.

Key design decisions:
- GRU input U[z,t] = concat(X, D, H, age, missing/applicability masks).
  D (tree vector) is passed directly even though D already enters the
  hypernetwork dynamic context (workplan rule).
- Sample interval: 5 seconds.
- History window: 60 steps (= 300 seconds look-back).
- Prediction horizons: 30 s (offset 6), 60 s (offset 12), 120 s (offset 24).
- Mock zones: 6.
- Observation channels (X): 6  — matches TARGET_VAR_NAMES order.
- Tree vector (D): 6  — must match Person 3 tree output channels.
- Spatial embedding (H): 8  — MOCK; final GNN output dim is not fixed.
- Age features: 6  — one per observation channel.
- Observation masks: 6  — one per channel (1 = valid/recent, 0 = stale/missing).
- Tree masks: 6  — one per tree channel (1 = applicable, 0 = inapplicable).
- Exit applicability mask: 1  — 1 for exit zones, 0 for non-exit zones.
- Total GRU input dim (U_DIM): 6+6+8+6+6+6+1 = 39.
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
# Temporal contract (mock defaults — tune only with justification)
# ---------------------------------------------------------------------------

#: Sample interval in seconds.  Changing this invalidates stored history.
SAMPLE_INTERVAL_SEC: int = 5

#: Number of historical steps fed to the GRU.  5 s × 60 = 300 s look-back.
HISTORY_STEPS: int = 60

#: Prediction horizons in seconds.  Corresponding step offsets:
#:   30 s → offset 6,  60 s → offset 12,  120 s → offset 24.
HORIZONS_SEC: List[int] = [30, 60, 120]

#: Derived step offsets (informational only).
HORIZON_OFFSETS: List[int] = [h // SAMPLE_INTERVAL_SEC for h in HORIZONS_SEC]


# ---------------------------------------------------------------------------
# Feature-vector dimensions (MOCK — reconcile with Persons 1–3)
# ---------------------------------------------------------------------------

#: Observation vector X: one value per channel.
X_DIM: int = NUM_CHANNELS        # 6

#: Decision-tree vector D: passed directly even though also in hypernetwork.
D_DIM: int = 6                   # 6  (mock; should match Person 3 tree channels)

#: Spatial GNN embedding H: MOCK.  Person 3 report notes 16-dim generated
#: node values but final GNN output dim is not fixed in the workplan.
H_DIM: int = 8                   # 8  MOCK

#: Age features: one normalised age per observation channel.
AGE_DIM: int = NUM_CHANNELS      # 6

#: Observation missing/stale masks: one per observation channel.
OBS_MASK_DIM: int = NUM_CHANNELS # 6

#: Tree applicability masks: one per tree channel.
TREE_MASK_DIM: int = D_DIM       # 6

#: Exit applicability flag (scalar mask): 1 for exit zones, 0 otherwise.
EXIT_MASK_DIM: int = 1           # 1

#: Total GRU input dimension U[z,t].
U_DIM: int = (
    X_DIM
    + D_DIM
    + H_DIM
    + AGE_DIM
    + OBS_MASK_DIM
    + TREE_MASK_DIM
    + EXIT_MASK_DIM
)  # 39

# Sanity assertion baked in at import time.
assert U_DIM == 39, f"U_DIM mismatch: expected 39, got {U_DIM}"


# ---------------------------------------------------------------------------
# Mock zone count
# ---------------------------------------------------------------------------

NUM_MOCK_ZONES: int = 6


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
    input_dim: int = U_DIM      # 39
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1        # only active when num_layers > 1
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
STREAM_A_ID: str = "stream_a_central_DEMO_FIXTURE"

MODEL_VERSION: str = "gru_forecaster_v1_mock"


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
