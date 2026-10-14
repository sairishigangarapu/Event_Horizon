"""
synthetic.py – Reproducible synthetic scenarios for Person 4 smoke tests.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

Generates:
  - Mock observation snapshots (as-of report snapshots) with realistic dynamics.
  - Mock tree estimates (D vectors) per zone per timestep.
  - Mock spatial GNN embeddings (H vectors) per zone.
  - Mock reference states with label_available_at delay.

All values are seeded and deterministic. Replace with real producer interfaces
(Person 1 observation API, Person 3 GNN/tree output) before interpreting
any model performance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    D_DIM,
    EXIT_ZONE_IDS,
    H_DIM,
    HISTORY_STEPS,
    HORIZONS_SEC,
    MISSING_AGE_SENTINEL,
    NUM_CHANNELS,
    NUM_MOCK_ZONES,
    OBS_STALE_THRESHOLD_SEC,
    SAMPLE_INTERVAL_SEC,
    VAR_NAMES,
    VAR_UNITS,
    VENUE_ZONE_IDS,
)

# ---------------------------------------------------------------------------
# Zone definitions (reconciled with Person 1 venue)
# ---------------------------------------------------------------------------

MOCK_ZONE_IDS: List[str] = list(VENUE_ZONE_IDS)

def is_exit_zone(zone_id: str) -> bool:
    return zone_id in EXIT_ZONE_IDS


# ---------------------------------------------------------------------------
# Dataclasses mirroring the schema contract
# ---------------------------------------------------------------------------

@dataclass
class MockObservation:
    """Minimal observation record compatible with Person 1 schema."""
    report_id: str
    scenario_id: str
    zone_id: str
    source_id: str
    variable: str
    value: float
    unit: str
    observed_at: datetime
    published_at: datetime
    received_at: datetime


@dataclass
class MockTreeOutput:
    """D[z,t]: decision-tree channel vector per zone per step."""
    scenario_id: str
    zone_id: str
    evaluation_time: datetime
    tree_version: str
    channel_names: List[str]
    channel_values: List[float]
    channel_masks: List[int]


@dataclass
class MockReferenceState:
    """Ground-truth label record, available only after label_available_at."""
    scenario_id: str
    zone_id: str
    variable: str
    value: float
    unit: str
    state_at: datetime
    label_available_at: datetime   # enforced delay; never use before this time


@dataclass
class MockScenario:
    """Container for one synthetic scenario's raw data."""
    scenario_id: str
    split: str   # "train", "val", or "test"
    start_time: datetime
    observations: List[MockObservation] = field(default_factory=list)
    tree_outputs: List[MockTreeOutput] = field(default_factory=list)
    reference_states: List[MockReferenceState] = field(default_factory=list)
    spatial_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthetic dynamics helpers
# ---------------------------------------------------------------------------

_TREE_CHANNEL_NAMES: List[str] = [
    "estimated_density",
    "estimated_speed",
    "estimated_inflow",
    "estimated_outflow",
    "estimated_queue_length",
    "probability_exit_open",
]

assert len(_TREE_CHANNEL_NAMES) == D_DIM


def _zone_phase(zone_idx: int) -> float:
    """Gives each zone a distinct phase offset for synthetic oscillations."""
    return zone_idx * (2 * math.pi / NUM_MOCK_ZONES)


def _synthetic_obs_value(
    var_idx: int,
    zone_idx: int,
    step: int,
    rng: random.Random,
) -> float:
    """
    Returns a plausible synthetic observation value for (variable, zone, step).

    Dynamics: slow sinusoidal base + small Gaussian noise.
    Each variable has its own amplitude/offset tuned to realistic physical ranges.
    """
    t = step * SAMPLE_INTERVAL_SEC
    phase = _zone_phase(zone_idx)

    # (mean, amplitude, noise_std)
    specs = [
        (1.5,  0.8, 0.05),   # density  [p/m²]
        (1.2,  0.4, 0.03),   # speed    [m/s]
        (2.0,  1.0, 0.10),   # inflow   [p/s]
        (1.8,  0.9, 0.10),   # outflow  [p/s]
        (3.0,  2.0, 0.20),   # queue    [persons]
        (1.0,  0.0, 0.0),    # exit_open binary (always open in mock base)
    ]
    mean, amp, noise_std = specs[var_idx]
    val = mean + amp * math.sin(2 * math.pi * t / 120.0 + phase)
    val += rng.gauss(0.0, noise_std)

    # Clamp physical plausibility
    if var_idx == 5:  # exit_open: binary
        val = 1.0 if rng.random() > 0.1 else 0.0
    else:
        val = max(0.0, val)
    return round(val, 4)


def _synthetic_tree_values(
    zone_idx: int,
    step: int,
    obs_values: List[float],
    rng: random.Random,
    zone_id: str,
) -> Tuple[List[float], List[int]]:
    """
    Mock D[z,t]: tree vector generated from visible obs_values with small noise.
    Masks: 0 for exit_risk on non-exit zones (inapplicable).
    """
    vals = []
    for k in range(D_DIM):
        if k < NUM_CHANNELS:
            v = obs_values[k] * (1.0 + rng.gauss(0.0, 0.02))
            v = max(0.0, v)
        else:
            v = rng.gauss(0.5, 0.1)
        vals.append(round(v, 4))

    masks = [1] * D_DIM
    if not is_exit_zone(zone_id):
        masks[D_DIM - 1] = 0   # est_exit_risk inapplicable for non-exit zones
    return vals, masks


def _synthetic_h_embedding(zone_idx: int, step: int, rng: random.Random) -> np.ndarray:
    """
    Mock spatial GNN embedding H[z,t] — H_DIM dimensions.
    MOCK: Replace with Person 3 GNN output before interpreting performance.
    """
    np_rng = np.random.default_rng(seed=zone_idx * 1000 + step)
    base = np_rng.normal(0.0, 0.5, size=H_DIM).astype(np.float32)
    jitter = np.array([rng.gauss(0.0, 0.01) for _ in range(H_DIM)], dtype=np.float32)
    return base + jitter


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

_SCENARIO_SPLITS: List[Tuple[str, str]] = [
    ("sc_train_01", "train"),
    ("sc_train_02", "train"),
    ("sc_train_03", "train"),
    ("sc_val_01",   "val"),
    ("sc_test_01",  "test"),
]

#: Delay before reference labels become available (simulates post-event reporting).
LABEL_DELAY_SEC: float = 60.0


def generate_scenario(
    scenario_id: str,
    split: str,
    num_steps: int,
    start_time: Optional[datetime] = None,
    seed: int = 0,
) -> MockScenario:
    """
    Generates one fully reproducible synthetic scenario.

    Args:
        scenario_id: Unique scenario identifier.
        split:       One of 'train', 'val', 'test'.
        num_steps:   Number of 5-second sample steps to generate.
        start_time:  Wall-clock anchor (defaults to a fixed UTC epoch).
        seed:        RNG seed for reproducibility.

    Returns:
        MockScenario containing observations, tree outputs, reference states
        and spatial embeddings for all zones.
    """
    if start_time is None:
        start_time = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)

    rng = random.Random(seed)
    scenario = MockScenario(
        scenario_id=scenario_id,
        split=split,
        start_time=start_time,
    )

    for step in range(num_steps):
        step_time = start_time + timedelta(seconds=step * SAMPLE_INTERVAL_SEC)
        # Observations published with a small random latency (0–3 s)
        latency = timedelta(seconds=rng.uniform(0.0, 3.0))

        for z_idx, zone_id in enumerate(MOCK_ZONE_IDS):
            # ---- observations ----
            obs_values: List[float] = []
            for v_idx, var in enumerate(VAR_NAMES):
                obs_val = _synthetic_obs_value(v_idx, z_idx, step, rng)
                obs_values.append(obs_val)
                obs = MockObservation(
                    report_id=f"{scenario_id}_{zone_id}_{var}_{step:04d}",
                    scenario_id=scenario_id,
                    zone_id=zone_id,
                    source_id=f"sensor_{zone_id}",
                    variable=var,
                    value=obs_val,
                    unit=VAR_UNITS[var],
                    observed_at=step_time,
                    published_at=step_time + latency,
                    received_at=step_time + latency,
                )
                scenario.observations.append(obs)

            # ---- tree output ----
            d_vals, d_masks = _synthetic_tree_values(
                z_idx, step, obs_values, rng, zone_id
            )
            tree_out = MockTreeOutput(
                scenario_id=scenario_id,
                zone_id=zone_id,
                evaluation_time=step_time,
                tree_version="tree_mock_v1",
                channel_names=_TREE_CHANNEL_NAMES,
                channel_values=d_vals,
                channel_masks=d_masks,
            )
            scenario.tree_outputs.append(tree_out)

            # ---- spatial embedding (static per zone per step) ----
            h_key = f"{zone_id}_{step}"
            scenario.spatial_embeddings[h_key] = _synthetic_h_embedding(
                z_idx, step, rng
            )

        # ---- reference states (one per zone × variable, with label delay) ----
        ref_time = step_time
        label_available = ref_time + timedelta(seconds=LABEL_DELAY_SEC)
        for z_idx, zone_id in enumerate(MOCK_ZONE_IDS):
            for v_idx, var in enumerate(VAR_NAMES):
                if var == "exit_open" and not is_exit_zone(zone_id):
                    continue   # exit_open only exists for exit zones
                ref_val = _synthetic_obs_value(v_idx, z_idx, step, rng)
                ref = MockReferenceState(
                    scenario_id=scenario_id,
                    zone_id=zone_id,
                    variable=var,
                    value=ref_val,
                    unit=VAR_UNITS[var],
                    state_at=ref_time,
                    label_available_at=label_available,
                )
                scenario.reference_states.append(ref)

    return scenario


def generate_all_scenarios(
    steps_per_scenario: int = HISTORY_STEPS + max(HORIZONS_SEC) // SAMPLE_INTERVAL_SEC + 5,
) -> List[MockScenario]:
    """
    Generates the full standard suite of synthetic scenarios.

    Uses the SCENARIO_SPLITS list; each scenario gets a deterministic seed
    derived from its index so that results are reproducible across runs.

    Returns:
        List of MockScenario objects ready for training and evaluation.
    """
    scenarios: List[MockScenario] = []
    base_time = datetime(2026, 10, 1, 8, 0, 0, tzinfo=timezone.utc)
    for i, (sc_id, split) in enumerate(_SCENARIO_SPLITS):
        sc_start = base_time + timedelta(hours=i)
        sc = generate_scenario(
            scenario_id=sc_id,
            split=split,
            num_steps=steps_per_scenario,
            start_time=sc_start,
            seed=i * 137,
        )
        scenarios.append(sc)
    return scenarios


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_tree_output(
    scenario: MockScenario,
    zone_id: str,
    evaluation_time: datetime,
) -> Optional[MockTreeOutput]:
    """Returns the tree output for the closest timestep at or before evaluation_time."""
    candidates = [
        t for t in scenario.tree_outputs
        if t.zone_id == zone_id and t.evaluation_time <= evaluation_time
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.evaluation_time)


def get_h_embedding(
    scenario: MockScenario,
    zone_id: str,
    step: int,
) -> np.ndarray:
    """Returns the cached spatial embedding for (zone_id, step), or zeros."""
    key = f"{zone_id}_{step}"
    if key in scenario.spatial_embeddings:
        return scenario.spatial_embeddings[key]
    return np.zeros(H_DIM, dtype=np.float32)
