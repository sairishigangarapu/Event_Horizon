"""
sequence.py – Chronological history window assembly and target-step validation.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

Responsibilities:
  - Assemble the fixed GRU input U[z,t] for every zone at every timestep.
  - Enforce chronological ordering; reject scrambled / out-of-order steps.
  - Build rolling 60-step history tensors and compute target indices.
  - Enforce the strict rule: only observations received_at <= issue_time enter
    the model input; reference/ground-truth data is NEVER included.

U[z,t] layout (39 dims):
  [X(6) | D(6) | H(8) | age(6) | obs_mask(6) | tree_mask(6) | exit_mask(1)]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    AGE_NORM_SEC,
    D_DIM,
    EXIT_MASK_DIM,
    H_DIM,
    HISTORY_STEPS,
    HORIZONS_SEC,
    MISSING_AGE_SENTINEL,
    NUM_CHANNELS,
    OBS_MASK_DIM,
    OBS_STALE_THRESHOLD_SEC,
    SAMPLE_INTERVAL_SEC,
    TREE_MASK_DIM,
    U_DIM,
    VAR_NAMES,
    X_DIM,
)
from .synthetic import (
    MOCK_ZONE_IDS,
    MockObservation,
    MockScenario,
    MockTreeOutput,
    get_h_embedding,
    get_tree_output,
    is_exit_zone,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepSnapshot:
    """
    All assembled inputs for one zone at one timestep.

    Fields:
        zone_id:     Zone identifier.
        step:        Step index within the scenario (monotonically increasing).
        issue_time:  Wall-clock time at which the forecast would be issued.
        u_vec:       Full U[z,t] feature vector of shape (U_DIM,).
    """
    zone_id: str
    step: int
    issue_time: datetime
    u_vec: np.ndarray   # shape (U_DIM,)


@dataclass
class WindowBatch:
    """
    Rolling HISTORY_STEPS window for all zones at one issue_time.

    Fields:
        issue_time:      Forecast issue time.
        u_history:       Array of shape (num_zones, HISTORY_STEPS, U_DIM).
        zone_ids:        Ordered list of zone IDs (matches axis-0 of u_history).
        target_times:    Dict mapping horizon_sec -> target datetime.
    """
    issue_time: datetime
    u_history: np.ndarray        # (num_zones, HISTORY_STEPS, U_DIM)
    zone_ids: List[str]
    target_times: Dict[int, datetime]


# ---------------------------------------------------------------------------
# Feature assembly helpers
# ---------------------------------------------------------------------------

def _as_naive_utc(dt: datetime) -> datetime:
    from datetime import timezone
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _build_x_and_masks(
    zone_id: str,
    issue_time: datetime,
    observations: List[MockObservation],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X (raw values), age (normalised), obs_mask from the observation list.

    Only observations with received_at <= issue_time are considered.
    Latest observation per variable wins.

    Returns:
        x_vec    (X_DIM,)         – raw values (default-filled if missing)
        age_vec  (NUM_CHANNELS,)  – normalised ages in [0, 1]
        obs_mask (OBS_MASK_DIM,)  – 1 if fresh, 0 if stale / missing
    """
    t = _as_naive_utc(issue_time)
    var_idx = {v: i for i, v in enumerate(VAR_NAMES)}

    x_vec    = np.zeros(X_DIM, dtype=np.float32)
    ages_sec = np.full(NUM_CHANNELS, MISSING_AGE_SENTINEL, dtype=np.float32)
    obs_mask = np.zeros(OBS_MASK_DIM, dtype=np.float32)

    _defaults = [0.8, 1.2, 1.0, 1.0, 0.0, 1.0]  # density…exit_open

    for obs in observations:
        if obs.zone_id != zone_id:
            continue
        recv = _as_naive_utc(obs.received_at)
        if recv > t:
            continue   # future-time input guard
        if obs.variable not in var_idx:
            continue
        idx = var_idx[obs.variable]
        age = max(0.0, (t - _as_naive_utc(obs.observed_at)).total_seconds())
        if age < ages_sec[idx]:
            ages_sec[idx] = age
            x_vec[idx] = float(obs.value)
            obs_mask[idx] = 1.0 if age <= OBS_STALE_THRESHOLD_SEC else 0.0

    # Fill defaults for unobserved variables
    for i in range(NUM_CHANNELS):
        if obs_mask[i] == 0.0:
            x_vec[i] = _defaults[i]

    age_vec = np.clip(ages_sec / AGE_NORM_SEC, 0.0, 1.0).astype(np.float32)
    return x_vec, age_vec, obs_mask


def _build_d_and_tree_mask(
    tree_out: Optional[MockTreeOutput],
    zone_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build D vector and tree applicability mask from a MockTreeOutput record.

    If no tree output is available, returns zeros and all-zero masks.

    Note: D is passed directly to the GRU even though it also enters the
    hypernetwork dynamic context. This dual-path is an explicit architectural
    requirement (P0 rule).
    """
    d_vec      = np.zeros(D_DIM, dtype=np.float32)
    tree_mask  = np.zeros(TREE_MASK_DIM, dtype=np.float32)

    if tree_out is not None:
        for i in range(D_DIM):
            d_vec[i]     = float(tree_out.channel_values[i])
            tree_mask[i] = float(tree_out.channel_masks[i])

    return d_vec, tree_mask


def _build_exit_mask(zone_id: str) -> np.ndarray:
    """Returns a 1-element mask: 1.0 for exit zones, 0.0 otherwise."""
    return np.array([1.0 if is_exit_zone(zone_id) else 0.0], dtype=np.float32)


def assemble_u_vec(
    zone_id: str,
    issue_time: datetime,
    observations: List[MockObservation],
    tree_out: Optional[MockTreeOutput],
    h_embedding: np.ndarray,
) -> np.ndarray:
    """
    Assembles the full U[z,t] feature vector.

    Layout:
        X(6) | D(6) | H(8) | age(6) | obs_mask(6) | tree_mask(6) | exit_mask(1)
      = 39 dimensions.

    Args:
        zone_id:     Zone identifier.
        issue_time:  The forecast issue time (no future data allowed).
        observations: All available observations (will be filtered by received_at).
        tree_out:    Decision-tree output at or before issue_time (may be None).
        h_embedding: Spatial GNN embedding H[z,t] of shape (H_DIM,).

    Returns:
        u_vec of shape (U_DIM,) = (39,).
    """
    x_vec, age_vec, obs_mask = _build_x_and_masks(zone_id, issue_time, observations)
    d_vec, tree_mask          = _build_d_and_tree_mask(tree_out, zone_id)
    exit_mask                 = _build_exit_mask(zone_id)

    u_vec = np.concatenate([
        x_vec,      # 6
        d_vec,      # 6
        h_embedding,# 8
        age_vec,    # 6
        obs_mask,   # 6
        tree_mask,  # 6
        exit_mask,  # 1
    ]).astype(np.float32)  # 39

    assert u_vec.shape == (U_DIM,), (
        f"U_DIM mismatch: expected ({U_DIM},), got {u_vec.shape}"
    )
    return u_vec


# ---------------------------------------------------------------------------
# Rolling window assembler
# ---------------------------------------------------------------------------

class SequenceWindowAssembler:
    """
    Stateful assembler that maintains a per-zone rolling history buffer.

    Usage:
        assembler = SequenceWindowAssembler()
        ...
        for step, issue_time in enumerate(timeline):
            batch = assembler.step(scenario, issue_time, step)
            # batch.u_history -> (num_zones, HISTORY_STEPS, U_DIM)
    """

    def __init__(
        self,
        zone_ids: Optional[List[str]] = None,
        history_steps: int = HISTORY_STEPS,
        horizons_sec: Optional[List[int]] = None,
    ):
        self._zone_ids      = zone_ids or list(MOCK_ZONE_IDS)
        self._history_steps = history_steps
        self._horizons_sec  = horizons_sec or list(HORIZONS_SEC)

        # Rolling buffers: {zone_id: deque-like list of U vecs}
        self._buffer: Dict[str, List[np.ndarray]] = {
            z: [] for z in self._zone_ids
        }
        self._last_step: Optional[int] = None

    def reset(self) -> None:
        """Clears all history buffers (call between scenarios)."""
        for z in self._zone_ids:
            self._buffer[z].clear()
        self._last_step = None

    def _validate_ordering(self, step: int) -> None:
        """Raises ValueError if steps arrive out of order."""
        if self._last_step is not None and step <= self._last_step:
            raise ValueError(
                f"Out-of-order step detected: received step={step} "
                f"but last processed step was {self._last_step}. "
                "History buffer requires monotonically increasing steps."
            )

    def step(
        self,
        scenario: MockScenario,
        issue_time: datetime,
        step: int,
    ) -> WindowBatch:
        """
        Processes one timestep for the given scenario.

        Args:
            scenario:   The MockScenario providing observations and tree outputs.
            issue_time: Wall-clock time of this step.
            step:       Monotonic step index (must be strictly increasing).

        Returns:
            WindowBatch with u_history of shape (num_zones, HISTORY_STEPS, U_DIM).

        Raises:
            ValueError: If steps arrive out of chronological order.
        """
        self._validate_ordering(step)
        self._last_step = step

        zone_sequences: List[np.ndarray] = []

        for z_id in self._zone_ids:
            # Gather observations available at this issue time
            avail_obs = [
                o for o in scenario.observations
                if o.zone_id == z_id
                and _as_naive_utc(o.received_at) <= _as_naive_utc(issue_time)
            ]

            # Get latest tree output at or before issue_time
            tree_out = get_tree_output(scenario, z_id, issue_time)

            # Get spatial embedding for this (zone, step) — mock
            h_emb = get_h_embedding(scenario, z_id, step)

            # Assemble current U[z,t]
            u_vec = assemble_u_vec(z_id, issue_time, avail_obs, tree_out, h_emb)

            # Append to rolling buffer
            self._buffer[z_id].append(u_vec)
            if len(self._buffer[z_id]) > self._history_steps:
                self._buffer[z_id].pop(0)

            # Build padded history window
            hist = self._buffer[z_id]
            if len(hist) < self._history_steps:
                # Left-pad with the earliest available vector
                pad_vec = hist[0]
                pad_count = self._history_steps - len(hist)
                padded = [pad_vec] * pad_count + hist
            else:
                padded = hist[-self._history_steps:]

            seq = np.stack(padded, axis=0).astype(np.float32)  # (HISTORY_STEPS, U_DIM)
            zone_sequences.append(seq)

        u_history = np.stack(zone_sequences, axis=0)  # (num_zones, HISTORY_STEPS, U_DIM)
        assert u_history.shape == (
            len(self._zone_ids), self._history_steps, U_DIM
        ), f"Unexpected u_history shape: {u_history.shape}"

        target_times = {
            h: issue_time + timedelta(seconds=h)
            for h in self._horizons_sec
        }

        return WindowBatch(
            issue_time=issue_time,
            u_history=u_history,
            zone_ids=list(self._zone_ids),
            target_times=target_times,
        )


def build_all_windows(
    scenario: MockScenario,
    history_steps: int = HISTORY_STEPS,
    horizons_sec: Optional[List[int]] = None,
) -> List[WindowBatch]:
    """
    Convenience wrapper that assembles every window in a scenario.

    Only steps where at least (history_steps) prior steps are available
    will produce usable history (earlier steps use left-padding).

    Returns:
        List of WindowBatch objects, one per timestep.
    """
    assembler = SequenceWindowAssembler(
        history_steps=history_steps,
        horizons_sec=horizons_sec,
    )
    assembler.reset()

    # Derive timeline from tree outputs (they track all steps)
    step_times = sorted(
        {t.evaluation_time for t in scenario.tree_outputs}
    )

    batches: List[WindowBatch] = []
    for step_idx, ts in enumerate(step_times):
        batch = assembler.step(scenario, ts, step_idx)
        batches.append(batch)

    return batches
