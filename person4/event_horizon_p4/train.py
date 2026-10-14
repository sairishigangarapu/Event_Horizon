"""
train.py – Synthetic-only smoke-test training and checkpoint saving.

DISCLAIMER: Synthetic scaffold only. NOT for real-world use.

Usage (CLI):
    python -m event_horizon_p4.train --epochs 2 --output artifacts/temporal_mock.pt

Training loop:
  1. Generate synthetic scenarios (train split).
  2. Build sequence windows per scenario.
  3. Fit a StandardScaler on training-split numeric inputs.
  4. Run GRU forward pass with MaskedMixedLoss.
  5. Save checkpoint + scaler + manifest JSON.

Notes:
  - Training is done jointly on all training scenarios to validate that the
    pipeline code path executes end-to-end (not to learn real distributions).
  - Model and scaler checkpoints are explicitly labelled
    synthetic_fixture_only_NOT_VALIDATED.
  - To coordinate with Person 3 (tree/GNN training), the scenario split IDs
    produced here should match the held-out split used in joint training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from .config import (
    EXIT_MASK_DIM,
    HISTORY_STEPS,
    HORIZONS_SEC,
    NUM_CHANNELS,
    NUM_MOCK_ZONES,
    NUM_NUMERIC_CHANNELS,
    SAMPLE_INTERVAL_SEC,
    TrainConfig,
    U_DIM,
    VAR_NAMES,
)
from .model import MaskedMixedLoss, TemporalGRUForecaster
from .sequence import WindowBatch, build_all_windows
from .synthetic import (
    MOCK_ZONE_IDS,
    generate_all_scenarios,
    is_exit_zone,
    MockScenario,
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _build_targets_for_batch(
    batch: WindowBatch,
    scenario: MockScenario,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build target tensors for one WindowBatch.

    Returns:
        targets    (num_zones, num_horizons, 6)
        obs_mask   (num_zones, 5)   — numeric channel masks from last step
        exit_mask  (num_zones,)     — 1 for exit zones

    Reference states are used ONLY here (training targets), never inside
    forecast_window or inference.
    """
    num_zones    = len(batch.zone_ids)
    num_horizons = len(HORIZONS_SEC)

    targets   = np.zeros((num_zones, num_horizons, NUM_CHANNELS), dtype=np.float32)
    obs_mask  = np.zeros((num_zones, NUM_NUMERIC_CHANNELS), dtype=np.float32)
    exit_mask = np.zeros(num_zones, dtype=np.float32)

    # Build reference lookup: (zone, variable, target_time) → value
    ref_lookup: Dict[Tuple[str, str, datetime], float] = {}
    for ref in scenario.reference_states:
        key = (ref.zone_id, ref.variable, ref.state_at)
        ref_lookup[key] = ref.value

    for z_idx, z_id in enumerate(batch.zone_ids):
        exit_mask[z_idx] = 1.0 if is_exit_zone(z_id) else 0.0

        # obs_mask from last U vec (layout: X(6) + age(6) + obs_mask(6) + D(6) + H(16))
        last_u = batch.u_history[z_idx, -1, :]
        obs_mask_start = 6 + 6  # X + age = 12
        obs_mask[z_idx] = last_u[obs_mask_start: obs_mask_start + NUM_NUMERIC_CHANNELS]

        for h_idx, h_sec in enumerate(HORIZONS_SEC):
            tgt_time = batch.target_times[h_sec]
            for v_idx, var in enumerate(VAR_NAMES):
                if var == "exit_open" and not is_exit_zone(z_id):
                    continue
                key = (z_id, var, tgt_time)
                if key in ref_lookup:
                    targets[z_idx, h_idx, v_idx] = ref_lookup[key]

    return targets, obs_mask, exit_mask


def _collect_training_data(
    scenarios: List[MockScenario],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Collects (u_history, targets, obs_mask, exit_mask) for all training scenarios.

    Returns:
        u_histories   – list of (num_zones, HISTORY_STEPS, U_DIM) arrays
        all_targets   – list of (num_zones, num_horizons, 6) arrays
        all_obs_masks – list of (num_zones, 5) arrays
        all_exit_masks – list of (num_zones,) arrays
    """
    u_histories: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_obs_masks: List[np.ndarray] = []
    all_exit_masks: List[np.ndarray] = []

    train_scenarios = [s for s in scenarios if s.split == "train"]
    print(f"[train] Collecting windows from {len(train_scenarios)} training scenarios...")

    for sc in train_scenarios:
        batches = build_all_windows(sc)
        # Skip early steps without enough context
        skip = max(0, HISTORY_STEPS - len(batches) // 2)
        for batch in batches[skip:]:
            targets, obs_mask, exit_mask = _build_targets_for_batch(batch, sc)
            u_histories.append(batch.u_history)
            all_targets.append(targets)
            all_obs_masks.append(obs_mask)
            all_exit_masks.append(exit_mask)

    print(f"[train] Collected {len(u_histories)} windows total.")
    return u_histories, all_targets, all_obs_masks, all_exit_masks


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def fit_scaler(u_histories: List[np.ndarray]) -> StandardScaler:
    """
    Fits a StandardScaler on the numeric (first X_DIM=6) features of U.
    Only trained on training-split histories. Scaler is NOT applied to
    masks or embeddings — those remain in their natural range.
    """
    # Flatten: (N, Z, T, U_DIM) → (N*Z*T, U_DIM)
    flat = np.concatenate(
        [h.reshape(-1, U_DIM) for h in u_histories], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(flat[:, :6])   # fit on X channels only
    return scaler


def apply_scaler(u_histories: List[np.ndarray], scaler: StandardScaler) -> List[np.ndarray]:
    """Applies scaler to X channels of each history array."""
    scaled = []
    for h in u_histories:
        h_copy = h.copy()
        orig_shape = h_copy.shape
        flat = h_copy.reshape(-1, U_DIM)
        flat[:, :6] = scaler.transform(flat[:, :6])
        scaled.append(flat.reshape(orig_shape))
    return scaled


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig = None,
    output_path: str = "artifacts/temporal_mock.pt",
    epochs: int = None,
) -> Tuple[TemporalGRUForecaster, StandardScaler, Dict]:
    """
    Runs the full synthetic training loop.

    Args:
        cfg:         TrainConfig (uses defaults if None).
        output_path: Path to save the model checkpoint.
        epochs:      Override cfg.epochs if provided.

    Returns:
        (model, scaler, manifest_dict)
    """
    cfg = cfg or TrainConfig()
    if epochs is not None:
        cfg.epochs = epochs

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # 1. Generate data
    print("[train] Generating synthetic scenarios...")
    scenarios = generate_all_scenarios()

    # 2. Collect windows
    u_histories, all_targets, all_obs_masks, all_exit_masks = _collect_training_data(scenarios)

    if not u_histories:
        raise RuntimeError("No training windows collected — check scenario generation.")

    # 3. Fit scaler on training data
    print("[train] Fitting StandardScaler on X channels...")
    scaler = fit_scaler(u_histories)
    scaled_histories = apply_scaler(u_histories, scaler)

    # 4. Build model
    model = TemporalGRUForecaster()
    model.train()
    criterion = MaskedMixedLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 5. Training loop
    n = len(scaled_histories)
    print(f"[train] Starting training: {cfg.epochs} epoch(s), {n} windows, "
          f"batch_size={cfg.batch_size} ...")

    epoch_losses: List[float] = []
    for epoch in range(cfg.epochs):
        indices = list(range(n))
        random.shuffle(indices)
        batch_losses: List[float] = []

        for batch_start in range(0, n, cfg.batch_size):
            batch_idx = indices[batch_start: batch_start + cfg.batch_size]

            u_batch   = torch.tensor(
                np.stack([scaled_histories[i] for i in batch_idx]), dtype=torch.float32
            )   # (B, Z, T, U_DIM) — but model expects (Z*B, T, U_DIM) or (Z, T, U_DIM)
            # We treat each (zone, window) pair independently:
            # Flatten to (B*Z, T, U_DIM)
            B, Z = u_batch.shape[0], u_batch.shape[1]
            u_flat = u_batch.view(B * Z, HISTORY_STEPS, U_DIM)

            t_batch  = torch.tensor(
                np.stack([all_targets[i] for i in batch_idx]), dtype=torch.float32
            ).view(B * Z, len(HORIZONS_SEC), NUM_CHANNELS)

            om_batch = torch.tensor(
                np.stack([all_obs_masks[i] for i in batch_idx]), dtype=torch.float32
            ).view(B * Z, NUM_NUMERIC_CHANNELS)

            em_batch = torch.tensor(
                np.stack([all_exit_masks[i] for i in batch_idx]), dtype=torch.float32
            ).view(B * Z)

            optimizer.zero_grad()
            preds = model(u_flat)   # (B*Z, H, 6)
            loss  = criterion(preds, t_batch, om_batch, em_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = float(np.mean(batch_losses))
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{cfg.epochs}  loss={avg_loss:.4f}")

    # 6. Save checkpoint
    print(f"[train] Saving checkpoint to {output_path} ...")
    manifest = {
        "fixture_label":    cfg.fixture_label,
        "model_version":    model.version,
        "u_dim":            U_DIM,
        "history_steps":    HISTORY_STEPS,
        "horizons_sec":     HORIZONS_SEC,
        "num_zones":        NUM_MOCK_ZONES,
        "zone_ids":         MOCK_ZONE_IDS,
        "var_names":        VAR_NAMES,
        "train_epochs":     cfg.epochs,
        "final_epoch_loss": epoch_losses[-1] if epoch_losses else None,
        "trained_at":       datetime.now(tz=timezone.utc).isoformat(),
        "WARNING":          "SYNTHETIC ONLY — NOT FOR REAL DECISIONS",
    }
    model.save(output_path, extra_meta={"manifest": manifest})

    # Save scaler
    scaler_path = output_path.replace(".pt", "_scaler.npy")
    np.save(scaler_path, {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    })

    # Save manifest JSON
    manifest_path = output_path.replace(".pt", "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[train] Done. Checkpoint: {output_path}")
    print(f"[train] Manifest:   {manifest_path}")
    print(f"[train] Scaler:     {scaler_path}")
    model.eval()
    return model, scaler, manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Person 4: synthetic-only smoke-test training."
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs (default: 2)."
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/temporal_mock.pt",
        help="Checkpoint output path."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate."
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cfg = TrainConfig(learning_rate=args.lr)
    train(cfg=cfg, output_path=args.output, epochs=args.epochs)


if __name__ == "__main__":
    _cli()
