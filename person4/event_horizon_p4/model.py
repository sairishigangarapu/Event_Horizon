"""
model.py – GRU temporal forecaster with six output heads per zone/horizon.

NOTE: This module provides temporal forecasting models; architectural
dimensions (U_DIM=40, H_DIM=16, HISTORY_STEPS=12) are actively reconciled
with the production ArchitecturePipeline.

Architecture:
  Input  : U[z,t] ∈ ℝ^(U_DIM=40), history length = HISTORY_STEPS = 12 (60 s look-back).
  GRU    : 2-layer bidirectional-disabled GRU, hidden_dim=64.
  Output : [Z, horizons, 6] where
             channels 0-4 → numeric heads (density, speed, inflow, outflow, queue)
             channel  5   → exit-open logit (raw, sigmoid applied at inference time)

Key rules enforced here:
  - No Sigmoid applied inside forward(); inference.py applies it + scale reversal.
  - Exit applicability mask is consumed by inference.py, not here.
  - The model holds NO reference to external data sources.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import (
    H_DIM,
    HISTORY_STEPS,
    HORIZONS_SEC,
    NUM_CHANNELS,
    NUM_NUMERIC_CHANNELS,
    U_DIM,
    VAR_NAMES,
    GRUConfig,
)


# ---------------------------------------------------------------------------
# Output metadata
# ---------------------------------------------------------------------------

#: Variable names in model output order.
OUTPUT_VAR_NAMES: List[str] = VAR_NAMES

#: Indices of numeric regression outputs.
NUMERIC_INDICES: List[int] = list(range(NUM_NUMERIC_CHANNELS))   # [0,1,2,3,4]

#: Index of exit-open logit.
EXIT_LOGIT_IDX: int = NUM_NUMERIC_CHANNELS                        # 5


# ---------------------------------------------------------------------------
# Per-horizon output head
# ---------------------------------------------------------------------------

class _HorizonHead(nn.Module):
    """
    One output head per prediction horizon.

    Numeric branch:  Linear → ReLU → Linear → ReLU  (non-negativity prior)
    Exit branch:     Linear → ReLU → Linear           (raw logit; no Sigmoid)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.numeric_branch = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_NUMERIC_CHANNELS),
            nn.ReLU(),   # physical non-negativity
        )
        self.exit_branch = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # No activation — raw logit returned; inference.py applies Sigmoid
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_last: (num_zones, hidden_dim)
        Returns:
            (num_zones, 6)  — 5 numeric + 1 exit logit
        """
        num_out  = self.numeric_branch(h_last)   # (Z, 5)
        exit_out = self.exit_branch(h_last)       # (Z, 1)
        return torch.cat([num_out, exit_out], dim=-1)   # (Z, 6)


# ---------------------------------------------------------------------------
# Main GRU model
# ---------------------------------------------------------------------------

class TemporalGRUForecaster(nn.Module):
    """
    Multivariate GRU temporal forecaster.

    Input:  U history  →  (num_zones, seq_len, U_DIM)
    Output: forecasts  →  (num_zones, num_horizons, 6)
              [zone, horizon, variable]
              channels 0-4: scaled regression (ReLU non-negative)
              channel  5  : raw exit logit (sigmoid at inference)

    One independent _HorizonHead is instantiated per horizon so that each
    horizon can learn a distinct projection from the shared GRU encoding.
    """

    def __init__(self, cfg: Optional[GRUConfig] = None):
        super().__init__()
        cfg = cfg or GRUConfig()

        self.input_dim    = cfg.input_dim
        self.hidden_dim   = cfg.hidden_dim
        self.num_layers   = cfg.num_layers
        self.num_horizons = cfg.num_horizons
        self.horizons_sec = list(cfg.horizons_sec)
        self.version      = cfg.version

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=cfg.dropout if self.num_layers > 1 else 0.0,
        )

        self.heads = nn.ModuleList([
            _HorizonHead(self.hidden_dim) for _ in range(self.num_horizons)
        ])

    def forward(self, u_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_history: (num_zones, seq_len, input_dim)

        Returns:
            forecasts: (num_zones, num_horizons, 6)
        """
        # gru_out: (Z, T, hidden_dim)
        gru_out, _ = self.gru(u_history)
        h_last = gru_out[:, -1, :]   # (Z, hidden_dim)

        horizon_preds = [
            self.heads[k](h_last).unsqueeze(1)   # (Z, 1, 6)
            for k in range(self.num_horizons)
        ]
        return torch.cat(horizon_preds, dim=1)   # (Z, H, 6)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, filepath: str, extra_meta: Optional[Dict] = None) -> None:
        """Saves checkpoint with config and optional extra metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "state_dict":   self.state_dict(),
            "input_dim":    self.input_dim,
            "hidden_dim":   self.hidden_dim,
            "num_layers":   self.num_layers,
            "num_horizons": self.num_horizons,
            "horizons_sec": self.horizons_sec,
            "h_dim":        extra_meta.get("h_dim", H_DIM) if extra_meta else H_DIM,
            "version":      self.version,
            **(extra_meta or {}),
        }
        torch.save(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TemporalGRUForecaster":
        """Loads checkpoint and reconstructs model."""
        ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
        cfg = GRUConfig(
            input_dim    = ckpt["input_dim"],
            hidden_dim   = ckpt["hidden_dim"],
            num_layers   = ckpt["num_layers"],
            num_horizons = ckpt["num_horizons"],
            horizons_sec = ckpt["horizons_sec"],
            version      = ckpt.get("version", "gru_forecaster_v1"),
        )
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Mixed loss (regression + binary classification)
# ---------------------------------------------------------------------------

class MaskedMixedLoss(nn.Module):
    """
    Per-variable masked loss combining:
      - MSE loss on 5 numeric channels (masked by obs_mask)
      - BCEWithLogits loss on exit-open channel (masked by exit_mask)

    Masking ensures that:
      - Stale/missing numeric observations do not contribute gradient.
      - Non-exit zones do not contribute to exit-open loss.

    Args:
        numeric_weight: Weight for the numeric MSE component.
        exit_weight:    Weight for the exit BCE component.
    """

    def __init__(self, numeric_weight: float = 1.0, exit_weight: float = 1.0):
        super().__init__()
        self.numeric_weight = numeric_weight
        self.exit_weight    = exit_weight
        self.mse_loss       = nn.MSELoss(reduction="none")
        self.bce_loss       = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        preds:      torch.Tensor,   # (Z, H, 6)
        targets:    torch.Tensor,   # (Z, H, 6)
        obs_mask:   torch.Tensor,   # (Z, 5)   — numeric channel masks
        exit_mask:  torch.Tensor,   # (Z,)     — 1 for exit zones
    ) -> torch.Tensor:
        """
        Returns scalar loss averaged over valid (zone, horizon, variable) entries.
        """
        # ---- numeric channels ----
        num_preds   = preds[..., :NUM_NUMERIC_CHANNELS]    # (Z, H, 5)
        num_targets = targets[..., :NUM_NUMERIC_CHANNELS]  # (Z, H, 5)
        num_loss_raw = self.mse_loss(num_preds, num_targets)  # (Z, H, 5)

        # Expand obs_mask to (Z, H, 5)
        obs_mask_exp = obs_mask.unsqueeze(1).expand_as(num_loss_raw)
        num_loss = (num_loss_raw * obs_mask_exp).sum() / (obs_mask_exp.sum() + 1e-8)

        # ---- exit channel ----
        exit_logit  = preds[..., EXIT_LOGIT_IDX]      # (Z, H)
        exit_target = targets[..., EXIT_LOGIT_IDX]    # (Z, H)
        exit_loss_raw = self.bce_loss(exit_logit, exit_target)  # (Z, H)

        exit_mask_exp = exit_mask.unsqueeze(1).expand_as(exit_loss_raw)
        exit_loss = (exit_loss_raw * exit_mask_exp).sum() / (exit_mask_exp.sum() + 1e-8)

        return self.numeric_weight * num_loss + self.exit_weight * exit_loss
