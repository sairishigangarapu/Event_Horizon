import os
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any

TARGET_VAR_NAMES = [
    "density",
    "speed",
    "inflow",
    "outflow",
    "queue_length",
    "exit_open",
]

class MultivariateTemporalForecaster(nn.Module):
    """
    Multivariate GRU temporal forecaster.
    Takes history sequence of U[z, t-L+1 : t] which includes raw observations,
    full decision-tree vector D[z, t], spatial GNN embedding H[z, t], age, and masks.

    Outputs multi-horizon predictions:
    [num_zones, num_horizons, num_variables]
    with separate numeric heads (positive ReLU/Linear) and exit probability head (Sigmoid).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_horizons: int = 3,
        horizons_sec: List[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_horizons = num_horizons
        self.horizons_sec = horizons_sec or [30, 60, 120]
        self.version = "gru_forecaster_v1"

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 5 numeric heads (density, speed, inflow, outflow, queue_length)
        # 1 probability head (exit_open)
        # For each horizon, we have a specialized projection head
        self.numeric_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 5),  # 5 numeric variables
                nn.ReLU()          # physical non-negativity constraint
            ) for _ in range(num_horizons)
        ])

        self.exit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()       # calibrated probability
            ) for _ in range(num_horizons)
        ])

    def forward(self, u_history: torch.Tensor) -> torch.Tensor:
        """
        u_history: [num_zones, seq_len, input_dim]
        Returns:
            forecasts: [num_zones, num_horizons, 6]
        """
        # gru output: [num_zones, seq_len, hidden_dim]
        out, _ = self.gru(u_history)
        h_last = out[:, -1, :]  # [num_zones, hidden_dim]

        horizon_outputs = []
        for h_idx in range(self.num_horizons):
            num_preds = self.numeric_heads[h_idx](h_last)  # [num_zones, 5]
            exit_prob = self.exit_heads[h_idx](h_last)     # [num_zones, 1]
            # Concatenate along variable dimension: [num_zones, 6]
            combined = torch.cat([num_preds, exit_prob], dim=-1)
            horizon_outputs.append(combined.unsqueeze(1))  # [num_zones, 1, 6]

        return torch.cat(horizon_outputs, dim=1)  # [num_zones, num_horizons, 6]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_horizons": self.num_horizons,
            "horizons_sec": self.horizons_sec,
            "version": self.version
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MultivariateTemporalForecaster":
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            num_horizons=checkpoint["num_horizons"],
            horizons_sec=checkpoint["horizons_sec"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.version = checkpoint.get("version", model.version)
        return model
