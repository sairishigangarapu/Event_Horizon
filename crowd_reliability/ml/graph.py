import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any

class SpatialGraphBuilder:
    """
    Constructs normalized adjacency and edge feature tensors from venue layout.
    """
    def __init__(self, zones: List[Dict[str, Any]], connections: List[Dict[str, Any]]):
        self.zones = zones
        self.connections = connections
        self.zone_ids = [z["zone_id"] for z in zones]
        self.zone_to_idx = {z_id: i for i, z_id in enumerate(self.zone_ids)}
        self.num_nodes = len(self.zone_ids)

        # Build edge index and edge attributes
        edge_indices = []
        edge_attrs = []

        for conn in connections:
            u = self.zone_to_idx[conn["from_zone"]]
            v = self.zone_to_idx[conn["to_zone"]]
            width = float(conn.get("width_m", 3.0))
            dist = float(conn.get("distance_m", 25.0))
            edge_indices.append([u, v])
            edge_attrs.append([width / 5.0, dist / 50.0])

            # If bidirectional, also add reverse
            if conn.get("direction") == "bidirectional":
                edge_indices.append([v, u])
                edge_attrs.append([width / 5.0, dist / 50.0])

        if len(edge_indices) > 0:
            self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_attr = torch.zeros((0, 2), dtype=torch.float32)

class EdgeAwareGNNLayer(nn.Module):
    """
    Edge-aware graph message passing layer:
    Aggregates neighbor features weighted by edge geometry (width, distance).
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.self_linear = nn.Linear(node_in_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, node_in_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_dim]
        num_nodes = x.size(0)
        self_feat = self.self_linear(x)

        if edge_index.size(1) == 0:
            return self.relu(self.out_norm(self_feat))

        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Aggregate messages into destination nodes (sum pooling)
        aggregated = torch.zeros((num_nodes, messages.size(-1)), dtype=x.dtype, device=x.device)
        aggregated.index_add_(0, dst, messages)

        out = self_feat + aggregated
        return self.relu(self.out_norm(out))

class SpatialGNN(nn.Module):
    """
    Two-layer edge-aware Spatial GNN encoding inter-zone pedestrian flow relationships.
    Computes H[t] spatial context per zone.
    """
    def __init__(self, node_in_dim: int = 18, edge_in_dim: int = 2, hidden_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.layer1 = EdgeAwareGNNLayer(node_in_dim, edge_in_dim, hidden_dim)
        self.layer2 = EdgeAwareGNNLayer(hidden_dim, edge_in_dim, embed_dim)
        self.embed_dim = embed_dim
        self.version = "gnn_edge_v1"

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x, edge_index, edge_attr)
        h2 = self.layer2(h1, edge_index, edge_attr)
        return h2  # [num_nodes, embed_dim]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "version": self.version}, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
        self.version = checkpoint.get("version", self.version)
