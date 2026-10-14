import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any

ZONE_TYPES = ["entrance", "corridor", "assembly", "stairway", "exit"]

class SpatialGraphBuilder:
    """
    Constructs normalized adjacency and edge feature tensors from venue layout.
    Also exposes static node descriptors consumed by the GraphHyperNetwork.
    """
    def __init__(self, zones: List[Dict[str, Any]], connections: List[Dict[str, Any]]):
        self.zones = zones
        self.connections = connections
        self.zone_ids = [z["zone_id"] for z in zones]
        self.zone_to_idx = {z_id: i for i, z_id in enumerate(self.zone_ids)}
        self.num_nodes = len(self.zone_ids)

        # Static node descriptors: zone-type one-hot (5) + capacity + area + exit flag = 8
        node_static = []
        for z in zones:
            onehot = [1.0 if z.get("zone_type") == t else 0.0 for t in ZONE_TYPES]
            node_static.append(onehot + [
                float(z.get("capacity_definition", 100.0)) / 400.0,
                float(z.get("area_m2", 50.0)) / 150.0,
                1.0 if z.get("has_exit", False) else 0.0,
            ])
        self.node_static = torch.tensor(node_static, dtype=torch.float32)

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

class GraphHyperNetwork(nn.Module):
    """
    Hypernetwork that GENERATES the node values and edge values consumed by the GNN.

    Node values  v[z,t] are generated from the zone's static descriptors and its dynamic
    context (observed variables, ages, masks and the FULL decision-tree vector D[z,t]).
    Edge values  e[(u,v),t] are generated from the connection's static geometry plus the
    latent codes of its two endpoint zones, so a link's effective behaviour (e.g. capacity,
    flow resistance) depends on the current state of the zones it joins.

    Channel preservation: D[z,t] enters the hypernetwork unreduced. The temporal forecaster
    still receives D[z,t] directly, independent of this module.
    """
    def __init__(
        self,
        dyn_dim: int = 24,
        static_node_dim: int = 8,
        static_edge_dim: int = 2,
        node_out_dim: int = 16,
        edge_out_dim: int = 4,
        hidden_dim: int = 32
    ):
        super().__init__()
        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim
        self.version = "hypernet_v1"

        self.node_encoder = nn.Sequential(
            nn.Linear(dyn_dim + static_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_head = nn.Linear(hidden_dim, node_out_dim)

        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + static_edge_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_head = nn.Linear(hidden_dim, edge_out_dim)
        self.softplus = nn.Softplus()  # edge values are non-negative (capacity/resistance-like)

    def forward(
        self,
        dyn_ctx: torch.Tensor,        # [num_nodes, dyn_dim]
        static_node: torch.Tensor,    # [num_nodes, static_node_dim]
        edge_index: torch.Tensor,     # [2, num_edges]
        static_edge: torch.Tensor     # [num_edges, static_edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.node_encoder(torch.cat([dyn_ctx, static_node], dim=-1))
        node_values = self.node_head(code)  # [num_nodes, node_out_dim]

        if edge_index.size(1) == 0:
            return node_values, torch.zeros((0, self.edge_out_dim), dtype=node_values.dtype)

        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat([code[src], code[dst], static_edge], dim=-1)
        edge_values = self.softplus(self.edge_head(self.edge_encoder(edge_in)))  # [num_edges, edge_out_dim]
        return node_values, edge_values

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "version": self.version}, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
        self.version = checkpoint.get("version", self.version)

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
