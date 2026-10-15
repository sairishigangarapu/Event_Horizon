import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

ZONE_TYPES = ["entrance", "corridor", "assembly", "stairway", "exit"]
NODE_STATIC_NAMES = [f"zone_type_{t}" for t in ZONE_TYPES] + ["capacity_norm", "area_norm", "has_exit"]
EDGE_STATIC_NAMES = ["width_norm", "distance_norm", "is_reverse_copy", "is_traversable"]
EDGE_DIRECTIONS = ("unidirectional", "bidirectional")
REVERSE_EDGE_SUFFIX = "__rev"

class SpatialGraphBuilder:
    """
    Constructs the message-passing graph, edge features and static node descriptors
    consumed by the GraphHyperNetwork.

    Every connection yields two message-passing edges, in connection order:
      forward  edge_id        from_zone -> to_zone
      reverse  edge_id__rev   to_zone -> from_zone
    Reverse copies let downstream state (e.g. a closed exit) reach upstream zones.
    They do not imply people can walk that way: physical movement is from_zone -> to_zone
    (as in CrowdSimulator), and only a "bidirectional" connection is traversable in reverse.

    Edge attributes (EDGE_STATIC_NAMES):
      width_norm, distance_norm   physical geometry (width/5 m, distance/50 m)
      is_reverse_copy             1 for the generated reverse message edge
      is_traversable              1 if people may physically move in this edge's direction
    """
    def __init__(
        self,
        zones: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        map_version: str = "v1.0"
    ):
        self.zones = zones
        self.connections = connections
        self.map_version = map_version
        self.zone_ids = [z["zone_id"] for z in zones]
        if len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError("Duplicate zone_id in venue layout")
        self.zone_to_idx = {z_id: i for i, z_id in enumerate(self.zone_ids)}
        self.num_nodes = len(self.zone_ids)

        # Static node descriptors: zone-type one-hot (5) + capacity + area + exit flag = 8
        # An explicitly unknown capacity (capacity_definition None) is only accepted when the zone
        # declares capacity_status "unknown"; it then contributes 0.0 and is listed in
        # unknown_capacity_zones. It is never replaced by an invented capacity value.
        node_static = []
        self.unknown_capacity_zones: List[str] = []
        for z in zones:
            onehot = [1.0 if z.get("zone_type") == t else 0.0 for t in ZONE_TYPES]
            if "capacity_definition" in z and z["capacity_definition"] is None:
                if z.get("capacity_status") != "unknown":
                    raise ValueError(f"Zone {z['zone_id']} has no capacity; declare capacity_status 'unknown' explicitly")
                self.unknown_capacity_zones.append(z["zone_id"])
                capacity_norm = 0.0
            else:
                capacity_norm = float(z.get("capacity_definition", 100.0)) / 400.0
            node_static.append(onehot + [
                capacity_norm,
                float(z.get("area_m2", 50.0)) / 150.0,
                1.0 if z.get("has_exit", False) else 0.0,
            ])
        self.node_static = torch.tensor(node_static, dtype=torch.float32)

        # Build edge index, edge ids and edge attributes (forward edge, then its reverse copy)
        edge_indices = []
        edge_attrs = []
        self.edge_ids: List[str] = []

        for conn in connections:
            direction = conn.get("direction", "unidirectional")
            if direction not in EDGE_DIRECTIONS:
                raise ValueError(f"Unknown direction '{direction}' on connection {conn.get('edge_id')}")
            for endpoint in ("from_zone", "to_zone"):
                if conn[endpoint] not in self.zone_to_idx:
                    raise ValueError(f"Connection {conn.get('edge_id')} references unknown zone '{conn[endpoint]}'")
            u = self.zone_to_idx[conn["from_zone"]]
            v = self.zone_to_idx[conn["to_zone"]]
            width = float(conn.get("width_m", 3.0)) / 5.0
            dist = float(conn.get("distance_m", 25.0)) / 50.0
            reverse_traversable = 1.0 if direction == "bidirectional" else 0.0

            edge_indices.append([u, v])
            edge_attrs.append([width, dist, 0.0, 1.0])
            self.edge_ids.append(conn["edge_id"])

            edge_indices.append([v, u])
            edge_attrs.append([width, dist, 1.0, reverse_traversable])
            self.edge_ids.append(conn["edge_id"] + REVERSE_EDGE_SUFFIX)

        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("Duplicate edge_id in venue layout")

        if len(edge_indices) > 0:
            self.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_attr = torch.zeros((0, len(EDGE_STATIC_NAMES)), dtype=torch.float32)

        self.num_edges = self.edge_index.size(1)
        assert self.num_edges == 2 * len(connections) == len(self.edge_ids)
        assert self.edge_attr.shape == (self.num_edges, len(EDGE_STATIC_NAMES))

    def metadata(self) -> Dict[str, Any]:
        """Identifiers and feature names that a spatial checkpoint must match."""
        return {
            "map_version": self.map_version,
            "zone_ids": list(self.zone_ids),
            "edge_ids": list(self.edge_ids),
            "node_static_names": list(NODE_STATIC_NAMES),
            "edge_static_names": list(EDGE_STATIC_NAMES),
        }

def _save_checkpoint(module: nn.Module, filepath: str):
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save({
        "state_dict": module.state_dict(),
        "version": module.version,
        "config": dict(module.config),
        "metadata": dict(module.metadata),
    }, filepath)

def _load_checkpoint(module: nn.Module, filepath: str):
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
    saved_config = checkpoint.get("config")
    if saved_config != module.config:
        raise ValueError(
            f"Incompatible {type(module).__name__} checkpoint {filepath}: "
            f"saved config {saved_config} does not match expected {module.config}"
        )
    module.load_state_dict(checkpoint["state_dict"])
    module.version = checkpoint.get("version", module.version)
    module.metadata = dict(checkpoint.get("metadata", {}))

class GraphHyperNetwork(nn.Module):
    """
    Hypernetwork that GENERATES the node values and edge values consumed by the GNN.

    Node values  Vn[z,t] are generated from the zone's static descriptors and its dynamic
    context (observed values, ages, report/freshness masks, the FULL decision-tree vector D[z,t]
    and its masks; see crowd_reliability.ml.spatial.SPATIAL_FEATURE_SCHEMA).
    Edge values  Ve[(u,v),t] are generated from the edge's static attributes plus the
    latent codes of its two endpoint zones, so a link's effective behaviour (e.g. capacity,
    flow resistance) depends on the current state of the zones it joins.

    Input normalization lives in buffers (saved with the weights) and is applied only to
    continuous features; entries marked invalid by dyn_valid are zeroed before and after it.

    Channel preservation: D[z,t] enters the hypernetwork unreduced. The temporal forecaster
    still receives D[z,t] directly, independent of this module.
    """
    def __init__(
        self,
        dyn_dim: int = 36,
        static_node_dim: int = 8,
        static_edge_dim: int = 4,
        node_out_dim: int = 16,
        edge_out_dim: int = 4,
        hidden_dim: int = 32,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim
        self.version = "hypernet_v2"
        self.config = {
            "dyn_dim": dyn_dim, "static_node_dim": static_node_dim, "static_edge_dim": static_edge_dim,
            "node_out_dim": node_out_dim, "edge_out_dim": edge_out_dim, "hidden_dim": hidden_dim,
        }
        self.metadata: Dict[str, Any] = {"seed": seed}

        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
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

        # Identity normalization until set_input_normalization is called
        self.register_buffer("input_mean", torch.zeros(dyn_dim))
        self.register_buffer("input_std", torch.ones(dyn_dim))
        self.register_buffer("input_continuous", torch.zeros(dyn_dim, dtype=torch.bool))

    def set_input_normalization(self, mean, std, continuous: torch.Tensor, scaler_id: str):
        self.input_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.input_std.copy_(torch.as_tensor(std, dtype=torch.float32))
        self.input_continuous.copy_(torch.as_tensor(continuous, dtype=torch.bool))
        self.metadata["scaler_id"] = scaler_id

    def normalize_inputs(self, dyn_ctx: torch.Tensor, dyn_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dyn_valid is None:
            dyn_valid = torch.ones_like(dyn_ctx)
        x = dyn_ctx * dyn_valid
        scaled = (x - self.input_mean) / self.input_std
        return torch.where(self.input_continuous, scaled, x) * dyn_valid

    def forward(
        self,
        dyn_ctx: torch.Tensor,        # [num_nodes, dyn_dim]
        static_node: torch.Tensor,    # [num_nodes, static_node_dim]
        edge_index: torch.Tensor,     # [2, num_edges]
        static_edge: torch.Tensor,    # [num_edges, static_edge_dim]
        dyn_valid: Optional[torch.Tensor] = None  # [num_nodes, dyn_dim], 1 = valid entry
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.normalize_inputs(dyn_ctx, dyn_valid)
        code = self.node_encoder(torch.cat([x, static_node], dim=-1))
        node_values = self.node_head(code)  # [num_nodes, node_out_dim]

        if edge_index.size(1) == 0:
            return node_values, torch.zeros((0, self.edge_out_dim), dtype=node_values.dtype)

        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat([code[src], code[dst], static_edge], dim=-1)
        edge_values = self.softplus(self.edge_head(self.edge_encoder(edge_in)))  # [num_edges, edge_out_dim]
        return node_values, edge_values

    def save(self, filepath: str):
        _save_checkpoint(self, filepath)

    def load(self, filepath: str):
        _load_checkpoint(self, filepath)

class EdgeAwareGNNLayer(nn.Module):
    """
    Edge-aware graph message passing layer.
    Each edge (u -> v) sends msg_mlp([x[u], edge_attr[u->v]]) to v, where edge_attr is the
    hypernetwork-generated edge value Ve; messages are sum-aggregated and added to a self
    transform. With no edges the aggregated term is zero and the layer reduces to the same
    self transform + LayerNorm + ReLU (the no-graph ablation).
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
    Two-layer edge-aware Spatial GNN over the hypernetwork-generated node values Vn[Z,16]
    and edge values Ve[E,4]. Computes the spatial embedding H[Z,embed_dim] per zone.
    """
    def __init__(
        self,
        node_in_dim: int = 16,
        edge_in_dim: int = 4,
        hidden_dim: int = 32,
        embed_dim: int = 16,
        seed: Optional[int] = None
    ):
        super().__init__()
        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
            self.layer1 = EdgeAwareGNNLayer(node_in_dim, edge_in_dim, hidden_dim)
            self.layer2 = EdgeAwareGNNLayer(hidden_dim, edge_in_dim, embed_dim)
        self.embed_dim = embed_dim
        self.version = "gnn_edge_v2"
        self.config = {
            "node_in_dim": node_in_dim, "edge_in_dim": edge_in_dim,
            "hidden_dim": hidden_dim, "embed_dim": embed_dim,
        }
        self.metadata: Dict[str, Any] = {"seed": seed}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x, edge_index, edge_attr)
        h2 = self.layer2(h1, edge_index, edge_attr)
        return h2  # [num_nodes, embed_dim]

    def save(self, filepath: str):
        _save_checkpoint(self, filepath)

    def load(self, filepath: str):
        _load_checkpoint(self, filepath)
