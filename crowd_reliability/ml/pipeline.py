import os
import uuid
import hashlib
import numpy as np
import torch
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any, Optional

from crowd_reliability.models.schema import (
    ZoneRecord, ConnectionRecord, ObservationRecord,
    TreeOutputRecord, ForecastRecord, ReferenceStateRecord
)
from crowd_reliability.models.manifest import PipelineManifest
from crowd_reliability.ml.tree_bank import DecisionTreeBank, TREE_CHANNELS
from crowd_reliability.ml.graph import SpatialGraphBuilder, SpatialGNN, GraphHyperNetwork
from crowd_reliability.ml.temporal import MultivariateTemporalForecaster, TARGET_VAR_NAMES
from crowd_reliability.simulation.scenarios import VenueLayout, get_default_venue

def _calc_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class ArchitecturePipeline:
    """
    End-to-End Prediction Pipeline implementing:
    1. Decision-tree channel preservation (never collapsed into a scalar)
    2. Spatial GNN contextualization
    3. Multivariate temporal forecasting across multiple horizons
    4. Strict data contract adherence
    """
    def __init__(
        self,
        venue: Optional[VenueLayout] = None,
        manifest: Optional[PipelineManifest] = None,
        lookback_steps: int = 12,
        horizons_sec: Optional[List[int]] = None
    ):
        self.venue = venue or get_default_venue()
        self.manifest = manifest or PipelineManifest()
        self.lookback_steps = lookback_steps
        self.horizons_sec = horizons_sec or [30, 60, 120]

        self.tree_bank = DecisionTreeBank()
        self.graph_builder = SpatialGraphBuilder(self.venue.zones, self.venue.connections)

        # Dynamic context dim fed to the hypernetwork:
        # Raw observed values (6) + ages (6) + masks (6) + tree values (6) = 24
        self.node_in_dim = 24

        # Hypernetwork generates the node values and edge values the GNN operates on
        self.hypernet = GraphHyperNetwork(
            dyn_dim=self.node_in_dim,
            static_node_dim=self.graph_builder.node_static.size(1),
            static_edge_dim=self.graph_builder.edge_attr.size(1),
            node_out_dim=16,
            edge_out_dim=4,
            hidden_dim=32
        )
        self.gnn = SpatialGNN(
            node_in_dim=self.hypernet.node_out_dim,
            edge_in_dim=self.hypernet.edge_out_dim,
            hidden_dim=32,
            embed_dim=16
        )
        # Most recent hypernetwork outputs, retained for inspection/export
        self.last_generated: Dict[str, np.ndarray] = {}

        # Temporal input dim:
        # X (18) + D (6) + H (16) = 40
        self.u_dim = 18 + 6 + self.gnn.embed_dim
        self.temporal_model = MultivariateTemporalForecaster(
            input_dim=self.u_dim,
            hidden_dim=64,
            num_layers=2,
            num_horizons=len(self.horizons_sec),
            horizons_sec=self.horizons_sec
        )

        # Rolling history buffer per scenario: {zone_id: list of U vectors}
        self.history_buffer: Dict[str, Dict[str, List[np.ndarray]]] = {}

    def extract_zone_features(
        self,
        zone_id: str,
        as_of_time: datetime,
        recent_observations: List[ObservationRecord]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts as-of snapshot:
        - raw_values: 6 variables
        - ages_sec: observation age in seconds
        - obs_masks: 1 if observed recently (within 90s), 0 if missing
        """
        var_indices = {v: i for i, v in enumerate(TARGET_VAR_NAMES)}
        raw_values = np.zeros(6, dtype=np.float32)
        ages_sec = np.full(6, 999.0, dtype=np.float32)
        obs_masks = np.zeros(6, dtype=np.float32)

        # Filter observations for this zone received at or before as_of_time
        zone_obs = [
            o for o in recent_observations
            if o.zone_id == zone_id and o.received_at <= as_of_time
        ]

        # Find latest report for each variable
        for obs in zone_obs:
            if obs.variable in var_indices:
                idx = var_indices[obs.variable]
                # Age calculation relative to as_of_time
                age = max(0.0, (as_of_time - obs.observed_at).total_seconds())
                if age < ages_sec[idx]:
                    ages_sec[idx] = age
                    raw_values[idx] = obs.value
                    obs_masks[idx] = 1.0 if age <= 90.0 else 0.0

        # Default fallback values for unobserved variables
        if obs_masks[0] == 0: raw_values[0] = 0.8  # density
        if obs_masks[1] == 0: raw_values[1] = 1.2  # speed
        if obs_masks[2] == 0: raw_values[2] = 1.0  # inflow
        if obs_masks[3] == 0: raw_values[3] = 1.0  # outflow
        if obs_masks[4] == 0: raw_values[4] = 0.0  # queue_length
        if obs_masks[5] == 0: raw_values[5] = 1.0  # exit_open

        # Normalize age to [0, 1] range (cap at 120s)
        norm_ages = np.clip(ages_sec / 120.0, 0.0, 1.0)
        return raw_values, norm_ages, obs_masks

    def step(
        self,
        scenario_id: str,
        as_of_time: datetime,
        recent_observations: List[ObservationRecord]
    ) -> Tuple[List[TreeOutputRecord], List[ForecastRecord]]:
        """
        Executes one evaluation step:
        1. Evaluates DecisionTreeBank for each zone -> D[z,t] vector.
        2. Builds graph input and evaluates SpatialGNN -> H[z,t].
        3. Assembles U[z,t] and appends to rolling history.
        4. Runs MultivariateTemporalForecaster -> multi-horizon forecasts.
        """
        tree_records: List[TreeOutputRecord] = []
        forecast_records: List[ForecastRecord] = []

        zone_lookup = {z["zone_id"]: z for z in self.venue.zones}
        num_zones = len(self.venue.zones)

        # 1. Evaluate Decision Tree Bank for each zone
        tree_vectors: List[List[float]] = []
        tree_masks_list: List[List[int]] = []
        node_features_list: List[np.ndarray] = []

        for z_data in self.venue.zones:
            z_id = z_data["zone_id"]
            has_exit = z_data.get("has_exit", False)
            raw_vals, norm_ages, obs_masks = self.extract_zone_features(z_id, as_of_time, recent_observations)

            # Combined input for decision tree
            dt_feat = np.concatenate([raw_vals, norm_ages, obs_masks])
            d_vals, d_masks = self.tree_bank.predict_zone(dt_feat, has_exit=has_exit)

            tree_vectors.append(d_vals)
            tree_masks_list.append(d_masks)

            tree_rec = TreeOutputRecord(
                scenario_id=scenario_id,
                zone_id=z_id,
                evaluation_time=as_of_time,
                tree_version=self.manifest.tree_version,
                channel_names=TREE_CHANNELS,
                channel_values=d_vals,
                channel_masks=d_masks
            )
            tree_records.append(tree_rec)

            # GNN node input: [raw (6), norm_ages (6), obs_masks (6), d_vals (6)] = 24
            node_feat = np.concatenate([raw_vals, norm_ages, obs_masks, np.array(d_vals, dtype=np.float32)])
            node_features_list.append(node_feat)

        # 2. Hypernetwork generates node values and edge values, then GNN contextualizes them
        dyn_ctx = torch.tensor(np.array(node_features_list), dtype=torch.float32)
        with torch.no_grad():
            node_values, edge_values = self.hypernet(
                dyn_ctx,
                self.graph_builder.node_static,
                self.graph_builder.edge_index,
                self.graph_builder.edge_attr
            )
            h_embeddings = self.gnn(
                node_values,
                self.graph_builder.edge_index,
                edge_values
            ).cpu().numpy()  # [num_zones, 16]
        self.last_generated = {
            "node_values": node_values.cpu().numpy(),
            "edge_values": edge_values.cpu().numpy(),
        }

        # 3. Assemble U[z, t] vector per zone and update history
        if scenario_id not in self.history_buffer:
            self.history_buffer[scenario_id] = {z_data["zone_id"]: [] for z_data in self.venue.zones}

        u_current_per_zone: List[np.ndarray] = []
        for i, z_data in enumerate(self.venue.zones):
            z_id = z_data["zone_id"]
            # U = [raw(6), age(6), mask(6), D(6), H(16)] = 40
            u_vec = np.concatenate([
                node_features_list[i],  # 24 features (including raw, age, mask, D)
                h_embeddings[i]         # 16 spatial embedding features
            ])
            self.history_buffer[scenario_id][z_id].append(u_vec)
            # Keep buffer size to lookback_steps
            if len(self.history_buffer[scenario_id][z_id]) > self.lookback_steps:
                self.history_buffer[scenario_id][z_id].pop(0)

            # Pad history if fewer than lookback_steps
            hist = self.history_buffer[scenario_id][z_id]
            if len(hist) < self.lookback_steps:
                padding = [hist[0]] * (self.lookback_steps - len(hist))
                seq = np.array(padding + hist, dtype=np.float32)
            else:
                seq = np.array(hist[-self.lookback_steps:], dtype=np.float32)
            u_current_per_zone.append(seq)

        # 4. Multivariate Temporal Forecasting
        # Shape: [num_zones, lookback_steps, u_dim]
        u_history_tensor = torch.tensor(np.array(u_current_per_zone), dtype=torch.float32)
        with torch.no_grad():
            # [num_zones, num_horizons, 6]
            forecasts_tensor = self.temporal_model(u_history_tensor).cpu().numpy()

        # Build ForecastRecord objects
        units = {
            "density": "p/m2",
            "speed": "m/s",
            "inflow": "p/s",
            "outflow": "p/s",
            "queue_length": "persons",
            "exit_open": "binary"
        }

        for i, z_data in enumerate(self.venue.zones):
            z_id = z_data["zone_id"]
            has_exit = z_data.get("has_exit", False)

            for h_idx, h_sec in enumerate(self.horizons_sec):
                target_at = as_of_time + timedelta(seconds=h_sec)

                for var_idx, var_name in enumerate(TARGET_VAR_NAMES):
                    # For non-exit zones, skip exit_open forecast or set to -1
                    if var_name == "exit_open" and not has_exit:
                        continue

                    val = float(forecasts_tensor[i, h_idx, var_idx])

                    f_rec = ForecastRecord(
                        forecast_id=f"f_arch_{uuid.uuid4().hex[:8]}",
                        stream_id="stream_b_architecture",
                        scenario_id=scenario_id,
                        zone_id=z_id,
                        variable=var_name,
                        unit=units[var_name],
                        issued_at=as_of_time,
                        received_at=as_of_time,
                        target_at=target_at,
                        value=round(val, 4),
                        model_or_provider_version=self.manifest.pipeline_version
                    )
                    forecast_records.append(f_rec)

        return tree_records, forecast_records

    def save_pipeline(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        tb_path = os.path.join(model_dir, "tree_bank.joblib")
        hn_path = os.path.join(model_dir, "hypernet.pt")
        gnn_path = os.path.join(model_dir, "gnn.pt")
        tm_path = os.path.join(model_dir, "temporal_model.pt")

        self.tree_bank.save(tb_path)
        self.hypernet.save(hn_path)
        self.gnn.save(gnn_path)
        self.temporal_model.save(tm_path)

        self.manifest.model_hashes = {
            "tree_bank.joblib": _calc_sha256(tb_path),
            "hypernet.pt": _calc_sha256(hn_path),
            "gnn.pt": _calc_sha256(gnn_path),
            "temporal_model.pt": _calc_sha256(tm_path),
        }
        self.manifest.save(os.path.join(model_dir, "manifest.json"))

    def load_pipeline(self, model_dir: str):
        manifest_path = os.path.join(model_dir, "manifest.json")
        if os.path.exists(manifest_path):
            self.manifest = PipelineManifest.load(manifest_path)

        def _verify_file(filename: str, filepath: str):
            if filename in self.manifest.model_hashes:
                expected = self.manifest.model_hashes[filename]
                actual = _calc_sha256(filepath)
                if expected != actual:
                    raise ValueError(f"Model integrity verification failed for {filename}")

        tb_path = os.path.join(model_dir, "tree_bank.joblib")
        if os.path.exists(tb_path):
            _verify_file("tree_bank.joblib", tb_path)
            self.tree_bank.load(tb_path)

        hn_path = os.path.join(model_dir, "hypernet.pt")
        if os.path.exists(hn_path):
            _verify_file("hypernet.pt", hn_path)
            self.hypernet.load(hn_path)

        gnn_path = os.path.join(model_dir, "gnn.pt")
        if os.path.exists(gnn_path):
            _verify_file("gnn.pt", gnn_path)
            self.gnn.load(gnn_path)

        tm_path = os.path.join(model_dir, "temporal_model.pt")
        if os.path.exists(tm_path):
            _verify_file("temporal_model.pt", tm_path)
            self.temporal_model = MultivariateTemporalForecaster.load(tm_path)
