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
from crowd_reliability.ml.graph import SpatialGraphBuilder
from crowd_reliability.ml.spatial import SpatialEncoder, stack_zone_features, write_spatial_artifact
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
        horizons_sec: Optional[List[int]] = None,
        spatial_seed: Optional[int] = 0
    ):
        self.venue = venue or get_default_venue()
        self.manifest = manifest or PipelineManifest()
        self.lookback_steps = lookback_steps
        self.horizons_sec = horizons_sec or [30, 60, 120]

        self.tree_bank = DecisionTreeBank()
        self.graph_builder = SpatialGraphBuilder(self.venue.zones, self.venue.connections)

        # Per-zone non-spatial features carried into U:
        # Raw observed values (6) + ages (6) + masks (6) + tree values (6) = 24
        self.node_in_dim = 24

        # Hypernetwork + GNN pair. The hypernetwork reads its own masked 36-feature context
        # (crowd_reliability.ml.spatial.SPATIAL_FEATURE_SCHEMA), which also includes the tree masks.
        self.spatial = SpatialEncoder(self.graph_builder, seed=spatial_seed)
        self.hypernet = self.spatial.hypernet
        self.gnn = self.spatial.gnn
        # Most recent hypernetwork outputs, retained for inspection/export
        self.last_generated: Dict[str, Any] = {}

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

    def extract_zone_features_detailed(
        self,
        zone_id: str,
        as_of_time: datetime,
        recent_observations: List[ObservationRecord]
    ) -> Dict[str, np.ndarray]:
        """
        As-of snapshot without fallback substitution (used by the spatial encoder):
        - values:   value of the selected report, 0 where no report exists
        - ages_sec: seconds since observed_at of the selected report, 999 where none exists
        - present:  1 if any report with received_at <= as_of_time exists
        - fresh:    1 if the selected report is at most 90 s old
        Per variable the report with the smallest age is selected (ties: first in input order).
        """
        var_indices = {v: i for i, v in enumerate(TARGET_VAR_NAMES)}
        values = np.zeros(6, dtype=np.float32)
        ages_sec = np.full(6, 999.0, dtype=np.float32)
        present = np.zeros(6, dtype=np.float32)
        fresh = np.zeros(6, dtype=np.float32)

        for obs in recent_observations:
            if obs.zone_id != zone_id or obs.received_at > as_of_time or obs.variable not in var_indices:
                continue
            idx = var_indices[obs.variable]
            age = max(0.0, (as_of_time - obs.observed_at).total_seconds())
            if age < ages_sec[idx]:
                ages_sec[idx] = age
                values[idx] = obs.value
                present[idx] = 1.0
                fresh[idx] = 1.0 if age <= 90.0 else 0.0

        return {"values": values, "ages_sec": ages_sec, "present": present, "fresh": fresh}

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
        detailed = self.extract_zone_features_detailed(zone_id, as_of_time, recent_observations)
        return self._legacy_zone_features(detailed)

    @staticmethod
    def _legacy_zone_features(detailed: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tree-bank / U features: fresh-only mask, fallback defaults, ages normalized to [0, 1]."""
        raw_values = detailed["values"].copy()
        ages_sec = detailed["ages_sec"]
        obs_masks = detailed["fresh"].copy()

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
        detailed_list: List[Dict[str, np.ndarray]] = []

        for z_data in self.venue.zones:
            z_id = z_data["zone_id"]
            has_exit = z_data.get("has_exit", False)
            detailed = self.extract_zone_features_detailed(z_id, as_of_time, recent_observations)
            detailed_list.append(detailed)
            raw_vals, norm_ages, obs_masks = self._legacy_zone_features(detailed)

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

            # Non-spatial U features: [raw (6), norm_ages (6), obs_masks (6), d_vals (6)] = 24
            node_feat = np.concatenate([raw_vals, norm_ages, obs_masks, np.array(d_vals, dtype=np.float32)])
            node_features_list.append(node_feat)

        # 2. Hypernetwork generates node values and edge values, then GNN contextualizes them
        spatial_out = self.spatial.encode_spatial(
            stack_zone_features(detailed_list),
            np.array(tree_vectors, dtype=np.float32),
            np.array(tree_masks_list, dtype=np.float32),
            tree_version=self.manifest.tree_version,
        )
        h_embeddings = spatial_out.H  # [num_zones, 16]
        self.last_generated = {
            "node_values": spatial_out.Vn,
            "edge_values": spatial_out.Ve,
            "zone_ids": spatial_out.zone_ids,
            "edge_ids": spatial_out.edge_ids,
            "version_metadata": spatial_out.version_metadata,
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
        tm_path = os.path.join(model_dir, "temporal_model.pt")

        self.tree_bank.save(tb_path)
        self.temporal_model.save(tm_path)
        # The hypernetwork+GNN pair goes into its own new versioned directory
        spatial_dir, spatial_hashes = write_spatial_artifact(self.spatial, model_dir)

        self.manifest.model_hashes = {
            "tree_bank.joblib": _calc_sha256(tb_path),
            "temporal_model.pt": _calc_sha256(tm_path),
            **spatial_hashes,
        }
        self.manifest.hypernet_version = self.hypernet.version
        self.manifest.gnn_version = self.gnn.version
        self.manifest.spatial_artifact_dir = spatial_dir
        self.manifest.training_status["hypernet"] = self.hypernet.metadata.get("training_status", "unknown")
        self.manifest.training_status["gnn"] = self.gnn.metadata.get("training_status", "unknown")
        self.manifest.training_status.setdefault("temporal", "untrained_initial")
        # Written last and atomically: this is the switch to the new artifact set
        self.manifest.save(os.path.join(model_dir, "manifest.json"))

    def load_pipeline(self, model_dir: str):
        manifest_path = os.path.join(model_dir, "manifest.json")
        if os.path.exists(manifest_path):
            self.manifest = PipelineManifest.load(manifest_path)

        def _verify_file(filename: str, filepath: str, required: bool = False):
            if filename in self.manifest.model_hashes:
                expected = self.manifest.model_hashes[filename]
                actual = _calc_sha256(filepath)
                if expected != actual:
                    raise ValueError(f"Model integrity verification failed for {filename}")
            elif required:
                raise ValueError(f"Model integrity verification failed for {filename}: no recorded hash")

        tb_path = os.path.join(model_dir, "tree_bank.joblib")
        if os.path.exists(tb_path):
            _verify_file("tree_bank.joblib", tb_path)
            self.tree_bank.load(tb_path)

        if self.manifest.spatial_artifact_dir:
            # Verify both files of the pair before loading either
            spatial_dir = os.path.join(model_dir, self.manifest.spatial_artifact_dir)
            for filename in ("hypernet.pt", "gnn.pt"):
                rel = f"{self.manifest.spatial_artifact_dir}/{filename}"
                _verify_file(rel, os.path.join(spatial_dir, filename), required=True)
            self.spatial.load(spatial_dir)
        elif any(os.path.exists(os.path.join(model_dir, f)) for f in ("hypernet.pt", "gnn.pt")):
            raise ValueError(
                "Legacy spatial checkpoint (hypernet.pt/gnn.pt at the model root, no spatial_artifact_dir) "
                f"is incompatible with {self.hypernet.version}/{self.gnn.version}. "
                "Run `python -m crowd_reliability.cli spatial-init` to create a compatible pair."
            )

        tm_path = os.path.join(model_dir, "temporal_model.pt")
        if os.path.exists(tm_path):
            _verify_file("temporal_model.pt", tm_path)
            self.temporal_model = MultivariateTemporalForecaster.load(tm_path)
