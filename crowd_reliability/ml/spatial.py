"""
Spatial encoder contract (Person 3): feature schema, feature construction and compatibility checks
for the GraphHyperNetwork + SpatialGNN pair.

Per-zone dynamic context fed to the hypernetwork (SPATIAL_FEATURE_SCHEMA, 36 features, in order):

  obs_value_<var>       latest observed value                 0 when report_present_<var> = 0
  obs_age_<var>         seconds since observed_at              0 when report_present_<var> = 0
  report_present_<var>  1 if any report received <= as_of     (binary)
  obs_fresh_<var>       1 if the selected report is <= 90 s   (binary; the pipeline's obs_mask)
  tree_<channel>        decision-tree estimate D[z,t]          0 when tree_mask_<channel> = 0
  tree_mask_<channel>   tree applicability mask from Person 2  (binary)

Observation semantics are those of ArchitecturePipeline.extract_zone_features_detailed:
  - only reports with received_at <= as_of are considered;
  - per zone and variable the report with the smallest age wins (ties: first in input order);
  - age is measured from observed_at; freshness is taken from the selected report;
  - conflicting reports are not resolved here (conflict assessment is ForecastComparator.assess_information).
Unlike extract_zone_features, stale-but-present values are kept (with their age) and no fallback
default is substituted for a missing value; absence is carried by report_present.

Normalization applies only to features of kind "continuous"; binary and probability features
pass through unchanged. Invalid entries are zeroed before normalization and again after it,
so a missing value never influences a statistic or an output.
"""
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from crowd_reliability.ml.graph import GraphHyperNetwork, SpatialGNN, SpatialGraphBuilder
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES
from crowd_reliability.ml.tree_bank import TREE_CHANNELS

OBS_UNITS = {
    "density": "p/m2",
    "speed": "m/s",
    "inflow": "p/s",
    "outflow": "p/s",
    "queue_length": "persons",
    "exit_open": "binary",
}
TREE_UNITS = {
    "estimated_density": "p/m2",
    "estimated_speed": "m/s",
    "estimated_inflow": "p/s",
    "estimated_outflow": "p/s",
    "estimated_queue_length": "persons",
    "probability_exit_open": "probability",
}

# Documented physical scales used before any training-split statistics exist ("physical_default").
DEFAULT_SCALES = {
    "density": (1.0, 1.0),
    "speed": (1.0, 0.3),
    "inflow": (1.5, 1.5),
    "outflow": (1.5, 1.5),
    "queue_length": (10.0, 25.0),
    "age": (30.0, 30.0),
}
TREE_TO_OBS = dict(zip(TREE_CHANNELS, TARGET_VAR_NAMES))

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    kind: str            # "continuous" | "binary" | "probability"
    unit: str
    mask: Optional[str]  # name of the binary feature that marks this one valid

def _build_schema() -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    for v in TARGET_VAR_NAMES:
        kind = "binary" if v == "exit_open" else "continuous"
        specs.append(FeatureSpec(f"obs_value_{v}", "obs_value", kind, OBS_UNITS[v], f"report_present_{v}"))
    for v in TARGET_VAR_NAMES:
        specs.append(FeatureSpec(f"obs_age_{v}", "obs_age", "continuous", "s", f"report_present_{v}"))
    for v in TARGET_VAR_NAMES:
        specs.append(FeatureSpec(f"report_present_{v}", "report_present", "binary", "flag", None))
    for v in TARGET_VAR_NAMES:
        specs.append(FeatureSpec(f"obs_fresh_{v}", "obs_fresh", "binary", "flag", None))
    for c in TREE_CHANNELS:
        kind = "probability" if c == "probability_exit_open" else "continuous"
        specs.append(FeatureSpec(f"tree_{c}", "tree_value", kind, TREE_UNITS[c], f"tree_mask_{c}"))
    for c in TREE_CHANNELS:
        specs.append(FeatureSpec(f"tree_mask_{c}", "tree_mask", "binary", "flag", None))
    return specs

SPATIAL_FEATURE_SCHEMA: List[FeatureSpec] = _build_schema()
SPATIAL_FEATURE_NAMES: List[str] = [s.name for s in SPATIAL_FEATURE_SCHEMA]
SPATIAL_DYN_DIM: int = len(SPATIAL_FEATURE_SCHEMA)
_INDEX = {name: i for i, name in enumerate(SPATIAL_FEATURE_NAMES)}

def schema_hash() -> str:
    payload = json.dumps([[s.name, s.group, s.kind, s.unit, s.mask] for s in SPATIAL_FEATURE_SCHEMA])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def continuous_mask() -> torch.Tensor:
    return torch.tensor([s.kind == "continuous" for s in SPATIAL_FEATURE_SCHEMA], dtype=torch.bool)

def default_normalization() -> Dict[str, Any]:
    """Physical-scale mean/std per feature (identity for non-continuous features)."""
    mean = np.zeros(SPATIAL_DYN_DIM, dtype=np.float32)
    std = np.ones(SPATIAL_DYN_DIM, dtype=np.float32)
    for i, s in enumerate(SPATIAL_FEATURE_SCHEMA):
        if s.kind != "continuous":
            continue
        if s.group == "obs_age":
            key = "age"
        elif s.group == "obs_value":
            key = s.name[len("obs_value_"):]
        else:
            key = TREE_TO_OBS[s.name[len("tree_"):]]
        mean[i], std[i] = DEFAULT_SCALES[key]
    return {"mean": mean, "std": std, "scaler_id": "physical_default"}

def fit_normalization(features: np.ndarray, valid: np.ndarray, scaler_id: str, eps: float = 1e-3) -> Dict[str, Any]:
    """
    Fits per-feature mean/std over VALID entries of continuous features only.
    features, valid: [N, SPATIAL_DYN_DIM]. Callers must pass training-split rows only.
    Features with no valid entries keep their physical default.
    """
    stats = default_normalization()
    cont = continuous_mask().numpy()
    for i in np.where(cont)[0]:
        col = features[valid[:, i] > 0, i]
        if col.size > 1:
            stats["mean"][i] = float(col.mean())
            stats["std"][i] = max(float(col.std()), eps)
    stats["scaler_id"] = scaler_id
    return stats

def build_spatial_features(
    obs_values: np.ndarray,
    obs_ages_sec: np.ndarray,
    report_present: np.ndarray,
    obs_fresh: np.ndarray,
    tree_values: np.ndarray,
    tree_masks: np.ndarray,
):
    """
    Assembles the unnormalized dynamic context and its validity mask.
    All inputs are [Z, 6] in TARGET_VAR_NAMES / TREE_CHANNELS order.
    Returns (features [Z, 36] float32 with invalid entries zeroed, valid [Z, 36] float32).
    """
    present = (np.asarray(report_present, dtype=np.float32) > 0).astype(np.float32)
    fresh = (np.asarray(obs_fresh, dtype=np.float32) > 0).astype(np.float32)
    tmask = (np.asarray(tree_masks, dtype=np.float32) > 0).astype(np.float32)
    ones = np.ones_like(present)

    blocks = [
        (np.asarray(obs_values, dtype=np.float32) * present, present),
        (np.asarray(obs_ages_sec, dtype=np.float32) * present, present),
        (present, ones),
        (fresh, ones),
        (np.asarray(tree_values, dtype=np.float32) * tmask, tmask),
        (tmask, ones),
    ]
    features = np.concatenate([b[0] for b in blocks], axis=1).astype(np.float32)
    valid = np.concatenate([b[1] for b in blocks], axis=1).astype(np.float32)
    assert features.shape[1] == SPATIAL_DYN_DIM
    return features, valid

def check_spatial_compat(metadata: Dict[str, Any], graph_metadata: Dict[str, Any]) -> None:
    """
    Single compatibility contract between a spatial checkpoint and the current graph/schema.
    Raises ValueError naming every mismatch.
    """
    problems = []
    if metadata.get("schema_hash") != schema_hash():
        problems.append("feature schema hash differs")
    if not metadata.get("scaler_id"):
        problems.append("scaler id missing")
    for key in ("zone_ids", "edge_ids", "edge_static_names", "node_static_names"):
        if list(metadata.get(key, [])) != list(graph_metadata.get(key, [])):
            problems.append(f"{key} differ")
    if problems:
        raise ValueError("Incompatible spatial checkpoint: " + "; ".join(problems))

def stack_zone_features(per_zone: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stacks per-zone outputs of extract_zone_features_detailed into [Z, 6] arrays."""
    return {key: np.stack([z[key] for z in per_zone]).astype(np.float32)
            for key in ("values", "ages_sec", "present", "fresh")}

@dataclass
class SpatialOutput:
    """Spatial encoder output. Vn/Ve are generated model internals, not forecasts or trust scores."""
    H: np.ndarray        # [Z, embed_dim] spatial embedding consumed by the temporal forecaster
    Vn: np.ndarray       # [Z, 16] hypernetwork-generated node values
    Ve: np.ndarray       # [E, 4] hypernetwork-generated non-negative edge values
    zone_ids: List[str]  # row order of H and Vn
    edge_ids: List[str]  # row order of Ve
    version_metadata: Dict[str, Any]

# Experiment outputs that must never be referenced by a pipeline manifest
NON_PROMOTABLE_STATUSES = ("smoke_trained_auxiliary",)

HYPERNET_FILE = "hypernet.pt"
GNN_FILE = "gnn.pt"
SPATIAL_MANIFEST_FILE = "spatial_manifest.json"

class SpatialEncoder:
    """
    Owns the GraphHyperNetwork + SpatialGNN pair for one graph. The pair is always saved,
    loaded and versioned together.
    """
    def __init__(self, graph: SpatialGraphBuilder, seed: Optional[int] = 0, hidden_dim: int = 32, embed_dim: int = 16):
        self.graph = graph
        self.hypernet = GraphHyperNetwork(
            dyn_dim=SPATIAL_DYN_DIM,
            static_node_dim=graph.node_static.size(1),
            static_edge_dim=graph.edge_attr.size(1),
            node_out_dim=16,
            edge_out_dim=4,
            hidden_dim=hidden_dim,
            seed=seed,
        )
        self.gnn = SpatialGNN(
            node_in_dim=self.hypernet.node_out_dim,
            edge_in_dim=self.hypernet.edge_out_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            seed=None if seed is None else seed + 1,
        )
        norm = default_normalization()
        self.set_normalization(norm)
        self.stamp(training_status="untrained_initial")

    @property
    def embed_dim(self) -> int:
        return self.gnn.embed_dim

    def set_normalization(self, norm: Dict[str, Any]):
        self.hypernet.set_input_normalization(norm["mean"], norm["std"], continuous_mask(), norm["scaler_id"])
        self.gnn.metadata["scaler_id"] = norm["scaler_id"]

    def stamp(self, training_status: str, pair_id: Optional[str] = None):
        """Writes the shared compatibility metadata onto both modules of the pair."""
        pair_id = pair_id or uuid.uuid4().hex
        common = dict(self.graph.metadata())
        common.update({
            "schema_hash": schema_hash(),
            "feature_names": list(SPATIAL_FEATURE_NAMES),
            "training_status": training_status,
            "pair_id": pair_id,
            "scaler_id": self.hypernet.metadata.get("scaler_id"),
        })
        self.hypernet.metadata.update(common)
        self.gnn.metadata.update(common)

    def version_metadata(self, tree_version: Optional[str] = None) -> Dict[str, Any]:
        return {
            "hypernet_version": self.hypernet.version,
            "gnn_version": self.gnn.version,
            "tree_version": tree_version,
            "schema_hash": schema_hash(),
            "scaler_id": self.hypernet.metadata.get("scaler_id"),
            "training_status": self.hypernet.metadata.get("training_status"),
            "pair_id": self.hypernet.metadata.get("pair_id"),
            "map_version": self.graph.map_version,
        }

    def forward_tensors(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable forward pass (used by training and tests). Returns H, Vn, Ve."""
        edge_index = self.graph.edge_index if edge_index is None else edge_index
        edge_attr = self.graph.edge_attr if edge_attr is None else edge_attr
        vn, ve = self.hypernet(features, self.graph.node_static, edge_index, edge_attr, dyn_valid=valid)
        h = self.gnn(vn, edge_index, ve)
        return h, vn, ve

    def encode_spatial(
        self,
        zone_features: Dict[str, np.ndarray],
        tree_values: np.ndarray,
        tree_masks: np.ndarray,
        tree_version: Optional[str] = None,
    ) -> SpatialOutput:
        """
        Inference entry point. zone_features is stack_zone_features(...) built only from
        observations received at or before the evaluation time; no reference states,
        targets or forecasts are accepted.
        """
        features, valid = build_spatial_features(
            zone_features["values"], zone_features["ages_sec"], zone_features["present"],
            zone_features["fresh"], tree_values, tree_masks,
        )
        self.hypernet.eval()
        self.gnn.eval()
        with torch.no_grad():
            h, vn, ve = self.forward_tensors(torch.from_numpy(features), torch.from_numpy(valid))
        return SpatialOutput(
            H=h.cpu().numpy(),
            Vn=vn.cpu().numpy(),
            Ve=ve.cpu().numpy(),
            zone_ids=list(self.graph.zone_ids),
            edge_ids=list(self.graph.edge_ids),
            version_metadata=self.version_metadata(tree_version),
        )

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.hypernet.save(os.path.join(directory, HYPERNET_FILE))
        self.gnn.save(os.path.join(directory, GNN_FILE))

    def load(self, directory: str):
        self.hypernet.load(os.path.join(directory, HYPERNET_FILE))
        self.gnn.load(os.path.join(directory, GNN_FILE))
        graph_meta = self.graph.metadata()
        check_spatial_compat(self.hypernet.metadata, graph_meta)
        check_spatial_compat(self.gnn.metadata, graph_meta)
        if self.hypernet.metadata.get("pair_id") != self.gnn.metadata.get("pair_id"):
            raise ValueError("Incompatible spatial checkpoint: hypernetwork and GNN are not from the same pair")

def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def write_spatial_artifact(encoder: SpatialEncoder, model_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    Writes the hypernetwork+GNN pair into a new versioned directory
    <model_dir>/spatial/<hypernet_version>__<gnn_version>__<hash8>/ via a staging directory:
    the pair is saved, reloaded and run once before the staging directory is renamed into place.
    Existing artifacts are never modified. Returns (relative_dir, {relative_file: sha256}).
    """
    status = encoder.hypernet.metadata.get("training_status")
    if status in NON_PROMOTABLE_STATUSES:
        raise ValueError(f"Spatial pair with training_status '{status}' is experimental and cannot become an active pair")
    spatial_root = os.path.join(model_dir, "spatial")
    os.makedirs(spatial_root, exist_ok=True)
    staging = os.path.join(spatial_root, f".staging-{uuid.uuid4().hex}")
    try:
        encoder.save(staging)
        manifest = {
            "hypernet_version": encoder.hypernet.version,
            "gnn_version": encoder.gnn.version,
            "metadata": encoder.hypernet.metadata,
            "hashes": {name: file_sha256(os.path.join(staging, name)) for name in (HYPERNET_FILE, GNN_FILE)},
        }
        with open(os.path.join(staging, SPATIAL_MANIFEST_FILE), "w") as f:
            json.dump(manifest, f, indent=2)

        # Validate the staged pair before it can be referenced
        probe = SpatialEncoder(encoder.graph, seed=None)
        probe.load(staging)
        z = encoder.graph.num_nodes
        h, vn, ve = probe.forward_tensors(torch.zeros(z, SPATIAL_DYN_DIM), torch.ones(z, SPATIAL_DYN_DIM))
        if h.shape != (z, probe.embed_dim) or ve.shape[0] != encoder.graph.num_edges or not torch.all(ve >= 0):
            raise ValueError("Staged spatial pair failed validation forward pass")

        pair_hash = hashlib.sha256("".join(manifest["hashes"].values()).encode()).hexdigest()[:8]
        name = f"{encoder.hypernet.version}__{encoder.gnn.version}__{pair_hash}"
        final = os.path.join(spatial_root, name)
        if os.path.exists(final):
            raise FileExistsError(f"Spatial artifact directory already exists: {final}")
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    rel_dir = f"spatial/{name}"
    hashes = {f"{rel_dir}/{fname}": digest for fname, digest in manifest["hashes"].items()}
    return rel_dir, hashes
