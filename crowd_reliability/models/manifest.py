import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class PipelineManifest(BaseModel):
    pipeline_version: str = "v1.0.0"
    tree_version: str = "dt_bank_v1"
    hypernet_version: str = "hypernet_v2"
    gnn_version: str = "gnn_edge_v2"
    temporal_version: str = "gru_forecaster_v1"
    scaler_version: str = "standard_scaler_v1"
    schema_version: str = "v1.0"
    training_cutoff: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scenario_split: str = "production"
    is_active: bool = True
    model_hashes: Dict[str, str] = Field(default_factory=dict)
    metrics_summary: Optional[Dict[str, Any]] = None
    # Fit status per component, e.g. {"hypernet": "untrained_initial"}; never implies validation
    training_status: Dict[str, str] = Field(default_factory=dict)
    # Relative directory holding the active hypernetwork+GNN pair; switching it is the only way
    # the active spatial pair changes (see crowd_reliability.ml.spatial.write_spatial_artifact)
    spatial_artifact_dir: Optional[str] = None

    def save(self, filepath: str):
        """Writes atomically: readers see either the previous or the new manifest, never a partial one."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = f"{filepath}.tmp"
        with open(tmp_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        os.replace(tmp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PipelineManifest":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
