import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class PipelineManifest(BaseModel):
    pipeline_version: str = "v1.0.0"
    tree_version: str = "dt_bank_v1"
    hypernet_version: str = "hypernet_v1"
    gnn_version: str = "gnn_edge_v1"
    temporal_version: str = "gru_forecaster_v1"
    scaler_version: str = "standard_scaler_v1"
    schema_version: str = "v1.0"
    training_cutoff: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scenario_split: str = "production"
    is_active: bool = True
    model_hashes: Dict[str, str] = Field(default_factory=dict)
    metrics_summary: Optional[Dict[str, Any]] = None

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str) -> "PipelineManifest":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
