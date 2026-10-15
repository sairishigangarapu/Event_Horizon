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
    scaler_version: Optional[str] = "standard_scaler_v1"
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
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        os.replace(tmp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PipelineManifest":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------- readiness

VALIDATED_STATUS = "trained_validated"
NEURAL_STAGES = ("hypernet", "gnn", "temporal")
EXPERIMENTAL_NOTICE = (
    "Stream B (architecture) forecasts come from neural stages that are not trained and validated "
    "({stages}). They demonstrate the data path, not forecast skill. Only the decision-tree bank is fitted."
)


class ModelNotReadyError(RuntimeError):
    """Raised in strict mode when the active pipeline is not trained and validated."""


def assess_readiness(manifest: "PipelineManifest") -> Dict[str, Any]:
    """
    Readiness of the active pipeline, read only from the manifest.

    production_ready requires every neural stage to be VALIDATED_STATUS and a
    recorded metrics_summary. Anything else is served as "experimental", with a
    notice that names the unvalidated stages. Missing statuses count as "unknown".
    """
    stages = {s: manifest.training_status.get(s, "unknown") for s in NEURAL_STAGES}
    unvalidated = [s for s, status in stages.items() if status != VALIDATED_STATUS]
    has_metrics = manifest.metrics_summary is not None
    ready = not unvalidated and has_metrics
    synthetic = all(status == "synthetic_validated" for status in stages.values())
    return {
        "synthetic_only": synthetic,
        "production_ready": ready,
        "serving_mode": "validated" if ready else "experimental",
        "pipeline_version": manifest.pipeline_version,
        "neural_stage_status": stages,
        "unvalidated_stages": unvalidated,
        "metrics_summary_present": has_metrics,
        "notice": None if ready else "Trained and evaluated on held-out synthetic scenarios only; real-world safety and venue transfer are unvalidated." if synthetic else EXPERIMENTAL_NOTICE.format(
            stages=", ".join(f"{s}: {stages[s]}" for s in unvalidated) or "no validation metrics recorded"),
    }
