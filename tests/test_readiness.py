"""Model readiness labelling, strict serving mode, and fail-loud artifact loading."""
import os
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from crowd_reliability import api as api_module
from crowd_reliability.config import settings
from crowd_reliability.ml.pipeline import (
    ArchitecturePipeline, ModelArtifactError, load_manifest, load_verified_tree_bank,
)
from crowd_reliability.ml.tree_bank import DecisionTreeBank
from crowd_reliability.models.manifest import (
    VALIDATED_STATUS, ModelNotReadyError, PipelineManifest, assess_readiness,
)
from crowd_reliability.simulation.scenarios import get_default_venue

PRODUCTION_DIR = os.path.join("models", "production")


@pytest.fixture
def saved_dir(tmp_path):
    """A complete, valid artifact set (fitted production tree bank + fresh neural stages)."""
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    pipeline.tree_bank = load_verified_tree_bank(PRODUCTION_DIR)
    pipeline.save_pipeline(str(tmp_path))
    return tmp_path


def _load(path):
    ArchitecturePipeline(venue=get_default_venue()).load_pipeline(str(path))


# ------------------------------------------------------------------ readiness

def test_production_manifest_is_reported_experimental():
    report = assess_readiness(load_manifest(PRODUCTION_DIR))
    assert report["production_ready"] is False
    assert report["serving_mode"] == "experimental"
    assert set(report["unvalidated_stages"]) == {"hypernet", "gnn", "temporal"}
    assert "not trained and validated" in report["notice"]


def test_validated_manifest_requires_all_stages_and_metrics():
    m = PipelineManifest(training_status={s: VALIDATED_STATUS for s in ("hypernet", "gnn", "temporal")})
    assert assess_readiness(m)["production_ready"] is False  # no metrics recorded yet
    m.metrics_summary = {"density_mae_30s": 0.1}
    report = assess_readiness(m)
    assert report["production_ready"] is True and report["notice"] is None
    m.training_status["gnn"] = "untrained_initial"
    assert assess_readiness(m)["unvalidated_stages"] == ["gnn"]


def test_require_ready_only_raises_in_strict_mode():
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    pipeline.load_pipeline(PRODUCTION_DIR)
    assert pipeline.require_ready(False)["serving_mode"] == "experimental"
    with pytest.raises(ModelNotReadyError):
        pipeline.require_ready(True)


# ----------------------------------------------------------- fail-loud loading

def test_complete_artifact_set_loads(saved_dir):
    _load(saved_dir)


@pytest.mark.parametrize("victim", ["manifest.json", "tree_bank.joblib", "temporal_model.pt"])
def test_missing_root_artifact_raises(saved_dir, victim):
    os.remove(saved_dir / victim)
    with pytest.raises(ModelArtifactError, match="Missing model artifact"):
        _load(saved_dir)


def test_missing_spatial_pair_raises(saved_dir):
    m = PipelineManifest.load(str(saved_dir / "manifest.json"))
    shutil.rmtree(saved_dir / m.spatial_artifact_dir)
    with pytest.raises(ModelArtifactError, match="Missing model artifact"):
        _load(saved_dir)
    m.spatial_artifact_dir = None
    m.save(str(saved_dir / "manifest.json"))
    with pytest.raises(ModelArtifactError, match="no spatial_artifact_dir"):
        _load(saved_dir)


def test_tree_bank_version_mismatch_raises(saved_dir):
    m = PipelineManifest.load(str(saved_dir / "manifest.json"))
    m.tree_version = "dt_bank_v999"
    m.save(str(saved_dir / "manifest.json"))
    with pytest.raises(ModelArtifactError, match="does not match manifest tree_version"):
        _load(saved_dir)


def test_unfitted_tree_bank_is_never_loaded_or_saved(saved_dir):
    bank = DecisionTreeBank()
    with pytest.raises(ModelArtifactError, match="not fitted"):
        pipeline = ArchitecturePipeline(venue=get_default_venue())
        pipeline.save_pipeline(str(saved_dir / "other"))
    # An unfitted bank written behind the manifest's back fails its hash, then its fit check.
    bank.save(str(saved_dir / "tree_bank.joblib"))
    with pytest.raises(ModelArtifactError, match="integrity verification failed"):
        _load(saved_dir)


# ------------------------------------------------------------------------ API

@pytest.fixture
def client():
    with TestClient(api_module.app) as c:  # runs the app lifespan, which creates the tables
        yield c


def test_api_and_pipeline_serve_the_same_tree_bank(client):
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    pipeline.load_pipeline(PRODUCTION_DIR)
    api_bank = api_module._load_tree_bank()
    rng = np.random.default_rng(0)
    for _ in range(25):
        row = rng.uniform(0, 3, size=18).astype(np.float32)
        assert api_bank.predict_zone(row, has_exit=True) == pipeline.tree_bank.predict_zone(row, has_exit=True)
    for name, tree in api_bank.trees.items():
        other = pipeline.tree_bank.trees[name].tree_
        for attr in ("feature", "threshold", "value"):
            assert np.array_equal(getattr(tree.tree_, attr), getattr(other, attr)), (name, attr)


def test_health_and_forecasts_report_experimental_status(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["serving_mode"] == "experimental"
    assert health["readiness"]["production_ready"] is False
    forecasts = client.get("/api/forecasts?scenario_id=crowd_surge").json()
    assert forecasts["stream_b_status"]["serving_mode"] == "experimental"
    manifest = client.get("/api/manifest").json()
    assert manifest["training_status"]["temporal"] == "untrained_initial"


def test_strict_mode_refuses_unvalidated_forecasts(client, monkeypatch):
    monkeypatch.setattr(settings, "require_validated_models", True)
    resp = client.get("/api/forecasts?scenario_id=crowd_surge")
    assert resp.status_code == 503
    assert "not trained and validated" in resp.json()["detail"]


def test_missing_artifacts_degrade_health_and_block_tree_outputs(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    health = client.get("/health").json()
    assert health["status"] == "degraded" and health["model_loaded"] is False and health["errors"]
    assert client.get("/api/scenarios/crowd_surge/tree-outputs").status_code == 503
    assert client.get("/api/forecasts?scenario_id=crowd_surge").status_code == 503


def test_seed_serving_strict_mode_stops_before_writing(monkeypatch, tmp_path):
    from crowd_reliability.cli import cmd_seed_serving
    monkeypatch.setattr(settings, "require_validated_models", True)
    csv_path = tmp_path / "central.csv"
    with pytest.raises(ModelNotReadyError):
        cmd_seed_serving(central_csv_path=str(csv_path))
    assert not csv_path.exists()
