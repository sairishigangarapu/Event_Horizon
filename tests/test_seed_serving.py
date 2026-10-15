"""seed-serving must populate every scenario from committed models without modifying them."""
import os
from datetime import datetime

import pytest

from crowd_reliability.cli import _model_dir_hashes, cmd_db_init, cmd_seed_serving
from crowd_reliability.database import SessionLocal
from crowd_reliability.models.schema import (
    ForecastRecord, ModelManifestRecord, ObservationRecord, ReferenceStateRecord, TreeOutputRecord,
)
from crowd_reliability.simulation.scenarios import SCENARIO_PRESETS

MODEL_DIR = os.path.join("models", "production")


def _clear_scenarios(db):
    ids = list(SCENARIO_PRESETS.keys())
    for model in (ObservationRecord, ReferenceStateRecord, ForecastRecord, TreeOutputRecord):
        db.query(model).filter(model.scenario_id.in_(ids)).delete(synchronize_session=False)
    db.query(ModelManifestRecord).delete()
    db.commit()


def test_seed_serving_populates_all_scenarios_and_leaves_models_untouched(tmp_path):
    cmd_db_init()
    db = SessionLocal()
    try:
        # A leftover row from an earlier seeding must be replaced, not duplicated.
        db.add(ObservationRecord(
            report_id="stale_leftover", scenario_id="crowd_surge", zone_id="concourse",
            source_id="old", variable="density", value=1.0, unit="p/m2",
            observed_at=datetime(2020, 1, 1), published_at=datetime(2020, 1, 1),
            received_at=datetime(2020, 1, 1),
        ))
        db.commit()

        before = _model_dir_hashes(MODEL_DIR)
        csv_path = str(tmp_path / "central.csv")
        cmd_seed_serving(seed=42, central_csv_path=csv_path)
        assert _model_dir_hashes(MODEL_DIR) == before
        assert os.path.getsize(csv_path) > 0

        assert db.query(ObservationRecord).filter_by(report_id="stale_leftover").count() == 0
        for sc in SCENARIO_PRESETS:
            for stream in ("stream_a_central", "stream_b_architecture"):
                assert db.query(ForecastRecord).filter_by(scenario_id=sc, stream_id=stream).count() > 0, (sc, stream)
            assert db.query(TreeOutputRecord).filter_by(scenario_id=sc).count() > 0
            assert db.query(ReferenceStateRecord).filter_by(scenario_id=sc).count() > 0

        active = db.query(ModelManifestRecord).filter_by(is_active=True).all()
        assert len(active) == 1 and active[0].model_hashes
    finally:
        _clear_scenarios(db)
        db.close()


def test_seed_serving_refuses_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing model artifacts"):
        cmd_seed_serving(model_dir=str(tmp_path), central_csv_path=str(tmp_path / "c.csv"))
