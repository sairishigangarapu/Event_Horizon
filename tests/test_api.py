from fastapi.testclient import TestClient
import pytest
from crowd_reliability.api import app
from crowd_reliability.cli import cmd_db_init

client = TestClient(app)

def setup_module(module):
    cmd_db_init()

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data

def test_api_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) >= 3
    ids = [s["scenario_id"] for s in scenarios]
    assert "crowd_surge" in ids
    assert "bottleneck_exit_closure" in ids

def test_api_venue():
    response = client.get("/api/venue")
    assert response.status_code == 200
    data = response.json()
    assert "zones" in data
    assert "connections" in data
    assert len(data["zones"]) >= 7

def test_api_manifest():
    response = client.get("/api/manifest")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_version" in data
    assert "tree_version" in data

def test_api_forecasts_and_assessments():
    response = client.get("/api/forecasts?scenario_id=crowd_surge")
    assert response.status_code == 200
    data = response.json()
    assert "comparisons" in data
    assert "total_comparisons" in data
    assert "limit" in data
    assert "offset" in data

    # Test pagination parameters
    paged_resp = client.get("/api/forecasts?scenario_id=crowd_surge&limit=10&offset=5")
    assert paged_resp.status_code == 200
    paged_data = paged_resp.json()
    assert paged_data["limit"] == 10
    assert paged_data["offset"] == 5
    assert len(paged_data["comparisons"]) <= 10

    resp_ass = client.get("/api/assessments?scenario_id=crowd_surge")
    assert resp_ass.status_code == 200

def test_api_assessments_invalid_timestamp():
    # Hostile input: malformed ISO timestamp should return 422, not unhandled 500
    resp = client.get("/api/assessments?scenario_id=crowd_surge&evaluated_at=not-a-valid-date")
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_api_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["tree_version"] == "dt_bank_v2"  # scaler-aware retraining now ships production v2


def test_api_tree_outputs_route():
    # P2 Section-G route: deterministic six-channel D[z,t] at as_of (or latest).
    resp = client.get("/api/scenarios/crowd_surge/tree-outputs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tree_version"] == "dt_bank_v2"
    assert data["scaler_version"] == "scaler_obs_only_v1"
    assert len(data["channel_metadata"]) == 6
    assert [c["name"] for c in data["channel_metadata"]] == [
        "estimated_density", "estimated_speed", "estimated_inflow",
        "estimated_outflow", "estimated_queue_length", "probability_exit_open",
    ]
    zones = data["zones"]
    # If seeded data is present, every zone carries all six channels + masks.
    if zones:
        z = zones[0]
        assert len(z["channel_values"]) == 6
        assert len(z["channel_masks"]) == 6
        # Exit channel applicability: 0 outside exits, 1 inside exits.
        exit_masks = [z2["channel_masks"][5] for z2 in zones]
        assert set(exit_masks) <= {0, 1}
        assert 1 in exit_masks  # the venue has at least two exits


def test_api_tree_outputs_invalid_as_of():
    resp = client.get("/api/scenarios/crowd_surge/tree-outputs?as_of=not-a-date")
    assert resp.status_code == 422
    assert "detail" in resp.json()



def test_api_tree_outputs_route_with_observations():
    # Exercises the populated branch that the empty-database test above skips:
    # real observations -> six channels per zone, exit applicability masks,
    # and as_of isolation (a report received after as_of must not change output).
    from datetime import datetime, timedelta
    from crowd_reliability.database import SessionLocal
    from crowd_reliability.models.schema import ObservationRecord

    scenario = "test_tree_outputs_seeded"
    t0 = datetime(2026, 1, 1, 0, 1, 0)
    as_of = t0 + timedelta(seconds=10)

    def obs(report_id, zone, variable, value, unit, received_at):
        return ObservationRecord(
            report_id=report_id, scenario_id=scenario, zone_id=zone, source_id="test_sensor",
            variable=variable, value=value, unit=unit,
            observed_at=received_at - timedelta(seconds=3),
            published_at=received_at - timedelta(seconds=2),
            received_at=received_at,
        )

    db = SessionLocal()
    try:
        db.add_all([
            obs("r1", "concourse", "density", 2.5, "p/m2", t0),
            obs("r2", "concourse", "speed", 0.4, "m/s", t0),
            obs("r3", "exit_1", "exit_open", 0.0, "binary", t0),
            obs("r4", "exit_1", "queue_length", 30.0, "persons", t0),
        ])
        db.commit()

        url = f"/api/scenarios/{scenario}/tree-outputs?as_of={as_of.isoformat()}"
        first = client.get(url)
        assert first.status_code == 200
        data = first.json()
        assert data["tree_version"] == "dt_bank_v2"
        assert data["evaluation_time"] == as_of.isoformat()

        zones = {z["zone_id"]: z for z in data["zones"]}
        assert "concourse" in zones and "exit_1" in zones
        for z in zones.values():
            assert z["channel_names"] == [c["name"] for c in data["channel_metadata"]]
            assert len(z["channel_values"]) == 6 and len(z["channel_masks"]) == 6
        assert zones["exit_1"]["channel_masks"][5] == 1
        assert zones["concourse"]["channel_masks"][5] == 0
        assert zones["concourse"]["channel_values"][5] == 0.0

        # Deterministic, and a report received after as_of is invisible.
        db.add(obs("r5", "concourse", "density", 9.0, "p/m2", as_of + timedelta(seconds=30)))
        db.commit()
        second = client.get(url)
        assert second.status_code == 200
        assert second.json() == data
    finally:
        db.query(ObservationRecord).filter(ObservationRecord.scenario_id == scenario).delete()
        db.commit()
        db.close()
