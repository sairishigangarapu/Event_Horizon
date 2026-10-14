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

