"""Twin Studio v3: trusted replay clock, causal notices and leakage guarantees."""
import copy
import hashlib
import inspect
import json

import pytest
from fastapi.testclient import TestClient

from services.twin_studio.v3 import engine
from services.twin_studio.v3.notices import derive_notices
from services.twin_studio.v3.operator_side import safe_metadata, session_from_reports
from services.twin_studio.v3.replay import ModelView, OperatorKey, ReplayError
from services.twin_studio.v3.replay_service import create_app
from services.twin_studio.v3.zone_sensing import DEFAULT_ZONE_SOURCES, generate_zone_observations
from services.twin_studio.v3.zone_states import derive_zone_states

OP, MODEL = "operator-" + "o" * 30, "model-" + "m" * 30


def _scene():
    return json.loads((engine.ARCHIVE_DIR / "examples/event_horizon_scene.json").read_text(encoding="utf-8"))


def _reports(scene, mode="custom"):
    res = engine.simulate(scene, mode)
    states = derive_zone_states(res, scene)
    return engine.route_observations(scene, res) + generate_zone_observations(states, DEFAULT_ZONE_SOURCES, scene["seed"], 5)


def _session(scene, reports, sid="s1"):
    return session_from_reports(sid, reports, safe_metadata(scene), DEFAULT_ZONE_SOURCES)


@pytest.fixture(scope="module")
def scene():
    return _scene()


@pytest.fixture(scope="module")
def reports(scene):
    return _reports(scene)


def _advance_to(ctrl, key, t):
    while ctrl.watermark_s < t:
        ctrl.advance(key)


def test_model_view_has_no_time_argument():
    assert list(inspect.signature(ModelView.snapshot).parameters) == ["self"]
    assert not any("time" in n or "as_of" in n for n in inspect.signature(ModelView.__init__).parameters if n != "ctrl")


def test_snapshot_respects_both_cutoffs_and_deduplicates(scene, reports):
    ctrl, key = _session(scene, reports)
    view = ctrl.model_view()
    _advance_to(ctrl, key, 60)
    s1 = view.snapshot()
    assert s1.as_of_s == 60 and s1.measurements
    assert all(m.observed_at_s <= 60 and m.available_at_s <= 60 for m in s1.measurements)
    late = [r for r in reports if r["observed_at_s"] <= 60 < r["available_at_s"]]
    assert late and not {r["observation_id"] for r in late} & {m.report_id for m in s1.measurements}
    assert view.snapshot().new_report_ids == ()                      # nothing new without advancing
    ctrl.advance(key)
    s2 = view.snapshot()
    assert s2.new_report_ids and not set(s2.new_report_ids) & set(s1.new_report_ids)


def test_only_the_operator_key_advances(scene, reports):
    ctrl, key = _session(scene, reports)
    with pytest.raises(ReplayError):
        ctrl.advance(OperatorKey("s1"))                             # forged key
    with pytest.raises(ReplayError):
        ctrl.advance(None)
    assert ctrl.advance(key) == 5


def test_notices_are_causal_and_carry_no_truth(scene, reports):
    ctrl, key = _session(scene, reports)
    view = ctrl.model_view()
    _advance_to(ctrl, key, 305)
    snap = view.snapshot()
    visible = {m.report_id for m in snap.measurements}
    assert snap.notices
    for n in snap.notices:
        assert n.available_at_s <= snap.as_of_s and set(n.evidence_report_ids) <= visible
        blob = json.dumps(n.to_dict())
        for forbidden in ("eventId", "incident", "collaps", "surge", "end", scene["events"][0]["id"]):
            assert forbidden not in blob.replace("evidence", "")
    assert any(n.notice_type == "exit_reported_closed" and n.entity_id == "T_GATE_E" for n in snap.notices)


def test_changing_a_future_incident_does_not_change_earlier_snapshots(scene):
    base = _scene()
    changed = _scene()
    changed["events"].append({"id": "late-block", "name": "late block", "type": "collapse", "routeId": changed["routes"][0]["id"],
                              "at": 0.4, "start": 200, "end": 260, "people": 0, "radiusM": 4})
    a, b = _reports(base), _reports(changed)
    assert a != b                                                    # the incident does change later reports
    ca, ka = _session(base, a, "a")
    cb, kb = _session(changed, b, "b")
    va, vb = ca.model_view(), cb.model_view()
    while ca.watermark_s < 195:
        ca.advance(ka); cb.advance(kb)
        sa, sb = va.snapshot(), vb.snapshot()
        assert [m.__dict__ for m in sa.measurements] == [m.__dict__ for m in sb.measurements]
        assert [n.to_dict() for n in sa.notices] == [n.to_dict() for n in sb.notices]


def test_future_reports_never_reach_an_earlier_snapshot(scene, reports):
    tampered = copy.deepcopy(reports)
    for r in tampered:
        if r["available_at_s"] > 100:
            for k, v in r["values"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "count_window_s":
                    r["values"][k] = v + 3 if isinstance(v, int) else v * 2 + 1
    ca, ka = _session(scene, reports, "a")
    cb, kb = _session(scene, tampered, "b")
    _advance_to(ca, ka, 100); _advance_to(cb, kb, 100)
    sa, sb = ca.model_view().snapshot(), cb.model_view().snapshot()
    assert [m.__dict__ for m in sa.measurements] == [m.__dict__ for m in sb.measurements]


def test_derive_notices_ignores_unavailable_evidence(scene, reports):
    ctrl, key = _session(scene, reports)
    all_ms = ctrl._measurements
    early = derive_notices(all_ms, 150)
    assert all(n.available_at_s <= 150 for n in early)
    assert all(max(n.evidence_observed_at_s) <= 150 for n in early)


# ---------------------------------------------------------------- HTTP boundary

@pytest.fixture
def client(scene, reports):
    app, registry = create_app(operator_token_sha256=hashlib.sha256(OP.encode()).hexdigest(),
                               model_token_sha256=hashlib.sha256(MODEL.encode()).hexdigest())
    registry["s1"] = _session(scene, reports)
    return TestClient(app)


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def test_model_cannot_advance_or_choose_its_cutoff(client):
    assert client.post("/operator/sessions/s1/advance", headers=_h(MODEL)).status_code == 403
    assert client.post("/operator/sessions/s1/advance").status_code == 401
    assert client.get("/model/sessions/s1/snapshot?time_s=999", headers=_h(MODEL)).status_code == 400
    assert client.get("/model/sessions/s1/snapshot?as_of=999", headers=_h(MODEL)).status_code == 400
    r = client.get("/model/sessions/s1/snapshot", headers={**_h(MODEL), "X-Replay-Time": "999", "X-As-Of": "999"})
    assert r.status_code == 200 and r.json()["as_of_s"] == 0
    assert client.get("/model/sessions/s1/snapshot", headers=_h(OP)).status_code == 403


def test_operator_advances_and_model_sees_deduplicated_increments(client):
    for _ in range(4):
        assert client.post("/operator/sessions/s1/advance", headers=_h(OP)).status_code == 200
    first = client.get("/model/sessions/s1/snapshot", headers=_h(MODEL)).json()
    assert first["as_of_s"] == 20 and first["new_report_ids"]
    again = client.get("/model/sessions/s1/snapshot", headers=_h(MODEL)).json()
    assert again["new_report_ids"] == [] and len(again["measurements"]) == len(first["measurements"])


@pytest.mark.parametrize("path", ["/api/observations/" + "a" * 32 + "/custom?time_s=5", "/api/render", "/docs",
                                  "/openapi.json", "/api/runs/" + "a" * 32 + "/run_manifest.json", "/hidden_ground_truth"])
def test_no_truth_render_or_download_routes(client, path):
    assert client.get(path, headers=_h(MODEL)).status_code == 404
