"""Twin Studio v3: scenario variation and splits, staleness rule, trained artifact compatibility."""
import json
from pathlib import Path

import pytest

from services.twin_studio.v3.ingest import ZONE_SCHEMA, Measurement, SourceSpec
from services.twin_studio.v3.replay import AuthorizedSnapshot, topology_from_venue
from services.twin_studio.v3.scenarios import REGIMES, gate_positions, base_scene, sample_all, sample_scenario
from services.twin_studio.v3.staleness import StalenessPolicy, assess_snapshot, materially_different, score
from services.twin_studio.zones import load_venue

REPO = Path(__file__).resolve().parents[1]
V3_MODELS = REPO / "models/environments/eh_twin_studio/v3"
TREE_DIR = V3_MODELS / "trees/dt_bank_v3_topo_1__mc_v1_n16_seed2026"
VENUE = load_venue()
TOPO = topology_from_venue(VENUE)


# ---------------------------------------------------------------- Monte Carlo variation and splits

def test_scenarios_vary_parameters_not_only_seeds():
    scs = sample_all(6)
    assert len({s.params["duration_s"] for s in scs}) > 1
    assert len({json.dumps(s.params["route_counts"], sort_keys=True) for s in scs}) == len(scs)
    assert len({json.dumps(s.params["route_sources"], sort_keys=True) for s in scs}) == len(scs)
    by = {r: [s for s in scs if s.regime == r] for r in REGIMES}
    assert all(not s.params["events"] for s in by["normal"] + by["observation_outage"])
    assert all(any(e["type"] == "surge" for e in s.params["events"]) for s in by["surge"])
    assert all(any(e["type"] == "collapse" for e in s.params["events"]) for s in by["blockage"])
    assert all({e["type"] for e in s.params["events"]} == {"surge", "collapse"} for s in by["compound"])
    assert all(s.params["degraded_reporting"] for s in by["observation_outage"])
    assert any(p.startswith("gate:") for s in scs for p in s.params["collapse_placements"])


def test_sampling_is_reproducible_and_recorded():
    a, b = sample_scenario("blockage", 3, 16), sample_scenario("blockage", 3, 16)
    assert a.record() == b.record() and a.scene == b.scene
    rec = a.record()
    assert {"scenario_id", "regime", "split", "seed", "params", "zone_sources", "generator_version"} <= set(rec)


def test_splits_are_whole_scenarios_stratified_by_regime():
    scs = sample_all(16)
    for r in REGIMES:
        splits = [s.split for s in scs if s.regime == r]
        assert splits.count("train") == 9 and splits.count("val") == 3 and splits.count("test") == 4
    assert len({s.scenario_id for s in scs}) == len(scs)          # one split per scenario, never per timestamp


def test_gate_positions_lie_on_routes_inside_gate_zones():
    pos = gate_positions(base_scene())
    assert {g for _, g, _ in pos} <= {"T_GATE_NW", "T_GATE_SW", "T_GATE_E"} and pos
    assert all(0 < at < 1 for _, _, at in pos)


# ---------------------------------------------------------------- staleness rule

SRC = (SourceSpec("zcam_1", "zone_camera", 5, 2, frozenset(TOPO.zone_ids), ZONE_SCHEMA),
       SourceSpec("zcam_2", "zone_camera", 10, 4, frozenset(TOPO.zone_ids), ZONE_SCHEMA))


def _m(source, var, value, observed, zone="T_COURTYARD"):
    delay = 2 if source == "zcam_1" else 4
    return Measurement(f"{source}:{zone}:{observed}", source, "zone_camera", "zone", zone, var, value, "u",
                       observed, observed + delay, ZONE_SCHEMA)


def _assess(ms, T, **kw):
    snap = AuthorizedSnapshot("s", T, tuple(ms), (), (), TOPO, SRC)
    return {(a.source_id, a.variable): a for a in assess_snapshot(snap, StalenessPolicy(age_multiplier=2.0), **kw)}


def test_fresh_report_needs_no_action_and_age_limit_is_delay_plus_k_intervals():
    a = _assess([_m("zcam_1", "density", 1.0, 100)], 105)[("zcam_1", "density")]
    assert a.max_age_s == 2 + 2 * 5 and a.state == "FRESH" and a.recommended_action == "NO_ACTION"
    old = _assess([_m("zcam_1", "density", 1.0, 100)], 120)[("zcam_1", "density")]
    assert old.state == "STALE_BY_AGE" and old.recommended_action == "VERIFY"
    assert old.reason_codes[0]["code"] == "AGE_EXCEEDS_LIMIT" and old.reason_codes[0]["age_s"] == 20


def test_conflict_and_newer_evidence_are_separate_from_age():
    ms = [_m("zcam_1", "density", 1.0, 100), _m("zcam_2", "density", 3.0, 100)]
    a = _assess(ms, 105)
    assert a[("zcam_1", "density")].conflict_status == "CONFLICT"
    assert a[("zcam_1", "density")].state == "CONFLICTING_EVIDENCE" and a[("zcam_1", "density")].recommended_action == "REVERIFY"
    newer = _assess([_m("zcam_2", "density", 1.0, 90), _m("zcam_1", "density", 2.5, 105)], 108)
    old = newer[("zcam_2", "density")]
    assert old.current_state_disagreement is True and old.state == "POTENTIALLY_OUTDATED"
    assert any(r["code"] == "NEWER_REPORT_DIFFERS" for r in old.reason_codes)
    agree = _assess([_m("zcam_2", "density", 1.0, 90), _m("zcam_1", "density", 1.05, 105)], 108)
    assert agree[("zcam_2", "density")].current_state_disagreement is False


def test_unvalidated_forecasts_never_confirm_outdatedness():
    ms = [_m("zcam_1", "density", 1.0, 100)]
    fc = {("zone", "T_COURTYARD", "density"): (4.0, "m")}
    a = _assess(ms, 105, forecasts=fc, forecasts_validated=False)[("zcam_1", "density")]
    assert a.forecast_change_warning == "UNAVAILABLE_MODEL_UNVALIDATED" and a.state == "FRESH"
    b = _assess(ms, 105, forecasts=fc, forecasts_validated=True)[("zcam_1", "density")]
    assert b.forecast_change_warning == "WARNING" and b.state == "FORECAST_CHANGE_WARNING" and b.recommended_action == "VERIFY"


def test_tolerances_and_scoring():
    assert not materially_different("density", 1.0, 1.2) and materially_different("density", 1.0, 1.5)
    assert materially_different("movement_category", "slow", "moving") and not materially_different("exit_open", 1.0, 1.0)
    s = score([(True, True), (True, False), (False, True), (False, False)])
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["false_positive_rate"] == 0.5


# ---------------------------------------------------------------- trained artifacts

@pytest.mark.skipif(not TREE_DIR.is_dir(), reason="trained v3 tree bank not present")
def test_tree_bank_matches_feature_schema_and_records_honest_status():
    from services.twin_studio.v3 import trees
    from services.twin_studio.v3.features import feature_names
    bank = trees.load(TREE_DIR)
    assert len(bank.models) == 6 and bank.version == "dt_bank_v3_topo_1"
    rep = json.loads((TREE_DIR / "tree_report.json").read_text(encoding="utf-8"))
    assert rep["feature_schema"] == "eh-v3-topo-features-1"
    assert set(rep["splits"]["train"]).isdisjoint(rep["splits"]["test"]) and set(rep["splits"]["val"]).isdisjoint(rep["splits"]["test"])
    val = json.loads((TREE_DIR / "tree_validation.json").read_text(encoding="utf-8"))
    assert set(val["validated_channels"]) <= set(rep["selection"]) and val["validation_scenarios"] == rep["splits"]["val"]


def _neural_dirs():
    root = V3_MODELS / "neural"
    return sorted(root.glob("*")) if root.is_dir() else []


@pytest.mark.skipif(not _neural_dirs(), reason="trained v3 neural checkpoint not present")
def test_neural_checkpoint_matches_twin_graph_and_never_overclaims():
    from services.twin_studio.v3 import neural
    model, rep = neural.load(_neural_dirs()[-1])
    assert model.enc.graph.map_version == "twin_venue_v1" and list(model.enc.graph.zone_ids) == list(TOPO.zone_ids)
    status = model.enc.hypernet.metadata["training_status"]
    assert status == rep["training_status"] == model.enc.gnn.metadata["training_status"]
    assert status in ("trained_validated", "trained_not_validated")
    if status == "trained_validated":
        assert len(rep["validation_wins_30s"]) >= 3
    else:
        assert len(rep["validation_wins_30s"]) < 3
    assert not model.temporal.training
    assert set(rep["splits"]["train"]).isdisjoint(rep["splits"]["test"])
