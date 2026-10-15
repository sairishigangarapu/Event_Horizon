"""Phase 2 tests: gate closure + Python/JS parity, zone-level states, causal observations."""
import copy
import importlib.util
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from services.twin_studio.engine import simulation as S
from services.twin_studio.observations import (FORBIDDEN_FIELDS, ObservationConfig, RunContext,
                                               generate_observations)
from services.twin_studio.zone_states import (QUEUE_SPEED_THRESHOLD_M_S, derive_zone_states,
                                              gate_open_at_instant, step_transitions)
from services.twin_studio.zones import OBSTACLE_KEY, OUTSIDE_KEY, ZoneAssigner, load_venue

ENGINE_DIR = Path(__file__).resolve().parents[1] / "services" / "twin_studio" / "engine"
NODE = shutil.which("node")
VENUE = load_venue()
ASSIGNER = ZoneAssigner(VENUE)
GATES = [{"id": z["zone_id"], **{k: z["geometry"][k] for k in "xywd"}} for z in VENUE["zones"] if z["zone_type"] == "exit"]
GATE_RECT = {g["id"]: g for g in GATES}
MODES = ("ideal", "custom", "random")

R_E_GATE = {"id": "r_e", "name": "Annex to E gate", "points": [{"x": 800, "y": 580}, {"x": 968, "y": 580}],
            "count": 60, "speed": 1.4, "width": 3}
R_NW_FAST = {"id": "r_nw", "name": "Fast across NW gate", "points": [{"x": 335, "y": 10}, {"x": 335, "y": 150}],
             "count": 30, "speed": 3.0, "width": 2}
R_COURT = {"id": "r_ct", "name": "Ends in courtyard", "points": [{"x": 300, "y": 300}, {"x": 400, "y": 350}],
           "count": 20, "speed": 1.2, "width": 3}
R_COURT_TO_SOUTH = {"id": "r_cs", "name": "Courtyard to south", "points": [{"x": 450, "y": 500}, {"x": 450, "y": 640}],
                    "count": 25, "speed": 1.3, "width": 3}


def scene(routes, events=(), duration=90, seed=5, gates=True):
    c = {"duration": duration, "scale": .1, "seed": seed, "randomCount": 1, "routes": list(routes), "events": list(events)}
    if gates:
        c["gates"] = GATES
    return c


OPEN = scene([R_E_GATE, R_NW_FAST, R_COURT, R_COURT_TO_SOUTH])
CLOSURES = scene([R_E_GATE, R_NW_FAST, R_COURT, R_COURT_TO_SOUTH], [
    {"id": "g_e", "name": "E gate closed", "type": "gate_closure", "gateId": "T_GATE_E", "start": 20, "end": 50},
    {"id": "g_nw", "name": "NW gate closed", "type": "gate_closure", "gateId": "T_GATE_NW", "start": 0, "end": 60}])
STUCK = scene([R_E_GATE], [{"id": "g", "name": "E shut late", "type": "gate_closure", "gateId": "T_GATE_E",
                             "start": 10, "end": 120}], duration=120)


# ---------------------------------------------------------------- Python / JavaScript parity

def run_js(config, modes=MODES, engine="engine.js"):
    script = ("const E=require(process.argv[1]);const c=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "const out={};for(const m of JSON.parse(process.argv[2]))out[m]=E.simulate(c,m);console.log(JSON.stringify(out))")
    raw = subprocess.check_output([NODE, "-e", script, str(ENGINE_DIR / engine), json.dumps(list(modes))],
                                  input=json.dumps(config).encode())
    return json.loads(raw)


def js_validate(config):
    script = ("const E=require(process.argv[1]);const c=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(E.validate(c)))")
    return json.loads(subprocess.check_output([NODE, "-e", script, str(ENGINE_DIR / "engine.js")],
                                              input=json.dumps(config).encode()))


def assert_same(a, b, path="$"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {set(a) ^ set(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        # JSON serialisation is the only boundary: JS integers like 0 arrive as int, Python may hold 0.0
        assert abs(a - b) <= 1e-9, f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


needs_node = pytest.mark.skipif(NODE is None, reason="node is required for the JavaScript parity tests")


@needs_node
@pytest.mark.parametrize("name,config", [("open_throughout", OPEN), ("closures", CLOSURES), ("stuck", STUCK)])
def test_python_javascript_parity(name, config):
    js = run_js(config)
    for mode in MODES:
        assert_same(S.simulate(config, mode), js[mode], f"{name}.{mode}")


@needs_node
@pytest.mark.parametrize("mutate", [
    lambda c: c["events"].append({"id": "x", "name": "Bad gate", "type": "gate_closure", "gateId": "NOPE", "start": 1, "end": 5}),
    lambda c: c["events"].append({"id": "x", "name": "Reversed", "type": "gate_closure", "gateId": "T_GATE_E", "start": 9, "end": 5}),
    lambda c: c["events"].append({"id": "x", "name": "Too late", "type": "gate_closure", "gateId": "T_GATE_E", "start": 5, "end": 999}),
    lambda c: c["events"].append({"id": "x", "name": "On a route", "type": "gate_closure", "gateId": "T_GATE_E", "start": 1, "end": 5, "routeId": "r_e"}),
    lambda c: c["events"].append({"id": "x", "name": "With people", "type": "gate_closure", "gateId": "T_GATE_E", "start": 1, "end": 5, "people": 3}),
    lambda c: c["events"].append({"id": "x", "name": "Unknown", "type": "teleport", "routeId": "r_e", "at": .5, "start": 1, "end": 5}),
    lambda c: c.update(gates=[{"id": "G", "x": 990, "y": 10, "w": 50, "d": 5}]),
    lambda c: c.update(gates=GATES + [dict(GATES[0])]),
    lambda c: (c["routes"].append({"id": "r_in", "name": "Spawns in E gate", "points": [{"x": 962, "y": 570}, {"x": 900, "y": 570}],
                                   "count": 5, "speed": 1, "width": 2}),
               c["events"].append({"id": "x", "name": "E shut", "type": "gate_closure", "gateId": "T_GATE_E", "start": 1, "end": 5})),
])
def test_invalid_incident_configurations_rejected_by_both_engines(mutate):
    bad = copy.deepcopy(OPEN)
    mutate(bad)
    with pytest.raises(ValueError):
        S.simulate(bad, "custom")
    assert js_validate(bad), "JavaScript engine accepted an invalid configuration"


def _load_original_python():
    spec = importlib.util.spec_from_file_location("twin_original_simulation", ENGINE_DIR / "original" / "simulation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORIGINAL_KEYS = ("mode", "events", "frames", "total", "completed", "peakQueue")
ORIGINAL_ROW_KEYS = ("run", "seed", "time_s", "route_id", "route_name", "spawned", "exited", "active", "completed",
                     "mean_speed_m_s", "mean_density_p_m2", "peak_local_density_p_m2", "queued", "net_flow")


def _original_view(result):
    view = {k: result[k] for k in ORIGINAL_KEYS}
    view["rows"] = [{k: r[k] for k in ORIGINAL_ROW_KEYS} for r in result["rows"]]
    return view


@needs_node
def test_without_gates_modified_engines_match_the_original_engines():
    plain = scene([R_E_GATE, R_NW_FAST, R_COURT], [
        {"id": "c", "name": "Collapse", "type": "collapse", "routeId": "r_e", "at": .5, "start": 10, "end": 30},
        {"id": "s", "name": "Surge", "type": "surge", "routeId": "r_ct", "at": .2, "start": 5, "end": 15, "people": 10}],
        gates=False)
    original = _load_original_python()
    js_orig = run_js(plain, engine="original/engine.js")
    js_mod = run_js(plain)
    for mode in MODES:
        mod = S.simulate(plain, mode)
        assert_same(_original_view(mod), _original_view(original.simulate(plain, mode)), f"py.{mode}")
        assert_same(_original_view(js_mod[mode]), _original_view(js_orig[mode]), f"js.{mode}")


# ---------------------------------------------------------------- gate-closure semantics

def test_gate_state_follows_schedule_with_documented_boundaries():
    res = S.simulate(CLOSURES, "custom")
    for t, flags in enumerate(res["gateOpen"]):
        assert flags["T_GATE_E"] == (not 20 <= t < 50)
        assert flags["T_GATE_NW"] == (not 0 <= t < 60)
        assert flags["T_GATE_SW"] is True
        for gid in flags:
            assert flags[gid] == gate_open_at_instant(res["events"], gid, t)
    assert all(all(f.values()) for f in S.simulate(CLOSURES, "ideal")["gateOpen"])  # ideal has no incidents


def _inside(gid, a):
    g = GATE_RECT[gid]
    return g["x"] <= a["x"] < g["x"] + g["w"] and g["y"] <= a["y"] < g["y"] + g["d"]


def test_no_agent_enters_a_closed_gate():
    res = S.simulate(CLOSURES, "custom")
    for gid, (start, end) in {"T_GATE_E": (20, 50), "T_GATE_NW": (0, 60)}.items():
        already = {a["id"] for a in res["frames"][start - 1] if _inside(gid, a)} if start > 0 else set()
        for k in range(start, end):     # frames[k] = state after step k, which is governed by the closure
            assert {a["id"] for a in res["frames"][k] if _inside(gid, a)} <= already
        assert not [c for c in res["completions"] if c["gateId"] == gid and start <= c["t"] < end]


def test_closed_narrow_gate_cannot_be_skipped_between_frames():
    """The NW gate is 1.2 m deep; fast agents move up to ~3.45 m per step, so in the open run some
    agents jump over it between frames. While closed, none get past it."""
    open_res = S.simulate(OPEN, "custom")
    jumped = False
    for k in range(1, len(open_res["frames"])):
        prev = {a["id"]: a for a in open_res["frames"][k - 1] if a["ri"] == 1}
        for a in open_res["frames"][k]:
            if a["ri"] == 1 and a["id"] in prev and prev[a["id"]]["y"] < 55 and a["y"] >= 67:
                jumped = True
    assert jumped, "fixture must contain a frame-to-frame jump over the open gate"
    closed = S.simulate(CLOSURES, "custom")
    for k in range(0, 60):
        assert all(a["y"] < 55 for a in closed["frames"][k] if a["ri"] == 1)
    assert any(a["ri"] == 1 and a["y"] >= 67 for k in range(60, 91) for a in closed["frames"][k])  # passes after reopening


def test_agents_queue_before_a_closed_gate_and_resume():
    res = S.simulate(CLOSURES, "custom")
    gate_x = GATE_RECT["T_GATE_E"]["x"]
    waiting = [a for a in res["frames"][45] if a["ri"] == 0]
    assert waiting and all(a["x"] < gate_x or _inside("T_GATE_E", a) for a in waiting)
    assert any(a["v"] < QUEUE_SPEED_THRESHOLD_M_S for a in waiting)
    assert [c for c in res["completions"] if c["gateId"] == "T_GATE_E" and c["t"] >= 50]


def test_completion_classification():
    res = S.simulate(OPEN, "ideal")
    by_route = defaultdict(set)
    for c in res["completions"]:
        by_route[c["ri"]].add(c["gateId"])
    assert by_route[0] == {"T_GATE_E"}           # completes inside the E gate: a real exit
    assert by_route[1] == {None}                 # crosses the NW gate but ends in the entrance
    assert by_route[2] == {None}                 # ends in the courtyard
    assert res["gateExits"] + res["routeCompletedNotExit"] == res["completed"]
    for row in res["rows"]:
        assert row["gate_exits"] + row["route_completed_not_exit"] == row["exited"]


# ---------------------------------------------------------------- zone-level states

@pytest.fixture(scope="module")
def closure_states():
    res = S.simulate(CLOSURES, "custom")
    return res, derive_zone_states(res, VENUE, 5), derive_zone_states(res, VENUE, 1)


def test_every_agent_accounted_for_exactly_once(closure_states):
    res, five, one = closure_states
    for st in one + five:
        acc = st["accounting"]
        assert acc["spawned_cumulative"] == (acc["in_zones"] + acc[OBSTACLE_KEY] + acc[OUTSIDE_KEY]
                                             + acc["gate_exits_cumulative"] + acc["route_completed_not_exit_cumulative"])
    assert one[-1]["accounting"]["spawned_cumulative"] == res["total"]


def test_occupancy_density_and_queue_match_agent_frames(closure_states):
    res, five, _ = closure_states
    area = {z["zone_id"]: z["effective_walkable_area_m2"] for z in VENUE["zones"]}
    for st in five[1:]:
        frame = res["frames"][st["simulation_second"] - 1]
        counts = ASSIGNER.count((a["x"], a["y"]) for a in frame)
        for zid, z in st["zones"].items():
            assert z["occupancy"] == counts[zid]
            assert z["density"] == pytest.approx(counts[zid] / area[zid])
            present = [a for a in frame if ASSIGNER.assign(a["x"], a["y"]).zone_id == zid]
            assert z["queue_length"] == sum(a["v"] < QUEUE_SPEED_THRESHOLD_M_S for a in present)
        assert st["accounting"][OUTSIDE_KEY] == counts[OUTSIDE_KEY]


def test_speed_validity_empty_versus_stationary():
    res = S.simulate(STUCK, "custom")
    states = derive_zone_states(res, VENUE, 5)
    for st in states:
        for z in st["zones"].values():
            assert (z["speed"] is None) == (z["occupancy"] == 0 or st["simulation_second"] == 0)
            assert z["speed_valid"] == (z["speed"] is not None)
    stopped = [st["zones"]["T_ANNEX"] for st in states if st["simulation_second"] >= 110]
    # Engine stop-line arithmetic leaves ~1e-16 m/s residues; truth is not rounded, so compare with a tolerance.
    assert stopped and all(z["occupancy"] > 0 and z["speed"] == pytest.approx(0.0, abs=1e-9) and z["speed_valid"] for z in stopped)
    assert all(z["queue_length"] == z["occupancy"] for z in stopped)


def test_flows_conserve_occupancy_at_every_step(closure_states):
    res, _, _ = closure_states
    prev = defaultdict(int)
    for st in step_transitions(res, ASSIGNER):
        now = defaultdict(int)
        for zone in st["now"].values():
            now[zone] += 1
        for zone in set(prev) | set(now) | set(st["inflow"]) | set(st["outflow"]):
            assert now[zone] - prev[zone] == st["inflow"].get(zone, 0) - st["outflow"].get(zone, 0), (st["step"], zone)
        prev = now


def test_inflow_and_outflow_are_real_boundary_crossings():
    res = S.simulate(scene([R_COURT_TO_SOUTH], duration=60), "ideal")
    states = derive_zone_states(res, VENUE, 5)
    crossed = set()
    for k in range(1, len(res["frames"])):
        prev = {a["id"]: ASSIGNER.assign(a["x"], a["y"]).zone_id for a in res["frames"][k - 1]}
        for a in res["frames"][k]:
            if prev.get(a["id"]) == "T_COURTYARD" and ASSIGNER.assign(a["x"], a["y"]).zone_id == "T_WALK_S":
                crossed.add(a["id"])
    out_court = sum(st["zones"]["T_COURTYARD"]["outflow_count"] or 0 for st in states)
    in_south = sum(st["zones"]["T_WALK_S"]["inflow_count"] or 0 for st in states)
    assert out_court == in_south == len(crossed) > 0
    assert all(st["zones"]["T_WALK_N"]["inflow_count"] in (0, None) for st in states)


def test_route_completion_outside_a_gate_is_never_a_gate_exit(closure_states):
    res, five, _ = closure_states
    for st in five[1:]:
        for zid, z in st["zones"].items():
            if not z["exit_applicable"]:
                assert z["gate_exits"] == 0
    total_not_exit = sum(z["route_completed_not_exit"] or 0 for st in five for z in st["zones"].values())
    total_gate = sum(z["gate_exits"] or 0 for st in five for z in st["zones"].values())
    covered = [c for c in res["completions"] if c["t"] < five[-1]["simulation_second"]]  # 5 s windows end at the last grid instant
    in_zone = [c for c in covered if ASSIGNER.assign(c["x"], c["y"]).zone_id is not None]
    assert total_gate == sum(c["gateId"] is not None for c in in_zone)
    assert total_not_exit == sum(c["gateId"] is None for c in in_zone)
    assert all(c["gateId"] is None for c in covered if ASSIGNER.assign(c["x"], c["y"]).zone_id is None)


def test_trailing_partial_interval_is_outside_the_five_second_grid(closure_states):
    """Duration 90 s: engine steps 0..90 give states at instants 1..91. The 5 s grid ends at 90, so
    step 90 (interval [90, 91)) is not aggregated; the 1 s derivation still covers it."""
    res, five, one = closure_states
    assert five[-1]["simulation_second"] == 90 and one[-1]["simulation_second"] == 91
    tail = [c for c in res["completions"] if c["t"] == 90]
    assert sum(z["route_completed_not_exit"] + z["gate_exits"] for z in one[-1]["zones"].values()) ==         sum(ASSIGNER.assign(c["x"], c["y"]).zone_id is not None for c in tail)


def test_exit_availability_only_for_gates_and_matches_schedule(closure_states):
    res, five, _ = closure_states
    for st in five:
        T = st["simulation_second"]
        for zid, z in st["zones"].items():
            if zid.startswith("T_GATE_"):
                assert z["exit_applicable"] and z["probability_exit_open"] == (0.0 if not gate_open_at_instant(res["events"], zid, T) else 1.0)
            else:
                assert not z["exit_applicable"] and z["probability_exit_open"] is None
    e = {st["simulation_second"]: st["zones"]["T_GATE_E"]["probability_exit_open"] for st in five}
    assert (e[15], e[20], e[45], e[50]) == (1.0, 0.0, 0.0, 1.0)   # closed exactly on [20, 50)


def test_five_second_grid_matches_one_second_derivation(closure_states):
    _, five, one = closure_states
    by_t = {st["simulation_second"]: st for st in one}
    assert [st["simulation_second"] for st in five] == list(range(0, 91, 5))
    for st in five:
        T = st["simulation_second"]
        assert st["interval_end_second"] == T and st["interval_start_second"] == (T - 5 if T else 0)
        for zid, z in st["zones"].items():
            assert z["occupancy"] == by_t[T]["zones"][zid]["occupancy"]          # instant field sampled at T
            if T:
                assert z["inflow_count"] == sum(by_t[t]["zones"][zid]["inflow_count"] for t in range(T - 4, T + 1))
                assert z["outflow_count"] == sum(by_t[t]["zones"][zid]["outflow_count"] for t in range(T - 4, T + 1))
                assert z["inflow"] == pytest.approx(z["inflow_count"] / 5)
            else:
                assert z["inflow"] is None and z["speed"] is None


# ---------------------------------------------------------------- causal observations

CTX = RunContext(scenario_id="twin_test", run_mode="custom", seed=5, epoch_utc=datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc))
EXACT = ObservationConfig(dropout_probability=0.0, camera_sigma={"sim_camera_A": 0.0, "sim_camera_B": 0.0},
                          count_noise_sigma_persons=0.0, zero_false_positive_probability=0.0,
                          exit_state_error_probability=0.0)


def _secs(iso):
    return (datetime.fromisoformat(iso.replace("Z", "+00:00")) - CTX.epoch_utc).total_seconds()


def test_report_timestamps_follow_the_contract(closure_states):
    _, five, _ = closure_states
    reports = generate_observations(five, VENUE, CTX)
    assert reports
    for r in reports:
        assert _secs(r["observed_at"]) == r["simulation_second"]
        assert _secs(r["observed_at"]) < _secs(r["published_at"]) <= _secs(r["received_at"])
        assert r["simulation_second"] % 15 == 0
    assert len({r["report_id"] for r in reports}) == len(reports)


def test_report_values_come_from_the_state_at_observed_at(closure_states):
    _, five, _ = closure_states
    by_t = {st["simulation_second"]: st for st in five}
    reports = generate_observations(five, VENUE, CTX, EXACT)
    assert reports
    for r in reports:
        truth = by_t[r["simulation_second"]]["zones"][r["zone_id"]][r["variable"]]
        assert r["value"] == pytest.approx(truth, abs=1e-6)
    exits = [r for r in reports if r["variable"] == "probability_exit_open"]
    assert exits and {r["zone_id"] for r in exits} <= {"T_GATE_NW", "T_GATE_SW", "T_GATE_E"}


def test_default_error_model_perturbs_values(closure_states):
    _, five, _ = closure_states
    by_t = {st["simulation_second"]: st for st in five}
    reports = generate_observations(five, VENUE, CTX)
    diffs = [r for r in reports if r["variable"] == "density"
             and r["value"] != round(by_t[r["simulation_second"]]["zones"][r["zone_id"]]["density"], 6)]
    assert diffs


def test_observations_reproducible_and_causal(closure_states):
    _, five, _ = closure_states
    a = generate_observations(five, VENUE, CTX)
    assert a == generate_observations(copy.deepcopy(five), VENUE, CTX)
    cut = 45
    future_changed = copy.deepcopy(five)
    for st in future_changed:
        if st["simulation_second"] > cut:
            for z in st["zones"].values():
                for k in ("density", "queue_length", "occupancy"):
                    z[k] = 999
    b = generate_observations(future_changed, VENUE, CTX)
    assert [r for r in a if r["simulation_second"] <= cut] == [r for r in b if r["simulation_second"] <= cut]
    assert [r for r in a if r["simulation_second"] > cut] != [r for r in b if r["simulation_second"] > cut]


def test_missing_and_undefined_values_produce_no_reports(closure_states):
    _, five, _ = closure_states
    by_t = {st["simulation_second"]: st for st in five}
    reports = generate_observations(five, VENUE, CTX, EXACT)
    for r in reports:
        assert by_t[r["simulation_second"]]["zones"][r["zone_id"]][r["variable"]] is not None
    assert not [r for r in reports if r["simulation_second"] == 0 and r["variable"] in ("speed", "inflow", "outflow")]
    none = ObservationConfig(dropout_probability=1.0)
    assert generate_observations(five, VENUE, CTX, none) == []
    outage = ObservationConfig(outages=({"source_id": "sim_gate_sensor", "zone_id": "T_GATE_E", "from_second": 0, "until_second": 60},))
    assert not [r for r in generate_observations(five, VENUE, CTX, outage)
                if r["zone_id"] == "T_GATE_E" and r["source_id"] == "sim_gate_sensor" and r["simulation_second"] < 60]


@pytest.mark.parametrize("bad", [
    dict(publish_latency_seconds=0), dict(base_delay_seconds=2, publish_latency_seconds=3),
    dict(transport_jitter_seconds=-1), dict(observation_every_seconds=7), dict(dropout_probability=1.5),
    dict(camera_b_zones=("NOT_A_ZONE",)),
    dict(outages=({"source_id": "sim_gate_sensor", "zone_id": "T_WALK_N", "from_second": 0, "until_second": 5},)),
])
def test_invalid_observation_configs_rejected(closure_states, bad):
    _, five, _ = closure_states
    with pytest.raises(ValueError):
        generate_observations(five, VENUE, CTX, ObservationConfig(**bad))


def test_no_incident_schedule_or_reference_fields_in_reports(closure_states):
    res, five, _ = closure_states
    reports = generate_observations(five, VENUE, CTX)
    for r in reports:
        assert not set(r) & set(FORBIDDEN_FIELDS)
    text = json.dumps(reports)
    for e in res["events"]:
        assert e["name"] not in text and f'"{e["id"]}"' not in text
    assert "gate_closure" not in text
