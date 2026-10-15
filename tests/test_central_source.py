"""Stream A (simulated central forecast) must be built from observations only."""
import copy
import inspect
from datetime import timedelta

import pytest

from crowd_reliability.services.central_source import CentralSourceAdapter, issue_grid
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import SCENARIO_PRESETS, get_default_venue
from crowd_reliability.simulation.simulator import CrowdSimulator


@pytest.fixture(scope="module")
def surge():
    venue = get_default_venue()
    sc = SCENARIO_PRESETS["bottleneck_exit_closure"]
    refs, _ = CrowdSimulator(venue=venue, seed=42).run_simulation(sc)
    obs = ObservationGenerator(seed=42).generate_observations(refs, sc)
    return venue, sc, obs


def _key(f):
    return (f.forecast_id, f.zone_id, f.variable, f.issued_at, f.target_at, f.value)


def test_generator_takes_no_reference_states():
    params = inspect.signature(CentralSourceAdapter.generate_simulated_forecasts).parameters
    assert "observations" in params
    assert not any("ref" in name for name in params)


def test_reports_received_after_issue_time_are_never_used(surge):
    # Regression: the fixture blended in the reference value at the target time.
    venue, sc, obs = surge
    grid = issue_grid([o.received_at for o in obs], 15)
    cut = grid[len(grid) // 2]
    adapter = CentralSourceAdapter()

    baseline = adapter.generate_simulated_forecasts(sc.scenario_id, obs, venue, issue_times=grid)
    tampered = copy.deepcopy(obs)
    for o in tampered:
        if o.received_at > cut:
            o.value = o.value * 10 + 100
    changed = adapter.generate_simulated_forecasts(sc.scenario_id, tampered, venue, issue_times=grid)

    early = lambda fs: sorted(_key(f) for f in fs if f.issued_at <= cut)
    assert early(baseline) and early(baseline) == early(changed)
    # Later issues do see the tampered reports, so the comparison is not vacuous.
    late = lambda fs: sorted(_key(f) for f in fs if f.issued_at > cut)
    assert late(baseline) != late(changed)


def test_unavailable_variables_are_not_fabricated(surge):
    venue, sc, obs = surge
    only_density = [o for o in obs if o.variable == "density" and o.zone_id == "concourse"]
    forecasts = CentralSourceAdapter().generate_simulated_forecasts(sc.scenario_id, only_density, venue)
    assert forecasts
    assert {(f.zone_id, f.variable) for f in forecasts} == {("concourse", "density")}


def test_exit_forecasts_only_for_exit_zones_and_ids_deterministic(surge):
    venue, sc, obs = surge
    a = CentralSourceAdapter().generate_simulated_forecasts(sc.scenario_id, obs, venue)
    b = CentralSourceAdapter().generate_simulated_forecasts(sc.scenario_id, obs, venue)
    exit_zones = {z["zone_id"] for z in venue.zones if z.get("has_exit")}
    assert {f.zone_id for f in a if f.variable == "exit_open"} <= exit_zones
    assert [_key(f) for f in a] == [_key(f) for f in b]
    assert len({f.forecast_id for f in a}) == len(a)
    for f in a:
        assert f.stream_id == "stream_a_central"
        assert f.received_at >= f.issued_at
        assert f.target_at - f.issued_at in (timedelta(seconds=30), timedelta(seconds=60), timedelta(seconds=120))
