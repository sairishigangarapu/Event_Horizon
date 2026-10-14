import pytest
from crowd_reliability.simulation.scenarios import get_default_venue, SCENARIO_PRESETS
from crowd_reliability.simulation.simulator import CrowdSimulator
from crowd_reliability.simulation.generator import ObservationGenerator

def test_simulation_repeatability_with_seed():
    """Validates that running simulator with fixed seed produces identical reference states."""
    venue = get_default_venue()
    sim1 = CrowdSimulator(venue=venue, seed=42)
    sim2 = CrowdSimulator(venue=venue, seed=42)

    sc = SCENARIO_PRESETS["crowd_surge"]
    refs1, _ = sim1.run_simulation(sc)
    refs2, _ = sim2.run_simulation(sc)

    assert len(refs1) == len(refs2)
    for r1, r2 in zip(refs1[:100], refs2[:100]):
        assert r1.zone_id == r2.zone_id
        assert r1.variable == r2.variable
        assert r1.value == r2.value
        assert r1.state_at == r2.state_at

def test_observation_generator():
    """Validates observation generator adds realistic delays and source provenance."""
    venue = get_default_venue()
    sim = CrowdSimulator(venue=venue, seed=42)
    sc = SCENARIO_PRESETS["crowd_surge"]
    refs, _ = sim.run_simulation(sc)

    gen = ObservationGenerator(seed=42)
    obs = gen.generate_observations(refs, sc)

    assert len(obs) > 0
    for o in obs[:50]:
        assert o.received_at >= o.published_at >= o.observed_at
        assert o.source_id is not None
        assert o.value >= 0.0 or o.variable == "exit_open"
