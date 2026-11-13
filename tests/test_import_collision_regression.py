"""Regression tests for the simulation module import collision (F-002).

Multiple directories in this repository define top-level modules named
``simulation`` (and ``observations``, ``lab_server``).  When archived test
modules are collected first, Python caches the archived ``simulation`` in
``sys.modules``.  Active application code (e.g. ``crowd_reliability.live_provider``)
then reuses the wrong module, causing ``ImportError`` for attributes that only
exist in the maintained ``studio/simulation.py``.

These tests guard against that regression by asserting that the currently
loaded ``simulation`` module is the active studio implementation and exposes
the expected API.
"""
import sys

import pytest


def test_active_simulation_module_has_live_steps():
    """The loaded simulation module must provide live_steps (studio version)."""
    import simulation
    assert hasattr(simulation, "live_steps"), (
        f"Wrong simulation module loaded: {simulation.__file__} lacks 'live_steps'. "
        f"Expected studio/simulation.py (has live_steps), got archive version."
    )


def test_simulation_module_path_is_not_archive():
    """The simulation module path must not be in an archive directory."""
    import simulation
    path = str(simulation.__file__)
    assert "archive" not in path, f"Archived simulation module loaded: {path}"
    assert "studio" in path or "simulation.py" in path, f"Unexpected simulation path: {path}"


def test_simulation_module_api_surface():
    """The active simulation module must expose the full maintained API."""
    import simulation
    expected = {"simulate", "validate", "normalize_routes", "live_steps", "path"}
    actual = {
        attr for attr in expected
        if hasattr(simulation, attr)
    }
    missing = expected - actual
    assert not missing, f"simulation module missing attributes: {missing}"


def test_live_provider_import_chain_loads_correct_simulation():
    """Importing live_provider must result in the studio simulation module."""
    # Force a clean state: remove any cached simulation module
    saved = sys.modules.pop("simulation", None)
    try:
        # Importing live_provider triggers sys.path.insert(0, studio/) then
        # 'from simulation import ...' — this must resolve to studio/simulation.py
        from crowd_reliability import live_provider  # noqa: F401 — triggers simulation import
        import simulation as sim_after
        path = str(sim_after.__file__)
        assert "archive" not in path, f"Archive simulation loaded via live_provider: {path}"
        assert hasattr(sim_after, "live_steps"), "live_steps not available after live_provider import"
    finally:
        # Restore whatever was in sys.modules before us
        if saved is not None:
            sys.modules["simulation"] = saved
