"""Isolated Twin Studio subsystem (environment_id "eh_twin_studio").

Phase 1 public interface:
  - zones.load_venue / zones.ZoneAssigner: frozen twin_venue_v1 zone model
  - timebase: engine-step <-> simulated-instant mapping (end-of-interval convention)
  - engine.simulation: unmodified copy of the Twin Studio Python engine (see engine/SOURCE.json)

The main application must use only these public modules; nothing here exposes reference truth.
"""
ENVIRONMENT_ID = "eh_twin_studio"
MAP_VERSION = "twin_venue_v1"
