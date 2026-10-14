"""
Adapter for externally supplied simulator scenario configs (e.g. the
Event_Horizon_Member1_Simulation_Starter pack) into ScenarioConfig objects that
CrowdSimulator can run on the default venue.

The starter pack uses a different zone naming scheme (Z_EXIT_E, Z_ANNEX, ...).
Zones that cannot be mapped onto the default venue are skipped with a warning;
the scenario still loads if its core fields are present.
"""
import json
import logging
import os
from typing import List, Optional

from crowd_reliability.simulation.scenarios import ScenarioConfig

logger = logging.getLogger(__name__)

# Starter-pack zone ids -> default venue zone ids (best-effort mapping).
STARTER_ZONE_MAP = {
    "Z_EXIT_E": "exit_2",
    "Z_EXIT_W": "exit_1",
    "Z_ANNEX": "concourse",
    "Z_CONCOURSE": "concourse",
}


def load_starter_scenario(filepath: str) -> Optional[ScenarioConfig]:
    """Loads one starter-pack JSON config. Returns None (with a warning) if unmappable."""
    try:
        with open(filepath) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable scenario config %s: %s", filepath, exc)
        return None

    scenario_id = raw.get("scenario_id")
    if not scenario_id or "seed" not in raw:
        logger.warning("Skipping %s: missing scenario_id or seed", filepath)
        return None

    base_rate = float(raw.get("arrival_rate_persons_per_second", 2.0))
    surge_start, surge_end, surge_mult = 9999, 9999, 1.0
    events = raw.get("arrival_events") or []
    if events:
        first = events[0]
        surge_start = int(first.get("from_second", 9999))
        surge_end = int(first.get("until_second", 9999))
        event_rate = float(first.get("rate_persons_per_second", base_rate))
        surge_mult = event_rate / base_rate if base_rate > 0 else 1.0

    exit_zone = "exit_1"
    exit_start, exit_end = 9999, 9999
    restrictions = raw.get("exit_restrictions") or []
    if restrictions:
        first = restrictions[0]
        raw_zone = first.get("exit_zone_id", "")
        mapped = STARTER_ZONE_MAP.get(raw_zone)
        if mapped is None:
            logger.warning(
                "Scenario %s: exit zone %s has no default-venue mapping; using exit_1",
                scenario_id, raw_zone,
            )
            mapped = "exit_1"
        exit_zone = mapped
        exit_start = int(first.get("from_second", 9999))
        exit_end = int(first.get("until_second", 9999))

    return ScenarioConfig(
        scenario_id=f"starter_{scenario_id}",
        title=raw.get("display_name", scenario_id),
        description=f"Imported from {os.path.basename(filepath)} (config {raw.get('configuration_version', 'unknown')})",
        duration_seconds=int(raw.get("duration_seconds", 1200)),
        step_seconds=int(raw.get("step_seconds", 5)),
        seed=int(raw["seed"]),
        surge_start_seconds=surge_start,
        surge_end_seconds=surge_end,
        surge_multiplier=surge_mult,
        exit_restriction_zone=exit_zone,
        exit_restriction_start=exit_start,
        exit_restriction_end=exit_end,
        sensor_dropout_rate=float(raw.get("observation_dropout_probability", 0.10)),
    )


def load_starter_scenarios(directory: str) -> List[ScenarioConfig]:
    """Loads all starter-pack scenario configs from a directory, sorted by filename."""
    scenarios: List[ScenarioConfig] = []
    if not os.path.isdir(directory):
        logger.warning("Starter scenario directory not found: %s", directory)
        return scenarios
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            cfg = load_starter_scenario(os.path.join(directory, name))
            if cfg is not None:
                scenarios.append(cfg)
    return scenarios
