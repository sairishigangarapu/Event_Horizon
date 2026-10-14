from typing import Dict, Any, List
from pydantic import BaseModel

class VenueLayout(BaseModel):
    zones: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]

def get_default_venue() -> VenueLayout:
    """
    Standard evaluation venue layout with 7 zones and 7 directed connections.
    Includes entrances, corridors, central concourse, stairway, and exits.
    """
    zones = [
        {"zone_id": "ent_a", "name": "Entrance North", "zone_type": "entrance", "capacity_definition": 150.0, "area_m2": 60.0, "has_exit": False, "x": 100, "y": 100},
        {"zone_id": "ent_b", "name": "Entrance South", "zone_type": "entrance", "capacity_definition": 150.0, "area_m2": 60.0, "has_exit": False, "x": 100, "y": 300},
        {"zone_id": "corridor_1", "name": "North Corridor", "zone_type": "corridor", "capacity_definition": 100.0, "area_m2": 40.0, "has_exit": False, "x": 250, "y": 120},
        {"zone_id": "corridor_2", "name": "South Corridor", "zone_type": "corridor", "capacity_definition": 100.0, "area_m2": 40.0, "has_exit": False, "x": 250, "y": 280},
        {"zone_id": "concourse", "name": "Central Concourse", "zone_type": "assembly", "capacity_definition": 400.0, "area_m2": 150.0, "has_exit": False, "x": 400, "y": 200},
        {"zone_id": "stair_1", "name": "Stairway East", "zone_type": "stairway", "capacity_definition": 80.0, "area_m2": 30.0, "has_exit": False, "x": 550, "y": 120},
        {"zone_id": "exit_1", "name": "Emergency Exit 1", "zone_type": "exit", "capacity_definition": 120.0, "area_m2": 50.0, "has_exit": True, "x": 550, "y": 280},
        {"zone_id": "exit_2", "name": "Main East Exit", "zone_type": "exit", "capacity_definition": 150.0, "area_m2": 60.0, "has_exit": True, "x": 700, "y": 120},
    ]

    connections = [
        {"edge_id": "e_ent_a_corr1", "from_zone": "ent_a", "to_zone": "corridor_1", "width_m": 3.5, "direction": "unidirectional", "distance_m": 25.0},
        {"edge_id": "e_ent_b_corr2", "from_zone": "ent_b", "to_zone": "corridor_2", "width_m": 3.5, "direction": "unidirectional", "distance_m": 25.0},
        {"edge_id": "e_corr1_conc", "from_zone": "corridor_1", "to_zone": "concourse", "width_m": 4.0, "direction": "unidirectional", "distance_m": 30.0},
        {"edge_id": "e_corr2_conc", "from_zone": "corridor_2", "to_zone": "concourse", "width_m": 4.0, "direction": "unidirectional", "distance_m": 30.0},
        {"edge_id": "e_conc_exit1", "from_zone": "concourse", "to_zone": "exit_1", "width_m": 3.0, "direction": "unidirectional", "distance_m": 20.0},
        {"edge_id": "e_conc_stair1", "from_zone": "concourse", "to_zone": "stair_1", "width_m": 2.5, "direction": "unidirectional", "distance_m": 25.0},
        {"edge_id": "e_stair1_exit2", "from_zone": "stair_1", "to_zone": "exit_2", "width_m": 3.0, "direction": "unidirectional", "distance_m": 20.0},
    ]

    return VenueLayout(zones=zones, connections=connections)

class ScenarioConfig(BaseModel):
    scenario_id: str
    title: str
    description: str
    duration_seconds: int = 300
    step_seconds: int = 5
    seed: int = 42
    surge_start_seconds: int = 60
    surge_end_seconds: int = 180
    surge_multiplier: float = 3.5
    exit_restriction_zone: str = "exit_1"
    exit_restriction_start: int = 80
    exit_restriction_end: int = 240
    sensor_dropout_rate: float = 0.15
    stale_report_probability: float = 0.20
    conflict_probability: float = 0.15

SCENARIO_PRESETS: Dict[str, ScenarioConfig] = {
    "crowd_surge": ScenarioConfig(
        scenario_id="crowd_surge",
        title="Sudden Inflow Surge",
        description="A sudden large influx of pedestrians at north and south entrances causes high concourse density and exit queuing.",
        duration_seconds=300,
        surge_start_seconds=45,
        surge_end_seconds=180,
        surge_multiplier=4.0,
        exit_restriction_start=9999,  # No exit restriction
        sensor_dropout_rate=0.10,
        stale_report_probability=0.25,
        conflict_probability=0.20,
    ),
    "bottleneck_exit_closure": ScenarioConfig(
        scenario_id="bottleneck_exit_closure",
        title="Bottleneck & Exit Closure",
        description="Emergency Exit 1 is blocked at t=70s, causing rapid queue growth, severe speed reduction, and rerouting via stairway.",
        duration_seconds=300,
        surge_start_seconds=30,
        surge_end_seconds=120,
        surge_multiplier=2.0,
        exit_restriction_zone="exit_1",
        exit_restriction_start=70,
        exit_restriction_end=220,
        sensor_dropout_rate=0.15,
        stale_report_probability=0.30,
        conflict_probability=0.25,
    ),
    "normal_evacuation": ScenarioConfig(
        scenario_id="normal_evacuation",
        title="Orderly Evacuation",
        description="Steady pedestrian flow with all routes and exits operating normally under nominal capacity.",
        duration_seconds=300,
        surge_start_seconds=9999,
        surge_end_seconds=9999,
        surge_multiplier=1.0,
        exit_restriction_start=9999,
        sensor_dropout_rate=0.05,
        stale_report_probability=0.10,
        conflict_probability=0.05,
    ),
}
