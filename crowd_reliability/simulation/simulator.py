import math
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any
from crowd_reliability.simulation.scenarios import VenueLayout, ScenarioConfig, get_default_venue
from crowd_reliability.models.schema import ReferenceStateRecord

TARGET_VARIABLES = [
    ("density", "p/m2"),
    ("speed", "m/s"),
    ("inflow", "p/s"),
    ("outflow", "p/s"),
    ("queue_length", "persons"),
    ("exit_open", "binary"),
]

class CrowdSimulator:
    """
    Deterministic pedestrian venue simulator implementing physical flow conservation,
    density-dependent walking speed, bottleneck queuing, and exit restrictions.
    """
    def __init__(self, venue: VenueLayout = None, seed: int = 42):
        self.venue = venue or get_default_venue()
        self.seed = seed
        self.rng = random.Random(seed)

    def run_simulation(
        self,
        scenario: ScenarioConfig,
        start_time: datetime = None
    ) -> Tuple[List[ReferenceStateRecord], Dict[str, Any]]:
        """
        Runs the simulation and produces timestamped ground-truth reference records.
        """
        if start_time is None:
            # Deterministic reference base time
            start_time = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)

        self.rng = random.Random(scenario.seed)

        # Internal state per zone: count of persons, queued persons
        zone_counts: Dict[str, float] = {z["zone_id"]: 10.0 + self.rng.uniform(0, 10) for z in self.venue.zones}
        zone_queues: Dict[str, float] = {z["zone_id"]: 0.0 for z in self.venue.zones}

        ref_records: List[ReferenceStateRecord] = []
        zone_lookup = {z["zone_id"]: z for z in self.venue.zones}

        # Track history for export / telemetry
        time_series: List[Dict[str, Any]] = []

        total_steps = scenario.duration_seconds // scenario.step_seconds
        dt = float(scenario.step_seconds)

        for step in range(total_steps):
            current_sec = step * scenario.step_seconds
            t_now = start_time + timedelta(seconds=current_sec)
            label_available_at = t_now  # Reference state is produced at t_now

            # Determine scenario active modifiers
            is_surge = (scenario.surge_start_seconds <= current_sec <= scenario.surge_end_seconds)
            surge_factor = scenario.surge_multiplier if is_surge else 1.0

            # Step 1: External arrivals at entrances
            base_arrival_rate = 2.0  # persons/second
            current_arrivals = base_arrival_rate * surge_factor * (1.0 + self.rng.uniform(-0.1, 0.1))

            inflows: Dict[str, float] = {z: 0.0 for z in zone_counts}
            outflows: Dict[str, float] = {z: 0.0 for z in zone_counts}

            inflows["ent_a"] += current_arrivals * 0.55
            inflows["ent_b"] += current_arrivals * 0.45

            # Step 2: Calculate density & speed for each zone
            densities: Dict[str, float] = {}
            speeds: Dict[str, float] = {}
            exit_statuses: Dict[str, float] = {}

            for z_id, z_data in zone_lookup.items():
                area = z_data["area_m2"]
                d = max(0.01, zone_counts[z_id] / area)
                densities[z_id] = d
                # Weidmann / Greenshields speed curve: v = v0 * max(0.1, 1 - d/5.0)
                v_free = 1.34  # m/s
                d_jam = 5.2    # p/m2
                v = max(0.12, v_free * (1.0 - (d / d_jam)))
                speeds[z_id] = v

                # Exit status
                if z_data.get("has_exit"):
                    is_restricted = (
                        z_id == scenario.exit_restriction_zone and
                        scenario.exit_restriction_start <= current_sec <= scenario.exit_restriction_end
                    )
                    exit_statuses[z_id] = 0.0 if is_restricted else 1.0
                else:
                    exit_statuses[z_id] = -1.0  # Not applicable mask indicator

            # Step 3: Compute transfers along connections
            for conn in self.venue.connections:
                from_z = conn["from_zone"]
                to_z = conn["to_zone"]
                width = conn["width_m"]
                v_from = speeds[from_z]
                d_from = densities[from_z]

                # Specific flow capacity: J = width * density * speed (p/s)
                # Limited by availability in from_z and space in to_z
                nominal_flow = width * min(d_from, 3.5) * v_from
                # Backpressure if destination is above critical capacity
                dest_cap = zone_lookup[to_z]["capacity_definition"]
                dest_occ = zone_counts[to_z]
                backpressure = max(0.1, 1.0 - (dest_occ / (dest_cap * 1.5)))

                flow = min(nominal_flow * backpressure, zone_counts[from_z] / dt)
                flow = max(0.0, flow)

                outflows[from_z] += flow
                inflows[to_z] += flow

            # Step 4: Exit discharges and bottleneck queuing
            for z_id, z_data in zone_lookup.items():
                if z_data.get("has_exit"):
                    exit_open = exit_statuses[z_id]
                    if exit_open > 0.5:
                        # Normal exit discharge capacity: ~ 3.5 p/s per exit
                        discharge_rate = 3.5 * (1.0 + self.rng.uniform(-0.05, 0.05))
                        # Drain queue first, then zone count
                        can_drain = min(discharge_rate, (zone_counts[z_id] + zone_queues[z_id]) / dt)
                        actual_drain = max(0.0, can_drain)
                        outflows[z_id] += actual_drain

                        # Queues dissolve
                        dissolve = min(zone_queues[z_id], actual_drain * dt)
                        zone_queues[z_id] -= dissolve
                    else:
                        # Blocked / heavily throttled exit
                        discharge_rate = 0.2
                        actual_drain = min(discharge_rate, zone_counts[z_id] / dt)
                        outflows[z_id] += actual_drain
                        # Incoming people wanting to exit get stuck in queue
                        queue_growth = max(0.0, (inflows[z_id] - actual_drain) * dt * 0.8)
                        zone_queues[z_id] += queue_growth

            # Step 5: Update counts using mass balance
            for z_id in zone_counts:
                delta = (inflows[z_id] - outflows[z_id]) * dt
                zone_counts[z_id] = max(0.5, zone_counts[z_id] + delta)

            # Step 6: Generate ReferenceStateRecords for each variable
            step_snapshot = {"timestamp": t_now.isoformat(), "step": step, "seconds": current_sec, "zones": {}}

            for z_id, z_data in zone_lookup.items():
                has_exit = z_data.get("has_exit", False)
                vals = {
                    "density": round(densities[z_id], 4),
                    "speed": round(speeds[z_id], 4),
                    "inflow": round(inflows[z_id], 4),
                    "outflow": round(outflows[z_id], 4),
                    "queue_length": round(zone_queues[z_id], 2),
                    "exit_open": 1.0 if has_exit and exit_statuses[z_id] > 0.5 else (0.0 if has_exit else -1.0)
                }
                step_snapshot["zones"][z_id] = vals

                for var_name, unit in TARGET_VARIABLES:
                    ref_records.append(
                        ReferenceStateRecord(
                            scenario_id=scenario.scenario_id,
                            zone_id=z_id,
                            variable=var_name,
                            value=vals[var_name],
                            unit=unit,
                            state_at=t_now,
                            label_available_at=label_available_at
                        )
                    )

            time_series.append(step_snapshot)

        telemetry = {
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "steps": total_steps,
            "start_time": start_time.isoformat(),
            "time_series": time_series
        }
        return ref_records, telemetry
