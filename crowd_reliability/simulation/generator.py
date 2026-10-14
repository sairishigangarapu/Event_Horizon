import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any
from crowd_reliability.models.schema import ReferenceStateRecord, ObservationRecord
from crowd_reliability.simulation.scenarios import ScenarioConfig

SENSOR_SOURCES = {
    "density": ["overhead_camera", "rfid_density_mesh"],
    "speed": ["optical_flow_cam", "radar_beacon"],
    "inflow": ["turnstile_in_counter", "door_beam_sensor"],
    "outflow": ["turnstile_out_counter", "exit_camera_vision"],
    "queue_length": ["queue_depth_camera", "manual_warden_patrol"],
    "exit_open": ["exit_turnstile_interlock", "magnetic_door_status", "steward_radio"],
}

class ObservationGenerator:
    """
    Transforms true reference states into realistic, imperfect observed sensor/human reports.
    Simulates observation delay, network latency, noise, sensor outages, stale reports, and conflicts.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_observations(
        self,
        ref_records: List[ReferenceStateRecord],
        scenario: ScenarioConfig
    ) -> List[ObservationRecord]:
        self.rng = random.Random(scenario.seed + 101)
        observations: List[ObservationRecord] = []

        # Group reference records by (zone_id, variable, state_at)
        for ref in ref_records:
            # Skip non-applicable exit_open records (value == -1.0)
            if ref.variable == "exit_open" and ref.value < 0:
                continue

            # Random sensor dropout / outage
            if self.rng.random() < scenario.sensor_dropout_rate:
                continue

            sources = SENSOR_SOURCES.get(ref.variable, ["generic_sensor"])
            primary_source = self.rng.choice(sources)

            # Realistic observation latency (0.5s to 3.0s)
            obs_delay_sec = self.rng.uniform(0.5, 3.0)
            # Transmission network latency (0.2s to 1.5s)
            net_delay_sec = self.rng.uniform(0.2, 1.5)

            observed_at = ref.state_at
            published_at = observed_at + timedelta(seconds=obs_delay_sec)
            received_at = published_at + timedelta(seconds=net_delay_sec)

            # Measurement noise
            if ref.variable == "exit_open":
                # Exit status usually clean, but 5% chance of misclassification
                val = ref.value if self.rng.random() > 0.05 else (1.0 - ref.value)
            else:
                noise = self.rng.gauss(0.0, 0.08 * max(0.5, ref.value))
                val = max(0.0, ref.value + noise)

            report_id = f"rep_{uuid.uuid4().hex[:10]}"
            obs = ObservationRecord(
                report_id=report_id,
                scenario_id=ref.scenario_id,
                zone_id=ref.zone_id,
                source_id=primary_source,
                variable=ref.variable,
                value=round(val, 4),
                unit=ref.unit,
                observed_at=observed_at,
                published_at=published_at,
                received_at=received_at
            )
            observations.append(obs)

            # Generate Stale Report condition:
            # An older observation is re-transmitted or persists with old timestamp
            if self.rng.random() < scenario.stale_report_probability:
                stale_age_sec = self.rng.uniform(45.0, 120.0)
                stale_observed_at = observed_at - timedelta(seconds=stale_age_sec)
                stale_val = max(0.0, ref.value * self.rng.uniform(0.3, 0.7))
                stale_obs = ObservationRecord(
                    report_id=f"rep_stale_{uuid.uuid4().hex[:8]}",
                    scenario_id=ref.scenario_id,
                    zone_id=ref.zone_id,
                    source_id=f"{primary_source}_delayed",
                    variable=ref.variable,
                    value=round(stale_val, 4),
                    unit=ref.unit,
                    observed_at=stale_observed_at,
                    published_at=published_at,
                    received_at=received_at
                )
                observations.append(stale_obs)

            # Generate Conflict condition:
            # A secondary source (e.g. manual warden) reports contradictory info
            if self.rng.random() < scenario.conflict_probability:
                alt_source = "manual_warden_patrol" if primary_source != "manual_warden_patrol" else "backup_optical"
                if ref.variable == "exit_open":
                    conflicting_val = 1.0 - val
                else:
                    conflicting_val = round(val * self.rng.choice([0.4, 2.2]), 3)

                conf_obs_at = observed_at + timedelta(seconds=self.rng.uniform(-1, 1))
                conf_pub_at = conf_obs_at + timedelta(seconds=self.rng.uniform(0.5, 2.5))
                conf_rec_at = conf_pub_at + timedelta(seconds=self.rng.uniform(0.2, 1.5))

                conflict_obs = ObservationRecord(
                    report_id=f"rep_conf_{uuid.uuid4().hex[:8]}",
                    scenario_id=ref.scenario_id,
                    zone_id=ref.zone_id,
                    source_id=alt_source,
                    variable=ref.variable,
                    value=conflicting_val,
                    unit=ref.unit,
                    observed_at=conf_obs_at,
                    published_at=conf_pub_at,
                    received_at=conf_rec_at
                )
                observations.append(conflict_obs)

        return observations
