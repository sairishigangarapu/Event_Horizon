import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class ZoneRecord(Base):
    __tablename__ = "zones"

    zone_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    geometry = Column(JSON, nullable=True)  # e.g. {"x": 100, "y": 200, "area_m2": 50.0}
    zone_type = Column(String, nullable=False)  # corridor, entrance, exit, stairway, assembly
    capacity_definition = Column(Float, nullable=False, default=100.0)  # max comfortable capacity
    map_version = Column(String, nullable=False, default="v1.0")

class ConnectionRecord(Base):
    __tablename__ = "connections"

    edge_id = Column(String, primary_key=True)
    from_zone = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    to_zone = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    width_m = Column(Float, nullable=False, default=3.0)
    direction = Column(String, nullable=False, default="bidirectional")  # unidirectional, bidirectional
    map_version = Column(String, nullable=False, default="v1.0")

class ObservationRecord(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String, index=True, nullable=False)
    scenario_id = Column(String, index=True, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    source_id = Column(String, nullable=False)  # e.g. "sensor_camera_1", "manual_warden_2"
    variable = Column(String, nullable=False)   # density, speed, inflow, outflow, queue_length, exit_open
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)       # p/m2, m/s, p/s, p/s, persons, binary
    observed_at = Column(DateTime, nullable=False)
    published_at = Column(DateTime, nullable=False)
    received_at = Column(DateTime, nullable=False)

class TreeOutputRecord(Base):
    __tablename__ = "tree_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(String, index=True, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    evaluation_time = Column(DateTime, index=True, nullable=False)
    tree_version = Column(String, nullable=False)
    channel_names = Column(JSON, nullable=False)   # ["estimated_density", "estimated_speed", ...]
    channel_values = Column(JSON, nullable=False)  # [1.4, 0.8, ...]
    channel_masks = Column(JSON, nullable=False)   # [1, 1, ...] (1 = valid/applicable, 0 = missing/inapplicable)

class ForecastRecord(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    forecast_id = Column(String, index=True, nullable=False)
    stream_id = Column(String, index=True, nullable=False)  # "stream_a_central" or "stream_b_architecture"
    scenario_id = Column(String, index=True, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    variable = Column(String, nullable=False)
    unit = Column(String, nullable=False)
    issued_at = Column(DateTime, nullable=False)
    received_at = Column(DateTime, nullable=False)
    target_at = Column(DateTime, index=True, nullable=False)
    value = Column(Float, nullable=False)
    model_or_provider_version = Column(String, nullable=False)

class ReferenceStateRecord(Base):
    __tablename__ = "reference_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(String, index=True, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id"), nullable=False)
    variable = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)
    state_at = Column(DateTime, index=True, nullable=False)
    label_available_at = Column(DateTime, nullable=False)

class AssessmentRecord(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_ref_id = Column(String, index=True, nullable=False)  # report_id or forecast_id
    target_type = Column(String, nullable=False)  # "observation" or "forecast"
    evaluated_at = Column(DateTime, nullable=False)
    observation_age_seconds = Column(Float, nullable=False)
    conflict_ids = Column(JSON, nullable=True)  # list of conflicting report IDs
    coverage = Column(Float, nullable=False)     # 0.0 to 1.0 evidence coverage ratio
    per_variable_flags = Column(JSON, nullable=True)  # {"stale": bool, "conflict": bool, "surge_detected": bool}

class ModelManifestRecord(Base):
    __tablename__ = "model_manifests"

    pipeline_version = Column(String, primary_key=True)
    tree_version = Column(String, nullable=False)
    hypernet_version = Column(String, nullable=False, default="hypernet_v1")
    gnn_version = Column(String, nullable=False)
    temporal_version = Column(String, nullable=False)
    scaler_version = Column(String, nullable=False)
    schema_version = Column(String, nullable=False)
    training_cutoff = Column(DateTime, nullable=False)
    scenario_split = Column(String, nullable=False)  # train, val, test, production
    is_active = Column(Boolean, nullable=False, default=False)
    metrics_summary = Column(JSON, nullable=True)
