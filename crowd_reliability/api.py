import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from crowd_reliability.config import settings
from crowd_reliability.database import get_db, init_db, SessionLocal
from crowd_reliability.models.schema import (
    ZoneRecord, ConnectionRecord, ObservationRecord,
    TreeOutputRecord, ForecastRecord, ReferenceStateRecord,
    AssessmentRecord, ModelManifestRecord
)
from crowd_reliability.models.manifest import PipelineManifest
from crowd_reliability.simulation.scenarios import (
    get_default_venue, SCENARIO_PRESETS, ScenarioConfig
)
from crowd_reliability.simulation.simulator import CrowdSimulator
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.services.central_source import CentralSourceAdapter
from crowd_reliability.services.comparator import ForecastComparator
from crowd_reliability.services.evaluator import EvaluationService

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Crowd Disaster Information Reliability API",
    description="Multivariate crowd forecasting with dual-stream comparison and information reliability assessment.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

comparator = ForecastComparator()
evaluator = EvaluationService()

@app.get("/")
def get_root():
    return {
        "title": "Crowd Disaster Information Reliability API",
        "version": "0.1.0",
        "app_env": settings.app_env,
        "docs_url": "/docs",
        "status": "operational"
    }

@app.get("/api/scenarios")
def list_scenarios():
    return [
        {
            "scenario_id": sc.scenario_id,
            "title": sc.title,
            "description": sc.description,
            "duration_seconds": sc.duration_seconds,
            "step_seconds": sc.step_seconds
        }
        for sc in SCENARIO_PRESETS.values()
    ]

@app.get("/api/venue")
def get_venue(db: Session = Depends(get_db)):
    zones = db.query(ZoneRecord).all()
    connections = db.query(ConnectionRecord).all()

    if not zones:
        v = get_default_venue()
        return {"zones": v.zones, "connections": v.connections}

    return {
        "zones": [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "geometry": z.geometry,
                "zone_type": z.zone_type,
                "capacity_definition": z.capacity_definition
            }
            for z in zones
        ],
        "connections": [
            {
                "edge_id": c.edge_id,
                "from_zone": c.from_zone,
                "to_zone": c.to_zone,
                "width_m": c.width_m,
                "direction": c.direction
            }
            for c in connections
        ]
    }

@app.get("/api/manifest")
def get_active_manifest(db: Session = Depends(get_db)):
    manifest_rec = db.query(ModelManifestRecord).filter(ModelManifestRecord.is_active == True).first()
    if not manifest_rec:
        return PipelineManifest().model_dump()
    return {
        "pipeline_version": manifest_rec.pipeline_version,
        "tree_version": manifest_rec.tree_version,
        "gnn_version": manifest_rec.gnn_version,
        "temporal_version": manifest_rec.temporal_version,
        "scaler_version": manifest_rec.scaler_version,
        "schema_version": manifest_rec.schema_version,
        "training_cutoff": manifest_rec.training_cutoff.isoformat() if manifest_rec.training_cutoff else None,
        "scenario_split": manifest_rec.scenario_split
    }

@app.get("/api/forecasts")
def get_forecasts(
    scenario_id: str = "crowd_surge",
    zone_id: Optional[str] = None,
    variable: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(ForecastRecord).filter(ForecastRecord.scenario_id == scenario_id)
    if zone_id:
        query = query.filter(ForecastRecord.zone_id == zone_id)
    if variable:
        query = query.filter(ForecastRecord.variable == variable)

    records = query.order_by(ForecastRecord.target_at.asc()).all()
    arch_recs = [r for r in records if r.stream_id == "stream_b_architecture"]
    cent_recs = [r for r in records if r.stream_id == "stream_a_central"]

    diffs = comparator.align_and_compare(arch_recs, cent_recs)

    return {
        "scenario_id": scenario_id,
        "zone_id": zone_id,
        "variable": variable,
        "count_arch": len(arch_recs),
        "count_cent": len(cent_recs),
        "comparisons": [
            {
                "zone_id": d.zone_id,
                "variable": d.variable,
                "unit": d.unit,
                "target_at": d.target_at.isoformat(),
                "arch_value": d.arch_forecast.value if d.arch_forecast else None,
                "cent_value": d.cent_forecast.value if d.cent_forecast else None,
                "signed_diff": d.signed_diff,
                "abs_diff": d.abs_diff,
                "arch_issue_time": d.arch_forecast.issued_at.isoformat() if d.arch_forecast else None,
                "cent_issue_time": d.cent_forecast.issued_at.isoformat() if d.cent_forecast else None,
            }
            for d in diffs
        ]
    }

@app.get("/api/assessments")
def get_assessments(
    scenario_id: str = "crowd_surge",
    zone_id: Optional[str] = None,
    evaluated_at: Optional[str] = None,
    db: Session = Depends(get_db)
):
    obs_query = db.query(ObservationRecord).filter(ObservationRecord.scenario_id == scenario_id)
    if zone_id:
        obs_query = obs_query.filter(ObservationRecord.zone_id == zone_id)
    observations = obs_query.all()

    if not observations:
        return {"assessments": []}

    t_eval = datetime.fromisoformat(evaluated_at) if evaluated_at else max(o.received_at for o in observations)

    # Variables to assess
    variables = ["density", "speed", "inflow", "outflow", "queue_length", "exit_open"]
    zones_to_eval = [zone_id] if zone_id else list(set(o.zone_id for o in observations))

    assessments = []
    for z in zones_to_eval:
        for v in variables:
            rec = comparator.assess_information(observations, t_eval, z, v)
            assessments.append({
                "zone_id": z,
                "variable": v,
                "evaluated_at": rec.evaluated_at.isoformat(),
                "observation_age_seconds": rec.observation_age_seconds,
                "conflict_count": len(rec.conflict_ids or []),
                "coverage": rec.coverage,
                "flags": rec.per_variable_flags
            })

    return {"scenario_id": scenario_id, "evaluated_at": t_eval.isoformat(), "assessments": assessments}

@app.get("/api/metrics")
def get_metrics(
    scenario_id: str = "crowd_surge",
    db: Session = Depends(get_db)
):
    forecasts = db.query(ForecastRecord).filter(ForecastRecord.scenario_id == scenario_id).all()
    reference_states = db.query(ReferenceStateRecord).filter(ReferenceStateRecord.scenario_id == scenario_id).all()

    if not forecasts or not reference_states:
        return {"scenario_id": scenario_id, "metrics": {}}

    res = evaluator.score_predictions(forecasts, reference_states)
    return {"scenario_id": scenario_id, "metrics": res}
