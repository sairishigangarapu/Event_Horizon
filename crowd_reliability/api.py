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
from crowd_reliability.ml.pipeline import (
    ArchitecturePipeline, ModelArtifactError, load_manifest, load_verified_tree_bank,
)
from crowd_reliability.models.manifest import assess_readiness
from crowd_reliability.ml.tree_bank import DecisionTreeBank
from crowd_reliability.ml.trees import TreeSnapshot, infer_tree_vector
from crowd_reliability.ml.graph import SpatialGraphBuilder
from crowd_reliability.ml.features import FEATURE_SCHEMA_VERSION
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

ALLOWED_ORIGINS = [
    "http://127.0.0.1:8501",
    "http://localhost:8501",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
        return {**PipelineManifest().model_dump(), **_artifact_status()}
    return {
        "pipeline_version": manifest_rec.pipeline_version,
        "tree_version": manifest_rec.tree_version,
        "hypernet_version": manifest_rec.hypernet_version,
        "gnn_version": manifest_rec.gnn_version,
        "temporal_version": manifest_rec.temporal_version,
        "scaler_version": manifest_rec.scaler_version,
        "schema_version": manifest_rec.schema_version,
        "training_cutoff": manifest_rec.training_cutoff.isoformat() if manifest_rec.training_cutoff else None,
        "scenario_split": manifest_rec.scenario_split,
        "model_hashes": manifest_rec.model_hashes or {},
        **_artifact_status(),
    }

@app.get("/api/forecasts")
def get_forecasts(
    scenario_id: str = "crowd_surge",
    zone_id: Optional[str] = None,
    variable: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    as_of: Optional[str] = None,
    db: Session = Depends(get_db)
):
    stream_b_status = _stream_b_status()
    query = db.query(ForecastRecord).filter(ForecastRecord.scenario_id == scenario_id)
    if zone_id:
        query = query.filter(ForecastRecord.zone_id == zone_id)
    if variable:
        query = query.filter(ForecastRecord.variable == variable)

    records = query.order_by(ForecastRecord.target_at.asc()).all()
    arch_recs = [r for r in records if r.stream_id == "stream_b_architecture"]
    cent_recs = [r for r in records if r.stream_id == "stream_a_central"]

    try:cutoff=datetime.fromisoformat(as_of) if as_of else datetime.now(timezone.utc)
    except ValueError:raise HTTPException(422,'Invalid as_of timestamp.')
    diffs = comparator.align_and_compare(arch_recs, cent_recs,as_of_time=cutoff)
    total_comparisons = len(diffs)
    paginated_diffs = diffs[offset : offset + limit]

    return {
        "scenario_id": scenario_id,
        "zone_id": zone_id,
        "variable": variable,
        "total_comparisons": total_comparisons,
        "limit": limit,
        "offset": offset,
        "stream_b_status": stream_b_status,
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
            for d in paginated_diffs
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

    # Validate any supplied timestamp before touching the database state, so a
    # malformed evaluated_at always yields 422 even when observations are empty.
    if evaluated_at:
        try:
            t_eval = datetime.fromisoformat(evaluated_at)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid ISO 8601 evaluated_at timestamp: '{evaluated_at}'"
            ) from exc
    elif observations:
        t_eval = max(o.received_at for o in observations)
    else:
        return {"assessments": []}

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
    evaluated_at: Optional[str] = None,
    db: Session = Depends(get_db)
):
    forecasts = db.query(ForecastRecord).filter(ForecastRecord.scenario_id == scenario_id).all()
    reference_states = db.query(ReferenceStateRecord).filter(ReferenceStateRecord.scenario_id == scenario_id).all()

    if not forecasts or not reference_states:
        return {"scenario_id": scenario_id, "metrics": {}}

    try:cutoff=datetime.fromisoformat(evaluated_at) if evaluated_at else datetime.now(timezone.utc)
    except ValueError:raise HTTPException(422,'Invalid evaluated_at timestamp.')
    res = evaluator.score_predictions(forecasts, reference_states,evaluation_time=cutoff)
    return {"scenario_id": scenario_id, "metrics": res}

# --- Shared interface freeze (PDF Section G) additions -----------------------

_TREE_BANK_CACHE: Dict[str, Any] = {}


def _load_tree_bank() -> DecisionTreeBank:
    """
    The serving tree bank: the same hash-verified models/production artifact the
    pipeline loads (crowd_reliability.ml.pipeline.load_verified_tree_bank).
    Raises ModelArtifactError instead of falling back to an unfitted bank.
    """
    key = os.path.abspath(settings.model_dir)
    if key not in _TREE_BANK_CACHE:
        _TREE_BANK_CACHE[key] = load_verified_tree_bank(settings.model_dir)
    return _TREE_BANK_CACHE[key]


def _artifact_status() -> Dict[str, Any]:
    """Training status and readiness read from the active manifest file."""
    try:
        manifest = load_manifest(settings.model_dir)
    except ModelArtifactError as exc:
        return {"training_status": None, "readiness": None, "artifact_error": str(exc)}
    return {"training_status": manifest.training_status, "readiness": assess_readiness(manifest)}


def _stream_b_status() -> Dict[str, Any]:
    """Readiness label for Stream B; in strict mode, refuse to serve unvalidated forecasts."""
    status = _artifact_status()
    readiness = status["readiness"]
    if readiness is None:
        raise HTTPException(status_code=503, detail=f"Model artifacts unavailable: {status['artifact_error']}")
    if settings.require_validated_models and not readiness["production_ready"]:
        raise HTTPException(status_code=503, detail=readiness["notice"])
    return {k: readiness[k] for k in ("serving_mode", "production_ready", "unvalidated_stages", "notice")}


@app.get("/health")
def health():
    status = _artifact_status()
    try:
        bank = _load_tree_bank()
        bank_error = None
    except ModelArtifactError as exc:
        bank, bank_error = None, str(exc)
    readiness = status["readiness"]
    ok = bank is not None and readiness is not None
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": bank is not None,
        "tree_version": bank.version if bank is not None else None,
        "serving_mode": readiness["serving_mode"] if readiness else None,
        "require_validated_models": settings.require_validated_models,
        "readiness": readiness,
        "errors": [e for e in (bank_error, status.get("artifact_error")) if e],
    }


@app.get("/api/scenarios/{scenario_id}/tree-outputs")
def get_tree_outputs(
    scenario_id: str,
    as_of: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """P2 route (Section G op 6): the named six-channel D[z,t] vector at as_of.

    Deterministic: zones sorted, channels in TREE_CHANNELS order, exit
    applicability mask 0 outside exits. All six values preserved.
    """
    observations = db.query(ObservationRecord).filter(
        ObservationRecord.scenario_id == scenario_id
    ).all()

    try:
        bank = _load_tree_bank()
    except ModelArtifactError as exc:
        raise HTTPException(status_code=503, detail=f"Tree bank unavailable: {exc}") from exc

    if as_of:
        try:
            eval_time = datetime.fromisoformat(as_of)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid ISO 8601 as_of timestamp: '{as_of}'"
            ) from exc
    elif observations:
        eval_time = max(o.received_at for o in observations)
    else:
        # No observations and no as_of: there is nothing to evaluate, but the
        # contract metadata (versions, channel order/units) is still returned so
        # the response shape is the same whether or not data has been seeded.
        return {
            "scenario_id": scenario_id,
            "evaluation_time": None,
            "tree_version": bank.version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "scaler_version": getattr(bank.scaler, "version", None) if bank.scaler is not None else None,
            "channel_metadata": bank.channel_metadata(),
            "zones": [],
        }

    zones = db.query(ZoneRecord).all()
    if zones:
        zone_exit_flags = {z.zone_id: z.zone_type == "exit" for z in zones}
    else:
        zone_exit_flags = {z["zone_id"]: bool(z.get("has_exit", False)) for z in get_default_venue().zones}
    zone_ids = sorted(zone_exit_flags.keys())
    if zones:
        descriptors=[dict(zone_id=z.zone_id,zone_type=z.zone_type,has_exit=z.zone_type=='exit',capacity_definition=z.capacity_definition,area_m2=(z.geometry or {}).get('area_m2',50.)) for z in zones]
        edges=[dict(edge_id=e.edge_id,from_zone=e.from_zone,to_zone=e.to_zone,width_m=e.width_m,direction=e.direction) for e in db.query(ConnectionRecord).all() if e.from_zone in zone_exit_flags and e.to_zone in zone_exit_flags]
        snapshot_graph=SpatialGraphBuilder(descriptors,edges)
    else:snapshot_graph=SpatialGraphBuilder(get_default_venue().zones,get_default_venue().connections)

    snapshot = TreeSnapshot(
        scenario_id=scenario_id,
        as_of_time=eval_time,
        observations=observations,
        zone_ids=zone_ids,
        zone_exit_flags=zone_exit_flags,
        graph=snapshot_graph,
    )
    result = infer_tree_vector(snapshot, bank)
    return {
        "scenario_id": scenario_id,
        "evaluation_time": result.evaluation_time.isoformat(),
        "tree_version": result.tree_version,
        "feature_schema_version": result.feature_schema_version,
        "scaler_version": result.scaler_version,
        "channel_metadata": result.channel_metadata,
        "zones": result.to_zone_records(),
    }
