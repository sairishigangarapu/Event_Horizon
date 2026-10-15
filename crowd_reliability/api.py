import os
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, aliased
from sqlalchemy import and_, func
from crowd_reliability.contracts import utc

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
    # F-023: In production environments, database schema should be managed via migrations
    # or explicit CLI 'db-init'; only auto-create in demo/development/test modes.
    if getattr(settings, "app_env", "demo") in ("demo", "test", "development"):
        init_db()
    yield

app = FastAPI(
    title="Crowd Disaster Information Reliability API",
    description="Multivariate crowd forecasting with dual-stream comparison and information reliability assessment.",
    version="0.1.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response

from crowd_reliability.auth import verify_auth, check_api_key
from crowd_reliability.rate_limit import ApiRateLimit
from crowd_reliability.body_limit import ApiBodyLimit
app.add_middleware(ApiBodyLimit)
_API_LIMIT=ApiRateLimit()

@app.middleware("http")
async def shared_api_boundary(request, call_next):
    if request.url.path.startswith('/api/') and request.method!='OPTIONS':
        if settings.app_env not in ('demo','test','development') and settings.api_rate_limit_per_minute>0 and request.url.path not in ('/api/ready','/api/health'):
            client=request.client.host if request.client else 'unknown'
            if not _API_LIMIT.allow(client,settings.api_rate_limit_per_minute):
                return JSONResponse(status_code=429,content={'detail':'API request rate exceeded; retry shortly.'},headers={'Retry-After':'1'})
        if request.url.path not in ('/api/ready','/api/runtime-config','/api/health'):
            try:check_api_key(request.headers.get('X-API-Key'))
            except HTTPException as exc:return JSONResponse(status_code=exc.status_code,content={'detail':exc.detail})
        origin=request.headers.get('Origin')
        if request.method not in ('GET','HEAD') and origin:
            same_origin=str(request.base_url).rstrip('/')
            if origin.rstrip('/')!=same_origin and origin not in ALLOWED_ORIGINS:
                return JSONResponse(status_code=403,content={'detail':'Cross-origin mutation is not allowed.'})
    return await call_next(request)

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
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

@app.get("/api/scenarios", dependencies=[Depends(verify_auth)])
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

@app.get("/api/venue", dependencies=[Depends(verify_auth)])
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
    try:serving=load_manifest(settings.model_dir)
    except ModelArtifactError as exc:raise HTTPException(503,'Serving manifest is unavailable.') from exc
    result={**serving.model_dump(mode='json'),**_artifact_status()}
    result['registry_pipeline_version']=manifest_rec.pipeline_version if manifest_rec else None
    result['registry_matches_serving']=manifest_rec.pipeline_version==serving.pipeline_version if manifest_rec else None
    return result

@app.get("/api/forecasts", dependencies=[Depends(verify_auth)])
def get_forecasts(
    scenario_id: str = "crowd_surge",
    zone_id: Optional[str] = None,
    variable: Optional[str] = None,
    limit: int = Query(default=500, ge=1, le=1000),
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

    try:cutoff=utc(datetime.fromisoformat(as_of)) if as_of else utc(datetime.now(timezone.utc))
    except ValueError:raise HTTPException(422,'Invalid as_of timestamp.')
    query=query.filter(ForecastRecord.stream_id.in_(['stream_b_architecture','stream_a_central']),
        ForecastRecord.received_at<=cutoff,ForecastRecord.issued_at<=cutoff)
    # Page physical forecast keys in SQL; never load an entire scenario to slice it.
    columns=[ForecastRecord.scenario_id,ForecastRecord.zone_id,ForecastRecord.variable,
        ForecastRecord.unit,ForecastRecord.issued_at,ForecastRecord.target_at]
    distinct_keys=query.with_entities(*columns).distinct()
    total_comparisons=distinct_keys.count()
    page=distinct_keys.order_by(*columns).offset(offset).limit(limit).subquery()
    match=and_(*(column==getattr(page.c,column.key) for column in columns))
    ranked=query.join(page,match).add_columns(func.row_number().over(
        partition_by=[ForecastRecord.stream_id,*columns],
        order_by=[ForecastRecord.received_at.desc(),ForecastRecord.forecast_id.desc(),ForecastRecord.id.desc()]
    ).label('_revision_rank')).subquery()
    forecast=aliased(ForecastRecord,ranked)
    records=db.query(forecast).filter(ranked.c._revision_rank==1).all()
    arch_recs=[r for r in records if r.stream_id=='stream_b_architecture']
    cent_recs=[r for r in records if r.stream_id=='stream_a_central']
    paginated_diffs=comparator.align_and_compare(arch_recs,cent_recs,as_of_time=cutoff)
    count_arch=query.filter(ForecastRecord.stream_id=='stream_b_architecture').count()
    count_cent=query.filter(ForecastRecord.stream_id=='stream_a_central').count()

    return {
        "scenario_id": scenario_id,
        "zone_id": zone_id,
        "variable": variable,
        "total_comparisons": total_comparisons,
        "limit": limit,
        "offset": offset,
        "stream_b_status": stream_b_status,
        "count_arch": count_arch,
        "count_cent": count_cent,
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
                "arch_model_version": d.arch_forecast.model_or_provider_version if d.arch_forecast else None,
                "cent_model_version": d.cent_forecast.model_or_provider_version if d.cent_forecast else None,
                "arch_channel_mask": d.arch_forecast.channel_mask if d.arch_forecast else None,
                "cent_channel_mask": d.cent_forecast.channel_mask if d.cent_forecast else None,
                "arch_missing_data_mask": d.arch_forecast.missing_data_mask if d.arch_forecast else None,
                "cent_missing_data_mask": d.cent_forecast.missing_data_mask if d.cent_forecast else None,
            }
            for d in paginated_diffs
        ]
    }

@app.get("/api/assessments", dependencies=[Depends(verify_auth)])
def get_assessments(
    scenario_id: str = "crowd_surge",
    zone_id: Optional[str] = None,
    evaluated_at: Optional[str] = None,
    include_provenance: bool = False,
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
    zones_lookup = {z.zone_id: z.zone_type for z in db.query(ZoneRecord).all()}

    assessments = []
    for z in zones_to_eval:
        z_type = zones_lookup.get(z)
        for v in variables:
            rec = comparator.assess_information(
                observations, t_eval, z, v,
                zone_type=z_type,
                scenario_type=scenario_id
            )
            assessments.append({
                "zone_id": z,
                "variable": v,
                "evaluated_at": rec.evaluated_at.isoformat(),
                "observation_age_seconds": rec.observation_age_seconds,
                "conflict_count": len(rec.conflict_ids or []),
                "coverage": rec.coverage,
                "coverage_basis": getattr(rec,"coverage_basis","No received source roster"),
                "conflict_graph": getattr(rec,"conflict_graph",{}) if include_provenance else {k:v for k,v in getattr(rec,"conflict_graph",{}).items() if k.startswith("num_")},
                "flags": rec.per_variable_flags
            })

    return {"scenario_id": scenario_id, "evaluated_at": t_eval.isoformat(), "assessments": assessments}

@app.get("/api/metrics", dependencies=[Depends(verify_auth)])
def get_metrics(
    scenario_id: str = "crowd_surge",
    evaluated_at: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:cutoff=utc(datetime.fromisoformat(evaluated_at)) if evaluated_at else utc(datetime.now(timezone.utc))
    except ValueError:raise HTTPException(422,'Invalid evaluated_at timestamp.')
    forecasts = db.query(ForecastRecord).filter(ForecastRecord.scenario_id == scenario_id).all()
    reference_states = db.query(ReferenceStateRecord).filter(ReferenceStateRecord.scenario_id == scenario_id).all()
    if not forecasts or not reference_states:
        return {"scenario_id": scenario_id, "metrics": {}}

    res = evaluator.score_predictions(forecasts, reference_states,evaluation_time=cutoff)
    return {"scenario_id": scenario_id, "metrics": res}

# --- Shared interface freeze (PDF Section G) additions -----------------------

_TREE_BANK_CACHE: Dict[str, Any] = {}
_TREE_BANK_LOCK = threading.Lock()


def _load_tree_bank() -> DecisionTreeBank:
    """
    The serving tree bank: the same hash-verified models/production artifact the
    pipeline loads (crowd_reliability.ml.pipeline.load_verified_tree_bank).
    Raises ModelArtifactError instead of falling back to an unfitted bank.
    """
    root=os.path.abspath(settings.model_dir)
    manifest=load_manifest(root)
    try:stamp=os.stat(os.path.join(root,'tree_bank.joblib'))
    except OSError as exc:raise ModelArtifactError('Missing tree bank artifact.') from exc
    key=(root,manifest.model_hashes.get('tree_bank.joblib'),stamp.st_mtime_ns,stamp.st_size)
    with _TREE_BANK_LOCK:
        if key not in _TREE_BANK_CACHE:
            _TREE_BANK_CACHE.clear()  # The serving process retains one verified active bank.
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


# --- Twin Studio v3 partial-information outputs (read-only) -------------------
# Serves only persisted pipeline outputs (services/twin_studio/v3/outputs_view.py). The API process
# never reads Twin run outputs, hidden truth, scenes or observation streams, and cannot start or
# advance a replay.

def _twin_v3():
    from services.twin_studio.v3 import outputs_view
    return outputs_view


@app.get("/api/twin-v3/evaluations")
def twin_v3_evaluations():
    return {"evaluations": _twin_v3().list_evaluations()}


@app.get("/api/twin-v3/evaluations/{eval_id}")
def twin_v3_report(eval_id: str):
    ov = _twin_v3()
    try:
        return ov.report(eval_id)
    except ov.OutputsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/twin-v3/evaluations/{eval_id}/{kind}")
def twin_v3_records(eval_id: str, kind: str, as_of_s: Optional[float] = Query(default=None, ge=0),
                    limit: int = Query(default=500, ge=1, le=5000), offset: int = Query(default=0, ge=0)):
    ov = _twin_v3()
    try:
        rows = ov.records(eval_id, kind, as_of_s=as_of_s)
    except ov.OutputsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"evaluation_id": eval_id, "kind": kind, "as_of_s": as_of_s, "total": len(rows),
            "records": rows[offset: offset + limit]}
