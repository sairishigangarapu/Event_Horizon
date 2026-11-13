"""Live observation-only inference; retrospective labels live in a separate store."""
import csv
import io
import math
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.ml.graph import SpatialGraphBuilder
from crowd_reliability.config import settings
from crowd_reliability.models.schema import ObservationRecord
from crowd_reliability.services.central_source import CentralSourceAdapter
from crowd_reliability.services.comparator import ForecastComparator
from crowd_reliability.simulation.scenarios import VenueLayout

router = APIRouter(prefix='/api/runtime', tags=['Live runtime'])
EPOCH = datetime(2026, 1, 1)
UNITS = dict(density='p/m2', speed='m/s', inflow='p/s', outflow='p/s', queue_length='persons', exit_open='binary')
SESSIONS = {}
# The inference path never receives this store.
EVALUATION = {}
LOCK = threading.RLock()
MAX_SESSIONS = 30
SESSION_TTL_SECONDS = 3600.0


def _evict_expired_sessions(now: float | None = None) -> int:
    cutoff = (time.monotonic() if now is None else now) - SESSION_TTL_SECONDS
    expired = [sid for sid, sess in SESSIONS.items() if sess.get('last_active_at', 0.0) < cutoff]
    for sid in expired:
        SESSIONS.pop(sid, None)
        EVALUATION.pop(sid, None)
    return len(expired)

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Point(Strict):
    x: float = Field(ge=0, le=1000)
    y: float = Field(ge=0, le=700)

class Route(Strict):
    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=100)
    points: list[Point] = Field(min_length=2, max_length=500)
    width: float = Field(ge=.5, le=20)

class Start(Strict):
    run_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    mode: Literal['ideal', 'custom', 'random']
    scale: float = Field(gt=0, le=2)
    duration: int = Field(ge=10, le=900)
    routes: list[Route] = Field(min_length=1, max_length=12)

class Report(Strict):
    observation_id: str
    source_id: str
    source_kind: Literal['camera', 'sensor', 'warden']
    route_id: str
    observed_at_s: float = Field(ge=0)
    available_at_s: float = Field(ge=0)
    values: dict[str, float | str]

class Advance(Strict):
    time_s: int = Field(ge=0, le=900)
    observations: list[Report] = Field(max_length=100000)

class Actual(Strict):
    time_s: int = Field(ge=0, le=900)
    available_at_s: int | None = Field(default=None,ge=0,le=3600)
    zone_id: str
    variable: Literal['density','speed','inflow','outflow','queue_length','exit_open']
    value: float

class Labels(Strict):
    actuals: list[Actual] = Field(max_length=100000)


def get_session(sid):
    if sid not in SESSIONS:
        raise HTTPException(404, 'Runtime session not found; render the scene again.')
    sess = SESSIONS[sid]
    sess['last_active_at'] = time.monotonic()
    return sess


def segment_distance(a,b,c,d):
    def cross(p,q,r):return (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x)
    def point_segment(p,q,r):
        dx=r.x-q.x;dy=r.y-q.y
        t=max(0,min(1,((p.x-q.x)*dx+(p.y-q.y)*dy)/(dx*dx+dy*dy or 1)))
        return math.hypot(p.x-q.x-t*dx,p.y-q.y-t*dy)
    if cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0:return 0
    return min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))


def venue_from_routes(data):
    """One measurable route corridor per zone; shared path points create graph edges.

    Route-wide reports cannot honestly be assigned to smaller unmeasured zones.
    Exit status stays missing unless an explicit gate observation is received.
    """
    zones, edges = [], []
    for r in data.routes:
        length = sum(math.hypot(a.x-b.x, a.y-b.y)*data.scale for a,b in zip(r.points,r.points[1:]))
        if length < 1:
            raise HTTPException(422, 'Routes must be at least one metre long.')
        zones.append(dict(zone_id=r.id, name=r.name, zone_type='exit', has_exit=True,
                          area_m2=length*r.width, capacity_definition=None,capacity_status='unknown',
                          x=sum(p.x for p in r.points)/len(r.points), y=sum(p.y for p in r.points)/len(r.points)))
    for i,a in enumerate(data.routes):
        for b in data.routes[i+1:]:
            distance=min(segment_distance(p,q,r,s)*data.scale for p,q in zip(a.points,a.points[1:]) for r,s in zip(b.points,b.points[1:]))
            if distance <= (a.width+b.width)/2:
                edges.append(dict(edge_id=f'connection_{len(edges)}',from_zone=a.id,to_zone=b.id,
                                  width_m=min(a.width,b.width),distance_m=distance,direction='bidirectional'))
    return VenueLayout(zones=zones, connections=edges)


@router.post('/sessions')
def start(data: Start):
    if len({r.id for r in data.routes}) != len(data.routes):
        raise HTTPException(422, 'Duplicate route IDs.')
    with LOCK:
        _evict_expired_sessions()
        if len(SESSIONS) >= MAX_SESSIONS:
            raise HTTPException(409, f'Runtime has {MAX_SESSIONS} active sessions; restart the local server or wait for idle sessions to expire.')
        venue=venue_from_routes(data)
        try:
            pipeline=ArchitecturePipeline(lookback_steps=settings.lookback_steps, horizons_sec=settings.horizons)
            pipeline.load_pipeline(settings.model_dir)
            readiness=pipeline.require_ready(settings.require_validated_models)
            # Verify artifacts against their saved training graph FIRST. Shared graph
            # weights can then run inductively on the scene graph; never rewrite hashes
            # or claim the new layout was validated by the original checkpoint.
            if settings.require_validated_models:
                raise ValueError('Custom route graphs require separate venue validation; strict serving is unavailable.')
            pipeline.venue=venue
            pipeline.graph_builder=SpatialGraphBuilder(venue.zones,venue.connections)
            pipeline.spatial.graph=pipeline.graph_builder
            readiness=dict(readiness, production_ready=False, serving_mode='experimental',
                custom_graph_transfer=True,
                notice=(readiness.get('notice') or '')+' Custom route graph uses shared checkpoint weights; this venue has not been validated.')
        except Exception as exc:
            raise HTTPException(503, f'Model cannot start: {exc}') from exc
        sid=uuid.uuid4().hex
        SESSIONS[sid]=dict(id=sid, run_id=data.run_id, mode=data.mode, duration=data.duration,
            venue=venue, pipeline=pipeline, readiness=readiness, time=-1, observations=[], seen={},
            forecasts=[], trees=[], steps=[], version=0, last_active_at=time.monotonic())
        EVALUATION[sid]={}
        return dict(session_id=sid, dashboard_url=f'/dashboard/?session={sid}', readiness=readiness)


def normalize(report, sid):
    mapping={'density_estimate_p_m2':'density','mean_speed_estimate_m_s':'speed',
             'entries_count':'inflow','exits_count':'outflow','queue_estimate':'queue_length',
             'exit_open_estimate':'exit_open'}
    records=[]
    for key,var in mapping.items():
        if key not in report.values:
            continue
        val=report.values[key]
        if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or val<0:
            raise HTTPException(422, f'Invalid numeric {key}')
        if var in ('inflow','outflow'):
            window=report.values.get('count_window_s',1)
            if isinstance(window,bool) or not isinstance(window,(int,float)) or not math.isfinite(window) or window<=0:
                raise HTTPException(422,'Invalid count window')
            val/=window
        if var=='exit_open' and val not in (0,1):
            raise HTTPException(422, 'Gate report must be 0 or 1')
        records.append(ObservationRecord(report_id=report.observation_id+':'+var,scenario_id=sid,
            zone_id=report.route_id,source_id=report.source_id,variable=var,value=val,unit=UNITS[var],
            observed_at=EPOCH+timedelta(seconds=report.observed_at_s),
            published_at=EPOCH+timedelta(seconds=report.available_at_s),
            received_at=EPOCH+timedelta(seconds=report.available_at_s)))
    return records


def serialize_forecast(f):
    return dict(stream=f.stream_id,zone_id=f.zone_id,variable=f.variable,unit=f.unit,
        issued_s=int((f.issued_at-EPOCH).total_seconds()),target_s=int((f.target_at-EPOCH).total_seconds()),
        available_s=int((f.received_at-EPOCH).total_seconds()),value=f.value,
        channel_mask=f.channel_mask,missing_data_mask=f.missing_data_mask,applicability=f.applicability,model_version=f.model_or_provider_version)


@router.post('/sessions/{sid}/advance')
def advance(sid: str, data: Advance):
    with LOCK:
        s=get_session(sid)
        if data.time_s>s['duration']:
            raise HTTPException(422, 'Time exceeds scenario duration.')
        # Clock increments exactly one second. Retried requests are idempotent.
        if data.time_s <= s['time']:
            return dict(time_s=s['time'], version=s['version'], duplicate=True)
        if data.time_s != s['time']+1:
            raise HTTPException(409,'Advance the runtime in order, one simulation second at a time.')
        ids={z['zone_id'] for z in s['venue'].zones}
        pending=[]
        for report in data.observations:
            if report.route_id not in ids or not 0<=report.observed_at_s<=report.available_at_s<=data.time_s:
                raise HTTPException(422,'Unknown zone or observation unavailable at this cutoff.')
            pending.extend(normalize(report,sid))
        pending_signatures={}
        for o in pending:
            signature=(o.zone_id,o.variable,o.source_id,o.value,o.observed_at,o.received_at)
            if o.report_id in pending_signatures and pending_signatures[o.report_id]!=signature:
                raise HTTPException(409,'A report ID was reused with different contents in one batch.')
            pending_signatures[o.report_id]=signature
            if o.report_id in s['seen'] and s['seen'][o.report_id]!=signature:
                raise HTTPException(409,'A report ID was reused with different contents.')
        for o in pending:
            if o.report_id not in s['seen']:
                s['observations'].append(o);s['seen'][o.report_id]=(o.zone_id,o.variable,o.source_id,o.value,o.observed_at,o.received_at)
        t=EPOCH+timedelta(seconds=data.time_s)
        # Model cadence remains the supplied five-second training contract.
        if data.time_s%5==0:
            trees,forecasts=s['pipeline'].step(sid,t,s['observations'])
            central=CentralSourceAdapter().generate_simulated_forecasts(sid,s['observations'],s['venue'],
                horizons_sec=settings.horizons,issue_times=[t])
            s['forecasts'].extend(forecasts+central)
            s['trees']=[dict(zone_id=r.zone_id,channels=r.channel_names,values=r.channel_values,masks=r.channel_masks) for r in trees]
        s['time']=data.time_s;s['version']+=1
        # Persist every cutoff for retrospective reports without exposing future labels.
        s['steps'].append(dict(time_s=s['time'],observations=len(s['observations'])))
        return dict(time_s=s['time'],version=s['version'],observation_count=len(s['observations']))


@router.post('/sessions/{sid}/evaluation')
def evaluation(sid: str, data: Labels):
    with LOCK:
        s=get_session(sid); zones={z['zone_id'] for z in s['venue'].zones}
        if any(a.time_s>s['time'] or a.zone_id not in zones or (a.available_at_s is not None and not a.time_s<=a.available_at_s<=s['time']) for a in data.actuals):
            raise HTTPException(422,'Evaluation labels must have arrived by the current clock and use known zones.')
        for a in data.actuals:
            EVALUATION[sid][(a.zone_id,a.variable,a.time_s)]=a.value
        s['version']+=1
        return dict(actual_count=len(EVALUATION[sid]))


def evidence(s):
    comparator=ForecastComparator();out=[]
    t=EPOCH+timedelta(seconds=max(0,s['time']))
    by_zone_var={}
    for o in s['observations']:
        if o.observed_at<=t:
            by_zone_var.setdefault((o.zone_id,o.variable),[]).append(o)
    for z in s['venue'].zones:
        for var in UNITS:
            recent=by_zone_var.get((z['zone_id'],var),[])
            rec=comparator.assess_information(recent,t,z['zone_id'],var,zone_type=z.get('zone_type'))
            flags=rec.per_variable_flags
            latest=max(recent,key=lambda o:(o.observed_at,o.received_at,o.report_id)) if recent else None
            # UI evidence index, NOT a trained probability or an input to forecasting.
            freshness=math.exp(-rec.observation_age_seconds/45)
            coverage=min(1,len({o.source_id for o in recent if (t-o.observed_at).total_seconds()<=90})/2)
            score=round(100*freshness*coverage*(.5 if flags.get('conflict') else 1)*( .75 if flags.get('rapidly_changing') else 1),1) if latest else None
            out.append(dict(zone_id=z['zone_id'],variable=var,observed_value=latest.value if latest else None,
                observation_age_s=rec.observation_age_seconds if latest else None,coverage=rec.coverage,
                coverage_basis=getattr(rec,'coverage_basis','No received source roster'),
                conflict_graph_summary={k:v for k,v in getattr(rec,'conflict_graph',{}).items() if k.startswith('num_')},
                conflict_count=len(rec.conflict_ids or []),flags=flags,trust_score=score))
    return out


def snapshot(sid):
    s=get_session(sid); clock=s['time'];actuals=EVALUATION[sid]
    ev=evidence(s); bykey={(e['zone_id'],e['variable']):e for e in ev}
    # Compare exact issue + exact target, never mix two forecast vintages.
    grouped={}
    for f in s['forecasts']:
        if f.received_at>EPOCH+timedelta(seconds=max(0,clock)):
            continue
        row=serialize_forecast(f);key=(row['zone_id'],row['variable'],row['issued_s'],row['target_s'])
        entry=grouped.setdefault(key,dict(zone_id=row['zone_id'],variable=row['variable'],unit=row['unit'],
            issued_s=row['issued_s'],target_s=row['target_s'],horizon_s=row['target_s']-row['issued_s'],
            architecture=None,central=None))
        stream='architecture' if f.stream_id=='stream_b_architecture' else 'central'
        entry[stream]=f.value
        for key in ('channel_mask','missing_data_mask','applicability','model_version'):entry[stream+'_'+key]=row[key]
    comparisons=list(grouped.values())
    metrics={}
    for row in comparisons:
        row['actual']=actuals.get((row['zone_id'],row['variable'],row['target_s'])) if row['target_s']<=clock else None
        row['trust_score']=bykey[(row['zone_id'],row['variable'])]['trust_score']
        row['difference']=round(row['architecture']-row['central'],4) if row['architecture'] is not None and row['central'] is not None else None
        for stream in ('architecture','central'):
            value=row[stream]
            if value is not None and row['actual'] is not None:
                key=f"{stream}/{row['variable']}/{row['horizon_s']}s"
                metrics.setdefault(key,[]).append(value-row['actual'])
    metrics={k:dict(count=len(v),mae=sum(abs(x) for x in v)/len(v),rmse=(sum(x*x for x in v)/len(v))**.5) for k,v in metrics.items()}
    latest_actual=[dict(zone_id=z,variable=v,time_s=t,value=a) for (z,v,t),a in actuals.items() if t==clock]
    return dict(alerts=alerts(s,ev,comparisons),cluster_assignments=getattr(s['pipeline'].tree_bank,'last_cluster_ids',None),session_id=sid,run_id=s['run_id'],mode=s['mode'],time_s=clock,duration=s['duration'],
        version=s['version'],readiness=s['readiness'],venue=s['venue'].model_dump(),
        observation_count=len(s['observations']),evidence=ev,tree_vectors=s['trees'],
        comparisons=comparisons,actuals=latest_actual,metrics=metrics,
        trust_method='UI evidence index (not accuracy): 100 × exp(-age/45) × min(1, fresh independent sources/2) × conflict penalty 0.5 × rapid-change penalty 0.75. Missing = unavailable. All five evidence labels remain separate. Score reflects evidence at dashboard clock, not forecast issue time.')


def alerts(s,evidence_rows,comparisons):
    """Information freshness and physical-risk advisories remain separate."""
    result=[]
    trained=all(s['pipeline'].manifest.training_status.get(k) in ('synthetic_validated','trained_validated') for k in ('hypernet','gnn','temporal'))
    for e in evidence_rows:
        if e['flags'].get('potentially_outdated'):
            result.append(dict(kind='information_stale',zone_id=e['zone_id'],variable=e['variable'],time_s=s['time'],age_s=e['observation_age_s'],reason='Last observation exceeds the variable freshness threshold. Age alone does not establish falsity.'))
    if trained:
        latest={}
        for row in comparisons:
            if row['target_s']<=s['time'] or row['architecture'] is None:continue
            key=(row['zone_id'],row['variable'],row['horizon_s'])
            if key not in latest or row['issued_s']>latest[key]['issued_s']:latest[key]=row
        for row in latest.values():
            # Demonstration policy, explicitly not a calibrated safety threshold.
            threshold={'density':3.,'queue_length':20.}.get(row['variable'])
            if threshold is not None and row['architecture']>=threshold:
                result.append(dict(kind='synthetic_risk_advisory',zone_id=row['zone_id'],variable=row['variable'],time_s=s['time'],target_s=row['target_s'],predicted_value=row['architecture'],threshold=threshold,reason='Synthetic forecast crossed a configurable demonstration threshold; not an incident-onset prediction.'))
    return result


@router.get('/sessions')
def sessions():
    with LOCK:
        return [dict(session_id=k,run_id=s['run_id'],mode=s['mode'],time_s=s['time'],duration=s['duration']) for k,s in SESSIONS.items()]


@router.get('/sessions/{sid}')
def report(sid: str):
    with LOCK:
        get_session(sid)
        return snapshot(sid)


@router.get('/sessions/{sid}/report.csv')
def export(sid: str):
    with LOCK:
        get_session(sid);data=snapshot(sid);out=io.StringIO()
        keys=['zone_id','variable','unit','issued_s','target_s','horizon_s','architecture','central','actual','difference','trust_score','architecture_model_version','central_model_version','architecture_channel_mask','central_channel_mask','architecture_missing_data_mask','central_missing_data_mask','architecture_applicability','central_applicability']
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(data['comparisons'])
        return Response(out.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="runtime_comparison.csv"'})
