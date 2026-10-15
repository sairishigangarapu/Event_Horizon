"""Operator-only synthetic live world; the forecaster receives released reports only."""
import copy
from pathlib import Path
import sys
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import Field
from typing import Literal
from crowd_reliability import runtime as rt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studio'))
from simulation import validate, live_steps
from observations import generate_observations, available_at, sources_for

router=APIRouter(prefix='/api/runtime/worlds',tags=['Synthetic live demonstrator'])
WORLDS={}
class Create(rt.Strict):
    scene: dict
class Tick(rt.Strict):
    expected_time_s: int = Field(ge=-1,le=900)
class Action(rt.Strict):
    kind: Literal['surge','close','open']
    route_id: str
    people: int = Field(default=50,ge=1,le=500)
    duration_s: int = Field(default=20,ge=1,le=120)

def world(wid):
    if wid not in WORLDS:raise HTTPException(404,'Live world not found; start a new demo.')
    return WORLDS[wid]

def view(w):
    result=w.get('result')
    return dict(world_id=w['id'],session_id=w['sid'],time_s=w['time'],duration=w['scene']['duration'],
        routes=w['scene']['routes'],scale=w['scene']['scale'],
        frame=result['frames'][-1] if result else [],
        active=result['active'] if result else 0,waiting=result['waiting'] if result else 0,
        completed=result['completed'] if result else 0,
        events=[e for e in w['scene']['events'] if e['end']>w['time']],
        runtime=rt.snapshot(w['sid']))

@router.post('')
def create(data:Create):
    scene=copy.deepcopy(data.scene)
    try:validate(scene);sources_for(scene)
    except (ValueError,TypeError,KeyError) as exc:raise HTTPException(422,str(exc)) from exc
    # Interactive worlds are bounded to keep one-worker demo hosting responsive.
    if scene['duration']>300 or sum(r['count'] for r in scene['routes'])>600:
        raise HTTPException(422,'Live demo limit: 300 seconds and 600 initial people.')
    with rt.LOCK:
        wid=uuid.uuid4().hex
        start=rt.Start(run_id=wid,mode='custom',scale=scene['scale'],duration=scene['duration'],
            routes=[{k:r[k] for k in ('id','name','points','width')} for r in scene['routes']])
        session=rt.start(start)
        w=dict(id=wid,sid=session['session_id'],scene=scene,time=-1,steps=live_steps(scene,'custom'),result=None)
        WORLDS[wid]=w
        return view(w)

@router.post('/{wid}/tick')
def tick(wid:str,data:Tick):
    with rt.LOCK:
        w=world(wid)
        # A retried request returns the committed frame rather than stepping twice.
        if data.expected_time_s<w['time']:return view(w)
        if data.expected_time_s!=w['time']:raise HTTPException(409,'Clock mismatch; reload the live world.')
        if w['time']>=w['scene']['duration']:return view(w)
        result=next(w['steps']);t=w['time']+1
        reports=[r for r in available_at(generate_observations(w['scene'],result),t) if r['available_at_s']>w['time']]
        rt.advance(w['sid'],rt.Advance(time_s=t,observations=reports))
        # Labels go to evaluation AFTER inference at this cutoff, never to its inputs.
        keys={'mean_density_p_m2':'density','mean_speed_m_s':'speed','spawned':'inflow','exited':'outflow','queued':'queue_length'}
        labels=[dict(time_s=t,zone_id=r['route_id'],variable=var,value=r[key])
            for r in result['rows'][-len(w['scene']['routes']):] for key,var in keys.items()]
        rt.evaluation(w['sid'],rt.Labels(actuals=labels))
        w.update(time=t,result=result)
        return view(w)

@router.post('/{wid}/actions')
def action(wid:str,data:Action):
    with rt.LOCK:
        w=world(wid);c=w['scene'];start=w['time']+1
        if start>=c['duration']:raise HTTPException(409,'This demo has finished.')
        if data.route_id not in {r['id'] for r in c['routes']}:raise HTTPException(422,'Unknown route.')
        if data.kind=='open':
            for e in list(c['events']):
                if e['type']=='collapse' and e['routeId']==data.route_id and e['end']>start:
                    if e['start']>=start:c['events'].remove(e)
                    else:e['end']=start
        else:
            candidate=copy.deepcopy(c)
            candidate['events'].append(dict(id=uuid.uuid4().hex,name='Live '+data.kind,type='surge' if data.kind=='surge' else 'collapse',
                routeId=data.route_id,at=.12 if data.kind=='surge' else .97,start=start,
                end=min(c['duration'],start+data.duration_s),people=data.people if data.kind=='surge' else 0,radiusM=3))
            try:validate(candidate)
            except ValueError as exc:raise HTTPException(422,str(exc)) from exc
            c['events'].append(candidate['events'][-1])
        return view(w)

@router.get('/{wid}')
def get(wid:str):
    with rt.LOCK:return view(world(wid))
