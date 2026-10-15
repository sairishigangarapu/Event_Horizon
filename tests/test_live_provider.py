"""Live intervention causality, ordered transport, and real-model integration."""
import copy
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient
from crowd_reliability.config import settings
from crowd_reliability import runtime as rt
from crowd_reliability.live_provider import WORLDS
from runtime_server import app
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studio'))
from simulation import live_steps, simulate
from observations import generate_observations, available_at

def scene():
    return dict(duration=40,scale=.1,seed=42,randomCount=1,events=[],routes=[dict(id='r',name='Main',color='#416bb7',count=40,speed=1.3,width=3,points=[dict(x=100,y=200),dict(x=500,y=200)])])

def test_live_intervention_preserves_history_and_observations():
    c=scene();steps=live_steps(c,'custom')
    for _ in range(7):before=next(steps)
    rows=copy.deepcopy(before['rows']);reports=available_at(generate_observations(c,before),6)
    c['events'].append(dict(id='surge',name='Live surge',type='surge',routeId='r',at=.12,start=7,end=12,people=30))
    after=next(steps)
    assert after['rows'][:len(rows)]==rows
    assert available_at(generate_observations(c,after),6)==reports
    assert after['rows'][-1]['spawned']>simulate(scene(),'custom')['rows'][7]['spawned']

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings,'model_dir',str(Path(__file__).resolve().parents[1]/'models/candidates/mc_v2'))
    rt.SESSIONS.clear();rt.EVALUATION.clear();WORLDS.clear()
    with TestClient(app) as c:yield c

def test_live_world_real_model_delay_retry_and_interventions(client):
    r=client.post('/api/runtime/worlds',json={'scene':scene()});assert r.status_code==200,r.text
    w=r.json();url='/api/runtime/worlds/'+w['world_id'];sid=w['session_id']
    assert client.post(url+'/tick',json={'expected_time_s':2}).status_code==409
    for t in range(36):
        if t==5:
            assert client.post(url+'/actions',json={'kind':'close','route_id':'r','duration_s':10}).status_code==200
        if t==8:
            assert client.post(url+'/actions',json={'kind':'surge','route_id':'r','people':30,'duration_s':5}).status_code==200
        if t==10:
            assert client.post(url+'/actions',json={'kind':'open','route_id':'r'}).status_code==200
        r=client.post(url+'/tick',json={'expected_time_s':t-1});assert r.status_code==200,r.text
        w=r.json();assert w['time_s']==t==w['runtime']['time_s']
        assert all((o.received_at-rt.EPOCH).total_seconds()<=t for o in rt.SESSIONS[sid]['observations'])
        assert all(x['actual'] is None for x in w['runtime']['comparisons'] if x['target_s']>t)
        if t==0:assert w['runtime']['observation_count']==0
        retry=client.post(url+'/tick',json={'expected_time_s':t-1}).json()
        assert retry['time_s']==t and retry['frame']==w['frame']
    assert w['runtime']['metrics'] and w['runtime']['observation_count']>0
    assert w['runtime']['tree_vectors']
    assert any(x['architecture'] is not None for x in w['runtime']['comparisons'])
    assert client.post(url+'/actions',json={'kind':'close','route_id':'missing'}).status_code==422
