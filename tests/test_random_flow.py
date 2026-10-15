"""New curves, density/direction contracts, hidden scenarios and honest evaluation."""
import copy
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient
from crowd_reliability.config import settings
from crowd_reliability import runtime as rt
from crowd_reliability.live_provider import WORLDS
from crowd_reliability.random_flow import Flow,make_scene,segment_ok
from crowd_reliability.flow_evaluation import plan_from_observations,monitor
from runtime_server import app
from simulation import live_steps,path

def test_curve_reproducibility_direction_density_and_walkable_space():
    c,m=make_scene(Flow(seed=731));again,_=make_scene(Flow(seed=731));assert c==again
    reverse,_=make_scene(Flow(seed=731,direction='inbound'))
    other,_=make_scene(Flow(seed=732));assert c['routes'][0]['points']!=other['routes'][0]['points']
    for a,b in zip(c['routes'],reverse['routes']):
        assert a['points']==b['points'][::-1]
        assert len(a['points'])>20
        assert all(segment_ok((p['x'],p['y']),(q['x'],q['y'])) for p,q in zip(a['points'],a['points'][1:]))
        assert abs(a['count']/(path(a,.1)['length']*a['width'])-.8)<.01
    initial=next(live_steps(c,'custom'))
    assert initial['active']==m['initial_population']
    assert initial['active']+initial['completed']+initial['waiting']<=initial['total']

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings,'model_dir',str(Path(__file__).resolve().parents[1]/'models/candidates/mc_v2'))
    rt.SESSIONS.clear();rt.EVALUATION.clear();WORLDS.clear()
    with TestClient(app) as c:yield c

def test_random_api_hidden_schedule_model_and_observation_only_planning(client):
    spec=dict(seed=731,density=.7,duration=90,scenario='combined',route_count=1)
    d=client.post('/api/runtime/worlds/draft',json=spec);assert d.status_code==200,d.text
    assert d.json()['scene']['events']==[]
    r=client.post('/api/runtime/worlds/random',json=spec);assert r.status_code==200,r.text
    w=r.json();wid=w['world_id'];sid=w['session_id'];url='/api/runtime/worlds/'+wid
    assert w['events']==[] and w['flow']['scenario']=='hidden until observed'
    assert WORLDS[wid]['scene']['events']
    for t in range(46):
        r=client.post(url+'/tick',json={'expected_time_s':t-1});assert r.status_code==200,r.text
        w=r.json();assert w['runtime']['time_s']==t
        assert all(e['start']<=t for e in w['events'])
        assert all((o.received_at-rt.EPOCH).total_seconds()<=t for o in rt.SESSIONS[sid]['observations'])
    assert w['assessment']['matured_forecasts']>0
    assert w['runtime']['metrics']
    before=copy.deepcopy(w['runtime']['comparisons'])
    r=client.post(url+'/plan',json={'horizon_s':40});assert r.status_code==200,r.text
    p=r.json()['plan'];assert len(p['trials'])>=2
    assert any(a['action']=='continue' for a in p['trials'])
    assert p['notice'].startswith('Best among tested')
    assert p['trials'][0]['exposure_index']<=p['trials'][-1]['exposure_index']
    assert r.json()['runtime']['comparisons']==before
    # Mutating the world's unseen incident schedule cannot alter a plan seeded by reports.
    private=copy.deepcopy(WORLDS[wid]['scene']['events']);WORLDS[wid]['scene']['events']=[]
    again=client.post(url+'/plan',json={'horizon_s':40}).json()['plan']
    assert p==again
    WORLDS[wid]['scene']['events']=private
    assert client.post(url+'/apply',json={'action':'meter'}).status_code==200
    assert WORLDS[wid]['scene']['meter_from']==46


def test_monitor_reports_missed_warning_instead_of_inventing_success():
    w={'time':0,'policy':{'density':2.5,'queue':15},'scene':{'routes':[{'id':'r'}]}}
    for t in range(8):
        w['time']=t
        a=monitor(w,{'comparisons':[]},{'rows':[{'route_id':'r','mean_density_p_m2':3,'queued':0}]})
    assert a['episodes'][0]['lead_time_s'] is None
    assert a['episodes'][0]['status']=='no_advance_warning'
    assert a['precision'] is None
