"""Causal integration contract against the real supplied model artifacts."""
import copy
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from crowd_reliability.config import settings
from crowd_reliability.runtime import SESSIONS, EVALUATION
from runtime_server import app

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings,'model_dir',str(Path(__file__).resolve().parents[1]/'models/production'))
    SESSIONS.clear();EVALUATION.clear()
    with TestClient(app) as c:yield c


def start(c):
    r=c.post('/api/runtime/sessions',json=dict(run_id='a'*32,mode='custom',scale=.1,duration=60,
       routes=[dict(id='r1',name='Main route',width=3,points=[dict(x=100,y=100),dict(x=500,y=100)])]))
    assert r.status_code==200,r.text
    return r.json()['session_id']


def obs(t,value=1):
    return dict(observation_id=f'camera:r1:{t}',source_id='camera_1',source_kind='camera',route_id='r1',
      observed_at_s=t,available_at_s=t,values=dict(density_estimate_p_m2=value,mean_speed_estimate_m_s=1.2))


def test_clock_validation_and_no_future_labels(client):
    sid=start(client);url=f'/api/runtime/sessions/{sid}'
    assert client.post(url+'/advance',json=dict(time_s=2,observations=[])).status_code==409
    assert client.post(url+'/advance',json=dict(time_s=0,observations=[obs(1)])).status_code==422
    assert client.post(url+'/advance',json=dict(time_s=0,observations=[obs(0)])).status_code==200
    assert client.post(url+'/advance',json=dict(time_s=0,observations=[obs(0)])).json()['duplicate']
    assert client.post(url+'/evaluation',json={'actuals':[dict(time_s=30,zone_id='r1',variable='density',value=99)]}).status_code==422
    assert client.get(url).json()['actuals']==[]
    assert all(x['actual'] is None for x in client.get(url).json()['comparisons'])


def test_actuals_do_not_change_inference_and_reports_mature(client):
    sid=start(client);url=f'/api/runtime/sessions/{sid}'
    for t in range(36):
        r=client.post(url+'/advance',json=dict(time_s=t,observations=[obs(t)]));assert r.status_code==200,r.text
    before=client.get(url).json()
    assert before['tree_vectors'][0]['channels'] and len(before['tree_vectors'][0]['values'])==6
    assert before['readiness']['production_ready'] is False
    assert client.post(url+'/evaluation',json={'actuals':[dict(time_s=30,zone_id='r1',variable='density',value=2)]}).status_code==200
    after=client.get(url).json()
    assert SESSIONS[sid]['pipeline'].history_buffer
    assert before['tree_vectors']==after['tree_vectors']
    assert [(x['central'],x['architecture']) for x in before['comparisons']]==[(x['central'],x['architecture']) for x in after['comparisons']]
    matured=[x for x in after['comparisons'] if x['target_s']==30 and x['variable']=='density']
    assert matured and all(x['actual']==2 for x in matured)
    assert after['metrics']['architecture/density/30s']['count']==1
    assert 'trust_score' in client.get(url+'/report.csv').text
    assert any(x['central'] is not None and x['architecture'] is not None for x in after['comparisons'])
    assert next(x for x in after['evidence'] if x['variable']=='exit_open')['trust_score'] is None


def test_unknown_and_forbidden_inputs(client):
    sid=start(client);url=f'/api/runtime/sessions/{sid}'
    assert client.post(url+'/advance',json=dict(time_s=0,observations=[],hidden_ground_truth={})).status_code==422
    report=obs(0);report['route_id']='wrong'
    assert client.post(url+'/advance',json=dict(time_s=0,observations=[report])).status_code==422
    assert client.get('/dashboard/').status_code==200


def test_custom_graph_connects_crossings_without_shared_vertices(client):
    from crowd_reliability.runtime import Start, venue_from_routes
    data=Start(run_id='a'*32,mode='ideal',duration=60,scale=.1,routes=[
       dict(id='h',name='Horizontal',width=2,points=[dict(x=100,y=300),dict(x=900,y=300)]),
       dict(id='v',name='Vertical',width=2,points=[dict(x=500,y=100),dict(x=500,y=600)])])
    venue=venue_from_routes(data)
    assert len(venue.connections)==1
    assert venue.connections[0]['distance_m']==0


def test_changed_report_id_is_rejected_atomically(client):
    sid=start(client);url=f'/api/runtime/sessions/{sid}'
    a=obs(0,1);b=obs(0,9)
    response=client.post(url+'/advance',json=dict(time_s=0,observations=[a,b]))
    assert response.status_code==409
    assert client.get(url).json()['time_s']==-1
    assert client.get(url).json()['observation_count']==0
