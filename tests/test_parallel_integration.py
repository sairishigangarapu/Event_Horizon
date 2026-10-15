"""Cross-branch regressions: API boundary, old DB upgrade, forecast vintages and masks."""
import copy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine,inspect,text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from crowd_reliability import runtime as rt
from crowd_reliability.config import settings
from crowd_reliability.database import upgrade_forecast_columns,get_db
from crowd_reliability.models.schema import Base,ForecastRecord
from crowd_reliability.api import app
import runtime_server  # Registers the integrated frontend/provider API.

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings,'auth_mode','local_demo')
    monkeypatch.setattr(settings,'model_dir',str(Path(__file__).resolve().parents[1]/'models/candidates/mc_v2'))
    rt.SESSIONS.clear();rt.EVALUATION.clear()
    with TestClient(app) as c:yield c

def test_auth_fails_closed_and_covers_new_and_old_apis(client,monkeypatch):
    monkeypatch.setattr(settings,'auth_mode','api_key');monkeypatch.setattr(settings,'app_api_key','')
    monkeypatch.setattr(settings,'central_api_key','upstream-is-not-an-inbound-key')
    assert client.get('/api/runtime/sessions',headers={'X-API-Key':'anything'}).status_code==503
    monkeypatch.setattr(settings,'app_api_key','test-session-only-key')
    for path in ['/api/runtime/sessions','/api/twin-v3/evaluations','/api/scenarios','/api/manifest']:
        assert client.get(path).status_code==401,path
        assert client.get(path,headers={'X-API-Key':'upstream-is-not-an-inbound-key'}).status_code==401
    assert client.get('/api/runtime/sessions',headers={'X-API-Key':'test-session-only-key'}).status_code==200
    assert client.get('/api/scenarios',headers={'X-API-Key':'test-session-only-key'}).status_code==200
    assert client.get('/dashboard/').status_code==200

def test_cross_origin_world_mutation_is_rejected(client):
    assert client.post('/api/runtime/worlds/random',json={},headers={'Origin':'https://attacker.example'}).status_code==403

def test_additive_upgrade_preserves_pre_audit_forecasts():
    engine=create_engine('sqlite:///:memory:')
    with engine.begin() as c:
        c.execute(text('CREATE TABLE forecasts (id INTEGER PRIMARY KEY, value FLOAT)'))
        c.execute(text('INSERT INTO forecasts VALUES (1, 4.5)'))
    assert set(upgrade_forecast_columns(engine))=={'channel_mask','missing_data_mask','applicability'}
    assert upgrade_forecast_columns(engine)==[]
    with engine.connect() as c:
        row=c.execute(text('SELECT value,channel_mask,missing_data_mask,applicability FROM forecasts')).one()
    assert tuple(row)==(4.5,None,None,1.0)

def test_core_api_pages_exact_forecast_keys_and_revisions_in_sql(client):
    engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
    Base.metadata.create_all(engine);Session=sessionmaker(bind=engine)
    t=datetime(2026,1,1)
    with Session() as db:
        for i in range(4):
            for stream in ('stream_a_central','stream_b_architecture'):
                for revision in range(3):
                    db.add(ForecastRecord(forecast_id=f'{stream}_{i}_{revision}',stream_id=stream,
                        scenario_id='integration',zone_id='r',variable='density',unit='p/m2',
                        issued_at=t+timedelta(seconds=i*5),received_at=t+timedelta(seconds=i*5+revision),
                        target_at=t+timedelta(seconds=i*5+30),value=i+revision,
                        model_or_provider_version='test'))
        db.commit()
    def override():
        with Session() as db:yield db
    app.dependency_overrides[get_db]=override
    try:
        result=client.get('/api/forecasts',params=dict(scenario_id='integration',as_of='2026-01-01T00:01:00Z',limit=2,offset=1))
        assert result.status_code==200,result.text
        d=result.json();assert d['total_comparisons']==4 and len(d['comparisons'])==2
        assert [r['arch_value'] for r in d['comparisons']]==[3,4]
        assert all(r['signed_diff']==0 for r in d['comparisons'])
    finally:app.dependency_overrides.pop(get_db,None);engine.dispose()

def test_real_provider_exports_masks_and_model_versions(client):
    import json
    expected=json.loads((Path(settings.model_dir)/'manifest.json').read_text())['pipeline_version']
    assert client.get('/api/manifest').json()['pipeline_version']==expected
    scene=dict(duration=40,scale=.1,seed=42,randomCount=1,events=[],routes=[dict(id='r',name='Route',count=30,speed=1.3,width=3,points=[dict(x=100,y=200),dict(x=500,y=200)])])
    w=client.post('/api/runtime/worlds',json={'scene':scene}).json();url='/api/runtime/worlds/'+w['world_id']
    for t in range(6):
        r=client.post(url+'/tick',json={'expected_time_s':t-1});assert r.status_code==200,r.text
    w=r.json();rows=w['runtime']['comparisons'];assert rows
    assert all(len(r['architecture_channel_mask'])==6 for r in rows if r['architecture'] is not None)
    assert all(len(r['architecture_missing_data_mask'])==6 for r in rows if r['architecture'] is not None)
    csv=client.get('/api/runtime/sessions/'+w['session_id']+'/report.csv')
    assert csv.status_code==200 and 'architecture_model_version' in csv.text

def test_rate_limit_refills_and_bounds_client_state():
    from crowd_reliability.rate_limit import ApiRateLimit
    limit=ApiRateLimit(max_clients=2)
    assert all(limit.allow('a',60,now=0) for _ in range(60))
    assert not limit.allow('a',60,now=0)
    assert limit.allow('a',60,now=1)
    limit.allow('b',60,now=1);limit.allow('c',60,now=1)
    assert len(limit.clients)==2


def test_person4_does_not_pair_different_forecast_vintages_or_units():
    from person4.event_horizon_p4.comparison import ForecastComparator
    from person4.event_horizon_p4.inference import ForecastRecord as P4
    t=datetime(2026,1,1)
    b=P4('b','stream_b_architecture','s','z','density','p/m2',t,t,t+timedelta(seconds=60),2,'test',60)
    a=P4('a','stream_a_central','s','z','density','p/m2',t+timedelta(seconds=30),t+timedelta(seconds=30),t+timedelta(seconds=60),1,'test',30)
    assert not ForecastComparator().compare([b],[a]).paired
    a.issued_at=t;a.received_at=t;a.horizon_sec=60
    report=ForecastComparator().compare([b],[a]);assert len(report.paired)==1


def test_person4_evaluator_filters_unavailable_forecasts_and_incompatible_labels():
    from person4.event_horizon_p4.inference import ForecastRecord as P4
    from person4.event_horizon_p4.evaluation import ReferenceEvaluator
    from person4.event_horizon_p4.synthetic import MockReferenceState
    t=datetime(2026,1,1);target=t+timedelta(seconds=30)
    forecast=P4('b','stream_b_architecture','s','z','density','p/m2',t,t+timedelta(seconds=90),target,2,'test',30)
    label=MockReferenceState('s','z','density',1,'p/m2',target,target)
    assert ReferenceEvaluator().score([forecast],[label],evaluation_time=target).n_matched==0
    forecast.received_at=t;label.unit='persons'
    assert ReferenceEvaluator().score([forecast],[label],evaluation_time=target).n_matched==0


def test_graph_unknown_capacity_contract_survives_both_branches():
    from crowd_reliability.ml.graph import SpatialGraphBuilder
    graph=SpatialGraphBuilder([dict(zone_id='z',zone_type='exit',has_exit=True,area_m2=10,capacity_definition=None,capacity_status='unknown')],[])
    assert graph.unknown_capacity_zones==['z']
    assert graph.node_static.shape==(1,8)


def test_canonical_assessment_reports_source_coverage_not_age_proxy():
    from crowd_reliability.services.comparator import ForecastComparator
    from crowd_reliability.models.schema import ObservationRecord
    t=datetime(2026,1,1,0,2)
    def obs(rid,source,age):return ObservationRecord(report_id=rid,scenario_id='s',zone_id='r',source_id=source,variable='density',value=1,unit='p/m2',observed_at=t-timedelta(seconds=age),published_at=t-timedelta(seconds=age),received_at=t-timedelta(seconds=age))
    a=ForecastComparator().assess_information([obs('a','camera',1),obs('b','warden',100)],t,'r','density')
    assert a.coverage==.5 and a.observation_age_seconds==1
    assert 'num_nodes' in a.conflict_graph

def test_central_csv_rejects_nonfinite_values_and_preserves_empty_schema(tmp_path):
    import csv
    from crowd_reliability.services.central_source import CentralSourceAdapter
    adapter=CentralSourceAdapter();empty=tmp_path/'empty.csv'
    adapter.export_to_csv([],str(empty));assert adapter.load_from_csv(str(empty))==[]
    t=datetime(2026,1,1)
    record=ForecastRecord(forecast_id='test',stream_id='stream_a_central',scenario_id='s',zone_id='r',variable='density',unit='p/m2',issued_at=t,received_at=t,target_at=t+timedelta(seconds=30),value=1,model_or_provider_version='test',channel_mask=[1,0,0,0,0,0],missing_data_mask=[0,1,1,1,1,1],applicability=1)
    file=tmp_path/'bad.csv';adapter.export_to_csv([record],str(file))
    with file.open() as f:rows=list(csv.DictReader(f));fields=list(rows[0])
    rows[0]['value']='nan'
    with file.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='Invalid forecast value'):adapter.load_from_csv(str(file))

def test_api_rejects_oversized_bodies_before_model_work(client,monkeypatch):
    monkeypatch.setattr(settings,'api_max_body_bytes',64)
    r=client.post('/api/runtime/worlds/random',content=b'x'*65,headers={'Content-Type':'application/json'})
    assert r.status_code==413
    r=client.post('/api/runtime/worlds/random',content=(part for part in [b'x'*40,b'x'*40]),headers={'Content-Type':'application/json'})
    assert r.status_code==413
