"""Adversarial boundary and learned-context regressions from the audit."""
import copy,json,subprocess,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pytest
from crowd_reliability.ml.features import build_feature_row
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.ml.tree_bank import DecisionTreeBank,TREE_CHANNELS
from crowd_reliability.ml.context import context_matrix,ContextExpertRouter
from crowd_reliability.ml.graph import SpatialGraphBuilder
from crowd_reliability.models.schema import ObservationRecord,ForecastRecord,ReferenceStateRecord
from crowd_reliability.services.comparator import ForecastComparator
from crowd_reliability.services.evaluator import EvaluationService
from crowd_reliability.simulation.scenarios import get_default_venue,SCENARIO_PRESETS
from crowd_reliability.simulation.simulator import CrowdSimulator
from crowd_reliability.simulation.generator import ObservationGenerator
T=datetime(2026,10,14,12)
Z='concourse'
def observation(**kwargs):
 d=dict(report_id='r',scenario_id='A',zone_id=Z,source_id='cam',variable='density',value=2.,unit='p/m2',observed_at=T,published_at=T,received_at=T);d.update(kwargs);return ObservationRecord(**d)
@pytest.mark.parametrize('change',[{'value':float('nan')},{'value':float('inf')},{'value':-1},{'unit':'persons'},{'observed_at':T+timedelta(seconds=100)}])
def test_invalid_observation_rejected(change):
 with pytest.raises(ValueError):build_feature_row(Z,T,[observation(**change)],scenario_id='A')
def test_equal_time_reports_have_stable_tie_break():
 a=observation(report_id='a',value=1);b=observation(report_id='b',value=9)
 assert np.array_equal(build_feature_row(Z,T,[a,b]),build_feature_row(Z,T,[b,a]))
def test_reused_id_rejected():
 with pytest.raises(ValueError):build_feature_row(Z,T,[observation(value=1),observation(value=9)])
def test_pipeline_isolates_scenarios_and_clock():
 pipeline=ArchitecturePipeline();result=pipeline.step('A',T,[observation(scenario_id='B',value=99)])
 assert pipeline.history_buffer['A'][Z][-1][0]==.8
 assert pipeline.step('A',T,[]) == result
 assert len(pipeline.history_buffer['A'][Z])==1
 with pytest.raises(ValueError):pipeline.step('A',T-timedelta(seconds=5),[])
@pytest.mark.parametrize('label',[0.,1.])
def test_single_class_exit_probability(label):
 x=np.zeros((12,18));y={k:np.ones(12) for k in TREE_CHANNELS};y[TREE_CHANNELS[-1]][:]=label
 bank=DecisionTreeBank();bank.fit(x,y);assert bank.predict_zone(x[0],True)[0][-1]==label

def forecast(id='a',scenario='A',unit='p/m2',value=2,issue=T):
 return ForecastRecord(forecast_id=id,scenario_id=scenario,zone_id=Z,variable='density',unit=unit,value=value,issued_at=issue,received_at=issue,target_at=T+timedelta(seconds=30),stream_id='stream_b_architecture')
@pytest.mark.parametrize('change',[{'scenario':'B'},{'unit':'persons'}])
def test_incompatible_forecasts_not_subtracted(change):
 rows=ForecastComparator().align_and_compare([forecast()],[forecast(id='b',**change)])
 assert all(r.signed_diff is None for r in rows)
def test_revision_selection_stable_and_cutoff_applied():
 a=forecast(id='a',value=1);b=forecast(id='b',value=9);b.received_at=T+timedelta(seconds=5);c=ForecastComparator()
 assert c.align_and_compare([a,b],[],T)[0].arch_forecast.value==1
 assert c.align_and_compare([a,b],[])[0].arch_forecast.value==c.align_and_compare([b,a],[])[0].arch_forecast.value==9
def test_labels_unavailable_until_release():
 ref=ReferenceStateRecord(scenario_id='A',zone_id=Z,variable='density',unit='p/m2',state_at=T+timedelta(seconds=30),label_available_at=T+timedelta(seconds=60),value=3)
 scorer=EvaluationService()
 assert not scorer.score_predictions([forecast()],[ref],evaluation_time=T+timedelta(seconds=59))
 assert scorer.score_predictions([forecast()],[ref],evaluation_time=T+timedelta(seconds=60))
def test_seed_reproduces_observation_ids_and_delayed_measurements():
 c=SCENARIO_PRESETS['bottleneck_exit_closure'].model_copy(update={'duration_seconds':140,'stale_report_probability':1.})
 refs,_=CrowdSimulator().run_simulation(c);gen=ObservationGenerator();a=gen.generate_observations(refs,c);b=gen.generate_observations(refs,c)
 encode=lambda rows:[(r.report_id,r.variable,r.observed_at,r.received_at,r.value) for r in rows]
 assert encode(a)==encode(b)
 lookup={(r.zone_id,r.variable,r.state_at):r.value for r in refs}
 stale=[r for r in a if r.report_id.startswith('rep_stale_')];assert stale
 assert all(r.value==lookup[(r.zone_id,r.variable,r.observed_at)] for r in stale)
def test_local_context_depends_on_topology_and_masks():
 venue=get_default_venue();graph=SpatialGraphBuilder(venue.zones,venue.connections);empty=SpatialGraphBuilder(venue.zones,[])
 x=np.ones((len(venue.zones),18),dtype=np.float32)
 assert context_matrix(x,graph).shape==(len(x),42)
 assert not np.array_equal(context_matrix(x,graph),context_matrix(x,empty))
 x[:,12:18]=0
 a=context_matrix(x,graph);x[:,:6]=999;b=context_matrix(x,graph)
 assert np.array_equal(a[:,18:30],b[:,18:30])
def test_context_router_serialization_and_unknown_fallback(tmp_path):
 rng=np.random.default_rng(1);x=rng.normal(size=(80,42));labels={c:np.ones(80) for c in TREE_CHANNELS}
 router=ContextExpertRouter().fit(x,labels,n_clusters=2,min_samples=10)
 values,masks,clusters=router.predict(np.full((1,42),999.),[True])
 assert clusters==[-1] and values.shape==(1,6)
 import joblib
 file=tmp_path/'router.joblib';joblib.dump(router,file)
 assert np.array_equal(values,joblib.load(file).predict(np.full((1,42),999.),[True])[0])
def test_utf8_report_even_under_windows_default_codec():
 code="""from pathlib import Path
old=Path.read_text
def windows(self,encoding=None,**kw):return old(self,encoding=encoding or 'cp1252',**kw)
Path.read_text=windows
import sys;sys.path.insert(0,'studio')
from reporting import render_report
assert 'text/html' not in render_report({})[:100]
print('UTF8_OK')
"""
 assert 'UTF8_OK' in subprocess.check_output([sys.executable,'-c',code],text=True)
def test_shipped_selected_model_is_trained_but_not_production_validated():
 root=Path('models/candidates/mc_v2')
 if not root.exists():pytest.skip('Run offline training first.')
 pipeline=ArchitecturePipeline();pipeline.load_pipeline(str(root))
 assert pipeline.tree_bank.context_model is not None
 status=pipeline.readiness();assert status['synthetic_only'] and not status['production_ready']
 report=json.loads((root/'selection_report.json').read_text(encoding='utf-8'))
 sets=[set(v) for v in report['split_seeds'].values()]
 assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
 assert report['selected']==min(report['trials'],key=lambda x:x['validation_loss'])['config']

def test_forecasts_of_different_horizons_never_compare():
 a=forecast(id='a',issue=T);b=forecast(id='b',issue=T+timedelta(seconds=5))
 rows=ForecastComparator().align_and_compare([a],[b])
 assert len(rows)==2 and all(r.signed_diff is None for r in rows)

def test_evaluation_requires_exact_target_timestamp():
 ref=ReferenceStateRecord(scenario_id='A',zone_id=Z,variable='density',unit='p/m2',state_at=T+timedelta(seconds=31),label_available_at=T+timedelta(seconds=31),value=3)
 assert not EvaluationService().score_predictions([forecast()],[ref],evaluation_time=T+timedelta(seconds=60))
