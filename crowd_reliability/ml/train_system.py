"""Offline Monte Carlo selection of clustered trees + hypernetwork/GNN/GRU.

Train/validation/test scenario seeds never overlap. No models are trained online.
Artifacts remain labelled synthetic_validated, never real-world validated.
"""
import argparse, copy, hashlib, json, platform
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import torch.nn.functional as F
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.ml.tree_bank import DecisionTreeBank,TREE_CHANNELS
from crowd_reliability.ml.features import FeatureScaler,build_feature_row
from crowd_reliability.ml.context import context_matrix,ContextExpertRouter
from crowd_reliability.ml.graph import SpatialGraphBuilder
from crowd_reliability.ml.spatial import SpatialEncoder,build_spatial_features,fit_normalization
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES,MultivariateTemporalForecaster
from crowd_reliability.simulation.simulator import CrowdSimulator
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import ScenarioConfig,get_default_venue
from crowd_reliability.contracts import utc

SCALE=np.array([1.,1.,2.,2.,10.,1.],dtype=np.float32)
CANDIDATES=[dict(hidden=32,lr=.003,depth=4,clusters=2),dict(hidden=64,lr=.001,depth=8,clusters=3)]

def monte_carlo_scenarios(seed,counts,duration=240):
    venue=get_default_venue();splits={}
    for si,(name,count) in enumerate(zip(('train','validation','test'),counts)):
        rng=np.random.default_rng(seed+si*100000)
        items=[]
        for i in range(count):
            value=int(rng.integers(0,2**30));start=int(rng.integers(35,100))
            cfg=ScenarioConfig(scenario_id=f'{name}_{value}',title=name,description='Seeded Monte Carlo',
                seed=value,duration_seconds=duration,step_seconds=5,surge_start_seconds=start,
                surge_end_seconds=start+int(rng.integers(25,75)),surge_multiplier=float(rng.uniform(1,5)),
                exit_restriction_zone='exit_1',exit_restriction_start=int(rng.integers(65,110)),
                exit_restriction_end=int(rng.integers(130,200)),sensor_dropout_rate=float(rng.uniform(.05,.35)),
                stale_report_probability=.2,conflict_probability=float(rng.uniform(.05,.15)))
            refs,_=CrowdSimulator(venue,seed=value).run_simulation(cfg)
            obs=ObservationGenerator(seed=value).generate_observations(refs,cfg)
            times=sorted({utc(r.state_at) for r in refs});lookup={(r.zone_id,r.variable,utc(r.state_at)):r.value for r in refs}
            byzone={z['zone_id']:[o for o in obs if o.zone_id==z['zone_id']] for z in venue.zones}
            base=np.array([[build_feature_row(z['zone_id'],t,byzone[z['zone_id']],cfg.scenario_id) for z in venue.zones] for t in times])
            labels=np.array([[[lookup[(z['zone_id'],var,t)] for var in TARGET_VAR_NAMES] for z in venue.zones] for t in times],dtype=np.float32)
            items.append(dict(id=cfg.scenario_id,seed=value,config=cfg.model_dump(),obs=obs,times=times,base=base,labels=labels))
        splits[name]=items
    return venue,splits

def targets(labels):return {c:labels[:,i] for i,c in enumerate(TREE_CHANNELS)}

def fit_bank(items,graph,candidate,seed):
    base=np.concatenate([s['base'].reshape(-1,18) for s in items]);labels=np.concatenate([s['labels'].reshape(-1,6) for s in items])
    scaler=FeatureScaler().fit(base,fit_scenario_ids=[s['id'] for s in items])
    bank=DecisionTreeBank(candidate['depth'],seed);bank.fit(base,targets(labels),scaler)
    context=np.concatenate([context_matrix(frame,graph) for s in items for frame in s['base']])
    bank.version="dt_bank_context_v3"
    bank.context_model=ContextExpertRouter().fit(context,targets(labels),candidate['clusters'],seed,max_depth=candidate['depth'])
    return bank

def model_inputs(items,bank,graph):
    output=[]
    for s in items:
        f=[];v=[];u=[]
        # Use the same masked values/age/presence convention as live pipeline.
        helper=ArchitecturePipeline(venue=get_default_venue())
        byzone={z['zone_id']:[o for o in s['obs'] if o.zone_id==z['zone_id']] for z in graph.zones}
        for t,base in zip(s['times'],s['base']):
            values,masks=bank.predict_context(base,graph)
            details=[helper.extract_zone_features_detailed(z['zone_id'],t,byzone[z['zone_id']]) for z in graph.zones]
            feat,valid=build_spatial_features(np.stack([d['values'] for d in details]),np.stack([d['ages_sec'] for d in details]),np.stack([d['present'] for d in details]),np.stack([d['fresh'] for d in details]),np.asarray(values),np.asarray(masks))
            f.append(feat);v.append(valid);u.append(np.concatenate([base,np.asarray(values)],axis=1))
        output.append(dict(**{k:s[k] for k in ('id','base','labels')},features=np.array(f),valid=np.array(v),u=np.array(u,dtype=np.float32)))
    return output

def windows(items,lookback=12,horizons=(6,12,24)):
    f=[];v=[];u=[];y=[];baseline=[]
    for s in items:
        for end in range(len(s['features'])-max(horizons)):
            indices=np.clip(np.arange(end-lookback+1,end+1),0,end)
            f.append(s['features'][indices]);v.append(s['valid'][indices]);u.append(s['u'][indices]);y.append(np.stack([s['labels'][end+h] for h in horizons],axis=1));baseline.append(np.repeat(s['base'][end,:,:6,None],len(horizons),axis=2).transpose(0,2,1))
    if not f:raise ValueError('No complete forecast windows.')
    result={k:torch.tensor(np.array(values),dtype=torch.float32) for k,values in [('f',f),('v',v),('u',u),('y',y),('baseline',baseline)]}
    result['mask']=result['y']>=0
    return result

def forward(enc,temporal,batch):
    # Batch disjoint copies of the venue graph for every history frame.
    b,t,z,_=batch['f'].shape;n=b*t;graph=enc.graph
    edges=graph.edge_index.repeat(1,n)+torch.arange(n).repeat_interleave(graph.num_edges).unsqueeze(0)*z
    vn,ve=enc.hypernet(batch['f'].reshape(-1,36),graph.node_static.repeat(n,1),edges,graph.edge_attr.repeat(n,1),batch['v'].reshape(-1,36))
    h=enc.gnn(vn,edges,ve).reshape(b,t,z,16)
    u=torch.cat([batch['u'],h],dim=-1).permute(0,2,1,3).reshape(b*z,t,40)
    return temporal(u).reshape(b,z,3,6)

def loss(pred,batch):
    mask=batch['mask'];scale=torch.tensor(SCALE)
    numeric=((pred[...,:5]-batch['y'][...,:5])/scale[:5])**2
    regression=numeric[mask[...,:5]].mean()
    exits=mask[...,5]
    binary=F.binary_cross_entropy(pred[...,5][exits].clamp(1e-6,1-1e-6),batch['y'][...,5][exits]) if exits.any() else 0
    return regression+.2*binary

def metrics(pred,batch):
    result={}
    for h,horizon in enumerate((30,60,120)):
        for i,var in enumerate(TARGET_VAR_NAMES):
            m=batch['mask'][:,:,h,i];error=pred[:,:,h,i][m]-batch['y'][:,:,h,i][m]
            result[f'{var}/{horizon}s']={'mae':float(error.abs().mean()),'rmse':float((error**2).mean().sqrt()),'count':int(m.sum())}
    return result

def fit_neural(graph,data,candidate,seed,epochs):
    torch.manual_seed(seed);enc=SpatialEncoder(graph,seed=seed,hidden_dim=candidate['hidden'])
    norm=fit_normalization(data['train']['f'].numpy().reshape(-1,36),data['train']['v'].numpy().reshape(-1,36),f'mc_train_{seed}')
    enc.set_normalization(norm)
    temporal=MultivariateTemporalForecaster(40)
    # Positive numeric heads avoid dead ReLU regressors during initial training.
    for head in temporal.numeric_heads:head[-1]=torch.nn.Softplus()
    temporal.numeric_activation='softplus'
    optimizer=torch.optim.Adam(list(enc.hypernet.parameters())+list(enc.gnn.parameters())+list(temporal.parameters()),lr=candidate['lr'])
    rng=np.random.default_rng(seed);best=None;bestloss=float('inf');curve=[];bad=0
    for epoch in range(epochs):
        temporal.train();enc.hypernet.train();enc.gnn.train()
        order=rng.permutation(len(data['train']['f']))
        for start in range(0,len(order),8):
            ids=order[start:start+8];batch={k:v[ids] for k,v in data['train'].items()}
            optimizer.zero_grad();value=loss(forward(enc,temporal,batch),batch);value.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.hypernet.parameters())+list(enc.gnn.parameters())+list(temporal.parameters()),1.)
            optimizer.step()
        temporal.eval();enc.hypernet.eval();enc.gnn.eval()
        with torch.no_grad():value=float(loss(forward(enc,temporal,data['validation']),data['validation']))
        curve.append(value)
        if value<bestloss-1e-6:
            bestloss=value;bad=0;best=copy.deepcopy((enc.hypernet.state_dict(),enc.gnn.state_dict(),temporal.state_dict()))
        else:bad+=1
        if bad>=6:break
    enc.hypernet.load_state_dict(best[0]);enc.gnn.load_state_dict(best[1]);temporal.load_state_dict(best[2]);temporal.eval()
    return enc,temporal,bestloss,curve

def run(out,seed=42,runs=6,epochs=20):
    torch.set_num_threads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Use a new output directory; existing model versions are preserved.')
    venue,splits=monte_carlo_scenarios(seed,(runs,max(2,runs//3),max(2,runs//3)))
    graph=SpatialGraphBuilder(venue.zones,venue.connections);selected=None;trials=[]
    for ci,candidate in enumerate(CANDIDATES):
        print('Candidate',ci+1,candidate,flush=True)
        bank=fit_bank(splits['train'],graph,candidate,seed)
        # Test observations/labels are not touched during selection.
        inputs={name:model_inputs(splits[name],bank,graph) for name in ('train','validation')}
        data={name:windows(items) for name,items in inputs.items()}
        enc,temporal,value,curve=fit_neural(graph,data,candidate,seed,epochs)
        trial=dict(config=candidate,validation_loss=value,curve=curve);trials.append(trial)
        if selected is None or value<selected[0]:selected=(value,bank,enc,temporal,copy.deepcopy(candidate))
    _,bank,enc,temporal,config=selected
    test=windows(model_inputs(splits['test'],bank,graph))
    with torch.no_grad():pred=forward(enc,temporal,test)
    summary={'scope':'synthetic_only','selection_objective':'masked scale-normalized regression MSE + 0.2 exit BCE',
        'seed':seed,'selected':config,'trials':trials,'heldout_metrics':metrics(pred,test),
        'persistence_metrics':metrics(test['baseline'],test),'split_ids':{k:[s['id'] for s in v] for k,v in splits.items()},
        'split_seeds':{k:[s['seed'] for s in v] for k,v in splits.items()},
        'training_runs':runs,'epochs_max':epochs,'calibrated_real_world':False,
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}
    enc.stamp(training_status='synthetic_validated')
    pipeline=ArchitecturePipeline(venue=venue);pipeline.tree_bank=bank;pipeline.spatial=enc;pipeline.gnn=enc.gnn;pipeline.hypernet=enc.hypernet;pipeline.temporal_model=temporal
    pipeline.manifest.training_status={k:'synthetic_validated' for k in ('hypernet','gnn','temporal','tree_bank')}
    pipeline.manifest.metrics_summary=summary;pipeline.manifest.pipeline_version='v2.0.0-synthetic'
    pipeline.save_pipeline(str(out))
    (out/'selection_report.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (out/'training_venue.json').write_text(json.dumps(venue.model_dump(),indent=2),encoding='utf-8')
    print('Saved',out,'(synthetic evaluation only)',flush=True)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='models/candidates/mc_v2');p.add_argument('--seed',type=int,default=42);p.add_argument('--runs',type=int,default=6);p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
    if a.runs<2 or a.epochs<1:p.error('Use at least two training runs and one epoch.')
    run(a.out,a.seed,a.runs,a.epochs)
if __name__=='__main__':main()
