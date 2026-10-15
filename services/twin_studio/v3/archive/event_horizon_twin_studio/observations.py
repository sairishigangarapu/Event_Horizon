"""Causal synthetic reporting. Never pass simulator results to inference clients."""
import csv
import json
import math
import shutil
from pathlib import Path
from simulation import RNG, number, integer

SOURCES = (
    {'id':'camera_1','kind':'camera','interval_s':2,'delay_s':2,'missing_probability':.1,'relative_noise':.08},
    {'id':'sensor_1','kind':'sensor','interval_s':1,'delay_s':1,'missing_probability':.03,'relative_noise':.03},
    {'id':'warden_1','kind':'warden','interval_s':10,'delay_s':5,'missing_probability':.15,'relative_noise':.15},
)

def sources_for(config):
    sources=config.get('observationSources',list(SOURCES))
    if not isinstance(sources,list) or not 1<=len(sources)<=30:raise ValueError('Use 1–30 observation sources.')
    ids=set();routes={r['id'] for r in config['routes']}
    for s in sources:
        if not isinstance(s,dict) or not isinstance(s.get('id'),str) or not s['id'] or s['id'] in ids:raise ValueError('Observation source IDs must be unique.')
        ids.add(s['id'])
        if s.get('kind') not in ('camera','sensor','warden'):raise ValueError('Unknown observation source kind.')
        for key,low,high in [('interval_s',1,900),('delay_s',0,3600)]:
            if not integer(s.get(key)) or not low<=s[key]<=high:raise ValueError('Invalid source '+key)
        for key in ('missing_probability','relative_noise'):
            if not number(s.get(key)) or not 0<=s[key]<=1:raise ValueError('Invalid source '+key)
        covered=s.get('route_ids',list(routes))
        if not isinstance(covered,list) or any(r not in routes for r in covered):raise ValueError('Invalid source route coverage.')
    return sources

def generate_observations(config,result):
    """Each measurement uses only its own time's row, never later rows/events."""
    out=[]
    for index,source in enumerate(sources_for(config)):
        rng=RNG(config['seed']+11731+index*7919)
        coverage=source.get('route_ids',[r['id'] for r in config['routes']])
        def noisy(value,count=False):
            estimate=max(0,value*(1+(rng.next()*2-1)*source['relative_noise']))
            return int(math.floor(estimate+.5)) if count else round(estimate,5)
        for row in result['rows']:
            t=row['time_s']
            if row['route_id'] not in coverage or t%source['interval_s']:continue
            if rng.next()<source['missing_probability']:continue
            if source['kind']=='camera':
                values={'visible_people_estimate':noisy(row['active'],True),'mean_speed_estimate_m_s':noisy(row['mean_speed_m_s']),'density_estimate_p_m2':noisy(row['mean_density_p_m2'])}
            elif source['kind']=='sensor':
                values={'entries_count':noisy(row['spawned'],True),'exits_count':noisy(row['exited'],True),'count_window_s':1}
            else:
                values={'queue_estimate':noisy(row['queued'],True),'movement_report':'slow' if row['active'] and row['mean_speed_m_s']<.2 else 'moving' if row['active'] else 'no_people_seen'}
            out.append({'observation_id':f"{source['id']}:{row['route_id']}:{t}",'source_id':source['id'],'source_kind':source['kind'],'route_id':row['route_id'],'observed_at_s':t,'available_at_s':t+source['delay_s'],'values':values})
    return sorted(out,key=lambda x:(x['available_at_s'],x['observation_id']))

def available_at(records,time_s):
    if not number(time_s) or time_s<0:raise ValueError('time_s must be a finite nonnegative number.')
    return [r for r in records if r['observed_at_s']<=time_s and r['available_at_s']<=time_s]

def write_jsonl(path,records):
    with path.open('w',encoding='utf-8') as f:
        for record in records:f.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')

def export_separated(directory,config,results):
    observable=directory/'observable';truth=directory/'hidden_ground_truth'
    observable.mkdir();truth.mkdir()
    (truth/'scenario.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for mode,result in results.items():
        write_jsonl(observable/(mode+'.jsonl'),generate_observations(config,result))
        folder=truth/mode;folder.mkdir()
        write_jsonl(folder/'frames.jsonl',({'time_s':t,'agents':frame,'waiting':result['waitingFrames'][t]} for t,frame in enumerate(result['frames'])))
        write_jsonl(folder/'incident_labels.jsonl',result['events'])
        with (folder/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
        (folder/'audit.json').write_text(json.dumps(result['audit']),encoding='utf-8')
    for folder in (observable,truth):
        shutil.make_archive(str(folder),'zip',root_dir=folder)
    (directory/'DATA_SEPARATION.txt').write_text('observable/: synthetic reports with observed_at_s and available_at_s. Only the cutoff API is for inference.\nhidden_ground_truth/: offline training/evaluation only. Studio CSVs, reports, manifests, scene files and render responses also contain truth. Never mount these into the inference process.\n',encoding='utf-8')

def read_observations(root,run_id,mode,time_s):
    # Caller validates run_id and mode. This reader has no truth-file access path.
    file=Path(root)/run_id/'observable'/(mode+'.jsonl')
    with file.open(encoding='utf-8') as f:return available_at([json.loads(line) for line in f],time_s)
