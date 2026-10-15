"""Retrospective incident scoring and observation-only intervention experiments."""
import copy
import math
from simulation import live_steps, path

def monitor(w,snapshot,result):
    t=w['time'];policy=w.get('policy',{'density':2.5,'queue':15})
    state=w.setdefault('monitor',dict(truth={},streak={},episodes=[],forecasts={},scored=set(),counts=dict(tp=0,fp=0,fn=0,tn=0)))
    for row in result['rows'][-len(w['scene']['routes']):]:
        zone=row['route_id'];risk=t>=3 and (row['mean_density_p_m2']>=policy['density'] or row['queued']>=policy['queue'])
        state['truth'][(zone,t)]=risk
        streak=state['streak'].get(zone,0)+1 if risk else 0;state['streak'][zone]=streak
        if streak==3:state['episodes'].append(dict(zone_id=zone,onset_s=t-2,confirmed_s=t))
    # Predictions are frozen at their issue time; truth never modifies their values.
    for row in snapshot['comparisons']:
        key=(row['zone_id'],row['issued_s'],row['target_s'])
        if row['variable'] not in ('density','queue_length') or row['architecture'] is None:continue
        record=state['forecasts'].setdefault(key,{'values':{}})
        record['values'][row['variable']]=row['architecture']
    for key,record in state['forecasts'].items():
        zone,issued,target=key
        if key in state['scored'] or target+2>t:continue
        if any((zone,k) not in state['truth'] for k in range(target,target+3)):continue
        pred=record['values'].get('density',0)>=policy['density'] or record['values'].get('queue_length',0)>=policy['queue']
        actual=all(state['truth'][(zone,k)] for k in range(target,target+3))
        state['counts']['tp' if pred and actual else 'fp' if pred else 'fn' if actual else 'tn']+=1;state['scored'].add(key)
    episodes=[]
    for e in state['episodes']:
        matching=[issued for (zone,issued,target),f in state['forecasts'].items() if zone==e['zone_id'] and issued<e['onset_s']<=target<=e['onset_s']+30 and (f['values'].get('density',0)>=policy['density'] or f['values'].get('queue_length',0)>=policy['queue'])]
        warning=min(matching) if matching else None
        episodes.append(dict(e,warning_s=warning,lead_time_s=e['onset_s']-warning if warning is not None else None,status='warned_before_onset' if warning is not None else 'no_advance_warning'))
    n=state['counts'];precision=n['tp']/(n['tp']+n['fp']) if n['tp']+n['fp'] else None;recall=n['tp']/(n['tp']+n['fn']) if n['tp']+n['fn'] else None
    return dict(policy=policy,sustained_seconds=3,episodes=episodes,counts=n.copy(),precision=precision,recall=recall,
        matured_forecasts=len(state['scored']),notice='Demo incidents are density or queue threshold breaches sustained for 3s. Forecast targets are scored only after their 3s outcome window. No-warning and false alarms remain visible.')

def plan_from_observations(w,snapshot,horizon=90):
    # Only the runtime evidence view and public graph geometry seed these trials.
    clock=snapshot['time_s'];routes=copy.deepcopy(w['scene']['routes']);scale=w['scene']['scale'];assumptions=[];events=[]
    for r in routes:
        evidence={e['variable']:e for e in snapshot['evidence'] if e['zone_id']==r['id']}
        density=evidence.get('density');speed=evidence.get('speed')
        if any(not e or e['observed_value'] is None or e['observation_age_s']>30 for e in (density,speed)):
            raise ValueError('Wait for recent density and speed reports on every route before comparing actions.')
        r['count']=max(0,round(density['observed_value']*path(r,scale)['length']*r['width']))
        r.update(initial_fill=True,lanes=r.get('lanes',max(1,int(r['width']/.6))),speed=1.3)
        inflow=evidence.get('inflow',{});r['arrival_rate']=min(1,max(0,float(inflow.get('observed_value') or 0)))
        queue=evidence.get('queue_length',{}).get('observed_value') or 0
        gate=evidence.get('exit_open',{}).get('observed_value')
        if gate==0 or queue>=w.get('policy',{}).get('queue',15) and speed['observed_value']<.3:
            events.append(dict(id='estimated_'+r['id'],name='Suspected bottleneck',type='collapse',routeId=r['id'],at=.97 if gate==0 else .7,start=0,end=horizon,people=0,radiusM=2.5))
            assumptions.append('A bottleneck on '+r['name']+' is inferred from received reports; its location and removability require operator confirmation.')
    if sum(r['count']+int(r['arrival_rate']*horizon) for r in routes)>2600 or any(r['count']>1500 for r in routes):raise ValueError('Observed population is above the bounded planning budget; reduce density for this demo.')
    base=dict(duration=horizon,scale=scale,seed=12345,randomCount=1,routes=routes,events=events)
    choices=[('continue','Continue current operation',False,False),('meter','Meter new arrivals',False,True)]
    if events:choices += [('clear','Inspect and clear suspected bottleneck',True,False),('clear_meter','Clear bottleneck + meter arrivals',True,True)]
    trials=[]
    for key,title,clear,meter in choices:
        config=copy.deepcopy(base)
        if clear:config['events']=[]
        if meter:config['meter_from']=0
        exposure=0;peak=0;clearance=None;last=None
        for last in live_steps(config,'custom'):
            rows=last['rows'][-len(routes):];t=rows[0]['time_s']
            exposure+=sum(r['queued']+max(0,r['mean_density_p_m2']-w.get('policy',{}).get('density',2.5))*r['active'] for r in rows)
            peak=max(peak,max(r['peak_local_density_p_m2'] for r in rows))
            if clearance is None and last['total']==last['completed']:clearance=t
        remaining=last['total']-last['completed']
        trials.append(dict(action=key,label=title,exposure_index=round(exposure,2),peak_local_density=round(peak,3),completed=last['completed'],remaining=remaining,waiting_outside=last['waiting'],clearance_s=clearance,clearance_censored=clearance is None,requires_inspection=clear))
    trials.sort(key=lambda x:(x['exposure_index'],x['remaining'],x['clearance_s'] if x['clearance_s'] is not None else horizon+1))
    return dict(as_of_s=clock,affected_routes=[e['routeId'] for e in events],horizon_s=horizon,recommended=trials[0]['action'],trials=trials,assumptions=assumptions+['Uniform positions reconstructed from latest observed route density; free walking speed assumed 1.3 m/s.','Current received inflow is held constant. Unknown future incidents are not available to these trials.','Ranking minimizes queue/density exposure, then remaining people. Outside waiting is reported separately; not all actions clear the whole venue.'],notice='Best among tested synthetic actions, not a guaranteed safe evacuation plan. Inferred obstruction clearing assumes inspection confirms it is feasible.')
