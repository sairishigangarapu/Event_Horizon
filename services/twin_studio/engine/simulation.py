"""Standard-library crowd engine, parity with engine.js.

Phase 2 (Twin Studio integration) additions to the original archive engine
(original kept byte-identical in engine/original/simulation.py):
  - optional scene field `gates`: [{id, x, y, w, d}] gate rectangles in canvas units;
  - incident type `gate_closure` {id, name, type, gateId, start, end}: the gate is closed during
    every engine step t with start <= t < end (step t covers [t, t+1) s), i.e. for the
    simulated interval [start, end);
  - a closed gate cannot be entered: an agent ahead of a gate span stops GATE_STOP_MARGIN_M
    before the span along its route (the check uses arc length, so no movement step can skip
    over a narrow gate), and an agent already inside a span holds position while it is closed;
  - completion classification: a route completion whose end position lies inside a gate
    rectangle (half-open, gates in scene order) is a gate exit; any other completion is
    `route_completed_not_exit`;
  - result logs: `spawns`, `completions`, per-step `gateOpen`; rows gain `gate_exits` and
    `route_completed_not_exit`. Existing fields and behaviour without gates are unchanged.
"""
import math

MASK = 0xffffffff
GATE_STOP_MARGIN_M = 0.5
class RNG:
    def __init__(self, seed): self.a = int(seed) & MASK
    def next(self):
        self.a = (self.a + 0x6D2B79F5) & MASK
        t = ((self.a ^ (self.a >> 15)) * (self.a | 1)) & MASK
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK
        t &= MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296

def path(route, scale):
    cum = [0.0]
    for a,b in zip(route['points'],route['points'][1:]):
        cum.append(cum[-1] + math.hypot(a['x']-b['x'],a['y']-b['y'])*scale)
    return {'cum':cum,'length':cum[-1]}

def position(route,p,s):
    i=1
    while i<len(p['cum'])-1 and s>p['cum'][i]: i+=1
    f=min(1,max(0,(s-p['cum'][i-1])/(p['cum'][i]-p['cum'][i-1] or 1)))
    a,b=route['points'][i-1:i+1]
    return {'x':a['x']+(b['x']-a['x'])*f,'y':a['y']+(b['y']-a['y'])*f}

def inside_gate(g,pos): return g['x']<=pos['x']<g['x']+g['w'] and g['y']<=pos['y']<g['y']+g['d']

def gate_spans(route,p,g):
    """Arc-length intervals [s_in, s_out] (metres) where the route polyline lies in the closed gate rectangle."""
    spans=[]
    pts=route['points']
    for i in range(1,len(pts)):
        a,b=pts[i-1],pts[i]; dx=b['x']-a['x']; dy=b['y']-a['y']
        seg=p['cum'][i]-p['cum'][i-1]
        if seg<=0: continue
        t0=0.0; t1=1.0; ok=True
        for q,r in ((-dx,a['x']-g['x']),(dx,g['x']+g['w']-a['x']),(-dy,a['y']-g['y']),(dy,g['y']+g['d']-a['y'])):
            if q==0:
                if r<0: ok=False; break
            else:
                u=r/q
                if q<0:
                    if u>t0: t0=u
                else:
                    if u<t1: t1=u
        if not ok or t0>t1: continue
        s0=p['cum'][i-1]+t0*seg; s1=p['cum'][i-1]+t1*seg
        if spans and s0<=spans[-1][1]: spans[-1][1]=max(spans[-1][1],s1)
        else: spans.append([s0,s1])
    return spans

def integer(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v==int(v)
def number(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def validate(c):
    if not isinstance(c,dict): raise ValueError('Scene must be a JSON object.')
    if not integer(c.get('duration')) or not 10<=c['duration']<=900: raise ValueError('Duration: 10–900 whole seconds.')
    if not number(c.get('scale')) or not 0<c['scale']<=2: raise ValueError('Scale: greater than 0 and at most 2 m/unit.')
    if not integer(c.get('seed')) or not 0<=c['seed']<=MASK: raise ValueError('Seed: 0–4294967295.')
    if not integer(c.get('randomCount')) or not 1<=c['randomCount']<=10: raise ValueError('Random collapses: 1–10.')
    routes=c.get('routes'); events=c.get('events')
    if not isinstance(routes,list) or not 1<=len(routes)<=12: raise ValueError('Draw or import 1–12 routes.')
    if not isinstance(events,list) or len(events)>100: raise ValueError('Maximum 100 incidents.')
    gates=c.get('gates',[])
    if not isinstance(gates,list) or len(gates)>20: raise ValueError('Gates must be a list of at most 20 rectangles.')
    gids=set()
    for g in gates:
        if not isinstance(g,dict) or not isinstance(g.get('id'),str) or not g['id'] or g['id'] in gids: raise ValueError('Gate IDs must be unique non-empty strings.')
        gids.add(g['id'])
        if not all(number(g.get(k)) for k in ('x','y','w','d')) or g['w']<=0 or g['d']<=0 or g['x']<0 or g['y']<0 or g['x']+g['w']>1000 or g['y']+g['d']>700: raise ValueError('Gate rectangles must be inside the map with positive size.')
    ids=set(); total=0
    for r in routes:
        if not isinstance(r,dict) or not isinstance(r.get('id'),str) or not r['id'] or r['id'] in ids: raise ValueError('Route IDs must be unique non-empty strings.')
        ids.add(r['id'])
        if not isinstance(r.get('name'),str) or not r['name'].strip(): raise ValueError('Name every route.')
        pts=r.get('points')
        if not isinstance(pts,list) or not 2<=len(pts)<=500: raise ValueError('Route needs 2–500 points.')
        for p in pts:
            if not isinstance(p,dict) or not number(p.get('x')) or not number(p.get('y')) or not 0<=p['x']<=1000 or not 0<=p['y']<=700: raise ValueError('Route points must be inside the map.')
        if path(r,c['scale'])['length']<1: raise ValueError('Route must be at least 1 metre long.')
        if not integer(r.get('count')) or not 0<=r['count']<=1500: raise ValueError('Route population: 0–1500.')
        if not number(r.get('speed')) or not .2<=r['speed']<=3: raise ValueError('Speed: 0.2–3 m/s.')
        if not number(r.get('width')) or not .5<=r['width']<=20: raise ValueError('Width: 0.5–20 m.')
        total+=r['count']
    eids=set()
    for e in events:
        if not isinstance(e,dict) or not isinstance(e.get('id'),str) or not e['id'] or e['id'] in eids: raise ValueError('Incident IDs must be unique.')
        eids.add(e['id'])
        if not isinstance(e.get('name'),str) or not e['name'].strip(): raise ValueError('Name every incident.')
        if e.get('type') not in ['surge','collapse','gate_closure']: raise ValueError('Unknown incident type.')
        if e['type']=='gate_closure':
            if e.get('gateId') not in gids: raise ValueError('Gate closure must name a configured gate.')
            if e.get('routeId') is not None or e.get('people',0)!=0 or 'at' in e: raise ValueError('Gate closure takes gateId, start and end only.')
            if not integer(e.get('start')) or not integer(e.get('end')) or not 0<=e['start']<e['end']<=c['duration']: raise ValueError(f"{e['name']}: require 0 ≤ start < end ≤ {c['duration']}s.")
            continue
        if e.get('routeId') not in ids: raise ValueError('Place every incident on a route.')
        if not number(e.get('at')) or not 0<=e['at']<=1: raise ValueError('Invalid incident location.')
        if not integer(e.get('start')) or not integer(e.get('end')) or not 0<=e['start']<e['end']<=c['duration']: raise ValueError(f"{e['name']}: require 0 ≤ start < end ≤ {c['duration']}s.")
        if e['type']=='surge':
            if not integer(e.get('people')) or not 1<=e['people']<=1500: raise ValueError('Surge: 1–1500 additional people.')
            total+=e['people']
    if total>3000: raise ValueError('Maximum 3000 total people per run.')
    closed={e['gateId'] for e in events if e['type']=='gate_closure'}
    for g in gates:
        if g['id'] not in closed: continue
        for r in routes:
            p=path(r,c['scale'])
            starts=[0.0]+[e['at']*p['length'] for e in events if e['type']=='surge' and e['routeId']==r['id']]
            for s0,s1 in gate_spans(r,p,g):
                if any(s0<=s<=s1 for s in starts): raise ValueError(f"Route '{r['name']}' spawns inside gate {g['id']}, which has a closure.")

def random_events(c):
    r=RNG(c['seed']+9182); events=[]
    for i in range(c['randomCount']):
        start=math.floor(c['duration']*(.15+r.next()*.55))
        ri=math.floor(r.next()*len(c['routes']))
        at=.15+r.next()*.7
        end=min(c['duration'],start+max(3,math.floor(c['duration']*(.08+r.next()*.15))))
        events.append({'id':'auto'+str(i),'name':'Random collapse '+str(i+1),'type':'collapse','routeId':c['routes'][ri]['id'],'at':at,'start':start,'end':end,'people':0})
    return events

def simulate(c,mode):
    validate(c)
    if mode not in ['ideal','custom','random']: raise ValueError('Unknown run mode.')
    r=RNG(c['seed']); routes=c['routes']; paths=[path(x,c['scale']) for x in routes]
    events=[] if mode=='ideal' else c['events'] if mode=='custom' else random_events(c)
    gates=c.get('gates',[]); closures=[e for e in events if e['type']=='gate_closure']
    spans=[[(g['id'],s0,s1) for g in gates for s0,s1 in gate_spans(x,paths[ri],g)] for ri,x in enumerate(routes)]
    agents=[]; frames=[]; rows=[]; schedules=[]; uid=0
    spawn_log=[]; completion_log=[]; gate_open=[]
    for ri,route in enumerate(routes):
        for i in range(int(route['count'])):
            uid+=1
            schedules.append({'time':math.floor(i/max(1,route['count'])*c['duration']*.65),'ri':ri,'at':0,'speed':route['speed']*(.85+r.next()*.3),'id':uid})
    for e in events:
        if e['type']!='surge': continue
        ri=next(i for i,x in enumerate(routes) if x['id']==e['routeId'])
        for i in range(int(e['people'])):
            uid+=1
            schedules.append({'time':int(e['start'])+math.floor(i/e['people']*(e['end']-e['start'])),'ri':ri,'at':e['at']*paths[ri]['length'],'speed':routes[ri]['speed']*(.85+r.next()*.3),'id':uid})
    schedules.sort(key=lambda a:(a['time'],a['id'])); cursor=0; completed=[0]*len(routes)
    gate_exits=[0]*len(routes); not_exit=[0]*len(routes)
    for t in range(int(c['duration'])+1):
        incoming=[0]*len(routes); outgoing=[0]*len(routes); step_gate=[0]*len(routes); step_other=[0]*len(routes)
        closed={e['gateId'] for e in closures if e['start']<=t<e['end']}
        gate_open.append({g['id']:g['id'] not in closed for g in gates})
        while cursor<len(schedules) and schedules[cursor]['time']==t:
            s=schedules[cursor]; cursor+=1
            agents.append({**s,'s':s['at'],'v':0,'done':False}); incoming[s['ri']]+=1
            spawn_log.append({'t':t,'id':s['id'],'ri':s['ri'],**position(routes[s['ri']],paths[s['ri']],s['at'])})
        active=[a for a in agents if not a['done']]; cells={}; groups=[[] for _ in routes]
        for a in active:
            a['pos']=position(routes[a['ri']],paths[a['ri']],a['s'])
            a['cell']=(math.floor(a['pos']['x']*c['scale']/2),math.floor(a['pos']['y']*c['scale']/2))
            cells[a['cell']]=cells.get(a['cell'],0)+1
            groups[a['ri']].append(a)
        for ri,route in enumerate(routes):
            p=paths[ri]; group=sorted(groups[ri],key=lambda a:(-a['s'],a['id'])); lead=float('inf')
            blocks=[e for e in events if e['type']=='collapse' and e['routeId']==route['id'] and e['start']<=t<e['end']]
            for a in group:
                v=a['speed']/(1+max(0,cells[a['cell']]/4-.5)*1.8)
                for b in blocks:
                    loc=b['at']*p['length']
                    if a['s']<=loc+1 and a['s']+v>=loc-1: v=max(0,loc-1-a['s'])
                for gid,s0,s1 in spans[ri]:
                    if gid not in closed: continue
                    if s0<=a['s']<=s1: v=0
                    elif a['s']<s0: v=min(v,max(0,s0-GATE_STOP_MARGIN_M-a['s']))
                v=min(v,max(0,lead-a['s']-.65)); a['v']=v; a['s']+=v; lead=a['s']
                if a['s']>=p['length']:
                    a['done']=True;outgoing[ri]+=1;completed[ri]+=1
                    end=position(route,p,a['s'])
                    gate=next((g['id'] for g in gates if inside_gate(g,end)),None)
                    if gate is None: not_exit[ri]+=1; step_other[ri]+=1
                    else: gate_exits[ri]+=1; step_gate[ri]+=1
                    completion_log.append({'t':t,'id':a['id'],'ri':ri,'x':end['x'],'y':end['y'],'gateId':gate})
            remaining=[a for a in group if not a['done']]
            rows.append({'run':mode,'seed':c['seed'],'time_s':t,'route_id':route['id'],'route_name':route['name'],'spawned':incoming[ri],'exited':outgoing[ri],'active':len(remaining),'completed':completed[ri],'mean_speed_m_s':sum(a['v'] for a in remaining)/len(remaining) if remaining else 0,'mean_density_p_m2':len(remaining)/(p['length']*route['width']),'peak_local_density_p_m2':max((cells[a['cell']]/4 for a in remaining),default=0),'queued':sum(a['v']<.1 for a in remaining),'net_flow':incoming[ri]-outgoing[ri],'gate_exits':step_gate[ri],'route_completed_not_exit':step_other[ri]})
        frames.append([{'id':a['id'],'ri':a['ri'],**position(routes[a['ri']],paths[a['ri']],a['s']),'v':a['v']} for a in active if not a['done']])
    return {'mode':mode,'events':events,'frames':frames,'rows':rows,'total':uid,'completed':sum(completed),'peakQueue':max(x['queued'] for x in rows),
            'gateExits':sum(gate_exits),'routeCompletedNotExit':sum(not_exit),'spawns':spawn_log,'completions':completion_log,'gateOpen':gate_open}

def normalize_routes(payload,scale=.1):
    if not isinstance(payload,dict) or not isinstance(payload.get('routes'),list): raise ValueError('Route JSON must contain a routes array.')
    legacy=payload.get('schema_version')=='path-editor-v1' or payload.get('coordinate_system')=='assumed_local_metres'
    colors=['#ff633e','#477fe5','#6d9c4b','#ac68bc']
    routes=[]
    for i,r in enumerate(payload['routes']):
        pts=[]
        for p in r.get('points',[]):
            x,y=(p['x'],p['y']) if isinstance(p,dict) else p
            pts.append({'x':round(float(x)/scale,8) if legacy else float(x),'y':round(float(y)/scale,8) if legacy else float(y)})
        routes.append({'id':str(r.get('id') or r.get('route_id') or 'import_'+str(i)),'name':r.get('name') or 'Imported route '+str(i+1),'color':r.get('color') or colors[i%4],'points':pts,'count':r.get('count',100),'speed':r.get('speed',1.3),'width':r.get('width',3)})
    return routes
