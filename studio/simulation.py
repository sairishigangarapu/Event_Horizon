"""Standard-library crowd engine, parity with dist/engine.js."""
import math

MASK = 0xffffffff
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
        if not integer(r.get('lanes',1)) or not 1<=r.get('lanes',1)<=20:raise ValueError('Lane count: 1–20.')
        if not number(r.get('arrival_rate',0)) or not 0<=r.get('arrival_rate',0)<=10:raise ValueError('Arrival rate: 0–10 people/s.')
        total+=r['count']+int(r.get('arrival_rate',0)*c['duration'])
    eids=set()
    for e in events:
        if not isinstance(e,dict) or not isinstance(e.get('id'),str) or not e['id'] or e['id'] in eids: raise ValueError('Incident IDs must be unique.')
        eids.add(e['id'])
        if not isinstance(e.get('name'),str) or not e['name'].strip(): raise ValueError('Name every incident.')
        if e.get('type') not in ['surge','collapse']: raise ValueError('Unknown incident type.')
        if e.get('routeId') not in ids: raise ValueError('Place every incident on a route.')
        if not number(e.get('at')) or not 0<=e['at']<=1: raise ValueError('Invalid incident location.')
        if not integer(e.get('start')) or not integer(e.get('end')) or not 0<=e['start']<e['end']<=c['duration']: raise ValueError(f"{e['name']}: require 0 ≤ start < end ≤ {c['duration']}s.")
        if e['type']=='collapse' and (not number(e.get('radiusM',3)) or not .25<=e.get('radiusM',3)<=20):raise ValueError('Collapse radius: 0.25–20 m.')
        if e['type']=='surge':
            if not integer(e.get('people')) or not 1<=e['people']<=1500: raise ValueError('Surge: 1–1500 additional people.')
            total+=e['people']
    if total>3000: raise ValueError('Maximum 3000 total people per run.')

def random_events(c):
    r=RNG(c['seed']+9182); events=[]
    for i in range(c['randomCount']):
        start=math.floor(c['duration']*(.15+r.next()*.55))
        ri=math.floor(r.next()*len(c['routes']))
        at=.15+r.next()*.7
        end=min(c['duration'],start+max(3,math.floor(c['duration']*(.08+r.next()*.15))))
        events.append({'id':'auto'+str(i),'name':'Random collapse '+str(i+1),'type':'collapse','routeId':c['routes'][ri]['id'],'at':at,'start':start,'end':end,'people':0})
    return events

EPS=1e-6

def intervals(route,p,center,radius,scale):
    hits=[]
    for i in range(1,len(route['points'])):
        a,b=route['points'][i-1:i+1]
        ax=a['x']*scale-center['x']*scale; ay=a['y']*scale-center['y']*scale
        dx=(b['x']-a['x'])*scale;dy=(b['y']-a['y'])*scale
        A=dx*dx+dy*dy
        if A<1e-12:continue
        B=2*(ax*dx+ay*dy);C=ax*ax+ay*ay-radius*radius;D=B*B-4*A*C
        if D<0:continue
        lo=max(0,(-B-math.sqrt(D))/(2*A));hi=min(1,(-B+math.sqrt(D))/(2*A))
        if lo<=hi:
            length=p['cum'][i]-p['cum'][i-1]
            hits.append([p['cum'][i-1]+lo*length,p['cum'][i-1]+hi*length])
    hits.sort(key=lambda h:h[0]);merged=[]
    for hit in hits:
        if merged and hit[0]<=merged[-1][1]+EPS:merged[-1][1]=max(merged[-1][1],hit[1])
        else:merged.append(hit)
    return merged

def barriers(c,events,paths):
    out=[]
    for e in events:
        if e['type']!='collapse':continue
        ri=next(i for i,r in enumerate(c['routes']) if r['id']==e['routeId'])
        center=position(c['routes'][ri],paths[ri],e['at']*paths[ri]['length']);radius=e.get('radiusM',3)
        affected=[]
        for i,r in enumerate(c['routes']):
            spans=intervals(r,paths[i],center,radius,c['scale'])
            if spans:affected.append({'ri':i,'intervals':spans})
        out.append({'eventId':e['id'],'name':e['name'],'start':e['start'],'end':e['end'],'center':center,'radiusM':radius,'routes':affected})
    return out

def grid(agents,scale):
    cells={}
    for a in agents:
        x=min(999.999999,max(0,a['pos']['x']));y=min(699.999999,max(0,a['pos']['y']))
        ix=math.floor(x*scale/2);iy=math.floor(y*scale/2);key=(ix,iy);a['cell']=key
        area=min(2,1000*scale-ix*2)*min(2,700*scale-iy*2)
        if key not in cells:cells[key]={'count':0,'area':area}
        cells[key]['count']+=1
    return cells

def live_steps(c,mode):
    validate(c)
    if mode not in ['ideal','custom','random']:raise ValueError('Unknown run mode.')
    random=RNG(c['seed']);routes=c['routes'];paths=[path(x,c['scale']) for x in routes]
    events=[] if mode=='ideal' else c['events'] if mode=='custom' else random_events(c)
    closures=barriers(c,events,paths);agents=[];frames=[];waiting_frames=[];rows=[];schedules=[];uid=0
    for ri,r in enumerate(routes):
        for i in range(int(r['count'])):
            uid+=1;schedules.append({'time':0 if r.get('initial_fill') else math.floor(i/max(1,r['count'])*c['duration']*.65),'ri':ri,'at':((i//r.get('lanes',1)+.5)/max(1,math.ceil(r['count']/r.get('lanes',1)))*paths[ri]['length']*.85) if r.get('initial_fill') else 0,'speed':r['speed']*(.85+random.next()*.3),'id':uid,'lane':i%r.get('lanes',1)})
        for i in range(int(r.get('arrival_rate',0)*c['duration'])):
            uid+=1;schedules.append({'time':1+math.floor(i/max(.001,r['arrival_rate'])),'ri':ri,'at':0,'speed':r['speed']*(.85+random.next()*.3),'id':uid,'lane':i%r.get('lanes',1),'arrival':True})
    for e in events:
        if e['type']!='surge':continue
        ri=next(i for i,r in enumerate(routes) if r['id']==e['routeId'])
        for i in range(int(e['people'])):
            uid+=1;schedules.append({'time':e['start']+math.floor(i/e['people']*(e['end']-e['start'])),'ri':ri,'at':e['at']*paths[ri]['length'],'speed':routes[ri]['speed']*(.85+random.next()*.3),'id':uid})
    schedules.sort(key=lambda s:(s['time'],s['id']));cursor=0;pending=[];outgoing=[0]*len(routes);blocked=[0]*len(routes);completed=[0]*len(routes);prevented=0;violations=[]
    scheduled_events={e['id'] for e in events}
    for t in range(int(c['duration'])+1):
        # Live operator actions are appended with future start times only.
        for e in events:
            if e['id'] in scheduled_events:continue
            scheduled_events.add(e['id'])
            if e['type']!='surge':continue
            ri=next(i for i,r in enumerate(routes) if r['id']==e['routeId'])
            for i in range(int(e['people'])):
                uid+=1;schedules.append({'time':e['start']+math.floor(i/e['people']*(e['end']-e['start'])),'ri':ri,'at':e['at']*paths[ri]['length'],'speed':routes[ri]['speed']*(.85+random.next()*.3),'id':uid})
        schedules.sort(key=lambda s:(s['time'],s['id']))
        closures=barriers(c,events,paths)
        active_blocks=[[] for _ in routes]
        for b in closures:
            if not b['start']<=t<b['end']:continue
            for rr in b['routes']:
                for lo,hi in rr['intervals']:active_blocks[rr['ri']].append({'lo':lo,'hi':hi,'eventId':b['eventId']})
        incoming=[0]*len(routes);scheduled=[0]*len(routes)
        while cursor<len(schedules) and schedules[cursor]['time']==t:
            s=schedules[cursor];cursor+=1;pending.append(s);scheduled[s['ri']]+=1
        keep=[]
        for s in pending:
            if s.get('arrival') and t>=c.get('meter_from',float('inf')):keep.append(s);continue
            if any(b['lo']-EPS<=s['at']<=b['hi']+EPS for b in active_blocks[s['ri']]):keep.append(s);continue
            agents.append({**s,'s':s['at'],'v':0,'done':False,'admittedAt':t});incoming[s['ri']]+=1
        pending=keep;active=[a for a in agents if not a['done']]
        for a in active:
            route=routes[a['ri']];p=paths[a['ri']];a['pos']=position(route,p,a['s'])
            lanes=route.get('lanes',1)
            if lanes>1:
                ahead=position(route,p,min(p['length'],a['s']+.05));dx=ahead['x']-a['pos']['x'];dy=ahead['y']-a['pos']['y'];norm=math.hypot(dx,dy) or 1
                offset=((a.get('lane',a['id']%lanes)+.5)/lanes-.5)*route['width']/c['scale']*.8
                a['pos']={'x':min(999,max(1,a['pos']['x']-dy/norm*offset)),'y':min(699,max(1,a['pos']['y']+dx/norm*offset))}
        cells=grid(active,c['scale'])
        for ri,route in enumerate(routes):
            p=paths[ri];remaining=[a for a in active if a['ri']==ri]
            rows.append({'run':mode,'seed':c['seed'],'time_s':t,'route_id':route['id'],'route_name':route['name'],'scheduled':scheduled[ri],'spawned':incoming[ri],'exited':outgoing[ri],'active':len(remaining),'waiting_entry':sum(a['ri']==ri for a in pending),'completed':completed[ri],'mean_speed_m_s':sum(a['v'] for a in remaining)/len(remaining) if remaining else 0,'mean_density_p_m2':len(remaining)/(p['length']*route['width']),'peak_local_density_p_m2':max((cells[a['cell']]['count']/cells[a['cell']]['area'] for a in remaining),default=0),'queued':sum(a['v']<.1 and a['admittedAt']<t for a in remaining),'blocked':blocked[ri],'net_flow':incoming[ri]-outgoing[ri]})
        frames.append([{'id':a['id'],'ri':a['ri'],'x':a['pos']['x'],'y':a['pos']['y'],'v':a['v'],'s_m':a['s']} for a in active])
        wg={}
        for a in pending:
            key=(a['ri'],a['at'])
            if key in wg:wg[key]['count']+=1
            else:wg[key]={'ri':a['ri'],'at':a['at'],'count':1}
        waiting_frames.append(list(wg.values()))
        yield {'mode':mode,'events':events,'barriers':closures,'frames':frames,'waitingFrames':waiting_frames,'rows':rows,'total':uid,'completed':sum(completed),'waiting':len(pending),'active':len(active),'peakQueue':max(r['queued'] for r in rows),'audit':{'preventedMoves':prevented,'violations':violations},'modelVersion':'kinematic-v3'}
        if t==c['duration']:break
        outgoing=[0]*len(routes);blocked=[0]*len(routes)
        for ri,route in enumerate(routes):
            group=sorted((a for a in active if a['ri']==ri),key=lambda a:(-a['s'],a['id']));leads={}
            for a in group:
                cell=cells[a['cell']];v=a['speed']/(1+max(0,cell['count']/cell['area']-.5)*1.8);before=a['s'];limited=False
                lane=a.get('lane',a['id']%route.get('lanes',1));lead=leads.get(lane,float('inf'))
                v=min(v,max(0,lead-a['s']-.65))
                for b in active_blocks[ri]:
                    if b['lo']-EPS<=a['s']<=b['hi']+EPS:
                        if v>0:limited=True
                        v=0
                    elif a['s']<b['lo']-EPS and a['s']+v>=b['lo']-EPS:v=max(0,b['lo']-EPS-a['s']);limited=True
                if limited:blocked[ri]+=1;prevented+=1
                a['v']=v;a['s']+=v;leads[lane]=a['s']
                for b in active_blocks[ri]:
                    if (before<b['lo']-EPS and a['s']>b['lo']+EPS) or (b['lo']-EPS<=before<=b['hi']+EPS and a['s']>before+EPS):violations.append({'time_s':t,'agent_id':a['id'],'route_id':route['id'],'event_id':b['eventId'],'from_m':before,'to_m':a['s']})
                if a['s']>=paths[ri]['length']-EPS:a['done']=True;outgoing[ri]+=1;completed[ri]+=1
    final=rows[-len(routes):]
    return {'mode':mode,'events':events,'barriers':closures,'frames':frames,'waitingFrames':waiting_frames,'rows':rows,'total':uid,'completed':sum(completed),'waiting':len(pending),'active':sum(r['active'] for r in final),'peakQueue':max(r['queued'] for r in rows),'audit':{'preventedMoves':prevented,'violations':violations},'modelVersion':'kinematic-v3'}

def simulate(c,mode):
    result=None
    for result in live_steps(c,mode):pass
    return result

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
