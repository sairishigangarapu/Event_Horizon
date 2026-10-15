"""Seeded winding routes constrained to the schematic venue's walkable space."""
import copy
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path
import random
from typing import Literal
from pydantic import Field
from crowd_reliability.runtime import Strict
from simulation import path

VENUE=json.loads((Path(__file__).resolve().parents[1]/'studio/venue_geometry.json').read_text())
AREAS=[a for a in VENUE['areas'] if a['type'] not in ('pond','gate')]
OBSTACLES=[a for a in VENUE['areas'] if a['type']=='pond']+[{**s,'d':s.get('d',s.get('h',10))} for s in VENUE['stalls']]
COLORS=['#e56c44','#3c7eaf','#829846']
class Flow(Strict):
    density:float=Field(default=.8,ge=.05,le=3)
    direction:Literal['outbound','inbound','mixed']='outbound'
    duration:int=Field(default=150,ge=40,le=300)
    seed:int=Field(default=731,ge=0,le=4294967295)
    route_count:int=Field(default=2,ge=1,le=3)
    scenario:Literal['random','normal','surge','closure','combined']='random'
    density_threshold:float=Field(default=2.5,ge=.5,le=6)
    queue_threshold:int=Field(default=15,ge=3,le=100)

def walkable(x,y,margin=10):
    for dx,dy in [(0,0),(margin,0),(-margin,0),(0,margin),(0,-margin),(margin*.7,margin*.7),(-margin*.7,-margin*.7),(margin*.7,-margin*.7),(-margin*.7,margin*.7)]:
        xx,yy=x+dx,y+dy
        if not any(a['x']<=xx<=a['x']+a['w'] and a['y']<=yy<=a['y']+a['d'] for a in AREAS):return False
        if any(a['x']<=xx<=a['x']+a['w'] and a['y']<=yy<=a['y']+a['d'] for a in OBSTACLES):return False
    return True

def segment_ok(a,b):
    n=max(1,math.ceil(math.dist(a,b)/3))
    return all(walkable(a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n) for i in range(n+1))

@lru_cache(maxsize=96)
def route_points(seed,index):
    rng=random.Random(seed+index*7919)
    nodes={(x,y) for x in range(200,951,10) for y in range(70,641,10) if walkable(x,y)}
    costs={n:1+rng.random()*.8 for n in sorted(nodes)}
    nearest=lambda p:min(nodes,key=lambda n:math.dist(n,p))
    start=nearest((375+index*12,90));end=nearest((215,615) if index%2==0 else (940,575))
    candidates=sorted(n for n in nodes if 295<n[0]<640 and 300<n[1]<570)
    via=rng.sample(candidates,2)
    def search(a,b):
        queue=[(0,a)];dist={a:0};prev={}
        while queue:
            _,u=heapq.heappop(queue)
            if u==b:
                out=[u]
                while u!=a:u=prev[u];out.append(u)
                return out[::-1]
            for dx,dy in [(10,0),(-10,0),(0,10),(0,-10),(10,10),(10,-10),(-10,10),(-10,-10)]:
                v=(u[0]+dx,u[1]+dy)
                if v not in nodes or not segment_ok(u,v):continue
                d=dist[u]+math.hypot(dx,dy)/10*costs[v]
                if d<dist.get(v,float('inf')):
                    dist[v]=d;prev[v]=u;heapq.heappush(queue,(d+math.dist(v,b)/10,v))
        raise ValueError('No walkable route between sampled points.')
    points=[]
    for a,b in zip([start]+via,via+[end]):points.extend(search(a,b)[bool(points):])
    reduced=[points[0]];i=0
    while i<len(points)-1:
        j=min(i+4,len(points)-1)
        while j>i+1 and not segment_ok(points[i],points[j]):j-=1
        reduced.append(points[j]);i=j
    # Corner cutting produces smooth, irregular curves. Each pass is constrained.
    for _ in range(3):
        smooth=[reduced[0]]
        for a,b in zip(reduced,reduced[1:]):smooth.extend([(a[0]*.75+b[0]*.25,a[1]*.75+b[1]*.25),(a[0]*.25+b[0]*.75,a[1]*.25+b[1]*.75)])
        smooth.append(reduced[-1])
        if all(segment_ok(a,b) for a,b in zip(smooth,smooth[1:])):reduced=smooth
    # Max 500 points, with no unsafe shortcuts.
    if len(reduced)>500:raise ValueError('Generated curve is too complex; choose another seed.')
    return [{'x':round(x,3),'y':round(y,3)} for x,y in reduced]

def make_scene(data:Flow,incidents=False):
    rng=random.Random(data.seed);routes=[]
    for i in range(data.route_count):
        points=copy.deepcopy(route_points(data.seed,i))
        reverse=data.direction=='inbound' or data.direction=='mixed' and i%2==1
        if reverse:points.reverse()
        r=dict(id=f'curve_{i+1}',name=f'Curve {i+1} · '+('inbound' if reverse else 'outbound'),color=COLORS[i],width=1.8,speed=1.3,points=points,initial_fill=True,lanes=3)
        area=path(r,.1)['length']*r['width'];r['count']=max(1,round(data.density*area));r['arrival_rate']=round(r['count']*.12/data.duration,3)
        routes.append(r)
    population=sum(r['count'] for r in routes)
    if population>1600 or any(r['count']>1500 for r in routes):raise ValueError(f'Density needs {population} initial people on these curves; live limit is 1600. Reduce density or route count.')
    scene=dict(duration=data.duration,scale=.1,seed=data.seed,randomCount=1,routes=routes,events=[])
    # This separate RNG never changes the routes or the model inputs directly.
    hidden=random.Random(data.seed^0x9e3779b9)
    scenario=hidden.choice(['normal','surge','closure','combined']) if data.scenario=='random' else data.scenario
    if incidents and scenario!='normal':
        route=hidden.choice(routes);start=hidden.randint(max(10,data.duration//5),max(11,data.duration//3))
        if scenario in ('closure','combined'):
            scene['events'].append(dict(id='hidden_closure',name='Random obstruction',type='collapse',routeId=route['id'],at=hidden.uniform(.55,.85),radiusM=2.5,start=start,end=min(data.duration,start+hidden.randint(30,60)),people=0))
        if scenario in ('surge','combined'):
            scene['events'].append(dict(id='hidden_surge',name='Random arrival surge',type='surge',routeId=route['id'],at=hidden.uniform(.08,.3),start=start+3,end=min(data.duration,start+15),people=min(200,max(40,round(population*.25)))))
    meta=dict(requested_density=data.density,initial_population=population,measured_initial_density=population/sum(path(r,.1)['length']*r['width'] for r in routes),direction=data.direction,route_count=len(routes),density_definition='Initial people divided by the sum of route-corridor areas; overlaps count per stream, not unique venue area.',scenario='hidden until observed',seed=data.seed)
    return scene,meta
