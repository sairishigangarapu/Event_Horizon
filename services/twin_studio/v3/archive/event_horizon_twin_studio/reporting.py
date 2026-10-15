"""Self-contained master report using the same renderer as the studio."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math
from simulation import simulate
ROOT=Path(__file__).resolve().parent

def color(d):
    stops=[(0,'b3d76d'),(1,'d8de62'),(2,'f6ba4e'),(4,'f57b38'),(6,'ce3d32')];i=0
    while i<len(stops)-2 and d>stops[i+1][0]:i+=1
    a,ca=stops[i];b,cb=stops[i+1];t=max(0,min(1,(d-a)/(b-a)))
    return '#'+''.join(f'{math.floor(int(ca[k:k+2],16)*(1-t)+int(cb[k:k+2],16)*t+.5):02x}' for k in [0,2,4])

def prepare_report(config,results,journal=None,meta=None):
    c=json.loads(json.dumps(config));c.pop('_decisionLog',None)
    if c.get('background'):c['background']='[Uploaded image omitted; see project export]'
    runs={}
    for mode,r in results.items():
        times={};heat={}
        for row in r['rows']:
            t=times.setdefault(row['time_s'],{'time_s':row['time_s'],'active':0,'queued':0,'waiting':0,'completed':0,'density':0})
            for dest,src in [('active','active'),('queued','queued'),('waiting','waiting_entry'),('completed','completed')]:t[dest]+=row.get(src,0)
            t['density']=max(t['density'],row['peak_local_density_p_m2'])
        unit=2/c['scale']
        for frame in r['frames']:
            cells={}
            for a in frame:
                ix=math.floor(min(999.999999,max(0,a['x']))/unit);iy=math.floor(min(699.999999,max(0,a['y']))/unit)
                key=(ix,iy);cells[key]=cells.get(key,0)+1
            for (ix,iy),count in cells.items():
                w=min(unit,1000-ix*unit);h=min(unit,700-iy*unit);area=w*h*c['scale']**2;density=count/area;key=f'{ix},{iy}'
                if key not in heat or heat[key]['density']<density:heat[key]={'key':key,'x':ix*unit,'y':iy*unit,'w':w,'h':h,'count':count,'area':area,'density':density,'color':color(density)}
        runs[mode]={'total':r['total'],'completed':r['completed'],'active':r.get('active',0),'waiting':r.get('waiting',0),'events':r['events'],'barriers':r.get('barriers',[]),'audit':r.get('audit',{}),'timeline':list(times.values()),'heat':list(heat.values()),'finalRows':[{**x,'admitted_total':sum(a['spawned'] for a in r['rows'] if a['route_id']==x['route_id'])} for x in r['rows'] if x['time_s']==c['duration']]}
    journal=journal if isinstance(journal,list) else []
    warnings=[f'“{e["name"]}” is a Surge (+{e["people"]} arrivals), not a collapse. Its saved type was preserved.' for e in c['events'] if e['type']=='surge' and 'collaps' in e['name'].lower()]
    if not journal or any(e.get('action')=='project_opened' and e.get('details',{}).get('historicalJournalAvailable') is False for e in journal):warnings.append('No historical decision journal was supplied. Earlier clicks and reasons cannot be reconstructed; this scene is the recorded baseline.')
    return {'version':3,'generatedAt':datetime.now(timezone.utc).isoformat(),'config':c,'runs':runs,'journal':journal,'meta':meta or {},'warnings':warnings}

def render_report(prepared):
    scripts='\n'.join((ROOT/'dist'/name).read_text() for name in ['density.js','venue.js','report.js'])
    data=json.dumps(prepared,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Event Horizon — Master Report</title></head><body><noscript>Enable JavaScript to display this self-contained report.</noscript><script>'+scripts+'\nconst reportData='+data+';let style=document.createElement("style");style.textContent=EHReport.styles;document.head.appendChild(style);document.body.innerHTML=EHReport.body(reportData);</script></body></html>'

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('scene');parser.add_argument('--out',default='Event_Horizon_Master_Report.html');args=parser.parse_args()
    raw=Path(args.scene).read_bytes();c=json.loads(raw);results={m:simulate(c,m) for m in ['ideal','custom','random']}
    prepared=prepare_report(c,results,c.get('decisionLog'),{'source':Path(args.scene).name,'sourceSha256':hashlib.sha256(raw).hexdigest(),'provenance':'Fresh v3 reruns of the uploaded scene; not a replay of previously uploaded CSVs.'})
    Path(args.out).write_text(render_report(prepared),encoding='utf-8');print(args.out)
