"""Local API + static studio, Python standard library only."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
from urllib.parse import urlsplit, parse_qs
import os, urllib.request, urllib.error
import argparse, base64, csv, json, re, tempfile, threading, uuid, webbrowser
from simulation import validate, simulate, normalize_routes
from reporting import prepare_report, render_report
from observations import sources_for, export_separated, read_observations
ROOT=Path(__file__).resolve().parent
NAMES={'ideal':'01_ideal_flow.csv','custom':'02_custom_disasters.csv','random':'03_random_collapses.csv'}
RUN_LOCK=threading.Lock()
SESSION=Path(os.environ.get('EH_OUTPUT_ROOT',str(ROOT/'outputs')));SESSION.mkdir(parents=True,exist_ok=True)
JOURNAL_LOCK=threading.Lock()
class LabHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def json_response(self,data,status=200):
        body=json.dumps(data,allow_nan=False,separators=(',',':')).encode()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_HEAD(self):
        if getattr(self.server,'inference_only',False):return self.json_response({'error':'Use GET /api/observations/<run>/<mode>?time_s=T'},404)
        super().do_HEAD()
    def do_GET(self):
        path=urlsplit(self.path).path
        observation=re.fullmatch(r'/api/observations/([a-f0-9]{32})/(ideal|custom|random)',path)
        if observation:
            try:
                query=parse_qs(urlsplit(self.path).query)
                if set(query)!={'time_s'} or len(query['time_s'])!=1:raise ValueError('Provide only time_s=T.')
                cutoff=float(query['time_s'][0])
                records=read_observations(getattr(self.server,'observation_root',SESSION),observation[1],observation[2],cutoff)
                return self.json_response({'as_of_s':cutoff,'observations':records})
            except FileNotFoundError:return self.json_response({'error':'Observation stream unavailable.'},404)
            except (ValueError,OverflowError):return self.json_response({'error':'time_s must be a finite nonnegative number.'},400)
        if getattr(self.server,'inference_only',False):
            if path=='/api/health':return self.json_response({'ok':True,'engine':'python','role':'observations-only'})
            return self.json_response({'error':'This service exposes observations only.'},404)
        if path=='/api/runtime-config':
            return self.json_response({'url':os.environ.get('EH_RUNTIME_URL','http://127.0.0.1:8001')})
        if path.startswith('/api/runtime/'):
            return self.runtime_proxy('GET')
        if path=='/api/health':return self.json_response({'ok':True,'engine':'python','version':3})
        m=re.fullmatch(r'/api/runs/([a-f0-9]{32})/(01_ideal_flow.csv|02_custom_disasters.csv|03_random_collapses.csv|run_manifest.json|master_report.html)',path)
        if m:
            file=SESSION/m[1]/m[2]
            if not file.is_file():return self.json_response({'error':'Run file unavailable; render again.'},404)
            data=file.read_bytes();self.send_response(200);self.send_header('Content-Type','text/csv' if file.suffix=='.csv' else 'text/html; charset=utf-8' if file.suffix=='.html' else 'application/json');self.send_header('Content-Disposition',f'attachment; filename="{file.name}"');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        super().do_GET()
    def runtime_proxy(self,method,body=None):
        base=os.environ.get('EH_RUNTIME_URL','http://127.0.0.1:8001')
        headers={'Content-Type':'application/json'}
        if self.headers.get('X-API-Key'):headers['X-API-Key']=self.headers['X-API-Key']
        request=urllib.request.Request(base+self.path,data=body,method=method,headers=headers)
        try:
            with urllib.request.urlopen(request,timeout=60) as response:
                payload=response.read()
                self.send_response(response.status)
                for header in ('Content-Type','Content-Disposition'):
                    if response.headers.get(header):self.send_header(header,response.headers[header])
                self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
                return
        except urllib.error.HTTPError as exc:
            return self.json_response(json.loads(exc.read()),exc.code)
        except (OSError,ValueError) as exc:
            return self.json_response({'error':'Runtime unavailable. Start with start_integrated.py. '+str(exc)},503)
    def do_POST(self):
        if getattr(self.server,'inference_only',False):return self.json_response({'error':'Inference service is read-only.'},403)
        try:
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:
                return self.json_response({'error':'Use the local studio URL.'},403)
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>12_000_000:raise ValueError('Upload must be under 12 MB.')
            raw=self.rfile.read(size)
            if self.path.startswith('/api/runtime/'):
                return self.runtime_proxy('POST',raw)
            data=json.loads(raw)
            if self.path=='/api/runtime-labels':
                run=data.get('run_id','');mode=data.get('mode');cutoff=data.get('time_s')
                if not re.fullmatch(r'[a-f0-9]{32}',run) or mode not in NAMES or not isinstance(cutoff,int) or not 0<=cutoff<=900:
                    raise ValueError('Invalid evaluation cutoff.')
                file=SESSION/run/'hidden_ground_truth'/mode/'metrics.csv'
                if not file.exists():raise ValueError('Render is no longer available.')
                keys={'mean_density_p_m2':'density','mean_speed_m_s':'speed','spawned':'inflow','exited':'outflow','queued':'queue_length'}
                with file.open(encoding='utf-8') as f:
                    actuals=[{'time_s':cutoff,'zone_id':row['route_id'],'variable':var,'value':float(row[key])} for row in csv.DictReader(f) if int(row['time_s'])==cutoff for key,var in keys.items()]
                return self.json_response({'actuals':actuals})
            if self.path=='/api/decisions':
                project=data.get('projectId','');entry=data.get('entry')
                if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',project) or not isinstance(entry,dict):raise ValueError('Invalid decision record.')
                if len(json.dumps(entry))>2_000_000:raise ValueError('Decision record is too large.')
                with JOURNAL_LOCK:
                    with (SESSION/('decisions_'+project+'.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
                return self.json_response({'ok':True})
            if self.path=='/api/routes':
                scale=data.get('scale',.1)
                if not isinstance(scale,(int,float)) or not 0<scale<=2:raise ValueError('Set a valid scale.')
                routes=normalize_routes(data.get('payload'),scale)
                validate({'routes':routes,'events':[],'duration':180,'scale':scale,'seed':42,'randomCount':3})
                return self.json_response({'routes':routes})
            if self.path=='/api/venue-image':
                url=data.get('image','');m=re.fullmatch(r'data:image/(png|jpeg|webp|gif);base64,(.+)',url)
                if not m:raise ValueError('Use PNG, JPEG, WEBP or GIF.')
                raw=base64.b64decode(m[2],validate=True)
                if len(raw)>8_000_000:raise ValueError('Image must be under 8 MB.')
                valid=raw.startswith(b'\x89PNG\r\n\x1a\n') or raw.startswith(b'\xff\xd8\xff') or raw[:6] in [b'GIF87a',b'GIF89a'] or (raw.startswith(b'RIFF') and raw[8:12]==b'WEBP')
                if not valid:raise ValueError('Image data is invalid.')
                asset=uuid.uuid4().hex; (SESSION/(asset+'.'+m[1])).write_bytes(raw)
                return self.json_response({'ok':True,'assetId':asset})
            if self.path=='/api/render':
                validate(data)
                sources_for(data)
                if not RUN_LOCK.acquire(blocking=False):return self.json_response({'error':'A render is already running.'},409)
                try:
                    results={m:simulate(data,m) for m in NAMES}
                    run_id=uuid.uuid4().hex;directory=SESSION/run_id;directory.mkdir()
                    export_separated(directory,data,results)
                    for mode,name in NAMES.items():
                        rows=results[mode]['rows']
                        with (directory/name).open('w',newline='',encoding='utf-8-sig') as f:
                            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    manifest={'model':'event-horizon-kinematic-v3','config':data,'events':{m:r['events'] for m,r in results.items()},'note':'Synthetic movement. Incident log is scenario truth; keep it out of prediction inputs.'}
                    (directory/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                    prepared=prepare_report(data,results,data.get('_decisionLog',[]),{'source':'Studio render','provenance':'Generated by the local Python engine from this saved render configuration.'})
                    (directory/'master_report.html').write_text(render_report(prepared),encoding='utf-8')
                    return self.json_response({'results':results,'runId':run_id})
                finally: RUN_LOCK.release()
            self.json_response({'error':'Unknown API endpoint.'},404)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:self.json_response({'error':str(exc)},400)
        except (BrokenPipeError,ConnectionResetError):pass

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8000);parser.add_argument('--no-browser',action='store_true');parser.add_argument('--inference-only',action='store_true');parser.add_argument('--observation-root',type=Path,default=SESSION);args=parser.parse_args()
    try:
        with ThreadingHTTPServer(('127.0.0.1',args.port),partial(LabHandler,directory=str(ROOT/'dist'))) as server:
            server.inference_only=args.inference_only;server.observation_root=args.observation_root
            url=f'http://127.0.0.1:{args.port}/';print(f'Event Horizon 3D Studio + Python API: {url}',flush=True)
            if not args.no_browser and not args.inference_only:webbrowser.open(url)
            try:server.serve_forever()
            except KeyboardInterrupt:print('\nStudio stopped.')
    except OSError as exc:print(f'Cannot start server: {exc}\nTry python lab_server.py --port 8001');raise SystemExit(1)
