"""Local API + static studio, Python standard library only."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
from urllib.parse import urlsplit
import argparse, base64, csv, json, re, tempfile, threading, uuid, webbrowser
from simulation import validate, simulate, normalize_routes
ROOT=Path(__file__).resolve().parent
NAMES={'ideal':'01_ideal_flow.csv','custom':'02_custom_disasters.csv','random':'03_random_collapses.csv'}
RUN_LOCK=threading.Lock()
SESSION=Path(tempfile.mkdtemp(prefix='event_horizon_'))
class LabHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def json_response(self,data,status=200):
        body=json.dumps(data,allow_nan=False,separators=(',',':')).encode()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/api/health':return self.json_response({'ok':True,'engine':'python','version':2})
        m=re.fullmatch(r'/api/runs/([a-f0-9]{32})/(01_ideal_flow.csv|02_custom_disasters.csv|03_random_collapses.csv|run_manifest.json)',path)
        if m:
            file=SESSION/m[1]/m[2]
            if not file.is_file():return self.json_response({'error':'Run file unavailable; render again.'},404)
            data=file.read_bytes();self.send_response(200);self.send_header('Content-Type','text/csv' if file.suffix=='.csv' else 'application/json');self.send_header('Content-Disposition',f'attachment; filename="{file.name}"');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        super().do_GET()
    def do_POST(self):
        try:
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:
                return self.json_response({'error':'Use the local studio URL.'},403)
            size=int(self.headers.get('Content-Length','0'))
            if size<=0 or size>12_000_000:raise ValueError('Upload must be under 12 MB.')
            data=json.loads(self.rfile.read(size))
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
                if not RUN_LOCK.acquire(blocking=False):return self.json_response({'error':'A render is already running.'},409)
                try:
                    results={m:simulate(data,m) for m in NAMES}
                    run_id=uuid.uuid4().hex;directory=SESSION/run_id;directory.mkdir()
                    for mode,name in NAMES.items():
                        rows=results[mode]['rows']
                        with (directory/name).open('w',newline='',encoding='utf-8-sig') as f:
                            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    manifest={'model':'event-horizon-kinematic-v2','config':data,'events':{m:r['events'] for m,r in results.items()},'note':'Synthetic movement. Incident log is scenario truth; keep it out of prediction inputs.'}
                    (directory/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                    return self.json_response({'results':results,'runId':run_id})
                finally: RUN_LOCK.release()
            self.json_response({'error':'Unknown API endpoint.'},404)
        except (ValueError,TypeError,KeyError,OverflowError) as exc:self.json_response({'error':str(exc)},400)
        except (BrokenPipeError,ConnectionResetError):pass

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8000);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    try:
        with ThreadingHTTPServer(('127.0.0.1',args.port),partial(LabHandler,directory=str(ROOT/'dist'))) as server:
            url=f'http://127.0.0.1:{args.port}/';print(f'Event Horizon 3D Studio + Python API: {url}',flush=True)
            if not args.no_browser:webbrowser.open(url)
            try:server.serve_forever()
            except KeyboardInterrupt:print('\nStudio stopped.')
    except OSError as exc:print(f'Cannot start server: {exc}\nTry python lab_server.py --port 8001');raise SystemExit(1)
