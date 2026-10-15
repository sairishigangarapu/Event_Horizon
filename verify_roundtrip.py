"""Actual HTTP smoke test. Starts/stops both services, uses temporary run outputs."""
import json,os,socket,subprocess,sys,tempfile,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def freeport():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def run():
 ports=(freeport(),freeport());base=f'http://127.0.0.1:{ports[0]}'
 def api(path,data=None):
  request=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode('utf-8'),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=60) as r:return json.load(r)
 with tempfile.TemporaryDirectory() as temp:
  env=dict(os.environ,EH_OUTPUT_ROOT=temp,PYTHONUTF8='1')
  children=[subprocess.Popen([sys.executable,'lab_server.py','--port',str(ports[0]),'--no-browser'],cwd=ROOT/'studio',env=dict(env,EH_RUNTIME_URL=f'http://127.0.0.1:{ports[1]}'),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL),subprocess.Popen([sys.executable,'-m','uvicorn','runtime_server:app','--host','127.0.0.1','--port',str(ports[1])],cwd=ROOT,env=dict(env,MODEL_DIR=str(ROOT/'models/candidates/mc_v2'),DATABASE_URL='sqlite:///'+temp+'/test.db'),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)]
  try:
   for _ in range(100):
    if any(p.poll() is not None for p in children):raise RuntimeError('A smoke-test service stopped.')
    try:
     api('/api/health');api('/api/runtime/sessions');break
    except (OSError,urllib.error.HTTPError):time.sleep(.1)
   else:raise RuntimeError('Startup timed out.')
   c={'duration':40,'scale':.1,'seed':42,'randomCount':1,'routes':[{'id':'r','name':'North — ಮುಖ್ಯ','color':'#ff633e','width':3,'count':30,'speed':1.3,'points':[{'x':100,'y':200},{'x':600,'y':200}]}], 'events':[{'id':'closed','name':'Exit — ಮುಚ್ಚಲಾಗಿದೆ','type':'collapse','routeId':'r','at':.6,'start':10,'end':25,'radiusM':3,'people':0}]}
   rendered=api('/api/render',c);rid=rendered['runId'];assert set(rendered['results'])=={'ideal','custom','random'}
   started=api('/api/runtime/sessions',{'run_id':rid,'mode':'custom','scale':c['scale'],'duration':c['duration'],'routes':[{k:c['routes'][0][k] for k in ('id','name','points','width')}]});sid=started['session_id']
   assert not started['readiness']['production_ready'] and started['readiness']['synthetic_only']
   for t in range(36):
    reports=api(f'/api/observations/{rid}/custom?time_s={t}')['observations'];assert all(r['available_at_s']<=t and r['observed_at_s']<=t for r in reports)
    api(f'/api/runtime/sessions/{sid}/advance',{'time_s':t,'observations':reports})
    labels=api('/api/runtime-labels',{'run_id':rid,'mode':'custom','time_s':t})
    api(f'/api/runtime/sessions/{sid}/evaluation',labels)
   snapshot=api(f'/api/runtime/sessions/{sid}');assert snapshot['observation_count']>0 and snapshot['tree_vectors'] and snapshot['comparisons'] and snapshot['metrics']
   assert all(r['actual'] is None for r in snapshot['comparisons'] if r['target_s']>35)
   assert snapshot['cluster_assignments'] is not None
   with urllib.request.urlopen(base+f'/api/runs/{rid}/master_report.html') as r:assert 'ಮುಖ್ಯ' in r.read().decode('utf-8')
   with urllib.request.urlopen(f'http://127.0.0.1:{ports[1]}/dashboard/') as r:assert r.status==200
   with urllib.request.urlopen(base+f'/api/runtime/sessions/{sid}/report.csv') as r:assert b'architecture' in r.read()
   print(json.dumps({'result':'PASS','clock_s':snapshot['time_s'],'received_measurements':snapshot['observation_count'],'comparisons':len(snapshot['comparisons']),'matured_metric_groups':len(snapshot['metrics']),'utf8_report':'PASS','future_actuals_hidden':'PASS','cluster_assignments':snapshot['cluster_assignments']},indent=2))
  finally:
   for p in children:p.terminate()
   for p in children:
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
if __name__=='__main__':run()
