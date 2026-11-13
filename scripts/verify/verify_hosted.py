"""Actual TCP single-public-port Railway smoke test, no account/secrets required."""
import json,os,socket,subprocess,sys,tempfile,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def run():
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 base=f'http://127.0.0.1:{port}'
 def request(path,data=None,origin=None):
  headers={'Content-Type':'application/json'}
  if origin:headers['Origin']=origin
  r=urllib.request.Request(base+path,None if data is None else json.dumps(data).encode(),headers)
  with urllib.request.urlopen(r,timeout=60) as response:return response.read(),response.headers
 def api(path,data=None,origin=None):return json.loads(request(path,data,origin)[0])
 with tempfile.TemporaryDirectory() as tmp:
  env=dict(os.environ,PORT=str(port),EH_OUTPUT_ROOT=tmp+'/outputs',DATABASE_URL='sqlite:///'+tmp+'/test.db',MODEL_DIR=str(ROOT/'models/candidates/mc_v2'))
  with open(tmp+'/server.log','w+',encoding='utf-8') as log:
   process=subprocess.Popen([sys.executable,'-m','uvicorn','hosted_server:app','--host','127.0.0.1','--port',str(port),'--workers','1'],cwd=ROOT,env=env,stdout=log,stderr=log)
   try:
    for _ in range(120):
     if process.poll() is not None:log.seek(0);raise RuntimeError(log.read())
     try:
      if api('/api/ready')['ok']:break
     except (OSError,urllib.error.HTTPError):time.sleep(.1)
    else:raise RuntimeError('Hosted startup timed out.')
    assert b'EVENT HORIZON' in request('/')[0]
    assert b'Runtime Dashboard' in request('/dashboard/')[0]
    assert api('/api/runtime-config')['url']==''
    assert request('/app.js')[1].get_content_type() in ('text/javascript','application/javascript')
    c={'duration':40,'scale':.1,'seed':42,'randomCount':1,'routes':[{'id':'r','name':'Hosted — ಮುಖ್ಯ','color':'#ff633e','width':3,'count':20,'speed':1.3,'points':[{'x':100,'y':200},{'x':600,'y':200}]}],'events':[]}
    try:api('/api/render',c,origin='https://other.example');raise AssertionError('Cross-origin Studio mutation accepted')
    except urllib.error.HTTPError as error:assert error.code==403
    render=api('/api/render',c,origin=base);rid=render['runId']
    session=api('/api/runtime/sessions',dict(run_id=rid,mode='custom',scale=.1,duration=40,routes=[{k:c['routes'][0][k] for k in ('id','name','points','width')}]))['session_id']
    for t in range(36):
     reports=api(f'/api/observations/{rid}/custom?time_s={t}')['observations']
     api(f'/api/runtime/sessions/{session}/advance',dict(time_s=t,observations=reports))
     labels=api('/api/runtime-labels',dict(run_id=rid,mode='custom',time_s=t))
     api(f'/api/runtime/sessions/{session}/evaluation',labels)
    result=api(f'/api/runtime/sessions/{session}');assert result['metrics'] and result['observation_count']>0
    assert all(row['actual'] is None for row in result['comparisons'] if row['target_s']>35)
    assert 'ಮುಖ್ಯ' in request(f'/api/runs/{rid}/master_report.html')[0].decode('utf-8')
    print(json.dumps({'result':'PASS','single_port':True,'same_origin_dashboard':True,'render':'PASS','clock_s':result['time_s'],'future_actuals_hidden':True,'unicode_report':'PASS'}))
   finally:
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':run()
