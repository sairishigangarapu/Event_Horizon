"""Run real Studio HTTP -> proxy -> model -> dashboard + report in one process tree."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]

def request(url,data=None):
    body=None if data is None else json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(url,data=body,headers={'Content-Type':'application/json'}),timeout=90) as r:
        return json.load(r)

def main():
    proc=subprocess.Popen([sys.executable,'start_integrated.py','--no-browser','--studio-port','8130','--runtime-port','8131'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                request('http://127.0.0.1:8131/health');request('http://127.0.0.1:8130/api/health');break
            except Exception:time.sleep(.5)
        else:raise AssertionError('Startup failed')
        c=json.loads((ROOT/'studio/examples/event_horizon_scene.json').read_text());c['duration']=40;c['events']=[]
        result=request('http://127.0.0.1:8130/api/render',c);run=result['runId']
        s=request('http://127.0.0.1:8130/api/runtime/sessions',dict(run_id=run,mode='ideal',duration=40,scale=c['scale'],routes=[{k:r[k] for k in ['id','name','width','points']} for r in c['routes']]))
        sid=s['session_id'];base=f'http://127.0.0.1:8130/api/runtime/sessions/{sid}'
        for t in range(36):
            obs=request(f'http://127.0.0.1:8130/api/observations/{run}/ideal?time_s={t}')
            request(base+'/advance',dict(time_s=t,observations=obs['observations']))
            labels=request('http://127.0.0.1:8130/api/runtime-labels',dict(run_id=run,mode='ideal',time_s=t))
            request(base+'/evaluation',labels)
        report=request(base)
        assert report['time_s']==35 and report['observation_count']>0
        assert any(r['actual'] is not None for r in report['comparisons'])
        assert any(r['central'] is not None and r['architecture'] is not None for r in report['comparisons'])
        assert report['metrics'] and len(report['tree_vectors'])==len(c['routes'])
        with urllib.request.urlopen('http://127.0.0.1:8131/dashboard/') as r:assert b'Dashboard' in r.read()
        with urllib.request.urlopen(base+'/report.csv') as r:assert b'architecture,central,actual' in r.read()
        print('PASS real HTTP: render -> delayed reports -> proxy -> six-channel model -> dual forecasts -> matured actuals -> metrics -> dashboard/report')
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill()

if __name__=='__main__':main()
