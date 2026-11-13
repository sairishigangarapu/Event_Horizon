"""Real TCP, authenticated frontend-to-backend random-flow integration."""
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]

def run():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}';key=secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory() as tmp:
        env=dict(os.environ,PORT=str(port),APP_ENV='test',AUTH_MODE='api_key',APP_API_KEY=key,
            MODEL_DIR=str(ROOT/'models/candidates/mc_v2'),DATABASE_URL='sqlite:///'+tmp+'/test.db',
            EH_OUTPUT_ROOT=tmp+'/outputs',EH_SMOKE_API_KEY=key,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with open(tmp+'/server.log','w+',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','hosted_server:app','--host','127.0.0.1','--port',str(port),'--workers','1'],cwd=ROOT,env=env,stdout=log,stderr=log)
            try:
                for _ in range(300):
                    if process.poll() is not None:log.seek(0);raise RuntimeError(log.read())
                    try:
                        with urllib.request.urlopen(base+'/api/ready',timeout=1):break
                    except OSError:time.sleep(.1)
                else:raise RuntimeError('Hosted startup timed out.')
                subprocess.run(['node','tests/live_dashboard_smoke.cjs',base],cwd=ROOT,env=env,timeout=120,check=True)
            finally:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':run()
