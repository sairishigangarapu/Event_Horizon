"""Starts the Studio first and the model runtime on a separate local port."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--studio-port',type=int,default=8000);p.add_argument('--runtime-port',type=int,default=8001);p.add_argument('--no-browser',action='store_true');args=p.parse_args()
    if args.studio_port==args.runtime_port:
        p.error('Use different ports for Studio and runtime.')
    trained=ROOT/'models/candidates/mc_v2'
    model_dir=os.environ.get('MODEL_DIR') or str(trained if (trained/'manifest.json').is_file() else ROOT/'models/production')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MODEL_DIR=model_dir,DATABASE_URL='sqlite:///'+str(ROOT/'data/runtime.db'),
             EH_RUNTIME_URL=f'http://127.0.0.1:{args.runtime_port}')
    (ROOT/'data').mkdir(exist_ok=True)
    processes=[]
    try:
        processes.append(subprocess.Popen([sys.executable,'lab_server.py','--port',str(args.studio_port),'--no-browser'],cwd=ROOT/'studio',env=env))
        processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','runtime_server:app','--host','127.0.0.1','--port',str(args.runtime_port)],cwd=ROOT,env=env))
        for _ in range(120):
            if any(x.poll() is not None for x in processes):
                raise RuntimeError('A service stopped. Read the error above; check dependencies and occupied ports.')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{args.runtime_port}/health',timeout=1).close()
                urllib.request.urlopen(f'http://127.0.0.1:{args.studio_port}/api/health',timeout=1).close()
                break
            except Exception: time.sleep(.5)
        else: raise RuntimeError('Startup timed out.')
        print(f'\nSTUDIO: http://127.0.0.1:{args.studio_port}/\nDASHBOARD: http://127.0.0.1:{args.runtime_port}/dashboard/\nOpen Studio, render a run, then press Play. Ctrl+C stops both services.',flush=True)
        if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.studio_port}/')
        while all(x.poll() is None for x in processes):time.sleep(.5)
    except KeyboardInterrupt:pass
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill()

if __name__=='__main__':main()
