"""Start the unpushed local review on one loopback port; never deploys anything."""
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
ROOT=Path(__file__).resolve().parent

def main():
    if sys.version_info[:2] not in ((3,11),(3,12)):
        raise SystemExit('Use Python 3.11 or 3.12 for this review package. On Windows run START_REVIEW_WINDOWS.bat after installing Python 3.12.')
    envdir=ROOT/'.venv_review';python=envdir/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():print('Creating the local review environment...',flush=True);venv.create(envdir,with_pip=True)
    fingerprint=hashlib.sha256((ROOT/'requirements.runtime.txt').read_bytes()+(ROOT/'requirements.railway.txt').read_bytes()).hexdigest()
    stamp=envdir/'review_dependencies.ok'
    if not stamp.exists() or stamp.read_text()!=fingerprint:
        print('First start: installing CPU-only dependencies. This can take several minutes.',flush=True)
        subprocess.run([str(python),'-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','torch>=2.1,<3'],check=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.railway.txt')],check=True)
        stamp.write_text(fingerprint)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=dict(os.environ,PORT=str(port),MODEL_DIR=str(ROOT/'models/candidates/mc_v2'),DATABASE_URL='sqlite:///'+str(ROOT/'data/local_review.db'),EH_OUTPUT_ROOT=str(ROOT/'data/review_outputs'),PYTHONUTF8='1')
    (ROOT/'data').mkdir(exist_ok=True)
    url=f'http://127.0.0.1:{port}/dashboard/'
    process=subprocess.Popen([str(python),'-m','uvicorn','hosted_server:app','--host','127.0.0.1','--port',str(port),'--workers','1'],cwd=ROOT,env=env)
    try:
        print('Loading the real model. Keep this terminal open.',flush=True)
        for _ in range(300):
            if process.poll() is not None:raise RuntimeError('Server exited; read the error above.')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/ready',timeout=1):break
            except OSError:time.sleep(.2)
        else:raise RuntimeError('Startup timed out.')
        print('\nLOCAL REVIEW: '+url+'\nNothing is uploaded or deployed. Ctrl+C stops it.',flush=True)
        webbrowser.open(url);process.wait()
    except KeyboardInterrupt:pass
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill()
if __name__=='__main__':main()
