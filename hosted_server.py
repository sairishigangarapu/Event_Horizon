"""Railway: one public ASGI port, Studio plus the separate runtime dashboard."""
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response

ROOT=Path(__file__).resolve().parent
# Defaults are established before the runtime imports its configuration.
os.environ.setdefault('MODEL_DIR',str(ROOT/'models/candidates/mc_v2'))
os.environ.setdefault('EH_OUTPUT_ROOT',str(ROOT/'data/studio_outputs'))
os.environ.setdefault('DATABASE_URL','sqlite:///'+str(ROOT/'data/hosted.db'))
os.environ.setdefault('EH_RUNTIME_URL','http://127.0.0.1:'+os.environ.get('PORT','8000'))
sys.path.insert(0,str(ROOT/'studio'))
from lab_server import LabHandler
from runtime_server import app
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.config import settings

previous_lifespan=app.router.lifespan_context
@asynccontextmanager
async def hosted_lifespan(application):
    # Fail startup instead of exposing an app whose model artifacts cannot load.
    model=ArchitecturePipeline();model.load_pipeline(settings.model_dir)
    application.state.model_readiness=model.readiness()
    del model
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(LabHandler,directory=str(ROOT/'studio/dist')))
    server.inference_only=False
    server.observation_root=Path(os.environ['EH_OUTPUT_ROOT'])
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    application.state.studio_origin=f'http://127.0.0.1:{server.server_port}'
    application.state.studio_thread=thread
    async with httpx.AsyncClient(timeout=120,trust_env=False) as client:
        application.state.studio_client=client
        try:
            async with previous_lifespan(application):yield
        finally:
            await asyncio.to_thread(server.shutdown);server.server_close();thread.join(timeout=5)
app.router.lifespan_context=hosted_lifespan
# The old API root described the service; the public root now serves Studio.
app.router.routes[:]=[r for r in app.router.routes if not (getattr(r,'path',None)=='/' and getattr(r,'name',None)=='get_root')]

@app.get('/api/runtime-config')
def runtime_config():
    # Empty means same origin in the browser, never a visitor's localhost.
    return {'url':''}

@app.get('/api/ready')
def ready():
    if not app.state.studio_thread.is_alive():return JSONResponse({'ok':False},status_code=503)
    return {'ok':True,'studio':True,'runtime':True,'model':app.state.model_readiness}

@app.api_route('/{studio_path:path}',methods=['GET','POST','HEAD'])
async def studio(request:Request,studio_path:str):
    if request.method=='POST':
        origin=request.headers.get('origin')
        if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
            return JSONResponse({'error':'Use the application origin.'},status_code=403)
    length=request.headers.get('content-length')
    try:
        if length and int(length)>12_000_000:return JSONResponse({'error':'Upload must be under 12 MB.'},status_code=413)
    except ValueError:return JSONResponse({'error':'Invalid content length.'},status_code=400)
    body=bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body)>12_000_000:return JSONResponse({'error':'Upload must be under 12 MB.'},status_code=413)
    query=('?'+request.url.query) if request.url.query else ''
    headers={'Content-Type':request.headers.get('content-type','application/json')}
    if request.method=='POST':headers['Origin']=app.state.studio_origin
    try:
        result=await app.state.studio_client.request(request.method,app.state.studio_origin+'/'+studio_path+query,content=bytes(body),headers=headers)
    except httpx.HTTPError:
        return JSONResponse({'error':'Studio is unavailable; retry shortly.'},status_code=503)
    forwarded={k:v for k,v in result.headers.items() if k.lower() in ('content-type','content-disposition','cache-control')}
    return Response(result.content,status_code=result.status_code,headers=forwarded)
