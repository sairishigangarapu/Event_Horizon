"""Bounded JSON/upload bodies before validation or model work."""
from starlette.responses import JSONResponse
from crowd_reliability.config import settings

class ApiBodyLimit:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or not scope.get('path','').startswith('/api/') or scope.get('method') not in ('POST','PUT','PATCH'):
            return await self.app(scope,receive,send)
        limit=settings.api_max_body_bytes
        headers=dict(scope.get('headers',[]))
        try:length=int(headers.get(b'content-length',b'0'))
        except ValueError:length=-1
        if length<0 or length>limit:
            return await JSONResponse({'detail':'API body must be at most '+str(limit)+' bytes.'},status_code=413)(scope,receive,send)
        chunks=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            chunk=message.get('body',b'');size+=len(chunk)
            if size>limit:
                return await JSONResponse({'detail':'API body must be at most '+str(limit)+' bytes.'},status_code=413)(scope,receive,send)
            chunks.append(chunk)
            if not message.get('more_body',False):break
        body=b''.join(chunks);sent=False
        async def replay():
            nonlocal sent
            if not sent:sent=True;return {'type':'http.request','body':body,'more_body':False}
            return await receive()
        return await self.app(scope,replay,send)
