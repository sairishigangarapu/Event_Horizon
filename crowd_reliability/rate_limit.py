"""Bounded process-local token bucket for production API requests."""
from collections import OrderedDict
import threading
import time

class ApiRateLimit:
    def __init__(self,max_clients=1024):
        self.clients=OrderedDict();self.lock=threading.Lock();self.max_clients=max_clients
    def allow(self,client,per_minute,now=None):
        now=time.monotonic() if now is None else now;rate=per_minute/60;burst=min(60,per_minute)
        with self.lock:
            tokens,at=self.clients.pop(client,(float(burst),now))
            tokens=min(burst,tokens+max(0,now-at)*rate)
            ok=tokens>=1
            self.clients[client]=(tokens-1 if ok else tokens,now)
            while len(self.clients)>self.max_clients:self.clients.popitem(last=False)
            return ok
