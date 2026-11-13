"""Shared access boundary for core, live-demo, hosted Studio and v3 API routes."""
import hmac
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from crowd_reliability.config import settings

api_key_header=APIKeyHeader(name='X-API-Key',auto_error=False)

_LOCALHOST_HOSTS = frozenset({'127.0.0.1', '::1', 'localhost', 'testclient'})

def check_api_key(value, client_host=None):
    """Shared access boundary.

    - local_demo: allows access from localhost only (no API key needed).
      Non-localhost requests are rejected (403) to prevent accidental
      network exposure of an unauthenticated service.
    - api_key: requires X-API-Key matching settings.app_api_key.
      Raises 503 if the key is not configured; 401 if the header is
      absent or incorrect.
    """
    if settings.auth_mode == 'local_demo':
        if client_host is not None and client_host not in _LOCALHOST_HOSTS:
            raise HTTPException(403, 'local_demo mode is only allowed from localhost')
        return
    expected = settings.app_api_key
    if settings.auth_mode != 'api_key' or not expected:
        raise HTTPException(503, 'API access is not configured.')
    if not value or not hmac.compare_digest(value, expected):
        raise HTTPException(401, 'Invalid or missing API key.')

def verify_auth(api_key=Security(api_key_header)):
    check_api_key(api_key)
    return True
