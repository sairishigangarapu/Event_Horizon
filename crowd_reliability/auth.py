"""Shared access boundary for core, live-demo, hosted Studio and v3 API routes."""
import hmac
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from crowd_reliability.config import settings

api_key_header=APIKeyHeader(name='X-API-Key',auto_error=False)

def check_api_key(value):
    if settings.auth_mode=='local_demo':return
    expected=settings.app_api_key
    if settings.auth_mode!='api_key' or not expected:
        raise HTTPException(503,'API access is not configured.')
    if not value or not hmac.compare_digest(value,expected):
        raise HTTPException(401,'Invalid or missing API key.')

def verify_auth(api_key=Security(api_key_header)):
    check_api_key(api_key)
    return True
