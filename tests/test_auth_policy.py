"""Regression tests for F-007: authentication policy.

Verifies that:
- local_demo mode allows localhost without API key, blocks non-localhost (403)
- api_key mode requires correct key, blocks missing/wrong keys (401)
- api_key mode with no key configured fails safely (503)
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crowd_reliability import runtime as rt
from crowd_reliability.config import settings
from crowd_reliability.api import app
from crowd_reliability.auth import check_api_key
import runtime_server  # noqa: F401 — registers runtime world/session routes


@pytest.fixture
def client(monkeypatch):
    """Fresh TestClient with local_demo auth and mc_v2 models."""
    monkeypatch.setattr(settings, 'auth_mode', 'local_demo')
    monkeypatch.setattr(settings, 'app_api_key', '')
    monkeypatch.setattr(settings, 'model_dir',
                        str(Path(__file__).resolve().parents[1] / 'models/candidates/mc_v2'))
    monkeypatch.setattr(settings, 'app_env', 'test')
    rt.SESSIONS.clear()
    rt.EVALUATION.clear()
    with TestClient(app) as c:
        yield c


class TestLocalDemoAuth:
    """local_demo mode: allows localhost without key, blocks non-localhost."""

    def test_local_demo_localhost_no_key_allowed(self, client):
        """TestClient defaults to 127.0.0.1 — local_demo should allow this."""
        r = client.get('/api/runtime/sessions')
        assert r.status_code == 200

    def test_local_demo_localhost_api_endpoint_allowed(self, client):
        """Even non-health endpoints should work from localhost in local_demo."""
        r = client.get('/api/runtime/sessions')
        assert r.status_code == 200

    def test_check_api_key_local_demo_blocks_non_localhost(self, monkeypatch):
        """Direct unit: local_demo + non-localhost host => 403."""
        monkeypatch.setattr(settings, 'auth_mode', 'local_demo')
        monkeypatch.setattr(settings, 'app_api_key', '')
        from fastapi import HTTPException

        for host in ("192.168.1.1", "10.0.0.1", "172.16.0.1", "203.0.113.5"):
            with pytest.raises(HTTPException) as exc_info:
                check_api_key(None, client_host=host)
            assert exc_info.value.status_code == 403, f"Expected 403 for {host}, got {exc_info.value.status_code}"

    def test_check_api_key_local_demo_allows_localhost(self, monkeypatch):
        """Direct unit: local_demo + localhost host => no exception."""
        monkeypatch.setattr(settings, 'auth_mode', 'local_demo')
        monkeypatch.setattr(settings, 'app_api_key', '')

        for host in ("127.0.0.1", "::1", "localhost"):
            # Should NOT raise
            check_api_key(None, client_host=host)


class TestApiKeyAuth:
    """api_key mode: requires correct key, fails safely when unconfigured."""

    def test_correct_key_allowed(self, monkeypatch):
        monkeypatch.setattr(settings, 'auth_mode', 'api_key')
        monkeypatch.setattr(settings, 'app_api_key', 'secret-key-123')
        check_api_key('secret-key-123')  # Should not raise

    def test_missing_key_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, 'auth_mode', 'api_key')
        monkeypatch.setattr(settings, 'app_api_key', 'secret-key-123')
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            check_api_key(None)
        assert exc_info.value.status_code == 401

    def test_wrong_key_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, 'auth_mode', 'api_key')
        monkeypatch.setattr(settings, 'app_api_key', 'secret-key-123')
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            check_api_key('wrong-key')
        assert exc_info.value.status_code == 401

    def test_unconfigured_key_fails_safely(self, monkeypatch):
        """api_key mode with no app_api_key => 503 (fail safe)."""
        monkeypatch.setattr(settings, 'auth_mode', 'api_key')
        monkeypatch.setattr(settings, 'app_api_key', '')
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            check_api_key('any-key')
        assert exc_info.value.status_code == 503

    def test_api_key_mode_via_http(self, client, monkeypatch):
        """Integration: api_key mode requires correct X-API-Key header."""
        monkeypatch.setattr(settings, 'auth_mode', 'api_key')
        monkeypatch.setattr(settings, 'app_api_key', 'test-key')

        # No key => 401
        assert client.get('/api/runtime/sessions').status_code == 401
        # Wrong key => 401
        assert client.get('/api/runtime/sessions', headers={'X-API-Key': 'wrong'}).status_code == 401
        # Correct key => 200
        assert client.get('/api/runtime/sessions', headers={'X-API-Key': 'test-key'}).status_code == 200
