"""Shared pieces of the Phase 4 APIs: token verification, durable session files, safe errors."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")


class ConfigError(RuntimeError):
    """Service configuration is missing or invalid; the service refuses to start."""


class TokenVerifier:
    """Accepts `Authorization: Bearer <token>` if SHA-256(token) is in the configured set.
    Only hashes are held server-side; tokens are never logged or echoed."""

    def __init__(self, sha256_hexes: list[str]):
        clean = [h.strip() for h in sha256_hexes if h.strip()]
        if not clean or not all(SHA256_HEX_RE.match(h) for h in clean):
            raise ConfigError("token hash file must contain one or more lowercase SHA-256 hex lines")
        self._hashes = [bytes.fromhex(h) for h in clean]

    @classmethod
    def from_file(cls, path: str | None) -> "TokenVerifier":
        if not path or not Path(path).is_file():
            raise ConfigError("token hash file not configured or not found")
        return cls(Path(path).read_text(encoding="utf-8").splitlines())

    def check(self, request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "authentication required", headers={"WWW-Authenticate": "Bearer"})
        digest = hashlib.sha256(token.strip().encode()).digest()
        if not any(hmac.compare_digest(digest, h) for h in self._hashes):
            raise HTTPException(401, "invalid credentials", headers={"WWW-Authenticate": "Bearer"})


class FileSessionStore:
    """One JSON file per session, replaced atomically (write temp + fsync + os.replace) BEFORE the
    API acknowledges a change, so an acknowledged registration survives a restart."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, sid: str) -> Path:
        if not SESSION_ID_RE.match(sid or ""):
            raise KeyError(sid)
        return self.dir / f"{sid}.json"

    def _write(self, sid: str, doc: dict) -> None:
        path = self._path(sid)
        tmp = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def create(self, doc: dict) -> str:
        sid = secrets.token_urlsafe(24)
        with self._lock:
            self._write(sid, {**doc, "session_id": sid})
        return sid

    def get(self, sid: str) -> dict:
        path = self._path(sid)
        if not path.is_file():
            raise KeyError(sid)
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, sid: str, fn) -> dict:
        """fn(doc) -> new doc (may raise HTTPException to abort without writing)."""
        with self._lock:
            doc = fn(self.get(sid))
            self._write(sid, doc)
            return doc


def install_safe_errors(app: FastAPI, log: logging.Logger) -> None:
    """Errors never echo request bodies, stack traces, storage paths, digests or credentials."""

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        fields = sorted({".".join(str(p) for p in e.get("loc", ()) if not isinstance(p, int)) for e in exc.errors()})
        return JSONResponse({"detail": "invalid request", "fields": fields}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.error("unhandled %s on %s", type(exc).__name__, request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)


def project(record: dict, fields) -> dict:
    """Explicit response allowlist: only the named fields leave the service."""
    return {k: record[k] for k in fields if k in record}
