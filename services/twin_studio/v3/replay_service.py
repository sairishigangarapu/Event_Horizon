"""HTTP boundary for trusted replay (Phase D).

Routes (nothing else exists; unknown paths are 404):
  POST /operator/sessions/{session_id}/advance   operator token; moves the watermark one step
  GET  /operator/sessions/{session_id}/status    operator token; watermark and end
  GET  /model/sessions/{session_id}/snapshot     model token; the current authorized snapshot

The model endpoint has no time parameter. Any query parameter is rejected with 400, and request
headers never influence the cutoff, so a client cannot request a future watermark. The model token
cannot advance a session. There are no static-file, render, manifest, scene, download or
hidden-truth routes. Tokens are compared by SHA-256 digest in constant time.

Enforced by this service: cutoff, deduplication per model client, token roles, route surface.
Not enforced here (deployment responsibility): network placement and OS/container isolation so the
model process cannot read outputs/ or reach the studio port 8000.
"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import FastAPI, HTTPException, Request

from services.twin_studio.v3.replay import ModelView, OperatorKey, ReplayController


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class _Session:
    def __init__(self, ctrl: ReplayController, key: OperatorKey):
        self.ctrl, self.key = ctrl, key
        self.views: dict[str, ModelView] = {}


def create_app(*, operator_token_sha256: str, model_token_sha256: str) -> tuple[FastAPI, dict]:
    """Returns (app, registry). The operator registers sessions with registry[session_id] = (ctrl, key)."""
    app = FastAPI(title="Twin v3 trusted replay", docs_url=None, redoc_url=None, openapi_url=None)
    sessions: dict[str, _Session] = {}

    class Registry(dict):
        def __setitem__(self, sid, value):
            ctrl, key = value
            sessions[sid] = _Session(ctrl, key)
            super().__setitem__(sid, True)

    registry = Registry()

    def _auth(request: Request, expected: str) -> str:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        digest = _digest(header[7:])
        if not hmac.compare_digest(digest, expected):
            raise HTTPException(403, "token not valid for this route")
        return digest

    def _session(sid: str) -> _Session:
        if sid not in sessions:
            raise HTTPException(404, "unknown session")
        return sessions[sid]

    @app.post("/operator/sessions/{sid}/advance")
    def advance(sid: str, request: Request):
        _auth(request, operator_token_sha256)
        s = _session(sid)
        if s.ctrl.finished:
            raise HTTPException(409, "replay already finished")
        return {"as_of_s": s.ctrl.advance(s.key)}

    @app.get("/operator/sessions/{sid}/status")
    def status(sid: str, request: Request):
        _auth(request, operator_token_sha256)
        s = _session(sid)
        return {"as_of_s": s.ctrl.watermark_s, "finished": s.ctrl.finished}

    @app.get("/model/sessions/{sid}/snapshot")
    def snapshot(sid: str, request: Request):
        digest = _auth(request, model_token_sha256)
        if request.query_params:
            raise HTTPException(400, "this endpoint takes no parameters; the replay time is set by the operator")
        s = _session(sid)
        view = s.views.setdefault(digest, s.ctrl.model_view())
        snap = view.snapshot()
        return {
            "as_of_s": snap.as_of_s,
            "new_report_ids": list(snap.new_report_ids),
            "measurements": [m.__dict__ for m in snap.measurements],
            "notices": [n.to_dict() for n in snap.notices],
        }

    return app, registry
