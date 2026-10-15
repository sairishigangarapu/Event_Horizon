"""Reference API (Phase 4): authenticated evaluation sessions over hidden reference truth.

Every route except /health requires the evaluator bearer token (only its SHA-256 is held here).
Storage: reference-reader credentials (eh-reference + approved eh-catalog markers only).

Server-side watermark (per session, in simulated seconds since the run epoch):
  * created at current_second = 0; only ever moves forward by POST /advance with 1..12 grid steps,
    capped at final_second = last_state_second + label_delay_seconds;
  * predictions are accepted only for issue_second == current_second (no backdating, no future
    issue times), horizon in the run's label horizons, target_second <= last_state_second, at most one
    prediction per (issue, horizon, zone, variable); they are written durably before the response;
  * a label is released only if label_available_at (= target_at + label_delay_seconds, from the
    reference bundle) <= the session watermark AND the prediction was registered at a watermark earlier
    than label_available_at;
  * a reference state is revealed only if state_at + label_delay_seconds <= the session watermark.
No client-supplied timestamp or request wall-clock time is ever used for eligibility.

Run: uvicorn --factory services.twin_studio.api.reference_app:app_from_env
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services.twin_studio.api.common import ConfigError, FileSessionStore, TokenVerifier, install_safe_errors, project
from services.twin_studio.api.timeutil import at, seconds_since
from services.twin_studio.bundle import RUN_ID_RE
from services.twin_studio.storage.readers import ReferenceStore, RunUnavailable

log = logging.getLogger("eh_twin.reference_api")
SERVICE = "eh-twin-reference-api"
MAX_ADVANCE_STEPS = 12
MAX_PREDICTIONS_PER_REQUEST = 5000
LABEL_FIELDS = ("zone_id", "variable", "unit", "horizon_seconds", "issue_second", "issued_at", "target_second",
                "target_at", "label_available_at", "value", "value_valid", "applicable", "measurement",
                "target_interval_start", "target_interval_end")
STATE_FIELDS = ("zone_id", "variable", "unit", "value", "value_valid", "applicable", "measurement",
                "simulation_second", "state_at", "label_available_at", "interval_start", "interval_end")
SESSION_FIELDS = ("session_id", "run_id", "current_second", "watermark_at", "sample_seconds", "horizons_seconds",
                  "label_delay_seconds", "target_variables", "last_state_second", "final_second",
                  "max_advance_steps")


class SessionCreate(BaseModel):
    run_id: str = Field(pattern=RUN_ID_RE.pattern)


class Prediction(BaseModel):
    zone_id: str = Field(min_length=1, max_length=64)
    variable: str = Field(min_length=1, max_length=64)
    horizon_seconds: int
    value: float


class PredictionBatch(BaseModel):
    issue_second: int
    predictions: list[Prediction] = Field(min_length=1, max_length=MAX_PREDICTIONS_PER_REQUEST)


class Advance(BaseModel):
    steps: int = Field(ge=1, le=MAX_ADVANCE_STEPS)


def create_app(store: ReferenceStore, sessions: FileSessionStore, evaluator: TokenVerifier) -> FastAPI:
    app = FastAPI(title="Event Horizon Twin Studio - Reference API", version="phase4",
                  docs_url=None, redoc_url=None, openapi_url=None)
    install_safe_errors(app, log)

    def auth(request: Request):
        evaluator.check(request)

    def run_or_404(run_id: str):
        try:
            return store.load(run_id)
        except RunUnavailable:
            raise HTTPException(404, "run not available") from None

    def session_or_404(sid: str) -> dict:
        try:
            return sessions.get(sid)
        except KeyError:
            raise HTTPException(404, "session not found") from None

    def view(doc: dict) -> dict:
        out = project(doc, SESSION_FIELDS)
        out["registered_predictions"] = len(doc.get("predictions", []))
        return out

    def available_second(run, iso: str) -> int:
        return seconds_since(run.manifest["epoch_utc"], iso)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": SERVICE}

    @app.post("/sessions", status_code=201, dependencies=[Depends(auth)])
    def create_session(body: SessionCreate):
        run = run_or_404(body.run_id)
        doc = {"run_id": run.run_id, "current_second": 0, "watermark_at": at(run.manifest["epoch_utc"], 0),
               "sample_seconds": run.sample_seconds, "horizons_seconds": run.horizons_seconds,
               "label_delay_seconds": run.label_delay_seconds, "target_variables": run.target_variables,
               "last_state_second": run.last_second, "final_second": run.last_second + run.label_delay_seconds,
               "max_advance_steps": MAX_ADVANCE_STEPS, "predictions": []}
        sid = sessions.create(doc)
        return view({**doc, "session_id": sid})

    @app.get("/sessions/{sid}", dependencies=[Depends(auth)])
    def get_session(sid: str):
        return view(session_or_404(sid))

    @app.post("/sessions/{sid}/predictions", status_code=201, dependencies=[Depends(auth)])
    def register(sid: str, body: PredictionBatch):
        run = run_or_404(session_or_404(sid)["run_id"])

        def add(doc):
            t = doc["current_second"]
            if body.issue_second != t:
                raise HTTPException(409, "issue_second must equal the session's current simulation time")
            existing = {(p["issue_second"], p["horizon_seconds"], p["zone_id"], p["variable"]) for p in doc["predictions"]}
            new = []
            for p in body.predictions:
                if not math.isfinite(p.value):
                    raise HTTPException(422, "prediction values must be finite")
                if p.horizon_seconds not in run.horizons_seconds:
                    raise HTTPException(422, "horizon is not a configured label horizon")
                if p.variable not in run.target_variables or p.zone_id not in run.zone_ids:
                    raise HTTPException(422, "unknown zone or target variable")
                target = t + p.horizon_seconds
                if target > run.last_second:
                    raise HTTPException(422, "target lies beyond the run timeline")
                key = (t, p.horizon_seconds, p.zone_id, p.variable)
                if key in existing:
                    raise HTTPException(409, "a prediction for this issue time, horizon, zone and variable exists")
                label = run.labels[key]
                avail = available_second(run, label["label_available_at"])
                if avail <= t:     # cannot happen with delay > 0; guards against any backdating path
                    raise HTTPException(409, "label already available; registration too late")
                existing.add(key)
                new.append({"issue_second": t, "horizon_seconds": p.horizon_seconds, "zone_id": p.zone_id,
                            "variable": p.variable, "value": p.value, "target_second": target,
                            "label_available_second": avail, "registered_at_second": t})
            return {**doc, "predictions": doc["predictions"] + new}

        doc = sessions.update(sid, add)
        return {"session_id": sid, "issue_second": doc["current_second"], "registered": len(body.predictions),
                "registered_total": len(doc["predictions"])}

    @app.post("/sessions/{sid}/advance", dependencies=[Depends(auth)])
    def advance(sid: str, body: Advance):
        run = run_or_404(session_or_404(sid)["run_id"])

        def step(doc):
            if doc["current_second"] >= doc["final_second"]:
                raise HTTPException(409, "session already at its final watermark")
            t = min(doc["current_second"] + body.steps * doc["sample_seconds"], doc["final_second"])
            return {**doc, "current_second": t, "watermark_at": at(run.manifest["epoch_utc"], t)}
        return view(sessions.update(sid, step))

    def released(doc: dict, run) -> tuple[list[dict], int]:
        out, pending = [], 0
        for p in doc["predictions"]:
            eligible = (p["label_available_second"] <= doc["current_second"]
                        and p["registered_at_second"] < p["label_available_second"])
            if not eligible:
                pending += 1
                continue
            label = run.labels[(p["issue_second"], p["horizon_seconds"], p["zone_id"], p["variable"])]
            err = abs(p["value"] - label["value"]) if label["value_valid"] and label["applicable"] else None
            out.append({"prediction": {k: p[k] for k in ("issue_second", "horizon_seconds", "zone_id", "variable",
                                                          "value")},
                        "label": project(label, LABEL_FIELDS), "absolute_error": err})
        return out, pending

    @app.get("/sessions/{sid}/labels", dependencies=[Depends(auth)])
    def labels(sid: str):
        doc = session_or_404(sid)
        run = run_or_404(doc["run_id"])
        out, pending = released(doc, run)
        return {"session_id": sid, "current_second": doc["current_second"], "released": out, "pending": pending}

    @app.get("/sessions/{sid}/reference-states", dependencies=[Depends(auth)])
    def reference_state(sid: str, simulation_second: int = Query(..., ge=0),
                        zone_id: str = Query(..., max_length=64), variable: str = Query(..., max_length=64)):
        doc = session_or_404(sid)
        run = run_or_404(doc["run_id"])
        rec = run.states.get((simulation_second, zone_id, variable))
        if rec is None:
            raise HTTPException(404, "no reference state for that instant, zone and variable")
        if available_second(run, rec["label_available_at"]) > doc["current_second"]:
            raise HTTPException(403, "reference state not yet available at the session watermark")
        return {"session_id": sid, "current_second": doc["current_second"], "state": project(rec, STATE_FIELDS)}

    @app.get("/sessions/{sid}/validation", dependencies=[Depends(auth)])
    def session_validation(sid: str):
        doc = session_or_404(sid)
        run = run_or_404(doc["run_id"])
        out, pending = released(doc, run)
        groups = defaultdict(list)
        for item in out:
            if item["absolute_error"] is not None:
                groups[(item["prediction"]["variable"], item["prediction"]["horizon_seconds"])].append(item["absolute_error"])
        metrics = [{"variable": v, "horizon_seconds": h, "n": len(e), "mae": sum(e) / len(e)}
                   for (v, h), e in sorted(groups.items())]
        return {"session_id": sid, "current_second": doc["current_second"], "released": len(out),
                "pending": pending, "metrics": metrics}

    @app.get("/runs/{run_id}/validation", dependencies=[Depends(auth)])
    def run_validation(run_id: str):
        checks, run = store.verify(run_id)
        return {"run_id": run_id, "verified": run is not None,
                "checks": [{"check": name, "passed": ok} for name, ok in checks]}

    return app


def app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    from services.twin_studio.storage.objects import s3_store_from_env
    evaluator = TokenVerifier.from_file(os.environ.get("EH_REF_TOKEN_SHA256_FILE"))
    session_dir = os.environ.get("EH_REF_SESSION_DIR")
    if not session_dir:
        raise ConfigError("EH_REF_SESSION_DIR is required")
    return create_app(ReferenceStore(s3_store_from_env("EH_REF_READER")), FileSessionStore(session_dir), evaluator)
