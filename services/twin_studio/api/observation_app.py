"""Observation API (Phase 4): serving-safe, read-only access to published observation bundles.

Watermark: every report/snapshot request is evaluated against a server-held REPLAY watermark
(simulated seconds since the run epoch). Replays are created and advanced only with the
playback-control token (held by the operator/evaluator harness, never by the forecaster). The
forecaster receives a replay_id and can read up to the watermark, never beyond it. `as_of` is kept
for compatibility but must be <= the watermark; it can only look back, never forward.

Storage: observation-reader credentials (eh-catalog + eh-observations only). Responses use explicit
field allowlists. Never served: reference states, labels, incident schedules, reference manifests,
input digests, reference paths, any credential or key material.

Run: uvicorn --factory services.twin_studio.api.observation_app:app_from_env
"""
from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services.twin_studio.api.common import (ConfigError, FileSessionStore, TokenVerifier, install_safe_errors,
                                             project)
from services.twin_studio.api.timeutil import at, seconds_since
from services.twin_studio.bundle import OBS_RECORD_FIELDS, RUN_ID_RE
from services.twin_studio.observations import CAMERA_VARIABLES, GATE_VARIABLES, UNITS
from services.twin_studio.storage.readers import IncompatibleSchema, ObservationStore, RunUnavailable

log = logging.getLogger("eh_twin.observation_api")
SERVICE = "eh-twin-observation-api"
RUN_LIST_FIELDS = ("run_id", "scenario_id", "environment_id", "map_version", "epoch_utc", "sample_seconds",
                   "observation_schema_version")
REPORT_FIELDS = tuple(f for f in OBS_RECORD_FIELDS)
VENUE_FIELDS = ("map_version", "environment_id", "coordinate_system", "zones", "obstacles", "connections")
REPLAY_FIELDS = ("replay_id", "run_id", "watermark_second", "watermark_at", "max_watermark_second", "sample_seconds")
MAX_ADVANCE_STEPS = 12


class ReplayCreate(BaseModel):
    run_id: str = Field(pattern=RUN_ID_RE.pattern)


class Advance(BaseModel):
    steps: int = Field(ge=1, le=MAX_ADVANCE_STEPS)


def create_app(store: ObservationStore, replays: FileSessionStore, control: TokenVerifier) -> FastAPI:
    app = FastAPI(title="Event Horizon Twin Studio - Observation API", version="phase4",
                  docs_url=None, redoc_url=None, openapi_url=None)
    install_safe_errors(app, log)

    def control_auth(request: Request):
        control.check(request)

    def run_or_404(run_id: str):
        try:
            return store.load(run_id)
        except IncompatibleSchema:
            raise HTTPException(409, "incompatible schema version") from None
        except RunUnavailable:
            raise HTTPException(404, "run not available") from None

    def replay_view(doc: dict) -> dict:
        return project(doc, REPLAY_FIELDS)

    def replay_for(run_id: str, replay_id: str) -> dict:
        try:
            doc = replays.get(replay_id)
        except KeyError:
            raise HTTPException(404, "replay not found") from None
        if doc["run_id"] != run_id:
            raise HTTPException(404, "replay not found")
        return doc

    def cutoff(run, doc: dict, as_of: str | None) -> int:
        """Effective observation time: the watermark, or an earlier validated as_of."""
        if as_of is None:
            return doc["watermark_second"]
        try:
            t = seconds_since(run.manifest["epoch_utc"], as_of)
        except ValueError:
            raise HTTPException(422, "invalid as_of timestamp") from None
        if t < 0:
            raise HTTPException(422, "as_of precedes the run epoch")
        if t > doc["watermark_second"]:
            raise HTTPException(409, "as_of is beyond the replay watermark")
        return t

    @app.get("/health")
    def health():
        return {"status": "ok", "service": SERVICE}

    @app.get("/runs")
    def runs():
        out = []
        for rid in store.list_run_ids():
            try:
                run = store.load(rid)           # only fully verified runs are listed
            except (RunUnavailable, IncompatibleSchema):
                continue
            out.append(project(run.catalog, RUN_LIST_FIELDS))
        return {"runs": out}

    @app.get("/runs/{run_id}/venue")
    def venue(run_id: str):
        return project(run_or_404(run_id).venue, VENUE_FIELDS)

    @app.post("/replays", status_code=201, dependencies=[Depends(control_auth)])
    def create_replay(body: ReplayCreate):
        run = run_or_404(body.run_id)
        doc = {"run_id": run.run_id, "watermark_second": 0, "watermark_at": at(run.manifest["epoch_utc"], 0),
               "max_watermark_second": run.max_received_second, "sample_seconds": run.manifest["sample_seconds"]}
        sid = replays.create(doc)
        return replay_view({**doc, "replay_id": sid})

    @app.post("/replays/{replay_id}/advance", dependencies=[Depends(control_auth)])
    def advance(replay_id: str, body: Advance):
        try:
            doc = replays.get(replay_id)
        except KeyError:
            raise HTTPException(404, "replay not found") from None
        run = run_or_404(doc["run_id"])

        def step(d):
            if d["watermark_second"] >= d["max_watermark_second"]:
                raise HTTPException(409, "replay already at its final watermark")
            t = min(d["watermark_second"] + body.steps * d["sample_seconds"], d["max_watermark_second"])
            return {**d, "watermark_second": t, "watermark_at": at(run.manifest["epoch_utc"], t)}
        return replay_view({**replays.update(replay_id, step), "replay_id": replay_id})

    @app.get("/replays/{replay_id}")
    def get_replay(replay_id: str):
        try:
            return replay_view({**replays.get(replay_id), "replay_id": replay_id})
        except KeyError:
            raise HTTPException(404, "replay not found") from None

    @app.get("/runs/{run_id}/reports")
    def reports(run_id: str, replay_id: str = Query(..., min_length=20, max_length=64),
                as_of: str | None = Query(None, max_length=40)):
        run = run_or_404(run_id)
        doc = replay_for(run_id, replay_id)
        t = cutoff(run, doc, as_of)
        visible = [project(r, REPORT_FIELDS) for r in run.reports if r["_received_second"] <= t]
        return {"run_id": run_id, "watermark_second": doc["watermark_second"], "as_of_second": t,
                "as_of": at(run.manifest["epoch_utc"], t), "count": len(visible), "reports": visible}

    @app.get("/runs/{run_id}/snapshot")
    def snapshot(run_id: str, replay_id: str = Query(..., min_length=20, max_length=64),
                 as_of: str | None = Query(None, max_length=40)):
        run = run_or_404(run_id)
        doc = replay_for(run_id, replay_id)
        t = cutoff(run, doc, as_of)
        coverage = run.manifest["source_coverage"]
        camera_zones = set(coverage.get("sim_camera_A", [])) | set(coverage.get("sim_camera_B", []))
        latest: dict[tuple, dict] = {}
        for r in run.reports:
            if r["_received_second"] > t:
                continue
            k = (r["zone_id"], r["variable"])
            best = latest.get(k)
            if best is None or (r["_observed_second"], r["_received_second"], r["report_id"]) > \
                    (best["_observed_second"], best["_received_second"], best["report_id"]):
                latest[k] = r
        entries = []
        for zone in run.venue["zones"]:
            zid, is_gate = zone["zone_id"], zone["zone_type"] == "exit"
            for variable in CAMERA_VARIABLES + GATE_VARIABLES:
                applicable = is_gate if variable in GATE_VARIABLES else True
                covered = applicable and (is_gate if variable in GATE_VARIABLES else zid in camera_zones)
                r = latest.get((zid, variable)) if applicable else None
                entries.append({
                    "zone_id": zid, "variable": variable, "unit": UNITS[variable],
                    "applicable": applicable, "covered": covered, "present": r is not None,
                    "value": r["value"] if r else None,
                    "source_id": r["source_id"] if r else None, "report_id": r["report_id"] if r else None,
                    "observed_at": r["observed_at"] if r else None, "received_at": r["received_at"] if r else None,
                    "age_seconds": (t - r["_observed_second"]) if r else None,
                })
        return {"run_id": run_id, "watermark_second": doc["watermark_second"], "as_of_second": t,
                "as_of": at(run.manifest["epoch_utc"], t), "entries": entries}

    return app


def app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    from services.twin_studio.storage.objects import s3_store_from_env
    control = TokenVerifier.from_file(os.environ.get("EH_OBS_CONTROL_TOKEN_SHA256_FILE"))
    replay_dir = os.environ.get("EH_OBS_REPLAY_DIR")
    if not replay_dir:
        raise ConfigError("EH_OBS_REPLAY_DIR is required")
    return create_app(ObservationStore(s3_store_from_env("EH_OBS_READER")), FileSessionStore(replay_dir), control)
