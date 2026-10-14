"""
Tests for the Event_Horizon dataset adapter (zones, projection, ingestion).

Uses a tiny synthetic dataset in a temp dir so the test is deterministic and
independent of the real (large) dataset files.
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from crowd_reliability.dataset.adapter import ingest_dataset, load_zone_flags
from crowd_reliability.dataset.zones import build_grid_zones, zone_exit_flags
from crowd_reliability.models.schema import Base, ObservationRecord, ReferenceStateRecord

GRID_COLS, GRID_ROWS = 8, 1


def _write_synthetic_dataset(dst: str):
    os.makedirs(dst, exist_ok=True)
    world_w, world_h = 32.0, 18.0
    # config: image widths drive px->m scaling
    with open(os.path.join(dst, "config_used.json"), "w") as f:
        json.dump({"version": "1.0.0", "world": {"width_m": world_w, "height_m": world_h},
                   "profiles": {"A": {"schema": "A", "image_width_px": 940},
                                "B": {"schema": "B", "image_width_px": 1280}}}, f)

    # agents at known x_m; links map foot_x_px -> x_m with slope 2 (x_m = 2*px).
    agents = [
        # agent_id, x_m, foot_x_px
        (0, 20.0, 10.0),   # longitudinal cell cx = 20/4 = 5 -> g_5_0
        (1, 12.0, 6.0),    # cx=3 -> g_3_0
        (2, 28.0, 14.0),   # cx=7 -> g_7_0 (exit)
    ]
    cols = ["scenario_id", "step", "t_s", "agent_id", "x_m", "y_m", "vx_mps", "vy_mps",
            "speed_mps", "heading_deg", "ax_mps2", "ay_mps2", "in_scene", "role", "paused_or_stopped"]
    gt_rows = []
    for agent_id, x_m, px in agents:
        for step in range(3):
            gt_rows.append(["tiny", step, step * 0.1, agent_id, x_m, 9.0, 0.5, 0.0,
                            0.5, 0.0, 0.0, 0.0, True, "walker", False])
    os.makedirs(os.path.join(dst, "ground_truth", "test"), exist_ok=True)
    pd.DataFrame(gt_rows, columns=cols).to_csv(
        os.path.join(dst, "ground_truth", "test", "tiny_gt_world.csv"), index=False)

    # Profile A tracks: foot_x_px = x_m/2 (recover via slope 2)
    prof_a_rows = []
    for agent_id, x_m, px in agents:
        for step in range(3):
            prof_a_rows.append([step, step * 0.1, agent_id, 0.9, 0, 0, 0, 0, 0, 0, px, px])
    # columns: frame_index,time_s,track_id,confidence,x1,y1,x2,y2,center_x,center_y,foot_x,foot_y
    os.makedirs(os.path.join(dst, "profile_A", "test"), exist_ok=True)
    pd.DataFrame(prof_a_rows, columns=[
        "frame_index", "time_s", "track_id", "confidence", "x1", "y1", "x2", "y2",
        "center_x", "center_y", "foot_x_px", "foot_y_px"]).to_csv(
        os.path.join(dst, "profile_A", "test", "tiny.csv"), index=False)

    # Profile B tracks: anchor_x columns, speed directly
    prof_b_rows = []
    for agent_id, x_m, px in agents:
        for step in range(3):
            prof_b_rows.append([step, step * 0.1, agent_id, 0.9, 0, 0, 0, 0, px, px, 0.5, "down"])
    os.makedirs(os.path.join(dst, "profile_B", "test"), exist_ok=True)
    pd.DataFrame(prof_b_rows, columns=[
        "frame", "time_s", "track_id", "confidence", "x1", "y1", "x2", "y2",
        "anchor_x", "anchor_y", "speed_px_s", "direction"]).to_csv(
        os.path.join(dst, "profile_B", "test", "tiny.csv"), index=False)

    # Profile A links: gt_ref_x_px = foot_x_px, gt_speed_px_s = 1.0
    link_rows = []
    for agent_id, x_m, px in agents:
        for step in range(3):
            link_rows.append([step, agent_id, agent_id, 1.0, 0, px, 0.0, 1.0])
    for prof in ("A", "B"):
        os.makedirs(os.path.join(dst, "links", f"profile_{prof}", "test"), exist_ok=True)
        pd.DataFrame(link_rows, columns=[
            "frame", "track_id", "gt_agent_id", "visible_ratio", "occlusion",
            "gt_ref_x_px", "gt_ref_y_px", "gt_speed_px_s"]).to_csv(
            os.path.join(dst, "links", f"profile_{prof}", "test", "tiny_links.csv"), index=False)

    # manifest: single run, test split
    with open(os.path.join(dst, "manifest.csv"), "w") as f:
        f.write("run_id,scenario,run_index,split,master_seed,n_agents,recorded_duration_s\n")
        f.write("tiny,tiny,0,test,26,3,0.3\n")


def test_grid_exit_rule_marks_only_downstream_cell():
    zones = build_grid_zones(32.0, 18.0, GRID_COLS, GRID_ROWS, exit_boundary="last_col")
    flags = zone_exit_flags(zones)
    assert sum(flags.values()) == 1
    assert flags["g_7_0"] is True
    assert flags["g_0_0"] is False


def _ingest_and_db(tmp_path):
    dst = str(tmp_path / "ds")
    _write_synthetic_dataset(dst)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    summary = ingest_dataset(dst, db, grid_cols=GRID_COLS, grid_rows=GRID_ROWS,
                             exit_boundary="last_col", eval_step_s=5)
    return db, summary


def test_dataset_ingest_produces_zones_obs_refs(tmp_path):
    db, summary = _ingest_and_db(tmp_path)

    assert summary["n_zones"] == GRID_COLS
    assert summary["scenarios"]["test"] == ["tiny"]

    zones = db.query(ReferenceStateRecord).count()
    refs = db.query(ReferenceStateRecord).all()
    assert len(refs) > 0
    assert {r.variable for r in refs} == {
        "density", "speed", "inflow", "outflow", "queue_length", "exit_open"}

    obs = db.query(ObservationRecord).all()
    assert len(obs) > 0
    sources = {o.source_id for o in obs}
    assert "profile_A_track" in sources and "profile_B_track" in sources

    # exit_open only labelled for exit zones
    exit_open_zones = {r.zone_id for r in refs if r.variable == "exit_open" and r.value >= 0}
    assert exit_open_zones == {"g_7_0"}


def test_dataset_projection_assigns_agents_to_correct_zones(tmp_path):
    db, _ = _ingest_and_db(tmp_path)
    # density observations exist in g_5_0, g_3_0, g_7_0 (from synthetic x_m of 20, 12, 28)
    dens_obs = (db.query(ObservationRecord.zone_id).filter(
        ObservationRecord.variable == "density").all())
    observed_zones = {z[0] for z in dens_obs}
    for expected in ("g_3_0", "g_5_0", "g_7_0"):
        assert expected in observed_zones, f"Projected to {observed_zones}"
