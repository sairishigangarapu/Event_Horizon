"""
Adapter that ingests the Event_Horizon dataset (data/dataset/) into the existing
record model so the same D[z,t] tree pipeline applies.

Mapping:
  - Zones: grid cells over the 32x18m world; perimeter cells along the exit
    boundary are exit zones (probability_exit_open applicability mask).
  - Labels (reference_states): true per-zone density/speed/inflow/outflow/
    queue_length/exit_open from ground_truth, sampled on the 5s grid.
  - Person-1 observations: per-zone density (track count / area) and speed
    (mean track speed) from profile_A and profile_B trackers, projected to
    world zones via an affine fit on links.csv (which pairs observed tracks
    with known ground-truth coordinates). Two profiles => two independent
    observation sources (useful for conflict/freshness testing).

Observations and labels never include forecasts. Scenario ids are the dataset
run_ids (e.g. normal_flow_001); train/val/test splits come from manifest.csv.
"""
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from crowd_reliability.dataset.zones import (
    apply_affine,
    assign_zone_ids,
    build_grid_zones,
    fit_affine,
    point_to_zone,
    zone_exit_flags,
)
from crowd_reliability.models.schema import ObservationRecord, ReferenceStateRecord, ZoneRecord, ConnectionRecord

DEFAULT_ANCHOR = datetime(2026, 10, 14, 10, 0, 0, tzinfo=timezone.utc)
DEFAULT_EVAL_STEP_S = 5


def _row_to_dict(row) -> dict:
    return {k: row[k] for k in row.index}


def _zone_area(zones: List[dict]) -> Dict[str, float]:
    return {z["zone_id"]: z["area_m2"] for z in zones}


def load_manifest(dataset_dir: str) -> List[dict]:
    with open(os.path.join(dataset_dir, "manifest.csv")) as f:
        return list(csv.DictReader(f))


def _m_per_px(dataset_dir: str, profile: str, width_m: float) -> float:
    with open(os.path.join(dataset_dir, "config_used.json")) as f:
        cfg = json.load(f)
    prof = cfg["profiles"][profile]
    return width_m / prof["image_width_px"]


def _calibrate_x(run_gt: pd.DataFrame, run_links: pd.DataFrame):
    """
    Fit a 1-D affine x_m = slope * gt_ref_x_px + intercept from links rows.

    Only the longitudinal axis (x) is recoverable: the perspective projection
    makes gt_ref_y_px uncorrelated with lateral y_m, so a full 2-D affine is
    degenerate. With rows=1 zoning (full cross-section), lateral position is
    irrelevant — x alone assigns each track to its longitudinal zone.
    """
    gt = run_gt[run_gt["in_scene"]]
    merged = run_links.merge(gt, left_on=["frame", "gt_agent_id"], right_on=["step", "agent_id"])
    if len(merged) < 3 or merged["gt_ref_x_px"].std(ddof=0) == 0:
        return None
    slope, intercept = np.polyfit(merged["gt_ref_x_px"], merged["x_m"], 1)
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return None
    return float(slope), float(intercept)


def _project_x(profile_df: pd.DataFrame, affine_x, x_col: str) -> np.ndarray:
    if affine_x is None or len(profile_df) == 0:
        return None
    slope, intercept = affine_x
    return slope * profile_df[x_col].to_numpy(dtype=float) + intercept


def _frame_for_t(t_s: float, dt: float) -> int:
    return int(round(t_s / dt))


def ingest_dataset(
    dataset_dir: str,
    db_session,
    grid_cols: int = 8,
    grid_rows: int = 1,
    exit_boundary: str = "last_col",
    eval_step_s: int = DEFAULT_EVAL_STEP_S,
    sim_dt_s: float = 0.1,
    anchor: datetime = DEFAULT_ANCHOR,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    """
    Loads zones + observations + reference states from the dataset into the DB.

    Longitudinal zoning (rows=1, full cross-section per cell) is used because the
    dataset's perspective projection recovers world x (flow direction) reliably but
    not lateral y. The downstream (last) cell is the exit zone.
    """
    zones = build_grid_zones(32.0, 18.0, grid_cols, grid_rows, exit_boundary=exit_boundary)
    exit_flags = zone_exit_flags(zones)
    areas = _zone_area(zones)

    width_m = 32.0
    m_per_px = {
        "A": _m_per_px(dataset_dir, "A", width_m),
        "B": _m_per_px(dataset_dir, "B", width_m),
    }

    manifest = load_manifest(dataset_dir)
    manifest_ids = [r["run_id"] for r in manifest]

    train_scenarios: List[str] = []
    val_scenarios: List[str] = []
    test_scenarios: List[str] = []
    scenario_ids_by_split = {"train": train_scenarios, "val": val_scenarios, "test": test_scenarios}

    if clear_existing:
        db_session.query(ObservationRecord).filter(
            ObservationRecord.scenario_id.in_(manifest_ids)
        ).delete(synchronize_session=False)
        db_session.query(ReferenceStateRecord).filter(
            ReferenceStateRecord.scenario_id.in_(manifest_ids)
        ).delete(synchronize_session=False)
        db_session.commit()

    # Persist zones + trivial intra-cell connections.
    for z in zones:
        db_session.merge(ZoneRecord(
            zone_id=z["zone_id"], name=z["name"],
            geometry={"x": z["x"], "y": z["y"], "area_m2": z["area_m2"]},
            zone_type=z["zone_type"], capacity_definition=z["capacity_definition"],
            map_version="grid_v1",
        ))
    for z in zones:
        eid = f"grid_{z['zone_id']}"
        if not db_session.query(ConnectionRecord).get(eid):
            db_session.add(ConnectionRecord(
                edge_id=eid, from_zone=z["zone_id"], to_zone=z["zone_id"],
                width_m=1.0, direction="bidirectional", map_version="grid_v1",
            ))
    db_session.commit()

    for run in manifest:
        run_id = run["run_id"]
        scenario = run["scenario"]
        split = run["split"]
        scenario_ids_by_split[split].append(run_id)

        gt_path = os.path.join(dataset_dir, "ground_truth", split, f"{run_id}_gt_world.csv")
        gt = pd.read_csv(gt_path)
        if "t_s" not in gt.columns:
            gt["t_s"] = gt["step"] * float(sim_dt_s)

        eval_times = np.arange(0, float(gt["t_s"].max()) + 1e-6, eval_step_s)

        # Zone membership of each ground-truth agent at every step (for reference).
        # Longitudinal zoning: clamp lateral y into the grid (y is well-projected
        # only along x; full cross-section is the zone's lateral extent).
        xs = gt["x_m"].to_numpy(dtype=float)
        ys = np.clip(gt["y_m"].to_numpy(dtype=float), 0.0, 18.0 - 1e-6)
        world_xy = np.column_stack([xs, ys])
        gt["zone_id"] = np.where(
            gt["in_scene"].to_numpy(),
            assign_zone_ids(world_xy, zones, width_m, 18.0, grid_cols, grid_rows),
            "",
        )

        # Reference states on the 5s grid (current state at time t).
        ref_records: List[ReferenceStateRecord] = []
        prev_zones_agents = None
        for t_idx, t in enumerate(eval_times):
            frame_t = _frame_for_t(float(t), sim_dt_s)
            snap = gt[gt["step"] == frame_t]
            t_dt = anchor + timedelta(seconds=float(t))
            snap_agents = snap[snap["in_scene"]]
            current_zones_agents = {
                z["zone_id"]: snap_agents[snap_agents["zone_id"] == z["zone_id"]]["agent_id"].tolist()
                for z in zones
            }
            for z in zones:
                in_zone = snap[snap["zone_id"] == z["zone_id"]]
                area = areas[z["zone_id"]]
                agents = in_zone[in_zone["in_scene"]]
                zid = z["zone_id"]

                # density p/m2
                ref_records.append(ReferenceStateRecord(
                    scenario_id=run_id, zone_id=zid, variable="density",
                    value=round(len(agents) / area, 4), unit="p/m2",
                    state_at=t_dt, label_available_at=t_dt,
                ))
                speed_vals = agents["speed_mps"].replace([np.inf, -np.inf], np.nan).dropna()
                ref_records.append(ReferenceStateRecord(
                    scenario_id=run_id, zone_id=zid, variable="speed",
                    value=round(float(speed_vals.mean()) if len(speed_vals) else 0.0, 4),
                    unit="m/s", state_at=t_dt, label_available_at=t_dt,
                ))
                # queue_length: stopped/paused agents in zone
                stopped = agents[agents["paused_or_stopped"].astype(bool) | (agents["speed_mps"] < 0.2)]
                ref_records.append(ReferenceStateRecord(
                    scenario_id=run_id, zone_id=zid, variable="queue_length",
                    value=round(float(len(stopped)), 4), unit="persons",
                    state_at=t_dt, label_available_at=t_dt,
                ))
                # inflow/outflow vs previous snapshot
                if prev_zones_agents is not None:
                    prev_ids = set(prev_zones_agents.get(zid, []))
                    cur_ids = set(current_zones_agents.get(zid, []))
                    inflow = len(cur_ids - prev_ids)
                    outflow = len(prev_ids - cur_ids)
                else:
                    inflow = outflow = 0
                ref_records.append(ReferenceStateRecord(
                    scenario_id=run_id, zone_id=zid, variable="inflow",
                    value=round(inflow / eval_step_s, 4), unit="p/s",
                    state_at=t_dt, label_available_at=t_dt,
                ))
                ref_records.append(ReferenceStateRecord(
                    scenario_id=run_id, zone_id=zid, variable="outflow",
                    value=round(outflow / eval_step_s, 4), unit="p/s",
                    state_at=t_dt, label_available_at=t_dt,
                ))
                # exit_open: applicable only in exit zones, label 1.0 (open)
                if exit_flags[zid]:
                    ref_records.append(ReferenceStateRecord(
                        scenario_id=run_id, zone_id=zid, variable="exit_open",
                        value=1.0, unit="binary", state_at=t_dt, label_available_at=t_dt,
                    ))
                else:
                    ref_records.append(ReferenceStateRecord(
                        scenario_id=run_id, zone_id=zid, variable="exit_open",
                        value=-1.0, unit="binary", state_at=t_dt, label_available_at=t_dt,
                    ))

            prev_zones_agents = current_zones_agents

        # Observations from both profiles (independent sources).
        obs_records: List[ObservationRecord] = []
        for profile in ("A", "B"):
            prof_path = os.path.join(dataset_dir, f"profile_{profile}", split, f"{run_id}.csv")
            links_path = os.path.join(dataset_dir, "links", f"profile_{profile}", split, f"{run_id}_links.csv")
            if not (os.path.exists(prof_path) and os.path.exists(links_path)):
                continue
            prof_df = pd.read_csv(prof_path)
            links_df = pd.read_csv(links_path)
            affine_x = _calibrate_x(gt, links_df)
            if affine_x is None or len(prof_df) == 0:
                continue

            if profile == "A" and "frame" not in prof_df.columns and "frame_index" in prof_df.columns:
                prof_df = prof_df.rename(columns={"frame_index": "frame"})

            x_col = "foot_x_px" if profile == "A" else "anchor_x"
            if x_col not in prof_df.columns:
                continue

            wx = _project_x(prof_df, affine_x, x_col)
            if wx is None:
                continue
            prof_df = prof_df.assign(wx=np.clip(wx, 0.0, 32.0 - 1e-6))
            # rows=1 grid: lateral position is unrecoverable from perspective and
            # irrelevant (full cross-section per cell). Zone by longitudinal x only.
            world_xy = np.column_stack([prof_df["wx"].to_numpy(dtype=float),
                                        np.full(len(prof_df), 9.0)])
            prof_df["zone_id"] = assign_zone_ids(world_xy, zones, width_m, 18.0, grid_cols, grid_rows)

            # Observed speed: profile B carries speed_px_s directly; profile A uses
            # the linked ground-truth speed (px/s -> m/s) when available.
            links_speed_lookup = {}
            if profile == "A" and len(links_df) and "gt_speed_px_s" in links_df.columns:
                pairs = links_df[["frame", "track_id", "gt_speed_px_s"]].copy()
                pairs["frame"] = pairs["frame"].astype(int)
                pairs["track_id"] = pairs["track_id"].astype(int)
                links_speed_lookup = dict(zip(
                    map(tuple, pairs[["frame", "track_id"]].values),
                    pairs["gt_speed_px_s"].values,
                ))

            # Sample observations on the 5s grid frames.
            for t in eval_times:
                frame_t = _frame_for_t(float(t), sim_dt_s)
                slice_df = prof_df[prof_df["frame"] == frame_t]
                t_dt = anchor + timedelta(seconds=float(t))
                rec_at = t_dt
                recv_at = rec_at + timedelta(seconds=1.0)  # small ingestion lag
                for z in zones:
                    zid = z["zone_id"]
                    tracks = slice_df[(slice_df["zone_id"] == zid) & (slice_df["zone_id"] != "")]
                    if tracks.empty:
                        continue
                    # density observation = track count / area (noisy: missing/occluded tracks)
                    obs_records.append(ObservationRecord(
                        report_id=f"o_{profile}_{run_id}_{zid}_{int(t)}",
                        scenario_id=run_id, zone_id=zid, source_id=f"profile_{profile}_track",
                        variable="density", value=round(len(tracks) / areas[zid], 4),
                        unit="p/m2", observed_at=rec_at, published_at=rec_at, received_at=recv_at,
                    ))
                    # speed observation
                    speed_vals = []
                    for _, row in tracks.iterrows():
                        sp = None
                        if profile == "B" and pd.notna(row.get("speed_px_s")):
                            sp = float(row["speed_px_s"])
                        else:
                            sp = links_speed_lookup.get((int(row["frame"]), int(row["track_id"])))
                        if sp is not None and np.isfinite(sp):
                            speed_vals.append(sp * m_per_px[profile])
                    if speed_vals:
                        obs_records.append(ObservationRecord(
                            report_id=f"o_{profile}_{run_id}_{zid}_{int(t)}_spd",
                            scenario_id=run_id, zone_id=zid, source_id=f"profile_{profile}_track",
                            variable="speed", value=round(float(np.mean(speed_vals)), 4),
                            unit="m/s", observed_at=rec_at, published_at=rec_at, received_at=recv_at,
                        ))

        db_session.bulk_save_objects(ref_records)
        db_session.bulk_save_objects(obs_records)
        db_session.commit()

    zones_json = os.path.join(dataset_dir, "grid_zones.json")
    with open(zones_json, "w") as f:
        json.dump({"zones": zones, "zone_exit_flags": exit_flags,
                   "grid": {"cols": grid_cols, "rows": grid_rows}}, f, indent=2)

    split_manifest = os.path.join(dataset_dir, "splits.json")
    with open(split_manifest, "w") as f:
        json.dump({"train": train_scenarios, "val": val_scenarios, "test": test_scenarios}, f, indent=2)

    return {
        "zones_json": zones_json,
        "split_manifest": split_manifest,
        "zone_exit_flags": exit_flags,
        "n_zones": len(zones),
        "scenarios": {"train": train_scenarios, "val": val_scenarios, "test": test_scenarios},
    }


def load_zone_flags(zones_json_path: str) -> Dict[str, bool]:
    with open(zones_json_path) as f:
        data = json.load(f)
    return data["zone_exit_flags"]


def load_split_scenarios(split_manifest_path: str, which: str = "train") -> List[str]:
    with open(split_manifest_path) as f:
        data = json.load(f)
    return data[which]
