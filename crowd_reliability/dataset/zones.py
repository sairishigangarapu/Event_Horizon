"""
Zoning for the Event_Horizon dataset (continuous 32x18m agent world).

Zones are a regular grid over [0, W) x [0, H). Perimeter cells along the configured
exit boundary are designated exit zones, giving the probability_exit_open channel an
applicability mask (exit cells only). Zone ids are stable strings so the same
D[z,t] vector contract (6 named channels + masks) applies as for the default venue.
"""
from typing import Any, Dict, List, Tuple

import numpy as np


def build_grid_zones(
    width_m: float,
    height_m: float,
    cols: int,
    rows: int,
    exit_boundary: str = "last_col",
) -> List[Dict[str, Any]]:
    """Partition the world into grid cells.

    exit_boundary: 'last_col' (default, downstream egress cell), 'last_row',
    'bottom', 'top', 'left', 'right', or '' for no exit zones.
    """
    cw = width_m / cols
    ch = height_m / rows
    zones: List[Dict[str, Any]] = []
    for cy in range(rows):
        for cx in range(cols):
            x0 = cx * cw
            y0 = cy * ch
            zid = f"g_{cx}_{cy}"
            is_exit = False
            if exit_boundary == "last_col" and cx == cols - 1:
                is_exit = True
            elif exit_boundary == "last_row" and cy == rows - 1:
                is_exit = True
            elif exit_boundary == "bottom" and cy == rows - 1:
                is_exit = True
            elif exit_boundary == "top" and cy == 0:
                is_exit = True
            elif exit_boundary == "left" and cx == 0:
                is_exit = True
            elif exit_boundary == "right" and cx == cols - 1:
                is_exit = True
            zones.append({
                "zone_id": zid,
                "name": f"grid_{cx}_{cy}",
                "zone_type": "exit" if is_exit else "corridor",
                "capacity_definition": cw * ch * 4.0,
                "area_m2": cw * ch,
                "has_exit": is_exit,
                "x": x0 + cw / 2,
                "y": y0 + ch / 2,
                "x0": x0, "y0": y0, "x1": x0 + cw, "y1": y0 + ch,
            })
    return zones


def zone_exit_flags(zones: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {z["zone_id"]: bool(z.get("has_exit", False)) for z in zones}


def point_to_zone(x: float, y: float, zones: List[Dict[str, Any]]) -> str:
    """Returns zone_id for a world (x,y), or '' if outside every cell."""
    for z in zones:
        if z["x0"] <= x < z["x1"] and z["y0"] <= y < z["y1"]:
            return z["zone_id"]
    return ""


def assign_zone_ids(
    world_xy: np.ndarray,
    zones: List[Dict[str, Any]],
    width_m: float,
    height_m: float,
    cols: int,
    rows: int,
) -> np.ndarray:
    """
    Vectorized world (x,y) -> zone_id. Returns array of zone_id strings (''),
    '' for points outside the grid. zones must be ordered as build_grid_zones
    outputs (cy-major: index = cy*cols + cx).
    """
    world_xy = np.asarray(world_xy)
    n = world_xy.shape[0]
    ids = np.array([""] * n, dtype=object)
    if n == 0:
        return ids
    cw = width_m / cols
    ch = height_m / rows
    xs = world_xy[:, 0]
    ys = world_xy[:, 1]
    cx = np.floor(xs / cw).astype(int)
    cy = np.floor(ys / ch).astype(int)
    valid = (cx >= 0) & (cx < cols) & (cy >= 0) & (cy < rows) & np.isfinite(xs) & np.isfinite(ys)
    idx = cy * cols + cx
    grid_ids = np.array([z["zone_id"] for z in zones], dtype=object)
    ids[valid] = grid_ids[idx[valid]]
    return ids


def fit_affine(px: np.ndarray, world: np.ndarray) -> np.ndarray:
    """
    Least-squares affine [x_m, y_m] = A @ [px, py, 1].
    px: (N,2) pixel coords, world: (N,2) metres. Returns 2x3 transform.
    """
    A = np.linalg.lstsq(
        np.column_stack([px, np.ones(len(px))]),
        world,
        rcond=None,
    )[0]
    return A.reshape(2, 3)


def apply_affine(A: np.ndarray, px: np.ndarray) -> np.ndarray:
    pts = np.column_stack([px, np.ones(len(px))])
    return (A @ pts.T).T
