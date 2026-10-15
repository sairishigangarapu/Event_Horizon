"""Zone model for the Twin Studio venue (twin_venue_v1).

Assigns a simulated agent position (Twin Studio canvas units) to exactly one of:
  - a modelled zone (status "zone"),
  - a non-walkable obstacle (status "obstacle", e.g. the pond), or
  - nowhere in the modelled venue (status "outside").

Rules (twin_venue_v1, approved Phase 0 decisions):
  - rectangles are half-open, [x, x+w) x [y, y+d), so a point on a shared boundary
    belongs to exactly one zone;
  - a gate zone takes priority over the host area it overlaps;
  - an obstacle takes priority over the area it sits in (it is not walkable);
  - agents that are not in any zone or obstacle are counted separately, never dropped.

Physical connections in the venue file are walkable links between zones. They are not
reverse message-passing edges of any graph model, and route completion is not a venue exit.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

VENUE_V1_PATH = Path(__file__).resolve().parent / "venue" / "twin_venue_v1.json"

STATUS_ZONE = "zone"
STATUS_OBSTACLE = "obstacle"
STATUS_OUTSIDE = "outside"
OUTSIDE_KEY = "outside_modelled_zones"
OBSTACLE_KEY = "inside_obstacle"


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    d: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.d


@dataclass(frozen=True)
class Assignment:
    status: str               # STATUS_ZONE | STATUS_OBSTACLE | STATUS_OUTSIDE
    zone_id: str | None       # set only when status == STATUS_ZONE
    obstacle_id: str | None   # set only when status == STATUS_OBSTACLE


def load_venue(path: str | Path = VENUE_V1_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _rect(geometry: dict[str, Any]) -> Rect:
    if geometry.get("type") != "Rectangle" or geometry.get("units") != "canvas_units":
        raise ValueError(f"Unsupported zone geometry: {geometry}")
    return Rect(geometry["x"], geometry["y"], geometry["w"], geometry["d"])


class ZoneAssigner:
    """Point-in-zone classification with gate > obstacle > host-area priority."""

    def __init__(self, venue: dict[str, Any] | None = None):
        self.venue = venue if venue is not None else load_venue()
        zones = self.venue["zones"]
        ids = [z["zone_id"] for z in zones]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate zone_id in venue")
        self.zone_ids: list[str] = ids
        self.scale_m_per_unit: float = self.venue["coordinate_system"]["scale_m_per_unit"]
        self._gates = [(z["zone_id"], _rect(z["geometry"])) for z in zones if z["zone_type"] == "exit"]
        self._areas = [(z["zone_id"], _rect(z["geometry"])) for z in zones if z["zone_type"] != "exit"]
        self._obstacles = [(o["obstacle_id"], _rect(o["geometry"])) for o in self.venue.get("obstacles", [])
                           if not o.get("walkable", False)]

    def assign(self, x: float, y: float) -> Assignment:
        for zone_id, r in self._gates:
            if r.contains(x, y):
                return Assignment(STATUS_ZONE, zone_id, None)
        for obstacle_id, r in self._obstacles:
            if r.contains(x, y):
                return Assignment(STATUS_OBSTACLE, None, obstacle_id)
        hits = [zone_id for zone_id, r in self._areas if r.contains(x, y)]
        if len(hits) > 1:
            raise ValueError(f"Point ({x}, {y}) lies in overlapping host areas {hits}; venue geometry is invalid")
        if hits:
            return Assignment(STATUS_ZONE, hits[0], None)
        return Assignment(STATUS_OUTSIDE, None, None)

    def count(self, positions: Iterable[tuple[float, float]]) -> dict[str, int]:
        """Occupancy per zone plus explicit outside/obstacle counts. Every position is counted once."""
        counts = Counter({zone_id: 0 for zone_id in self.zone_ids})
        counts[OUTSIDE_KEY] = 0
        counts[OBSTACLE_KEY] = 0
        for x, y in positions:
            a = self.assign(x, y)
            if a.status == STATUS_ZONE:
                counts[a.zone_id] += 1
            elif a.status == STATUS_OBSTACLE:
                counts[OBSTACLE_KEY] += 1
            else:
                counts[OUTSIDE_KEY] += 1
        return dict(counts)
