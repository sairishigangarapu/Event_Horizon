"""Trusted replay controller for the v3 partial-information system (Phase D).

The operator/evaluator owns the replay clock. The model-facing ``ModelView`` exposes only the
current authorized snapshot: it takes no time argument, and nothing it returns refers to files,
manifests, scenes or hidden truth.

  ReplayController.create(...) -> (controller, OperatorKey)
  controller.advance(key)       operator only: moves the watermark forward by one step
  controller.model_view()       -> ModelView; view.snapshot() -> AuthorizedSnapshot at the watermark

A snapshot contains the validated measurements with observed_at_s <= T and available_at_s <= T
(the archive's own cutoff rule), the ids first visible since the view's previous snapshot
(deduplication as the replay advances), causal notices visible at T, and inference-safe static
metadata (venue topology and source settings; no incidents, seeds or routes' futures).

What is enforced where: in-process, these objects are a contract, not a security boundary (Python
code can always reach private state). The enforced boundary is the HTTP replay service
(replay_service.py), whose model endpoint has no time parameter and refuses query parameters,
combined with process/container isolation that gives the model no filesystem access to outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from services.twin_studio.v3.ingest import IngestContext, Measurement, ReportNormalizer, SourceSpec
from services.twin_studio.v3.notices import DEFAULT_DISPATCH_DELAY_S, Notice, derive_notices


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class StaticTopology:
    """Inference-safe venue metadata: zones, gates, physical connections. No scenario content."""
    map_version: str
    zone_ids: tuple
    gate_ids: tuple
    zone_types: dict
    zone_area_m2: dict
    connections: tuple        # (edge_id, from_zone, to_zone, width_m, distance_m, traversable)
    zone_centroid_m: dict = None   # static rectangle centroids in metres (venue geometry)


@dataclass(frozen=True)
class AuthorizedSnapshot:
    session_id: str
    as_of_s: float
    measurements: tuple            # Measurement, all with observed/available <= as_of_s
    new_report_ids: tuple
    notices: tuple                 # Notice, available_at_s <= as_of_s
    topology: StaticTopology
    sources: tuple                 # SourceSpec

    def latest(self, entity_type: str, entity_id: str, variable: str):
        best = None
        for m in self.measurements:
            if m.entity_type == entity_type and m.entity_id == entity_id and m.variable == variable:
                if best is None or (m.observed_at_s, m.available_at_s, m.report_id) > (best.observed_at_s, best.available_at_s, best.report_id):
                    best = m
        return best


class OperatorKey:
    """Capability returned once, to the operator, when a controller is created."""
    __slots__ = ("_session",)

    def __init__(self, session: str):
        self._session = session


def topology_from_venue(venue: dict) -> StaticTopology:
    zones = venue["zones"]
    scale = float(venue["coordinate_system"]["scale_m_per_unit"])
    return StaticTopology(
        map_version=venue["map_version"],
        zone_ids=tuple(z["zone_id"] for z in zones),
        gate_ids=tuple(z["zone_id"] for z in zones if z["zone_type"] == "exit"),
        zone_types={z["zone_id"]: z["zone_type"] for z in zones},
        zone_area_m2={z["zone_id"]: float(z["effective_walkable_area_m2"]) for z in zones},
        connections=tuple((c["edge_id"], c["from_zone"], c["to_zone"], float(c["width_m"]), float(c["distance_m"]),
                           bool(c.get("physically_traversable", True))) for c in venue["connections"]),
        zone_centroid_m={z["zone_id"]: ((z["geometry"]["x"] + z["geometry"]["w"] / 2) * scale,
                                        (z["geometry"]["y"] + z["geometry"]["d"] / 2) * scale) for z in zones},
    )


class ReplayController:
    def __init__(self, session_id: str, measurements: list[Measurement], topology: StaticTopology,
                 sources: dict[str, SourceSpec], *, step_s: int, start_s: float, end_s: float, dispatch_delay_s: int):
        self.session_id = session_id
        self._measurements = sorted(measurements, key=lambda m: (m.available_at_s, m.observed_at_s, m.report_id))
        self._topology = topology
        self._sources = tuple(sources[k] for k in sorted(sources))
        self._step = step_s
        self._watermark = float(start_s)
        self._end = float(end_s)
        self._delay = dispatch_delay_s
        self._key: OperatorKey | None = None

    @classmethod
    def create(cls, session_id: str, reports: Iterable[dict], ctx: IngestContext, topology: StaticTopology, *,
               step_s: int = 5, start_s: float = 0, end_s: float, dispatch_delay_s: int = DEFAULT_DISPATCH_DELAY_S):
        """Validates every report up front (fail fast), then returns (controller, operator key)."""
        if step_s <= 0 or end_s < start_s:
            raise ReplayError("invalid replay schedule")
        measurements = ReportNormalizer(ctx).normalize_many(reports)
        ctrl = cls(session_id, measurements, topology, ctx.sources, step_s=step_s, start_s=start_s, end_s=end_s,
                   dispatch_delay_s=dispatch_delay_s)
        key = OperatorKey(session_id)
        ctrl._key = key
        return ctrl, key

    @property
    def watermark_s(self) -> float:
        return self._watermark

    @property
    def finished(self) -> bool:
        return self._watermark >= self._end

    def advance(self, key: OperatorKey) -> float:
        if key is not self._key:
            raise ReplayError("only the operator key of this session can advance the replay")
        if self.finished:
            raise ReplayError("replay already reached its end")
        self._watermark = min(self._end, self._watermark + self._step)
        return self._watermark

    def _snapshot(self, already: set) -> AuthorizedSnapshot:
        T = self._watermark
        visible = tuple(m for m in self._measurements if m.available_at_s <= T and m.observed_at_s <= T)
        new_ids = tuple(sorted({m.report_id for m in visible} - already))
        already.update(new_ids)
        return AuthorizedSnapshot(self.session_id, T, visible, new_ids,
                                  tuple(derive_notices(visible, T, self._delay)), self._topology, self._sources)

    def model_view(self) -> "ModelView":
        return ModelView(self)


class ModelView:
    """Model-facing access: the current snapshot only. There is no way to request another time."""
    __slots__ = ("_ctrl", "_seen")

    def __init__(self, ctrl: ReplayController):
        self._ctrl = ctrl
        self._seen: set = set()

    @property
    def as_of_s(self) -> float:
        return self._ctrl.watermark_s

    def snapshot(self) -> AuthorizedSnapshot:
        return self._ctrl._snapshot(self._seen)
