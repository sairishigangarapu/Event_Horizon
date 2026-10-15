"""Versioned zone-level sensing extension for Twin Studio v3 (schema eh-v3-zone-sensing-1).

The canonical v3 archive reports per route only. The main pipeline needs zone-level evidence, so
this extension adds three synthetic source kinds, generated with the same principles as the
archive's observations.py:
  * zone_camera   density, queue and (when the zone has moving agents) mean speed of one zone;
  * zone_counter  entries and exits across the zone boundary over the reporting interval;
  * gate_sensor   open/closed state of a gate zone.
Each report uses only the zone state at its own observation instant (observed_at_s), becomes
available at observed_at_s + delay_s, may be missing, and carries multiplicative noise that cannot
create a value from zero (as in the archive). Values that do not exist (speed of an empty zone)
are omitted, never reported as zero. A seeded RNG independent of the engine and of the canonical
sources keeps movement and canonical reports unchanged. Settings are uncalibrated assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from services.twin_studio.v3.engine import simulation_module
from services.twin_studio.v3.ingest import ZONE_SCHEMA, SourceSpec, rid_expected

KINDS = ("zone_camera", "zone_counter", "gate_sensor")
SEED_OFFSET = 51787          # distinct from the archive's movement (seed) and reporting (seed + 11731) streams


@dataclass(frozen=True)
class ZoneSource:
    id: str
    kind: str
    interval_s: int
    delay_s: int
    missing_probability: float
    relative_noise: float = 0.0          # zone_camera / zone_counter
    error_probability: float = 0.0       # gate_sensor state flips
    zone_ids: tuple | None = None        # None: every applicable zone

    def to_dict(self) -> dict:
        return dict(self.__dict__)


DEFAULT_ZONE_SOURCES = (
    ZoneSource("zcam_1", "zone_camera", interval_s=5, delay_s=3, missing_probability=0.05, relative_noise=0.08),
    ZoneSource("zcount_1", "zone_counter", interval_s=5, delay_s=2, missing_probability=0.03, relative_noise=0.03),
    ZoneSource("gate_1", "gate_sensor", interval_s=5, delay_s=1, missing_probability=0.02, error_probability=0.02),
)


def validate_sources(sources: Iterable[ZoneSource], zone_ids: Iterable[str], gate_ids: Iterable[str],
                     sample_seconds: int) -> list[ZoneSource]:
    zone_ids, gate_ids = set(zone_ids), set(gate_ids)
    seen, out = set(), []
    for s in sources:
        if s.id in seen or not s.id:
            raise ValueError("zone source ids must be unique and non-empty")
        seen.add(s.id)
        if s.kind not in KINDS:
            raise ValueError(f"unknown zone source kind {s.kind!r}")
        if not (isinstance(s.interval_s, int) and s.interval_s >= sample_seconds and s.interval_s % sample_seconds == 0):
            raise ValueError("zone source interval_s must be a positive multiple of the zone-state grid")
        if not (isinstance(s.delay_s, int) and 0 <= s.delay_s <= 3600):
            raise ValueError("zone source delay_s must be a whole number in [0, 3600]")
        for p in (s.missing_probability, s.relative_noise, s.error_probability):
            if not 0 <= p <= 1:
                raise ValueError("probabilities and noise must lie in [0, 1]")
        allowed = gate_ids if s.kind == "gate_sensor" else zone_ids
        if s.zone_ids is not None and not set(s.zone_ids) <= allowed:
            raise ValueError(f"source {s.id} covers zones it cannot observe")
        out.append(s)
    return out


def coverage(source: ZoneSource, zone_ids: Iterable[str], gate_ids: Iterable[str]) -> tuple:
    allowed = sorted(gate_ids) if source.kind == "gate_sensor" else sorted(zone_ids)
    return tuple(sorted(source.zone_ids)) if source.zone_ids is not None else tuple(allowed)


def source_specs(sources: Iterable[ZoneSource], zone_ids, gate_ids) -> dict[str, SourceSpec]:
    return {s.id: SourceSpec(s.id, s.kind, s.interval_s, s.delay_s, frozenset(coverage(s, zone_ids, gate_ids)), ZONE_SCHEMA)
            for s in sources}


def generate_zone_observations(states: list[dict], sources: Iterable[ZoneSource], seed: int,
                               sample_seconds: int) -> list[dict]:
    """Zone-level reports from zone states (one state per instant on the sample grid)."""
    RNG = simulation_module().RNG
    if not states:
        return []
    zone_ids = sorted(states[0]["zones"])
    gate_ids = sorted(z for z, s in states[0]["zones"].items() if s["exit_applicable"])
    sources = validate_sources(sources, zone_ids, gate_ids, sample_seconds)
    out = []
    for index, src in enumerate(sources):
        rng = RNG(seed + SEED_OFFSET + index * 7919)
        covered = coverage(src, zone_ids, gate_ids)

        def noisy(value: float, count: bool = False):
            est = max(0.0, value * (1 + (rng.next() * 2 - 1) * src.relative_noise))
            return int(est + 0.5) if count else round(est, 5)

        for state in states:
            t = state["time_s"]
            if t % src.interval_s:
                continue
            for zid in covered:
                zs = state["zones"][zid]
                if rng.next() < src.missing_probability:
                    continue
                if src.kind == "zone_camera":
                    values = {"density_estimate_p_m2": noisy(zs["density"]), "queue_estimate": noisy(zs["queue_length"], True)}
                    if zs["speed"] is not None:
                        values["mean_speed_estimate_m_s"] = noisy(zs["speed"])
                elif src.kind == "zone_counter":
                    if zs["inflow_count"] is None:
                        continue                  # no completed counting window at t = 0
                    # interval may span several sample windows: sum the windows inside (t - interval, t]
                    ins = outs = 0
                    complete = True
                    for back in range(0, src.interval_s, sample_seconds):
                        prior = _state_at(states, t - back, sample_seconds)
                        if prior is None or prior["zones"][zid]["inflow_count"] is None:
                            complete = False      # the counting window would start before t = 0
                            break
                        ins += prior["zones"][zid]["inflow_count"]
                        outs += prior["zones"][zid]["outflow_count"]
                    if not complete:
                        continue
                    values = {"inflow_count": noisy(ins, True), "outflow_count": noisy(outs, True),
                              "count_window_s": src.interval_s}
                else:
                    state_open = zs["exit_open"] >= 0.5
                    if rng.next() < src.error_probability:
                        state_open = not state_open
                    values = {"exit_state": "open" if state_open else "closed"}
                out.append({"observation_id": rid_expected(src.id, zid, t), "source_id": src.id,
                            "source_kind": src.kind, "zone_id": zid, "observed_at_s": t,
                            "available_at_s": t + src.delay_s, "values": values, "schema_version": ZONE_SCHEMA})
    return sorted(out, key=lambda r: (r["available_at_s"], r["observation_id"]))


def _state_at(states: list[dict], t: int, sample_seconds: int):
    i = t // sample_seconds
    if 0 <= i < len(states) and states[i]["time_s"] == t:
        return states[i]
    return None
