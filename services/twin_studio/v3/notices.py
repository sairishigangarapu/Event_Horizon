"""Causal event notices derived only from available observations (schema eh-v3-notices-1).

The v3 archive has no model-facing event feed, and the simulator's incident schedule (start/end,
event ids, labels) is hidden truth. A notice here is a published operational message whose only
input is validated measurements that were already available at the replay time:

  rule_id                 trigger (all from available measurements)                notice_type
  ROUTE_SLOW_MOVEMENT     warden movement_category == "slow" on a route               possible_obstruction
  ROUTE_STALL             two consecutive camera reports on a route with speed        possible_obstruction
                          < STALL_SPEED_M_S and visible_people >= STALL_MIN_PEOPLE
  GATE_REPORTED_CLOSED    gate_sensor reports exit_open == 0                           exit_reported_closed
  GATE_REPORTED_REOPENED  gate_sensor reports open after a closed report               exit_reported_open
  ZONE_HIGH_DENSITY       zone_camera density >= HIGH_DENSITY_P_M2                     high_density

Notices fire on onsets: when a source's reports switch from not meeting a rule to meeting it
(one notice per episode, not one per report).
issued_at_s = available_at_s of the measurement that completes the rule;
available_at_s = issued_at_s + dispatch_delay_s. A notice is visible only once available_at_s <= T.
Notices carry the evidence report ids and their observation times, never simulator event ids,
incident names, end times or labels. They state what was reported, not that it is true.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from services.twin_studio.v3.ingest import Measurement

SCHEMA = "eh-v3-notices-1"
STALL_SPEED_M_S = 0.2
STALL_MIN_PEOPLE = 5
HIGH_DENSITY_P_M2 = 4.0
DEFAULT_DISPATCH_DELAY_S = 5


@dataclass(frozen=True)
class Notice:
    notice_id: str
    notice_type: str
    rule_id: str
    entity_type: str
    entity_id: str
    evidence_report_ids: tuple
    evidence_observed_at_s: tuple       # (first, last)
    issued_at_s: float
    available_at_s: float
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["evidence_report_ids"] = list(self.evidence_report_ids)
        d["evidence_observed_at_s"] = list(self.evidence_observed_at_s)
        return d


def _notice(rule, ntype, entity_type, entity_id, evidence: list[Measurement], delay) -> Notice:
    trigger = max(evidence, key=lambda m: (m.available_at_s, m.report_id))
    ids = tuple(m.report_id for m in evidence)
    nid = "ntc_" + hashlib.sha256(f"{rule}|{entity_id}|{'|'.join(ids)}".encode()).hexdigest()[:16]
    return Notice(nid, ntype, rule, entity_type, entity_id, ids,
                  (min(m.observed_at_s for m in evidence), max(m.observed_at_s for m in evidence)),
                  trigger.available_at_s, trigger.available_at_s + delay)


def derive_notices(measurements: Iterable[Measurement], as_of_s: float,
                   dispatch_delay_s: int = DEFAULT_DISPATCH_DELAY_S) -> list[Notice]:
    """Notices visible at as_of_s, from measurements already available at as_of_s."""
    avail = sorted((m for m in measurements if m.available_at_s <= as_of_s and m.observed_at_s <= as_of_s),
                   key=lambda m: (m.observed_at_s, m.available_at_s, m.report_id))
    out: list[Notice] = []
    by = {}
    for m in avail:
        by.setdefault((m.source_kind, m.entity_id, m.variable), []).append(m)

    def onsets(ms, flagged):
        """Measurements where the reported condition switches from not flagged to flagged."""
        prev = False
        for m in ms:
            now = flagged(m)
            if now and not prev:
                yield m
            prev = now

    for (kind, entity, var), ms in by.items():
        if kind == "warden" and var == "movement_category":
            out += [_notice("ROUTE_SLOW_MOVEMENT", "possible_obstruction", "route", entity, [m], dispatch_delay_s)
                    for m in onsets(ms, lambda m: m.value == "slow")]
        elif kind == "gate_sensor" and var == "exit_open":
            out += [_notice("GATE_REPORTED_CLOSED", "exit_reported_closed", "zone", entity, [m], dispatch_delay_s)
                    for m in onsets(ms, lambda m: m.value == 0.0)]
            prev = None
            for m in ms:
                if prev is not None and prev.value == 0.0 and m.value == 1.0:
                    out.append(_notice("GATE_REPORTED_REOPENED", "exit_reported_open", "zone", entity, [prev, m], dispatch_delay_s))
                prev = m
        elif kind == "zone_camera" and var == "density":
            out += [_notice("ZONE_HIGH_DENSITY", "high_density", "zone", entity, [m], dispatch_delay_s)
                    for m in onsets(ms, lambda m: m.value >= HIGH_DENSITY_P_M2)]

    cams = {}
    for m in avail:
        if m.source_kind == "camera" and m.variable in ("speed", "visible_people"):
            cams.setdefault((m.source_id, m.entity_id, m.report_id), {})[m.variable] = m
    by_route = {}
    for (src, route, rid), vals in sorted(cams.items(), key=lambda kv: kv[1]["speed"].observed_at_s if "speed" in kv[1] else 0):
        if "speed" in vals and "visible_people" in vals:
            by_route.setdefault((src, route), []).append(vals)
    for (src, route), seq in by_route.items():
        stalled = [v["speed"].value < STALL_SPEED_M_S and v["visible_people"].value >= STALL_MIN_PEOPLE for v in seq]
        for i in range(1, len(seq)):
            if stalled[i] and stalled[i - 1] and not (i >= 2 and stalled[i - 2]):
                out.append(_notice("ROUTE_STALL", "possible_obstruction", "route", route,
                                   [seq[i - 1]["speed"], seq[i]["speed"]], dispatch_delay_s))
    return sorted((n for n in out if n.available_at_s <= as_of_s), key=lambda n: (n.available_at_s, n.notice_id))
