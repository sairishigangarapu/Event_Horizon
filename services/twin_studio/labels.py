"""Reference records and future labels from the Phase 2 zone-state history (Phase 3).

Both are produced ONLY from zone_states.derive_zone_states output (the authoritative reference),
never from route rows, observations or predictions.

Label availability (approved v0.2 decision D2, `label_delay_seconds`, default 60 s):
  reference state:  label_available_at = state_at  + label_delay_seconds
  future label:     label_available_at = target_at + label_delay_seconds
Future labels: for every canonical issue instant t and horizon h (30, 60, 120 s; multiples of the
5 s grid), the target is the reference state at instant t+h. Targets beyond the last canonical
instant are excluded (counted in the manifest), never extrapolated or replaced by another state.
Interval variables (speed, inflow, outflow) at a target instant T describe [T-5, T), exactly as
the reference state does.
"""
from __future__ import annotations

from typing import Any

from services.twin_studio import ENVIRONMENT_ID, MAP_VERSION
from services.twin_studio.timebase import instant_to_utc
from services.twin_studio.zone_states import UNITS

REFERENCE_SOURCE_TYPE = "SIMULATOR_HIDDEN_REFERENCE"
LABEL_SOURCE_TYPE = "HIDDEN_FUTURE_LABEL"
REFERENCE_VARIABLES = ("occupancy", "density", "speed", "queue_length", "inflow", "outflow",
                       "probability_exit_open", "gate_exits", "route_completed_not_exit")
INTERVAL_VARIABLES = {"speed", "inflow", "outflow", "gate_exits", "route_completed_not_exit"}
DEFAULT_LABEL_VARIABLES = ("density", "speed", "inflow", "outflow", "queue_length", "probability_exit_open")
DEFAULT_HORIZONS_SECONDS = (30, 60, 120)
DEFAULT_LABEL_DELAY_SECONDS = 60


def _iso(epoch, seconds: int) -> str:
    return instant_to_utc(epoch, seconds).isoformat(timespec="seconds").replace("+00:00", "Z")


def _value_fields(zone_state: dict, variable: str) -> dict[str, Any]:
    applicable = zone_state["exit_applicable"] if variable == "probability_exit_open" else True
    value = zone_state[variable] if applicable else None
    return {"value": value, "value_valid": value is not None, "applicable": applicable}


def reference_records(states: list[dict], *, run_id: str, scenario_id: str, epoch, sample_seconds: int,
                      label_delay_seconds: int = DEFAULT_LABEL_DELAY_SECONDS) -> list[dict]:
    records = []
    for st in states:
        t = st["simulation_second"]
        for zone_id in sorted(st["zones"]):
            zs = st["zones"][zone_id]
            for variable in REFERENCE_VARIABLES:
                rec = {
                    "run_id": run_id, "scenario_id": scenario_id, "run_mode": st["run_mode"], "seed": st["seed"],
                    "environment_id": ENVIRONMENT_ID, "map_version": MAP_VERSION,
                    "zone_id": zone_id, "variable": variable, "unit": UNITS[variable],
                    **_value_fields(zs, variable),
                    "measurement": "interval" if variable in INTERVAL_VARIABLES else "instant",
                    "simulation_second": t,
                    "state_at": _iso(epoch, t),
                    "label_available_at": _iso(epoch, t + label_delay_seconds),
                    "source_type": REFERENCE_SOURCE_TYPE,
                }
                if variable in INTERVAL_VARIABLES:
                    rec["interval_start"] = _iso(epoch, st["interval_start_second"])
                    rec["interval_end"] = _iso(epoch, st["interval_end_second"])
                records.append(rec)
    return records


def future_labels(states: list[dict], *, run_id: str, scenario_id: str, epoch, sample_seconds: int,
                  horizons_seconds=DEFAULT_HORIZONS_SECONDS, variables=DEFAULT_LABEL_VARIABLES,
                  label_delay_seconds: int = DEFAULT_LABEL_DELAY_SECONDS) -> tuple[list[dict], dict[str, int]]:
    """Returns (labels, excluded_counts_by_horizon). Excluded = issue instants whose target lies past the run."""
    for h in horizons_seconds:
        if h <= 0 or h % sample_seconds:
            raise ValueError(f"horizon {h}s must be a positive multiple of the {sample_seconds}s reference grid")
    by_t = {st["simulation_second"]: st for st in states}
    last = states[-1]["simulation_second"]
    labels, excluded = [], {}
    for h in horizons_seconds:
        excluded[str(h)] = 0
        for st in states:
            t = st["simulation_second"]
            target = t + h
            if target > last:
                excluded[str(h)] += 1
                continue
            tst = by_t[target]  # exists: the grid is contiguous in steps of sample_seconds
            for zone_id in sorted(tst["zones"]):
                zs = tst["zones"][zone_id]
                for variable in variables:
                    rec = {
                        "run_id": run_id, "scenario_id": scenario_id, "run_mode": st["run_mode"], "seed": st["seed"],
                        "environment_id": ENVIRONMENT_ID, "map_version": MAP_VERSION,
                        "zone_id": zone_id, "variable": variable, "unit": UNITS[variable],
                        **_value_fields(zs, variable),
                        "horizon_seconds": h,
                        "issue_second": t, "issued_at": _iso(epoch, t),
                        "target_second": target, "target_at": _iso(epoch, target),
                        "label_available_at": _iso(epoch, target + label_delay_seconds),
                        "measurement": "interval" if variable in INTERVAL_VARIABLES else "instant",
                        "source_type": LABEL_SOURCE_TYPE,
                    }
                    if variable in INTERVAL_VARIABLES:
                        rec["target_interval_start"] = _iso(epoch, tst["interval_start_second"])
                        rec["target_interval_end"] = _iso(epoch, tst["interval_end_second"])
                    labels.append(rec)
    return labels, excluded
