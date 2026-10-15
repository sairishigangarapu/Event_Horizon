"""Offline evaluation and calibration of the staleness policy (Phase H.3; authorized offline use).

For each scenario run, the replay is stepped exactly as for inference; at every 5 s instant T every
source's latest report per entity/variable is assessed. Its evaluation target is whether the
report is materially inconsistent with the hidden true state at T (staleness.materially_different
with variable-specific tolerances; categorical movement reports compare categories).

Metrics: precision / recall / false-positive rate of "alert" (recommended_action != NO_ACTION),
by source kind, variable and regime, plus detection delay: for each entity/variable, an episode
starts when any assessed report becomes inconsistent and lasts while one stays inconsistent; the
delay is the time to the first alert within the episode (episodes without one are undetected).
The age multiplier k is chosen on validation scenarios (max F1) and test is reported once.
Primary target: a report that was consistent with the true state at its observation time but is
materially inconsistent at T ("became outdated"). Reports already wrong when observed are
measurement errors no age rule can detect; they are counted separately, and the all-report
"inconsistent now" target is kept as a secondary figure. Zone counters are compared with true
rates over the same counting-window length.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from services.twin_studio.v3.artifacts import OfflineAccess, read_hidden_truth
from services.twin_studio.v3.operator_side import session_from_run_dir
from services.twin_studio.v3.staleness import StalenessPolicy, assess_snapshot, materially_different, score, truth_value

K_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _route_rows(truth: dict) -> dict:
    rows = {}
    for r in truth["metrics"]:
        rows[(int(r["time_s"]), r["route_id"])] = r
    return rows


def replay_records(run_dir: Path, regime: str, policies: dict, access: OfflineAccess) -> dict:
    """{policy_name: [record, ...]} with one record per assessed report per instant."""
    (ctrl, key), meta, _ = session_from_run_dir(run_dir, "custom")
    truth = read_hidden_truth(run_dir, "custom", access)
    zstates = {s["time_s"]: s for s in truth["zone_states"]}
    rows = _route_rows(truth)
    view = ctrl.model_view()
    out = {name: [] for name in policies}
    while True:
        T = int(ctrl.watermark_s)
        if T > meta["duration_s"]:
            break
        snap = view.snapshot()
        intervals = {src.source_id: src.interval_s for src in snap.sources}
        for name, policy in policies.items():
            for a in assess_snapshot(snap, policy):
                def at(t):
                    win = [zstates.get(t - back) for back in range(0, intervals[a.source_id], 5)]
                    return truth_value(a, zstates.get(t), rows.get((t, a.entity_id)) if a.entity_type == "route" else None,
                                       zone_window_states=win if all(win) else None)
                tv_now, tv_obs = at(T), at(int(a.observed_at_s))
                if tv_now is None or tv_obs is None:
                    continue
                wrong_at_observation = materially_different(a.variable, a.value, tv_obs)
                inconsistent_now = materially_different(a.variable, a.value, tv_now)
                out[name].append({"T": T, "regime": regime, "source_kind": a.source_kind, "variable": a.variable,
                                  "entity": f"{a.entity_type}:{a.entity_id}", "state": a.state, "alert": a.alert,
                                  "conflict": a.conflict_status, "age_s": a.age_s,
                                  "wrong_at_observation": wrong_at_observation,
                                  "inconsistent_now": inconsistent_now,
                                  # primary target: correct when observed, materially outdated at T
                                  "inconsistent": (not wrong_at_observation) and inconsistent_now})
        if ctrl.finished or T + 5 > meta["duration_s"]:
            break
        ctrl.advance(key)
    return out


def detection_delays(records: list[dict]) -> dict:
    by = defaultdict(lambda: defaultdict(lambda: [False, False]))
    for r in records:
        cell = by[(r["entity"], r["variable"])][r["T"]]
        cell[0] |= r["inconsistent"]
        cell[1] |= r["alert"]
    delays, undetected = [], 0
    for _, timeline in by.items():
        onset, detected = None, False
        for T in sorted(timeline):
            inc, alert = timeline[T]
            if inc and onset is None:
                onset, detected = T, False
            if onset is not None and not detected and alert:
                delays.append(T - onset)
                detected = True
            if not inc and onset is not None:
                if not detected:
                    undetected += 1
                onset = None
        if onset is not None and not detected:
            undetected += 1
    total = len(delays) + undetected
    return {"episodes": total, "detected_fraction": round(len(delays) / total, 4) if total else None,
            "median_delay_s": statistics.median(delays) if delays else None,
            "p90_delay_s": sorted(delays)[int(0.9 * (len(delays) - 1))] if delays else None}


def summarize(all_records: list[dict]) -> dict:
    """Primary metrics on reports that were correct when observed (target: became outdated by T).
    Reports already wrong at observation time are measurement errors, reported separately."""
    records = [r for r in all_records if not r["wrong_at_observation"]]
    def sc(rs, target="inconsistent"):
        s = score([(r["alert"], r[target]) for r in rs])
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
    groups = {"overall": sc(records),
              "measurement_errors": {"reports_wrong_at_observation": sum(r["wrong_at_observation"] for r in all_records),
                                     "of_assessed": len(all_records),
                                     "alerted": sum(r["alert"] for r in all_records if r["wrong_at_observation"])},
              "secondary_inconsistent_now_all_reports": sc(all_records, "inconsistent_now")}
    for key in ("source_kind", "variable", "regime"):
        groups[f"by_{key}"] = {v: sc([r for r in records if r[key] == v]) for v in sorted({r[key] for r in records})}
    groups["conflicting_sources"] = sc([r for r in records if r["conflict"] == "CONFLICT"])
    groups["detection"] = detection_delays(records)
    groups["state_counts"] = {s: sum(1 for r in records if r["state"] == s) for s in sorted({r["state"] for r in records})}
    return groups


def calibrate_and_evaluate(runs_root: Path, index: list[dict], access: OfflineAccess, k_grid=K_GRID) -> dict:
    policies = {f"k={k}": StalenessPolicy(age_multiplier=k) for k in k_grid}
    split_records = {"val": {n: [] for n in policies}, "test": {n: [] for n in policies}}
    for rec in index:
        if rec["split"] not in split_records:
            continue
        got = replay_records(Path(runs_root) / rec["run_dir"], rec["regime"], policies, access)
        for n in policies:
            split_records[rec["split"]][n].extend(got[n])
    val = {n: summarize(rs)["overall"] for n, rs in split_records["val"].items()}
    chosen = max(val, key=lambda n: (val[n]["f1"] or 0.0, -float(n.split("=")[1])))
    return {"policy_version": StalenessPolicy().version, "k_grid": list(k_grid), "validation_overall": val,
            "chosen_on_validation": chosen,
            "test_chosen": summarize(split_records["test"][chosen]),
            "test_initial_k2": summarize(split_records["test"]["k=2.0"]),
            "note": "Targets compare each report with the hidden true state at the evaluation instant. "
                    "Model-based evidence is not used (policy uses age, observed conflicts and newer reports)."}
