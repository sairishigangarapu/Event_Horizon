"""
Spatial-only smoke experiment for the hypernetwork + GNN pair (Person 3).

What it tests: whether the spatial stack learns a useful representation at all, and whether graph
message passing helps over the identical model without edges. A throwaway linear head on H
predicts the simulator reference state at t+h. This is NOT the forecasting system (the temporal
forecaster is not involved), and its outputs are labelled "smoke_trained_auxiliary": they can
never become the active production pair (see spatial.NON_PROMOTABLE_STATUSES).

Data validity rules enforced here:
  - inputs at time t use only observations with received_at <= t (build_inputs never sees references);
  - simulator reference states are used only as targets (and, separately, by the diagnostic oracle baseline);
  - generator "stale" reports (report_id rep_stale_*) are excluded by default, because their values are
    derived from the current reference state under a back-dated observed_at;
  - train/val/test are disjoint by seed; test also uses held-out scenario parameter ranges;
  - normalization statistics and target statistics are fitted on the training split only;
  - model selection uses the validation split; the test split is evaluated once at the end.

Usage: python -m crowd_reliability.ml.spatial_train [--seeds 0 1 2] [--horizons 5 30]
"""
import argparse
import copy
import csv
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sklearn
import torch
import torch.nn as nn

from crowd_reliability.ml.pipeline import ArchitecturePipeline, _calc_sha256
from crowd_reliability.ml.spatial import SpatialEncoder, build_spatial_features, fit_normalization, schema_hash
from crowd_reliability.ml.temporal import TARGET_VAR_NAMES
from crowd_reliability.models.manifest import PipelineManifest
from crowd_reliability.simulation.generator import ObservationGenerator
from crowd_reliability.simulation.scenarios import ScenarioConfig, get_default_venue
from crowd_reliability.simulation.simulator import CrowdSimulator

STATUS = "smoke_trained_auxiliary"
STEP_SECONDS = 5
EXCLUDED_REPORT_PREFIXES = ("rep_stale_",)
EXIT_VAR = TARGET_VAR_NAMES.index("exit_open")
SPLIT_BASE_SEEDS = {"train": 1000, "val": 2000, "test": 3000}

# ---------------------------------------------------------------- scenarios

def sample_config(rng: np.random.Generator, split: str, index: int) -> ScenarioConfig:
    """Train/val share parameter ranges; test uses held-out ranges (stronger surges, later closures)."""
    seed = SPLIT_BASE_SEEDS[split] + index
    if split == "test":
        surge_mult = rng.uniform(3.0, 4.5)
        has_closure = True
        closure_start = rng.uniform(140, 180)
        closure_len = rng.uniform(60, 100)
    else:
        surge_mult = rng.uniform(1.0, 3.0)
        has_closure = rng.random() < 0.6
        closure_start = rng.uniform(40, 140)
        closure_len = rng.uniform(60, 120)
    surge_start = rng.uniform(20, 100)
    surge_len = rng.uniform(60, 140)
    return ScenarioConfig(
        scenario_id=f"{split}_{index:03d}",
        title=f"Sampled {split} scenario {index}",
        description="Sampled by spatial_train for the spatial smoke experiment",
        duration_seconds=300,
        step_seconds=STEP_SECONDS,
        seed=seed,
        surge_start_seconds=int(surge_start),
        surge_end_seconds=int(surge_start + surge_len),
        surge_multiplier=float(surge_mult),
        exit_restriction_zone="exit_1",
        exit_restriction_start=int(closure_start) if has_closure else 9999,
        exit_restriction_end=int(closure_start + closure_len) if has_closure else 9999,
        sensor_dropout_rate=float(rng.uniform(0.05, 0.15)),
        stale_report_probability=float(rng.uniform(0.10, 0.30)),
        conflict_probability=float(rng.uniform(0.05, 0.20)),
    )

def make_splits(n_train: int, n_val: int, n_test: int) -> Dict[str, List[ScenarioConfig]]:
    splits = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        rng = np.random.default_rng(SPLIT_BASE_SEEDS[split])
        splits[split] = [sample_config(rng, split, i) for i in range(n)]
    return splits

def filter_reports(observations, include_stale: bool):
    if include_stale:
        return list(observations)
    return [o for o in observations if not o.report_id.startswith(EXCLUDED_REPORT_PREFIXES)]

# ---------------------------------------------------------------- datasets

def build_inputs(pipeline: ArchitecturePipeline, observations, times) -> Dict[str, np.ndarray]:
    """
    Model inputs at each time in `times`, from observations only (never references).
    Returns features/valid [T, Z, 36] and the tree estimates D [T, Z, 6].
    """
    zones = pipeline.venue.zones
    by_zone = {z["zone_id"]: [o for o in observations if o.zone_id == z["zone_id"]] for z in zones}
    feats, valids, trees = [], [], []
    for t in times:
        per_zone, tvals, tmasks = [], [], []
        for z in zones:
            detailed = pipeline.extract_zone_features_detailed(z["zone_id"], t, by_zone[z["zone_id"]])
            raw, ages, masks = pipeline._legacy_zone_features(detailed)
            d_vals, d_masks = pipeline.tree_bank.predict_zone(np.concatenate([raw, ages, masks]), has_exit=z.get("has_exit", False))
            per_zone.append(detailed)
            tvals.append(d_vals)
            tmasks.append(d_masks)
        f, v = build_spatial_features(
            np.stack([d["values"] for d in per_zone]), np.stack([d["ages_sec"] for d in per_zone]),
            np.stack([d["present"] for d in per_zone]), np.stack([d["fresh"] for d in per_zone]),
            np.array(tvals, dtype=np.float32), np.array(tmasks, dtype=np.float32),
        )
        feats.append(f)
        valids.append(v)
        trees.append(np.array(tvals, dtype=np.float32))
    return {"features": np.stack(feats), "valid": np.stack(valids), "tree": np.stack(trees)}

def build_reference_grid(references, zone_ids: List[str], times) -> Tuple[np.ndarray, np.ndarray]:
    """Reference states [T, Z, 6] and their applicability mask (exit_open is -1 where not applicable)."""
    t_index = {t: i for i, t in enumerate(times)}
    z_index = {z: i for i, z in enumerate(zone_ids)}
    v_index = {v: i for i, v in enumerate(TARGET_VAR_NAMES)}
    grid = np.zeros((len(times), len(zone_ids), 6), dtype=np.float32)
    for r in references:
        grid[t_index[r.state_at], z_index[r.zone_id], v_index[r.variable]] = r.value
    mask = np.ones_like(grid)
    mask[..., EXIT_VAR] = (grid[..., EXIT_VAR] >= 0).astype(np.float32)
    return grid, mask

def build_scenario(pipeline: ArchitecturePipeline, cfg: ScenarioConfig, include_stale: bool) -> Dict[str, Any]:
    venue = pipeline.venue
    refs, _ = CrowdSimulator(venue=venue, seed=cfg.seed).run_simulation(cfg)
    obs = filter_reports(ObservationGenerator(seed=cfg.seed).generate_observations(refs, cfg), include_stale)
    times = sorted({r.state_at for r in refs})
    inputs = build_inputs(pipeline, obs, times)
    ref_grid, ref_mask = build_reference_grid(refs, pipeline.graph_builder.zone_ids, times)
    seconds = np.array([(t - times[0]).total_seconds() for t in times])
    closure = (seconds >= cfg.exit_restriction_start) & (seconds <= cfg.exit_restriction_end)
    return {"inputs": inputs, "ref": ref_grid, "ref_mask": ref_mask, "closure": closure, "n_reports": len(obs)}

def horizon_samples(scenarios: List[Dict[str, Any]], h_steps: int, first_step: int = 2) -> Dict[str, np.ndarray]:
    """Stacks (input at t, target at t+h) pairs across scenarios."""
    out = {k: [] for k in ("features", "valid", "tree", "oracle", "target", "target_mask", "closure")}
    for s in scenarios:
        n = s["ref"].shape[0]
        idx = np.arange(first_step, n - h_steps)
        out["features"].append(s["inputs"]["features"][idx])
        out["valid"].append(s["inputs"]["valid"][idx])
        out["tree"].append(s["inputs"]["tree"][idx])
        out["oracle"].append(s["ref"][idx])
        out["target"].append(s["ref"][idx + h_steps])
        out["target_mask"].append(s["ref_mask"][idx + h_steps])
        out["closure"].append(s["closure"][idx])
    return {k: np.concatenate(v) for k, v in out.items()}

# ---------------------------------------------------------------- model

def batched_forward(enc: SpatialEncoder, features: torch.Tensor, valid: torch.Tensor, use_graph: bool) -> torch.Tensor:
    """Runs B graphs as one block-diagonal graph. use_graph=False removes all edges (same modules)."""
    b, z, f = features.shape
    g = enc.graph
    static = g.node_static.repeat(b, 1)
    if use_graph:
        offsets = (torch.arange(b) * z).repeat_interleave(g.num_edges)
        edge_index = g.edge_index.repeat(1, b) + offsets
        edge_attr = g.edge_attr.repeat(b, 1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, g.edge_attr.size(1)))
    vn, ve = enc.hypernet(features.reshape(b * z, f), static, edge_index, edge_attr, dyn_valid=valid.reshape(b * z, f))
    h = enc.gnn(vn, edge_index, ve)
    return h.reshape(b, z, -1)

def fit_target_stats(target: np.ndarray, mask: np.ndarray, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Train-only target standardization; exit_open stays on its 0/1 scale."""
    mean, std = np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)
    for v in range(6):
        if v == EXIT_VAR:
            continue
        col = target[..., v][mask[..., v] > 0]
        mean[v], std[v] = col.mean(), max(col.std(), eps)
    return mean, std

def train_model(graph, data, norm, tstats, use_graph: bool, seed: int, epochs: int, batch: int, lr: float, patience: int):
    torch.manual_seed(seed)
    enc = SpatialEncoder(graph, seed=seed)
    enc.set_normalization(norm)
    head = nn.Linear(enc.embed_dim, 6)
    params = list(enc.hypernet.parameters()) + list(enc.gnn.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    t_mean, t_std = (torch.from_numpy(a) for a in tstats)

    def tensors(split):
        d = data[split]
        y = (torch.from_numpy(d["target"]) - t_mean) / t_std
        return (torch.from_numpy(d["features"]), torch.from_numpy(d["valid"]), y, torch.from_numpy(d["target_mask"]))

    train_t, val_t = tensors("train"), tensors("val")

    def loss_on(x, v, y, m):
        pred = head(batched_forward(enc, x, v, use_graph))
        return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)

    rng = np.random.default_rng(seed)
    n = train_t[0].shape[0]
    curve, best, best_val, bad = [], None, float("inf"), 0
    for epoch in range(epochs):
        enc.hypernet.train(); enc.gnn.train()
        order = rng.permutation(n)
        total, count = 0.0, 0
        for start in range(0, n, batch):
            idx = torch.from_numpy(order[start:start + batch])
            loss = loss_on(*(t[idx] for t in train_t))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)
        enc.hypernet.eval(); enc.gnn.eval()
        with torch.no_grad():
            val_loss = loss_on(*val_t).item()
        curve.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best = copy.deepcopy((enc.hypernet.state_dict(), enc.gnn.state_dict(), head.state_dict()))
        else:
            bad += 1
            if bad >= patience:
                break
    enc.hypernet.load_state_dict(best[0])
    enc.gnn.load_state_dict(best[1])
    head.load_state_dict(best[2])
    enc.hypernet.eval(); enc.gnn.eval()
    return enc, head, curve, best_val

def predict(enc, head, d, tstats, use_graph: bool) -> np.ndarray:
    with torch.no_grad():
        pred = head(batched_forward(enc, torch.from_numpy(d["features"]), torch.from_numpy(d["valid"]), use_graph)).numpy()
    pred = pred * tstats[1] + tstats[0]
    pred[..., EXIT_VAR] = np.clip(pred[..., EXIT_VAR], 0.0, 1.0)
    return pred

# ---------------------------------------------------------------- metrics

def metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, select: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    out = {}
    for v, name in enumerate(TARGET_VAR_NAMES):
        m = mask[..., v] > 0
        if select is not None:
            m = m & select
        if m.sum() == 0:
            continue
        err = pred[..., v][m] - target[..., v][m]
        row = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())), "n": int(m.sum())}
        if v == EXIT_VAR:
            row["accuracy"] = float(((pred[..., v][m] >= 0.5) == (target[..., v][m] >= 0.5)).mean())
        out[name] = row
    return out

def _git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"

# ---------------------------------------------------------------- main

def run(args) -> str:
    started = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    run_dir = os.path.join(args.out_root, run_id)
    os.makedirs(run_dir, exist_ok=False)

    # Read-only use of the active tree bank, verified against the active manifest
    manifest = PipelineManifest.load(os.path.join(args.model_dir, "manifest.json"))
    tb_path = os.path.join(args.model_dir, "tree_bank.joblib")
    tb_hash = _calc_sha256(tb_path)
    if manifest.model_hashes.get("tree_bank.joblib") != tb_hash:
        raise ValueError("tree_bank.joblib does not match the active manifest")
    pipeline = ArchitecturePipeline(venue=get_default_venue())
    pipeline.tree_bank.load(tb_path)
    graph = pipeline.graph_builder

    splits = make_splits(args.n_train, args.n_val, args.n_test)
    print(f"[spatial_train] run {run_id}: building scenarios (stale reports {'INCLUDED' if args.include_stale else 'excluded'})")
    built = {s: [build_scenario(pipeline, cfg, args.include_stale) for cfg in cfgs] for s, cfgs in splits.items()}
    for s, cfgs in splits.items():
        closures = sum(c.exit_restriction_start < 9999 for c in cfgs)
        print(f"  {s:5s}: {len(cfgs)} scenarios, seeds {cfgs[0].seed}-{cfgs[-1].seed}, "
              f"{closures} with exit_1 closure, {sum(b['n_reports'] for b in built[s])} reports")

    curves_rows, metric_rows, summary = [], [], {}
    for h_sec in args.horizons:
        h_steps = h_sec // STEP_SECONDS
        data = {s: horizon_samples(built[s], h_steps) for s in built}
        norm = fit_normalization(
            data["train"]["features"].reshape(-1, data["train"]["features"].shape[-1]),
            data["train"]["valid"].reshape(-1, data["train"]["valid"].shape[-1]),
            scaler_id=f"fit_{run_id}_h{h_sec}",
        )
        tstats = fit_target_stats(data["train"]["target"], data["train"]["target_mask"])
        test = data["test"]
        conc = graph.zone_ids.index("concourse")
        closure_sel = np.zeros(test["target"].shape[:2], dtype=bool)
        closure_sel[test["closure"], conc] = True

        results = {
            "tree_persistence": [metrics(test["tree"], test["target"], test["target_mask"])],
            "oracle_persistence_DIAGNOSTIC": [metrics(test["oracle"], test["target"], test["target_mask"])],
        }
        closure_results = {
            "tree_persistence": [metrics(test["tree"], test["target"], test["target_mask"], closure_sel)],
            "oracle_persistence_DIAGNOSTIC": [metrics(test["oracle"], test["target"], test["target_mask"], closure_sel)],
        }
        for model_name, use_graph in (("no_graph", False), ("graph", True)):
            results[model_name], closure_results[model_name] = [], []
            for seed in args.seeds:
                enc, head, curve, best_val = train_model(graph, data, norm, tstats, use_graph, seed,
                                                         args.epochs, args.batch, args.lr, args.patience)
                for row in curve:
                    curves_rows.append({"horizon_s": h_sec, "model": model_name, "seed": seed, **row})
                pred = predict(enc, head, test, tstats, use_graph)  # test evaluated once, after selection
                results[model_name].append(metrics(pred, test["target"], test["target_mask"]))
                closure_results[model_name].append(metrics(pred, test["target"], test["target_mask"], closure_sel))
                print(f"  h={h_sec}s {model_name:8s} seed={seed}: epochs={len(curve)} best_val={best_val:.4f}")
                if use_graph:
                    enc.stamp(training_status=STATUS)
                    ckpt = os.path.join(run_dir, "checkpoints", f"h{h_sec}_graph_seed{seed}")
                    enc.save(ckpt)
                    torch.save({"state_dict": head.state_dict(), "target_mean": tstats[0].tolist(),
                                "target_std": tstats[1].tolist(), "status": STATUS}, os.path.join(ckpt, "aux_head.pt"))

        for subset, res in (("all_zones", results), ("concourse_during_closure", closure_results)):
            for model_name, per_seed in res.items():
                for var in TARGET_VAR_NAMES:
                    vals = [r[var] for r in per_seed if var in r]
                    if not vals:
                        continue
                    maes, rmses = [v["mae"] for v in vals], [v["rmse"] for v in vals]
                    metric_rows.append({
                        "horizon_s": h_sec, "subset": subset, "model": model_name, "variable": var,
                        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                        "n_seeds": len(vals), "n": vals[0]["n"],
                        "accuracy_mean": float(np.mean([v["accuracy"] for v in vals])) if "accuracy" in vals[0] else "",
                    })

        # Conclusions: per variable, on the all-zones test subset
        verdict = {}
        for var in TARGET_VAR_NAMES:
            g = [r[var]["mae"] for r in results["graph"]]
            ng = [r[var]["mae"] for r in results["no_graph"]]
            tp = results["tree_persistence"][0][var]["mae"]
            verdict[var] = {
                "learns_beyond_tree_persistence": bool(np.mean(g) < tp),
                "graph_beats_no_graph_every_seed": bool(all(a < b for a, b in zip(g, ng))),
                "graph_beats_no_graph_no_seed": bool(all(a >= b for a, b in zip(g, ng))),
            }
        summary[f"h{h_sec}"] = verdict

    with open(os.path.join(run_dir, "curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves_rows[0].keys()))
        w.writeheader()
        w.writerows(curves_rows)
    with open(os.path.join(run_dir, "metrics_test.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)

    run_manifest = {
        "run_id": run_id,
        "status": STATUS,
        "not_deployable": True,
        "purpose": "Spatial-only smoke experiment: does HNet+GNN learn, and does message passing help vs no edges",
        "caveats": [
            "Auxiliary linear head on H; the temporal forecaster is not involved; not a forecasting result.",
            "Tree estimates come from the active tree bank, which was fitted in-sample on the preset scenarios (not out-of-sample).",
            "Generator stale reports carry values derived from the current reference state; they are "
            + ("INCLUDED (diagnostic run, invalid evidence)." if args.include_stale else "excluded from inputs."),
            "Conflict reports are kept; their observed_at is within +/-1 s of the state they were derived from.",
            "One venue layout only; no claim of generalization to other venues.",
            "Simulator report_ids use uuid4, so report ids (not values) differ between runs.",
        ],
        "excluded_report_kinds": [] if args.include_stale else list(EXCLUDED_REPORT_PREFIXES),
        "horizons_s": args.horizons,
        "seeds": args.seeds,
        "hyperparameters": {"epochs_max": args.epochs, "batch": args.batch, "lr": args.lr, "patience": args.patience,
                            "loss": "masked MSE on train-standardized targets (exit_open unscaled)"},
        "splits": {s: [c.model_dump() for c in cfgs] for s, cfgs in splits.items()},
        "inputs": {"schema_hash": schema_hash(), "tree_bank_sha256": tb_hash, "tree_version": manifest.tree_version,
                   "graph": graph.metadata()},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
        "git": _git_head(),
        "summary": summary,
        "runtime_s": round(time.time() - started, 1),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, default=str)

    print_report(metric_rows, summary)
    print(f"[spatial_train] wrote {run_dir}")
    return run_dir

def print_report(rows, summary):
    for subset in ("all_zones", "concourse_during_closure"):
        print(f"\n=== Test MAE ({subset}); mean +/- std over seeds ===")
        print(f"{'h':>4} {'variable':<13} {'tree_pers':>10} {'oracle*':>9} {'no_graph':>16} {'graph':>16}")
        keyed = {(r["horizon_s"], r["model"], r["variable"]): r for r in rows if r["subset"] == subset}
        for h in sorted({r["horizon_s"] for r in rows}):
            for var in TARGET_VAR_NAMES:
                if (h, "graph", var) not in keyed:
                    continue
                def cell(m, std=False):
                    r = keyed.get((h, m, var))
                    if r is None:
                        return "-"
                    return f"{r['mae_mean']:.4f}+/-{r['mae_std']:.4f}" if std else f"{r['mae_mean']:.4f}"
                print(f"{h:>4} {var:<13} {cell('tree_persistence'):>10} {cell('oracle_persistence_DIAGNOSTIC'):>9} "
                      f"{cell('no_graph', True):>16} {cell('graph', True):>16}")
    print("  * oracle uses the reference state at t: diagnostic only, not deployable")
    print("\n=== Conclusions per variable ===")
    for h, verdict in summary.items():
        for var, v in verdict.items():
            graph_claim = ("graph better on every seed" if v["graph_beats_no_graph_every_seed"]
                           else "graph worse on every seed" if v["graph_beats_no_graph_no_seed"] else "inconclusive")
            print(f"  {h} {var:<13} learns beyond tree persistence: {v['learns_beyond_tree_persistence']!s:<5}  graph vs no-graph: {graph_claim}")

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 30])
    p.add_argument("--n-train", type=int, default=24)
    p.add_argument("--n-val", type=int, default=8)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--include-stale", action="store_true", help="Diagnostic only: results are not valid evidence")
    p.add_argument("--model-dir", default="models/production", help="Read-only source of the tree bank")
    p.add_argument("--out-root", default="models/spatial/experiments")
    args = p.parse_args(argv)
    if any(h % STEP_SECONDS for h in args.horizons):
        p.error(f"horizons must be multiples of {STEP_SECONDS} s")
    run(args)

if __name__ == "__main__":
    main()
