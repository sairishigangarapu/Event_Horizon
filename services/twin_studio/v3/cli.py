"""Reproducible commands for the Twin Studio v3 pipeline. Every command refuses to overwrite outputs.

  python -m services.twin_studio.v3.cli build-dataset   --name mc_v1_n16_seed2026 --n-per-regime 16 --seed 2026
  python -m services.twin_studio.v3.cli train-trees     --dataset mc_v1_n16_seed2026
  python -m services.twin_studio.v3.cli train-neural    --dataset mc_v1_n16_seed2026
  python -m services.twin_studio.v3.cli eval-staleness  --dataset mc_v1_n16_seed2026
  python -m services.twin_studio.v3.cli render          --scene <scene.json> [--report]
  python -m services.twin_studio.v3.cli run-pipeline    --run-dir <outputs/run_id> [--k 1.0]
  python -m services.twin_studio.v3.cli write-manifest  --dataset mc_v1_n16_seed2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from services.twin_studio.v3 import CANONICAL_ARCHIVE_SHA256, engine
from services.twin_studio.v3.artifacts import OfflineAccess
from services.twin_studio.v3.dataset import DATA_ROOT, build_sample_store, load_samples, render_scenarios

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "models" / "environments" / "eh_twin_studio" / "v3"


def _ds(name: str) -> Path:
    return DATA_ROOT / "mc" / name


def _tree_dir(name: str) -> Path:
    from services.twin_studio.v3 import trees
    return MODELS / "trees" / f"{trees.BANK_VERSION}__{name}"


def _neural_dir(name: str) -> Path:
    from services.twin_studio.v3 import neural, trees
    return MODELS / "neural" / f"{neural.VERSION}__{trees.BANK_VERSION}__{name}"


def _staleness_path(name: str) -> Path:
    return MODELS / "staleness" / f"staleness_report__eh-v3-staleness-1__{name}.json"


def _oos(name: str, data: dict) -> dict:
    from services.twin_studio.v3 import trees
    z = np.load(_ds(name) / "derived" / f"tree_oos__{trees.BANK_VERSION}.npz")
    return {sid: {"values": z[f"{sid}__values"], "masks": z[f"{sid}__masks"]} for sid in data}


def build_dataset(name: str, n: int, seed: int):
    from services.twin_studio.v3.scenarios import GENERATOR_VERSION, sample_all
    root = _ds(name)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    sc = sample_all(n, master_seed=seed)
    render_scenarios(sc, root / "runs")
    build_sample_store(root / "runs", root / "samples", OfflineAccess("dataset_construction"))
    (root / "dataset_manifest.json").write_text(json.dumps({"generator_version": GENERATOR_VERSION, "master_seed": seed,
                                                            "n_per_regime": n, "scenarios": [s.record() for s in sc]}, indent=2),
                                                encoding="utf-8")


def train_trees(name: str):
    from services.twin_studio.v3 import trees
    _, data = load_samples(_ds(name) / "samples")
    bank, out = trees.train(data, seed=0, folds=4)
    report = out["report"] | {"dataset": name}
    tdir = trees.save(bank, report, _tree_dir(name))
    val = sorted(k for k, d in data.items() if d["split"] == "val")
    m = trees.evaluate(bank, data, val)
    detail = {v: {"metric": "brier" if v == "exit_open" else "mae",
                  "tree": e["tree"]["brier" if v == "exit_open" else "mae"],
                  "latest_report_baseline": e["latest_report_baseline"]["brier" if v == "exit_open" else "mae"]}
              for v, e in m.items()}
    for d in detail.values():
        d["validated"] = d["tree"] <= d["latest_report_baseline"]
    (tdir / "tree_validation.json").write_text(json.dumps({
        "rule": "a channel is validated if its validation-scenario error (MAE; Brier for exit_open) is no worse than the zone's latest-report baseline",
        "validated_channels": [v for v, d in detail.items() if d["validated"]], "validation_scenarios": val, "detail": detail},
        indent=2), encoding="utf-8")
    oos = _ds(name) / "derived" / f"tree_oos__{trees.BANK_VERSION}.npz"
    oos.parent.mkdir(exist_ok=True)
    if oos.exists():
        raise FileExistsError(f"{oos} exists")
    np.savez_compressed(oos, **{f"{s}__values": v["values"] for s, v in out["oos"].items()},
                        **{f"{s}__masks": v["masks"] for s, v in out["oos"].items()})


def train_neural(name: str):
    from services.twin_studio.v3 import neural, trees
    _, data = load_samples(_ds(name) / "samples")
    model, rep = neural.train(data, _oos(name, data), epochs=120, patience=12)
    neural.save(model, rep | {"dataset": name}, _neural_dir(name), tree_version=trees.BANK_VERSION)


def eval_staleness(name: str):
    from services.twin_studio.v3 import staleness_eval
    index = json.loads((_ds(name) / "runs" / "index.json").read_text(encoding="utf-8"))
    rep = staleness_eval.calibrate_and_evaluate(_ds(name) / "runs", index, OfflineAccess("evaluation"))
    target = _staleness_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.write_text(json.dumps(rep | {"dataset": name}, indent=2), encoding="utf-8")


def write_manifest(name: str):
    from services.twin_studio.v3 import neural, pipeline, routing, staleness, trees
    tdir, ndir = _tree_dir(name), _neural_dir(name)
    trep = json.loads((tdir / "tree_report.json").read_text(encoding="utf-8"))
    tval = json.loads((tdir / "tree_validation.json").read_text(encoding="utf-8"))
    nrep = json.loads((ndir / "neural_report.json").read_text(encoding="utf-8"))
    srep = json.loads(_staleness_path(name).read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {
        "environment_id": "eh_twin_studio", "map_version": "twin_venue_v1", "canonical_archive_sha256": CANONICAL_ARCHIVE_SHA256,
        "engine_model_version": "kinematic-v3", "pipeline_version": pipeline.VERSION, "dataset": name,
        "components": {
            "tree_bank": {"version": trees.BANK_VERSION, "feature_schema": trep["feature_schema"],
                          "status": "trained_heldout_evaluated", "validated_channels": tval["validated_channels"],
                          "artifact": str((tdir / "tree_bank.joblib").relative_to(REPO)).replace("\\", "/"),
                          "sha256": sha(tdir / "tree_bank.joblib"), "sklearn_version": trep["sklearn_version"]},
            "hypernet": {"version": "hypernet_v2", "status": nrep["training_status"],
                         "sha256": sha(ndir / "spatial" / "hypernet.pt"), "pair_id": nrep["pair_id"]},
            "gnn": {"version": "gnn_edge_v2", "status": nrep["training_status"], "sha256": sha(ndir / "spatial" / "gnn.pt"),
                    "pair_id": nrep["pair_id"]},
            "temporal": {"version": "gru_forecaster_v3_twin", "status": nrep["training_status"],
                         "sha256": sha(ndir / "temporal_model.pt"), "u_schema": nrep["u_schema"],
                         "horizons_s": nrep["horizons_s"], "validation_wins_30s": nrep["validation_wins_30s"]},
            "staleness_policy": {"version": staleness.POLICY_VERSION, "age_multiplier": float(srep["chosen_on_validation"].split("=")[1]),
                                 "status": "calibrated_on_validation_reported_on_test"},
            "routing": {"version": routing.VERSION, "status": "rule_based_decision_support_not_validated_against_outcomes"},
        },
        "not_implemented": ["hypernetwork+GNN feedback loop (candidate copy, validation gate, atomic promotion)",
                            "Monte Carlo hyperparameter selection for the hypernetwork/GNN (out of policy, ARCHITECTURE section 10)"],
    }
    target = MODELS / f"manifest_v3__{name}.json"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("build-dataset", "train-trees", "train-neural", "eval-staleness", "write-manifest"):
        p = sub.add_parser(cmd)
        p.add_argument("--dataset", "--name", dest="name", default="mc_v1_n16_seed2026")
        if cmd == "build-dataset":
            p.add_argument("--n-per-regime", type=int, default=16)
            p.add_argument("--seed", type=int, default=2026)
    r = sub.add_parser("render")
    r.add_argument("--scene", required=True)
    r.add_argument("--report", action="store_true")
    r.add_argument("--out", default=str(DATA_ROOT / "runs"))
    p = sub.add_parser("run-pipeline")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", default="mc_v1_n16_seed2026")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--out", default=str(DATA_ROOT / "evaluations"))
    a = ap.parse_args(argv)
    if a.cmd == "build-dataset":
        build_dataset(a.name, a.n_per_regime, a.seed)
    elif a.cmd == "train-trees":
        train_trees(a.name)
    elif a.cmd == "train-neural":
        train_neural(a.name)
    elif a.cmd == "eval-staleness":
        eval_staleness(a.name)
    elif a.cmd == "write-manifest":
        write_manifest(a.name)
    elif a.cmd == "render":
        from services.twin_studio.v3.runs import render_run
        scene = json.loads(Path(a.scene).read_text(encoding="utf-8"))
        print(render_run(scene, Path(a.out), include_report=a.report))
    elif a.cmd == "run-pipeline":
        from services.twin_studio.v3.pipeline import run
        from services.twin_studio.v3.staleness import StalenessPolicy
        k = a.k if a.k is not None else float(json.loads(_staleness_path(a.dataset).read_text(encoding="utf-8"))
                                              ["chosen_on_validation"].split("=")[1])
        print(run(Path(a.run_dir), Path(a.out), tree_dir=_tree_dir(a.dataset), neural_dir=_neural_dir(a.dataset),
                  policy=StalenessPolicy(age_multiplier=k)))


if __name__ == "__main__":
    main()
