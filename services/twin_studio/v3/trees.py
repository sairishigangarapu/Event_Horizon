"""Six-channel topology-aware decision-tree bank for Twin Studio v3 (Phase F).

Channels (fixed order, never merged): density, speed, inflow, outflow, queue_length (regression)
and exit_open (classification probability, gate zones only; masked elsewhere).
Inputs: eh-v3-topo-features-1 rows built from authorized snapshots (features.py).
Labels: hidden zone-state values at the same instant (current state), offline only.

Training protocol:
  * splits are whole Monte Carlo scenarios (train / val / test), stratified by regime;
  * per channel, max_depth and min_samples_leaf are selected on validation scenarios;
  * out-of-sample tree outputs for training scenarios come from grouped K-fold cross-fitting
    (a scenario is never predicted by a model that saw it); val/test outputs come from the model
    fitted on all training scenarios;
  * test metrics are reported once, per channel and regime, against two baselines: the zone's own
    latest report (falling back to the training mean when there is none) and the training mean.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from services.twin_studio.v3.features import SCHEMA_VERSION as FEATURE_SCHEMA, VARIABLES, feature_names

BANK_VERSION = "dt_bank_v3_topo_1"
GRID = [(d, leaf) for d in (4, 6, 8, 10, 12) for leaf in (5, 20)]
EXIT = VARIABLES.index("exit_open")
OWN_VALUE = {v: feature_names().index(f"own_value_{v}") for v in VARIABLES}
IS_GATE = feature_names().index("exit_applicable")


def _rows(data: dict, sids: list[str]):
    X, Y, M, G, R = [], [], [], [], []
    for sid in sids:
        d = data[sid]
        n_t, n_z, _ = d["X"].shape
        X.append(d["X"].reshape(n_t * n_z, -1))
        Y.append(d["Y"].reshape(n_t * n_z, -1))
        M.append(d["M"].reshape(n_t * n_z, -1))
        G += [sid] * (n_t * n_z)
        R += [d["regime"]] * (n_t * n_z)
    return np.vstack(X), np.vstack(Y), np.vstack(M), np.array(G), np.array(R)


def _model(j: int, depth: int, leaf: int, seed: int):
    cls = DecisionTreeClassifier if j == EXIT else DecisionTreeRegressor
    return cls(max_depth=depth, min_samples_leaf=leaf, random_state=seed)


def _fit_channel(j, X, y, depth, leaf, seed):
    m = _model(j, depth, leaf, seed)
    m.fit(X, y.astype(int) if j == EXIT else y)
    return m


def _predict_channel(j, model, X):
    if j != EXIT:
        return model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def _score(j, pred, y):
    if j == EXIT:
        return float(np.mean((pred - y) ** 2))           # Brier score
    return float(np.mean(np.abs(pred - y)))              # MAE


@dataclass
class TreeBankV3:
    models: list = field(default_factory=list)
    hyperparams: list = field(default_factory=list)
    train_means: list = field(default_factory=list)
    version: str = BANK_VERSION

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """values[Z, 6], masks[Z, 6]; exit channel masked outside gate zones."""
        values = np.zeros((len(X), len(VARIABLES)))
        masks = np.ones_like(values)
        gate = X[:, IS_GATE] == 1.0
        for j, model in enumerate(self.models):
            if j == EXIT:
                values[gate, j] = _predict_channel(j, model, X[gate]) if gate.any() else []
                masks[~gate, j] = 0.0
            else:
                values[:, j] = np.maximum(0.0, _predict_channel(j, model, X))
        return values, masks


def train(data: dict, *, seed: int = 0, folds: int = 4) -> tuple[TreeBankV3, dict]:
    sids = {s: sorted(k for k, d in data.items() if d["split"] == s) for s in ("train", "val", "test")}
    Xtr, Ytr, Mtr, Gtr, _ = _rows(data, sids["train"])
    Xva, Yva, Mva, _, _ = _rows(data, sids["val"])
    bank = TreeBankV3()
    selection = {}
    for j, var in enumerate(VARIABLES):
        tr = Mtr[:, j] == 1.0
        va = Mva[:, j] == 1.0
        if j == EXIT:
            tr &= Xtr[:, IS_GATE] == 1.0
            va &= Xva[:, IS_GATE] == 1.0
        scores = {}
        for depth, leaf in GRID:
            m = _fit_channel(j, Xtr[tr], Ytr[tr, j], depth, leaf, seed)
            scores[(depth, leaf)] = _score(j, _predict_channel(j, m, Xva[va]), Yva[va, j])
        best = min(scores, key=lambda k: (scores[k], k[0]))
        selection[var] = {"max_depth": best[0], "min_samples_leaf": best[1], "val_score": scores[best],
                          "val_metric": "brier" if j == EXIT else "mae"}
        bank.hyperparams.append({"max_depth": best[0], "min_samples_leaf": best[1]})
        bank.models.append(_fit_channel(j, Xtr[tr], Ytr[tr, j], *best, seed))
        bank.train_means.append(float(np.mean(Ytr[tr, j])))

    # Grouped cross-fitting: out-of-sample outputs for every training scenario.
    order = sids["train"]
    fold_of = {sid: i % folds for i, sid in enumerate(order)}
    oos = {}
    for f in range(folds):
        fit_sids = [s for s in order if fold_of[s] != f]
        out_sids = [s for s in order if fold_of[s] == f]
        Xf, Yf, Mf, _, _ = _rows(data, fit_sids)
        fold_bank = TreeBankV3(hyperparams=bank.hyperparams)
        for j in range(len(VARIABLES)):
            sel = Mf[:, j] == 1.0
            if j == EXIT:
                sel &= Xf[:, IS_GATE] == 1.0
            hp = bank.hyperparams[j]
            fold_bank.models.append(_fit_channel(j, Xf[sel], Yf[sel, j], hp["max_depth"], hp["min_samples_leaf"], seed))
        for sid in out_sids:
            oos[sid] = _predict_scenario(fold_bank, data[sid])
    for sid in sids["val"] + sids["test"]:
        oos[sid] = _predict_scenario(bank, data[sid])

    report = {"bank_version": BANK_VERSION, "feature_schema": FEATURE_SCHEMA, "selection": selection,
              "splits": sids, "folds": folds, "sklearn_version": sklearn.__version__,
              "test_metrics": evaluate(bank, data, sids["test"]),
              "feature_importance_neighbour_share": _neighbour_share(bank)}
    return bank, {"report": report, "oos": oos}


def _predict_scenario(bank: TreeBankV3, d: dict) -> dict:
    n_t, n_z, n_f = d["X"].shape
    vals, masks = bank.predict(d["X"].reshape(n_t * n_z, n_f))
    return {"values": vals.reshape(n_t, n_z, -1), "masks": masks.reshape(n_t, n_z, -1)}


def _neighbour_share(bank: TreeBankV3) -> dict:
    names = feature_names()
    nb = np.array([n.startswith("nb_") for n in names])
    return {var: round(float(m.feature_importances_[nb].sum()), 4) for var, m in zip(VARIABLES, bank.models)}


def evaluate(bank: TreeBankV3, data: dict, sids: list[str]) -> dict:
    X, Y, M, _, R = _rows(data, sids)
    pred, pmask = bank.predict(X)
    out = {}
    for j, var in enumerate(VARIABLES):
        sel = (M[:, j] == 1.0) & (pmask[:, j] == 1.0)
        own = X[:, OWN_VALUE[var]]
        persist = np.where(np.isnan(own), bank.train_means[j], own)
        mean = np.full(len(X), bank.train_means[j])
        def metrics(p, s):
            if not s.any():
                return None
            err = p[s] - Y[s, j]
            d = {"n": int(s.sum()), "mae": round(float(np.mean(np.abs(err))), 4),
                 "rmse": round(float(math.sqrt(np.mean(err ** 2))), 4)}
            if j == EXIT:
                d["brier"] = round(float(np.mean(err ** 2)), 4)
                d["accuracy"] = round(float(np.mean((p[s] >= 0.5) == (Y[s, j] >= 0.5))), 4)
            return d
        entry = {"tree": metrics(pred[:, j], sel), "latest_report_baseline": metrics(persist, sel),
                 "train_mean_baseline": metrics(mean, sel), "by_regime": {}}
        for reg in sorted(set(R)):
            s = sel & (R == reg)
            entry["by_regime"][reg] = {"tree": metrics(pred[:, j], s), "latest_report_baseline": metrics(persist, s)}
        out[var] = entry
    return out


def save(bank: TreeBankV3, report: dict, directory: Path) -> Path:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} exists; model artifacts are never overwritten")
    directory.mkdir(parents=True)
    joblib.dump({"models": bank.models, "hyperparams": bank.hyperparams, "train_means": bank.train_means,
                 "version": bank.version, "feature_names": feature_names()}, directory / "tree_bank.joblib")
    (directory / "tree_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return directory


def load(directory: Path) -> TreeBankV3:
    blob = joblib.load(Path(directory) / "tree_bank.joblib")
    if blob["feature_names"] != feature_names():
        raise ValueError("tree bank feature schema does not match eh-v3-topo-features-1")
    return TreeBankV3(blob["models"], blob["hyperparams"], blob["train_means"], blob["version"])
