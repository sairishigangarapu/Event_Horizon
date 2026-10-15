"""Initial training of the Twin v3 hypernetwork + GNN + GRU (Phase G, eh-v3-neural-1).

Architecture and dimensions are the existing, versioned ones (unchanged, not Monte Carlo selected):
SpatialEncoder (hypernetwork Vn[Z,16], Ve[E,4] >= 0; two-layer edge-aware GNN H[Z,16]) and
MultivariateTemporalForecaster (2-layer GRU, 40 inputs, horizons 30 s and 60 s).

Inputs per zone and instant (all from authorized snapshots; see dataset.py):
  spatial dynamic context (36): observation value/age/present/fresh per variable, tree values and masks
  U (40) = value x present (6), min(age / 120 s, 1) (6), present (6), tree value x tree mask (6), H (16)
Tree values are the out-of-sample (cross-fitted) outputs of the v3 tree bank, never in-sample fits.
Targets: hidden zone-state values at T + h (offline labels), masked where undefined.
Loss: masked Huber on per-variable scaled numeric targets (scale = training std, no centring because
the numeric heads are non-negative) + binary cross-entropy for exit_open on gate zones.
Selection: early stopping on validation scenarios. Test scenarios are evaluated once.

Status rule (recorded with the checkpoint): "trained_validated" only if, on validation scenarios at
30 s, Stream B beats the best observation-only baseline (latest report, report EWMA, held tree
estimate) on at least 3 of the 5 numeric variables; otherwise "trained_not_validated".
The hypernetwork/GNN feedback loop (candidate copy, gated promotion) is not part of this pass.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crowd_reliability.ml.graph import SpatialGraphBuilder
from crowd_reliability.ml.spatial import SpatialEncoder, build_spatial_features, fit_normalization
from crowd_reliability.ml.temporal import MultivariateTemporalForecaster
from services.twin_studio.pipeline_adapter import venue_layout
from services.twin_studio.v3.features import VARIABLES
from services.twin_studio.zones import load_venue

VERSION = "eh-v3-neural-1"
U_SCHEMA = "eh-v3-u-1"
HORIZONS_S = (30, 60)
STEP_S = 5
LOOKBACK = 12
EXIT = VARIABLES.index("exit_open")
NUMERIC = [j for j in range(len(VARIABLES)) if j != EXIT]
EWMA_ALPHA = 0.35
EWMA_LOOKBACK_S = 90.0


def graph_builder(venue: dict | None = None) -> SpatialGraphBuilder:
    venue = venue or load_venue()
    layout = venue_layout(venue)
    return SpatialGraphBuilder(layout.zones, layout.connections, map_version=venue["map_version"])


def scenario_tensors(d: dict, tree_values: np.ndarray, tree_masks: np.ndarray) -> dict:
    """Per-scenario arrays: spatial context [T,Z,36] + valid, non-spatial U part [T,Z,24]."""
    values = np.nan_to_num(d["obs_values"], nan=0.0)
    ages = np.nan_to_num(d["obs_ages"], nan=0.0)
    present, fresh = d["obs_present"], d["obs_fresh"]
    feats, valid = [], []
    for t in range(values.shape[0]):
        f, v = build_spatial_features(values[t], ages[t], present[t], fresh[t], tree_values[t], tree_masks[t])
        feats.append(f)
        valid.append(v)
    age_norm = np.where(present > 0, np.minimum(ages / 120.0, 1.0), 1.0)
    u24 = np.concatenate([values * present, age_norm, present, tree_values * tree_masks], axis=2).astype(np.float32)
    return {"feats": np.stack(feats), "valid": np.stack(valid), "u24": u24,
            "Y": d["Y"].astype(np.float32), "M": d["M"].astype(np.float32), "times": d["times"]}


def _windows(n_t: int) -> np.ndarray:
    """Index of the LOOKBACK steps ending at each t (left-padded by repeating step 0)."""
    idx = np.arange(n_t)[:, None] - np.arange(LOOKBACK - 1, -1, -1)[None, :]
    return np.maximum(idx, 0)


class Model(torch.nn.Module):
    def __init__(self, graph: SpatialGraphBuilder, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = SpatialEncoder(graph, seed=seed)
        self.hypernet, self.gnn = self.enc.hypernet, self.enc.gnn
        self.temporal = MultivariateTemporalForecaster(input_dim=24 + self.enc.embed_dim, hidden_dim=64, num_layers=2,
                                                       num_horizons=len(HORIZONS_S), horizons_sec=list(HORIZONS_S))

    def spatial(self, feats: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feats/valid [B, Z, 36] -> H [B, Z, 16] (block-diagonal batch of graphs)."""
        b, z, f = feats.shape
        g = self.enc.graph
        offsets = (torch.arange(b) * z).repeat_interleave(g.num_edges)
        edge_index = g.edge_index.repeat(1, b) + offsets
        vn, ve = self.hypernet(feats.reshape(b * z, f), g.node_static.repeat(b, 1), edge_index,
                               g.edge_attr.repeat(b, 1), dyn_valid=valid.reshape(b * z, f))
        return self.gnn(vn, edge_index, ve).reshape(b, z, -1)

    def forward_scenario(self, s: dict) -> torch.Tensor:
        """Forecasts for every instant of one scenario: [T, Z, H, 6] (exit head = probability)."""
        feats, valid = torch.from_numpy(s["feats"]), torch.from_numpy(s["valid"])
        H = self.spatial(feats, valid)                                   # [T, Z, 16]
        U = torch.cat([torch.from_numpy(s["u24"]), H], dim=2)           # [T, Z, 40]
        win = torch.from_numpy(_windows(U.shape[0]))
        seq = U[win]                                                     # [T, L, Z, 40]
        t, l, z, f = seq.shape
        out = self.temporal(seq.permute(0, 2, 1, 3).reshape(t * z, l, f))
        return out.reshape(t, z, len(HORIZONS_S), -1)


def _targets(s: dict):
    """Labels at T + h and masks (0 where T + h is beyond the run)."""
    n_t = s["Y"].shape[0]
    Y = np.zeros((n_t, s["Y"].shape[1], len(HORIZONS_S), len(VARIABLES)), dtype=np.float32)
    M = np.zeros_like(Y)
    for k, h in enumerate(HORIZONS_S):
        off = h // STEP_S
        if off < n_t:
            Y[: n_t - off, :, k] = np.nan_to_num(s["Y"][off:], nan=0.0)
            M[: n_t - off, :, k] = s["M"][off:]
    return torch.from_numpy(Y), torch.from_numpy(M)


def _loss(pred, Y, M, scale):
    num = [j for j in NUMERIC]
    p, y, m = pred[..., num] / scale, Y[..., num] / scale, M[..., num]
    huber = F.huber_loss(p, y, reduction="none", delta=1.0)
    l_num = (huber * m).sum() / m.sum().clamp(min=1.0)
    pe, ye, me = pred[..., EXIT].clamp(1e-5, 1 - 1e-5), Y[..., EXIT], M[..., EXIT]
    l_exit = (F.binary_cross_entropy(pe, ye, reduction="none") * me).sum() / me.sum().clamp(min=1.0)
    return l_num + l_exit


def train(data: dict, oos: dict, *, seed: int = 0, epochs: int = 120, patience: int = 12, lr: float = 1e-3,
          batch_scenarios: int = 4, log=print) -> tuple[Model, dict]:
    graph = graph_builder()
    sids = {s: sorted(k for k, d in data.items() if d["split"] == s) for s in ("train", "val", "test")}
    tens = {sid: scenario_tensors(data[sid], oos[sid]["values"], oos[sid]["masks"]) for sid in data}
    tr_feats = np.concatenate([tens[s]["feats"].reshape(-1, 36) for s in sids["train"]])
    tr_valid = np.concatenate([tens[s]["valid"].reshape(-1, 36) for s in sids["train"]])
    norm = fit_normalization(tr_feats, tr_valid, scaler_id=f"{VERSION}_train_split")
    ys = np.concatenate([data[s]["Y"].reshape(-1, len(VARIABLES)) for s in sids["train"]])
    ms = np.concatenate([data[s]["M"].reshape(-1, len(VARIABLES)) for s in sids["train"]])
    scale = torch.tensor([max(float(np.nanstd(ys[ms[:, j] > 0, j])), 1e-3) for j in NUMERIC], dtype=torch.float32)

    model = Model(graph, seed)
    model.enc.set_normalization(norm)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    targets = {sid: _targets(tens[sid]) for sid in data}
    rng = np.random.default_rng(seed)
    best, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(sids["train"])
        t0, tr_loss = time.time(), 0.0
        for i in range(0, len(order), batch_scenarios):
            opt.zero_grad()
            loss = sum(_loss(model.forward_scenario(tens[s]), *targets[s], scale) for s in order[i:i + batch_scenarios])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss)
        model.eval()
        with torch.no_grad():
            val = float(np.mean([float(_loss(model.forward_scenario(tens[s]), *targets[s], scale)) for s in sids["val"]]))
        history.append({"epoch": epoch, "train_loss": tr_loss / max(1, len(order)), "val_loss": val, "seconds": time.time() - t0})
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"epoch {epoch} train {history[-1]['train_loss']:.4f} val {val:.4f} ({history[-1]['seconds']:.1f}s)")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    report = {"version": VERSION, "u_schema": U_SCHEMA, "horizons_s": list(HORIZONS_S), "lookback_steps": LOOKBACK,
              "seed": seed, "splits": sids, "history": history, "best_val_loss": best, "target_scale": scale.tolist(),
              "normalization_scaler_id": norm["scaler_id"]}
    report["val_metrics"] = evaluate(model, data, tens, sids["val"])
    report["test_metrics"] = evaluate(model, data, tens, sids["test"])
    report["status_rule"] = ("trained_validated if Stream B beats the best observation-only baseline on validation "
                             "at 30 s for >= 3 of 5 numeric variables")
    wins = [v for v in ("density", "speed", "inflow", "outflow", "queue_length")
            if report["val_metrics"]["30"][v]["stream_b"]["mae"] <= min(report["val_metrics"]["30"][v][b]["mae"]
                                                                           for b in ("latest_report", "report_ewma", "tree_hold"))]
    report["validation_wins_30s"] = wins
    report["training_status"] = "trained_validated" if len(wins) >= 3 else "trained_not_validated"
    return model, report


def _report_ewma(d: dict) -> np.ndarray:
    """Observation-only EWMA of distinct reports (reconstructed from per-instant latest value + age)."""
    times, vals, ages, present = d["times"], d["obs_values"], d["obs_ages"], d["obs_present"]
    n_t, n_z, n_v = vals.shape
    out = np.full_like(vals, np.nan)
    for z in range(n_z):
        for v in range(n_v):
            seen = []
            for t in range(n_t):
                if present[t, z, v]:
                    obs = times[t] - ages[t, z, v]
                    if not seen or seen[-1][0] != obs:
                        seen.append((obs, vals[t, z, v]))
                window = [x for o, x in seen if o >= times[t] - EWMA_LOOKBACK_S]
                if window:
                    level = window[0]
                    for x in window[1:]:
                        level = EWMA_ALPHA * x + (1 - EWMA_ALPHA) * level
                    out[t, z, v] = level
    return out


def evaluate(model: Model, data: dict, tens: dict, sids: list[str]) -> dict:
    res = {str(h): {v: {} for v in VARIABLES} for h in HORIZONS_S}
    acc = {(h, v, k): [] for h in HORIZONS_S for v in VARIABLES for k in ("stream_b", "latest_report", "report_ewma", "tree_hold")}
    for sid in sids:
        s, d = tens[sid], data[sid]
        with torch.no_grad():
            pred = model.forward_scenario(s).numpy()
        Y, M = (x.numpy() for x in _targets(s))
        latest = np.where(d["obs_present"] > 0, d["obs_values"], np.nan)
        ewma = _report_ewma(d)
        tree = s["u24"][..., 18:24]
        for k, h in enumerate(HORIZONS_S):
            for j, v in enumerate(VARIABLES):
                sel = M[:, :, k, j] > 0
                truth = Y[:, :, k, j][sel]
                for name, p in (("stream_b", pred[:, :, k, j]), ("latest_report", latest[..., j]),
                                ("report_ewma", ewma[..., j]), ("tree_hold", tree[..., j])):
                    acc[(h, v, name)].append((p[sel], truth))
    for (h, v, name), parts in acc.items():
        p = np.concatenate([a for a, _ in parts]) if parts else np.array([])
        y = np.concatenate([b for _, b in parts]) if parts else np.array([])
        ok = ~np.isnan(p)
        if name != "stream_b":
            p = np.where(ok, p, np.nanmean(p) if ok.any() else 0.0)     # baselines without evidence: their own mean
        err = p - y
        entry = {"n": int(len(y)), "mae": round(float(np.mean(np.abs(err))), 4) if len(y) else None,
                 "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4) if len(y) else None,
                 "coverage": round(float(ok.mean()), 4) if len(ok) else None}
        if v == "exit_open" and len(y):
            entry["brier"] = round(float(np.mean(err ** 2)), 4)
        res[str(h)][v][name] = entry
    return res


def save(model: Model, report: dict, directory: Path, tree_version: str) -> Path:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} exists; model artifacts are never overwritten")
    status = report["training_status"]
    model.enc.stamp(training_status=status)
    model.enc.save(str(directory / "spatial"))
    model.temporal.version = "gru_forecaster_v3_twin"
    model.temporal.save(str(directory / "temporal_model.pt"))
    report = report | {"tree_version": tree_version, "pair_id": model.enc.hypernet.metadata["pair_id"],
                       "map_version": model.enc.graph.map_version}
    (directory / "neural_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return directory


def load(directory: Path) -> tuple[Model, dict]:
    directory = Path(directory)
    report = json.loads((directory / "neural_report.json").read_text(encoding="utf-8"))
    model = Model(graph_builder(), seed=report["seed"])
    model.enc.load(str(directory / "spatial"))
    model.temporal = MultivariateTemporalForecaster.load(str(directory / "temporal_model.pt"))
    model.eval()
    return model, report
