# Event Horizon — Person 4: Temporal Forecast, Comparison & Evaluation

> **⚠️ DISCLAIMER**: Synthetic implementation scaffold only.  
> This is **NOT** a trained or validated disaster-forecasting system.  
> All values and model outputs are for interface, tensor-shape, and software smoke tests.  
> **Do NOT use for real emergency decisions or claim real-world accuracy.**

---

## Overview

This package implements Person 4's responsibilities from the **Event Horizon 5-Person Workload Plan** (Manipal Hackathon 2026):

| Responsibility | Module |
|---|---|
| GRU input assembly `U[z,t]` | `sequence.py` |
| GRU forecaster (6 outputs × 3 horizons) | `model.py` |
| Synthetic scenario generation | `synthetic.py` |
| Smoke-test training & checkpoint saving | `train.py` |
| Checkpoint loading + inference | `inference.py` |
| Persistence & moving-average baselines | `baseline.py` |
| Stream A demo fixture adapter | `central_adapter.py` |
| Stream A/B channel-wise comparison | `comparison.py` |
| Delayed reference label evaluation | `evaluation.py` |
| End-to-end demo runner | `demo.py` |
| Central configuration | `config.py` |

---

## Mock Contract

| Parameter | Value | Note |
|---|---|---|
| Sample interval | 5 s | Fixed — changing invalidates stored history |
| History window | 60 steps | 5 s × 60 = 300 s look-back |
| Horizons | 30 s / 60 s / 120 s | Offsets: 6, 12, 24 steps |
| Zones (mock) | 6 | Reconcile with Person 1 venue map |
| Observation channels X | 6 | `density, speed, inflow, outflow, queue_length, exit_open` |
| Tree vector D | 6 | Must match Person 3 tree output channels |
| Spatial embedding H | **8 (MOCK)** | Final GNN output dim not fixed in workplan |
| Age features | 6 | One normalised age per observation channel |
| Observation masks | 6 | 1 = fresh (< 90 s), 0 = stale / missing |
| Tree masks | 6 | 1 = applicable, 0 = inapplicable |
| Exit mask | 1 | 1 for exit zones, 0 otherwise |
| **Total U_DIM** | **39** | Asserted at import time |

---

## GRU Input Layout (U[z,t])

```
index:  0     5 | 6    11 | 12   19 | 20   25 | 26   31 | 32   37 | 38
        X (6)   | D (6)   | H (8)   | age (6) | obs_m(6)| tree_m(6)| exit_m(1)
```

**Key rules**:
- D (tree vector) is passed **directly**, even though D also enters the hypernetwork dynamic context.
- No reference/ground-truth data ever enters `assemble_u_vec()`.
- Observations with `received_at > issue_time` are rejected (future-data guard).

---

## Setup

Python 3.10+ is recommended.

```bash
# From the person4/ directory:
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e '.[test]'
```

---

## Run Tests

```bash
cd person4/
pytest -q
```

Test files cover:
- `test_tensor_shapes.py` — U_DIM=39, WindowBatch shapes, model output `(Z, H, 6)`, exit applicability masking
- `test_forecast_alignment.py` — pairing rules, unit mismatch, unavailable labels, causal filter, no merged trust probability
- `test_sequence_timestamps.py` — out-of-order rejection, future-data exclusion, stale masking, padding
- `test_evaluation_delay.py` — `label_available_at` enforcement, MAE/RMSE, exit classification metrics

---

## Smoke-Test Training

```bash
cd person4/
python -m event_horizon_p4.train --epochs 2 --output artifacts/temporal_mock.pt
```

Writes:
- `artifacts/temporal_mock.pt` — model checkpoint (explicitly labelled `synthetic_fixture_only_NOT_VALIDATED`)
- `artifacts/temporal_mock_scaler.npy` — fitted StandardScaler parameters
- `artifacts/temporal_mock_manifest.json` — training provenance metadata

---

## End-to-End Demo

```bash
cd person4/
python -m event_horizon_p4.demo --checkpoint artifacts/temporal_mock.pt --epochs 2
```

Writes to `data/fixtures/person4/`:
- `mock_architecture_forecasts.json` — Stream B (architecture) forecast sample
- `mock_central_demo_fixture.json` — Stream A demo fixture (⚠️ not real data)
- `mock_comparison_differences.json` — per-channel signed & absolute differences
- `mock_evaluation_metrics.json` — MAE/RMSE + exit Brier score
- `mock_evaluation_metrics.csv` — CSV export for notebook ingestion
- `mock_persistence_baseline_comparison.json` — persistence baseline vs Stream A

---

## Module Map

```
person4/
├── pyproject.toml
├── README.md
├── event_horizon_p4/
│   ├── __init__.py
│   ├── config.py            ← All constants and dimension annotations
│   ├── synthetic.py         ← Reproducible synthetic scenario generator
│   ├── sequence.py          ← Rolling window assembly, U[z,t] constructor
│   ├── model.py             ← GRU model + MaskedMixedLoss
│   ├── train.py             ← Smoke-test training loop + checkpoint saving
│   ├── inference.py         ← TemporalForecaster + ForecastRecord
│   ├── baseline.py          ← Persistence + moving-average baselines
│   ├── central_adapter.py   ← Stream A demo fixture (⚠️ explicitly labelled)
│   ├── comparison.py        ← Channel-wise A/B comparison
│   ├── evaluation.py        ← Delayed label scoring + CSV export
│   └── demo.py              ← End-to-end smoke test runner
└── tests/
    ├── __init__.py
    ├── test_tensor_shapes.py
    ├── test_forecast_alignment.py
    ├── test_sequence_timestamps.py
    └── test_evaluation_delay.py
```

---

## Key Engineering Rules

1. **Reference states are targets only.** `forecast_window` / `assemble_u_vec` never accept reference data.
2. **At issue time, only `received_at <= issue_time` observations may enter model input.**
3. **Forecast records carry** `issue_at`, `received_at`, `target_at`, `zone_id`, `variable`, `unit`, `stream_id`, `model_version`.
4. **Stream A remains explicitly labelled** as a demo fixture until a genuine provider is configured.
5. **Exit applicability mask**: `exit_open` records are **omitted** (not set to zero) for non-exit zones.
6. **Comparisons join on** `(scenario_id, zone_id, variable, unit, target_at)` only; unit mismatches are skipped.
7. **Reference metrics are computed only after** `label_available_at <= evaluation_time`.
8. **Model raw output**: channels 0-4 = scaled regression (ReLU); channel 5 = raw exit logit (Sigmoid applied at inference).
9. **No merged "trust probability"** is published as the sole comparison result.

---

## Integration Points with Other Persons

| Person | What Person 4 needs | How to update |
|---|---|---|
| Person 1 | Real observation records (replace `MockObservation`) | Update `sequence.py::_build_x_and_masks()` |
| Person 2 | Fixed-expiry information baseline contract | Align in `baseline.py` |
| Person 3 | Real tree vector D (replace `MockTreeOutput`) and GNN embedding H | Update `config.py::D_DIM`, `H_DIM`; update `sequence.py::assemble_u_vec()` |

---

*Manipal Hackathon 2026 • Round 2 • Event Horizon • Person 4*
