# Attack Surface and Architecture Map

## Repository Overview

The **Crowd Disaster Information Reliability** system is a Python-based crowd-disaster forecasting and reliability assessment pipeline. It generates synthetic crowd simulation data, trains ML models (tree banks, hypernetworks, GNNs, GRU temporal forecasters), serves forecasts via a FastAPI API, and provides a Streamlit dashboard for evaluation.

### Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Runtime | Python | 3.12 |
| Simulation | Seeded Python zone/agent simulation | custom |
| Data handling | NumPy, pandas | 2.5.4, 3.0.6 |
| Tree bank | scikit-learn DecisionTreeRegressor | 1.9.1 |
| Graph learning | PyTorch, PyTorch Geometric | 2.14.1 |
| Temporal | PyTorch GRU (2 layers, dropout=0.1) | 2.14.1 |
| API | FastAPI, Uvicorn | 0.143.0 |
| Dashboard | Streamlit, Plotly | 1.65.0 |
| Persistence | SQLite (sqlalchemy) | — |
| Config | Pydantic Settings (env vars) | 2.14.1 |
| Testing | pytest | — |
| Dependency scan | pip-audit | No known vulnerabilities |

### Dependency Vulnerability Scan

```
$ pip-audit
No known vulnerabilities found
```

Installed versions: torch 2.14.1+cu130, fastapi 0.143.0, pydantic 2.14.1, pandas 3.0.6, numpy 2.5.4, sklearn 1.9.1, streamlit 1.65.0.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI ENTRY POINTS (cli.py)                   │
│  ┌───────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐ │
│  │db-init    │  │seed-demo     │  │replay      │  │evaluate  │ │
│  │train-trees│  │spatial-init  │  │monte-carlo │  │export    │ │
│  └─────┬─────┘  └──────┬───────┘  └──────┬─────┘  └────┬─────┘ │
│        │               │                  │              │         │
└────────┼───────────────┼──────────────────┼──────────────┼─────────┘
         │               │                  │              │
         ▼               ▼                  ▼              ▼
  ┌──────────┐  ┌─────────────┐    ┌────────────────┐  ┌──────────┐
  │ database │  │ ml/pipeline │    │ ml/pipeline    │  │ ml/tree  │
  │  (SQLite)│  │             │    │  .step()       │  │   _train │
  │          │  │  load_      │    │  └───────────┐  │  │          │
  │ zones    │  │  pipeline() │    │  hypernet    │  │  │  train_  │
  │ conns    │  │  save_      │    │  encode_     │  │  │  tree_   │
  │ obs      │  │  pipeline() │    │  spatial()   │  │  │  _bank   │
  │ refs     │  └──────┬──────┘    │  temporal    │  │  └──────────┘
  │ forecasts│         │           │  .forward()  │  │
  │ assess  │         │           └──────┬───────┘  │
  └──────────┘         │                  │          │
                       │                  │          │
         ┌─────────────┼──────────────────┼──────────┘
         │             │                  │
         │             │  ┌───────────────▼──────────────┐
         │             │  │ ml/temporal.py               │
         │             │  │  MultivariateTemporalForecaster│
         │             │  │  ┌─────────────────────────┐ │
         │             │  │  │ GRU (2 layers,         │ │
         │             │  │  │  dropout=0.1)          │ │
         │             │  │  │  ┌─────┐ ┌─────┐      │ │
         │             │  │  │  │GRU 1│→│GRU 2│      │ │
         │             │  │  │  └─────┘ └─────┘      │ │
         │             │  │  └─────────────────────────┘ │
         │             │  │  NO .eval() CALLED          │
         │             │  │  (BUG: dropout active      │
         │             │  │  during inference)          │
         │             │  └──────────────────────────────┘
         │             │
         │             │  ┌──────────────────────────────┐
         │             │  │ ml/spatial.py                │
         │             │  │  SpatialEncoder              │
         │             │  │  ┌─────────────┐            │
         │             │  │  │Hypernet MLP │→ edge_vals │
         │             │  │  │GNN (GCN)    │→ node_emb  │
         │             │  │  └─────────────┘            │
         │             │  │  NO .eval() CALLED          │
         │             │  └──────────────────────────────┘
         │             │
         │             │  ┌──────────────────────────────┐
         │             │  │ ml/tree_bank.py              │
         │             │  │  DecisionTreeBank            │
         │             │  │  ┌─────────────────────────┐ │
         │             │  │  │6-channel tree vectors   │ │
         │             │  │  │predict_zone() → heuristics│ │
         │             │  │  │if not is_fitted          │ │
         │             │  │  └─────────────────────────┘ │
         │             │  └──────────────────────────────┘
         │             │
         └─────────────┴──────────────────────────────────┐
                                                         │
┌─────────────────────────────────────────────────────┐  │
│                    API (api.py)                     │  │
│  FastAPI + Uvicorn @ 127.0.0.1:8000                │  │
│  ┌──────────────┬──────────────┬───────────────┐    │  │
│  │ /api/forecasts │ /api/assessments │ /tree-outputs │    │  │
│  │ NO AUTH       │ NO AUTH        │ NO AUTH   │    │  │
│  │ NO RATE LIMIT │ Uses comparator │ NO AUTH │    │  │
│  │ O(N) PAGINATION│ NOT canonical  │ (BUG)   │    │  │
│  └──────┬───────┴────────────────┘               │  │
│         │                                          │  │
│         │  ┌───────────────────────────────────┐   │  │
│         │  │ CentralSourceAdapter (imported    │   │  │
│         │  │ but ignores settings)            │   │  │
│         │  └───────────────────────────────────┘   │  │
│         │                                          │  │
│         │  ┌───────────────────────────────────┐   │  │
│         │  │ _TREE_BANK_CACHE (thread-unsafe)  │   │  │
│         │  └───────────────────────────────────┘   │  │
│         │                                          │  │
│         │  ┌───────────────────────────────────┐   │  │
│         │  │ init_db() on every startup        │   │  │
│         │  └───────────────────────────────────┘   │  │
│         │                                          │  │
│         └──────────────────────────────────────────┘  │
│                                                      │
│  CORSMiddleware (allow_credentials=True)              │
│  NO auth middleware                                    │
│  NO rate limiting                                      │
│  NO TLS                                                  │
│  NO exception handlers                               ||
└─────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    DASHBOARD (dashboard/app.py)               │
│  Streamlit + Plotly @ 127.0.0.1:8501                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ st.markdown(unsafe_allow_html=True)                     │  │
│  │ f"<b>{z.name}</b>" — XSS surface                       │  │
│  │ z.geometry.get("x", 0) — crashes if None               │  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ assess_information() without zone_type/scenario_type   │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   ARTIFACT STORAGE                            │
│  models/                                                        │
│  ├── production/                                               │
│  │   ├── tree_bank.joblib          (dt_bank_v2, has scaler)   │
│  │   ├── temporal_model.pt         (UNTRAINED)               │
│  │   ├── manifest.json            (is_active=true, but       │
│  │   │                              training_status=untrained)│
│  │   └── spatial/hypernet_v2__gnn_edge_v2__d1018ba3/         │
│  │       ├── hypernet.pt  (UNTRAINED)                         │
│  │       ├── gnn.pt     (UNTRAINED)                           │
│  │       └── spatial_manifest.json                            │
│  ├── trees/                                                  │
│  │   ├── tree_bank_dt_bank_v2.joblib                          │
│  │   └── scaler_scaler_obs_only_v1.joblib                    │
│  └── demo/                                                   │
│      └── spatial_output_sample.json                          │
│                                                              │
│  NO integrity verification for missing artifacts → silent    │
│  fallback to unfitted defaults                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Trust Boundaries

### 1. Client ↔ API (HTTP)
- **Boundary:** Internet/local network → FastAPI process
- **Controls:** None (no auth, no TLS, no rate limiting)
- **Finding ref:** F-004, F-007, F-020, F-030

### 2. API ↔ Database (SQLite)
- **Boundary:** Application → local file-based database
- **Controls:** SQLAlchemy ORM (SQL injection not possible via ORM); foreign keys disabled
- **Finding ref:** F-016 (FK pragmas off)

### 3. API ↔ Model Artifacts (filesystem)
- **Boundary:** Application → model files in `models/`
- **Controls:** SHA-256 hash verification in load_pipeline (for known artifacts)
- **Finding ref:** F-001, F-003 (missing artifacts silently fall back), F-036 (FileNotFoundError unhandled)

### 4. CLI ↔ Database
- **Boundary:** Command line → local database
- **Controls:** None (no confirmation on destructive operations)
- **Finding ref:** F-005 (silent DB writes), F-019 (path traversal), F-034 (stale data)

### 5. Dashboard ↔ API
- **Boundary:** Streamlit process → HTTP calls to FastAPI
- **Controls:** None (no auth on either side)
- **Finding ref:** F-004, F-006

### 6. Data ingestion (CSV → DB)
- **Boundary:** External CSV file → database records
- **Controls:** None (no schema validation, no injection protection)
- **Finding ref:** F-025 (crashes on malformed input)

---

## Data Flow Summary

```
1. Simulation: CrowdSimulator.seed=42 → reference states
                    ↓
2. Observation: ObservationGenerator(seed=42) → observations
                    ↓
3. Ingestion: ingest_dataset() → SQLite DB (observations, reference_states)
                    ↓
4. Tree Training: train_tree_bank() → DecisionTreeBank + FeatureScaler
                    ↓
5. Pipeline Assembly: ArchitecturePipeline
   - Step: observations → tree_bank.predict → spatial features + temporal GRU
                    ↓
6. Forecast Serving: /api/forecasts, /api/assessments (NO AUTH, NO RATE LIMIT)
                    ↓
7. Dashboard: Streamlit → API calls → Plotly charts (XSS surface)
```

### Critical Data Flows

| Flow | Trust Level | Notes |
|---|---|---|
| Seed demo → train trees → export → mc-refit → benchmark | Untrusted | Pipeline not deterministic (F-002); seed=42 in simulator but seed=0 in spatial encoder (F-018) |
| API → DB query → Python sort → slice | Untrusted | O(N) memory load per request; no pagination limits (F-008) |
| CSV import → ForecastRecord | Untrusted | No validation; crashes on malformed values (F-025) |
| Zone data → Dashboard HTML | Untrusted | DB-derived data rendered via unsafe_allow_html (F-006) |
| .env → Config → CentralSourceAdapter | Untrusted | CENTRAL_API_KEY ignored by adapter (F-022); extra='ignore' drops typos (F-021) |

---

## Attack Surface Summary

| Component | Exposure | Vulnerabilities |
|---|---|---|
| `/api/forecasts` | Network (no auth, no TLS) | Unauthenticated data exposure (F-004), DoS via O(N) pagination (F-008), no rate limiting (F-007) |
| `/api/assessments` | Network (no auth) | Uses wrong comparator (F-009), no P2 thresholds (F-009), O(N×V×Z) nested loop (F-009) |
| `/api/scenarios/*` | Network (no auth) | Unauthenticated access, no input validation (F-038) |
| `/api/scenarios/{id}/tree-outputs` | Network (no auth) | Thread-unsafe cache (F-015), no auth (F-004) |
| `/health` | Network (no auth) | Health check reveals system state |
| `cmd_replay` | Local CLI (any privileges) | Silent DB writes (F-005), wrong spatial seed (F-018) |
| `cmd_dataset_ingest` | Local CLI (any privileges) | Path traversal (F-019), destructive without confirmation (F-020) |
| Dashboard | Local network (Streamlit) | XSS surface (F-006), crash on None geometry (F-017) |
| Model artifacts | Filesystem (read-only in normal use) | Silent fallback to untrained model (F-001, F-003), hash verification only for known files (F-036) |
| Configuration | .env file | Dead API key (F-035), silent env var dropping (F-021) |
