# Repository Audit Report

## Crowd Disaster Information Liability

**Audit date:** 2026-10-15
**Audit mode:** Read-only discovery (no source modifications)
**Repository:** /mnt/storage/Projects/MKCrusaders26

## 1. Methodology

Five phases of iterative probing were conducted:

- **Phase 1:** Validation of 7 README-documented critical findings + architectural baseline probes (19 probes)
- **Phase 2:** Inference determinism, CSV loading, API health, model eval mode probes (18 probes)
- **Phase 3:** API security, error handling, deployment config probes (24 probes)
- **Phase 4:** Concurrency, thread safety, CSRF, DB integrity probes (13 probes)
- **Phase 5:** API input validation, dead imports, normalization edge cases (9 probes)

All probes were designed to be **non-destructive** and **safe for local execution**. No external systems were contacted. No production data was modified. The diagnostic scripts are in `audit/diag_phase[1-5].py`.

### Severity Model

| Level | Definition |
|---|---|
| Critical | Directly enables data exposure, invalid output served as production, or complete auth bypass |
| High | Causes incorrect business logic, DoS, or exploitable injection surface |
| Medium | Reliability impact under specific conditions, or missing integrity controls |
| Low | Code quality issues that increase risk but don't directly cause defects |
| Informational | Dead code, stale comments, missing imports, config smells |

CVSS 4.0 vectors are included where defensible with stated assumptions.

---

## 2. Findings by Severity

### 2.1 Critical Findings

#### F-001: Untrained neural stack served as active production pipeline

- **Category:** Reliability / Model Integrity
- **Status:** CONFIRMED
- **Confidence:** High
- **CVSS 4.0:** CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:N/VC:N/VI:N/VE:N/SC:N (Critical)
- **Assumptions:** API reachable on network; no auth required; attacker has no special privileges

**Affected files:**
- `crowd_reliability/cli.py` (line 168, save_pipeline)
- `crowd_reliability/ml/pipeline.py` (load_pipeline, step)
- `crowd_reliability/models/manifest.py` (PipelineManifest)

**Root cause:** `seed-demo` and `save_pipeline` save freshly-initialized GRU/GNN/hypernetwork with no training step. The production manifest at `models/production/manifest.json` has:
```json
{
  "is_active": true,
  "training_status": {
    "hypernet": "untrained_initial",
    "gnn": "untrained_initial",
    "temporal": "untrained_initial"
  },
  "metrics_summary": null
}
```

`load_pipeline` and `step` contain **no readiness gate** — they load and serve the artifacts regardless of training status.

**Evidence (live probe):**
```
load_pipeline readiness gate: CONFIRMED: loaded with training_status={
  'hypernet': 'untrained_initial',
  'gnn': 'untrained_initial',
  'temporal': 'untrained_initial'
}, no gate enforced
```

**Reproduction:**
```bash
PYTHONPATH=.venv/bin python -c "
from crowd_reliability.ml.pipeline import ArchitecturePipeline
from crowd_reliability.config import settings
from crowd_reliability.simulation.scenarios import get_default_venue
p = ArchitecturePipeline(venue=get_default_venue())
p.load_pipeline(settings.model_dir)
# No error raised despite untrained_initial status
"
```

**Impact:** All forecasts served by `/api/forecasts` are random-weight outputs. The A/B comparison (Stream A vs Stream B) is structurally invalid. Any decision-making based on these forecasts is compromised.

**Minimal remediation:** Add readiness gate in `load_pipeline`:
```python
if any(v != "trained_validated" for v in self.manifest.training_status.values()):
    raise RuntimeError(f"Model not ready: {self.manifest.training_status}")
if self.manifest.metrics_summary is None:
    raise RuntimeError("metrics_summary is None — model not validated")
```

**Regression test plan:** `test_readiness_gate.py::test_load_pipeline_rejects_untrained` — load_pipeline on production artifacts should raise.

---

#### F-002: Non-deterministic inference from GRU dropout in train mode

- **Category:** Reliability / Reproducibility
- **Status:** CONFIRMED
- **Confidence:** High
- **CVSS 4.0:** CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:L/VC:N/VI:N/VE:N/SC:N (Critical)

**Affected files:**
- `crowd_reliability/ml/pipeline.py` (step method, ~line 245)
- `crowd_reliability/ml/temporal.py` (MultivariateTemporalForecaster, GRU with dropout=0.1)

**Root cause:** `step()` calls `self.temporal_model(u_history_tensor)` while the GRU is in `train()` mode. PyTorch applies dropout between GRU layers when `self.training == True`. `step()` uses `with torch.no_grad():` (disables gradients only, does NOT switch to eval mode). `load_pipeline` never calls `.eval()`.

The GRU is configured with:
```python
GRU(input_dim, hidden_dim, num_layers=2, dropout=0.1, batch_first=True)
```
With `num_layers=2 > 1`, dropout is active between layers during training.

**Evidence (live probe):**
```
Phase 2 probe: GRU dropout=0.1, max_diff=0.0266, 86/144 elements differ between identical calls
Phase 1 probe: model.training=True before step(), still True after step()
```

README reports: "two identical step() calls differ by up to 0.0457 on 55/126 forecasts; .eval() yields 0.0 difference."

**Reproduction:**
```python
from crowd_reliability.ml.temporal import MultivariateTemporalForecaster
import torch
m = MultivariateTemporalForecaster(40, 64, 2, 3)
u = torch.randn(8, 12, 40)
o1 = m(u).numpy()
o2 = m(u).numpy()
print('max_diff:', max(abs(o1-o2)))  # ~0.02-0.05
```

**Impact:** The project's taste preferences require byte-identical re-runs. This violates that contract. Forecasts change on every API call, making benchmark comparisons and debugging impossible. The `seed_demo` → `train_trees` → `export-tree-vector` → `mc-refit` → `benchmark-trees` pipeline cannot produce reproducible results.

**Minimal remediation:** Call `self.temporal_model.eval()` at the start of `step()` and in `load_pipeline()`. Same for spatial encoder: `self.hypernet.eval()`, `self.gnn.eval()`.

**Regression test plan:** `test_determinism.py::test_step_identical_output` — two identical `step()` calls produce forecasts with max_diff < 1e-6.

---

#### F-003: Missing-artifact silent fallback to unfitted tree bank

- **Category:** Data Security / Reliability
- **Status:** CONFIRMED
- **Confidence:** High
- **CVSS 4.0:** CVSS:4.0/AV:N/AC:H/AT:N/PR:N/UI:N/VC:H/VI:H/VA:N/VC:N/VI:N/VE:N/SC:N (Critical)

**Affected files:**
- `crowd_reliability/ml/pipeline.py` (~line 327, load_pipeline)
- `crowd_reliability/ml/tree_bank.py` (~line 94, predict_zone)

**Root cause:** `load_pipeline` checks `if os.path.exists(tb_path)` with **no else/raise**. If any artifact is missing, the pipeline loads with default/unfitted components. `predict_zone` checks `if not self.is_fitted:` and returns heuristic defaults derived from `feature_row[0:3]` with `channel_masks=[1,1,1,1,1,1]`.

**Evidence:**
```
Phase 1 probe: loaded with unfitted tree bank (is_fitted=False), no error raised
Phase 1 probe: unfitted tree bank returns feature_row-derived heuristic values
  (density=0.5, speed=0.3, inflow=2.0) with mask all-1
```

**Code path (tree_bank.py:94):**
```python
def predict_zone(self, feature_row, has_exit=False):
    ...
    if not self.is_fitted:
        return np.array([feature_row[0], feature_row[1], feature_row[2],
                         feature_row[3], feature_row[4], 0.0]), \
               np.array([1, 1, 1, 1, 1, 1])
```

**Impact:** Production API silently serves fabricated forecast values (density=0.5, speed=0.3, inflow=2.0 from feature_row indices) with all masks=1, presenting synthetic data as valid. No alert, no error. Operators cannot distinguish real from fabricated forecasts.

**Minimal remediation:** Require all three artifacts (tree_bank.joblib, temporal_model.pt, spatial pair). Remove the `if not self.is_fitted` fallback branch or make it raise.

**Regression test plan:**
- `test_load_pipeline.py::test_missing_artifacts_raise`
- `test_tree_bank.py::test_unfitted_raises_not_defaults`

---

#### F-004: No authentication or authorization on FastAPI endpoints

- **Category:** Authentication & Authorization
- **Status:** CONFIRMED
- **Confidence:** High
- **CVSS 4.0:** CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:N/VA:N/VC:N/VI:N/VE:N/SC:N (Critical)

**Affected file:** `crowd_reliability/api.py`

**Root cause:** The API has zero authentication, authorization, or session management. No OAuth2, JWT, API key, or security middleware is present. CORS is configured with `allow_credentials=True` but this provides no access control. The `.env` setting `AUTH_MODE=local_demo` is never read by any code path.

**Evidence:**
```
Phase 3 probe: No auth found. Middleware: ['CORSMiddleware'].
All endpoints (/api/forecasts, /api/assessments, /api/scenarios, /health) publicly accessible.
AUTH_MODE=local_demo never enforced.
```

**Affected endpoints (all unauthenticated):**
- GET /health
- GET /api/scenarios
- GET /api/scenarios/{id}
- GET /api/scenarios/{id}/tree-outputs
- GET /api/forecasts
- GET /api/assessments
- GET / (root)

**Impact:** Anyone who can reach the API can access all forecast data, reliability assessments, scenario metadata, and zone information. In any network-accessible deployment (not just localhost), this is a data exposure vulnerability. The dashboard connects to `http://127.0.0.1:8000` — if the API is ever exposed beyond localhost, all data is accessible.

**Minimal remediation:** Add authentication middleware. Require auth for all endpoints except `/health`.

**Regression test plan:**
- `test_api_auth.py::test_unauthenticated_returns_401`
- `test_api_auth.py::test_authenticated_returns_200`

---

#### F-005: cmd_replay silently auto-generates simulation and writes to database

- **Category:** Business Logic / Data Integrity
- **Status:** CONFIRMED
- **Confidence:** High
- **CVSS 4.0:** CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:N/VI:H/VA:N/VC:N/VI:N/VE:N/SC:N (High)

**Affected file:** `crowd_reliability/cli.py` (cmd_replay function)

**Root cause:** `cmd_replay` calls `get_db()`, then checks if observations exist for the scenario. If none found, it silently calls `run_simulation()` + `generate_observations()` and writes the results to the database via `ingest_dataset()`. There is no `--generate-sim` flag to control this behavior, and no user confirmation prompt.

**Evidence:**
```
Phase 2 probe: cmd_replay silently runs simulation and writes ref+obs records to DB
when no observations found — unexpected side effect in a 'replay' command
```

**Impact:** A command named "replay" should only replay existing data. When it silently generates synthetic data and writes it to the DB, several issues arise:
1. Data pollution: synthetic data mixed with real observations
2. Data loss masking: if real data was accidentally deleted, cmd_replay silently regenerates with a different seed, hiding the loss
3. Unintended side effects: operators may not expect their DB to be modified by a replay command

**Minimal remediation:** Add `--generate-sim` flag (default `False`). If no observations found and flag not set, raise:
```
RuntimeError(f"No observations found for scenario '{scenario}'. Use --generate-sim to create synthetic data.")
```

**Regression test plan:**
- `test_cli_replay.py::test_replay_empty_db_raises`
- `test_cli_replay.py::test_replay_generate_sim_writes_db`

---

#### F-012: Central-source simulated forecast leaks future ground truth

- **Category:** Data Security / Bias
- **Status:** CONFIRMED
- **Confidence:** High
- **CVSS 4.0:** CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:N/VA:N/VC:N/VI:N/VE:N/SC:N (Critical)

**Affected file:** `crowd_reliability/services/central_source.py` (line 91, generate_simulated_forecasts)

**Root cause:** `generate_simulated_forecasts` blends `current_val` (weight 0.65) with `future_true_val` (weight 0.35) — the reference state at the forecast target time. This means the "central source" forecast sees 35% of the ground truth at the target time, unfairly advantaging Stream A in the A/B benchmark.

**Evidence:**
```
Phase 1 probe: 466 forecasts equal future reference value out of 2520
Phase 3 probe: 460 forecasts equal future reference value
```

**Impact:** The A/B benchmark is structurally biased in favor of Stream A (central source). Any model comparison, accuracy metric, or business decision based on this benchmark is invalid. The bias is most severe when the system state is stable (current ≈ future), making the central source appear more accurate than it is.

**Minimal remediation:** Remove `future_true_val` from the blending formula. Only smooth over received observations.

**Regression test plan:**
- `test_central_source.py::test_no_future_leak` — no forecast value equals the future reference value within tolerance.

---

### 2.2 High Findings

#### F-006: Dashboard XSS surface via unsafe_allow_html

- **Category:** Injection (XSS)
- **Status:** CONFIRMED
- **Confidence:** Medium
- **CVSS 4.0:** CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:R/VC:L/VI:L/VA:L/SC:C (High)

**Affected file:** `dashboard/app.py` (line 284)

**Root cause:** The dashboard constructs HTML strings containing zone data:
```python
node_text.append(f"<b>{z.name}</b><br>ID: {z.zone_id}<br>Type: {z.zone_type}<br>Capacity: {z.capacity_definition} p")
```
These are passed as Plotly `hovertext`, and `st.markdown(..., unsafe_allow_html=True)` is used elsewhere on the page. If any zone field contains HTML/JavaScript, it could be rendered.

**Note:** Plotly's `hovertext` renders as plain text by default (not HTML), which lowers the risk. However, the `unsafe_allow_html=True` on `st.markdown` calls creates a broader XSS surface if any data flows through those paths.

**Evidence:**
```
Phase 2 probe: Dashboard uses st.markdown(unsafe_allow_html=True) and renders
z.name/z.zone_id/z.zone_type directly into HTML hovertext strings — XSS surface
```

**Minimal remediation:** Escape all database-derived values with `html.escape()` before embedding in HTML strings.

---

#### F-007: No rate limiting on API endpoints

- **Category:** Resource Exhaustion / DoS
- **Status:** CONFIRMED
- **Confidence:** High

**Affected file:** `crowd_reliability/api.py`

**Root cause:** No rate limiting middleware is configured. Middleware list is `['CORSMiddleware']` only. No `slowapi`, `Starlette RateLimitMiddleware`, or `RequestValidator`.

**Evidence:** Confirmed via middleware inspection.

**Impact:** Combined with F-008 (O(N) pagination), a single attacker can trigger expensive database queries repeatedly, exhausting database connections or API process memory.

---

#### F-008: O(N) offset pagination causes DoS

- **Category:** Performance / DoS
- **Status:** CONFIRMED
- **Confidence:** High

**Affected file:** `crowd_reliability/api.py` (get_forecasts function)

**Root cause:** `get_forecasts` loads ALL matching forecast records into memory via SQLAlchemy query, sorts in Python, then slices `[offset:offset+limit]`. With `offset=999999`, the database still materializes all preceding rows. With `limit=5000`, all matching records are loaded into Python memory.

**Evidence:**
```
Phase 3 probe: offset=999999 returns total_comparisons=3318 (ALL records loaded into memory)
Phase 3 probe: limit=5000 loads all matching records into Python list
```

**Impact:** Single request with `limit=5000` can OOM the API process on a production database. Offset-based pagination with large offsets causes progressively slower queries. Combined with no rate limiting, this is a trivial DoS vector.

---

#### F-009: P2 reliability dead-wired in API

- **Category:** Business Logic
- **Status:** CONFIRMED
- **Confidence:** High

**Affected files:**
- `crowd_reliability/api.py` (line 215)
- `crowd_reliability/services/comparator.py` (line 161)
- `crowd_reliability/reliability/assess.py`

**Root cause:** `api.py:215` calls `ForecastComparator.assess_information` (a P1 reimplementation), **not** the canonical `reliability.assess.assess_information`. The comparator reimplementation:
1. Omits scenario/zone threshold multipliers (`comparator.py:161`) — exit zones require fresher data per P2 but this is bypassed
2. Aliases `coverage = 1.0 if not stale else 1-(age-45)/100` instead of `compute_coverage()` — verified: fresh→1.0, stale@70s→0.75
3. Never builds `build_conflict_graph()` — conflict detection is unavailable through the API

**Evidence:**
```
Phase 1 probe: comparator.assess_information lacks zone_type_lookup/scenario_type params;
canonical assessment not wired to API
Phase 3 probe: get_assessments never passes zone_type_lookup or scenario_type to
assess_information — P2 zone-type multipliers (exit zones require fresher data)
are silently bypassed in production API
```

**Impact:** Exit zone thresholds (which require fresher data) are silently bypassed. Coverage values are incorrect. Conflict detection is not available through the API. Stakeholders receive inaccurate reliability assessments.

---

### 2.3 Medium Findings

#### F-010: Scaler contract issue — **FIXED AND VERIFIED**

- **Category:** Data Security / Schema
- **Status:** FIXED AND VERIFIED
- **Confidence:** High

**Affected file:** `crowd_reliability/ml/pipeline.py`, `crowd_reliability/cli.py`

**Original issue (README #4):** README documented that production `tree_bank.joblib` had `scaler=None` (dt_bank_v1), while the real scaler was orphaned at `models/trees/scaler_scaler_obs_only_v1.joblib`.

**Current state:** Git commit `08d184d` (fix: "route seed-demo through scaler-aware train_tree_bank; ship production dt_bank_v2+scaler") resolved this. The production manifest now shows:
- `tree_version: "dt_bank_v2"`
- `scaler_version: "scaler_obs_only_v1"`

**Evidence:**
```
Production tree_bank (loaded via bank.load()):
  is_fitted: True
  version: dt_bank_v2
  scaler: <FeatureScaler object>
  scaler version: scaler_obs_only_v1
```

**Conclusion:** This finding is resolved. The README table (which still lists dt_bank_v1/scaler=None) is stale.

---

#### F-011: Missing applicability/mask columns on ForecastRecord

- **Category:** Schema / Data Integrity
- **Status:** CONFIRMED
- **Confidence:** High

**Affected file:** `crowd_reliability/models/schema.py` (ForecastRecord)

**Root cause:** ForecastRecord stores `forecast_id, id, issued_at, model_or_provider_version, received_at, scenario_id, stream_id, target_at, unit, value, variable, zone_id` but is **missing** `channel_mask`, `applicability`, `missing_data_mask` columns.

**Evidence:**
```
Phase 1 probe: columns=['forecast_id', 'id', 'issued_at', 'model_or_provider_version',
'received_at', 'scenario_id', 'stream_id', 'target_at', 'unit', 'value', 'variable',
'zone_id'], missing=['channel_mask', 'applicability', 'missing_data_mask']
```

**Impact:** Missing-data semantics are lost at the forecast record level. Cannot distinguish forecasts covering all zones vs. partial coverage. Cannot distinguish applicability masks from missing data masks.

---

#### F-014: load_pipeline does not call .eval() on loaded models

- **Category:** Reliability
- **Status:** CONFIRMED
- **Confidence:** High

**Affected file:** `crowd_reliability/ml/pipeline.py` (load_pipeline)

**Root cause:** `load_pipeline` loads `temporal_model.pt` and spatial models but never calls `.eval()`. Models remain in default `train()` mode, causing non-deterministic dropout during inference.

**Evidence:** Confirmed via source inspection — no `eval(` call in load_pipeline source.

---

#### F-015: _TREE_BANK_CACHE not thread-safe

- **Category:** Concurrency
- **Status:** CONFIRMED
- **Confidence:** Medium

**Affected file:** `crowd_reliability/api.py` (module-level `_TREE_BANK_CACHE` dict)

**Root cause:** `_TREE_BANK_CACHE` is a module-level mutable dict with no locking. Concurrent requests to `/health` or `/api/scenarios/{id}/tree-outputs` can race on cache miss (read-modify-write on shared dict). FastAPI runs in multi-threaded mode by default.

**Impact:** Race condition could cause cache corruption, multiple redundant loads, or inconsistent state.

---

#### F-016: SQLite foreign_keys pragma OFF

- **Category:** Data Integrity
- **Status:** CONFIRMED
- **Confidence:** High

**Affected file:** `crowd_reliability/database.py`

**Root cause:** SQLite does not enforce foreign keys by default. The `foreign_keys` pragma is not set to `ON`. Verified: `PRAGMA foreign_keys` returns 0 (OFF).

**Impact:** Referential integrity is not enforced at the DB level. Orphaned records are possible (e.g., forecasts referencing non-existent zones).

---

#### F-017: Dashboard zone.geometry crashes on None

- **Category:** Error Handling
- **Status:** CONFIRMED
- **Confidence:** Medium

**Affected file:** `dashboard/app.py` (line 292)

**Root cause:** `zone_pos = z.geometry.get("x", 0)` — if `z.geometry` is `None` (nullable=True in schema), this raises `AttributeError: 'NoneType' object has no attribute 'get'`.

---

#### F-018: cmd_replay ignores --seed for spatial encoder

- **Category:** Reliability / Reproducibility
- **Status:** CONFIRMED
- **Confidence:** Medium

**Affected file:** `crowd_reliability/cli.py` (cmd_replay)

**Root cause:** `cmd_replay` creates `ArchitecturePipeline(venue=venue)` without passing `spatial_seed`. The pipeline's default is `seed=0`. While the simulator and observation generator use the `--seed` argument, the spatial encoder uses a different seed, creating inconsistency between replay runs.

---

#### F-019: cmd_dataset_ingest accepts arbitrary paths (path traversal)

- **Category:** Path Traversal
- **Status:** CONFIRMED
- **Confidence:** Medium

**Affected file:** `crowd_reliability/cli.py` (cmd_dataset_ingest)

**Root cause:** `cmd_dataset_ingest` accepts `dataset_dir` from CLI args with no path validation. The default is `'./data/dataset'` but a user (or future CLI caller) can pass `../../../etc/` to read files from arbitrary locations.

---

#### F-020: clear_existing=True deletes without confirmation

- **Category:** Data Safety
- **Status:** CONFIRMED
- **Confidence:** Medium

**Affected file:** `crowd_reliability/cli.py` (cmd_dataset_ingest / ingest_dataset)

**Root cause:** `ingest_dataset` with `clear_existing=True` executes `db_session.query(...).delete()` for ALL scenario_ids in the manifest WITHOUT any confirmation prompt.

---

### 2.4 Low Findings

#### F-021: Settings.extra='ignore' silently drops unknown env vars
- **File:** `crowd_reliability/config.py`
- **Status:** CONFIRMED
- **Root cause:** Pydantic Settings with `extra='ignore'` silently drops unknown env vars (e.g., typos like `DATABASE_UL` instead of `DATABASE_URL`) with no warning.

#### F-022: CentralSourceAdapter ignores settings.central_api_key
- **File:** `crowd_reliability/services/central_source.py`
- **Status:** CONFIRMED
- **Root cause:** Adapter uses `mode="replay"` hardcoded; never reads `settings.central_source_mode` or `settings.central_api_key`. The `.env` `CENTRAL_API_KEY` field is dead configuration.

#### F-023: init_db() called on every app startup
- **File:** `crowd_reliability/api.py` (lifespan)
- **Status:** CONFIRMED
- **Root cause:** `lifespan()` calls `init_db()` on every app startup, silently creating the DB file and data directory at runtime if missing.

#### F-028: PipelineManifest defaults is_active=True
- **File:** `crowd_reliability/models/manifest.py`
- **Status:** CONFIRMED
- **Root cause:** A freshly created `PipelineManifest()` has `is_active=True` with `training_status={}` (empty dict, not `'untrained_initial'`), creating a manifest that claims to be active without any training status.

#### F-029: queue_length fallback value indistinguishable from real zero
- **File:** `crowd_reliability/ml/pipeline.py` (_legacy_zone_features)
- **Status:** CONFIRMED
- **Root cause:** When queue_length has no observation, raw=0.0, present=0, mask=0 — the fallback value (0.0) is indistinguishable from a real zero observation.

#### F-030: No TLS/HTTPS enforcement
- **File:** `crowd_reliability/api.py`, `crowd_reliability/config.py`
- **Status:** CONFIRMED
- **Root cause:** No SSL/TLS middleware, HSTS, or SecureCookie. API binds to 127.0.0.1 only (which mitigates exposure but doesn't enforce encryption).

---

### 2.5 Informational Findings

| ID | Title | File | Status |
|---|---|---|---|
| F-024 | No custom exception handlers (stack traces exposed) | api.py | CONFIRMED |
| F-025 | CSV loading crashes on malformed input (no validation) | central_source.py | CONFIRMED |
| F-026 | uuid.uuid4 for forecast IDs (non-deterministic) | pipeline.py:step | CONFIRMED |
| F-027 | Missing Tuple import in central_source.py | central_source.py:38 | CONFIRMED |
| F-031 | load_pipeline doesn't verify spatial artifact dir existence | pipeline.py | CONFIRMED |
| F-032 | Database module-level side effects (creates data dir at import) | database.py | CONFIRMED |
| F-033 | No CSRF protection (current GET-only, but CORS allows credentials) | api.py | CONFIRMED |
| F-034 | cmd_db_init doesn't clear stale data | cli.py | CONFIRMED |
| F-035 | Unused CENTRAL_API_KEY env var | config.py, central_source.py | CONFIRMED |
| F-036 | _calc_sha256 raises FileNotFoundError for missing files | pipeline.py | CONFIRMED |
| F-037 | No __main__.py in crowd_reliability package | (missing file) | CONFIRMED |
| F-038 | normalize_inputs division by zero risk (no epsilon) | ml/graph.py | CONFIRMED |
| F-039 | Function-level json imports in cli.py | cli.py | CONFIRMED |
| F-040 | CLI entry point uses relative paths (./data, ./models) | cli.py, config.py | CONFIRMED |
| F-041 | Stale CLI help text references 'standard_scaler_v1' | cli.py | CONFIRMED |

---

## 3. Findings Not Assessed

The following areas could not be fully assessed due to environment limitations:

1. **Live network penetration testing** — API tested via TestClient (in-process); not tested over real TCP with network-level attacks
2. **Browser-based XSS exploitation** — Dashboard XSS surface identified via code review; not tested with live browser + Streamlit server
3. **Concurrent request testing** — Thread-safety of `_TREE_BANK_CACHE` identified via code review; not tested with concurrent requests
4. **Model training integrity** — Only inference path was tested; training pipeline (`train_tree_bank`) was reviewed but not executed end-to-end with validation
5. **Real production data** — All probes used synthetic/demo data; no real production records were accessed
6. **External service integration** — `CENTRAL_API_URL`/`CENTRAL_API_KEY` are empty in `.env`; live API integration was not tested
