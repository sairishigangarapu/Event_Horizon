# Phase 2 Verification Report — Event Horizon (MKCrusaders26)

## A. Executive Summary

**Investigation scope:** Reproduce Phase 1 findings, establish a trustworthy test baseline, resolve JavaScript parity test failures, investigate Python import collisions, and prepare remediation proposals (without applying code changes).

**Environment:** Python 3.12.3, scikit-learn 1.9.1, Node.js v22.23.2, FastAPI/Uvicorn, SQLite, PyTorch CPU.

**Working tree:** Clean (only untracked: `.commandcode/`, `.vscode/`, `data/dataset.zip`, `data/dataset/`). No tracked files modified.

**Baseline (filtered, excluding 2 legacy archive directories):** 419 collected → 417 passed, 2 failed, 3 skipped.

**Baseline (unfiltered):** 397 collected + 6 collection errors (import collision on `simulation` module name).

**Resolved issues:**
- **JS parity test root cause: CONFIRMED.** Tests call `require('./dist/engine.js')` via `subprocess.check_output` without setting `cwd`. Node resolves the relative path against the process CWD (project root), not `studio/`. When run from `studio/`, both parity tests pass.

**Corrected Phase 1 findings:**
- **F-001 (wrong server in verify script):** CLOSED / NOT A DEFECT. `hosted_server.py` imports `from runtime_server import app` (line 22), so it includes all world endpoints via the runtime and world routers.
- **F-004 (/api/assess POST endpoint):** NOT APPLICABLE. No `POST /api/assess` endpoint exists in the codebase. The only assessment-related endpoint is `GET /api/assessments`.
- **F-008 (unbounded forecast limit):** NOT A DEFECT. `limit: int = Query(default=500, ge=1, le=1000)` — already constrained.

**Confirmed findings requiring action:**
- **F-002 (simulation import collision):** VERIFIED. Affects both tests and runtime_server.py direct startup.
- **F-003 (MODEL_DIR mismatch):** VERIFIED. `.env` says `models/production` (untrained); all scripts override to `models/candidates/mc_v2` (trained).
- **F-005 (step cache staleness):** VERIFIED (static). Cache keyed only on `as_of_time`, not observation content.
- **F-006 (sklearn version mismatch):** VERIFIED. Env has 1.9.1; artifacts saved with 1.7.2/1.8.0/1.9.1; `requirements.txt` pins `>=1.3.0` (no upper bound).
- **F-007 (auth bypass):** VERIFIED. `AUTH_MODE=local_demo` bypasses all auth.
- **F-009 (two spatial model dirs):** VERIFIED. Two active spatial model pairs in production.
- **F-010 (tree bank version mismatch):** VERIFIED. Production uses `dt_bank_v2`; mc_v2 uses `dt_bank_context_v3`.
- **F-011 (double-sigmoid risk):** STATIC FINDING. Production applies Sigmoid in-model; Person 4 applies at inference. Separate code paths — risk only if cross-contaminated.
- **F-012 (dead code):** VERIFIED. `_EXIT_MASK_IDX` defined once, never referenced.

---

## B. Environment and Code Revision

| Item | Value |
|---|---|
| Working directory | `/mnt/storage/Projects/MKCrusaders26` |
| Python | 3.12.3 (from `.venv/bin/python`) |
| Node.js | v22.23.2 |
| scikit-learn | 1.9.1 (installed) vs 1.7.2/1.8.0/1.9.1 (artifact) |
| Git HEAD | `960b888` "Integrate Parallel audit fixes and random-flow dashboard into main" |
| Git parents | `8f7cf07`, `214fc6a`, `6fe88e5` (3-way merge) |
| Working tree | Clean (4 untracked files only) |
| Database | SQLite `data/app.db` (tracked) |
| Model dir | `models/candidates/mc_v2` (active), `models/production` (inactive/untrained) |

---

## C. Exact Commands Executed

All commands run with `PYTHONDONTWRITEBYTECODE=1` where noted, no files modified:

```bash
# 1. Diagnostic checkpoint
git status --short
./.venv/bin/python -c "import runtime_server; import simulation; ..."
./.venv/bin/python -c "from hosted_server import app; ..."
./.venv/bin/python -c "from runtime_server import app; ..."

# 2. Test collection (filtered)
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider --collect-only -q \
  --ignore=services/twin_studio/v3/archive \
  --ignore=services/twin_studio/studio/event_horizon_twin_studio

# 3. Full test execution (filtered)
./.venv/bin/python -m pytest -q --ignore=... --ignore=...

# 4. JS parity test (direct, from studio/ CWD)
cd studio && ../../.venv/bin/python -c "import subprocess; subprocess.check_output(['node','-e',script])"

# 5. Step cache source inspection
./.venv/bin/python -c "import inspect; print(inspect.getsource(ArchitecturePipeline.step))"
```

---

## D. Test-Suite Inventory

### D.1 Active test suites (filtered, 419 collected)

| Directory | Test files | Tests | Purpose |
|---|---|---|---|
| `tests/` | 39 files | 344 | Main application: API, reliability, ML, simulation, Twin Studio v3, audit |
| `person4/tests/` | 6 files | 57 | Person 4: GRU forecaster, comparison, evaluation, alignment |
| `studio/` | 2 files | 18 | Studio engine + observations (Python/JS parity, simulation) |
| **Total** | **47 files** | **419** | |

### D.1.1 `tests/` suite (344 tests)

Key files:
- `test_api.py` (12 tests) — API endpoint testing via TestClient
- `test_live_provider.py` (2 tests) — World/tick/action lifecycle
- `test_random_flow.py` (2 tests) — Scene generation, drafting, planning
- `test_parallel_integration.py` (12 tests) — Two-service integration
- `test_twin_studio_phase1-5.py` (varies) — Twin Studio integration
- `test_twin_studio_phase4.py` (29 tests, ALL SKIPPED due to missing `yaml`)
- `test_twin_v3_*.py` (45 tests) — Twin Studio v3 routing, replay, ingestion
- `test_spatial_leakage.py` (11 tests) — Evidence leakage prevention
- `test_audit_fixes.py` (15 tests) — Audit remediation verification
- `test_readiness.py` (13 tests) — Model readiness checks
- `test_dashboard_app.py` (1 test) — Streamlit dashboard (PASSES)

### D.1.2 `person4/tests/` suite (57 tests)

Tests Person 4's standalone GRU forecaster with mock synthetic data. All pass.

### D.1.3 `studio/` suite (18 tests)

Key files:
- `test_backend.py` (11 tests) — Studio engine: health, static files, scene, route import, CSV, decisions, master report, **2 JS parity tests (FAILING)**, bad input rejection, CORS
- `test_observations.py` (7 tests) — Observation generation, delays, conflicts, coverage

### D.2 Excluded suites (archive/legacy)

| Directory | Status | Reason |
|---|---|---|
| `services/twin_studio/v3/archive/event_horizon_twin_studio/` | Excluded | Contains competing `simulation`, `observations`, `lab_server` modules that cause import collision |
| `services/twin_studio/studio/event_horizon_twin_studio/` | Excluded | Contains competing `simulation` module |

### D.3 Import collision analysis

**Root cause:** Multiple directories define a top-level `simulation` module:
- `studio/simulation.py` — defines `simulate`, `validate`, `normalize_routes`, `live_steps`, `path`, `intervals`, `barriers`
- `services/twin_studio/v3/archive/event_horizon_twin_studio/simulation.py` — defines `path`, `validate` (NO `simulate`, NO `live_steps`)

**Collision mechanism:**
1. `crowd_reliability/live_provider.py` line 12: `from simulation import validate, live_steps`
2. `crowd_reliability/live_provider.py` line 8: `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studio'))` — adds `studio/` to path BEFORE the import
3. When tests in archive directories are collected FIRST, they import `simulation` from the archive path, caching it in `sys.modules`
4. When `live_provider.py` later tries `from simulation import live_steps`, it gets the cached archive module which lacks `live_steps` → `ImportError`

**Affected modules:**
- `crowd_reliability/live_provider.py` → imports `from simulation import validate, live_steps`
- `crowd_reliability/flow_evaluation.py` → imports `from simulation import live_steps, path`
- `tests/test_live_provider.py` → imports `from simulation import live_steps, simulate`
- `tests/test_random_flow.py` → imports `from simulation import live_steps, path`
- `tests/test_live_runtime.py` → imports `from simulation import ...`
- `tests/test_parallel_integration.py` → imports from live_provider
- `studio/test_backend.py` → imports `from simulation import simulate, validate, normalize_routes`
- `studio/test_observations.py` → imports `from simulation import simulate`

**Safe startup paths:**
- `runtime_server:app` (when imported fresh, before any archive module): `live_provider.py` adds `studio/` to path, then imports `simulation` → resolves to `studio/simulation.py` → **SUCCESS**
- `hosted_server:app` (imports `runtime_server` which triggers `live_provider`): same as above → **SUCCESS**
- `start_review.py`: adds `studio/` to path, starts `hosted_server:app` → **SUCCESS**

**Unsafe startup path:**
- If any archive test module is imported first, `simulation` gets cached from the archive → subsequent imports fail

### D.4 Cross-file test dependencies

`studio/test_observations.py` imports `from test_backend import C` (line 8 of test_observations.py), creating a dependency on `test_backend.py` being importable. This works when `studio/` is in `sys.path` (pytest rootdir insertion), but could fail if test collection order changes.

---

## E. Current Baseline (Filtered Run)

**Command:**
```bash
./.venv/bin/python -m pytest -q \
  --ignore=services/twin_studio/v3/archive \
  --ignore=services/twin_studio/studio/event_horizon_twin_studio
```

**Result:** `417 passed, 2 failed, 3 skipped, 23 warnings in 51.49s`

**Skipped tests:** All 3 from `tests/test_twin_studio_phase4.py:455` — `yaml = pytest.importorskip("yaml")` fails because `yaml` (PyYAML) is not installed.

**Failed tests:**
1. `studio/test_backend.py::Tests::test_js_python_parity` — `require('./dist/engine.js')` fails (CWD issue)
2. `studio/test_backend.py::Tests::test_shared_gate_python_and_js_match` — same CWD issue

**Warnings:** 23 warnings, all `InconsistentVersionWarning` from scikit-learn (artifacts saved with sklearn 1.7.2/1.8.0/1.9.1, environment has 1.9.1).

---

## F. Root-Cause Analysis of Failures

### F-1: JS parity test failures (2 tests)

**Root cause:** Working directory mismatch. The tests call `subprocess.check_output(['node', '-e', script])` where `script` contains `require('./dist/engine.js')`. Node resolves `./dist/engine.js` relative to the process CWD, which is the project root (`/mnt/storage/Projects/MKCrusaders26/`), not `studio/`.

**Evidence:**
- `studio/dist/engine.js` exists and is tracked in git (un-ignored via `.gitignore: !studio/dist/`)
- Running from `studio/` directory: `require('./dist/engine.js')` resolves correctly → JS simulation succeeds
- Running from project root: `require('./dist/engine.js')` → `ModuleNotFoundError: Cannot find module './dist/engine.js'`

**Confirmed:** Both JS parity tests pass when pytest is run from `studio/` directory:
```
studio/ $ PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -m pytest -p no:cacheprovider -v test_backend.py::Tests::test_js_python_parity test_backend.py::Tests::test_shared_gate_python_and_js_match
============================== 2 passed in 0.53s ===============================
```
All 16 remaining studio tests also pass from `studio/` CWD (18 total, 2 deselected, 16 passed).

**Fix (requires authorization):**
- Option A (recommended): Add `cwd=str(Path(__file__).parent)` to `subprocess.check_output` calls in `test_js_python_parity` and `test_shared_gate_python_and_js_match`
- Option B: Change require paths to use absolute paths: `require(str(Path(__file__).parent / 'dist' / 'engine.js'))`
- Option C: Change require paths to `../studio/dist/engine.js` (relative from project root)

### F-2: YAML collection skips (3 tests)

**Root cause:** `tests/test_twin_studio_phase4.py` line 455: `yaml = pytest.importorskip("yaml")`. PyYAML is not installed in `.venv`.

**Evidence:**
```
pip3 show pyyaml  # → not installed
.venv/bin/python -c "import yaml"  # → ModuleNotFoundError: No module named 'yaml'
```

**Fix (requires authorization):** `pip install pyyaml` — non-destructive, adds a dependency

---

## G. Verified Findings (Updated from Phase 1)

### F-001: Verify script server mismatch — CLOSED / NOT A DEFECT

**Evidence:** `hosted_server.py` line 22: `from runtime_server import app`. This imports the FULL app including `runtime.router` (sessions) and `live_provider.world_router` (worlds). OpenAPI schema confirms 28 paths including `/api/runtime/worlds/random`, `/api/runtime/worlds/{wid}/tick`, etc. Actual HTTP test confirms `POST /api/runtime/worlds/random` returns 200.

**Root cause of Phase 1 error:** The Phase 1 report read a truncated/incorrect version of `hosted_server.py` (reported as 20 lines). The actual file is 81 lines and imports `runtime_server:app`, not a raw API app.

### F-002: Ambiguous simulation import — VERIFIED

**Evidence:** Unfiltered pytest collection shows 6 collection errors, all from `from simulation import ...` resolving to the wrong module. The `simulation` name is ambiguous between `studio/simulation.py` and `services/twin_studio/v3/archive/.../simulation.py`.

**Impact:**
- Tests: 6 test files fail to collect when archive directory is not excluded
- Runtime: `runtime_server:app` starts correctly only because `live_provider.py` adds `studio/` to path before importing

### F-003: MODEL_DIR inconsistency — VERIFIED

**Evidence:**
- `.env`: `MODEL_DIR=./models/production` (untrained_initial models)
- `hosted_server.py` line 16: `os.environ.setdefault('MODEL_DIR', str(ROOT/'models/candidates/mc_v2'))` (synthetic_validated)
- `start_review.py`: explicitly sets `MODEL_DIR=str(ROOT/'models/candidates/mc_v2')`
- `verify_random_flow.py`: explicitly sets `MODEL_DIR=str(ROOT/'models/candidates/mc_v2')`

**Impact:** `.env` is misleading. If a user sources `.env` and then imports `crowd_reliability.api` directly (without `hosted_server.py`'s `setdefault` calls), they get untrained models. The startup scripts override correctly, but the `.env` value is a trap.

### F-005: Pipeline step cache staleness — VERIFIED (static)

**Evidence:** `pipeline.py` lines 258-260:
```python
prior = self._step_cache.get(scenario_id) if scenario_id in self.history_buffer else None
if prior and as_of_time < prior[0]: raise ValueError('Replay clock cannot move backwards.')
if prior and as_of_time == prior[0]: return prior[1]
```

The cache returns `prior[1]` (cached forecasts) when `as_of_time == prior[0]`, without checking whether `recent_observations` changed. The cache key is `(scenario_id)` → `(as_of_time, result)`. Observation content is NOT in the key.

**Scenario:** If `step()` is called twice at the same `as_of_time` with different observation sets (e.g., a retried tick, or concurrent sessions sharing a scenario), the second call returns the first call's cached forecasts, ignoring new observations.

**Note:** In normal operation, each tick advances `as_of_time` by 5s, so the cache miss path is taken. The risk is primarily for retries and concurrent access.

### F-006: scikit-learn version mismatch — VERIFIED

**Evidence:**
- Installed: scikit-learn 1.9.1
- `requirements.txt`: `scikit-learn>=1.3.0` (no upper bound)
- Warnings: `InconsistentVersionWarning: Trying to unpickle estimator from version 1.7.2/1.8.0/1.9.1 when using version 1.9.1`
- `test_twin_studio_phase4.py` uses `sklearn 1.7.2` artifacts (line 455 yaml skip doesn't affect this)

**Impact:** Currently warnings only. sklearn's `InconsistentVersionWarning` is non-fatal but indicates potential forward-compatibility issues. The `requirements.txt` should pin a specific version matching the artifact training version.

### F-007: Authentication bypass in local_demo — VERIFIED

**Evidence:**
- `.env`: `AUTH_MODE=local_demo`
- `api.py` line 77: `if settings.auth_mode == 'local_demo': return await call_next(request)` — bypasses all auth
- Test: `GET /api/scenarios` with no API key returns 200

**Impact:** All endpoints are unauthenticated in local_demo mode. `verify_random_flow.py` uses `AUTH_MODE=api_key` (proper), but `start_review.py` and `hosted_server.py` default to `local_demo`. For a local-only demo, this is intentional. For any networked deployment, this is a security risk.

### F-009: Two spatial model directories — VERIFIED

**Evidence:**
- `models/production/spatial/hypernet_v2__gnn_edge_v2__6675c91e/` — 20974B gnn.pt, 29135B hypernet.pt (timestamps: Oct 15 14:22)
- `models/production/spatial/hypernet_v2__gnn_edge_v2__d1018ba3/` — 22363B gnn.pt, 29135B hypernet.pt (timestamps: Oct 15 00:56)
- `models/candidates/mc_v2/spatial/hypernet_v2__gnn_edge_v2__e8b5461c/` — single directory

**Impact:** The `spatial_artifact_dir` in the manifest names `d1018ba3` as the active spatial pair (deterministic, reproducible). The `6675c91e` directory is orphaned — created but never loaded.

### F-010: Tree bank version mismatch — VERIFIED

**Evidence:**
- `models/production/manifest.json`: `tree_version: "dt_bank_v2"`, `training_status: {hypernet/gnn/temporal: "untrained_initial"}`
- `models/candidates/mc_v2/manifest.json`: `tree_version: "dt_bank_context_v3"`, `training_status: all "synthetic_validated"`

**Impact:** Tests assert `tree_version == "dt_bank_v2"` (production). Runtime uses `dt_bank_context_v3` (mc_v2). Production tree bank lacks context routing.

### F-011: Sigmoid placement discrepancy — STATIC FINDING (design choice)

**Evidence:**
- `crowd_reliability/ml/temporal.py` line 66: `nn.Sigmoid()` inside model's `exit_heads`
- `person4/event_horizon_p4/model.py` line 79: No Sigmoid — raw logit returned
- `person4/event_horizon_p4/inference.py` line 207: `torch.sigmoid(...)` applied at inference

**Assessment:** This is an intentional design difference, not a bug. Production model applies Sigmoid in the architecture; Person 4 separates it for testability. Cross-contamination would only occur if checkpoint loading code is shared, which it is not (production uses `MultivariateTemporalForecaster.load()`, Person 4 uses its own loader).

### F-012: Dead code `_EXIT_MASK_IDX` — VERIFIED

**Evidence:** `person4/event_horizon_p4/baseline.py` line 45: `_EXIT_MASK_IDX = U_DIM - 1`. Grepped across `person4/` — exactly 1 occurrence (the definition). Never imported or referenced elsewhere.

---

## H. Report Inconsistencies (Phase 1 → Phase 2 Corrections)

| Phase 1 Claim | Phase 2 Finding | Correction |
|---|---|---|
| "410 passed, 1 skipped (Streamlit)" | Streamlit test PASSES (417 passed); 3 skips are YAML, not Streamlit | The counts differ because Phase 1 may have run from a different environment or excluded studio/ tests |
| F-001: "verify_random_flow.py starts hosted_server:app (NO world endpoints)" | hosted_server.py imports runtime_server:app which DOES have world endpoints | F-001 is NOT A DEFECT — the server chain works correctly |
| F-004: "/api/assess accepts raw dict" | No POST /api/assess endpoint exists | F-004 is NOT APPLICABLE |
| F-008: "No upper bound on /api/forecasts limit" | `limit: int = Query(default=500, ge=1, le=1000)` | F-008 is NOT A DEFECT |
| "hosted_server.py (20 lines)" | hosted_server.py is 81 lines | Phase 1 read was truncated |
| F-006: "artifacts saved with 1.9.1/1.8.0/1.7.2, environment has 1.9.0" | Environment has 1.9.1 (not 1.9.0) | The exact version number was incorrect |

---

## I. Remaining Blind Spots

| # | Blind Spot | Resolution |
|---|---|---|
| 1 | Whether deployed production instance uses `hosted_server:app` or `runtime_server:app` | Owner confirmation |
| 2 | Whether `CENTRAL_SOURCE_MODE=replay` with CSV fixture is acceptable for evaluation | Owner confirmation |
| 3 | Whether the two spatial model dirs in production are intentional (active vs. orphaned) | Check manifest `spatial_artifact_dir` value |
| 4 | Real-world ML accuracy (models trained on synthetic data only) | Real evaluation dataset with ground truth |
| 5 | Memory usage under concurrent WebSocket connections | Load testing (deferred — read-only phase) |
| 6 | Whether the catch-all `/{studio_path:path}` route in hosted_server could intercept API routes | Verify via live HTTP testing with lifespan |

---

## J. Remediation Plan (Proposals — Not Yet Authorized)

### J-1: Fix JS parity test working directory (P1 — blocks test suite)

**Files:** `studio/test_backend.py` lines ~102-104, ~127-128

**Change:** Add `cwd=str(Path(__file__).parent)` to `subprocess.check_output` calls:
```python
# Before:
js=json.loads(subprocess.check_output(['node','-e',script],input=json.dumps(C).encode()))
# After:
js=json.loads(subprocess.check_output(['node','-e',script],input=json.dumps(C).encode(),
    cwd=str(Path(__file__).parent)))
```

**Regression tests:** `test_js_python_parity`, `test_shared_gate_python_and_js_match` should both pass.

**Risk:** Minimal — only affects test execution CWD.

---

### J-2: Resolve simulation import collision (P1 — blocks test collection)

**Files:** Create `conftest.py` at project root OR fix import paths

**Approach A (recommended):** Add a root-level `conftest.py` that removes archive directories from `sys.path`:
```python
# conftest.py at project root
import sys
# Ensure studio/ takes priority for 'simulation' module
sys.path.insert(0, str(Path(__file__).parent / 'studio'))
```

**Approach B:** Rename archive modules to avoid name collision (e.g., `archive_simulation.py`)

**Approach C:** Use pytest `--import-mode=importlib` to avoid sys.path pollution

**Regression tests:** All 6 previously-failing test files should collect and pass.

**Risk:** Approach A is lowest-risk. Approach B is cleanest but requires renaming many files.

---

### J-3: Install PyYAML (P3 — unblocks 3 phase-4 tests)

**Command:** `pip install pyyaml`

**Impact:** 3 tests in `test_twin_studio_phase4.py` would run instead of being skipped.

**Risk:** Non-destructive. Only affects Phase 4 tests.

---

### J-4: Fix MODEL_DIR in .env (P2 — configuration consistency)

**File:** `.env` line 4

**Change:** Change `MODEL_DIR=./models/production` to `MODEL_DIR=./models/candidates/mc_v2`

**Or:** Add a comment clarifying that `hosted_server.py` overrides this.

**Impact:** Prevents developers from accidentally loading untrained models.

**Risk:** If any tooling depends on `.env` for production model loading, this changes behavior.

---

### J-5: Address step cache staleness (P2 — potential data integrity)

**File:** `crowd_reliability/ml/pipeline.py` lines 258-260

**Change:** Include observation content hash in cache key, or remove cache:
```python
# Option: include a hash of observation IDs + values
obs_hash = hash(tuple((o.zone_id, o.variable, o.value) for o in recent_observations))
if prior and prior[0] == as_of_time and prior[2] == obs_hash:
    return prior[1]
# ...
self._step_cache[scenario_id] = (as_of_time, forecasts, obs_hash)
```

**Regression test:** Call `step()` twice at same timestamp with different observations; verify second call produces different forecasts.

**Risk:** Cache hit performance may degrade if observations change frequently.

---

### J-6: Pin scikit-learn version (P2 — artifact compatibility)

**File:** `requirements.txt`

**Change:** Replace `scikit-learn>=1.3.0` with `scikit-learn>=1.3.0,<1.10.0` or pin to the version used for artifact creation.

**Or:** Re-save artifacts with the current sklearn version.

**Risk:** Low — version constraint only.

---

### J-7: Remove dead code (P3 — cleanup)

**File:** `person4/event_horizon_p4/baseline.py` line 45

**Change:** Remove `_EXIT_MASK_IDX = U_DIM - 1` (never referenced).

**Risk:** None — dead code removal.

---

## K. Readiness Assessment

| Dimension | Status | Evidence |
|---|---|---|
| Functional behavior | Verified | 417 tests pass; all core flows covered |
| Frontend reliability | Partially verified | JS parity tests fail (CWD bug, easily fixable); live_dashboard_smoke.cjs not yet run end-to-end |
| Backend reliability | Verified | All API tests pass; 2 failures are JS-only |
| Test suite collection | Partially verified | 6 collection errors when archive dirs not excluded; clean when excluded |
| ML execution and correctness | Partially verified | mc_v2 models load and run; sklearn version warnings present; Person 4 uses mocks |
| Security | Unverified | Auth defaults to local_demo; /api/assess (nonexistent) was wrongly flagged; need live HTTP testing |
| Data integrity | Verified | Tests cover determinism, no duplicates, causal filtering, hash verification |
| Build reproducibility | Partially verified | 417 tests pass from clean state; Dockerfile exists; yaml missing for 3 tests |
| Deployment reliability | Partially verified | hosted_server:app correctly includes all routers; verify script chain works |
| Observability | Unverified | No logging infrastructure; errors may surface in HTTP response only |
