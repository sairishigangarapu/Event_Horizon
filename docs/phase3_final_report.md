# Phase 3 Final Report — Event Horizon (MKCrusaders26)

**Date:** 2026-10-15  
**Git HEAD:** `960b888` (0 commits behind/ahead of origin/main)  
**Working tree:** 8 tracked files modified, 4 new test files + conftest.py added

---

## 1. Changes Made and Rationale

### J-1: JS/Python parity test working directory (F-002 test impact)

**File:** `studio/test_backend.py` (2 lines changed)

**Problem:** Two JS parity tests called `subprocess.check_output(['node','-e',...])` with `require('./dist/engine.js')` in the Node script. Node resolves `./dist/engine.js` relative to the process CWD (project root), not `studio/`. The artifact exists and is tracked; only the path was wrong.

**Fix:** Added `cwd=str(Path(__file__).parent)` to both `subprocess.check_output` calls (lines 58, 74). `Path` was already imported.

**Behavioral consequence:** Tests now work from any CWD. The real JS engine is still exercised — no assertions changed, no tests skipped/disabled.

---

### J-2: Simulation import collision (F-002)

**File:** `conftest.py` (new, root-level)

**Problem:** Multiple directories define top-level `simulation.py`. The archived version lacks `live_steps`. When archive tests are collected first, the archived module is cached in `sys.modules`, causing 6 collection errors.

**Fix:** Root `conftest.py` that:
1. Prepends `studio/` to `sys.path[0]` before any test collection — ensures `simulation` resolves to `studio/simulation.py` (has `live_steps`)
2. Declares `collect_ignore` for the two archive directories — `services/twin_studio/v3/archive` and `services/twin_studio/studio/event_horizon_twin_studio`

**Result:** 0 collection errors without any `--ignore` flags. 91 additional tests are now collected (modules that import `simulation` directly).

**Commands documented in conftest.py:**
- Maintained suite: `pytest` (no flags needed)
- Archive suite: `pytest services/twin_studio/v3/archive services/twin_studio/studio/event_horizon_twin_studio`
- Person 4 only: `pytest person4/tests/`
- Studio only: `pytest studio/`
- Main only: `pytest tests/`

---

### J-3: PyYAML test dependency (F-011)

**Files:** `requirements.txt` (1 line added), PyYAML installed in `.venv`

**Problem:** 3 tests in `test_twin_studio_phase4.py` were skipped due to missing `yaml` module (`pytest.importorskip("yaml")`). PyYAML was already declared in `requirements.integration.txt` but not in `requirements.txt` (the dev requirements), so the `.venv` was set up without it.

**Analysis:** `yaml` is imported ONLY in `tests/test_twin_studio_phase4.py` (for Docker compose validation). Not used in any runtime code. **Conclusion: test-only dependency.**

**Fix:** Added `PyYAML>=6.0,<7` to `requirements.txt` (dev/test requirements). Installed via `uv pip install 'PyYAML>=6.0,<7' --python ./.venv/bin/python` (PyYAML 6.0.3).

**Result:** 3 tests changed from `skipped` to `passed` (they validate Docker compose security properties). No additional failures revealed.

---

### J-4: MODEL_DIR configuration consistency (F-003)

**Files:** `crowd_reliability/config.py` (1 line), `.env.example` (1 line)

**Problem:** `.env` (gitignored) and `config.py` default both pointed to `models/production` (untrained_initial models). Only startup scripts (`start_review.py`, `verify_random_flow.py`) and `hosted_server.py`'s `setdefault` overrode to `mc_v2` (trained).

**Analysis:**
- `config.py` default: `./models/production` → untrained
- `.env` (gitignored): `MODEL_DIR=./models/production` → overrides config default via pydantic-settings `env_file`
- `.env.example` (tracked): `MODEL_DIR=./models/production` → misleading template
- `hosted_server.py`: `os.environ.setdefault('MODEL_DIR', mc_v2)` → only sets if NOT in env; if `.env` sourced, production is used
- `start_review.py` / `verify_random_flow.py`: explicitly set `MODEL_DIR=mc_v2` (overrides)

**Fix:**
1. Changed `config.py` default from `./models/production` to `./models/candidates/mc_v2` — new users without `.env` get trained models
2. Changed `.env.example` from `./models/production` to `./models/candidates/mc_v2` — correct template for new users
3. **Did NOT modify `.env`** (gitignored, user-specific — respect user's explicit choice)

**Policy established:**
- Class default: `mc_v2` (trained) — safe for local dev/demo
- `.env`: user's explicit configuration — takes precedence (even if it says production)
- Production deployment: must explicitly set `MODEL_DIR` (e.g., `MODEL_DIR=./models/production` in deployment env)
- `hosted_server.py`'s `setdefault` remains as a safety net for direct uvicorn startup without `.env`

---

### J-5: Pipeline step-cache correctness (F-005)

**File:** `crowd_reliability/ml/pipeline.py` (cached store, 2 edits in `step()`)

**Problem:** `ArchitecturePipeline.step()` cached forecasts keyed only on `(scenario_id, as_of_time)`. When called twice at the same timestamp with different observation inputs, the cache returned stale results without recomputing.

**Root cause:** Cache check at original line 260:
```python
if prior and as_of_time == prior[0]:return prior[1]
```
No observation content was in the cache key.

**Fix:** Added an observation fingerprint to the cache key:
1. Compute `obs_fingerprint = hash(frozenset(...))` over all semantic observation fields (zone_id, variable, value, unit, source_id, observed_at, published_at, received_at)
2. Cache hit only if `as_of_time == prior[0] AND prior[2] == obs_fingerprint`
3. Cache store: `self._step_cache[scenario_id] = (as_of_time, (tree_records, forecast_records), obs_fingerprint)`

**Fields deliberately excluded from fingerprint:** `report_id` (volatile metadata, not a forecasting input — different report_ids with same content produce identical forecasts).

**Fingerprint properties:**
- **Includes:** every field that affects `select_reports` filtering or `_zone_inputs` feature extraction
- **Excludes:** `report_id`, `id` (ORM PK), `scenario_id` (already in cache key)
- **Ordering consistent:** Uses `frozenset` of tuples (order-independent)
- **Immutable:** observations are treated as read-only; the fingerprint captures their effective content

**Performance impact:** Minimal — `hash(frozenset(...))` is O(n) where n is observation count (typically <100 per zone per tick). Cache still hits for true retries (identical observations, same timestamp).

---

### J-6: scikit-learn compatibility (F-006)

**File:** `requirements.txt` (1 line changed)

**Analysis:**
- Installed: scikit-learn 1.9.1
- `requirements.deploy.txt`: `scikit-learn==1.9.1` (with comment: "committed tree bank and scaler were pickled with it")
- `requirements.lock.txt`: `scikit-learn==1.9.1`
- `requirements.runtime.txt`: `scikit-learn==1.8.0` (outdated)
- `requirements.txt` (dev): `scikit-learn>=1.3.0` (too broad)
- mc_v2 artifacts: pickled with 1.7.2 (tree bank) and 1.8.0 (scaler, KMeans)
- Production artifacts: pickled with 1.7.2 (tree bank), 1.8.0 (scaler/KMeans)

**Warnings observed:** `InconsistentVersionWarning` — non-fatal (sklearn explicitly allows this with a warning). No loading or inference errors.

**Fix:** Tightened `requirements.txt` from `>=1.3.0` to `>=1.9.1,<1.10.0` — matches installed, deploy, and lock versions.

**Not done (justified):**
- Did NOT regenerate artifacts (heavy compute, not explicitly supported workflow)
- Did NOT change `requirements.runtime.txt` (intentional 1.8.0 pin for railway production runtime; production artifacts are not the active inference path)
- Did NOT silence warnings via code — they are informative and non-fatal

**Status:** Known compatibility risk documented. Warnings persist but are non-fatal. Recommended next step: regenerate mc_v2 artifacts under sklearn 1.9.1 in a dedicated PR after load testing.

---

### J-7: Dead code removal (F-012)

**File:** `person4/event_horizon_p4/baseline.py` (3 lines removed)

**Removed:** `_EXIT_MASK_IDX = U_DIM - 1` (line 45, plus its docstring comment)

**Verification:** `grep -rn "_EXIT_MASK_IDX" person4/` → exactly 1 occurrence (the definition). Never imported or referenced in any source or test.

---

### F-007: Authentication hardening (auth policy)

**Files:** `crowd_reliability/auth.py` (rewritten `check_api_key`), `crowd_reliability/api.py` (1 line added)

**Problem:** `local_demo` mode bypassed ALL authentication via `check_api_key` returning immediately. If the service was bound to a network interface, any request would be accepted without a key.

**Fix:**
1. Added `client_host` parameter to `check_api_key`
2. In `local_demo` mode: allow localhost (`127.0.0.1`, `::1`, `localhost`, `testclient`) without API key; reject non-localhost with 403
3. In `api_key` mode: unchanged behavior (401 missing key, 401 wrong key, 503 unconfigured)
4. Middleware passes `request.client.host` to `check_api_key`

**TestClient compatibility:** Added `testclient` to `_LOCALHOST_HOSTS` (Starlette TestClient uses this as the client host).

**Policy established:**
- `local_demo` mode: works for localhost only (development/demo)
- `api_key` mode: required for network-facing deployment
- Fail safe: `api_key` mode without `app_api_key` configured → 503
- No credentials in logs (uses `hmac.compare_digest` for constant-time comparison)

---

## 2. Files Modified (8 tracked + 4 new + 1 new)

| File | Change type | Lines changed |
|---|---|---|
| `studio/test_backend.py` | Modified (J-1) | 2 insertions, 2 deletions |
| `requirements.txt` | Modified (J-3, J-6) | 2 insertions, 2 deletions |
| `crowd_reliability/ml/pipeline.py` | Modified (J-5) | 14 insertions, 2 deletions |
| `crowd_reliability/config.py` | Modified (J-4) | 1 insertion, 1 deletion |
| `.env.example` | Modified (J-4) | 1 insertion, 1 deletion |
| `crowd_reliability/auth.py` | Modified (F-007) | 16 insertions, 5 deletions |
| `crowd_reliability/api.py` | Modified (F-007) | 2 insertions, 1 deletion |
| `person4/event_horizon_p4/baseline.py` | Modified (J-7) | 0 insertions, 3 deletions |
| `conftest.py` | New (J-2) | 38 lines |
| `tests/test_import_collision_regression.py` | New (J-2) | 64 lines |
| `tests/test_step_cache_regression.py` | New (J-5) | 139 lines |
| `tests/test_model_dir_config.py` | New (J-4) | 74 lines |
| `tests/test_auth_policy.py` | New (F-007) | 115 lines |

---

## 3. Test Results: Before vs After

| Metric | Before (Phase 2) | After (Phase 3) | Δ |
|---|---|---|---|
| Collected | 419 | 510 | +91 |
| Passed | 417 | 510 | +93 |
| Failed | 2 | 0 | -2 |
| Skipped | 3 | 0 | -3 |
| Collection errors | 6 (unfiltered) | 0 | -6 |
| Warnings | 23 | 27 | +4 |

**Warning breakdown:** All 27 warnings are `InconsistentVersionWarning` from sklearn (non-fatal). The +4 vs Phase 2 is from additional model loading in new tests.

### Per-suite breakdown (after):

| Suite | Tests | Status |
|---|---|---|
| `tests/` | 435 | 435 passed |
| `person4/tests/` | 57 | 57 passed |
| `studio/` | 18 | 18 passed |
| **Total** | **510** | **510 passed** |

---

## 4. Command Summary

### Run the full maintained test suite:
```bash
cd /mnt/storage/Projects/MKCrusaders26
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider
```

### Run individual suites:
```bash
# Person 4 only
./.venv/bin/python -m pytest -p no:cacheprovider person4/tests/

# Studio only
./.venv/bin/python -m pytest -p no:cacheprovider studio/

# Main tests only
./.venv/bin/python -m pytest -p no:cacheprovider tests/

# Auth policy only
./.venv/bin/python -m pytest -p no:cacheprovider tests/test_auth_policy.py

# Cache regression only
./.venv/bin/python -m pytest -p no:cacheprovider tests/test_step_cache_regression.py
```

### Start the application (local demo):
```bash
cd /mnt/storage/Projects/MKCrusaders26
# Demo (local_demo auth, mc_v2 models, localhost only):
./.venv/bin/python -m uvicorn hosted_server:app --host 127.0.0.1 --port 8000

# Interactive review (separate venv):
python start_review.py
```

---

## 5. Outstanding Risks

| Risk | Severity | Impact | Mitigation |
|---|---|---|---|
| sklearn version warnings | Low | Non-fatal; `InconsistentVersionWarning` only | Document; recommend artifact regeneration under 1.9.1 in dedicated PR |
| Local `.env` overrides model dir | Low | User's local `.env` may still point to `models/production` (untrained) | Documented in `test_effective_model_dir_reports_what_is_loaded`; `.env.example` corrected |
| Auth: `local_demo` allows any localhost | Medium | If service exposed on network, localhost restriction doesn't help | Use `api_key` mode for network-facing; bind to `127.0.0.1` |
| Orphaned spatial model dir (F-009) | P3 | Wasted disk; no functional impact (manifest selects correct dir) | Investigate after core correctness work |
| Archive test suites excluded | P3 | 2 legacy directories not collected by default | Documented in conftest.py; can run explicitly |

---

## 6. Intentionally NOT Changed

| Item | Reason |
|---|---|
| `.env` file | Gitignored, user-specific; user's explicit configuration takes precedence |
| `requirements.runtime.txt` sklearn pin (1.8.0) | Intentional for railway production runtime; not the active inference path |
| `requirements.lock.txt` | Already at 1.9.1; no change needed |
| Model artifacts | Not regenerated (heavy compute; non-fatal warnings) |
| `hosted_server.py` `setdefault` | Redundant with config.py default change but harmless; kept as safety net |
| Archive directories | Not deleted; excluded via conftest.py `collect_ignore` |
| `runtime_server.py` | No changes needed — imports `crowd_reliability.api` and `runtime.router`; model loading uses `settings.model_dir` which now defaults to mc_v2 |

---

## 7. Finding Status Summary

| Finding | Phase 1 Status | Phase 2 Status | Phase 3 Action | Status |
|---|---|---|---|---|
| F-001 | Verified | Closed / Not a defect | No action | Closed |
| F-002 | Suspected | Verified | Fixed (J-2: conftest.py) | Resolved |
| F-003 | Static | Verified | Fixed (J-4: config default) | Resolved |
| F-004 | Verified | Not applicable | No action | Closed |
| F-005 | Static | Verified | Fixed (J-5: cache fingerprint) | Resolved |
| F-006 | Verified | Verified | Fixed (J-6: version pin) | Mitigated |
| F-007 | Static | Verified | Fixed (F-007: localhost restriction) | Resolved |
| F-008 | Static | Not a defect | No action | Closed |
| F-009 | Verified | Verified | Not addressed (orphaned dir; P3) | Deferred |
| F-010 | Verified | Verified | Not treated as defect (intentional tiers) | Closed |
| F-011 | Suspected | Static finding | Fixed (J-3: PyYAML install) | Resolved |
| F-012 | Static | Verified | Fixed (J-7: dead code removed) | Resolved |

---

## 8. Adversarial Test Coverage

The new regression tests cover these adversarial scenarios:

| Test | Adversarial scenario |
|---|---|
| `test_cache_is_invalidated_when_observations_change` | Same timestamp, different observations → must recompute (spy on model forward pass) |
| `test_cache_hit_returns_same_result_for_identical_inputs` | Same timestamp, same observations → cache hit (model not called) |
| `test_active_simulation_module_has_live_steps` | Correct `simulation` module loaded (not archive) |
| `test_simulation_module_path_is_not_archive` | Module path does not contain "archive" |
| `test_simulation_module_api_surface` | All expected API functions present |
| `test_live_provider_import_chain_loads_correct_simulation` | Import chain through `live_provider` resolves correctly |
| `test_class_default_model_dir_is_mc_v2` | Default config points to trained model |
| `test_explicit_env_var_takes_precedence` | Explicit env var overrides default |
| `test_mc_v2_manifest_is_trained` | Loaded manifest has trained status |
| `test_production_manifest_is_untrained` | Documents production = untrained |
| `test_check_api_key_local_demo_blocks_non_localhost` | 403 for non-localhost in local_demo |
| `test_check_api_key_local_demo_allows_localhost` | No exception for localhost in local_demo |
| `test_correct_key_allowed` / `test_missing_key_rejected` / `test_wrong_key_rejected` / `test_unconfigured_key_fails_safely` | api_key mode edge cases |
| `test_api_key_mode_via_http` | Full HTTP auth flow with TestClient |
