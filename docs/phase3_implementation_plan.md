# Phase 3 Implementation Plan — Event Horizon (MKCrusaders26)

## 0. Status

**Phase 2 report:** `docs/phase2_verification_report.md` (auth complete, untracked evidence file).
**Working tree:** Clean (4 untracked files only: `.commandcode/`, `.vscode/`, `data/dataset.zip`, `data/dataset/`).
**Git HEAD:** `960b888` — 0 commits behind/ahead of `origin/main`.
**Authorization status:** None yet. This plan awaits owner sign-off before any file edits.

## 1. Count Reconciliation

| Metric | Value | Notes |
|---|---|---|
| Collected (pytest --collect-only) | 419 | Excludes 2 legacy archive dirs |
| Passed | 417 | All collected tests that ran |
| Failed | 2 | Both JS parity tests (F-J1) |
| Skipped | 3 | `test_twin_studio_phase4.py` compose tests (line 455 `importorskip("yaml")`) |
| Total outcomes | 422 (419 + 3) | 3 module-level skips are NOT in collection count |
| Warnings | 23 | All `InconsistentVersionWarning` from sklearn 1.9.1 unpickling 1.7.2/1.8.0 artifacts |

**The Phase 2 report's numbers are internally consistent.** The apparent discrepancy ("419 collected" vs "417+2+3=422") arises because the 3 skipped tests are module-level skips (from `importorskip` at module scope in `test_twin_studio_phase4.py` line 455) that are reported as outcomes but not included in the `--collect-only` count. The 26 keyring tests in the same file ARE collected and pass; only the 3 yaml-dependent compose tests after line 455 are skipped at import time.

**The "29 tests in Phase 4 suite" in the Phase 2 report counts ALL `def test` definitions in the file (26 keyring + 3 compose = 29), of which 26 are collected/pass and 3 are skipped.** No inconsistency exists — it is a descriptive count vs. an outcome count.

**Authoritative filtered baseline command:**
```bash
cd /mnt/storage/Projects/MKCrusaders26
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider \
  --ignore=services/twin_studio/v3/archive \
  --ignore=services/twin_studio/studio/event_horizon_twin_studio \
  -rs --tb=short -q
```
Result: `417 passed, 2 failed, 3 skipped, 23 warnings in 93.89s`

**Unfiltered collection (for import collision verification):**
```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m pytest -p no:cacheprovider --collect-only -q
```
Result: 397 collected + 6 collection errors (import collision on `simulation` module name).

## 2. Proposed Fix Order

| Order | Finding | Priority | Risk | Estimated Effort |
|---|---|---|---|---|
| 1 | J-1: JS parity test CWD (F-002's test impact) | P0 | Low | 5 min |
| 2 | J-3: PyYAML test dependency (F-011) | P1 | Very low | 1 min + 3 min test |
| 3 | J-5: Step cache contract (F-005) | P1 | Medium | 10 min design + impl |
| 4 | J-2: Simulation import collision | P1 | High | 15 min design + impl |
| 5 | J-4: MODEL_DIR consistency | P2 | Medium | 10 min analysis + impl |
| 6 | J-6: scikit-learn compatibility | P1 | Medium | 15 min analysis |
| 7 | J-7: Dead code removal | P3 | Very low | 2 min |
| 8 | F-007: Authentication policy | P1 | High | Design + impl |

## 3. Fix Details

### Fix 1: J-1 — JavaScript/Python Parity Test Working Directory (P0)

**Finding ID:** Phase 1 F-002 (test impact). Phase 2 J-1.

**Files to modify:** `studio/test_backend.py`

**Root cause:** `subprocess.check_output(['node','-e',script])` with `require('./dist/engine.js')` in the script string. Node resolves `./dist/engine.js` relative to CWD (project root), not `studio/`.

**Regression tests:**
- `studio/test_backend.py::Tests::test_js_python_parity` — currently fails from project root
- `studio/test_backend.py::Tests::test_shared_gate_python_and_js_match` — currently fails from project root

**Proposed diff (Option A — `cwd` parameter):**
```python
# In test_js_python_parity and test_shared_gate_python_and_js_match:
# Add cwd=str(Path(__file__).parent) to subprocess.check_output calls
js = json.loads(subprocess.check_output(
    ['node','-e', script],
    input=json.dumps(C).encode(),
    cwd=str(Path(__file__).parent)  # <--- added
))
```

**Behavioral consequence:** Tests now pass regardless of pytest CWD. The JS engine is still exercised (no test logic changes, no skips, no disabled assertions). `studio/dist/engine.js` must still exist (it does, tracked in git).

**Alternative (Option B):** Use absolute path in the Node require string:
```python
require(str(Path(__file__).parent / 'dist' / 'engine.js'))
```
This avoids the CWD issue entirely but requires changing the JS code string.

**Verification:** Run `studio/test_backend.py` from project root — all 11 tests should pass (including 2 formerly failing). Run filtered suite — should go from `417 passed, 2 failed, 3 skipped` to `419 passed, 3 skipped`.

**Dependencies:** None. Can be done first.

---

### Fix 2: J-3 — Missing PyYAML Test Dependency (P1)

**Finding ID:** Phase 1 F-011. Phase 2 J-3.

**Files to inspect:** `requirements.txt`, `requirements.*.txt`, `setup.py` or `pyproject.toml`

**Determination needed:** Is PyYAML a:
- **Required runtime dependency?** (used by inference/serving code)
- **Optional feature dependency?** (only used by deploy/compose tooling)
- **Test/development dependency?** (only used by test code)

**Evidence so far:** `yaml` is used in `tests/test_twin_studio_phase4.py` (line 455) to parse Docker Compose files. These tests validate deployment configuration, not runtime inference.

**Proposed diff:**
- If test-only: Add `PyYAML>=6.0` to `requirements.test.txt` (or `requirements.dev.txt`)
- If runtime: Add `PyYAML>=6.0` to `requirements.txt` and `requirements.runtime.txt`

**Regression tests:**
- `tests/test_twin_studio_phase4.py::test_compose_network_isolation`
- `tests/test_twin_studio_phase4.py::test_compose_credentials_are_scoped`
- `tests/test_twin_studio_phase4.py::test_images_pinned_and_secrets_ignored`

**Verification:** These 3 tests should change from `skipped` to either `passed` or `failed` (if there are additional issues uncovered in the compose files).

**Dependencies:** None. Can be done independently.

**Note:** If installing PyYAML reveals test failures in the compose tests, those must be investigated as separate findings.

---

### Fix 3: J-5 — Pipeline Step-Cache Correctness (P1)

**Finding ID:** Phase 1 F-005. Phase 2 J-5.

**Files to inspect:** `crowd_reliability/ml/pipeline.py`, `crowd_reliability/contracts.py`

**Contract questions to answer:**
1. Is repeated evaluation at the same timestamp intended to be idempotent? → **Likely NO** for the live demo path, where new observations drive new forecasts. The cache exists to avoid recomputation for the SAME inputs (e.g., a retry that resends the same data).
2. Can observations legitimately arrive at the same timestamp? → YES — the observation window is 5s, and observations can arrive at any point during the window. A tick at T=0 collects all observations received before T=5.
3. Should a second call with changed observations return cached, recompute, or reject? → **Recompute** — if observations changed, the forecast must reflect the new inputs. The cache should only serve results computed with the SAME observations.
4. Can shared scenario_id cause cross-session interference? → YES — `_step_cache` is keyed only on `scenario_id`, not session_id. If two sessions use the same scenario_id, they share the cache.
5. Are observations immutable after evaluation? → Need to verify (observations can be edited/augmented, as shown by `test_editing_hidden_truth_after_render_changes_no_pipeline_output`).

**Cache key inspection (pipeline.py lines 258-260):**
```python
prior = self._step_cache.get(scenario_id) if scenario_id in self.history_buffer else None
if prior and as_of_time < prior[0]: raise ValueError('Replay clock cannot move backwards.')
if prior and as_of_time == prior[0]: return prior[1]
```

The cache stores `(as_of_time, (tree_records, forecast_records))` keyed on `scenario_id`. When `as_of_time` matches, it returns the cached result without checking observations.

**Regression test to write:**
```python
# tests/test_step_cache_invalidation.py
def test_step_cache_returns_stale_when_observations_change():
    """F-005 regression: same as_of_time, different observations should recompute."""
    pipeline = ArchitecturePipeline(...)  # load mc_v2 models
    obs_v1 = [create_observation(zone="ent_a", variable="density", value=0.5, received_at="2026-10-15T10:00:00Z")]
    obs_v2 = [create_observation(zone="ent_a", variable="density", value=0.9, received_at="2026-10-15T10:00:00Z")]
    t = utc("2026-10-15T10:00:00Z")
    _, forecast_v1 = pipeline.step("scenario_1", t, obs_v1)
    _, forecast_v2 = pipeline.step("scenario_1", t, obs_v2)
    assert forecast_v1 != forecast_v2  # If changed, cache must be invalidated
```

**Proposed fix:** Include a fingerprint of the observation set in the cache key:
```python
# Compute fingerprint before cache check
obs_fingerprint = hash(tuple(sorted(
    (o.scenario_id, o.zone_id, o.variable, o.value, utc(o.received_at).isoformat())
    for o in recent_observations
)))

# Cache stores (as_of_time, obs_fingerprint, result)
# Cache hit only if as_of_time matches AND obs_fingerprint matches
```

**Behavioral consequence:** Cache hits become more selective. Inference cost may increase slightly for retries with identical observations (same as before), but recomputation for changed observations is required for correctness.

**Risk assessment:** The cache key must include EVERY semantically relevant input:
- `as_of_time` (currently included)
- Observation values (density, speed, inflow, outflow, queue_length, exit_open)
- Observation timestamps (received_at, observed_at, published_at)
- Observation sources (source_id)
- NOT include volatile fields like report_id (these can differ for the same semantic observation)

**Dependencies:** Requires F-002/J-2 fix to be tested (import collision must be resolved). Can be implemented independently.

---

### Fix 4: J-2 — Simulation Import Collision (P1)

**Finding ID:** Phase 1 F-002. Phase 2 J-2.

**Files to inspect:** `crowd_reliability/live_provider.py`, `crowd_reliability/flow_evaluation.py`, `crowd_reliability/random_flow.py`, `studio/simulation.py`, archived `simulation.py` files.

**Root cause:** Multiple directories define a top-level `simulation.py` module. `live_provider.py` uses `sys.path.insert(0, studio_dir)` before `from simulation import live_steps`, but this is order-dependent and fragile.

**Three approaches to evaluate:**

**Approach A — Test discovery policy:**
- Keep `--ignore` flags in pytest configuration for archive dirs
- Add a `pytest.ini` or `pyproject.toml` section with `testpaths` and `norecursedirs`
- Document: "Run maintained tests with: `pytest --ignore=... --ignore=...`"

**Approach B — Package-qualified imports:**
- Rename top-level modules to package imports: `from simulation import ...` → `from studio.simulation import ...`
- Requires: `studio/` to be a package (`studio/__init__.py`)
- Requires: changes to all imports in `live_provider.py`, `flow_evaluation.py`, `random_flow.py`, `runtime_server.py`, and test files

**Approach C — pytest import mode:**
- Use `--import-mode=importlib` which uses `importlib` instead of `sys.path` manipulation
- Should prevent `sys.modules` pollution
- Needs verification: does it work with the cross-file `from test_backend import C` import in `studio/test_observations.py`?

**Recommended approach:** **C first (importlib), then A as belt-and-suspenders.** The `_importmode=importlib` approach addresses the root cause (sys.modules pollution) without renaming modules. If it doesn't fully resolve the collision (e.g., due to `sys.path.insert` in `live_provider.py`), fall back to A (documented ignores) plus a root `conftest.py` that ensures `studio/` is on the path correctly.

**Active startup analysis:**
- `runtime_server:app` startup: `live_provider.py` adds `studio/` to path, then imports `simulation` → resolves to `studio/simulation.py` (has `live_steps`) → **works**
- This is NOT a defect in active startup — the `sys.path.insert` ordering ensures the correct module is imported first. The issue is ONLY in test collection where archive test modules may import `simulation` before `live_provider.py` runs.

**Regression test:**
```python
# tests/test_simulation_import.py
def test_correct_simulation_module_loaded():
    """F-002 regression: simulation module must have live_steps (studio version, not archive)."""
    import simulation
    assert hasattr(simulation, 'live_steps'), f"Wrong simulation module: {simulation.__file__}"
    assert 'studio' in simulation.__file__ or 'simulation.py' in simulation.__file__
```

**Dependencies:** Fix 3 (J-5 cache) depends on this for test execution.

---

### Fix 5: J-4 — MODEL_DIR Configuration Consistency (P2)

**Finding ID:** Phase 1 F-003. Phase 2 J-4.

**Files to inspect:** `.env`, `.env.example`, `crowd_reliability/config.py`, `crowd_reliability/api.py`, `hosted_server.py`, `runtime_server.py`, `start_review.py`, `verify_random_flow.py`

**Configuration ownership analysis:**
- `.env` sets `MODEL_DIR=./models/production` (untrained_initial)
- `hosted_server.py` line 24: `os.environ.setdefault('MODEL_DIR', str(ROOT/'models/candidates/mc_v2'))` — only sets if NOT already set
- `start_review.py`: explicitly sets `MODEL_DIR=mc_v2` (overrides)
- `verify_random_flow.py`: explicitly sets `MODEL_DIR=mc_v2` (overrides)
- `runtime_server.py`: imports `runtime.router` which uses `live_provider` which uses `settings.model_dir`
- `crowd_reliability/config.py`: `model_dir = Path(os.environ.get('MODEL_DIR', '.'))`

**Problem:** If a developer runs `python runtime_server.py` without sourcing `.env` and without `MODEL_DIR` set, they get `.` (current directory, no models). If they source `.env`, they get `models/production` (untrained). Only `hosted_server.py` and scripts set the correct `mc_v2` path.

**Proposed fix:** Change `.env` to point to `models/candidates/mc_v2` OR make `hosted_server.py`'s `setdefault` unconditional (override `.env`).

**Determination needed:** Is `models/production` ever used intentionally? Check if any deployment script sets `MODEL_DIR=production`.

**Regression test:**
```python
# tests/test_model_dir_config.py
def test_hosted_server_uses_mc_v2():
    """J-4 regression: hosted_server must select mc_v2 models."""
    from hosted_server import create_app
    # or: check os.environ after import
def test_runtime_uses_mc_v2():
    """J-4 regression: runtime_server must load mc_v2 manifest."""
```

**Dependencies:** None. Can be done independently after F-003 analysis.

---

### Fix 6: J-6 — scikit-learn Artifact Compatibility (P1)

**Finding ID:** Phase 1 F-006. Phase 2 J-6.

**Files to inspect:** `models/production/manifest.json`, `models/candidates/mc_v2/manifest.json`, `crowd_reliability/ml/pipeline.py` (load method)

**Evidence:**
- Installed: sklearn 1.9.1
- Artifacts saved with: 1.7.2 (production), 1.8.0, 1.9.1 (mc_v2)
- `requirements.txt`: `scikit-learn>=1.3.0` (no upper bound)
- Warnings: `InconsistentVersionWarning` (non-fatal)

**Active inference path:** Uses `models/candidates/mc_v2/` → artifacts saved with 1.7.2 and 1.9.1. Loading works with warnings.

**Strategies to evaluate:**
1. **Pin to 1.9.1** and regenerate mc_v2 artifacts with 1.9.1 (eliminates one warning source)
2. **Pin to 1.7.2** (matches most artifact versions) — requires environment downgrade
3. **Pin to 1.8.0** (matches one artifact version) — requires downgrade
4. **Remove pin constraint** and accept warnings — status quo

**Recommended approach:** Pin `scikit-learn>=1.9.1,<1.10.0` in `requirements.txt` and regenerate mc_v2 artifacts with 1.9.1. This eliminates the `InconsistentVersionWarning` for the active path. Production artifacts (1.7.2) would still warn but are not used by the active path.

**Dependencies:** Requires running the mc_v2 training pipeline (heavy compute). May not be feasible without authorization and resources.

---

### Fix 7: J-7 — Remove Dead Code (P3, low priority)

**Finding ID:** Phase 1 F-012. Phase 2 J-7.

**File:** `person4/event_horizon_p4/baseline.py` line 45: `_EXIT_MASK_IDX = U_DIM - 1`

**Evidence:** `grep -rn "_EXIT_MASK_IDX" person4/` → exactly 1 occurrence (the definition). Never imported or referenced anywhere.

**Proposed diff:** Remove line 45.

**Regression tests:** All `person4/tests/` (57 tests) — should all still pass.

**Dependencies:** None. Can be done anytime.

---

### Fix 8: F-007 — Authentication Policy (P1, design decision)

**Finding ID:** Phase 1 F-007. Phase 2 J-7 (security follow-up).

**Files to inspect:** `crowd_reliability/config.py`, `crowd_reliability/api.py`

**Current behavior:**
- `.env`: `AUTH_MODE=local_demo`
- `config.py`: `auth_mode = os.environ.get('AUTH_MODE', 'local_demo')`
- `api.py` line 77: `if settings.auth_mode == 'local_demo': return await call_next(request)` (bypass all auth)
- `runtime_server.py`: uses `api.py`'s app → inherits local_demo default

**Proposed design:**
- **Default to `api_key` mode** (safe by default)
- `local_demo` only when explicitly set: `AUTH_MODE=local_demo`
- In `api_key` mode: verify `x-api-key` header against configured key
- Fail closed (403) when key is missing/mismatched
- Test both modes via TestClient

**Owner decision required:**
1. What is the intended deployment scenario? (local demo only, or network-facing?)
2. If network-facing: what is the API key strategy? (single shared key, per-user keys, external auth provider?)

---

## 4. Dependencies and Interactions

| Fix | Depends On | Blocking |
|---|---|---|
| J-1 (JS parity) | None | No |
| J-3 (PyYAML) | None | No |
| J-5 (cache) | J-2 (import collision) for regression tests | Yes — tests won't run with 6 collection errors |
| J-2 (import collision) | None | Partially — can run with `--import-mode=importlib` |
| J-4 (MODEL_DIR) | None | No |
| J-6 (sklearn) | J-4 (model loading) | Yes — need to confirm which model path is active |
| J-7 (dead code) | None | No |
| F-007 (auth) | None | No |

**Critical path:** J-2 (import collision) must be resolved before J-5 regression tests can run in the full suite. J-1 and J-3 should be done first to maximize test pass rate.

## 5. Post-Fix Baseline Target

| Metric | Current | Target (after J-1, J-3) |
|---|---|---|
| Collected | 419 | 422 (+3 from PyYAML) |
| Passed | 417 | 419 (+2 from JS parity fix) |
| Failed | 2 | 0 |
| Skipped | 3 | 0 |
| Errors | 0 (filtered) | 0 |

*(PyYAML install + JS CWD fix: 419 passed, 0 skipped, 0 errors on filtered run)*

## 6. Owner Decisions Required Before Implementation

1. **Authentication policy (F-007):** Should the default `AUTH_MODE` change from `local_demo` to `api_key`? What is the intended deployment scenario?
2. **MODEL_DIR correction (J-4):** Should `.env` be changed to point to `models/candidates/mc_v2`, or should `hosted_server.py` override `.env` unconditionally? Is `models/production` used by any deployment?
3. **Import collision approach (J-2):** Prefer pytest `--import-mode=importlib` (Approach C) or root `conftest.py` with documented ignores (Approach A)?
4. **sklearn strategy (J-6):** Should we pin to 1.9.1 and regenerate artifacts, or accept warnings? (Regeneration requires training pipeline execution.)
5. **Phase 2 report file:** Should `docs/phase2_verification_report.md` be committed to git, or should it be kept as a working document?

## 7. Files That Would Be Modified (Upon Authorization)

| File | Fix | Change Type |
|---|---|---|
| `studio/test_backend.py` | J-1 | Test-only (add `cwd=`) |
| `requirements.txt` | J-3, J-6 | Dependency declaration |
| `requirements.test.txt` | J-3 | Add PyYAML if test-only |
| `crowd_reliability/ml/pipeline.py` | J-5 | Cache key enhancement |
| `tests/test_step_cache_invalidation.py` | J-5 | New regression test |
| `conftest.py` (root) | J-2 | New file |
| `pytest.ini` or `pyproject.toml` | J-2 | Test config |
| `.env` | J-4 | Config correction |
| `person4/event_horizon_p4/baseline.py` | J-7 | Dead code removal |
| `crowd_reliability/config.py` | F-007 | Auth mode default change |

## 8. Verification Process (Per Fix)

For each authorized fix:
1. State finding ID and demonstrated failure
2. Apply minimal diff only
3. Inspect `git diff`
4. Run targeted regression test (must exercise real dependency, not mock)
5. Run relevant component suite
6. Run filtered maintained suite (419-test baseline)
7. Record actual pass/fail/skip/error counts
8. Update evidence ledger in report

## 9. Deferred Until After Core Fixes

- F-009 (orphaned spatial model dir) — only after J-4 model selection is finalized
- F-011 (sigmoid design) — design difference, no action needed
- Adversarial test matrix (T-01 through T-28) — will be executed against the post-fix baseline
