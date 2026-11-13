# Event Horizon — Crowd Reliability & Forecasting

**Repository:** `https://github.com/sairishigangarapu/Event_Horizon.git`

Start locally with Python 3.11/3.12: `python start_review.py` (Windows: `scripts\launchers\START_REVIEW_WINDOWS.bat`). The launcher opens `/dashboard/`, where density, direction, duration and seed control the curved random-flow world and the existing forecasting backend. Studio remains at `/`.

For an already prepared environment: `python -m uvicorn hosted_server:app --host 127.0.0.1 --port 8000 --workers 1`. Set `MODEL_DIR` explicitly to choose a compatible artifact directory. Defaults for hosted/local review use the historical synthetic `mc_v2` checkpoint; custom-route prediction is experimental and the recorded stress runs include missed advance warnings.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `crowd_reliability/` | Core FastAPI backend, ML pipeline (`ml/`), data contracts, reliability engine, simulation, and services |
| `person4/` | Standalone `event_horizon_p4` temporal GRU package and tests |
| `services/twin_studio/` | Preserved Twin Studio v3 source-of-truth engine, replay, and A* routing |
| `studio/` | Interactive Studio frontend (`dist/`), Python observation/reporting server, and JS/Python tests |
| `runtime_frontend/` | FastAPI-served runtime dashboard assets |
| `dashboard/` | Legacy Streamlit evaluator dashboard |
| `scripts/` | Verification scripts (`scripts/verify/`) and Windows launchers (`scripts/launchers/`) |
| `models/` | Versioned model manifests and checkpoints (`production`, `candidates/mc_v2`, `environments/`) |
| `data/` | Demo fixtures and local SQLite/output directories |
| `deploy/` | Docker, Railway, Hugging Face Space, and Twin Studio deployment configs |
| `docs/` | Architecture specification, implementation status, training notes, and audit reconciliation |
| `tests/` | Automated Python test suites (440+ tests) and smoke checks |

Read [current architecture](docs/ARCHITECTURE.md), [implementation status](docs/IMPLEMENTATION_STATUS.md) and [integration/audit evidence](docs/INTEGRATION_AUDIT_RECONCILIATION.md) before the historical setup notes below. The main branch preserves Twin Studio v3 as a separate environment; its models and A* recommendations are not silently substituted into the route-based live model.

Optional protected mode: `AUTH_MODE=api_key` plus an independently generated `APP_API_KEY`. Use the dashboard/Studio **API access** button to supply it for the browser session. `CENTRAL_API_KEY` is only upstream-provider configuration. `APP_ENV=production` uses explicit `python -m crowd_reliability db-init`; local demo startup initializes the DB and additively upgrades forecast mask columns. Production API requests use a bounded process-local token bucket (default 600/minute); configure `API_RATE_LIMIT_PER_MINUTE` for your workload.

Full verification: install CPU torch, then `pip install -r requirements.integration.txt`; run `pytest tests -q`, `pytest person4/tests -q`, `python scripts/verify/verify_hosted.py`, `python scripts/verify/verify_roundtrip.py` and `python scripts/verify/verify_random_flow.py`. Live MinIO/network integration requires the separate documented Compose test environment.

Railway's live source remains **bugbounty**. Main receives tested integration code; its CI workflow does not deploy. Publishing the new frontend requires separate approval and promotion to the deployment branch.

---

# Railway deployment

The integrated app is prepared for Railway: Studio at `/`, runtime dashboard at `/dashboard/`, health check at `/api/ready`. See [Railway setup](deploy/railway/README.md). Local Windows startup still uses `scripts\launchers\START_WINDOWS.bat`.

# Windows rendering and audited pipeline update

Read [docs/FIXES_AND_TRAINING.md](docs/FIXES_AND_TRAINING.md) for the corrected launcher, implemented audit fixes, selected synthetic model and limitations. Older audit/status sections below describe earlier snapshots.

> **Integrated Studio + Runtime:** start with [docs/START_HERE.md](docs/START_HERE.md). Double-click `scripts\launchers\START_WINDOWS.bat`. The original backend documentation follows.

# Crowd Disaster Information Reliability
## Hackathon README — implementation-aligned template

> **Status:** Architecture and documentation draft. The application repository, implemented stack, team roster and deployment were not supplied. Fields marked **TO COMPLETE** and the proposed command interface below must be replaced or verified before submission. This package contains documentation and Mermaid source only; it is not a runnable application.

## 1. Project and team

| Field | Value |
| --- | --- |
| Project title | TO COMPLETE — approved project name |
| Team name | TO COMPLETE |
| Team members and roles | TO COMPLETE — names, roles and institution if required |
| Contact | TO COMPLETE — public project contact |
| Repository | TO COMPLETE — evaluator-accessible URL |
| Submission version | TO COMPLETE — tag and commit |
| Hosted demo | TO COMPLETE — URL, or declare local-only evaluation |
| Demo availability | TO COMPLETE — evaluation dates and hosting owner |
| Licence | TO COMPLETE — chosen project licence |

## 2. Problem and solution summary

During a crowd emergency, observations and forecasts may stop reflecting the current situation. This project uses crowd-disaster simulations to assess report freshness, conflicts and changing conditions, and compares two forecast streams:

1. A forecast supplied by a configured central source.
2. A forecast generated from separate decision-tree output channels, a hypernetwork that generates GNN node and edge values, spatial GNN context and multivariate temporal forecasting.

Density, speed, inflow, outflow, queue length and exit availability remain separate inputs and outputs. Decision-tree values are never collapsed into a single signal before forecasting. The dashboard preserves both forecast streams and their channel-specific differences.

Simulator reference states support evaluation and candidate-model training. They are separate from the observations used for production inference.

## 3. Architecture and documentation

- [Architecture specification](ARCHITECTURE.md) — component roles, model interfaces, data contracts and training lifecycle.
- [Architecture Mermaid source](diagrams/architecture.mmd).
- [Dataflow Mermaid source](diagrams/dataflow.mmd).

GitHub-compatible Markdown viewers render the embedded Mermaid diagrams. The standalone .mmd files contain the same editable Mermaid syntax.

## 4. Proposed stack

The following is a reference implementation choice, not an inventory of verified dependencies.

| Layer | Proposed technology |
| --- | --- |
| Runtime | Python 3.11 |
| Simulation | Seeded Python zone/agent simulation; final simulator TO COMPLETE |
| Data handling | NumPy, pandas |
| Tree bank | scikit-learn decision-tree estimators |
| Graph learning | PyTorch and PyTorch Geometric |
| Node/edge value generation | PyTorch MLP hypernetwork (`GraphHyperNetwork`) feeding the GNN |
| Temporal forecasting | PyTorch GRU with separate numeric and exit-state heads |
| API | FastAPI and Uvicorn |
| Evaluator dashboard | Streamlit and Plotly |
| Persistence | SQLite for local runs; filesystem for scenario files and model artifacts |
| Configuration | Environment variables and scenario configuration files |
| Tests | pytest |
| Diagrams | Mermaid |

Pin the actual compatible package versions in the submitted dependency lock file. Record the simulator, model artifact and schema versions with the release.

## 5. Supported environments and prerequisites

**Target environments, not yet runtime-tested:** Linux x86-64, Windows through WSL2, and macOS with a compatible Python/PyTorch installation. CPU execution is the intended default for bundled evaluation scenarios; a GPU is optional.

Required for the proposed implementation:
- Python 3.11 and pip.
- Git when cloning the repository.
- A modern browser for the local dashboard.
- Disk space for scenarios, SQLite data and model checkpoints.
- Network access for initial dependency installation; the bundled demo is intended to run offline afterwards.

**TO COMPLETE:** tested OS versions, minimum RAM/storage, observed CPU runtime, optional GPU/CUDA configuration and exact dependency versions. Do not describe a platform as tested until a clean setup succeeds there.

## 6. Installation

**The commands in sections 6–9 define a proposed repository interface. The named files and modules are not included in this documentation package. Replace them with actual commands or implement this interface, then verify on a clean machine.**

Proposed repository entry points:

| Path | Intended purpose |
| --- | --- |
| requirements.lock.txt | Pinned runtime and test dependencies |
| .env.example | Non-secret default configuration |
| crowd_reliability/cli.py | Database, demo-data, replay, training and evaluation commands |
| crowd_reliability/api.py | FastAPI application |
| dashboard/app.py | Streamlit evaluator interface |
| data/demo/ | Bundled synthetic scenarios and central-source replay fixtures |
| models/production/ | Compatible production pipeline and manifest |
| tests/ | Automated tests |

Proposed shell setup:

```bash
git clone <REPOSITORY_URL> crowd-reliability
cd crowd-reliability
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
cp .env.example .env
```

The commands above target Bash on Linux, WSL2 or macOS. If native Windows support is added, include the tested PowerShell equivalents.

## 7. Environment variables

Proposed names and defaults; align these with the final implementation. A configuration example is included in this package.

| Variable | Default | Purpose |
| --- | --- | --- |
| APP_ENV | demo | Execution mode |
| DATABASE_URL | sqlite:///./data/app.db | Local metadata and result database |
| SCENARIO_DIR | ./data/demo | Bundled scenario location |
| MODEL_DIR | ./models/production | Active pipeline artifacts |
| CENTRAL_SOURCE_MODE | replay | Use bundled central forecasts for independent evaluation |
| CENTRAL_FORECAST_PATH | ./data/demo/central_forecast.csv | Disclosed central-source replay file |
| CENTRAL_API_URL | empty | Optional real central-provider endpoint |
| CENTRAL_API_KEY | empty | Optional team-managed provider credential |
| SIM_SEED | 42 | Default scenario seed |
| STEP_SECONDS | 5 | Observation time step |
| LOOKBACK_STEPS | 60 | Temporal history length |
| FORECAST_HORIZONS_SECONDS | 30,60,120 | Forecast horizons |
| API_HOST | 127.0.0.1 | Local bind address |
| API_PORT | 8000 | Local API port |
| API_BASE_URL | http://127.0.0.1:8000 | Dashboard API address |
| DASHBOARD_PORT | 8501 | Local dashboard port |
| AUTH_MODE | local_demo | Proposed local evaluator mode |

Actual provider credentials are not part of this document. The bundled replay path must remain usable without them. Public hosting must use the final deployment's documented evaluator access configuration; local_demo describes the local-only flow.

## 8. Database setup and demo assets

The proposed local database is SQLite and does not require an external database account.

Proposed initialization:

```bash
python -m crowd_reliability.cli db-init
python -m crowd_reliability.cli seed-demo --seed 42
```

Expected initialization:
- Create the local data directory and database.
- Create tables for zones, connections, observations, tree outputs, forecasts, reference states, assessments, metrics and model manifests.
- Load the bundled demonstration scenario and source metadata.
- Verify that the bundled central forecast fixture and production checkpoint exist.

Reference-state records are accessed by evaluation/training paths. Production features are built from the observed stream.

**TO COMPLETE:** actual migration command, migration version, asset filenames, download sizes if any, and repeat-run behaviour. Bundle or provide free access to required checkpoints and fixtures; evaluation must not depend on retraining a large model.

## 9. Run commands and evaluator walkthrough

Start the API in terminal 1:

```bash
source .venv/bin/activate
python -m uvicorn crowd_reliability.api:app --host 127.0.0.1 --port 8000
```

Start the dashboard in terminal 2:

```bash
source .venv/bin/activate
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Proposed local addresses:
- Dashboard: http://127.0.0.1:8501
- API documentation: http://127.0.0.1:8000/docs

Proposed independent evaluation:
1. Open the local dashboard; the local demo flow is designed not to require an account.
2. Select the bundled crowd-surge scenario.
3. Start or replay the simulation.
4. Select a zone and inspect density, speed, inflow, outflow, queue length and exit availability separately.
5. Inspect both forecast streams and their issue/target times.
6. Review report freshness, conflict indicators and channel-specific differences.
7. Reveal reference outcomes and export evaluation results.

Proposed CLI alternatives:

```bash
python -m crowd_reliability.cli replay --scenario crowd_surge --seed 42
python -m crowd_reliability.cli evaluate --split test
```

**TO COMPLETE:** verify entry points, scenario identifier, successful startup output and expected demo duration.

## 10. Deployment, evaluator access and external services

**Current documentation status:** no hosted deployment or test account has been supplied or verified.

Choose and complete the final submission route:
- **Local evaluation:** provide the working commands, bundled scenarios and checkpoints, with no required login or paid services.
- **Hosted evaluation:** provide the exact URL and either a standard non-administrative test account or a documented safe self-service account-creation process.

If login is required, complete:

| Access item | Submission value |
| --- | --- |
| Login or registration URL | TO COMPLETE |
| Standard evaluator account or account-creation steps | TO COMPLETE |
| Evaluator permissions | TO COMPLETE — scenario replay, inspection and export |
| Account expiry/reset process | TO COMPLETE |
| Evaluation availability period | TO COMPLETE |

Evaluators must not need administrative, infrastructure, cloud-console or database credentials. The team must maintain hosted availability and any required external-service access/quota throughout evaluation. Evaluators must not be asked to purchase API credits.

The intended default demonstration uses bundled replay fixtures and local compute. If live central-source access is enabled, identify the provider and keep the disclosed replay mode available for independent testing.

## 11. Test instructions

Proposed automated commands:

```bash
python -m pytest tests/ -q
python -m crowd_reliability.cli evaluate --split test
```

Expected test coverage:
- Schema, timestamp, unit and zone-ID handling.
- Preservation of every decision-tree channel in temporal inputs.
- Graph and history-window construction.
- Hypernetwork output shapes, edge-value non-negativity and dependence on the observed state, including the tree channels.
- Forecast alignment by zone, variable and target timestamp.
- Separation of central and architecture forecast records.
- Reference-based scoring by variable and horizon.
- Seeded scenario repeatability.
- Compatible pipeline version loading and promotion.

Report metrics separately for each variable and horizon. Include simple baseline comparisons and reserve complete scenarios for final testing.

**TO COMPLETE:** actual test count, pass/fail summary, tested commit, evaluation command output and benchmark results. No software tests or model benchmarks were run as part of this documentation-only task.

## 12. Known limitations — scope and implementation status

- The defined prototype scope is crowd-based disaster simulation.
- Initial evaluation uses synthetic scenarios; real-world field validation is outside this documented prototype.
- Forecast coverage is limited to the implemented variables, venue layouts and horizons.
- Runtime commands, environment support, model artifacts and deployment remain to be verified against the implementation.

## 13. AI-tool disclosure

| Tool | Confirmed contribution |
| --- | --- |
| ChatGPT | Architecture discussion, design refinement and preparation of this README and architecture documentation |
| Other AI tools | TO COMPLETE — list tools and actual uses, or state none |

Add any AI-assisted code, image, dataset or presentation generation performed during implementation. Do not claim those activities occurred solely because documentation was AI-assisted.

## 14. Third-party resources and dependencies

The libraries below are proposed dependencies. Record actual versions and licences from the installed release and disclose any copied/adapted code, pretrained weights, venue maps or external datasets.

| Resource | Intended role | Official reference |
| --- | --- | --- |
| NumPy | Numerical arrays | https://numpy.org/ |
| pandas | Tabular time-series processing | https://pandas.pydata.org/ |
| scikit-learn | Decision-tree bank | https://scikit-learn.org/ |
| PyTorch | Neural training and temporal model | https://pytorch.org/ |
| PyTorch Geometric | Spatial graph model | https://pytorch-geometric.readthedocs.io/ |
| FastAPI / Uvicorn | API runtime | https://fastapi.tiangolo.com/ ; https://www.uvicorn.org/ |
| Streamlit / Plotly | Evaluator interface and charts | https://streamlit.io/ ; https://plotly.com/python/ |
| SQLite | Local persistence | https://www.sqlite.org/ |
| pytest | Automated testing | https://docs.pytest.org/ |
| Mermaid | Diagram source and rendering | https://mermaid.js.org/ |

**TO COMPLETE:** material external assets, source URLs, licence/attribution terms, modifications and pretrained-model details. No external dataset or pretrained checkpoint is included in this documentation package.

## 15. Synthetic-data disclosure

The proposed demonstration generates pedestrian/zone states and observations through a crowd simulator. It varies arrival patterns, route choices, restrictions, sampling intervals and observation availability.

The release must include:
- Simulator name/version and generation configuration.
- Random seeds, venue definitions, units and target-label definitions.
- Train/validation/test scenario assignments.
- Identification of synthetic reports and synthetic central-source forecasts.
- Any externally sourced components and their permissions/licences.

A replay fixture or separately generated baseline used as the central source must be labelled “simulated central forecast.” It must not be described as an actual authority feed.

## 16. Containers

Containers are not specified or included in this documentation-only package. The proposed evaluator path uses a Python virtual environment.

If the submitted implementation uses containers, add its actual Dockerfile(s), any Compose configuration, and tested build, run, port, volume and stop commands here. The final README must match the implementation choice.

## 17. Before submission

Replace project/team/access fields; align commands and configuration with actual code; pin dependencies; bundle the required assets; record clean-machine test results; and complete AI, synthetic-data and third-party disclosures. Remove template instructions only after the corresponding facts and evaluator steps have been verified.

## 18. Security & model-readiness audit (verified)

An adversarial code audit traced the **integrated forecast path** end-to-end: `cli.cmd_replay`/`cmd_seed_demo` → `ArchitecturePipeline.step` (`crowd_reliability/ml/pipeline.py`) → `DecisionTreeBank.predict_zone` (`ml/tree_bank.py`) → `SpatialEncoder.encode_spatial` (`ml/spatial.py`) → `MultivariateTemporalForecaster.forward` (`ml/temporal.py`). Findings below are confirmed against the artifacts in `models/production/` by live probe, not by docstrings/tests.

### Verified production readiness

| Component | Artifact | State | Risk |
| --- | --- | --- | --- |
| Decision tree bank | `models/production/tree_bank.joblib` | `dt_bank_v1`, fitted, **6 channels preserved**, `scaler=None` | Medium — scaler contract not enforced (see 2) |
| Temporal GRU | `models/production/temporal_model.pt` | `gru_forecaster_v1`, `untrained_initial`, `metrics_summary=null` | **Critical — served untrained** |
| Hypernetwork + GNN | `models/production/spatial/.../` | `untrained_initial` (pair `5a89a8…`) | **Critical — served untrained** |
| Manifest | `models/production/manifest.json` | `is_active=true`, no readiness gate | **Critical** |
| Forecast output | `schema.ForecastRecord` | 6 canonical channels; `exit_open` bounded in [0,1] (Sigmoid / `predict_proba`); numeric heads `ReLU`-clamped ≥ 0 | Channel layout/bounds OK; masks & units missing |
| Reliability | `reliability/assess.py` + `services/comparator.py` | 5 independent labels present | P2 thresholds/provenance not wired to API (see 5) |

### Open critical issues (do not serve as production-ready without these)

1. **[CRITICAL] Untrained neural stack served as active.** `seed-demo`/`save_pipeline` (cli.py:168) save freshly-initialized GRU/GNN/hypernetwork with no training step; `load_pipeline` and `step` contain **no readiness gate**. `manifest.training_status` = `untrained_initial` yet `is_active=true`. `/api/forecasts` and `replay` serve random-weight forecasts. **Fix:** add a readiness gate in `load_pipeline` that raises if any `training_status != trained_validated` or `metrics_summary is None`.

2. **[CRITICAL] Non-deterministic inference.** `step` (pipeline.py:245) calls `self.temporal_model(...)` while the model is in `train()` mode → GRU dropout (0.1) fires on every call. Verified: two identical `step()` calls differ by up to 0.0457 on 55/126 forecasts; `.eval()` yields 0.0 difference. Violates the byte-identical re-run contract. **Fix:** `self.temporal_model.eval()` before inference (ideally enforced in `load_pipeline`).

3. **[CRITICAL] Missing-artifact silent fallback.** `load_pipeline` (pipeline.py:327) `if os.path.exists(tb_path)` has no `else`/raise; a missing `tree_bank.joblib`/`temporal_model.pt`/spatial pair leaves an **unfitted** bank whose `predict_zone` (`tree_bank.py:94`) hands back heuristic defaults (`feature_row[0]`, `[1]`, `[2]`) with `channel_masks=[1,1,1,1,1,1]` — presenting synthetic values as valid. **Fix:** require all three artifacts; remove/ gate the `not self.is_fitted` branch.

4. **[CONTRACT] Scaler dropped at production inference.** `seed-demo` (cli.py:186) fits the bank **without** a scaler, so production `tree_bank.joblib` has `scaler=None` while `manifest.scaler_version="standard_scaler_v1"`. The real scaler at `models/trees/scaler_scaler_obs_only_v1.joblib` (for the v2 dataset bank) is orphaned. `load_pipeline` hash-checks the file but never verifies scaler presence/version. **Fix:** fit+attach+save the scaler in `seed-demo` (as `tree_train.train_tree_bank` does) and assert `bank.scaler` is present and `bank.version == manifest.tree_version` after load.

5. **[CONTRACT] P2 reliability features dead-wired.** `api.py:215` calls `ForecastComparator.assess_information` (a P1 reimplementation), **not** the canonical `reliability.assess.assess_information`. The comparator reimplementation (a) omits scenario/zone threshold multipliers (`comparator.py:161`), (b) aliases `coverage = 1.0 if not stale else 1-(age-45)/100` instead of `compute_coverage` (verified: fresh→1.0, stale@70s→0.75), and (c) never builds `build_conflict_graph`. `compute_coverage`/`build_conflict_graph` are exercised only by tests. **Fix:** route `/api/assessments` through `reliability.assess_information` with `zone_type_lookup`/`scenario_type`; delete the comparator reimplementation.

6. **[CONTRACT] Mask & unit integrity.** A zone with **100% missing sensors** still emits default-substituted values with `channel_masks=[1,1,1,1,1,1]` (verified for `exit_1`) — applicability and missing-data semantics are conflated (pipeline.py:100-118, tree_bank.py:116). `TreeOutputRecord` schema stores `channel_names/values/masks` with **no `channel_units`**; `ForecastRecord` has **no applicability/missing-data mask** column. **Fix:** add `channel_units`/`applicability_mask`/`missing_data_mask`; derive masks from observation presence, not only zone type.

7. **[EDGE] Central-source simulated forecast leaks future ground truth.** `central_source.py:91` blends `current_val` with `future_true_val` (the reference state at the forecast target time), unfairly advantaging Stream A in the A/B benchmark. **Fix:** smooth over received observations only.

