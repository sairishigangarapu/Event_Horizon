# Twin Studio v3 reconciliation and partial-information pipeline

**Status (15 October 2026): implemented and host-tested, not complete.** The neural stages are trained but do not meet their validation rule, and the real storage/network acceptance tests are still blocked by Docker. Code lives in `services/twin_studio/v3/`; Twin-specific models in `models/environments/eh_twin_studio/v3/`; generated data in `data/environments/eh_twin_studio/v3/` (large runs git-ignored). `models/production` is unchanged.

**Canonical archive:** `Event_Horizon_Twin_Studio.zip`, SHA-256 `3fb226ce73e8ead281a31bb8e17c55173a04bc3a1e6903a797614f24503d9f7a`, 334,449 bytes, 31 files, engine `kinematic-v3`. A byte-identical copy with per-file hashes is in `services/twin_studio/v3/archive/` and `SOURCE.json` (git EOL conversion disabled for it). The previous archive (`e3d2e8df…`) was replaced at the same workspace path and is not used as a baseline; the Phase 1–7 subsystem in `services/twin_studio/` remains bound to it and is unchanged.

---

## 1. Source differences (Phase A)

| Area | Previous archive (+ Phase 2 repo extension) | v3 archive | Consequence |
|---|---|---|---|
| Frame semantics | Row/frame recorded after the step's movement | Frame t = state at instant t before movement; `v` = displacement over [t−1, t]; final frame is a snapshot | Phase 1 `timebase.py` / Phase 2 `zone_states.py` do not apply; v3 has its own derivation |
| Collapse | Stops agents within ±1 m on its own route | Circle of `radiusM` (default 3 m) blocks every intersecting route span; blocked sources make arrivals wait (`waiting_entry`); crossing audit | Blockages are spatial; barrier spans are exported |
| Density in speed rule | count / 4 m² | count / clipped cell area | Edge cells differ |
| Spacing vs barrier | Barrier, then 0.65 m spacing | Spacing, then barrier | Different queue movement |
| Queue | `v < 0.1` | `v < 0.1` and admitted before t | New arrivals not queued |
| CSV | 14 columns | 17 columns (`scheduled`, `waiting_entry`, `blocked`); `exited` = previous step | The archive README's schema line omits `blocked` |
| Gates | Repo extension `gate_closure` | None | Not ported; v3 barriers model blockage |
| Observations | None in archive | Route-level `camera` / `sensor` / `warden`, `observed_at_s` / `available_at_s`, own RNG | No zone-level or exit-state reports |
| Hidden truth | None | `hidden_ground_truth/` (`scenario.json`, `frames.jsonl`, `incident_labels.jsonl`, `metrics.csv`, `audit.json`) + ZIP | Physically separated |
| API | Studio only | `GET /api/observations/{run}/{mode}?time_s=T`, `--inference-only` (port 8001) | Caller chooses `time_s` (no trusted clock) |
| Reports | None | `reporting.py`, `analysis/` examples, decision journal | Reports, manifests, scenes and render responses carry truth |
| Venue | `venue.js` `0168ae7e…` | Identical | `twin_venue_v1.json` (11 zones, 16 connections) still valid |

Archive example CSVs reproduce exactly from `examples/event_horizon_scene.json` (0 value differences, 602 rows each).

## 2. Artifact ingestion and use (Phases B, C)

| Class | Files | Permitted use | Truth-bearing |
|---|---|---|---|
| OBSERVATION_STREAM | `observable/<mode>.jsonl`, `observable_ext/zone_sensing/<mode>.jsonl` | replay source (operator side), training, evaluation | no |
| OBSERVATION_ARCHIVE | `observable.zip` (contains later reports) | training, evaluation | treated as yes |
| HIDDEN_TRUTH | `hidden_ground_truth/**`, `hidden_ground_truth.zip`, `hidden_ground_truth_ext/**`, `scenario_params.json` | training, evaluation (requires `OfflineAccess`) | yes |
| ANALYSIS_CSV | `01_…`, `02_…`, `03_…csv` (fresh or `analysis/`) | training, evaluation, operator reporting | yes |
| OPERATOR_REPORT | `master_report.html`, `report_data.json`, `run_manifest.json`, `decisions_*.jsonl` | operator reporting, provenance | yes |
| SCENE_CONFIG | `examples/event_horizon_scene.json` | simulation input, operator reporting | yes |
| UNCLASSIFIED | anything else | none (fail closed) | yes |

Every record keeps its relative path, class, permitted uses, run id, origin (`fresh_run` / `archive_example` / `mc_dataset`), SHA-256, size, schema version and timestamp semantics (`artifacts.py`). No file is ever an inference input; the feature builder accepts only an `AuthorizedSnapshot` and rejects artifact records, paths, raw reports and dicts (tested). CSVs are validated (header, types, mode, seed, 0..duration grid, row count, duplicates, `net_flow`, cumulative `completed`) and cross-checked across runs (identical seed, routes and grid; identical ideal/random base arrivals).

Strict ingestion (`ingest.py`) validates exact field sets per family, source identity and kind, run/route/zone identity and coverage, gate-only exit sensors, finite values in physical bounds (catches unit errors such as density 150), integer counts, `observed_at_s ≤ available_at_s`, latency = source delay, the source's reporting grid, report-id format, and duplicate ids (identical copies ignored, conflicting copies rejected). Units are converted once (counts per window → persons/s). Simulation seconds are kept; UTC mapping requires a timezone-aware epoch. Rendered runs are written by the archive's own export procedure (`runs.py`); extensions live in sibling folders so the archive's ZIPs match their folders.

## 3. Route/zone mapping and its limits

Zone labels (`zone_states.py`, offline) come from v3 per-agent frames and the existing `ZoneAssigner` (gate > obstacle > host area; outside and obstacle counted, never dropped), on a 5 s grid with v3 frame semantics: occupancy, density (effective walkable area), queue (`v < 0.1` among agents present at T−1), speed (mean per-agent movement over (T−5, T], none for an empty zone), inflow/outflow (zone transitions incl. admissions/completions over (T−5, T]), and exit-open for gates = 0 if an active v3 barrier circle intersects the gate rectangle, else 1 (non-gates: not applicable). Invariants checked for every run: agent accounting and per-zone flow conservation.

Verified limits: 99.66% of agent positions in the example custom run fall in a zone (0.34% outside the modelled rectangles); Route 1 is 2.4% and Route 2 0.8% outside by length; every zone-to-zone transition along both routes is a physical connection; both routes end outside the venue (route completion is not an exit); `T_CONNECTOR` and `T_GATE_NW` are never visited. A zone crossed within one 1 s step is not registered. The exit label treats any barrier overlap with a gate as closed (conservative). Route-level aggregates are never divided among zones.

Zone-level observations come from the versioned extension `eh-v3-zone-sensing-1` (`zone_camera`, `zone_counter`, `gate_sensor`), generated from the zone state at each report's own observation instant, with configurable interval, delay, missingness, noise and coverage; absent values are omitted, never zero. Canonical route-level reports are unchanged.

## 4. Scenario variation and tree training (Phases E, F)

Dataset `mc_v1_n16_seed2026`: 80 scenarios, 16 per regime (normal, surge, blockage, observation_outage, compound), generator `eh-v3-mc-1`. Each varies duration (240–360 s), route populations and speeds, surge timing/location/size, collapse timing/location/radius (half placed on a gate), and every reporting source's interval, delay, missingness, noise and coverage (degraded under outages; optional redundant camera). All parameters and seeds are recorded. Splits are whole scenarios, stratified by regime: 45 train, 15 validation, 20 test. Regimes are metadata only; no clustering selects a model.

Tree features (`eh-v3-topo-features-1`, 58 per zone): own latest value/age/present/fresh per variable; neighbour mean, coverage fraction, youngest age and relative gap over physical connections; static degree, connected width, distance, area, zone type, exit applicability. Missing values stay NaN with masks. No hidden values, future reports, run files or GNN outputs (tested: neighbour reports change a zone's neighbour features only; hidden-label edits and future reports change nothing).

Bank `dt_bank_v3_topo_1`: six channels, depth/leaf selected on validation; out-of-sample outputs for training scenarios by 4-fold grouped cross-fitting. Held-out test (20 scenarios):

| Channel | Tree | Latest-report baseline | Validated on validation split |
|---|---|---|---|
| density MAE | 0.0245 | 0.0241 | yes (0.0220 vs 0.0233) |
| speed MAE | 0.116 | 0.120 | yes |
| inflow MAE | 0.244 | 0.250 | yes |
| outflow MAE | 0.207 | 0.239 | yes |
| queue_length MAE | 3.68 | 3.70 | **no** (4.22 vs 3.66) |
| exit_open Brier | 0.013 | 0.028 | yes |

Neighbour features carry 2–15% of each channel's importance. Note: on test, density is marginally worse than its baseline.

## 5. Training status of every component (Phase G)

| Component | Status | Evidence |
|---|---|---|
| Tree bank `dt_bank_v3_topo_1` | trained, held-out evaluated; 5/6 channels validated | `trees/.../tree_report.json`, `tree_validation.json` |
| Hypernetwork `hypernet_v2` + GNN `gnn_edge_v2` + GRU `gru_forecaster_v3_twin` | **trained_not_validated** | `neural/.../neural_report.json` |
| Staleness policy `eh-v3-staleness-1` | calibrated on validation, reported on test | `staleness/...json` |
| Routing `eh-v3-routing-1` | rule-based decision support, not validated against outcomes | tests only |
| HNet+GNN feedback loop | not implemented | — |

Neural training: architecture and dimensions unchanged (no Monte Carlo hyperparameter selection, per ARCHITECTURE §10); trained end to end on training scenarios with out-of-sample tree outputs and labels at 30 s and 60 s; early stopping on validation (41 epochs). Rule: validated only if Stream B beats the best observation-only baseline (latest report, report EWMA, held tree estimate) at 30 s on ≥ 3 of 5 numeric variables on validation. It wins 2 (inflow, outflow). Held-out test MAE, Stream B vs best baseline: inflow 0.304 vs 0.346 (30 s) and 0.348 vs 0.437 (60 s); outflow 0.260 vs 0.284 and 0.292 vs 0.367; density 0.036 vs 0.036 and 0.044 vs 0.048; speed 0.188 vs 0.171 and 0.212 vs 0.222; queue 16.3 vs 5.6 and 18.9 vs 8.3. Queue length is the clear failure.

## 6. Staleness rule and held-out metrics (Phase H)

At replay time T per source's latest report: `age_s = T − observed_at_s`, `delivery_latency_s = available_at_s − observed_at_s`, `max_age_s = delay_s + k · interval_s`. Separate fields: freshness, conflict (other source, within the larger interval, beyond tolerance), coverage (fresh sources), current-state disagreement (newer report from another source), forecast-change warning (used only if the neural status is validated; currently `UNAVAILABLE_MODEL_UNVALIDATED`), presentation state, action (`NO_ACTION` / `VERIFY` / `REVERIFY`) and reason codes with evidence ids and times. Tree estimates are not used as staleness evidence (not part of the evaluated policy).

Target: a report consistent with truth at its observation time that is materially inconsistent at T (variable tolerances; categorical exact; zone counters compared over the same window). Reports already wrong at observation are measurement errors (88 of 75,806 on test). k chosen on validation by F1: **k = 1.0**.

| Test (20 scenarios) | Precision | Recall | False-positive rate | Episodes detected | Median / p90 delay |
|---|---|---|---|---|---|
| k = 1.0 (chosen) | 0.116 | 0.220 | 0.135 | 96% of 188 | 0 / 5 s |
| k = 2.0 (initial) | 0.162 | 0.113 | 0.047 | 83% of 188 | 0 / 10 s |

By regime (k = 1): observation_outage recall 0.60 (FPR 0.44); normal 0.04; blockage 0.07. Gate-sensor and route-sensor staleness are not caught (precision/recall ≈ 0): with 1–10 s reporting, an outdated report is replaced within its own age limit, which an age rule cannot flag by construction. Episode detection counts an alert from any source on the entity/variable, so it overstates per-report detection. Age-based staleness is weak in this synthetic setting.

## 7. Event notices and leakage guarantees (Phase D)

Trusted replay (`replay.py`): the operator owns the watermark (`OperatorKey`); the model's `ModelView.snapshot()` takes no time argument and returns validated measurements with `observed_at_s ≤ T` and `available_at_s ≤ T`, ids new since its last snapshot, causal notices and inference-safe topology/source metadata. HTTP boundary (`replay_service.py`): operator-token advance; model-token snapshot with no parameters (any query → 400; time headers ignored); no static, render, manifest, scene, download or truth routes. In-process objects are a contract, not a security boundary; OS/container isolation of the model process is a deployment responsibility (as in the archive's `DATA_OUTPUTS.md`). The integrated pipeline uses the in-process view; the HTTP service is implemented and tested but not deployed.

Notices (`eh-v3-notices-1`) are derived only from available measurements, on onsets: possible obstruction (slow warden report; two stalled camera reports), exit reported closed / reopened (gate sensor), high density (zone camera). They carry evidence ids and times, never event ids, names, end times or labels. Gate-sensor errors (2–5% per report) create spurious single onsets.

Leakage tests (all pass): adding a future incident leaves every earlier snapshot and notice byte-identical; future reports do not change earlier snapshots or features; editing hidden labels, the scene's events and the manifest after rendering leaves forecasts, assessments, notices and recommendations byte-identical; forged keys cannot advance; future timestamps and headers cannot move the cutoff.

## 8. A* objective and tests (Phase I)

Graph: `twin_venue_v1`, re-validated against rectangle geometry (16 declared = 16 derived connections). Cost per edge (seconds): distance / speed estimate (fresh observed speed in [0.2, 3.45] m/s, else free speed 1.3 m/s slowed by the v3 engine's density rule, else 1.3 m/s) + queue × 0.65 m / speed + 20 s if the entered zone lacks fresh evidence; +60 s for a gate without a fresh status. A fresh "closed" report removes the gate; only a fresh "open" report verifies it. Capacity is not used (unknown). Heuristic: straight-line centroid distance to the gate / 3.45 m/s (admissible; equal to Dijkstra on 75 random cases). Output: status (`ROUTE_FOUND`, `NO_CONFIDENT_ROUTE`, `ROUTING_DISABLED`), exit, path, cost and breakdown, evidence times, closures, alternatives. Scenes not on the canonical venue disable routing. Tested: closures, missing and stale gate status, congestion, missing evidence, disconnected graph, misaligned scenes. Recommendations never route to a reported-closed gate in the three pipeline runs. A* does not reroute Twin Studio agents.

## 9. Integration and pipeline runs (Phase J)

`pipeline.py`: trusted replay → strict ingestion → topology features → trees → HNet/GNN → GRU → forecasts persisted (fsync) at issue time → notices → staleness → triggered A* → reference evaluation after the replay (labels eligible at target + 60 s; the evaluator advances its own label clock). Streams: `stream_a_observation_ewma` (observation-only EWMA, 90 s; the Phase 6 Twin evaluator uses latest-value persistence instead) and `stream_b_architecture` (with its status). Main API: read-only `GET /api/twin-v3/evaluations`, `/{id}`, `/{id}/{forecasts|assessments|notices|recommendations}?as_of_s=` over persisted outputs only.

Runs (outputs in `data/environments/eh_twin_studio/v3/evaluations/`): example scene (fresh render with master report), `mc_blockage_015` and `mc_compound_014` (held-out test). All eligible labels scored. Stream B beats Stream A on inflow in all three runs at both horizons, and on density in the two test runs; it is far worse on queue length (e.g. 25.4 vs 8.2 at 30 s).

## 10. Tests run

| Command | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider --ignore=tests/twin_phase4_integration` | 371 passed (78 new v3 tests; Phase 1–7 and main suites unchanged) |
| `cd person4 && python -m pytest -q` | 47 passed |
| Archive, clean extraction: `PYTHONUTF8=1 python -m unittest test_observations test_backend` | 18 passed (default Windows encoding: 2 errors, archive defect) |
| Archive: `node test_engine.js` / `test_regressions.js` / `test_ui.js` | 17 / 11 / 10 passed |
| `tests/twin_phase4_integration` (real MinIO) and the main-app network probe | **not run** (Docker daemon did not answer; not retried) |

Reproduction: `python -m services.twin_studio.v3.cli build-dataset | train-trees | train-neural | eval-staleness | write-manifest | render | run-pipeline` (see `cli.py`).

## 11. Unresolved

- Neural stages `trained_not_validated`; queue length forecasting fails; feedback loop not implemented.
- Staleness by age has low precision/recall here; a model-informed rule needs validated models and its own evaluation.
- Real storage/network acceptance blocked (Docker); the Phase 4 MinIO stack serves the previous archive's bundle format, not v3 outputs.
- The trusted HTTP replay service is not deployed; process isolation is not configured.
- The v3 tree bank was pickled with scikit-learn 1.7.2 locally; `requirements.deploy.txt` pins 1.9.1.
- Twin capacities remain unknown; camera coverage and noise are uncalibrated assumptions.
- Main dashboard has no v3 page yet (API routes only).

## 12. Production artifacts

`models/production` SHA-256 hashes were recorded before the work and verified unchanged after every test run.
