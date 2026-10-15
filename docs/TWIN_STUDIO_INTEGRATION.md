# Twin Studio integration

**Status:** Phases 1–3 are implemented and tested. Phases 4–8 have not started: there are no services, APIs, object storage or main-pipeline integration yet.

The Twin Studio simulator becomes the canonical simulated-world and reference-state source for the environment `eh_twin_studio`. It lives in the isolated subsystem `services/twin_studio/`. The main application may use only its public modules, and no main-pipeline code imports the subsystem yet.

## Source and engine provenance (`services/twin_studio/engine/SOURCE.json`)

| File | sha256 | Status |
|---|---|---|
| Archive `Event_Horizon_Twin_Studio.zip` | `e3d2e8df…3f97d` | Newest in the workspace at audit time |
| `engine/original/simulation.py` | `ce39174c…` | Byte-identical to the archive's `simulation.py`; never edited |
| `engine/original/engine.js` | `866882f1…` | Byte-identical to the archive's `dist/engine.js`; never edited |
| `engine/simulation.py` | `9798781f…` | **Modified in Phase 2** (gate closure, completion classification, logs) |
| `engine/engine.js` | `5ec10f13…` | **Modified in Phase 2**, mirroring `simulation.py` operation by operation |

The browser UI engine in the archive is not part of this subsystem and is unchanged.

## Approved decisions

| ID | Decision |
|---|---|
| Zones | 11 zones: 8 walkable rectangles (the courtyard counts as walkable) plus 3 gate zones. A gate overrides its host area. The pond is a non-walkable obstacle. Agents outside every zone are counted separately. |
| Connections | **16** physical bidirectional connections, exactly the geometric contacts: 13 shared boundaries plus 3 gate attachments. The Phase 0 audit report said 17; that was an arithmetic error in the audit, corrected here. |
| Capacity | `null`. Twin Studio defines none. Person 1 must set it before Phase 5, and adapters must fail safely until then. |
| Exits | A `gate_closure` incident, parity-tested. Only route completions inside a gate zone count as exits. |
| Storage | Docker Compose + MinIO, with separate read-only accounts for the observation and reference APIs (Phase 4) |
| Models | A separate environment directory `models/environments/eh_twin_studio/` with untrained models. The tree bank is labelled out-of-environment. `models/production` is never modified. |

## Phase 1: canonical data model (verified)

- `venue/twin_venue_v1.json`: frozen zones, obstacle, connections and coordinate system (0.1 m per canvas unit).
- `zones.py` (`ZoneAssigner`): half-open rectangles, priority gate > obstacle > host area, with explicit `outside_modelled_zones` and `inside_obstacle` counts.
- `timebase.py`: engine step k covers [k, k+1) s, and its output is the state at instant k+1 s. Instant 0 is the empty venue. Canonical 5 s grid; interval fields aggregate over [T−Δ, T).

| Zone | Type | Walkable m² | Notes |
|---|---|---|---|
| T_ENTRANCE | entrance | 189.15 | NW gate overlap removed |
| T_CONNECTOR | corridor | 60.0 | |
| T_WALK_N | corridor | 490.5 | |
| T_WALK_W | corridor | 321.0 | SW gate overlap removed |
| T_WALK_S | corridor | 341.25 | |
| T_WALK_E | corridor | 256.5 | |
| T_COURTYARD | assembly | 985.65 | pond removed; stalls not modelled |
| T_ANNEX | corridor | 491.0 | E gate overlap removed |
| T_GATE_NW | exit | 7.2 | |
| T_GATE_SW | exit | 8.0 | |
| T_GATE_E | exit | 7.0 | |

## Phase 2: gate closure, zone states, causal observations (verified)

### 2A. Gate closure (both engines)

**Scene field:** `gates: [{id, x, y, w, d}]`, gate rectangles in canvas units. The subsystem passes the three gate zones from `twin_venue_v1`.

**Incident:** `{id, name, type: "gate_closure", gateId, start, end}`, with integer seconds and `0 ≤ start < end ≤ duration`.
- The gate is closed during every engine step t with `start ≤ t < end`, which is the simulated interval [start, end).
- At instants, the gate is closed at T exactly when `start ≤ T < end`.
- Closures are active in the `custom` run only. `ideal` has no incidents, and `random` has only seeded random collapses, unchanged.

**Blocking:**
- The engine precomputes, for every route, the arc-length spans where the polyline lies inside each gate rectangle (closed-rectangle clipping, so it is conservative).
- While a gate is closed, an agent ahead of a span stops 0.5 m before it (`GATE_STOP_MARGIN_M`), and an agent already inside a span holds position.
- The cap is applied to the movement step before the agent moves, so a fast agent cannot jump over the 1.2 m NW gate between frames. In the open-gate fixture that jump does happen; under closure it doesn't (tested).
- There is no rerouting. Blocked agents wait, and nothing claims they choose another exit.

**Completions:**
- A completion whose end position lies inside a gate rectangle (half-open) counts as a gate exit.
- Every other completion is `route_completed_not_exit`.
- Rows gain `gate_exits` and `route_completed_not_exit`, and their sum equals the original `exited`. The result gains `gateExits`, `routeCompletedNotExit`, `spawns`, `completions` and per-step `gateOpen`.

**Validation rejects:**
- an unknown `gateId`;
- reversed or out-of-range intervals;
- a gate closure that also carries `routeId`, `people` or `at`;
- unknown incident types;
- gate rectangles outside the map or with duplicate IDs;
- a route that spawns (at its start or at a surge marker) inside a gate that has a closure.

**Parity and compatibility:**
- Python and JS agree on every output field, in all three modes, for the open, closure and stuck-queue scenes. Floats match within 1e-9; the only tolerance needed is for JSON's int/float representation.
- Without gates, both modified engines reproduce every original output field of the archive engines.
- The archive's own 17 JS engine checks pass against the modified `engine.js`.

### 2B. Zone-level reference states (`zone_states.py`)

The source is the engine's per-agent frames, spawn and completion logs, and incident list. Positions are assigned only through `ZoneAssigner`, and route-level rows are never used. A state at instant T with sampling interval Δ looks like this:

| Field | Kind | Rule | Unit |
|---|---|---|---|
| occupancy | instant | Agents assigned to the zone at T, each counted once | persons |
| density | instant | occupancy / effective walkable area (gate overlaps and the pond excluded) | persons/m² |
| queue_length | instant | Agents present at T with last-step speed < **0.1 m/s** (the engine's own queued rule) | persons |
| probability_exit_open | instant | Gates only: 1.0 open / 0.0 closed at T. Non-gates: `null`, `exit_applicable = false` | probability |
| speed | interval [T−Δ, T) | Mean over agents present at T of each agent's mean engine speed over its active steps in the window. Empty zone: `null`, `speed_valid = false`. A fully stopped zone has valid speed ≈ 0, with engine residues around 1e−16 left unrounded. | m/s |
| inflow / outflow | interval | Actual zone changes between consecutive 1 s frames, plus spawns into the spawn-position zone and completions out of the completion-position zone, divided by Δ. Raw counts are kept as `inflow_count` / `outflow_count`. | persons/s |
| gate_exits / route_completed_not_exit | interval | Completions in the zone, split by whether they ended inside a gate | persons |
| gate_open_fraction | interval | Gates only: share of the window's steps with the gate open | fraction |
| accounting | instant | Agents in zones, `inside_obstacle`, `outside_modelled_zones`, and cumulative spawned, gate exits and non-exit completions | persons |

**Verified invariants:**
- Spawned equals agents in zones + obstacle + outside + gate exits + non-exit completions, at every 1 s and 5 s state.
- Occupancy change equals inflow − outflow for every zone and pseudo-zone at every step.
- Occupancy, density and queue match the agent frames.
- 5 s instant fields equal the 1 s states at the same instants.
- 5 s interval counts equal the sum of the 1 s counts over [T−4, T].

**Unsupported or limited (`zone_states.UNSUPPORTED`):**
- **Intermediate zone visits:** a zone crossed entirely within one 1 s step isn't counted. That's possible only for zones shorter than one step's movement, such as an open gate crossed mid-route.
- **Trailing partial interval:** the 5 s grid ends at the last multiple of 5 s. For a 90 s scene, step [90, 91) appears only in the 1 s derivation.

### 2C. Causal observations (`observations.py`)

This ports the repaired v0.2 generator, with the same timing contract and error model.

**Timing:**
- `observed_at` is a state instant on the observation grid (default every 15 s, a multiple of the 5 s grid).
- The value comes from the state at that instant only.
- `published_at = observed_at + publish_latency` (≥ 1 s, default 3 s).
- `received_at = published_at + transport latency`, where transport = (base delay 7 s − publish latency) + jitter 0–14 s.
- So `observed_at < published_at ≤ received_at` always holds.

**Error model** (v0.2 decision D4 defaults, uncalibrated):

| Setting | Default |
|---|---|
| Camera A noise σ | 6% (multiplicative) |
| Camera B noise σ | 10% (multiplicative) |
| Additive count noise (when truth > 0) | 0.5 persons |
| Zero-count false positives | probability 0.02, size 1–2 persons |
| Gate-state misclassification | 0.02 |
| Dropout | 0.05 |
| Outages | configurable; none by default |

**Coverage:**
- Camera A sees all 11 zones.
- The gate sensor sees the 3 gates.
- Camera B covers **no zones by default**. Its v0.2 coverage was specific to that venue, and no Twin coverage has been approved.

**Causality and reproducibility:**
- Each report draws from its own RNG keyed by (scenario, mode, seed, source, zone, variable, instant), so no report depends on any other state.
- Mutating future states leaves every earlier report byte-identical (tested).
- Report IDs derive from the same key.

**Missingness:**
- Undefined truth produces no report: empty-zone speed, flows at instant 0, and exit state outside gates.
- Dropped or outage reports are absent, never zero.

**Isolation:** reports never contain incident names, IDs, types, `state_at`, `label_available_at`, occupancy or accounting fields (tested).

**Identity:** reports and states share `scenario_id`, `run_mode`, `seed`, `environment_id` and `map_version` through `RunContext`. The deterministic `run_id` belongs to Phase 3.

### Worked example: one agent crossing a zone boundary

Scene: one route from (450, 540) to (450, 640), 1 agent, seed 3, `ideal` run. The courtyard/south-walkway boundary is at y = 575.

| Engine step k | Interval | State instant | y | v (m/s) | Zone |
|---|---|---|---|---|---|
| spawn (step 0) | | | 540.000 | | T_COURTYARD |
| 0 | [0, 1) | 1 | 553.859 | 1.386 | T_COURTYARD |
| 1 | [1, 2) | 2 | 567.718 | 1.386 | T_COURTYARD |
| **2** | **[2, 3)** | **3** | **581.577** | 1.386 | **T_WALK_S** |
| 3–6 | | 4–7 | 595.4 → 637.0 | 1.386 | T_WALK_S |
| completion (step 7) | | | 640.000 | | T_WALK_S, `gateId: null` → `route_completed_not_exit` |

- **Step 2 transition:** T_COURTYARD outflow +1, T_WALK_S inflow +1.
- **1 s state at instant 3:**
  - courtyard: occupancy 0, out 1, speed `null` (empty);
  - south walkway: occupancy 1, in 1, queue 0, speed 1.386.
- **5 s state at instant 5 (interval [0, 5)):**
  - courtyard: inflow 1 (the spawn) and outflow 1 (the crossing), so 0.2 persons/s each; occupancy 0.
  - south walkway: inflow 0.2 persons/s (count 1), outflow 0, occupancy 1, density 1 / 341.25 = 0.00293 persons/m², queue 0 (v ≥ 0.1).
  - accounting: spawned 1 = in zones 1.
- **Completion at step 7:** inside T_WALK_S and not in a gate, so it is counted under `route_completed_not_exit` in the window [5, 10), never as a gate exit.

## Phase 3: deterministic runs, future labels, immutable local bundles (verified)

Modules: `run_identity.py`, `labels.py`, `bundle.py`. There are no services, database or object storage. Those belong to Phase 4.

### Defect fixed before publication

Phase 2 observation records carried `run_mode` and `seed`, and both leak hidden future information:
- `run_mode` reveals whether incidents exist at all (`ideal` means none).
- `seed` plus the scene configuration is enough to recompute the seeded random-collapse schedule.

The generator no longer emits either field. They still key the per-report random streams internally, so no value changed. The bundle writer additionally projects every report onto an explicit allowlist. The physical model and error defaults are unchanged.

### Run identity

- **Format:** `run_id = "ehts_" + SHA-256(canonical_json(run_inputs))[:32]` (128 bits). The full 64-hex digest is stored as `run_input_sha256` in the reference manifest.
- **Canonical JSON:** sorted keys, separators `(",", ":")`, ASCII escapes, and NaN/Infinity rejected.
- **`run_inputs` fields:**
  - `environment_id`, `map_version`, `venue_sha256` (venue file with LF line endings);
  - `code_fingerprint_sha256`, computed over `engine/simulation.py`, `zones.py`, `timebase.py`, `zone_states.py`, `observations.py`, `labels.py`, `bundle.py`, `run_identity.py` and `__init__.py`, with LF normalisation;
  - `schema_versions`;
  - `scenario_id`, `mode`, `seed`, and the full `scene`, with the frozen venue gates injected and every incident included;
  - `epoch_utc` (normalised to `Z`), `sample_seconds`;
  - `observation_config`, including outages;
  - `label_config`: horizons, delay and target variables.
- **Excluded:** wall-clock time, UUIDs, file times and output hashes are never inputs. The input identity (`run_input_sha256`) is kept distinct from the output hashes (`files.*.sha256` in each manifest).

### Bundles (one simulation execution feeds both)

```
<root>/observations/<run_id>/  observations.jsonl, venue.json, manifest.json          (serving-safe)
<root>/reference/<run_id>/     reference_states.jsonl, accounting.jsonl, future_labels.jsonl,
                               incidents.json, run_inputs.json, manifest.json          (reference-only)
<root>/published/<run_id>.json completion marker
<root>/.staging/               per-operation staging (never read as a run)
```

**Observation records** use exactly these fields: `report_id, run_id, scenario_id, zone_id, source_id, variable, value, unit, simulation_second, observed_at, published_at, received_at, source_type, environment_id, map_version, schema_version`.

**Observation manifest** (allowlist):
- `bundle_kind`, `bundle_schema_version`, `run_id`, `environment_id`, `map_version`, `venue_sha256`
- `scenario_id`, `epoch_utc`, `sample_seconds`, `observation_schema_version`, `observation_every_seconds`
- `source_coverage`, `variables` (units), and `files` (sha256, bytes, record counts)

It does **not** contain: mode, seed, incidents, outages, noise or dropout parameters, the input digest, code fingerprints, or any reference path.

Validation scans every observation file for forbidden tokens: `run_mode`, `seed`, `state_at`, `label_available_at`, `target_at`, the reference source types, `gate_closure`, `collapse`, `surge`, reference filenames, and every incident's exact `id` and `name`.

**Reference manifest:** full provenance. It records mode, seed, `run_input_sha256`, the code-fingerprint breakdown, original and modified engine hashes, the label rule and exclusions, the `UNSUPPORTED` notes, `observation_manifest_sha256` (the cross-link) and file hashes. `run_inputs.json` and `incidents.json` are reference-only. Raw agent frames are **not** stored; zone states plus accounting suffice for reference and audit.

### Reference records and future labels

- **Reference records:** one per (instant, zone, variable), with variables `occupancy, density, speed, queue_length, inflow, outflow, probability_exit_open, gate_exits, route_completed_not_exit`.
  - Each record carries `value`, `value_valid`, `applicable` (false for `probability_exit_open` outside gates), `measurement` (`instant` or `interval`, with `interval_start`/`interval_end` = [T−5, T)), `state_at`, and `label_available_at = state_at + 60 s`.
- **Future labels:** for every grid issue instant t, horizon h ∈ {30, 60, 120} s and the six forecast variables, the value is the reference state at `target_at = t + h`.
  - `label_available_at = target_at + 60 s`, the approved v0.2 rule (`label_delay_seconds`, default 60). It's configurable and part of the run identity.
  - Interval targets carry `target_interval_start`/`target_interval_end`.
  - Targets past the last grid instant are excluded and counted per horizon in the manifest. Nothing is extrapolated or substituted.
  - Horizons must be multiples of the 5 s grid, or generation raises `ValueError`.

### Staging and publication protocol

1. Build both bundles in memory from **one** simulation. Unless disabled, build a second time and require byte-identical output; otherwise raise `BundleIntegrityError` (nondeterminism).
2. Run `validate_outputs`, 38 checks covering identity and cross-links, schema and map versions, allowlists, forbidden tokens, timestamp ordering, grid alignment, interval windows, duplicate keys, zone IDs, exit applicability, missing-value flags, label horizons, label availability, labels equal to reference states, and file hashes. Any failure raises `BundleValidationError` and nothing is written.
3. If a completion marker exists, verify the published run against the fresh manifests. A match is `already_published` (no-op); a mismatch is `BundleIntegrityError`.
4. Write both bundles to a private staging directory, fsync, and re-verify.
5. Install each bundle with a **non-overwriting** `os.rename`. If a directory already exists, it must verify against the expected manifest, or the operation raises `BundleIntegrityError` and leaves it untouched.
6. Write `published/<run_id>.json` last, via temp file and atomic `os.replace`. A run is complete **only** when this marker exists and `verify_published` passes.
7. Remove only this operation's staging directory.

If the process is interrupted before step 6, the bundles stay invisible as a run. The next identical publish verifies them and writes the marker (status `recovered`).

The marker's `published_at_utc` is the only wall-clock value, and it's never part of identity or integrity comparison.

**Commands:**
- Publish: `python -m services.twin_studio.bundle --spec spec.json --root DIR`
- Verify (in Python): `services.twin_studio.bundle.verify_published(root, run_id)`

### Residual risks and decisions (Phase 3)

- **Brute-forcing incidents from identifiers.** `run_id` and `report_id` are hashes over inputs that include the incidents and the seed. An observer who already knew every other input exactly (routes, populations, observation config, epoch) could brute-force the incident schedule offline. The operational side doesn't have those inputs. Resolved in Phase 4 by keyed HMAC identifiers.
- **Scenario names.** `scenario_id` is user-chosen and appears in both bundles. Validation blocks the obvious leaking words (`surge`, `collapse`, `gate_closure`), but operators must not encode incident information in scenario names.
- **Name collisions.** An incident whose `id` or `name` equals a string that legitimately appears in an observation file (for example a zone name) is rejected by validation, deliberately conservatively.

## Phase 4: object storage, APIs and keyed identifiers (implemented; live verification BLOCKED)

Full details: [TWIN_STUDIO_PHASE4_DEPLOYMENT.md](TWIN_STUDIO_PHASE4_DEPLOYMENT.md).

- **Keyed identifiers** (`keyed_ids.py`, scheme `hmac_sha256_v1`). The public run ID, report IDs and public scenario alias are HMACs under versioned, domain-separated keys.
  - The Phase 3 unkeyed digest stays in the reference bundle only.
  - This closes the Phase 3 residual risk about brute-forcing incident schedules from IDs.
- **Single serializer.** `build_run(spec, identity)` and `validate_outputs(out, identity)` are the Phase 3 code with an optional keyring. The MinIO publisher uploads exactly the bytes the local Phase 3 publication verified.
- **Storage.** MinIO with buckets `eh-observations`, `eh-reference` and `eh-catalog`, each with object lock and versioning.
  - Writes are conditional creates, and readers fetch by version pin.
  - The catalog marker is written last, and readers fail closed.
  - Separate least-privilege identities for the publisher, the observation reader and the reference reader.
- **APIs.**
  - Observation API: replay watermark under operator control.
  - Reference API: evaluator token and a server-side session watermark.
  - Compose networks keep the main application away from the reference API and MinIO.
- **Not verified live.** The real-MinIO integration suite (`tests/twin_phase4_integration`) and the network probe have **not been executed**.
  - On 2026-10-15, C: had 0.26 GB free and the Docker daemon did not answer within 20 s (`docker version` timed out twice).
  - Recovery is an operator step (see [deployment doc §6](TWIN_STUDIO_PHASE4_DEPLOYMENT.md#6-operating-it)).
  - None of the storage guarantees (policies, object lock, conditional writes, network isolation) count as verified until that suite and the probe pass.

## Phase 5: Twin Studio → main pipeline (implemented, host-tested)

Module: `services/twin_studio/pipeline_adapter.py`. It changes one main-repo file, `crowd_reliability/ml/graph.py` (Person 3's own module).

- **Observation contract.**
  - Records must carry exactly the Phase 3 observation allowlist, for `eh_twin_studio` / `twin_venue_v1` / `eh-twin-schema-1.0` and the expected run.
  - Any extra field (for example a label or reference field), missing field, unknown variable, unit or zone, foreign environment, non-numeric value, or broken timestamp order raises `AdapterError`. Nothing is dropped or guessed.
- **Six separate channels.** Twin `density, speed, inflow, outflow, queue_length, probability_exit_open` map 1:1 onto pipeline `density, speed, inflow, outflow, queue_length, exit_open` (tree channels `estimated_*` / `probability_exit_open`).
  - Units are renamed only (`persons/m2`→`p/m2`, `persons/s`→`p/s`, `probability`→`binary`); values are unchanged.
  - Zone IDs and timezone-aware timestamps are preserved.
  - Exit forecasts are produced for gate zones only.
- **Capacity is unknown and stays unknown.**
  - Zones are passed with `capacity_definition=None, capacity_status="unknown"`.
  - `SpatialGraphBuilder` now accepts a missing capacity only when it is declared unknown: it uses 0.0 and lists the zone in `unknown_capacity_zones`. An undeclared `None` raises an error.
  - The main venue's capacity features are unchanged (test).
- **Environment separation.**
  - `models/environments/eh_twin_studio/environment_manifest.json` records the environment's status. It holds no weights.
  - Nothing is loaded from `models/production`.
  - The production tree bank (trained on the main simulator) is not used. `UnavailableTreeBank` returns all six channels with mask 0, instead of the unfitted bank's heuristic values with mask 1.
  - The hypernetwork, GNN and GRU are deterministic seeded initialisations, `untrained_initial`.
- **Determinism and the GRU `.eval()` issue (Person 4).**
  - The problem: `MultivariateTemporalForecaster` has `dropout=0.1` between GRU layers. It is built in train mode, `load()` doesn't call `eval()`, and `ArchitecturePipeline.step()` uses only `no_grad`. Main-pipeline forecasts are therefore non-deterministic. `test_upstream_gru_train_mode_is_nondeterministic_evidence` demonstrates this.
  - What the adapter does: it calls `temporal_model.eval()` (and `eval()` on the hypernetwork and GNN) and seeds GRU initialisation inside a forked torch RNG. Person 4's module is unchanged.
  - Recommended upstream fix (for Person 4 to decide): call `model.eval()` at the end of `MultivariateTemporalForecaster.load()` and in `ArchitecturePipeline.__init__`/`load_pipeline`.
- **Pipeline fallback values.** `_legacy_zone_features` (main contract) still fills absent channels with fixed defaults (0.8, 1.2, …) **together with mask 0**. Masks therefore keep missingness explicit; the defaults are not observations.

## Phase 6: both streams, persisted forecasts, delayed labels (implemented, host-tested)

Module: `services/twin_studio/evaluation.py`. It is the offline evaluator: it holds the reference token and runs on the reference network or through the opt-in loopback route.

- **Stream B:** `TwinArchitectureRunner` (architecture pipeline, untrained).
- **Stream A:** `observation_persistence`, an observation-only baseline: the latest received report, held constant over each horizon.
  - The repo's `CentralSourceAdapter.generate_simulated_forecasts` builds its forecasts *from reference states*, so it is **not** used for Twin Studio.
- **Each 5 s issue step** follows this order:
  1. Read reports up to the observation replay watermark.
  2. Compute both streams' forecasts.
  3. Append them to `forecasts.jsonl` and fsync.
  4. Register them with the reference API (one session per stream).
  5. Advance the replay and both sessions one step together.
- **Forecasts are saved before labels can exist.** Each forecast is persisted at its issue watermark, which is strictly before `target + 60 s`. The reference API independently rejects any issue time other than the session's current one.
- **Labels and metrics.**
  - Labels come only from `GET /sessions/{id}/labels` after the server watermark passes `label_available_at`.
  - Metrics are per stream × variable × horizon: MAE and RMSE, plus Brier score and accuracy for `probability_exit_open`.
  - Inapplicable labels are excluded. Invalid (undefined) labels are counted as `missing_label`, never imputed.
- **Outputs.** `report.json` contains `skill_claim: none - untrained models; a replay demonstrates the data path, not forecast accuracy`. Evaluation outputs are never overwritten.
- **Leakage test.** `test_mutating_future_truth_does_not_change_issued_forecasts` adds 100 to every future label after publication. The forecast log and observation access are byte-identical; only the revealed labels change.

## Phase 7: dashboard reveal flow (implemented, host-tested)

The page is `dashboard/pages/1_Twin_Studio_Evaluation.py`, a new page in the existing Streamlit app; `app.py` is unchanged. Its view model is `services/twin_studio/report_view.py`.

The flow, in order:
1. Select a run and evaluation.
2. Inspect observations with observed, published and received timestamps.
3. View forecasts per stream and horizon.
4. Press **Reveal eligible labels**. This shows only labels the reference service already released to the evaluator.
5. View metrics.

Throughout, there are warnings for the untrained and unavailable components, and a list of unavailable fields (tree channels, capacity, reference-only variables).

The page reads evaluator output files only (`EH_TWIN_EVALUATIONS_DIR`, default `data/twin_evaluations`). A test checks that neither the page nor its view model references reference-API URLs, tokens, HTTP clients, boto3 or the reference store. The page is executed headless with Streamlit `AppTest`.

## Original app and fidelity to the ZIP (2026-10-15)

**The original app is included unchanged.** `services/twin_studio/studio/event_horizon_twin_studio/` is a byte-identical extraction of `Event_Horizon_Twin_Studio.zip` (SHA-256 `e3d2e8df…3f97d`; per-file hashes in `studio/SOURCE.json`; guarded by `tests/test_twin_studio_studio_app.py`). It is the final UI: run `python lab_server.py --port 8000` from that folder.

**Fidelity audit.** The original engines were compared with the integrated engines:
- **Corpus:** the archive's sample scene plus 40 seeded random scenes (1–12 polyline routes; surges and collapses; scales 0.05, 0.1 and 0.2), each in `ideal`, `custom` and `random` mode.
- **What was compared:** every result field exactly, including per-agent positions and velocities at every step.

| Comparison | Result |
|---|---|
| Python original vs integrated, no gates / gates present but open | 123/123 and 123/123 identical (2,558,418 agent-steps) |
| JavaScript original vs integrated, same two cases | 123/123 and 123/123 identical |
| Original Python vs original JavaScript (a property of the archive) | Identical structure and events; numeric difference ≤ 1.1e-13 relative |
| `T_GATE_E` closed over [15, 40) vs open | Divergence starts at frame 15 on the gated route only; the unrelated route is unchanged; no gate exits during the closure |
| Zone-state and observation derivation | No mutation of the simulation result; changing all states after 40 s leaves the 119 reports observed at or before 40 s unchanged |
| Venue vs archive `dist/venue.js` | All 12 rectangles represented (11 zones + pond); areas recompute exactly; the 16 connections recompute from rectangle adjacency |

**Non-physics changes** in the integrated `engine.js`:
- unknown modes are rejected instead of being treated as `random`;
- `csv()` gains two columns.

**Movement.** The engines move agents along polylines. The 16 connections exist only for the spatial model; there is no graph path planning.

## Tests (Phase 8 acceptance run, 2026-10-15)

| Command | Result |
|---|---|
| `pytest tests/test_twin_studio_phase1.py tests/test_twin_studio_phase2.py tests/test_twin_studio_phase3.py` | 99 passed |
| `pytest tests/test_twin_studio_phase4.py` | 41 passed |
| `pytest tests/test_twin_studio_phase5_6.py tests/test_twin_studio_phase7.py` | 29 passed (26 + 3) |
| `pytest tests` (everything, main + Twin) | 265 passed, 1 failed: `test_api_tree_outputs_route` (known; see below). Main-only: 96 passed, 1 failed, unchanged from before Phase 4. |
| `cd person4 && pytest` | 47 passed |
| Archive tests `test_backend.py` / `test_engine.js` / `test_ui.js` | 8 / 17 / 7 passed |
| Engine provenance (`engine/SOURCE.json`) | modified and original hashes match |
| Production artifacts vs Phase 0 baseline | 12 / 12 byte-identical |
| **Real MinIO suite** `--profile test run --rm tests` | **NOT RUN** (Docker blocked) |
| **Network probe** `--profile test run --rm main-app-probe` | **NOT RUN** (Docker blocked) |

Host tests ran with `DATABASE_URL=sqlite:///<tmp>/x.db PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider --basetemp=<dir on D:>`, because C: is full.

**Known failure: `test_api_tree_outputs_route`** (owner: P2, sairishigangarapu).
- On an empty database, `GET /api/scenarios/{id}/tree-outputs` returns `{"scenario_id", "zones": []}` without `tree_version`, but the test reads `tree_version` unconditionally.
- The documented seeding command (`seed-demo`) calls `save_pipeline`, which rewrites `models/production`, so it cannot be run under the current constraints.
- Not changed here.

## Owner dependencies and open items

| Owner | Item |
|---|---|
| Operator | Free space on C: or move Docker Desktop's disk image to D:, then run the Phase 4 live suite and the probe (deployment doc §6). |
| Person 4 | GRU `.eval()` fix in `temporal.py` / `pipeline.py`. Worked around in the Twin adapter only. |
| P2 | `test_api_tree_outputs_route` empty-database behaviour vs test expectation. No Twin Studio tree bank exists (channels masked). |
| Person 1 | Twin venue capacities (null) and camera B coverage (empty). |
| Team | `CentralSourceAdapter` simulated mode reads reference truth; it must not be used as Stream A on Twin Studio. |

## Integration checklist

| # | Item | Phase | Status |
|---|---|---|---|
| 1 | Audit the package and repo; record decisions | 0 | Done |
| 2 | Isolated subsystem `services/twin_studio/` | 1 | Done |
| 3 | Frozen `twin_venue_v1.json` (11 zones, 16 connections) | 1 | Done, verified against raw geometry |
| 4 | Zone assigner (gate > obstacle > host; outside counted) | 1 | Done, exact lattice test |
| 5 | Engine provenance (originals preserved, modifications recorded) | 1–2 | Done |
| 6 | End-of-interval time mapping | 1 | Done, verified on engine frames |
| 7 | Capacity values | Person 1 | Open (null; handled as unknown, never invented) |
| 8 | `gate_closure` incident + Python/JS parity | 2 | Done |
| 9 | Zone-level state sequence | 2 | Done (limits listed above) |
| 10 | Causal observation generator from the same run | 2 | Done (camera B coverage open) |
| 11 | Camera B coverage for the Twin venue | Person 1 decision | Open (empty by default) |
| 12 | Deterministic run ID, staging, validation, future labels, manifests, immutable bundles | 3 | Done (local filesystem) |
| 12a | Keyed (HMAC) run/report identifiers | 4 | Done (host-tested) |
| 13 | MinIO, observation API, reference API with server-side watermark, Compose | 4 | Implemented; **live MinIO and network verification blocked (Docker)** |
| 14 | Main-pipeline adapter; environment model dir; GRU `.eval()` | 5 | Done (host-tested); upstream GRU fix pending with Person 4 |
| 15 | Stream A (observations only), persisted forecasts, delayed labels, evaluator | 6 | Done (host-tested in-process); live API run pending Docker |
| 16 | Dashboard reveal flow and docs | 7 | Done (host-tested with AppTest) |
| 17 | Acceptance review | 8 | Host part done; live Phase 4 suite outstanding |
