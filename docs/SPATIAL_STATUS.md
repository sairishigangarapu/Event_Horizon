> Historical handoff notes. Current integrated behavior and validation status are recorded in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and [INTEGRATION_AUDIT_RECONCILIATION.md](INTEGRATION_AUDIT_RECONCILIATION.md). Statements below about active versions, untrained stages or incomplete wiring may describe the earlier snapshot.

# Spatial ML status: graph builder, hypernetwork, GNN (Person 3)

**Date:** 14 October 2026 · **Owner:** Person 3 (Spatial ML) · **Branch:** `main`, changes uncommitted at time of writing

**Status in one line:** the spatial path is implemented, tested and active in `models/production`, using **untrained initial weights**. A separate spatial-only smoke experiment shows that the stack can learn, and that message passing helps for density, speed and flows but **not for queue length**. That experiment is not a forecasting result.

---

## 1. What was implemented

| Area | Change | Files |
|---|---|---|
| Graph builder | Each connection becomes a forward edge plus a `<edge_id>__rev` reverse message-passing copy (7 connections give E = 14, asserted from data). Edge attributes: `width_norm, distance_norm, is_reverse_copy, is_traversable`. Reverse copies let downstream state, such as an exit closure, reach upstream zones; they do **not** imply people can walk that way. `edge_ids` stay aligned with Ve rows. An unknown direction or zone raises an error. | `crowd_reliability/ml/graph.py` |
| Feature schema | One central 36-feature per-zone schema: `obs_value`, `obs_age`, `report_present`, `obs_fresh`, `tree_value` (full D[z,t]), `tree_mask`. Every feature has a kind (continuous / binary / probability), a unit and its governing mask. | `crowd_reliability/ml/spatial.py` |
| Missing data | A missing report is no longer encoded as a 999 s age or a fabricated fallback value. Absence is carried by `report_present`, and real stale values are kept with their age. Invalid entries are zeroed before **and** after normalization. | `spatial.py`, `pipeline.py` (`extract_zone_features_detailed`) |
| Hypernetwork | Input normalization lives in buffers saved with the weights and applies **only** to continuous features. Init is seeded. Checkpoints carry config, schema hash, scaler id, zone/edge IDs, pair id and training status. Outputs: Vn[Z,16], Ve[E,4] ≥ 0 (softplus). | `graph.py` |
| GNN | Unchanged two-layer edge-aware architecture, with stale docstrings and defaults corrected, seeded init and the same checkpoint metadata. Output H[Z,16]. | `graph.py` |
| Interface | `SpatialEncoder.encode_spatial(zone_features, tree_values, tree_masks) -> SpatialOutput(H, Vn, Ve, zone_ids, edge_ids, version_metadata)`. It takes no reference states, targets or forecasts. | `spatial.py` |
| Integration | `ArchitecturePipeline.step()` calls `encode_spatial`. **The temporal input U is unchanged** (40 dims = raw 6, age 6, mask 6, D 6, H 16), and the tree-bank and U features are byte-identical to before (tested). | `pipeline.py` |
| Artifacts | The hypernetwork+GNN pair lives in its own versioned directory, `models/production/spatial/<hypernet>__<gnn>__<hash8>/`. `manifest.json` points at it through `spatial_artifact_dir`, and the manifest is replaced atomically, so readers see either the old pair or the new pair, never a mix. A manifest without the pointer but with root-level `hypernet.pt`/`gnn.pt` is rejected with a clear "legacy" error. | `spatial.py`, `pipeline.py`, `models/manifest.py` |
| Fit-status labelling | The manifest gains `training_status`. Production now reads `hypernet/gnn/temporal: untrained_initial`, `tree_bank: fitted_unvalidated`. | `models/manifest.py` |
| CLI | `python -m crowd_reliability.cli spatial-init [--dry-run] [--seed N]` creates an untrained pair, validates it, keeps a byte-exact `manifest.prev.json` (an existing backup is preserved under a timestamped name), switches the pointer, verifies the load, and restores the backup bytes if the load fails. | `crowd_reliability/cli.py` |
| Smoke experiment | `python -m crowd_reliability.ml.spatial_train`. See section 3. Its checkpoints are labelled `smoke_trained_auxiliary` and code blocks them from becoming an active pair. | `crowd_reliability/ml/spatial_train.py` |

### Active production artifact

| Item | Value |
|---|---|
| Pipeline version | v1.1.0 (was v1.0.0) |
| Active spatial pair | `spatial/hypernet_v2__gnn_edge_v2__6675c91e` (hypernet_v2, gnn_edge_v2) |
| Training status | untrained initial weights (seed 0) |
| Unchanged artifacts | `tree_bank.joblib` (b1045036…), `temporal_model.pt` (a5cd1176…), legacy root `hypernet.pt`/`gnn.pt` |
| Backup | `manifest.prev.json`, byte-identical to the previous manifest (sha256 115e4693…) |

**Rollback** needs both steps. Restoring the manifest alone is not enough:
1. Restore `manifest.prev.json` over `manifest.json`.
2. Check out the code from before this change. The current code deliberately rejects the legacy root-level checkpoints.

---

## 2. Evidence: tests and checks

Run with `DATABASE_URL=sqlite:///<temp>/t.db python -m pytest -q -p no:cacheprovider`.

**Result: 51 passed, 1 failed (52 collected; 36 are new spatial tests).** The failure, `test_api_assessments_invalid_timestamp`, existed before this work and belongs to the API (see section 5).

New tests (`tests/test_graph_shapes.py`, 25; `tests/test_spatial_leakage.py`, 11) cover:
- **Shapes and edges:** shapes derived from data; E = 2 × connections; Ve ≥ 0; edge ids and direction flags correct for uni- and bidirectional connections.
- **Masks, one test each:** an absent report's value and age have no effect, including at the raw hypernetwork input; `obs_fresh` matters; a masked tree channel's value has no effect; `tree_mask` matters; an observed zero differs from a missing value; binary and probability features are never normalized.
- **Sensitivity and gradients:** each of the six tree channels individually changes Vn, and every parameter receives a gradient.
- **Graph effects:**
  - Removing an edge changes only nodes 2-hop reachable from its destination.
  - Reverse copies let the concourse respond to exit-zone state; with forward edges only, it cannot.
  - The no-graph path equals the message-free full path.
- **Checkpoints:** exact round-trip; reordered zones, a mismatched pair or a mismatched schema/scaler are rejected; same seed gives the same init.
- **Leakage:**
  - Observations received after the cutoff change no spatial input, tree output or forecast.
  - Mutating the reference archive changes nothing.
  - `encode_spatial` accepts no target/reference arguments.
  - Experiment inputs are built without references, and stale reports are excluded.
- **Artifact lifecycle:**
  - Dry-run writes nothing; the switch keeps tree/GRU hashes.
  - A failed validation leaves no trace; a failed post-switch load restores the manifest bytes.
  - The backup is byte-exact and an older backup is preserved; a legacy manifest raises a clear error.
  - A smoke-trained pair cannot be made active.

**Post-switch checks on the real `models/production`:**
- Before/after hashes: only `manifest.json` changed, plus the added backup and spatial directory.
- The pipeline loads, and the hypernetwork and GNN have matching pair ids.
- One inference step gives Vn (8,16) and Ve (14,4) with min Ve 0.62; all values are finite and 126 forecasts are produced.
- `replay --scenario crowd_surge` runs end to end (907 steps).
- Replay MAE values are **not** quality evidence, because the hypernetwork, GNN and GRU all have initial weights.

---

## 3. Smoke-experiment findings

**Run:** `models/spatial/experiments/20261014T130452Z_661bcb/`. It contains `run_manifest.json` (configs, seeds, versions, caveats), `curves.csv`, `metrics_test.csv`, and six `smoke_trained_auxiliary` checkpoints.

**Question asked:** given only information available at time t, can HNet+GNN plus a throwaway linear head predict the simulator reference state at t+h? And does message passing help compared with the identical model with no edges?

**Design:**
- **Splits (disjoint seeds):**
  - Train: 24 scenarios (seeds 1000–1023).
  - Val: 8 scenarios (2000–2007), used for early stopping and selection.
  - Test: 8 scenarios (3000–3007), using **held-out parameter ranges**: stronger surges and later exit closures. Test was evaluated once.
- **Training:** horizons 5 s and 30 s; 3 seeds per model.
- **Normalization:** input and target statistics fitted on train only.
- **Baselines:**
  - *Tree persistence*: the current tree estimate as the prediction. It uses only inputs available at t.
  - *Oracle persistence*: the true state at t. Diagnostic only, not deployable.

**Test MAE, all zones (mean ± std over 3 seeds)**

| h | Variable | Tree persistence | No-graph | Graph |
|---|---|---|---|---|
| 5 s | density (p/m²) | 0.215 | 0.080 ± 0.003 | **0.063 ± 0.002** |
| 5 s | speed (m/s) | 0.055 | 0.021 ± 0.001 | **0.016 ± 0.001** |
| 5 s | inflow (p/s) | 0.616 | 0.349 ± 0.009 | **0.262 ± 0.007** |
| 5 s | outflow (p/s) | 0.605 | 0.226 ± 0.005 | **0.189 ± 0.004** |
| 5 s | queue_length (persons) | 2.600 | **0.897 ± 0.079** | 1.019 ± 0.087 |
| 5 s | exit_open | 0.167 | 0.088 ± 0.008 | 0.091 ± 0.008 |
| 30 s | density | 0.253 | 0.176 ± 0.006 | **0.153 ± 0.009** |
| 30 s | speed | 0.065 | 0.045 ± 0.001 | **0.039 ± 0.002** |
| 30 s | inflow | 0.763 | 0.663 ± 0.007 | **0.536 ± 0.028** |
| 30 s | outflow | 0.720 | 0.518 ± 0.004 | **0.431 ± 0.024** |
| 30 s | queue_length | 3.233 | 2.457 ± 0.092 | 2.721 ± 0.308 |
| 30 s | exit_open | 0.190 | 0.143 ± 0.007 | 0.146 ± 0.009 |

**Findings:**
- **The spatial stack learns:** both learned models beat tree persistence on every variable and horizon.
- **Message passing helps for density, speed, inflow and outflow**, on every seed at both horizons. At the concourse during exit closures, graph density error is about 33% lower than no-graph at 5 s and about 27% lower at 30 s.
- **Message passing does not help for queue length.** The graph model is worse on every seed at 5 s, and inconclusive at 30 s with high variance (±0.31). This is an open problem, not a rounding issue.
- **Exit-open probability is inconclusive** at both horizons.

**Limitations of this evidence:**
- An auxiliary head on H; the GRU forecaster is not involved, so this says nothing about end-to-end forecast quality.
- Three initialization seeds on **one fixed data split**, so ± reflects seed variation, not data variation.
- Tree persistence is a weak baseline, made weaker by in-sample tree fitting (section 5).
- One venue layout only, so there is no claim about other venues.
- Stale reports were excluded, because their generated values leak current state (section 5).
- Conflict reports were kept; their timestamps are within ±1 s of the state they came from.

---

## 4. Known limitations of the spatial path

- **Untrained:** the active production pair has initial weights. H currently carries structure but no learned meaning.
- **Scope of the "upstream" signal:** reverse edges are a modelling choice. They make information flow both ways, but carry no physical flow claim (`is_traversable` = 0 on one-way reverse copies).
- **Queue length:** see section 3. Not investigated yet.
- **Not committed:** all changes are uncommitted on `main`. The folder `models/spatial/` is untracked and not gitignored, and whether it belongs in version control is still undecided.
- **API unchanged:** the API still serves forecasts precomputed in the database, so this change reaches the API only once data is re-seeded or replayed.

---

## 5. Cross-team handoffs (identified by Person 3, owned by others)

| # | Owner | Issue | Evidence | Suggested fix | Impact on spatial work |
|---|---|---|---|---|---|
| H1 | **Person 4** | The GRU is never switched to `eval()` before inference, so its 0.1 dropout stays active. Identical inputs give different Stream B forecasts. | Two identical `step()` calls produced different forecasts (e.g. ent_a speed 0.064 vs 0.081); `tests/test_spatial_leakage.py` works around it explicitly. | Call `self.temporal_model.eval()` before inference in `ArchitecturePipeline.step()` (or when loading). Add a determinism test. | Spatial output is deterministic already; forecasts are not. |
| H2 | **Person 1** | Generator stale reports (`rep_stale_*`) take their value from the **current** reference state (×0.3–0.7) under a back-dated `observed_at`, so they leak present-time information. | `simulation/generator.py:79-97` | Derive the stale value from the reference state at the stale `observed_at`. | The smoke experiment excludes these reports; production inference cannot detect them. |
| H3 | **Person 2** | The tree bank is fitted in-sample on all three presets, with no out-of-sample predictions available for downstream training. | `cli.py` `cmd_seed_demo` fits on all preset scenarios | Scenario-split tree fitting plus chronological out-of-sample predictions for neural training. | Blocks valid joint spatial+temporal training. |
| H4 | Person 2 | `tree_bank.joblib` was pickled with sklearn 1.9.1; this environment has 1.7.2 (InconsistentVersionWarning). | Warning on every load | Pin and regenerate in the deployment environment. | None directly. |
| H5 | Person 5 | `test_api_assessments_invalid_timestamp` fails on an empty database: the endpoint returns before validating the timestamp. The README still says "not a runnable application". | Test output | Validate the timestamp before the empty-data early return. | None. |
| H6 | Person 1 | `report_id` uses `uuid4`, so IDs aren't reproducible from the seed. Docstring says 7 zones; there are 8. `ConnectionRecord` has no `distance_m`. | `generator.py`, `scenarios.py`, `models/schema.py` | Seeded IDs; fix the docstring; add the field. | None directly. |

---

## 6. Open scope decision

Two candidate next steps, both pending sign-off:
1. Investigate the queue-length regression.
2. Train a stronger spatial model, alone or jointly with Person 4's GRU.

Joint training is blocked on H3 (out-of-sample trees) and should follow H1 and H2.
---

This actually clarifies your remaining work quite a bit. The queue-length issue is now better understood, and Person 2's changes may remove one of the blockers to proper spatial-model training. Neither finding requires you to redesign the architecture immediately.

## 1. Queue-length regression: what the findings mean

The important distinction is between where the graph hurts and where both models fail.

| Finding                                                    | Interpretation                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Graph model leaks small queues into zones that never queue | Spatial message passing spreads queue-related signals into zones where they are inappropriate.    |
| At `exit_1`, graph and no-graph are tied                   | The graph currently offers no demonstrated advantage at the zone where the real queue forms.      |
| Both models underpredict queues during closures            | This is a shared forecasting failure, not a regression caused exclusively by the GNN.             |
| Bias grows from roughly −10 to −30 persons                 | The models increasingly underestimate queue buildup at longer horizons.                           |
| At 5 seconds, both barely beat persistence                 | The learned predictions offer little improvement over simply repeating the latest observed queue. |

The result is that queue prediction needs a targeted modelling improvement, but replacing the graph architecture is not yet justified.

Your three proposed directions are reasonable experiments:

- Weight the queue loss toward exit zones. This makes errors in the zones where queues actually form contribute more to training. The risk is overfitting to `exit_1` or neglecting queues elsewhere if future layouts introduce them.
- Add zone type to the prediction head. This gives the head explicit information about whether a zone is an exit, concourse or another zone type, rather than expecting it to infer that entirely from the learned representation.
- Predict queue growth directly. Instead of predicting only the future queue \\(Q\_{t+h}\\), predict its change relative to the current queue: \\(\Delta Q = Q\_{t+h}-Q_t\\). This may make the target easier to learn, although the model must still capture queue growth during closures.

I would test these as separate ablations, starting with the zone-type input and direct queue-growth prediction. Keep the original model as the control, use the same splits and seeds, and evaluate queue-active zones separately from zones whose queue is always zero.

## 2. Person 2's changes: what they unblock

The key distinction is between producing out-of-sample tree predictions and actually using them in your spatial pipeline.

Person 2's new output

`dt_bank_v2` + Monte Carlo refits + out-of-sample predictions

Handoff H3: verify the contract

Which predictions are out of sample? How are they aligned with scenarios, timestamps and zones?

Spatial pipeline

Currently uses `dt_bank_v1`, so the new output is not yet consumed.

Person 2's work potentially solves the in-sample prediction problem, but we should confirm exactly what the new predictions cover before marking H3 complete.

You should ask Person 2 to clarify:

1. Are the OOS predictions generated for every relevant scenario, zone, timestamp and tree feature?
2. Are predictions produced without training on the scenario or target being evaluated? What is the Monte Carlo refit and holdout procedure?
3. Are the outputs aligned with the six channels in your spatial feature schema, including masks and missing values?
4. Is there a supported way for the pipeline to load `dt_bank_v2` and consume `oos_predictions.jsonl`, or are those currently separate experimental artifacts?

Until that is established and integrated, keep the H3 note open. Having the new files in the repository does not mean your pipeline is using them.

## 3. What this means for your workload

I would update your status as follows:

| Item                                          | Status                                                       |
| --------------------------------------------- | ------------------------------------------------------------ |
| Spatial architecture and pipeline integration | Complete                                                     |
| Initial graph/no-graph experiment             | Complete                                                     |
| Queue-length failure analysis                 | Initial diagnosis complete; improvement experiments pending  |
| H3 tree handoff                               | Potentially unblocked; awaiting confirmation and integration |
| End-to-end training with Person 4             | Still pending                                                |
| Production-quality forecasting claims         | Not yet supported                                            |

My recommendation: don't start by changing your architecture. First, resolve H3 with Person 2. If the new OOS predictions meet the required contract, integrate them through the agreed interface and rerun the spatial experiment with leakage-safe features. Then test the queue-specific changes.

That order matters: otherwise, you could spend time tuning a spatial model using tree features that are later replaced, making the experiment difficult to interpret.

Your engineering deliverable remains largely complete. The remaining work is now a bounded sequence of data-handoff verification, controlled experiments and joint forecasting validation—not an open-ended instruction to keep adding complexity.

---

## 7. Spatial inference deliverable and Person 4 compatibility (Person 3)

### Deliverables

| Deliverable | Where | Evidence |
|---|---|---|
| Callable spatial inference | `ArchitecturePipeline.spatial_snapshot(as_of_time, recent_observations) -> (SpatialOutput, zone_inputs)`, the same path `step()` uses, without touching the GRU; lower level: `SpatialEncoder.encode_spatial(...)` | `tests/test_spatial_contract.py` |
| All six tree channels reach the hypernetwork's dynamic input | Feature schema slots `tree_estimated_density` … `tree_probability_exit_open` (+ masks) | Per-channel test on the **active production pair** |
| Vn[Z,16], Ve[E,4] ≥ 0 (softplus), H[Z,H_dim] | `SpatialOutput.H/Vn/Ve` with `zone_ids` / `edge_ids` row order | Shape and finiteness tests on production artifacts |
| Checkpoint/manifest match the loaded architecture | `models/production/spatial/hypernet_v2__gnn_edge_v2__6675c91e/`, `manifest.json` | Test compares the saved `config`/`version` with the loaded modules |
| Sample output for a known scenario and time | `data/demo/spatial_output_sample.json`, from `python -m crowd_reliability.cli export-spatial-sample` (bottleneck_exit_closure, seed 42, t = 150 s, 1,714 observations received by then) | Includes versions, `pair_id`, `training_status`, D, H, Vn, Ve and evidence values |
| Uses generated values and no reference truth | `spatial_snapshot` accepts only observations and ignores reports received after as_of | Tests: H = gnn(Vn, Ve) exactly; zeroing Ve or scaling Vn changes H; invariant to reference mutation and to late reports |

**Status:** weights are `untrained_initial`. The sample demonstrates the interface and data flow, not learned spatial meaning.

### Actual H dimension: 16, read from the checkpoints

`spatial_contract(encoder)` reads every dimension from the loaded modules' configs: `h_dim` = GNN `embed_dim` = **16**, `vn_dim` = 16, `ve_dim` = 4, Z = 8, E = 14. The main pipeline's temporal model matches it: `input_dim` 40 = 24 + 16.

### Mismatch with `origin/feat/person4` (handoff to Person 4)

The branch's `person4/event_horizon_p4/config.py` sets `H_DIM = 8  # MOCK`. Changing that constant is **not enough**, because the two input vectors also differ in layout and length:

| | Main pipeline `U` (current) | Person 4 branch `U` |
|---|---|---|
| Layout | raw(6) · age(6) · obs_mask(6) · D(6) · H(16) | X(6) · D(6) · H(8) · age(6) · obs_mask(6) · tree_mask(6) · exit_mask(1) |
| Dimension | 40 | 39 (becomes 47 with H = 16) |
| History | 12 steps (`lookback_steps`) | 60 steps |
| Tree masks / exit mask in U | no / no | yes / yes |

**What Person 4 needs to do (proposed, owned by Person 4):**
1. Size the spatial slot from `spatial_contract(...)["h_dim"]` (or the loaded GNN config) instead of a constant, and assert it at load time.
2. Agree on **one** U layout with the main pipeline. Then the GRU `input_dim`, the U scaler, the trained weights and the manifest all follow from that layout. Any GRU checkpoint must record the `h_dim`, the spatial `pair_id` and the hypernet/GNN versions it was trained with, and refuse to load against a different spatial pair.
3. Replace `MockTreeOutput` with Person 2's real tree output, and the H mock with `spatial_snapshot(...)`.
