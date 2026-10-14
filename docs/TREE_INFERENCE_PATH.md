# Tree Models & Reliability — Inference Path Confirmation
## Member 2 — Decision-tree bank, integration contract, and evidence reliability

**Status:** Confirmed from code + artifact inspection + test run (see Section 8).
**Date:** 2026-10-14 · **Commit context:** `main` at `16146b1` (P2 freshness/conflict graph), current tree code at `4ac2be4`+ (bank version `dt_bank_v2`).

This file answers the outstanding question for Member 2: *do the shipped tree artifact, the preprocessing/scaler, and the output contract match what the integrated forecast path actually uses?* The answer is **partially — the contract is sound, but the shipped production artifact is stale and its scaler is missing** (Section 6). Everything below is sourced to concrete modules and artifacts and pinned to exact versions.

---

## 1. The canonical six-channel output

Defined once, in `crowd_reliability/ml/tree_bank.py`, and never reordered. This list **is** the export order and the channel dimension of `D[z,t]`:

| Order | Channel (tree output) | source variable | unit | kind | applicability |
|---|---|---|---|---|---|
| 0 | `estimated_density` | `density` | `p/m2` | regression | `all_zones` |
| 1 | `estimated_speed` | `speed` | `m/s` | regression | `all_zones` |
| 2 | `estimated_inflow` | `inflow` | `p/s` | regression | `all_zones` |
| 3 | `estimated_outflow` | `outflow` | `p/s` | regression | `all_zones` |
| 4 | `estimated_queue_length` | `queue_length` | `persons` | regression | `all_zones` |
| 5 | `probability_exit_open` | `exit_open` | `probability` | classification | `exit_zones_only` |

- The parallel raw-variable ordering is `crowd_reliability.ml.temporal.TARGET_VAR_NAMES`:
  `["density","speed","inflow","outflow","queue_length","exit_open"]`. It maps 1:1 to `TREE_CHANNELS` (see `tree_bank.py` / `spatial.py` `TREE_TO_OBS`).
- Channels are **never collapsed, summed, or averaged** before forecasting. The bank keeps a separate estimator per channel (5× `DecisionTreeRegressor` + 1× `DecisionTreeClassifier`) — `tree_bank.py:43-50`. This is the non-negotiable representation rule in `docs/ARCHITECTURE.md` §2.

## 2. Output contract: names, units, ordering, zone IDs, time, masks

The published inference function is `infer_tree_vector` in `crowd_reliability/ml/trees.py`:

```python
def infer_tree_vector(snapshot: TreeSnapshot, bank: DecisionTreeBank) -> TreeVectorResult
```

It returns a `TreeVectorResult` (`trees.py:38`) that carries every piece of the contract:

- `scenario_id: str`
- `evaluation_time: datetime` — the snapshot `as_of_time` (ISO-serialised on export). This is the per-zone evaluation timestamp.
- `zone_ids: List[str]` — **sorted** (`trees.py:111`: `zone_ids = sorted(snapshot.zone_ids)`). Stable across runs.
- `values: np.ndarray [Z, 6] float64` — the six channels in `TREE_CHANNELS` order, per zone.
- `masks: np.ndarray [Z, 6] int64` — per-zone, per-channel validity/applicability mask.
- `channel_metadata: List[Dict]` — name/unit/kind/source_variable/applicability (from `CHANNEL_METADATA`).
- `tree_version: str` and `feature_schema_version: str` and `scaler_version: Optional[str]` — pinned versions.

Masks are produced by `DecisionTreeBank.predict_zone` (`tree_bank.py:80`):

```python
mask = 1 if (channel != "probability_exit_open" or has_exit) else 0
```

So `probability_exit_open` is `1` (valid) only in exit zones and `0` (inapplicable) everywhere else; the other five channels are always `1`. This is asserted in `tests/test_tree_channels.py::test_exit_channel_applicability_mask`. Missing-data vs. inapplicability is the same mask field: a `0` means “do not trust/use this channel for this zone at this time.”

Per-zone records carry names, units, values and masks explicitly (`TreeVectorResult.to_zone_records`, `trees.py:51`):

```python
{"scenario_id", "zone_id", "evaluation_time", "tree_version",
 "feature_schema_version", "scaler_version",
 "channel_names": [...6], "channel_units": [...6],
 "channel_values": [...6], "channel_masks": [...6]}
```

JSON round-trips losslessly for all six values + masks (`TreeVectorResult.to_json` / `from_json`; `tests/test_tree_channels.py::test_all_six_channels_survive_serialization_roundtrip`).

## 3. The callable inference interface (actual artifact + preprocessing/scaler)

There is one canonical entry point used by the evaluation CLI and the replay path:

```python
from crowd_reliability.ml.trees import TreeSnapshot, TreeVectorResult, infer_tree_vector
from crowd_reliability.ml.tree_bank import DecisionTreeBank
```

The flow inside `infer_tree_vector` (`trees.py:102-138`):

1. Zone ids sorted.
2. For each zone, build the **fixed-order, observed-only 18-dim feature row** via `build_feature_row` (`features.py:48`).
   - Order: `[raw_values(6), norm_ages(6), obs_masks(6)]` in `TARGET_VAR_NAMES` order — 1:1 with `TREE_CHANNELS` (`features.py:55-57`).
   - Age normalised to `[0,1]` via `clip(age/120, 0, 1)`; `MISSING_AGE_SENTINEL = 999` → normalised to `1.0`.
   - **No RNG** at inference time. Identical snapshot + identical bank ⇒ identical output (asserted `test_inference_is_deterministic`).
   - `feature_schema_version = "obs_only_v1"`; `OBSERVED_ONLY_FEATURES` is the locked 18-name list (`features.py:19`). Reordering raises a version bump, not a silent change.
3. `bank.predict_zone(feature_row, has_exit=...)` (`tree_bank.py:80`):
   - Applies the bank’s **embedded scaler** if present: `if self.scaler is not None and self.is_fitted: x = self.scaler.transform(x)`.
   - Runs each of the six trees, returns `[values, masks]`.

The CLI surface (`crowd_reliability/cli.py`):

| Command | What it does |
|---|---|
| `export-tree-vector` | Loads a bank, runs `infer_tree_vector` on a scenario snapshot, writes `TreeVectorResult.to_json`. |
| `train-trees` | Fits versioned `FeatureScaler` + bank via `train_tree_bank`, writes `tree_bank_<v>.joblib`, `scaler_<v>.joblib`, `train_metadata.json`. |
| `replay` | Runs `ArchitecturePipeline.step` per time step; trees + forecasts. |
| `mc-refit` / `benchmark-trees` | Seeded refits + per-channel OOS metrics. |

Loading order in `cmd_export_tree_vector` (`cli.py:471-481`):

```python
bank = DecisionTreeBank()
versioned_path = "./models/trees/tree_bank_dt_bank_v2.joblib"   # preferred
prod_path      = "./models/production/tree_bank.joblib"          # fallback
```

i.e. it prefers the standalone trained artifact and only falls back to the production bank. **If neither exists it warns and uses the unfitted heuristic fallback** — so a missing artifact does not silently produce “real” predictions.

The `src/` package is a thin re-export shim (taste: shim pattern over a second package):

```python
# src/ml/__init__.py
from crowd_reliability.ml.tree_bank import ...  # CHANNEL_METADATA, TREE_CHANNELS, DecisionTreeBank
from crowd_reliability.ml.trees import ...      # TreeSnapshot, TreeVectorResult, infer_tree_vector
from crowd_reliability.ml.tree_train import ...# LeakageError, train_tree_bank, ...
```

So `from src.ml import infer_tree_vector` and `from crowd_reliability.ml.trees import infer_tree_vector` resolve to the **same** code. Real implementation lives in `crowd_reliability/`, `src/` only re-exports.

## 4. Freshness, conflict and coverage — distinct fields, five labels

These are produced by `crowd_reliability/reliability/assess.py::assess_information(reports, as_of, expected_coverage, thresholds, scenario_type, zone_type_lookup)`. Freshness, conflict and coverage are **separate fields**; nothing is fused into a single synthetic trust score.

### Four independently-owned modules

| Module | Responsibility | Key export |
|---|---|---|
| `reliability/freshness.py` | Age + fixed (P0/P1) thresholds | `observation_age_seconds`, `FIXED_*` constants |
| `reliability/thresholds.py` | Scenario/zone-dependent P2 thresholds | `DEFAULT_FRESHNESS_THRESHOLDS`, `FreshnessThresholds.for_variable()` |
| `reliability/conflicts.py` | Dedup, comparable grouping, disagreement, provenance graph | `deduplicate_reports`, `detect_conflicts`, `ConflictGraph` |
| `reliability/coverage.py` | Observed vs expected evidence sources | `compute_coverage` |

### The five information-status labels (independent booleans)

```python
FRESHNESS_LABELS = [
    "recently_verified",
    "potentially_outdated",
    "conflicting_evidence",
    "rapidly_changing",
    "insufficient_current_evidence",
]
```

They are **independent** (multiple may be true at once) and **conservative by design**: a stale report is flagged `potentially_outdated` (needs verification) — it is **never** asserted false solely because of age (`assess.py:82-84`).

### Distinct output fields (never fused)

`assess_information` returns, per report and per zone/zvar:

- `age_seconds` — observation age (separate number, not a label).
- `conflict_ids` — list of disagreeing report IDs (empty when none).
- `conflict_provenance` — per-zone conflict edges with `mean`, `relative_deviation`, `observed_at`, `sources`.
- `coverage_ratio` — observed/expected evidence ratio (separate from freshness).
- `threshold_used` — the `(recent_sec, outdated_sec)` pair actually applied.
- `labels` — the five booleans above.

Plus a top-level `conflict_graph` (num_nodes, num_conflict_edges, num_duplicate_edges) and `coverage_overall`.

### P2 explainable freshness thresholds (defaults)

| variable | recently_verified (s) | potentially_outdated (s) |
|---|---|---|
| `exit_open` | 5 | 15 |
| `inflow` / `outflow` | 6 | 25 |
| `speed` | 10 | 35 |
| `queue_length` | 15 | 40 |
| `density` | 20 | 45 |

Multiplied by scenario (`SCENARIO_MULTIPLIERS`) and zone-type (`ZONE_TYPE_MULTIPLIERS`) factors, all defaulting to `1.0` so behaviour is explainable and reproducible (`thresholds.py:38-65`). Conflict parameters: `CONFLICT_WINDOW_S=30`, `CONFLICT_REL_DEVIATION=0.4`, `CONFLICT_MEAN_EPSILON=0.1`.

### Service wrapper

`crowd_reliability/services/comparator.py::ForecastComparator.assess_information` is the per-zone/variable convenience that delegates to the reliability module and returns an `AssessmentRecord` whose `per_variable_flags` carries **both** the legacy flags (`stale`, `conflict`, `missing`) and the five independent labels, with `observation_age_seconds`, `coverage`, and `conflict_ids` as separate fields (`comparator.py:100-184`).

## 5. The integrated forecast path consumes the tree vector

`ArchitecturePipeline.step` (`crowd_reliability/ml/pipeline.py:144`) is the end-to-end inference used by `replay`/`seed-demo`:

```
OBS ──> build features ──> tree_bank.predict_zone ──> D[z,t] (6 values + 6 masks)
        _legacy_zone_features (18-dim, see §5.1)       \
                                                        ├──> node_features U_part (24 dims)
OBS ──> extract_zone_features_detailed ─> stack ─> encode_spatial(tree_values, tree_masks) ─> H[z,t] (16)
                                                                  \
U[z,t] = concat([raw(6), age(6), obs_mask(6), D(6), H(16)]) = 40 dims
temporal_model(U[z, t-L+1:t]) ──> forecasts[num_zones, num_horizons, 6]
```

- `node_in_dim = 24`, `u_dim = 40` (asserted `test_pipeline_temporal_input_channel_preservation`).
- `D` is appended **after** the observation block and **before** `H` — `pipeline.py:195`:
  `node_feat = np.concatenate([raw_vals, norm_ages, obs_masks, np.array(d_vals)])`
  ⇒ in `U`, the six tree channels sit at indices **18:24**, in `TREE_CHANNELS` order.

## 6. Temporal input: six tree channels — confirmed, with a documented caveat

The user requirement: *the temporal input receives the six real tree channels without reordering, losing masks, or substituting synthetic values.*

### What is confirmed

- **Six real channels, no reordering:** `D` (the six `predict_zone` outputs) is placed contiguously at `U[18:24]` in `TREE_CHANNELS` order; it is never permuted. `tests/test_tree_channels.py` and `test_pipeline_temporal_input_channel_preservation` pin this.
- **No synthetic substitution of the tree channels themselves:** for a **fitted** bank, `d_vals` come from `tree.predict(x)` / `predict_proba`; the only branch that substitutes defaults is the *unfitted* bank fallback in `predict_zone` (`tree_bank.py:93-103`), and the production bank reports `is_fitted = True`. So the six D values are genuine predictions, not heuristics.
- **Masks are not lost** — they are routed, not dropped. `predict_zone` returns `d_masks` (6 ints); in `step()` these become `tree_masks_list` and are passed **directly** into the spatial encoder: `encode_spatial(stack_zone_features(...), tree_values, tree_masks, ...)` → `build_spatial_features` consumes `tree_mask_<channel>` for all six channels (`spatial.py:144-168`). The tree masks are therefore present in the spatial context that produces `H`; they are simply **not** duplicated into the GRU input `U`. The observation masks (`obs_masks`, the freshness mask at `U[12:18]`) **are** in `U`. So no mask is discarded — each is consumed by the consumer that needs it (tree masks → spatial GNN; observation masks → temporal GRU).

### The caveat that was *not* synthetic substitution

- The **observation features** feeding the trees (`raw_vals`, `U[0:6]`) carry agreed `DEFAULT_VALUES` for variables with no fresh report (`features.py:88-90` / `pipeline.py:132-138`). This is an intrinsic property of the *observed-only* schema shared by both `build_feature_row` (standalone path) and `_legacy_zone_features` (integrated path) — it is the same rule, not a divergence. The six **tree** channels themselves are never defaulted.
- `extract_zone_features_detailed` (`pipeline.py:77`) is documented as the *no-substitution* as-of snapshot used by the spatial encoder; `_legacy_zone_features` (used for the tree input) applies the schema defaults. Both feed their respective consumers consistently with `build_feature_row`.

## 7. Preprocessing / scaler contract — the discrepancy (answering the outstanding question)

> “Whether their current artifact, preprocessing and output contract are what the integrated forecast path will use.”

The **contract is sound**, but the **shipped production artifact is stale and scaler-less**. Concretely:

**A. The current code produces `dt_bank_v2` + `scaler_obs_only_v1`.** Confirmed in `tree_bank.py:41` (`self.version = "dt_bank_v2"`), `features.py:12` (`FeatureScaler.VERSION = "scaler_obs_only_v1"`), and the test `test_train_tree_bank_deterministic_same_seed` which asserts `meta["tree_version"] == "dt_bank_v2"` and `scaler.version == "scaler_obs_only_v1"`.

**B. The standalone trained artifacts match the code.** `models/trees/tree_bank_dt_bank_v2.joblib` loads as version `dt_bank_v2`, `is_fitted=True`, with an **embedded `FeatureScaler`** (`scaler_obs_only_v1`, 18 fitted features). `predict_zone` applies it (`scaler is not None and is_fitted`). Its sidecar `models/trees/train_metadata.json` records `tree_version=dt_bank_v2`, `scaler_version=scaler_obs_only_v1`, seed 42, depth 6, 960 train samples.

**C. The production artifact does NOT match.** `models/production/tree_bank.joblib` loads as:

```
version = dt_bank_v1   (not v2)
is_fitted = True
scaler  = None         (NONE — bank has no embedded scaler; only keys trees/is_fitted/version are persisted)
```

i.e. it was written by an **older** `save()` (3 keys, no scaler/max_depth/random_state) and a **pre-v2** code version (version string `dt_bank_v1`). The current `DecisionTreeBank.save()` writes 6 keys including `scaler`, `max_depth`, `random_state` (`tree_bank.py:124-133`); this artifact predates that.

**D. The production manifest compounds the mismatch.** `models/production/manifest.json` claims:

```json
"tree_version": "dt_bank_v1",
"scaler_version": "standard_scaler_v1",
"training_status": { "hypernet": "untrained_initial", "gnn": "untrained_initial", "temporal": "untrained_initial", "tree_bank": "fitted_unvalidated" }
```

- `"standard_scaler_v1"` **does not exist** anywhere in the repo. The only scaler version string in the codebase is `FeatureScaler.VERSION = "scaler_obs_only_v1"`. There is no artifact named `standard_scaler_v1`, and the production bank has no scaler to apply.
- `tree_version "dt_bank_v1"` does not match the current code (`dt_bank_v2`) or the standalone trained artifact (`dt_bank_v2`).

**E. Behavioural consequence (verified by direct artifact load).** Feeding the *same* 18-dim feature row to both banks:

| Bank | Scaler | density (ch 0) | ... | probability_exit_open (exit zone) |
|---|---|---|---|---|
| `models/trees/tree_bank_dt_bank_v2.joblib` | `scaler_obs_only_v1` applied | 1.3839 | … | 0.0 |
| `models/production/tree_bank.joblib` | **none** (unscaled) | 0.8259 | … | 1.0 |

The production bank runs **unscaled** because `predict_zone` sees `self.scaler is None`. That happens to be *internally* consistent with how `cmd_seed_demo` trains it (`cli.py:161`: `pipeline.tree_bank.fit(X, Y)` — **no scaler passed**), so the shipped production pipeline is self-consistent: train and infer both unscaled. **But** it diverges from the standalone trained bank (v2 + scaler) that `infer_tree_vector` / `train-trees` / `export-tree-vector` actually use.

**F. Schema-level compatibility is fine.** `_legacy_zone_features` (used by `ArchitecturePipeline.step`) builds the identical 18-dim vector as `build_feature_row` (same 6+6+6 order, same `DEFAULT_VALUES`, same `clip(age/120,0,1)`, same 90s mask cutoff). So if the v2 bank (with its embedded `scaler_obs_only_v1`) were dropped into the production pipeline, `predict_zone` would scale it correctly and the integrated path would honour the scaler. The feature *schema* is the contract; the *artifact* is the problem.

### Recommended fix (scope caveat: not applied here)

1. Regenerate the production tree bank via the scaler-aware path (`train_tree_bank`, which embeds `scaler_obs_only_v1`) rather than `cmd_seed_demo`’s scaler-less `pipeline.tree_bank.fit(...)`, so `models/production/tree_bank.joblib` becomes a `dt_bank_v2` with an embedded scaler — matching `models/trees/tree_bank_dt_bank_v2.joblib` and the current code.
2. Fix `manifest.json`: set `scaler_version` to `"scaler_obs_only_v1"` (or to `null`/omit when the bank is intentionally scaler-free), and `tree_version` to `"dt_bank_v2"`.
3. Make `cmd_seed_demo` route through `train_tree_bank` + `save_training_artifacts` (or at least `bank.fit(X, Y, scaler=scaler)`) so the integrated path and the standalone path share one training contract. `cmd_export_tree_vector` already prefers the v2 artifact; align the rest to match.

> No code change was made in this pass — this file is the confirmation/delivery. Applying the fix above is the follow-up that makes the shipped artifact match the code contract.

## 8. Evidence: model status + reproducible evaluation results

### Model status (production manifest, `models/production/manifest.json`)

| Component | Version | training_status |
|---|---|---|
| tree bank | `dt_bank_v1` (stale — see §7) | `fitted_unvalidated` |
| hypernetwork | `hypernet_v2` | `untrained_initial` |
| GNN | `gnn_edge_v2` | `untrained_initial` |
| temporal (GRU forecaster) | `gru_forecaster_v1` | `untrained_initial` |
| scaler | `standard_scaler_v1` (phantom¹) | n/a (production bank has none) |
| pipeline | `v1.1.0` | `is_active = true` |

¹ No scaler artifact with this id exists; the real id is `scaler_obs_only_v1` (`features.py:102`).

Manifest integrity is enforced on load: `ArchitecturePipeline.load_pipeline` re-computes SHA-256 of `tree_bank.joblib`, `temporal_model.pt` and the spatial pair and aborts on mismatch (`pipeline.py:312-349`; `spatial_init` refuses to switch the spatial pair if `tree_bank.joblib`/`temporal_model.pt` don’t match the manifest).

### Reproducible evaluation results

All stored under `models/trees/`, regenerated by `train-trees` / `mc-refit` / `dataset-train` with seed 42.

**Standalone tree bank, held-out (`models/trees/holdout_benchmark.json`** — dt_bank_v2 vs `normal_evacuation`, n=160 per regression channel):

| channel | MAE | RMSE | extra |
|---|---|---|---|
| estimated_density | 0.0312 | 0.0492 | n/a |
| estimated_speed | 0.0080 | 0.0127 | |
| estimated_inflow | 0.1232 | 0.2291 | |
| estimated_outflow | 0.0729 | 0.1950 | |
| estimated_queue_length | 0.0339 | 0.4285 | |
| probability_exit_open | — | — | accuracy 1.0, brier 0.0, n=40 |

**Monte-Carlo OOS (`models/trees/mc/mc_benchmarks.json`** — 5 seeded refits, held-out scenario held out of fitting):

| channel | MAE | RMSE | extra |
|---|---|---|---|
| estimated_density | 0.0349 | 0.0535 | |
| estimated_speed | 0.0088 | 0.0137 | |
| estimated_inflow | 0.1781 | 0.3954 | |
| estimated_outflow | 0.1982 | 0.3983 | |
| estimated_queue_length | 0.0130 | 0.1561 | |
| probability_exit_open | — | — | accuracy 1.0, brier 0.0, n=36 |

Determinism is asserted, not assumed: `tests/test_tree_train.py::test_monte_carlo_refit_seeded_deterministic_and_versioned` checks that the **same** seed reproduces **byte-identical** `oos_predictions.jsonl`, and a different seed produces different output. Per-run `mc_metadata.json` records `seed`, `tree_version`, `scaler_version`, `feature_schema_version`.

**Dataset run (`models/trees/dataset/`** — Event_Horizon ingestion, dt_bank_v2, 12 train / 6 test scenarios, 912 samples): see `dataset_benchmark.json` and `train_metadata.json`. Same contract (`obs_only_v1` schema, scaler embedded).

### Test results (run against this tree, `16146b1`)

```
tests/test_tree_channels.py  tests/test_channels.py  tests/test_tree_train.py   -> 12 passed
tests/test_pipeline.py  tests/test_reliability_p2.py  tests/test_freshness.py  tests/test_alignment.py -> 19 passed
```

Covering: all-six-channels-preserved, exit-zone applicability mask, inference determinism, version=`dt_bank_v2`/scaler=`scaler_obs_only_v1` assertions, `U` dims (node=24, temporal=40), pipeline save/load + SHA integrity, and the five independent freshness labels.

---

## 9. Where the code lives (quick map)

| Concern | Module |
|---|---|
| Six-channel definition + bank | `crowd_reliability/ml/tree_bank.py` |
| Frozen inference entry point + result/record contract | `crowd_reliability/ml/trees.py` |
| Feature row + versioned scaler | `crowd_reliability/ml/features.py` |
| Training, MC refit, benchmark | `crowd_reliability/ml/tree_train.py` |
| Integrated end-to-end step (trees → spatial → temporal) | `crowd_reliability/ml/pipeline.py` |
| Temporal forecaster (GRU, 6-head output) | `crowd_reliability/ml/temporal.py` |
| Spatial encoder (36-dim context incl. tree masks) | `crowd_reliability/ml/spatial.py` |
| Evidence reliability, 5 labels, conflict graph | `crowd_reliability/reliability/{freshness,thresholds,conflicts,coverage,assess}.py` |
| Service wrapper (legacy + 5 labels) | `crowd_reliability/services/comparator.py` |
| Schema/records | `crowd_reliability/models/schema.py` |
| Pipeline manifest + atomic save + SHA verify | `crowd_reliability/models/manifest.py` |
| CLI | `crowd_reliability/cli.py` |
| `src/ml` / `src/reliability` shims | re-export the above (literal import paths also work) |

## 10. Summary

- The **six-channel output** (`TREE_CHANNELS`), units, fixed ordering, zone-sorting, evaluation timestamp, and applicability/missing-data masks are **locked and confirmed** by `tree_bank.py` / `trees.py` and the test suite.
- The **callable inference interface** is `infer_tree_vector(snapshot, bank)`, which uses the real tree artifact and its **embedded** scaler; the CLI prefers `models/trees/tree_bank_dt_bank_v2.joblib`.
- **Freshness, conflict and coverage** are emitted as **separate fields** with the **five independent status labels** (`recently_verified`, `potentially_outdated`, `conflicting_evidence`, `rapidly_changing`, `insufficient_current_evidence`) from `reliability/assess.py`.
- The **temporal input does receive the six real tree channels** contiguously at `U[18:24]` in `TREE_CHANNELS` order, with real (non-synthetic) predictions for a fitted bank; the tree **masks are consumed by the spatial GNN** (not dropped), and observation masks sit at `U[12:18]`.
- **Open issue:** the shipped `models/production/tree_bank.joblib` is a **stale `dt_bank_v1` with no scaler**, and `manifest.json` references a **phantom** `scaler_version: "standard_scaler_v1"`. The feature schema is compatible, but the production artifact must be regenerated through the scaler-aware `train_tree_bank` path (→ `dt_bank_v2` + `scaler_obs_only_v1`) to match the current code contract. See §7 for the concrete fix.
