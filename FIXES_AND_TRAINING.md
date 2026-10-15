> Historical integrated-package change log. The preserved mc_v2 experiment below searched joint configurations. The current trainer selects tree parameters only, uses scenario-level out-of-sample tree features and fixes the neural configuration. Current status: [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

# Integrated update: Windows rendering and adversarial fixes

This update modifies the Integrated package, preserving the venue floor plan, Clear all routes control, separate Studio and Runtime Dashboard, user routes, three scenarios, heatmaps and master report.

## Start on Windows

1. Stop the previous server (Ctrl+C).
2. Extract this ZIP into a new folder.
3. Run START_WINDOWS.bat and keep its terminal open.
4. Studio opens at http://127.0.0.1:8000. The separate dashboard is http://127.0.0.1:8001/dashboard/.
5. Open your saved scene and render. Press Play to advance the runtime clock.

The screenshot's charmap/0x9d failure came from reading report JavaScript using the Windows default text encoding. Those reads now explicitly use UTF-8, and the launcher sets UTF-8 for both processes. A regression test emulates cp1252 defaults; an HTTP roundtrip includes Kannada route and incident names.

The launcher installs dependencies on first use and records successful installation, so later launches skip repeated installation. If dependencies change or a prior install is broken, remove .venv/eh_installed.ok to retry. To reuse your already installed environment, copy the old .venv folder into the extracted project before starting, or run the new start_integrated.py with that environment's Python. Python virtual environments must remain on the same computer and Python installation; recreate one if relocation fails.

## What is implemented

- Shared observation validation: finite nonnegative values, canonical units, timestamp ordering, UTC normalization, scenario isolation, stable tie-breaking and immutable report IDs.
- Single-class exit-tree correction. All-closed labels produce P(open)=0; all-open labels produce 1. Missing reference targets are masked rather than invented.
- Deterministic synthetic report IDs. Stale reports now replay historical states instead of backdating current truth.
- Replay history rejects backwards time and handles repeated ticks without appending history. The live bridge maintains one-second transport and five-second model cadence.
- Exact forecast quantities/targets, deterministic revision selection, scenario/unit separation, as-of filtering and label-release filtering in the main evaluator.
- Camera dataset ingestion requires independently sourced calibration. It no longer uses hidden speed-link labels, and camera speeds derive from observable tracks. Dataset graph cells have physical neighbor links. Missing exit status is not hardcoded open.
- Clustering is explicitly over observed zone/time snapshots, not individual pedestrian IDs. Training-only KMeans normalization and cluster assignments route to contextual tree experts. Unknown/distant clusters use a global contextual fallback.
- Context has 42 features: 18 observed values/ages/masks, 6 masked neighbor values, 6 neighbor availability indicators, 8 static node features and 4 local edge summaries. Hidden Monte Carlo state is never used as live context.
- Offline Monte Carlo model selection compares two configurations covering hidden width, learning rate, tree depth and cluster count. Joint gradients train the hypernetwork, message-passing GNN and GRU. The hypernetwork generates node/edge representations; it does not claim to generate a new GNN architecture per timestep.
- Dashboard alerts distinguish stale evidence from synthetic forecast-risk advisories. A moving average remains a baseline; fixed +30/+60/+120 second forecast horizons are not predicted incident-onset times.

## Included trained checkpoint

The original models/production artifacts are preserved. The launcher defaults to models/candidates/mc_v2 when present; setting MODEL_DIR overrides that selection.

The selected checkpoint was trained on 6 Monte Carlo scenario seeds, selected on 2 separate validation seeds, and evaluated on 2 untouched test seeds. Each seed samples surge magnitude/timing, closure timing, dropout and source conflicts. Both candidate configurations use identical split seeds. Check models/candidates/mc_v2/selection_report.json for the full objective, candidate losses, split identities and per-variable/horizon errors. No claim of globally optimal hyperparameters is made.

Selected tested configuration: hidden width 32, learning rate 0.003, tree depth 4, 2 clusters. The other tested candidate used width 64, learning rate 0.001, depth 8 and 3 clusters. The maximum epoch count was 20 with validation early stopping.

Held-out synthetic 30-second MAE:

| Variable | Selected model | Persistence baseline |
| --- | ---: | ---: |
| Density (people/m²) | 0.1483 | 0.1947 |
| Speed (m/s) | 0.0531 | 0.1208 |
| Queue (people) | 0.9470 | 1.9218 |

These are narrow synthetic benchmarks on the repository's default venue, not measurements of field safety performance. They use a persistence baseline; they are not a claim of beating the central EWMA. Training/testing a custom floor plan or real-camera domain remains separate work. Custom-route transfer is explicitly experimental; readiness stays production_ready=false. The initial checkpoints remain untrained. The trained checkpoint is tagged synthetic_validated, which does not pass the real-world strict serving gate.

Reproduce offline training into a NEW directory:

    python -m crowd_reliability.ml.train_system --out models/candidates/my_run --seed 42 --runs 6 --epochs 20

The trainer refuses to overwrite an existing manifest. Keep final test seeds out of model selection. Monte Carlo samples and measurement generation are seeded. Exact floating-point training reproducibility is scoped to the same CPU/Python/library environment; runtime inference uses evaluation mode.

The contextual tree bank version is dt_bank_context_v3. It preserves all six channels. The temporal input remains 40-dimensional, keeping the full tree vector alongside spatial embeddings. Direct contextual-tree snapshot inference requires a graph; the live runtime supplies one automatically.

## Output separation and runtime boundary

Studio still writes observable/ and hidden_ground_truth/ separately. Inference consumes only reports released by the replay clock. Retrospective truth arrives through a separate evaluation endpoint after the target time; it does not enter the forecaster. A complete observable ZIP includes later reports, so live inference must use the cutoff API/ordered bridge, not read the full archive.

The integrated operator dashboard intentionally shows retrospective truth next to predictions. This is a local operator/evaluation application, not a secure multi-user deployment. Run the standalone Studio observation-only service with an observation-only filesystem mount/account if isolating a separate ML consumer. A process with the whole project, evaluation endpoints or complete filesystem can read truth; logical separation is not OS access control.

Risk advisories use demonstration thresholds (density 3 people/m²; queue 20 people), are explicitly synthetic, and are separate from freshness flags. They are not calibrated crowd-safety thresholds. No injury or true physical crowd-collapse model was added.

## Checks

    python -m pytest tests -q --ignore=tests/test_dashboard_app.py --disable-warnings
    python -m pytest person4/tests -q --disable-warnings
    python verify_roundtrip.py

In studio/:

    python -m unittest test_observations test_backend
    node test_engine.js
    node test_regressions.js
    node test_ui.js
    node test_live.js

The old Streamlit dashboard test requires Streamlit, which the lightweight integrated runtime does not install. It remains available through the full requirements.txt development environment. The separate runtime dashboard is served by FastAPI and exercised by HTTP tests. Visual browser/Windows hardware verification remains a user-side check.

The new verify_roundtrip.py launches both actual servers on temporary ports, renders three scenarios with Unicode names, sends only released observations through the proxy, runs the trained contextual model, releases retrospective labels, checks future actuals stay hidden, opens the dashboard route and downloads the report CSV and UTF-8 master report. Temporary outputs and services are cleaned up.
