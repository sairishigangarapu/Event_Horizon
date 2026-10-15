# Observable reports and hidden truth

Run START_WINDOWS.bat as before, open your scene and render all three scenarios.
Each Python render now writes two separate directories under outputs/<run-id>/:

| Output | Contents | Intended use |
| --- | --- | --- |
| observable/ | ideal.jsonl, custom.jsonl, random.jsonl | Synthetic camera, counter and warden reports |
| hidden_ground_truth/ | scenario.json, per-mode exact metrics, trajectories, waiting agents, incident labels and crossing audits | Offline training and retrospective evaluation |

Each folder also has its own ZIP alongside it. OPEN_OUTPUTS_WINDOWS.bat opens the output directory. Use the newest run folder. Existing three CSV downloads remain exact simulator analysis outputs, not observation features. Reports, scene JSON, manifests and /api/render responses also contain truth; only the simulator operator should use the studio API.

## Inference and replay

Start START_INFERENCE_WINDOWS.bat in a second terminal. This starts a separate read-only observation API at port 8001. It has no static file route, scenario/configuration endpoint, rendering or truth download endpoint.

Request, replacing RUN_ID with the folder name:

    GET http://127.0.0.1:8001/api/observations/RUN_ID/custom?time_s=30

Modes: ideal, custom, random. The response contains ONLY reports where both observed_at_s <= 30 and available_at_s <= 30. A report observed at 29 but available at 34 is excluded. Missing reports are absent, never converted to zero. Reports may arrive after simulation duration.

The caller supplies replay time T; the API is not a wall-clock scheduler. To emulate streaming, advance T and deduplicate by observation_id. Do not allow an ML model to choose its own future cutoff. The simulator operator owns run selection and replay time.

For process-level isolation, copy only each run's observable/ folder to a clean tree preserving RUN_ID/observable/, and start the API with:

    python lab_server.py --inference-only --port 8001 --observation-root path_to_clean_tree

Run ML under an account/container that can access only that clean tree and the observation API, not the simulator output filesystem, studio port 8000, complete project archive or full observation ZIP. HTTP separation is enforced by the handler; operating-system isolation must be configured by whoever runs ML. Full observable JSONL/ZIP includes later reports and is for offline storage; inference uses the cutoff endpoint.

## Report model

Defaults assume route-wide camera views, entry/exit counters and route patrols. These are synthetic reporting assumptions, not calibrated CCTV coverage.

| Source | Interval | Reporting delay | Missing probability | Relative noise |
| --- | --- | --- | --- | --- |
| camera_1 | 2 s | 2 s | 0.10 | 0.08 |
| sensor_1 | 1 s | 1 s | 0.03 | 0.03 |
| warden_1 | 10 s | 5 s | 0.15 | 0.15 |

Camera: visible population, density and speed estimates. Sensor: observed entry/exit counter estimates over the current one-second window. Warden: queue estimate and current movement category. No source receives event names, future incidents, seed, true blockage flags, future arrivals or exact agent trajectories. Every observation has source identity, observed timestamp and availability timestamp. Multiplicative uniform noise cannot create false positives from zero; more realistic detection errors require calibration.

Reports use an independent deterministic RNG, keeping movement unchanged. The same scene, seed and source settings reproduce the same reports. Observations are generated from current/past movement only. Model-training splits must group whole runs/scenarios; never randomly split adjacent timestamps across training and test sets.

Custom sources are accepted by POST /api/render in observationSources, for example:

```json
[{"id":"cam_north","kind":"camera","route_ids":["YOUR_ROUTE_ID"],"interval_s":2,"delay_s":4,"missing_probability":0.1,"relative_noise":0.08}]
```

Omitted route_ids means all routes; [] means no covered routes. The current UI uses defaults; configure custom source settings through the Python API. Do not assume source settings edited in scene JSON are retained by the current UI.

## Checks

    python -m unittest test_observations test_backend
    node test_engine.js
    node test_regressions.js
    node test_ui.js

Tests cover timestamp boundaries, invalid/nonfinite cutoffs, dropout, coverage, repeatability, changing a future incident without changing earlier observations, physical export separation and HTTP refusal of truth/static/render access in inference mode. The existing kinematic movement model and its calibration limits still apply. Incident labels describe simulated obstructions/surges, not validated injury or physical crowd collapse.
