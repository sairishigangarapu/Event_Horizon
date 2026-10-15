# New: separated observation and truth exports

See [DATA_OUTPUTS.md](DATA_OUTPUTS.md) for the two outputs and observation-only replay API. Every Python render writes these automatically.

# Event Horizon — 3D Twin Studio v3

Your original venue sketch is loaded as a 3D schematic: entrance, ring pathways, vendor courtyard, pond, eastern annex and three illustrative exits. Routes start empty and remain yours to draw. The sketch is included in `docs/original_venue_sketch.png` for reference. Dimensions, heights, stalls, trees and exit positions are assumptions rather than surveyed geometry.

## Start on Windows

1. Extract the entire ZIP into a new folder.
2. Stop the old server with Ctrl+C and close its browser tab.
3. Open `event_horizon_twin_studio`; double-click `START_WINDOWS.bat`.
4. The new browser window opens at `http://127.0.0.1:8000/`. The top badge should say **PYTHON API ONLINE**.

Or run in the extracted folder:

```powershell
python lab_server.py
```

Keep that terminal open. Use `python lab_server.py --port 8001` if the old port is occupied. Do not open index.html directly, and do not use `python -m http.server` if you want the Python API.

Python 3.10+ and a modern browser are sufficient; the local app has no external packages, CDN dependencies or API keys. WebGL is needed for 3D. Plan view is available if WebGL is disabled.

## Use the studio

- **3D**: drag to orbit, scroll to zoom, View to reset the camera. Your original venue is already loaded. Plan switches to its overhead drawing. The crowd uses small 3D markers with colored bodies and light heads.
- **Venue**: upload a PNG/JPG/WEBP/GIF using the visible file picker. The picture is decoded, validated by the Python API and shown as a plan; select 3D to see it as a ground reference. A single picture is not automatically reconstructed into geometry. Use original venue restores the built-in 3D sketch model. Set the scale from a measured overhead distance. Default 0.1 m/unit gives a 100 m × 70 m canvas; these dimensions are assumed.
- **Routes**: Draw a route, click successive ground positions, and Finish route. Dragging in 3D rotates instead of placing a point. Set population, name, speed and width. Undo removes the last draft point. Routes are centerlines; draw around obstacles.
- **Import routes**: choose your earlier `routes (2).json` using the Routes file picker. Both path-editor-v1 and current Studio route arrays are supported. Legacy metre coordinates convert using the selected scale; IDs and waypoint geometry are preserved. Import replaces the route list and clears manual incidents. No routes are imported automatically.
- **Incidents**: name the event, choose surge or collapse, set start/end, and pick a location on a drawn route. Add, edit and delete work. Require 0 ≤ start < end ≤ duration, in integer seconds.
- **Runs**: Render all three. Select a result card, then Play or scrub. Switching between 3D and Plan preserves the run and timestamp. Density is a Plan overlay. Download each CSV or all three. Browsers may ask to allow multiple downloads; individual buttons work independently.
- **Save/open project**: a Studio scene JSON contains routes, manual incidents, settings and optional background. Use the dedicated Routes import for old route-only files. Save a project if you want to retain it between sessions.

## v3 fixes and master record

- Heatmap is on by default in both 3D and Plan, on a fixed 0–6+ people/m² legend. It uses actual occupied 2 m cells, not event labels. Display choices no longer switch you out of 3D.
- Collapses have a selectable spatial radius (default 3 m). Every drawn path intersecting that footprint is blocked, including another route and repeated visits to the same place. New arrivals inside a blocked source wait outside admission; existing pedestrians inside stop; pedestrians already downstream can continue. Start is inclusive and end is the reopening time. The final frame is a snapshot and does not advance an extra second.
- Surge names do not override their type. A Surge named “collapse” still adds people and gets an explicit warning. Unsaved incident form changes must be saved or discarded before rendering.
- The selected run and currently active incidents are displayed on the map. Ideal excludes all imposed incidents.
- **Master report** exports the current configuration, recorded decisions, maps, metrics and all three run comparisons. It works before rendering too, with no run results claimed.
- Every committed route/incident/settings edit, route drawing action, important display choice, rejected render, export and user note is timestamped with details and before/after configuration differences. Draft field typing is captured when saved or rejected; it is not a keystroke recorder. Add a decision note to record your reasoning.
- Autosave retains the scene and journal in this browser when storage is available. Save project includes the journal for transfer. The local backend appends decision records under outputs/decisions_<project-id>.jsonl. Large images can exceed browser storage; the journal status tells you to Save project explicitly.
- Every local render writes its three CSVs, manifest and master_report.html under outputs/<run-id>/. The on-demand report includes the most recent browser decisions; the server report captures the render-time journal.
- Your unchanged uploaded scene is in examples/event_horizon_scene.json. Open it using Open project. Fresh v3 CSVs and its master report are in analysis/. Its three events named “crowd collapse” are actually Surge events; this has been flagged, not silently converted. Earlier editing history was not supplied and cannot be reconstructed.
- To generate a report from a saved scene without opening the interface: `python reporting.py examples/event_horizon_scene.json --out master_report.html`. The self-contained report can be opened in your browser and printed to PDF.

## Backend integration

`lab_server.py` serves the interface and these local APIs:

| Endpoint | Behavior |
|---|---|
| GET `/api/health` | Confirms the Python engine is online |
| POST `/api/venue-image` | Validates and stores uploaded image bytes for the server session |
| POST `/api/routes` | Converts and validates imported route JSON |
| POST `/api/render` | Validates the full scene and renders all three runs in Python |
| GET `/api/runs/{runId}/{filename}` | Downloads CSVs, manifest or master report |
| POST `/api/decisions` | Appends a project decision to its persistent local journal |

The backend uses `simulation.py`, with seeded random behavior and movement outputs checked against the JavaScript engine. Generated files live in a persistent outputs folder beside the server. Download them to retain them. The browser also keeps result frames for playback. Hosted static mode uses a browser worker when the Python API is unavailable; its badge says **BROWSER ENGINE**. A local Python server is not deployed to static hosting.

## Three CSVs

| Filename | Base population | Events |
|---|---|---|
| `01_ideal_flow.csv` | Your route populations | No imposed incident |
| `02_custom_disasters.csv` | Same base + manual surge arrivals | Your named incidents |
| `03_random_collapses.csv` | Same base population | Seeded random collapses; excludes manual incidents |

Schemas and timestamp grids match. Base arrivals and per-agent nominal speed draws are identical across runs. Ideal means no imposed disaster; it may still queue at high population.

Each CSV has one row per route per one-second simulation step:

`run, seed, time_s, route_id, route_name, scheduled, spawned, exited, active, waiting_entry, completed, mean_speed_m_s, mean_density_p_m2, peak_local_density_p_m2, queued, net_flow`

Spawned/exited are per-step counts; completed is cumulative. Mean density is active people divided by route length × stated width. Peak local density is an occupied 2 m × 2 m cell estimate across routes; edge cells use their clipped area. scheduled counts intended arrivals and spawned counts admissions. waiting_entry records scheduled people waiting to enter at a blocked source; it is excluded from inside density. Queued means speed below 0.1 m/s. Net flow is spawned minus exited. Endpoints mean route completion, not automatically a surveyed venue exit.

Export run config + incident log downloads scenario truth separately. Keep future incident schedules out of prediction inputs. These CSVs contain synthetic movement features, not a trained real-world prediction model.

## Modeling limits

The 3D renderer visualizes the same ground-plane simulation frames as Plan; it does not add a physical 3D simulation. Pedestrians follow polylines with coarse density-based slowing and 0.65 m front-to-back spacing. Movement is single-file along each centerline; route width affects displayed corridors and average density. Base arrivals span the first 65% of duration. Surges add people at the marker over the event interval. Collapses block passage temporarily; no falls, injury or pressure mechanics are modeled. Random incidents select route, location and time using the seed.

Venue images do not infer walls or walkable geometry. Routes can cross visual obstacles if you draw them that way. No automatic free-space pathfinding, multi-lane passing, CCTV connection, perspective correction, biomechanics or operational warning model is included. Accurate digital-twin behavior still requires surveyed geometry and calibration against observed movement.

Limits: 12 routes, 3000 total people per run, 900 seconds, 10 random collapses. Start with 100–500 people; large runs use substantial memory. The 3D scene has assumed vendor-stall heights of roughly 2.4 m and marker heights around 1.4 m at default scale. Decorative trees and entrance structure are illustrative.

## Checks

```sh
node test_engine.js
node test_ui.js
node test_regressions.js
python test_backend.py
```

49 checks passed: 17 existing engine checks, 11 barrier/heatmap/report regressions, 10 DOM/camera checks and 11 actual backend/report checks. Backend tests include image upload, legacy route import, all three renders, downloadable CSVs, invalid timing and Python/JS numerical parity. DOM checks cover startup with the browser's reserved location property, route creation, upload, rendering and download controls. Browser visual/WebGL hardware QA was unavailable in the build environment; the 3D display needs checking on your laptop.

## Stack and disclosure

Buildless HTML/CSS/JavaScript, Canvas, native WebGL, browser worker fallback, and Python standard-library API. No database, containers, credentials or environment variables needed. The outputs directory contains persistent local data and is excluded from source packaging. Node is needed only for development tests, not to run the application. OpenAI Codex assisted implementation. Synthetic crowd data only; geometry is based on the user's sketch. No third-party library or dataset is bundled.
