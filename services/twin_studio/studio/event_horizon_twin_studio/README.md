# Event Horizon — 3D Twin Studio v2

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

## Backend integration

`lab_server.py` serves the interface and these local APIs:

| Endpoint | Behavior |
|---|---|
| GET `/api/health` | Confirms the Python engine is online |
| POST `/api/venue-image` | Validates and stores uploaded image bytes for the server session |
| POST `/api/routes` | Converts and validates imported route JSON |
| POST `/api/render` | Validates the full scene and renders all three runs in Python |
| GET `/api/runs/{runId}/{filename}` | Downloads generated CSVs or incident manifest |

The backend uses `simulation.py`, with seeded random behavior and movement outputs checked against the JavaScript engine. Generated files live in a temporary session folder. Download them to retain them. The browser also keeps result frames for playback. Hosted static mode uses a browser worker when the Python API is unavailable; its badge says **BROWSER ENGINE**. A local Python server is not deployed to static hosting.

## Three CSVs

| Filename | Base population | Events |
|---|---|---|
| `01_ideal_flow.csv` | Your route populations | No imposed incident |
| `02_custom_disasters.csv` | Same base + manual surge arrivals | Your named incidents |
| `03_random_collapses.csv` | Same base population | Seeded random collapses; excludes manual incidents |

Schemas and timestamp grids match. Base arrivals and per-agent nominal speed draws are identical across runs. Ideal means no imposed disaster; it may still queue at high population.

Each CSV has one row per route per one-second simulation step:

`run, seed, time_s, route_id, route_name, spawned, exited, active, completed, mean_speed_m_s, mean_density_p_m2, peak_local_density_p_m2, queued, net_flow`

Spawned/exited are per-step counts; completed is cumulative. Mean density is active people divided by route length × stated width. Peak local density is a coarse occupied 2 m × 2 m cell estimate across routes. Queued means speed below 0.1 m/s. Net flow is spawned minus exited. Endpoints mean route completion, not automatically a surveyed venue exit.

Export run config + incident log downloads scenario truth separately. Keep future incident schedules out of prediction inputs. These CSVs contain synthetic movement features, not a trained real-world prediction model.

## Modeling limits

The 3D renderer visualizes the same ground-plane simulation frames as Plan; it does not add a physical 3D simulation. Pedestrians follow polylines with coarse density-based slowing and 0.65 m front-to-back spacing. Movement is single-file along each centerline; route width affects displayed corridors and average density. Base arrivals span the first 65% of duration. Surges add people at the marker over the event interval. Collapses block passage temporarily; no falls, injury or pressure mechanics are modeled. Random incidents select route, location and time using the seed.

Venue images do not infer walls or walkable geometry. Routes can cross visual obstacles if you draw them that way. No automatic free-space pathfinding, multi-lane passing, CCTV connection, perspective correction, biomechanics or operational warning model is included. Accurate digital-twin behavior still requires surveyed geometry and calibration against observed movement.

Limits: 12 routes, 3000 total people per run, 900 seconds, 10 random collapses. Start with 100–500 people; large runs use substantial memory. The 3D scene has assumed vendor-stall heights of roughly 2.4 m and marker heights around 1.4 m at default scale. Decorative trees and entrance structure are illustrative.

## Checks

```sh
node test_engine.js
node test_ui.js
python test_backend.py
```

17 movement/validation tests, 7 DOM/camera math checks and 8 actual HTTP API tests passed. Backend tests include image upload, legacy route import, all three renders, downloadable CSVs, invalid timing and Python/JS numerical parity. DOM checks cover startup with the browser's reserved location property, route creation, upload, rendering and download controls. Browser visual/WebGL hardware QA was unavailable in the build environment; the 3D display needs checking on your laptop.

## Stack and disclosure

Buildless HTML/CSS/JavaScript, Canvas, native WebGL, browser worker fallback, and Python standard-library API. No database, containers, credentials or environment variables needed. Node is needed only for development tests, not to run the application. OpenAI Codex assisted implementation. Synthetic crowd data only; geometry is based on the user's sketch. No third-party library or dataset is bundled.
