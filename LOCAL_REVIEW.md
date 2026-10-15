# Unpublished random-flow review

Nothing in this package pushes code, publishes a site or deploys to Railway.

## Watch first

Open Event_Horizon_Local_Preview.html in Chrome or Edge. Choose one of the three recorded tests, press Play, scrub the timeline, or jump to 60s for the clearance comparison. The recordings contain actual outputs from the trained Python backend. Offline playback does not perform new inference.

## Run the full interactive system

Extract the ZIP completely. Install Python 3.12 (or 3.11). On Windows double-click START_REVIEW_WINDOWS.bat. On macOS/Linux run `python3 start_review.py`. First launch creates a local environment and installs CPU-only dependencies; it may take several minutes. Keep the terminal open. The dashboard opens automatically at a printed localhost address.

Choose density (people/m²), crowd direction, duration, route count and seed. Generate new curves, then Start live test. Curves are constrained to the schematic venue's walkable space. Choose Random for an unknown normal/surge/obstruction/combined scenario, or a specific test condition. The forecasting model is not given the scenario schedule.

Use 1× playback for a real-time demonstration, 4× or As fast as possible to test quickly. Density initializes the population along route corridors, with a small sustained arrival flow. Opposing directions reverse alternating streams. This is a kinematic synthetic demonstrator, not calibrated crowd physics or real CCTV synchronization.

Compare clearance actions pauses the world and reconstructs candidate states from current received reports. It compares continued operation, metered arrivals and, where reports suggest a bottleneck, clearing that bottleneck with/without metering. The recommendation is best among these synthetic trials. Clearing requires operator confirmation that inspection found it feasible. Apply and Resume to observe the actual synthetic response. Unknown future incidents are not used by the planner.

## What the local tests actually found

The current checkpoint runs end-to-end, but the new geometry and flow distribution expose poor generalization. Neither recorded stress test produced a warning before the first confirmed congestion incident. False positives and missed target windows appear in the results. The normal control had no sustained congestion but did have false positive forecasts. No model was retrained or thresholds tuned to hide these failures.

The incident test is density >=2.5 people/m² or queue >=15 people for 3 consecutive seconds. These are explicit demonstration policies, not operational safety thresholds. Fixed +30/+60/+120s forecasts are scored after their outcome windows mature. A warning is an advance warning only if issued before onset and its predicted target lies between onset and 30 seconds after onset.

Density means initial population divided by summed route-corridor area. Overlapping streams count separately; this is not unique floor-area occupancy. Full evacuation time is marked beyond the planning horizon if anyone, including people waiting outside, remains.

## Verification

The standalone review originally passed 152 backend tests. The broader main/parallel integration evidence is recorded in docs/INTEGRATION_AUDIT_RECONCILIATION.md. Local frontend/API integration passed curved-route generation, real model ticking, forecast rows, incident metrics, pause and clearance comparison. The existing Studio Python and JavaScript suites are also included. Browser visual verification could not run in the authoring environment because a browser download was blocked.

Inference receives time-released observation reports only. Truth is sent separately to retrospective evaluation after inference. This is logical separation in one operator process, not an access-control guarantee. Sessions are in memory and are lost when stopped. The live demo is capped at 300 seconds and 1600 initial people (3000 total scheduled); large runs slow playback rather than skipping ticks.
