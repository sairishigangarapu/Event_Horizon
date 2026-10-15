# Railway deployment

The root Dockerfile now serves the integrated Studio at `/` and the separate runtime dashboard at `/dashboard/`. One public ASGI process binds to Railway's `PORT` on `0.0.0.0`. Studio's internal HTTP server uses a private ephemeral loopback port. `/api/ready` verifies model artifacts loaded and Studio is running. Browser links and requests stay on the deployment origin; no visitor-local backend URL is exposed.

## Deploy from GitHub

1. Connect Railway and give its GitHub integration access to Manipal-Hackathon-2026/MKCrusaders26.
2. Create a service from that repository, branch main, root directory `/`.
3. Railway reads railway.json and the root Dockerfile. No custom build/start command is needed.
4. Generate a public domain after the service is running.
5. Open `/`, load the example scene, render, play, and open `/dashboard/`.

The app needs no OpenAI API key and no external database for the synthetic demonstration. It loads the included synthetic-trained checkpoint by default. It does not claim validated field safety predictions.

## CLI equivalent

From the repository root, after authenticating to the intended Railway account:

    npm install --global @railway/cli
    railway login
    railway init --name event-horizon
    railway add --service app
    railway up --service app
    railway domain --service app

If a project already exists, use `railway link` instead of creating another project. `railway up` uploads local source and does not itself create a public domain.

## GitHub Actions

`.github/workflows/railway.yml` runs the backend, Person 4, Studio, frontend, two-service and single-port tests, then builds the container. Railway deployment runs only after that job succeeds AND these repository variables are configured:

- RAILWAY_PROJECT_ID
- RAILWAY_SERVICE_ID
- RAILWAY_ENVIRONMENT (optional; defaults to production)

Add an environment-scoped project token as the GitHub secret RAILWAY_TOKEN. Never commit a token. Alternatively use Railway's GitHub autodeploy; choose one deployment trigger to avoid duplicate deployments. If CI is the gate, disable independent Railway push autodeploy or configure its checks gate.

The legacy Hugging Face workflow is manual-only and uses the preserved Dockerfile under deploy/hf_space. It no longer deploys automatically on each main push.

## Runtime storage and capacity

Keep one replica and one Uvicorn worker because live session state is in memory. Restarting/redeploying discards live sessions. Scene decisions and generated files default to `/app/data/studio_outputs`; without a mounted Railway volume they are ephemeral. For retained run files mount a writable volume and set EH_OUTPUT_ROOT to a directory on it. Configure DATABASE_URL separately if retaining SQLite data. Test actual memory usage with the intended population/duration before increasing simulation load.

## Verify

    python verify_hosted.py

This starts the actual hosted ASGI entrypoint on a temporary port and verifies page/asset routing, same-origin backend configuration, render, released observations, trained runtime inference, matured labels, hidden future actuals and a Unicode master report.

A local Docker engine was not available during preparation, so the container build must be verified by the workflow or Railway builder. The actual Python hosted process passed its end-to-end TCP test.

If GitHub reports a failed job with no runner steps/logs, the code tests have not run in that job. Inspect its annotations and organization Actions availability/billing/runner policy; changing application source cannot repair an account-level runner restriction.

References:
- https://docs.railway.com/config-as-code/reference
- https://docs.railway.com/deployments/healthchecks
- https://docs.railway.com/cli/deploying
