#!/usr/bin/env bash
# Starts the API, the dashboard and the proxy. If any of them exits, the
# container exits too, so the platform restarts it instead of serving half an app.
set -euo pipefail
cd /app

python -m uvicorn crowd_reliability.api:app --host 127.0.0.1 --port 8000 &

python -m streamlit run dashboard/app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true &

nginx -e /dev/stderr -c /app/deploy/hf_space/nginx.conf -g 'daemon off;' &

wait -n
echo "[start.sh] a service exited; stopping the container" >&2
exit 1
