# Twin Studio Phase 4 service image: publisher, observation API, reference API, integration tests.
# Contains code only. No secret is copied or baked in; all secrets arrive as Compose file secrets
# under /run/secrets at run time (see .dockerignore at the repository root).
FROM python:3.12.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY deploy/twin_studio/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt \
 && useradd --system --uid 10001 --create-home eh \
 && mkdir -p /var/lib/eh-publisher /var/lib/eh-observation /var/lib/eh-reference \
 && chown eh /var/lib/eh-publisher /var/lib/eh-observation /var/lib/eh-reference
WORKDIR /app
COPY services /app/services
COPY tests/twin_phase4_integration /app/tests/twin_phase4_integration
COPY deploy/twin_studio/specs /app/specs
USER eh
