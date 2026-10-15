# Main app image (Hugging Face Space, Docker SDK). One container, one public port (7860):
#   nginx :7860  ->  FastAPI :8000  for /api/*, /health, /docs, /redoc, /openapi.json
#                ->  Streamlit :8501 for everything else (including its websocket)
# The demo database is seeded at build time from the committed models with
# `seed-serving`, which never writes to models/. Spaces run the container as uid 1000.
FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/home/user

RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user

WORKDIR /app
COPY requirements.deploy.txt ./
RUN pip install -r requirements.deploy.txt

COPY --chown=user:user crowd_reliability ./crowd_reliability
COPY --chown=user:user dashboard ./dashboard
COPY --chown=user:user models/production ./models/production
COPY --chown=user:user data/demo ./data/demo
COPY --chown=user:user deploy/hf_space ./deploy/hf_space
RUN chown user:user /app

USER user
RUN python -m crowd_reliability.cli seed-serving --seed 42

EXPOSE 7860
CMD ["bash", "deploy/hf_space/start.sh"]
