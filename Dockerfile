FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.runtime.txt requirements.railway.txt ./
# CPU wheels avoid shipping CUDA libraries into a CPU Railway service.
RUN pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.1,<3' \
 && pip install -r requirements.railway.txt
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser crowd_reliability ./crowd_reliability
COPY --chown=appuser:appuser models/production ./models/production
COPY --chown=appuser:appuser models/candidates/mc_v2 ./models/candidates/mc_v2
COPY --chown=appuser:appuser runtime_frontend ./runtime_frontend
COPY --chown=appuser:appuser studio ./studio
COPY --chown=appuser:appuser runtime_server.py hosted_server.py ./
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn hosted_server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips '*' "]
