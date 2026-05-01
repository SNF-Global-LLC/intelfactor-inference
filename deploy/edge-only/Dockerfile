# IntelFactor.ai — Edge Station Dockerfile
# Runs inference pipeline + local API on NVIDIA Jetson (JetPack 6.x).
#
# Build:  docker build -f deploy/edge-only/Dockerfile -t intelfactor-station .
# Run:    see deploy/edge-only/docker-compose.yml

FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

WORKDIR /app

# System deps (curl for healthcheck, GL for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install with Jetson extras (flask, opencv, llama-cpp, inference-sdk)
RUN pip install --no-cache-dir ".[jetson]"

# Create runtime directories
RUN mkdir -p /data/evidence /opt/intelfactor/config

# Copy default configs into image
RUN cp -r configs/* /opt/intelfactor/config/ 2>/dev/null || true

# Environment defaults — all overridable via docker-compose / .env
ENV STORAGE_MODE=local \
    SQLITE_DB_PATH=/data/local.db \
    DB_PATH=/data/local.db \
    EVIDENCE_DIR=/data/evidence \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    STATION_ID=station_01 \
    ENABLE_LOCAL_LLM=false \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Non-root user
RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /data /opt/intelfactor/config

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

EXPOSE ${API_PORT}

USER appuser

CMD ["python", "-c", "from packages.inference.cli import run_station; run_station()"]
