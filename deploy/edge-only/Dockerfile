# IntelFactor.ai — Edge Station Dockerfile
# Runs inference pipeline + local API on NVIDIA edge devices.

FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy full project for pip install
COPY . .

# Install project with jetson extras (includes all real deps)
RUN pip install --no-cache-dir ".[jetson]"

# Create data directories
RUN mkdir -p /data/evidence /data/models /opt/intelfactor/config

# Copy default config if available
RUN cp -r configs/* /opt/intelfactor/config/ 2>/dev/null || true

# Environment defaults
ENV STORAGE_MODE=local
ENV SQLITE_DB_PATH=/data/local.db
ENV EVIDENCE_DIR=/data/evidence
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the station (calls run_station() which starts runtime + API)
CMD ["python", "-c", "from packages.inference.cli import run_station; run_station()"]
