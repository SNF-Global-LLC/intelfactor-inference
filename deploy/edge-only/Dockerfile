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

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir flask opencv-python-headless numpy

# Copy application code
COPY packages/ ./packages/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p /data/evidence /data/models

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

# Run the station
CMD ["python", "-m", "packages.inference.cli", "serve", "--api-only"]
