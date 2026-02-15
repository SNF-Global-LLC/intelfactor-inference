# Claude Session Context

## Project: IntelFactor Inference Engine

Edge-first industrial quality inspection platform for NVIDIA devices.

## Current State (2026-02-14)

### Critical Fixes Applied (California MVP)

1. **Dockerfile CMD** - Fixed broken entrypoint
   - Was: `python -m packages.inference.cli serve --api-only` (doesn't exist)
   - Now: `python -c "from packages.inference.cli import run_station; run_station()"`

2. **Dependency Installation** - Fixed missing deps
   - Was: manual `pip install flask opencv numpy` (incomplete)
   - Now: `pip install ".[jetson]"` (installs from pyproject.toml)

3. **API Switch** - Use v2 with evidence endpoints
   - Was: `from packages.inference.api import create_app`
   - Now: `from packages.inference.api_v2 import create_app`

4. **Sync Dockerfile** - Fixed to include full project
   - Was: only copies sync_cloud.py (missing storage, schemas)
   - Now: copies full project, `pip install ".[sync]"`

### Completed
- [x] Storage abstraction layer (SQLite for local, cloud-ready interface)
- [x] API v2 with evidence serving endpoints
- [x] Docker Compose profiles (edge-only, hybrid)
- [x] Cloud sync agent for hybrid mode
- [x] Real YOLOv8 output parser (replaced stub)
- [x] Manifest.jsonl support for evidence sync
- [x] Model setup script (scripts/setup_models.sh)
- [x] intelfactor-doctor CLI entry point
- [x] **Deployment fixes for California MVP**
- [x] **RUNBOOK_MVP.md** created
- [x] 78 passing tests

### Files Modified for Deployment Fix
```
deploy/edge-only/Dockerfile      # Fixed CMD, proper pip install
deploy/hybrid/Dockerfile.sync    # Fixed to include full project
packages/inference/cli.py        # Switched to api_v2.create_app
docs/RUNBOOK_MVP.md              # NEW: Deployment guide
```

### Quick Deploy Commands

```bash
# Edge-only (no cloud)
cd deploy/edge-only
cp .env.example .env && vim .env
docker compose up -d
curl http://localhost:8080/health

# Hybrid (with cloud sync)
cd deploy/hybrid
cp .env.example .env && vim .env
docker compose up -d
docker logs -f intelfactor-sync-agent
```

### Repository
- **GitHub**: https://github.com/tonesgainz/intelfactor-inference
- **Tests**: 78 passing (pytest)

### API Endpoints (v2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/events` | GET/POST | Defect events |
| `/api/triples` | GET/PATCH | Causal triples |
| `/api/feedback` | POST | Operator feedback |
| `/api/v1/evidence/<id>` | GET | Evidence metadata |
| `/api/v1/evidence/<id>/image.jpg` | GET | Evidence image |
| `/api/v1/evidence/manifest` | GET | Evidence manifest |

### Environment Variables
```bash
# Required
STORAGE_MODE=local
SQLITE_DB_PATH=/data/local.db
EVIDENCE_DIR=/data/evidence
STATION_ID=station_01

# For hybrid mode
CLOUD_API_URL=https://api.intelfactor.ai
CLOUD_API_KEY=ifk_your_key
S3_BUCKET=intelfactor-evidence
```

### Tests
```bash
source .venv/bin/activate
python -m pytest tests/ -v
# 78 passed
```
