# Claude Session Context

## Project: IntelFactor Inference Engine

Edge-first industrial quality inspection platform for NVIDIA devices.

## Current State (2026-02-14)

### Completed
- [x] Storage abstraction layer (SQLite for local, cloud-ready interface)
- [x] API v2 with evidence serving endpoints
- [x] Docker Compose profiles (edge-only, hybrid)
- [x] Cloud sync agent for hybrid mode
- [x] Doctor script for system health checks
- [x] 19 passing tests (storage + API)
- [x] Technical architecture documentation
- [x] Local mode documentation
- [x] Cloud integration guide (app.intelfactor.ai)
- [x] Makefile for build/deploy automation
- [x] Deploy script (scripts/deploy.sh)
- [x] Updated README with quick start

### Repository
- **GitHub**: https://github.com/tonesgainz/intelfactor-inference
- **Commits**: 5

### Key Files Created This Session
```
packages/inference/storage/          # Storage abstraction layer
├── __init__.py
├── factory.py                       # STORAGE_MODE routing
├── base.py                          # Abstract interfaces
├── sqlite_base.py                   # SQLite connection + migrations
├── sqlite_events.py                 # Event store
├── sqlite_evidence.py               # Evidence store
└── sqlite_triples.py                # Triple store

packages/inference/api_v2.py         # Enhanced API with evidence endpoints
packages/inference/sync_cloud.py     # Cloud sync agent
packages/inference/.env.example      # Environment template

deploy/edge-only/                    # Edge-only deployment
├── docker-compose.yml
├── Dockerfile
└── .env.example

deploy/hybrid/                       # Hybrid deployment
├── docker-compose.yml
├── Dockerfile.sync
└── .env.example

docs/ARCHITECTURE.md                 # Technical architecture (666 lines)
docs/LOCAL_MODE.md                   # Local deployment guide

scripts/doctor.py                    # System health checks

tests/test_storage.py                # Storage layer tests (9 tests)
tests/test_api_v2.py                 # API v2 tests (10 tests)
```

### Next Steps (User's Question)
User asked: "how would I build this and integrate with https://app.intelfactor.ai"

Planned response would cover:
1. **Build Docker images** for edge-only and hybrid modes
2. **Configure cloud sync** with app.intelfactor.ai API
3. **Set up authentication** (API keys, bearer tokens)
4. **Deploy to edge device** (Jetson/GPU server)
5. **Verify end-to-end flow**: camera → inference → evidence → cloud sync

### Environment
- Working directory: `/Users/tonyadmin/edgefirst/intelfactor-inference`
- Python venv: `.venv` (pytest, flask, numpy installed)
- Git remote: `origin` → `github.com:tonesgainz/intelfactor-inference.git`

### Key Environment Variables
```bash
STORAGE_MODE=local|cloud
SQLITE_DB_PATH=/opt/intelfactor/data/local.db
EVIDENCE_DIR=/opt/intelfactor/data/evidence
STATION_ID=station_01
API_PORT=8080

# For hybrid mode:
CLOUD_API_URL=https://api.intelfactor.ai
CLOUD_API_KEY=<your_key>
S3_BUCKET=intelfactor-evidence
```

### Architecture Summary
```
Camera → Vision (TensorRT) → RCA Pipeline → Dashboard
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
              [EDGE-ONLY]                    [HYBRID]
              SQLite + local              SQLite + cloud sync
              No cloud deps               → app.intelfactor.ai
```

### Tests
```bash
source .venv/bin/activate
python -m pytest tests/test_storage.py tests/test_api_v2.py -v
# 19 passed
```
