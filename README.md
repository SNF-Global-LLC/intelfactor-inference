# IntelFactor Inference Engine

Edge-native manufacturing quality inspection + closed-loop root cause analysis.

Runs entirely on NVIDIA Jetson — no cloud required. Detects defects, explains why they happened, recommends what to do about it, and learns from operator feedback.

## Quick Start

### Docker (Recommended)

```bash
# Clone
git clone https://github.com/tonesgainz/intelfactor-inference.git
cd intelfactor-inference

# Deploy edge-only (no cloud)
./scripts/deploy.sh edge-only

# Or deploy hybrid (with cloud sync to app.intelfactor.ai)
./scripts/deploy.sh hybrid

# Dashboard: http://localhost:8080
```

### Using Make

```bash
make build          # Build Docker images
make deploy-edge    # Deploy edge-only mode
make deploy-hybrid  # Deploy hybrid mode
make test           # Run tests
make help           # Show all commands
```

### Manual Install

```bash
pip install -e ".[jetson]"
intelfactor-station doctor --config configs/station.yaml
intelfactor-station run --config configs/station.yaml
```

## Architecture

```
Camera ──► Vision (YOLOv8/TensorRT) ──► DefectIQ Rules ──► RCA Pipeline
                                                              │
                ┌─────────────────────────────────────────────┘
                ▼
    ┌──── Accumulator (SQLite, 30-day rolling, z-score anomaly)
    ├──── Correlator (process parameters, Pearson drift detection)
    ├──── Explainer (Qwen 3B via llama.cpp, bilingual CN/EN)
    └──── Recommender (SOP-linked actions, causal triples)
                │
                ▼
    Operator Dashboard (Chinese-primary, accept/reject feedback)
                │
                ▼
    Causal Triple Store (defect → cause → outcome, verified by operators)
```

## Deployment Modes

| Mode | Description | Cloud Dependency |
|------|-------------|------------------|
| **Edge-Only** | Everything local, no internet required | None |
| **Hybrid** | Local + sync to app.intelfactor.ai | Optional (outbound only) |

### Edge-Only Mode

```bash
cd deploy/edge-only
cp .env.example .env
# Edit .env with your STATION_ID, CAMERA_URI
docker compose up -d
```

### Hybrid Mode (Cloud Integration)

```bash
cd deploy/hybrid
cp .env.example .env
# Edit .env with:
#   - CLOUD_API_URL=https://api.intelfactor.ai
#   - CLOUD_API_KEY=your_key_from_app.intelfactor.ai
docker compose up -d
```

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for full cloud integration guide.

## Project Structure

```
packages/inference/
├── providers/              # GPU-agnostic vision + language providers
│   ├── resolver.py         #   Auto-detect hardware, select optimal models
│   ├── vision_trt.py       #   TensorRT inference (YOLOv8)
│   └── language_llama.py   #   llama.cpp inference (Qwen 3B)
├── rca/                    # 4-Layer Root Cause Analysis
│   ├── accumulator.py      #   Defect pattern accumulation + anomaly
│   ├── correlator.py       #   Process parameter correlation + drift
│   ├── explainer.py        #   Bilingual explanation via SLM
│   └── recommender.py      #   SOP-linked recommendations
├── storage/                # Storage abstraction layer
│   ├── factory.py          #   STORAGE_MODE routing (local/cloud)
│   ├── sqlite_events.py    #   SQLite event store
│   └── sqlite_evidence.py  #   Evidence index + filesystem
├── api_v2.py               # REST API with evidence endpoints
├── sync_cloud.py           # Cloud sync agent
├── evidence.py             # JPEG evidence writer with disk quota
└── static/index.html       # Operator dashboard (Chinese-primary)

deploy/
├── edge-only/              # Single-device deployment
├── hybrid/                 # Edge + cloud sync
└── hub/                    # Site hub (Postgres + Grafana)

docs/
├── ARCHITECTURE.md         # Technical architecture
├── INTEGRATION.md          # Cloud integration guide
└── LOCAL_MODE.md           # Local deployment guide
```

## Hardware Support

| Device | VRAM | Role | Vision Model | Language Model |
|--------|------|------|--------------|----------------|
| Orin Nano Super | 8GB | Station | YOLOv8n FP16 | Qwen-2.5-3B Q4 |
| Orin NX 16GB | 16GB | Station | YOLOv8m FP16 | Qwen-2.5-7B Q4 |
| AGX Orin 64GB | 64GB | Hub | YOLOv8l FP16 | Qwen-2.5-14B Q8 |
| GPU Server | 24GB+ | Site hub | YOLOv8x FP16 | Qwen-2.5-14B vLLM |

Auto-detected via `CapabilityResolver`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/events` | GET/POST | Defect events |
| `/api/v1/evidence/:id/image.jpg` | GET | Evidence image |
| `/api/triples` | GET/PATCH | Causal triples |
| `/api/recommendations` | GET | Pending recommendations |
| `/api/feedback` | POST | Operator feedback |

Full API reference: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#api-design)

## Tests

```bash
# Create venv and run tests
python3 -m venv .venv && source .venv/bin/activate
pip install pytest flask numpy
python -m pytest tests/ -v
# 19 passed
```

## Documentation

- [Technical Architecture](docs/ARCHITECTURE.md) — System design, data flow, scaling
- [Cloud Integration](docs/INTEGRATION.md) — Connecting to app.intelfactor.ai
- [Local Mode](docs/LOCAL_MODE.md) — Edge-only deployment guide
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) — Bilingual CN/EN instructions

## License

Proprietary — IntelFactor.ai
