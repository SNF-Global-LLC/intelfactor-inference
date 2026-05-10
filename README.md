# IntelFactor Inference Engine

Edge-native manufacturing quality inspection + root cause analysis.

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

## Local Operator Console

`/inspect` is the local edge operator console for factory inspection. It is designed to run offline against local SQLite and local evidence files. It does not require cloud sync, Kafka, S3, CloudFront, training jobs, or the IntelBase web dashboard in the inspection hot path.

### Run `/inspect` Locally

For API-only local validation without camera hardware:

```bash
mkdir -p data/evidence
EDGE_API_KEY=local-dev-key \
STATION_API_KEY=local-dev-key \
DB_PATH=data/operator-console.db \
SQLITE_DB_PATH=data/operator-console.db \
STORAGE_MODE=local \
EVIDENCE_DIR=data/evidence \
STATION_ID=station_dev \
python3 -c "from packages.inference.api_v2 import create_app; app=create_app(); app.run(host='127.0.0.1', port=8080, debug=False)"
```

Open:

```text
http://localhost:8080/inspect
```

The console calls local `/api/...` endpoints. Enter the same local edge API key in the console before loading status, history, queue data, or recording operator actions. If the key is missing or wrong, the API fails closed.

### Seed Local Inspection Rows

Use the seed script to create local PASS, REVIEW, and DEFECT inspection rows for UI validation:

```bash
python3 scripts/seed_operator_console.py \
  --db-path data/operator-console.db \
  --evidence-dir data/evidence \
  --station-id station_dev
```

The script is safe to rerun. It replaces the same deterministic seed inspection IDs and writes placeholder evidence files under `data/evidence/operator-console-seed/`.

### Before Jetson Validation

The local API-only path proves the operator console, local SQLite reads, review actions, and evidence routing. It does not prove FLIR capture, TensorRT engine loading, real camera status, touch-screen ergonomics, or station supervisor recovery. Use [docs/JETSON_OPERATOR_CONSOLE_VALIDATION.md](docs/JETSON_OPERATOR_CONSOLE_VALIDATION.md) before calling the console production-ready on Jetson hardware.

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
    Operator Console (confirm_defect / override_to_pass)
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
└── static/inspect.html     # Local edge operator console

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
# Create venv and run focused edge/API tests
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" flask
python3 -m pytest tests/test_api_v2.py tests/test_sync_inspections.py tests/test_sync_cloud.py -q
```

## Documentation

- [Technical Architecture](docs/ARCHITECTURE.md) — System design, data flow, scaling
- [Cloud Integration](docs/INTEGRATION.md) — Connecting to app.intelfactor.ai
- [Local Mode](docs/LOCAL_MODE.md) — Edge-only deployment guide
- [Jetson Operator Console Validation](docs/JETSON_OPERATOR_CONSOLE_VALIDATION.md) — Hardware validation checklist for `/inspect`
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) — Bilingual CN/EN instructions

## License

Proprietary — IntelFactor.ai
