# AGENTS.md

> This file is written for AI coding agents. Expect the reader to know nothing about the project.
> All information below is derived from the actual codebase. Do not assume anything not documented here.

---

## Project Overview

**IntelFactor Inference Engine** is an edge-first AI platform for manufacturing quality inspection and predictive maintenance. It runs entirely on NVIDIA Jetson devices (Orin Nano, Orin NX, AGX Orin) and GPU servers, with optional cloud synchronization. The system performs real-time defect detection, 4-layer root cause analysis (RCA), bilingual (Chinese/English) explanation generation, and closed-loop learning from operator feedback.

The core value proposition is **offline-first autonomy**: the station runs fully without internet; cloud connectivity is additive and never gating.

---

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Language | Python >=3.10 | No PyTorch on Jetson at runtime |
| Web API | Flask >=3.0 | REST API v2 in `packages/inference/api_v2.py` |
| Vision Inference | TensorRT (YOLOv8) | Device-specific `.engine` files; no runtime PyTorch |
| Language Inference | llama.cpp (Jetson) / vLLM (GPU server) | Qwen-2.5 GGUF models |
| Edge Storage | SQLite (WAL mode) | Default; `STORAGE_MODE=local` |
| Cloud Storage | DynamoDB / S3 / Postgres | Hybrid sync only; direct `STORAGE_MODE=cloud` is stubbed |
| Message Bus | MQTT (optional) | For machine health sensor ingestion |
| Container | Docker + NVIDIA Container Toolkit | Jetson base: `nvcr.io/nvidia/l4t-ml:r36.2.0-py3` |
| Build | setuptools via `pyproject.toml` | No `setup.py` |
| Lint | ruff >=0.3 | `target-version = "py310"`, `line-length = 120` |
| Test | pytest >=8.0 | No `conftest.py`; inline fixtures per file |

---

## Project Structure

```
packages/
├── inference/                 # Core inference engine
│   ├── api_v2.py              # Main Flask REST API (~1,250 lines, ~30 endpoints)
│   ├── cli.py                 # CLI entry points: station, hub, doctor
│   ├── inspection.py          # Inspection transaction orchestrator
│   ├── doctor.py              # Pre-flight system diagnostics
│   ├── schemas.py             # Core dataclasses / enums
│   ├── evidence.py            # JPEG ring buffer writer with disk quota
│   ├── ingest.py              # Continuous camera ingest (RTSP/GigE/USB)
│   ├── capture.py             # Single-frame capture backends
│   ├── verdict.py             # Deterministic PASS/REVIEW/FAIL logic
│   ├── annotate.py            # OpenCV bounding box annotation
│   ├── sync*.py               # Cloud sync workers and utilities
│   ├── providers/             # Vision + language provider abstractions
│   │   ├── resolver.py        # Hardware auto-detection + model selection
│   │   ├── vision_trt.py      # TensorRT YOLO inference
│   │   ├── vision_roboflow.py # Roboflow hosted inference fallback
│   │   ├── language_llama.py  # llama.cpp SLM inference
│   │   ├── language_vllm.py   # vLLM server inference
│   │   └── stub.py            # Stub providers for dev/test
│   ├── rca/                   # 4-layer Root Cause Analysis pipeline
│   │   ├── accumulator.py     # Defect rate anomaly detection
│   │   ├── correlator.py      # Process parameter drift (Pearson)
│   │   ├── explainer.py       # Bilingual explanation via SLM
│   │   └── recommender.py     # SOP-linked action recommendations
│   └── storage/               # Storage abstraction layer
│       ├── factory.py         # STORAGE_MODE routing (local/cloud)
│       ├── sqlite_events.py   # SQLite event store
│       ├── sqlite_evidence.py # Evidence index + filesystem
│       └── sqlite_triples.py  # Causal triple store
├── ingestion/                 # Machine health sensor ingestion
│   ├── sensor_service.py      # MQTT subscriber, SQLite WAL store, z-scoring
│   └── schemas.py             # Sensor dataclasses
├── policy/                    # Maintenance rules engine
│   └── maintenance_iq.py      # Threshold-based HEALTHY/WARNING/CRITICAL verdicts
└── visibility/                # Production metrics & observability
    ├── metrics_api.py         # Flask blueprint for /api/metrics/*
    └── production_metrics.py  # Throughput, cycle time, utilization, idle detection

deploy/
├── edge-only/                 # Single-device deployment (Docker)
├── hybrid/                    # Edge + cloud sync sidecar
├── hub/                       # Site hub: Postgres + MinIO + Prometheus + Grafana
├── aws/                       # AWS ECS + Datadog Agent task definitions
├── station/                   # Bare-metal setup scripts
└── systemd/                   # systemd service unit

configs/
├── station.yaml               # Default production station config
├── dev-mac-station.yaml       # Mac dev config (stub providers, webcam)
├── flir-blackfly-station.yaml # Office/lab config with FLIR camera
├── station-roboflow.yaml      # Roboflow-hosted vision dev config
└── wiko_taxonomy.yaml         # Defect taxonomy for Wiko Cutlery

docs/                          # Architecture, integration, deployment guides
tests/                         # 15 Python test files (no conftest.py)
training/                      # YOLO training pipeline scripts & configs
scripts/                       # Deployment, build, validation, and utility scripts
migrations/                    # SQL schema migrations
```

---

## Build, Install, and Dev Server

### Install Dependencies

```bash
pip install -e ".[dev]"
```

After installation, console scripts are registered:
- `intelfactor-station` — Start station runtime + API
- `intelfactor-hub` — Start site hub runtime
- `intelfactor-doctor` — Run pre-flight diagnostics

> **PATH Gotcha**: After `pip install`, executables may land in `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is set.

### Run the Dev Server (API-only, no hardware)

`make dev` attempts full station startup including TensorRT model loading, which will fail in cloud VMs or machines without the hardware. Instead, start the API standalone:

```bash
mkdir -p data/evidence
DB_PATH=data/dev.db \
STORAGE_MODE=local \
SQLITE_DB_PATH=data/dev.db \
EVIDENCE_DIR=data/evidence \
STATION_ID=station_dev \
python3 -c "
import os
os.makedirs('data/evidence', exist_ok=True)
from packages.inference.api_v2 import create_app
app = create_app()
app.run(host='0.0.0.0', port=8080, debug=False)
"
```

This starts the Flask API on port 8080 with local SQLite storage and no camera/vision/language model dependencies.

### Seed Data for UI Testing

```bash
python3 scripts/seed_operator_console.py \
  --db-path data/operator-console.db \
  --evidence-dir data/evidence \
  --station-id station_dev
```

The script is idempotent. It writes placeholder evidence files under `data/evidence/operator-console-seed/`.

---

## Build and Test Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev]"` |
| Lint | `python3 -m ruff check packages/ tests/` |
| All tests | `python3 -m pytest tests/ -v` |
| API tests only | `python3 -m pytest tests/test_api_v2.py -v` |
| Sensor + maintenance tests | `python3 -m pytest tests/test_sensor_service.py tests/test_maintenance_iq.py -v` |
| Storage tests | `python3 -m pytest tests/test_storage.py -v` |
| Smoke test (needs running API) | `./scripts/production_smoke_test.sh` |
| Doctor (system checks) | `python3 scripts/doctor.py --full` |

### Docker Build / Deploy (Jetson/hardware required)

| Task | Command |
|------|---------|
| Build edge image | `make build-edge` |
| Build sync agent | `make build-sync` |
| Deploy edge-only | `make deploy-edge` |
| Deploy hybrid | `make deploy-hybrid` |
| Deploy hub | `make deploy-hub` |
| Build TRT engine | `make build-trt MODEL=yolov8n.onnx PRECISION=fp16` |

---

## Code Style Guidelines

- **Linter/Formatter**: ruff (no Black configured).
- **Config**: `pyproject.toml`:
  ```toml
  [tool.ruff]
  target-version = "py310"
  line-length = 120
  ```
- **Import style**: Standard library → third-party → internal packages.
- **Type hints**: Used where helpful but not enforced by a type checker.
- **Schemas**: All data schemas are stdlib `dataclasses`. No Pydantic dependency on edge.
- **String literals**: Project supports Chinese-primary bilingual UI. `JSON_AS_ASCII = False` is set in Flask.
- **Docstrings**: Mixed; not all modules have comprehensive docstrings.

---

## Testing Strategy

- **Runner**: pytest with `-v --tb=short` (configured in `pyproject.toml`).
- **Coverage**: pytest-cov is listed in dev extras but not run by default.
- **No shared fixtures**: There is **no `conftest.py`** in `tests/` or anywhere in the repo. Each test file defines its own inline fixtures and helper factories (`_make_detection()`, `_make_event()`, `_post_reading()`, etc.).
- **Storage isolation**: Tests that touch `packages.inference.storage.factory` frequently reset module-level singletons (`_event_store`, `_evidence_store`, `_triple_store = None`) to avoid cross-test pollution.
- **Env patching**: `monkeypatch` or direct `os.environ` manipulation is common for `STORAGE_MODE`, `SQLITE_DB_PATH`, `DB_PATH`, `EVIDENCE_DIR`, and `STATION_API_KEY`.
- **SQLite in tests**: Almost all integration tests spin up real SQLite databases in temp directories (`tmp_path` / `tempfile.TemporaryDirectory()`) rather than mocking storage.
- **No CI/CD**: There are no GitHub Actions workflows or pre-commit hooks.

### Known Pre-existing Test Failures

- `test_maintenance_api.py` and `test_taxonomy.py` have ~25 failures and 7 errors that are **pre-existing codebase issues**, not environment problems.
- 198+ tests pass.

---

## Deployment Architecture

Four deployment modes are supported:

| Mode | Description | Cloud Dependency |
|------|-------------|------------------|
| **Edge-Only** | Everything on a single Jetson/GPU box. SQLite + local filesystem. | None |
| **Hybrid** | Edge runs locally; a sidecar sync agent pushes to cloud API via HTTPS. | Optional (outbound only) |
| **Hub** | On-premises site data zone: Postgres + MinIO + Prometheus + Grafana. | None |
| **AWS ECS + Datadog** | Cloud-native container deployment with Datadog observability (ECS Explorer, APM, Logs). | Datadog |

### Datadog on AWS

See `deploy/aws/` for the complete Datadog AWS stack:

1. **AWS Integration** (`cloudformation-datadog-integration.yaml`) — IAM role allowing Datadog to pull CloudWatch metrics, events, and resource metadata from your AWS account.
2. **Datadog Forwarder** (`cloudformation-datadog-forwarder.yaml`) — Lambda function that forwards CloudWatch Logs, S3 events, and SNS messages to Datadog.
3. **ECS Agent** (`cloudformation-ecs-datadog.yaml`) — Datadog Agent deployed as an ECS daemon service with ECS Explorer enabled (`DD_ECS_TASK_COLLECTION_ENABLED`).

Local Docker Compose files also include an optional `datadog-agent` sidecar for dev parity.

Quick deploy:
```bash
cd deploy/aws
export DD_API_KEY="..."
export DD_EXTERNAL_ID="..."
./setup-datadog-aws.sh
```

### Environment Variables (Critical for Dev)

| Variable | Purpose | Dev Notes |
|----------|---------|-----------|
| `STORAGE_MODE` | `local` (SQLite) or `cloud` (stubs) | Always `local` for dev |
| `SQLITE_DB_PATH` | SQLite DB for event/triple store | Required |
| `DB_PATH` | SQLite DB for production metrics | **Must match `SQLITE_DB_PATH`** in dev |
| `EVIDENCE_DIR` | JPEG evidence storage path | Required |
| `STATION_ID` | Unique station identifier | e.g. `station_dev` |
| `STATION_API_KEY` / `EDGE_API_KEY` | API key for auth | Required for most `/api/*` endpoints |
| `CAMERA_PROTOCOL` | `file`, `usb`, `rtsp`, `pyspin`, `webcam` | Use `file` or stub for dev |
| `CAMERA_URI` | Device path or RTSP URL | |
| `VISION_ENGINE_PATH` | Path to `.engine` file | Not needed with stub providers |
| `ENABLE_LOCAL_LLM` | `true` / `false` | Set `false` for dev without GGUF models |
| `EVIDENCE_MAX_GB` | FIFO disk quota (default 50) | |
| `CLOUD_API_URL` | Cloud API base URL | Only for hybrid sync |
| `CLOUD_API_KEY` | Cloud API bearer token | Only for hybrid sync |
| `DD_API_KEY` | Datadog API key | Only when using Datadog Agent |
| `DD_SITE` | Datadog site (`datadoghq.com`, `datadoghq.eu`) | Defaults to `datadoghq.com` |

### No External Services Needed for Dev/Test

All storage is SQLite (stdlib). No MQTT broker, Docker, Postgres, MinIO, GPU, or camera hardware is required for development or testing. Use stub providers and `STORAGE_MODE=local`.

---

## Security Considerations

- **API key auth**: All `/api/*` routes (except `/api/maintenance/*`) require an API key via `X-Edge-Api-Key`, `X-API-Key`, or `Authorization: Bearer <token>`. The API **fails closed** — missing or invalid keys return 401.
- **NOTIFY-ONLY safety rail**: The system suggests actions but never commands PLCs/actuators without operator confirmation.
- **Single-tenant by design** at the station level.
- **Secrets in `.env`**: `.env` and `.env.*` are gitignored. Only `.env.example` files are committed.
- **Model files are gitignored**: `.engine`, `.gguf`, `.pt`, `.onnx`, `.bin` are excluded from version control.
- **No PyTorch at edge runtime**: Reduces attack surface on deployed devices.

---

## Key Conventions and Gotchas

1. **DB_PATH vs SQLITE_DB_PATH**: The API reads `DB_PATH` for production metrics and `SQLITE_DB_PATH` for the event store. Both must be set to the same path for local dev or metrics will write to separate databases.
2. **TensorRT engines are device-specific**: Never copy `.engine` files between different GPU architectures. Always build on the target Jetson using `scripts/build_trt_engine.sh` or `make build-trt`.
3. **`make dev` won't work without hardware**: It attempts full station startup including TensorRT model loading. Use the API-only standalone snippet above for cloud dev.
4. **Storage singleton resets in tests**: When writing tests that touch storage, reset `packages.inference.storage.factory._event_store`, `_evidence_store`, and `_triple_store` to `None` to avoid cross-test state.
5. **No `conftest.py`**: Every test file is self-contained. Do not create a shared `conftest.py` unless the team decides to adopt one.
6. **Codacy is local-only**: The `.codacy/` directory is gitignored and auto-installed by the Codacy CLI/MCP tooling. It is not part of committed CI.
7. **Chinese UI support**: Operator console and maintenance action templates include Chinese strings. `JSON_AS_ASCII = False` is set globally in Flask.
8. **`deploy.md` is stale**: That file describes an older FastAPI-based architecture and explicitly warns it does not apply to the current repo. Refer to `docs/JETSON_QUICKSTART.md`, `docs/LOCAL_MODE.md`, and `docs/ARCHITECTURE.md` for current deployment guidance.
9. **Model setup is multi-step**: `scripts/setup_models.sh` + `scripts/build_trt_engine.sh` + `scripts/download_qwen.sh` (or manual download) are required to populate `/opt/intelfactor/models/` on a real station.

---

## Hardware Tiers (Runtime)

`CapabilityResolver` auto-detects hardware via `nvidia-smi`, VRAM, and Jetson platform checks (`/etc/nv_tegra_release`). Override via `INTELFACTOR_DEVICE_CLASS`.

| Device | VRAM | Vision Model | Language Model |
|--------|------|--------------|----------------|
| Orin Nano Super | 8GB | YOLOv8n FP16 | Qwen-2.5-3B Q4_K_M (llama.cpp) |
| Orin NX 16GB | 16GB | YOLOv8m FP16 | Qwen-2.5-7B Q4_K_M |
| AGX Orin 64GB | 64GB | YOLOv8l FP16 | Qwen-2.5-14B Q8 |
| GPU Server | 24GB+ | YOLOv8x FP16 | Qwen-2.5-14B vLLM |

---

## Useful Documentation References

| File | Topic |
|------|-------|
| `docs/ARCHITECTURE.md` | System design, data flow, scaling |
| `docs/JETSON_QUICKSTART.md` | Jetson Orin Nano deployment guide |
| `docs/LOCAL_MODE.md` | Edge-only local deployment |
| `docs/INTEGRATION.md` | Cloud integration guide |
| `docs/EDGE_CLOUD_CONTRACT.md` | Edge-to-cloud sync API contract (v1.0, frozen) |
| `docs/BACKEND_IMPLEMENTATION_GUIDE.md` | Cloud backend implementation reference |
| `docs/JETSON_OPERATOR_CONSOLE_VALIDATION.md` | Hardware validation checklist for `/inspect` |
| `docs/DEPLOYMENT_GUIDE.md` | Bilingual CN/EN deployment instructions |
| `deploy/aws/README.md` | Datadog Agent on ECS setup guide |
| `README.md` | Human-facing quick start |
| `CLAUDE.md` | Full architecture reference (comprehensive) |
