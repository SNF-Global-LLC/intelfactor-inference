# IntelFactor Inference Engine — Development Guide

Edge-first AI platform for manufacturing quality inspection and predictive maintenance.
Runs autonomously on NVIDIA Jetson. Cloud is additive, never gating.

---

## Architecture

```
Camera (RTSP/USB)
      │
      ▼
CameraIngest.on_frame()
      │
      ▼
StationRuntime.process_frame()
      ├── VisionProvider.detect(frame)        ← TensorRT YOLO → DetectionResult
      ├── EvidenceWriter.write(frame)          ← JPEG + JSON sidecar + manifest.jsonl
      └── RCAPipeline.ingest(result)           ← DefectAccumulator (FAIL/REVIEW only)

[Background: every 5 min]
      ▼
RCAPipeline.run_rca()
      ├── Layer 1: DefectAccumulator.check_anomalies()   ← z-score vs 30-day baseline
      ├── Layer 2: ProcessCorrelator.correlate(alert)    ← Pearson r on process params
      ├── Layer 3: RCAExplainer.explain(...)             ← SLM prompt → bilingual JSON
      └── Layer 4: ActionRecommender.recommend()+triple  ← SOP ref + CausalTriple

[MQTT sensor stream, continuous]
      ▼
SensorService.ingest_reading()
      ├── _extract_scalar()                   ← normalise vibration/current/acoustic
      ├── _score() vs BaselineProfile         ← z-score → HealthVerdict
      └── SQLite sensor_events + baselines

[Background: every hour]
      ▼
SensorService.recompute_baselines()   ← excludes live window to protect baseline

[API request]
POST /api/maintenance/feedback  →  MaintenanceAction updated
POST /api/feedback              →  CausalTriple PENDING → VERIFIED/DISPUTED
```

**Design principles:**
- Edge-authoritative: Jetson decides locally. Cloud is escalation only.
- NOTIFY-ONLY safety rail: system suggests, never executes.
- Offline-first: SQLite WAL buffers everything. Kafka/cloud replay on reconnect.
- Evidence-driven: every verdict is traceable to image + detection JSON + audit log.
- Confidence routing: >0.85 auto-pass/fail, 0.5–0.85 human verify, <0.5 manual.

---

## Project Structure

```
intelfactor-inference/
├── packages/
│   ├── inference/              # Core inference, API, storage, RCA
│   │   ├── api_v2.py           # Flask REST server (30 endpoints)
│   │   ├── cli.py              # Entry points: run_station, run_hub, run_doctor
│   │   ├── doctor.py           # Pre-flight diagnostics (CheckResult, DoctorReport)
│   │   ├── evidence.py         # JPEG ring buffer, manifest.jsonl
│   │   ├── ingest.py           # Camera ingest (RTSP/USB via OpenCV)
│   │   ├── schemas.py          # Shared dataclasses (no Pydantic)
│   │   ├── sync_cloud.py       # Hybrid-mode cloud sync agent
│   │   ├── modes/
│   │   │   └── runtime.py      # StationRuntime + SiteHubRuntime
│   │   ├── providers/
│   │   │   ├── base.py         # Abstract VisionProvider, LanguageProvider
│   │   │   ├── resolver.py     # CapabilityResolver — hardware auto-detect
│   │   │   ├── vision_trt.py   # TensorRT YOLO provider
│   │   │   ├── language_llama.py   # llama.cpp backend (Jetson)
│   │   │   └── language_vllm.py    # vLLM backend (GPU server)
│   │   ├── rca/
│   │   │   ├── pipeline.py     # RCAPipeline orchestrator
│   │   │   ├── accumulator.py  # Layer 1: DefectAccumulator
│   │   │   ├── correlator.py   # Layer 2: ProcessCorrelator
│   │   │   ├── explainer.py    # Layer 3: RCAExplainer (SLM)
│   │   │   └── recommender.py  # Layer 4: ActionRecommender
│   │   └── storage/
│   │       ├── base.py         # Abstract EventStore, EvidenceStore, TripleStore
│   │       ├── factory.py      # Singleton factory (STORAGE_MODE env var)
│   │       └── sqlite_*.py     # SQLite implementations
│   ├── ingestion/              # Machine Health Copilot — sensor pipeline
│   │   ├── schemas.py          # SensorReading, SensorEvent, BaselineProfile,
│   │   │                       # MaintenanceVerdict, MaintenanceAction
│   │   └── sensor_service.py   # MQTT subscriber, SQLite WAL, z-score scoring
│   └── policy/
│       └── maintenance_iq.py   # Stateless rules engine — HEALTHY/WARNING/CRITICAL
├── configs/
│   ├── station.yaml            # Reference station config (copy + customise)
│   └── wiko_taxonomy.yaml      # 13 defect classes, bilingual, SOP sections
├── deploy/
│   ├── edge-only/              # Single Jetson, no cloud
│   ├── hybrid/                 # Station + cloud sync sidecar
│   ├── hub/                    # Multi-station: Postgres + MinIO + Grafana
│   ├── station/                # Bare-metal scripts
│   └── systemd/                # systemd unit file (non-Docker)
├── scripts/
│   ├── build_trt_engine.sh     # Compile TRT engine on target hardware (CRITICAL)
│   ├── verify_trt_engine.py    # Load-test a .engine file
│   ├── smoke_test.py           # End-to-end pipeline validation
│   ├── doctor.py               # System health checks (called by make doctor)
│   └── setup_models.sh         # Download Qwen-2.5 GGUF
└── tests/                      # 163+ passing tests
```

---

## RCA Pipeline (4 layers, all on Jetson)

| Layer | Class | What it does |
|-------|-------|-------------|
| 1 | `DefectAccumulator` | SQLite WAL, 30-day rolling window, z-score on 4-hour buckets. Records FAIL+REVIEW only. |
| 2 | `ProcessCorrelator` | Loads `process_parameters` from `edge.yaml`. Pearson r on 30-min windows. |
| 3 | `RCAExplainer` | Wraps any `LanguageProvider`. Structured prompt → bilingual ZH/EN JSON. Statistical fallback if SLM fails. |
| 4 | `ActionRecommender` | NOTIFY-ONLY. SOP-mapped actions. Stores `CausalTriple` (PENDING → VERIFIED/DISPUTED on feedback). |

**`CausalTriple`** is the data flywheel: `defect_event_id + cause_parameter + operator_action`. Rejection reasons are stored as training signals.

---

## Sensor Pipeline (Machine Health Copilot)

| Class | Location | What it does |
|-------|----------|-------------|
| `SensorService` | `packages/ingestion/sensor_service.py` | MQTT subscriber for vibration/current/acoustic. SQLite WAL. Rolling 4-hour baseline (excludes live window). Z-score scoring on ingest. |
| `MaintenanceIQ` | `packages/policy/maintenance_iq.py` | Stateless rules engine. Takes list of `SensorEvent`, returns `MaintenanceVerdict`. Per-machine threshold overrides via `edge.yaml`. |

Thresholds: `HEALTHY` (z < 2.0), `WARNING` (z ≥ 2.0), `CRITICAL` (z ≥ 3.5).
Multi-sensor: worst verdict wins. Tie-break on highest z_score (conservative).

paho-mqtt is optional — `SensorService` works without a broker (`ingest_reading()` directly).

---

## Hardware Tiers

| DeviceClass | RAM | Jetson? | Vision model | Language backend |
|-------------|-----|---------|--------------|-----------------|
| `orin_nano` | 8GB | ✓ | yolov8n TRT FP16 | Qwen2.5-3B INT4 (llama.cpp) |
| `orin_nx` | 16GB | ✓ | yolov8s TRT FP16 | Qwen2.5-3B INT4 (llama.cpp) |
| `agx_orin` | 64GB | ✓ | YOLO26 TRT INT8 | Qwen2.5-7B INT4 (llama.cpp) |
| `thor_t4000` | 64GB | ✓ | YOLO26 TRT INT8 | Qwen2.5-7B FP16 (vLLM) |
| `thor_t5000` | 128GB | ✓ | YOLO26 TRT INT8 | Qwen2.5-20B INT4 (vLLM) |
| `gpu_server` | varies | ✗ | YOLO26 TRT INT8 | Qwen2.5-20B INT4 (vLLM) |

`CapabilityResolver` auto-detects via `/etc/nv_tegra_release` + `nvidia-smi`.
Override: `INTELFACTOR_DEVICE_CLASS=orin_nano`.

---

## TensorRT Engine Build

**TRT engines are device-specific — build ON the target Jetson.**

```bash
# FP16 (recommended for Orin Nano — best accuracy/speed tradeoff)
make build-trt MODEL=yolov8n.pt PRECISION=fp16

# INT8 (fastest, needs calibration images)
make build-trt-int8 MODEL=yolov8n.pt CALIB_DIR=./calibration_images/

# Verify an existing engine
make verify-trt ENGINE=/opt/intelfactor/models/vision/yolov8n_fp16.engine

# Direct script (more options)
./scripts/build_trt_engine.sh yolov8n.onnx fp16 --workspace 2048
./scripts/build_trt_engine.sh yolov8n.onnx int8 --calib ./calib/ --skip-verify
```

**Workflow for a new Jetson deployment:**
1. Export ONNX on x86 dev machine: `yolo export model=yolov8n.pt format=onnx`
2. Copy `.onnx` to Jetson via Tailscale/SCP
3. On Jetson: `make build-trt MODEL=yolov8n.onnx PRECISION=fp16`
4. Build takes 5–20 min. Engine + manifest written to `/opt/intelfactor/models/vision/`
5. `make doctor` to verify

The build script produces a `_manifest.json` alongside the engine:
```json
{
  "model_name": "yolov8n",
  "precision": "fp16",
  "device_model": "NVIDIA Jetson Orin Nano",
  "engine_sha256": "...",
  "onnx_sha256": "...",
  "build_date": "2026-02-14T08:00:00Z",
  "input_shape": [1, 3, 640, 640]
}
```

---

## API Endpoints

All served by Flask on `http://0.0.0.0:8080`.

### Core Quality Inspection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + storage mode |
| `/api/status` | GET | Full station status |
| `/api/events` | GET/POST | List/create defect events |
| `/api/events/<id>` | GET | Single event |
| `/api/triples` | GET | List causal triples |
| `/api/triples/<id>` | GET/PATCH | Get/update triple |
| `/api/feedback` | POST | Operator accept/reject recommendation |
| `/api/alerts` | GET | Unacknowledged anomaly alerts |
| `/api/recommendations` | GET | Pending triples awaiting operator |
| `/api/drift` | GET | Current process parameter drift |
| `/api/reading` | POST | Manual process parameter reading |
| `/api/pipeline/stats` | GET | Full RCA pipeline statistics |
| `/api/triples/stats` | GET | Triple collection stats |

### Evidence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/evidence/<id>` | GET | Evidence metadata |
| `/api/v1/evidence/<id>/image.jpg` | GET | Evidence JPEG |
| `/api/v1/evidence/<id>/thumb.jpg` | GET | Evidence thumbnail |
| `/api/v1/evidence/manifest` | GET | Date-partitioned manifest |
| `/api/evidence/stats` | GET | Disk usage stats |

### Machine Health Copilot (`/api/maintenance/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/maintenance/status` | GET | Sensor service + IQ health |
| `/api/maintenance/events` | GET | Sensor events (filterable) |
| `/api/maintenance/verdicts` | GET | Latest machine verdicts |
| `/api/maintenance/baselines` | GET | Active baseline profiles |
| `/api/maintenance/alerts` | GET | WARNING/CRITICAL events |
| `/api/maintenance/actions` | GET | Pending maintenance actions |
| `/api/maintenance/actions/<id>` | GET | Single action |
| `/api/maintenance/feedback` | POST | Operator accept/reject action |
| `/api/maintenance/reading` | POST | Manual sensor reading |
| `/api/maintenance/recompute-baselines` | POST | Force baseline recompute |

---

## Storage

| Store | DB file | Tables | Purpose |
|-------|---------|--------|---------|
| `EventStore` | `local.db` | `defect_events`, `anomaly_alerts` | DefectAccumulator events |
| `EvidenceStore` | `local.db` | `evidence_metadata` | JPEG sidecar records |
| `TripleStore` | `triples.db` | `causal_triples` | RCA feedback loop |
| `SensorService` | `sensors.db` | `sensor_events`, `baseline_profiles` | Machine health data |

All SQLite, WAL mode, `PRAGMA synchronous=NORMAL`. Thread-safe for concurrent reads.

Cloud backends (`DynamoDBEventStore`, etc.) are `NotImplementedError` stubs — designed for future implementation. Controlled by `STORAGE_MODE` env var.

---

## Configuration

### station.yaml (key fields)

```yaml
station_id: station_1
mode: station_only          # station_only | station_plus_hub
data_dir: /opt/intelfactor/data
model_dir: /opt/intelfactor/models

rca:
  z_score_threshold: 2.5
  anomaly_check_interval_sec: 300
  window_hours: 4.0

process_parameters:
  grinding_rpm:
    unit: RPM
    target: 3000
    tolerance: 50
    data_source: mqtt
    mqtt_topic: station_1/grinding/rpm

machine_health:
  enabled: true
  thresholds:
    warning: 2.0
    critical: 3.5
  sop_map:
    vibration: "SOP M.3.1"
    current: "SOP M.2.4"
    acoustic: "SOP M.4.2"
```

### edge.yaml (per-deployment, from SOP compiler)

Contains: `process_parameters`, `sop_map`, `defect_taxonomy`, `machine_health` block.
Read by `StationRuntime._load_edge_yaml()` at startup.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_MODE` | `local` | `local` or `cloud` |
| `SQLITE_DB_PATH` | `/data/local.db` | Main SQLite DB |
| `EVIDENCE_DIR` | `/data/evidence` | JPEG storage root |
| `STATION_ID` | `station_01` | Station identifier |
| `CAMERA_URI` | `/dev/video0` | RTSP URL or USB device |
| `VISION_MODEL_PATH` | `/models/vision` | Directory with `.engine` files |
| `LLM_MODEL_PATH` | `/models/llm` | Directory with `.gguf` files |
| `API_HOST` | `0.0.0.0` | Flask bind address |
| `API_PORT` | `8080` | Flask port |
| `MQTT_HOST` | `localhost` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `EVIDENCE_MAX_GB` | `50` | FIFO disk quota for JPEG evidence |
| `INTELFACTOR_DEVICE_CLASS` | _(auto)_ | Override hardware detection |

---

## Development

```bash
# Run without Docker (API + RCA only, no camera, stub models)
make dev

# Full test suite
make test                          # 163+ tests
make test-sensors                  # sensor + maintenance IQ tests only
make test-api                      # Flask API tests (requires flask installed)

# Linting
make lint                          # ruff check packages/ tests/

# Pre-flight diagnostics
make doctor                        # checks GPU, models, disk, camera

# TRT engine build (on Jetson)
make build-trt MODEL=yolov8n.pt PRECISION=fp16
make verify-trt ENGINE=/opt/intelfactor/models/vision/yolov8n_fp16.engine

# Smoke test (end-to-end on hardware)
python3 scripts/smoke_test.py
```

---

## Deployment

### Edge-only (single Jetson, no cloud)

```bash
cd deploy/edge-only
cp .env.example .env && vim .env
docker compose up -d
curl http://localhost:8080/health
```

### Hybrid (station + cloud sync sidecar)

```bash
cd deploy/hybrid
cp .env.example .env && vim .env   # set CLOUD_API_URL, CLOUD_API_KEY, S3_BUCKET
docker compose up -d
docker logs -f intelfactor-sync-agent
```

### Hub (multi-station, Postgres + Grafana)

```bash
cd deploy/hub
docker compose up -d
# Grafana: http://localhost:3000 (admin/admin)
# Station dashboards: Defect Rate, Cross-line Drift, Triple Acceptance, System Health
```

### Systemd (bare-metal, non-Docker)

```bash
sudo cp deploy/systemd/intelfactor-station.service /etc/systemd/system/
sudo systemctl enable --now intelfactor-station
journalctl -u intelfactor-station -f
```

---

## Code Style

- **Python 3.10+** — type hints required everywhere
- **Black** — 100-char line limit (configured in pyproject.toml as 120; use judgment)
- **Ruff** — rules E, F, I, W (`make lint`)
- **snake_case** functions and variables
- **PascalCase** classes
- **No Pydantic** — all schemas are stdlib `dataclasses`
- **No PyTorch on Jetson** — TensorRT only
- **httpx over requests** — for any HTTP client calls
- Imports: `from __future__ import annotations` at top of every module

---

## Key Constraints

| Constraint | Reason |
|-----------|--------|
| No PyTorch on Jetson | TensorRT for inference, llama.cpp for SLM. PyTorch too large for Orin Nano. |
| Cloud for escalation only | Edge-authoritative. Latency + airgap requirements. |
| NOTIFY-ONLY safety rail | Never send commands to PLCs/actuators without operator confirmation. |
| All schemas are stdlib dataclasses | No Pydantic dependency on edge device. |
| Single-tenant by design | One station, one operator console. Multi-tenant is `intelbase` (cloud layer). |
| paho-mqtt optional | `SensorService` works in broker-less test/dev environments. |
| TRT engines are device-specific | Must build on target hardware. Never copy between different GPU SKUs. |

---

## Known Gaps

| Gap | Impact | Workaround |
|-----|--------|-----------|
| No OTA model update mechanism | Manual `scp` + restart required for model updates | Planned: manifest.json pull + SHA256 verify |
| Camera ingest not tested on real hardware | USB + RTSP paths are code-complete but unvalidated on live Jetson + camera | smoke_test.py covers stub mode |
| Cloud storage backends are stubs | `STORAGE_MODE=cloud` raises `NotImplementedError` | Use `local` mode; sync via hybrid profile |
| Sync agent heartbeat endpoint referenced but not wired | `/api/sync/heartbeat` in API but `sync_cloud.py` doesn't call it | Non-blocking for MVP |
| Hub Dockerfile previously stale | `deploy/hub/Dockerfile.sync` was missing deps | Fixed: now mirrors `deploy/hybrid/Dockerfile.sync` |
| Wiko taxonomy in vision_trt.py | `_estimate_severity` has hardcoded cutlery defect types | Move to `configs/` YAML lookup |
