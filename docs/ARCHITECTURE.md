# IntelFactor Technical Architecture

## Overview

IntelFactor is an edge-first industrial quality inspection platform that runs entirely on NVIDIA edge devices with optional cloud synchronization. The system performs real-time defect detection, root cause analysis, and continuous learning from operator feedback.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTELFACTOR ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │   Camera    │    │   Vision    │    │    RCA      │    │  Operator   │  │
│   │   Ingest    │───▶│  Pipeline   │───▶│  Pipeline   │───▶│  Dashboard  │  │
│   │  (RTSP/USB) │    │ (TensorRT)  │    │ (4-Layer)   │    │  (Flask)    │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      STORAGE LAYER                                   │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│   │  │   SQLite    │  │  Evidence   │  │   Causal    │  │  Metrics   │  │   │
│   │  │   Events    │  │  JPEG/JSON  │  │   Triples   │  │  Timeseries│  │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                        │
│            [EDGE-ONLY MODE]               [HYBRID MODE]                     │
│            Local dashboard                 Cloud sync                       │
│            No external deps                S3 + Cloud API                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Modes

### Edge-Only Mode
Everything runs on a single NVIDIA device. No cloud dependency.

```
┌──────────────────────────────────────────────────────────────┐
│                    NVIDIA EDGE DEVICE                         │
│                  (Jetson Orin / GPU Server)                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐   │
│  │ Camera  │──▶│ Ingest  │──▶│ Vision  │──▶│ RCA Pipeline│   │
│  │ RTSP/USB│   │ Thread  │   │TensorRT │   │  4-Layer    │   │
│  └─────────┘   └─────────┘   └─────────┘   └─────────────┘   │
│                                    │              │           │
│                                    ▼              ▼           │
│                            ┌─────────────────────────┐        │
│                            │    Evidence Writer      │        │
│                            │  /data/evidence/*.jpg   │        │
│                            └─────────────────────────┘        │
│                                    │                          │
│  ┌─────────────────────────────────┴──────────────────────┐  │
│  │                    SQLite Database                      │  │
│  │  /data/local.db (events, triples, alerts, heartbeats)  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                    │                          │
│                                    ▼                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                  Flask API Server                        │  │
│  │                  http://0.0.0.0:8080                     │  │
│  │  /api/events, /api/evidence, /api/triples, /api/health  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                    │                          │
│                                    ▼                          │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Operator Dashboard (index.html)             │  │
│  │              Chinese-primary, bilingual UI               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Operator Browser
                http://<edge-ip>:8080
```

### Hybrid Mode
Edge runs independently; cloud sync for cross-site analytics.

```
┌─────────────────────────────────────────┐
│          EDGE DEVICE (Station)          │
├─────────────────────────────────────────┤
│  Camera → Vision → RCA → Dashboard      │
│              │                          │
│              ▼                          │
│  ┌───────────────────────────────────┐  │
│  │  SQLite + Evidence (Local)        │  │
│  │  Operates independently           │  │
│  └───────────────────────────────────┘  │
│              │                          │
│              ▼                          │
│  ┌───────────────────────────────────┐  │
│  │     Cloud Sync Agent              │  │
│  │  - Batch POST to Cloud API        │  │
│  │  - S3 upload for evidence         │  │
│  │  - Watermark-based incremental    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
              │
              │ HTTPS (outbound only)
              ▼
┌─────────────────────────────────────────┐
│            CLOUD / HUB                  │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Cloud   │  │ Postgres│  │   S3    │  │
│  │ API     │  │ (Hub)   │  │Evidence │  │
│  └─────────┘  └─────────┘  └─────────┘  │
│       │            │            │       │
│       └────────────┼────────────┘       │
│                    ▼                    │
│  ┌───────────────────────────────────┐  │
│  │   HQ Dashboard (Multi-Site)       │  │
│  │   Cross-line analytics            │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

KEY RULE: Edge keeps running if cloud dies.
```

---

## Component Architecture

### 1. Camera Ingest (`ingest.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA INGEST                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Sources:           Thread Pool:        Output:        │
│  ┌─────────────┐         ┌──────────┐        ┌──────────┐   │
│  │ RTSP Stream │────────▶│ Capture  │───────▶│ Frame    │   │
│  │ USB Camera  │         │ Thread   │        │ Queue    │   │
│  │ CSI (Jetson)│         │          │        │ (Ring)   │   │
│  │ GigE Vision │         │ Watchdog │        │          │   │
│  └─────────────┘         │ Reconnect│        └──────────┘   │
│                          └──────────┘              │        │
│                                                    ▼        │
│  Stats:                                     To Vision       │
│  - FPS actual/target                        Pipeline        │
│  - Frames captured/dropped                                  │
│  - Reconnect count                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Vision Pipeline (`providers/`)

```
┌─────────────────────────────────────────────────────────────┐
│                    VISION PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Capability Resolver                     │    │
│  │  Auto-detect: GPU, VRAM, Jetson/Server, CUDA caps   │    │
│  │  Select: Optimal model + quantization per device    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│           ┌──────────────┼──────────────┐                   │
│           ▼              ▼              ▼                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ TensorRT    │ │ Triton      │ │ ONNX        │           │
│  │ (Jetson)    │ │ (Server)    │ │ (Fallback)  │           │
│  │             │ │             │ │             │           │
│  │ YOLOv8n-l   │ │ YOLOv8x     │ │ YOLOv8n     │           │
│  │ FP16/INT8   │ │ FP16        │ │ FP32        │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│           │              │              │                   │
│           └──────────────┼──────────────┘                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 Detection Result                     │    │
│  │  - event_id, timestamp, station_id                  │    │
│  │  - detections: [{defect_type, confidence, bbox}]    │    │
│  │  - verdict: PASS | FAIL | REVIEW                    │    │
│  │  - inference_ms, model_version                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. RCA Pipeline (`rca/`)

```
┌─────────────────────────────────────────────────────────────┐
│                  4-LAYER RCA PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: ACCUMULATOR                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - SQLite rolling window (30 days default)          │    │
│  │  - Defect rate tracking per (station, defect_type)  │    │
│  │  - Z-score anomaly detection (baseline comparison)  │    │
│  │  - Emits: AnomalyAlert when spike detected          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  Layer 2: CORRELATOR                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Process parameter ingestion (grinding_rpm, etc.) │    │
│  │  - Pearson correlation with defect rates            │    │
│  │  - Drift detection (target ± tolerance)             │    │
│  │  - Emits: ProcessCorrelation with drift_pct         │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  Layer 3: EXPLAINER                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Qwen SLM (3B-14B depending on hardware)          │    │
│  │  - Bilingual output (Chinese primary, English)      │    │
│  │  - Context: defect + parameter drift + history      │    │
│  │  - Emits: RCAExplanation (cause_zh, cause_en)       │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  Layer 4: RECOMMENDER                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - SOP section lookup (defect → procedure)          │    │
│  │  - Action recommendation (adjust parameter to X)    │    │
│  │  - Urgency assignment (normal, high, critical)      │    │
│  │  - Emits: CausalTriple for operator review          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  CAUSAL TRIPLE                       │    │
│  │  The fundamental unit of IntelFactor's data moat    │    │
│  │                                                      │    │
│  │  DEFECT ──────────▶ CAUSE ──────────▶ OUTCOME       │    │
│  │  (what failed)      (why it failed)   (what worked) │    │
│  │                                                      │    │
│  │  Verified by operator accept/reject feedback        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4. Storage Layer (`storage/`)

```
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE ABSTRACTION                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Storage Factory                        │    │
│  │                                                      │    │
│  │  STORAGE_MODE=local ──▶ SQLite + Filesystem         │    │
│  │  STORAGE_MODE=cloud ──▶ DynamoDB + S3 (future)      │    │
│  │                                                      │    │
│  │  get_event_store()    → EventStore interface        │    │
│  │  get_evidence_store() → EvidenceStore interface     │    │
│  │  get_triple_store()   → TripleStore interface       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  LOCAL MODE IMPLEMENTATION:                                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  SQLite Database (/data/local.db)                   │    │
│  │                                                      │    │
│  │  Tables:                                             │    │
│  │  ├── defect_events     (event_id, timestamp, ...)   │    │
│  │  ├── evidence_index    (event_id, image_path, ...)  │    │
│  │  ├── causal_triples    (triple_id, defect, cause)   │    │
│  │  ├── anomaly_alerts    (alert_id, z_score, ...)     │    │
│  │  └── device_heartbeats (station_id, status, ...)    │    │
│  │                                                      │    │
│  │  Features:                                           │    │
│  │  - WAL mode for concurrent reads                    │    │
│  │  - Auto-migration on startup                        │    │
│  │  - Thread-local connections                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Evidence Filesystem (/data/evidence/)              │    │
│  │                                                      │    │
│  │  Structure:                                          │    │
│  │  evidence/                                           │    │
│  │  ├── 2026-02-13/                                    │    │
│  │  │   ├── evt_20260213_143022_a1b2c3.jpg            │    │
│  │  │   ├── evt_20260213_143022_a1b2c3.json           │    │
│  │  │   └── manifest.jsonl                             │    │
│  │  └── 2026-02-14/                                    │    │
│  │      └── ...                                        │    │
│  │                                                      │    │
│  │  Features:                                           │    │
│  │  - Date-partitioned directories                     │    │
│  │  - FIFO deletion when quota exceeded                │    │
│  │  - Configurable max_disk_gb (default 50GB)          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## API Design

### REST Endpoints

```
┌─────────────────────────────────────────────────────────────┐
│                    API ENDPOINTS                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HEALTH                                                      │
│  GET  /health                   System health check          │
│                                                              │
│  EVENTS                                                      │
│  GET  /api/events              List events (paginated)       │
│  GET  /api/events/:id          Get single event              │
│  POST /api/events              Create event                  │
│                                                              │
│  EVIDENCE                                                    │
│  GET  /api/v1/evidence/:id           Metadata JSON           │
│  GET  /api/v1/evidence/:id/image.jpg Full image              │
│  GET  /api/v1/evidence/:id/thumb.jpg Thumbnail               │
│  GET  /api/v1/evidence/manifest      Daily manifest          │
│  GET  /api/evidence/stats            Disk usage stats        │
│                                                              │
│  TRIPLES                                                     │
│  GET   /api/triples            List causal triples           │
│  GET   /api/triples/:id        Get single triple             │
│  PATCH /api/triples/:id        Update triple (feedback)      │
│  GET   /api/triples/stats      Triple statistics             │
│                                                              │
│  RCA                                                         │
│  GET  /api/alerts              Active anomaly alerts         │
│  GET  /api/recommendations     Pending recommendations       │
│  POST /api/feedback            Record operator feedback      │
│  GET  /api/drift               Parameter drift status        │
│  POST /api/reading             Record manual reading         │
│                                                              │
│  PIPELINE                                                    │
│  GET  /api/status              Full station status           │
│  GET  /api/pipeline/stats      Pipeline statistics           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Dashboard ──▶ GET /api/events?verdict=FAIL&limit=25        │
│                        │                                     │
│                        ▼                                     │
│              ┌─────────────────┐                            │
│              │   Flask Route   │                            │
│              │  routes/events  │                            │
│              └────────┬────────┘                            │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                            │
│              │ Storage Factory │                            │
│              │ get_event_store │                            │
│              └────────┬────────┘                            │
│                       │                                      │
│          ┌────────────┴────────────┐                        │
│          ▼                         ▼                        │
│  [STORAGE_MODE=local]     [STORAGE_MODE=cloud]              │
│  ┌─────────────────┐      ┌─────────────────┐              │
│  │ SQLiteEventStore│      │ DynamoEventStore│              │
│  │  SELECT * FROM  │      │  query(...)     │              │
│  │  defect_events  │      │                 │              │
│  └─────────────────┘      └─────────────────┘              │
│          │                         │                        │
│          └────────────┬────────────┘                        │
│                       ▼                                      │
│              ┌─────────────────┐                            │
│              │  JSON Response  │                            │
│              │ {events: [...]} │                            │
│              └─────────────────┘                            │
│                       │                                      │
│                       ▼                                      │
│              Dashboard renders event list                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE SUPPORT MATRIX                              │
├──────────────────┬───────┬──────────┬─────────────────┬────────────────────┤
│ Device           │ VRAM  │ Role     │ Vision Model    │ Language Model     │
├──────────────────┼───────┼──────────┼─────────────────┼────────────────────┤
│ Orin Nano Super  │ 8GB   │ Station  │ YOLOv8n FP16    │ Qwen-2.5-3B Q4_K_M │
│ Orin NX 16GB     │ 16GB  │ Station  │ YOLOv8m FP16    │ Qwen-2.5-7B Q4_K_M │
│ AGX Orin 64GB    │ 64GB  │ Hub      │ YOLOv8l FP16    │ Qwen-2.5-14B Q8    │
│ RTX 4090         │ 24GB  │ Server   │ YOLOv8x FP16    │ Qwen-2.5-14B vLLM  │
│ L4/A10           │ 24GB  │ Cloud    │ YOLOv8x FP16    │ Qwen-2.5-14B vLLM  │
└──────────────────┴───────┴──────────┴─────────────────┴────────────────────┘

Capability Resolver automatically selects optimal configuration based on:
- nvidia-smi GPU detection
- Available VRAM
- Jetson platform detection (tegra check)
- CUDA compute capability
```

---

## Network Topology

### Single Station (Edge-Only)

```
┌──────────────────────────────────────────┐
│              Factory LAN                  │
│                                           │
│  ┌──────────┐      ┌──────────────────┐  │
│  │ Camera   │      │  NVIDIA Edge     │  │
│  │ (RTSP)   │─────▶│  Device          │  │
│  │ :554     │      │                  │  │
│  └──────────┘      │  ┌────────────┐  │  │
│                    │  │ Station    │  │  │
│                    │  │ Container  │  │  │
│  ┌──────────┐      │  │ :8080      │  │  │
│  │ Operator │◀────▶│  └────────────┘  │  │
│  │ Browser  │      │                  │  │
│  └──────────┘      └──────────────────┘  │
│                                           │
└──────────────────────────────────────────┘

Ports:
- 554:  Camera RTSP (inbound to edge)
- 8080: Dashboard + API (LAN only)
```

### Multi-Station with Hub

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FACTORY SITE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Production Line 1              Production Line 2         Production Line 3 │
│  ┌─────────────────┐           ┌─────────────────┐       ┌─────────────────┐│
│  │ Station 1       │           │ Station 2       │       │ Station 3       ││
│  │ Orin NX 16GB    │           │ Orin NX 16GB    │       │ Orin Nano 8GB   ││
│  │                 │           │                 │       │                 ││
│  │ Camera → Infer  │           │ Camera → Infer  │       │ Camera → Infer  ││
│  │ SQLite + Evid   │           │ SQLite + Evid   │       │ SQLite + Evid   ││
│  │ Local Dashboard │           │ Local Dashboard │       │ Local Dashboard ││
│  └────────┬────────┘           └────────┬────────┘       └────────┬────────┘│
│           │                             │                         │         │
│           └─────────────────────────────┼─────────────────────────┘         │
│                                         │                                    │
│                                         ▼                                    │
│                          ┌─────────────────────────────┐                    │
│                          │        SITE HUB             │                    │
│                          │   (AGX Orin / GPU Server)   │                    │
│                          │                             │                    │
│                          │  ┌───────┐ ┌───────┐       │                    │
│                          │  │Postgres│ │ MinIO │       │                    │
│                          │  │ :5432  │ │ :9000 │       │                    │
│                          │  └───────┘ └───────┘       │                    │
│                          │  ┌───────┐ ┌───────┐       │                    │
│                          │  │Grafana│ │Prometh│       │                    │
│                          │  │ :3000 │ │ :9090 │       │                    │
│                          │  └───────┘ └───────┘       │                    │
│                          │                             │                    │
│                          │  Cross-line analytics       │                    │
│                          │  Pattern correlation        │                    │
│                          └─────────────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Data Flow:
- Each station operates independently (local SQLite)
- Sync service copies SQLite → Postgres (batch, configurable interval)
- Hub provides cross-line dashboards via Grafana
```

---

## Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  NETWORK SECURITY                                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Edge devices on isolated factory VLAN            │    │
│  │  - No inbound internet connectivity required        │    │
│  │  - Outbound HTTPS only for hybrid mode sync         │    │
│  │  - Camera RTSP streams on dedicated network         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  API SECURITY                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - API key header for all requests (X-API-Key)      │    │
│  │  - CORS restricted to known origins                 │    │
│  │  - Rate limiting on sensitive endpoints             │    │
│  │  - No PII in evidence images (anonymized)           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  DATA SECURITY                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - SQLite database on encrypted volume (optional)   │    │
│  │  - Evidence FIFO deletion (configurable retention)  │    │
│  │  - No cloud storage required in edge-only mode      │    │
│  │  - Audit log for operator actions                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  CLOUD SYNC SECURITY (Hybrid Mode Only)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Bearer token authentication                      │    │
│  │  - TLS 1.3 for all cloud API calls                 │    │
│  │  - S3 server-side encryption (AES-256)             │    │
│  │  - IAM roles with least privilege                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Scaling Considerations

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────┐
│                    SCALING PATTERNS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STATION SCALING (Add more lines)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Each production line gets its own station device   │    │
│  │  - Independent operation                            │    │
│  │  - Local storage                                    │    │
│  │  - No cross-line dependency                         │    │
│  │  - Hub aggregates for analytics only                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  HUB SCALING (Multi-site)                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Site 1 Hub ─┐                                      │    │
│  │  Site 2 Hub ─┼──▶ Regional Cloud API                │    │
│  │  Site 3 Hub ─┘    (Optional)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  THROUGHPUT TARGETS                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Per Station:                                       │    │
│  │  - Vision: 30 FPS @ 1080p (TensorRT)               │    │
│  │  - RCA: <100ms per explanation (Qwen 3B)           │    │
│  │  - API: 100 req/s (Flask + gunicorn)               │    │
│  │  - Evidence: 1000 images/hour write capacity       │    │
│  │                                                      │    │
│  │  Per Hub:                                            │    │
│  │  - Postgres: 10M events (90-day retention)         │    │
│  │  - Grafana: 10 concurrent dashboard users          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Artifacts

```
deploy/
├── edge-only/
│   ├── docker-compose.yml    # Single-container station
│   ├── Dockerfile            # NVIDIA L4T base
│   └── .env.example          # Configuration template
│
├── hybrid/
│   ├── docker-compose.yml    # Station + sync-agent
│   ├── Dockerfile.sync       # Cloud sync container
│   └── .env.example          # Cloud config template
│
├── hub/
│   ├── docker-compose.yml    # Postgres + MinIO + Grafana
│   ├── init-db.sql           # Schema migrations
│   └── grafana/dashboards/   # Pre-built dashboards
│
└── systemd/
    └── intelfactor-station.service
```

---

## Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY STACK                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  METRICS (Prometheus)                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  intelfactor_frames_processed_total                 │    │
│  │  intelfactor_defects_detected_total{type}           │    │
│  │  intelfactor_inference_latency_ms                   │    │
│  │  intelfactor_rca_explanation_latency_ms             │    │
│  │  intelfactor_evidence_disk_bytes                    │    │
│  │  intelfactor_triple_acceptance_rate                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  LOGS (Structured JSON)                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  {"level":"INFO","component":"vision",              │    │
│  │   "event_id":"evt_123","inference_ms":45}           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  HEALTH CHECKS                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  GET /health → {status, storage_mode, runtime}      │    │
│  │  scripts/doctor.py → Pre-flight diagnostics         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  DASHBOARDS (Grafana)                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  - Defect Rate by Station                           │    │
│  │  - Cross-Line Drift Correlation                     │    │
│  │  - Triple Acceptance Rate                           │    │
│  │  - System Health (GPU, disk, latency)               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

| Aspect | Edge-Only | Hybrid |
|--------|-----------|--------|
| **Cloud dependency** | None | Optional sync |
| **Storage** | SQLite + filesystem | SQLite + S3/Postgres |
| **Dashboard** | Local (port 8080) | Local + Cloud |
| **Offline operation** | Full capability | Full capability |
| **Cross-site analytics** | No | Yes (via hub) |
| **Deployment** | 1 container | 2 containers |

**Key Design Principles:**
1. **Edge-first**: Core inspection runs locally, cloud is optional
2. **Offline-capable**: Factory keeps running if network dies
3. **Hardware-adaptive**: Auto-selects optimal models per device
4. **Operator-centric**: Chinese-primary UI, simple accept/reject workflow
5. **Closed-loop**: Operator feedback improves system over time
