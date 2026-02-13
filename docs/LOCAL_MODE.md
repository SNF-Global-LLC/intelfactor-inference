# IntelFactor.ai — Local Mode Guide

Run the full IntelFactor stack on a single NVIDIA edge device (Jetson or GPU server) with no cloud dependencies.

## Quick Start

### 1. Prerequisites

- NVIDIA GPU with CUDA support (Jetson Orin, RTX, or datacenter GPU)
- Docker with NVIDIA Container Toolkit
- Camera (USB, CSI, or RTSP)

### 2. Deploy Edge-Only Stack

```bash
cd deploy/edge-only

# Copy and edit environment
cp .env.example .env
nano .env  # Set STATION_ID, CAMERA_URI, MODELS_DIR

# Start services
docker compose up -d

# Check logs
docker compose logs -f station
```

### 3. Access Dashboard

Open `http://<edge-ip>:8080` in your browser.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NVIDIA Edge Device                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Camera     │───▶│  Inference   │───▶│  Evidence    │       │
│  │   Capture    │    │  Pipeline    │    │  Writer      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                              │                   │              │
│                              ▼                   ▼              │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   SQLite     │    │  /data/      │       │
│                      │   local.db   │    │  evidence/   │       │
│                      └──────────────┘    └──────────────┘       │
│                              │                   │              │
│                              ▼                   │              │
│                      ┌──────────────────────────────────┐       │
│                      │         REST API (Flask)         │       │
│                      │         http://localhost:8080    │       │
│                      └──────────────────────────────────┘       │
│                                      │                          │
└──────────────────────────────────────│──────────────────────────┘
                                       │
                                       ▼
                              ┌──────────────┐
                              │  Dashboard   │
                              │  (Browser)   │
                              └──────────────┘
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_MODE` | `local` | Storage backend (`local` or `cloud`) |
| `SQLITE_DB_PATH` | `/opt/intelfactor/data/local.db` | SQLite database path |
| `EVIDENCE_DIR` | `/opt/intelfactor/data/evidence` | Evidence storage directory |
| `STATION_ID` | `station_01` | Unique station identifier |
| `CAMERA_URI` | `/dev/video0` | Camera source (device or RTSP URL) |
| `API_PORT` | `8080` | API server port |
| `EVIDENCE_MAX_GB` | `50` | Max evidence storage (FIFO deletion) |

### Storage Layout

```
/opt/intelfactor/data/
├── local.db              # SQLite database (events, triples, alerts)
└── evidence/
    ├── 2026-02-13/
    │   ├── evt_20260213_143022_a1b2c3.jpg
    │   ├── evt_20260213_143022_a1b2c3.json
    │   └── ...
    └── 2026-02-14/
        └── ...
```

## API Endpoints

### Events

```bash
# List recent events
curl http://localhost:8080/api/events?limit=10

# Get single event
curl http://localhost:8080/api/events/evt_20260213_143022_a1b2c3

# Filter by verdict
curl http://localhost:8080/api/events?verdict=FAIL
```

### Evidence

```bash
# Get evidence metadata
curl http://localhost:8080/api/v1/evidence/evt_20260213_143022_a1b2c3

# Get evidence image
curl http://localhost:8080/api/v1/evidence/evt_20260213_143022_a1b2c3/image.jpg -o image.jpg

# List evidence by date
curl http://localhost:8080/api/v1/evidence/manifest?date=2026-02-13
```

### Triples (Causal Analysis)

```bash
# List pending triples
curl http://localhost:8080/api/triples?status=pending

# Record operator feedback
curl -X PATCH http://localhost:8080/api/triples/abc123 \
  -H "Content-Type: application/json" \
  -d '{"operator_action": "accepted", "operator_id": "op_001"}'
```

### Health

```bash
curl http://localhost:8080/health
```

## System Doctor

Run the doctor script to verify system health:

```bash
# Basic checks
python scripts/doctor.py

# Full checks including camera
python scripts/doctor.py --full

# Test specific camera
python scripts/doctor.py --camera rtsp://192.168.1.100:554/stream
```

## Upgrading to Hybrid Mode

To add cloud sync while keeping local-first operation:

1. Switch to hybrid deployment:
   ```bash
   cd deploy/hybrid
   cp .env.example .env
   nano .env  # Add CLOUD_API_URL, CLOUD_API_KEY
   docker compose up -d
   ```

2. The edge box continues running independently. If cloud dies, local operation is unaffected.

## Troubleshooting

### No GPU detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Camera not accessible

```bash
# List video devices
ls -la /dev/video*

# Test with ffmpeg
ffmpeg -f v4l2 -i /dev/video0 -frames 1 test.jpg
```

### Database issues

```bash
# Check database
sqlite3 /opt/intelfactor/data/local.db ".tables"
sqlite3 /opt/intelfactor/data/local.db "SELECT COUNT(*) FROM defect_events"
```
