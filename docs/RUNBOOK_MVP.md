# California MVP: Station → Cloud → app.intelfactor.ai

Quick deployment guide for edge-first inference with cloud sync.

## Prerequisites

- NVIDIA Jetson (Orin Nano/NX/AGX) or GPU server
- Docker + NVIDIA Container Toolkit
- API key from https://app.intelfactor.ai

## Phase 1: Edge-Only Mode (No Cloud)

### 1.1 Clone and Setup

```bash
git clone https://github.com/tonesgainz/intelfactor-inference.git
cd intelfactor-inference

# Setup models first
./scripts/setup_models.sh --from-onnx yolov8n.onnx
# Or download pre-trained:
./scripts/setup_models.sh --quick
```

### 1.2 Configure

```bash
cd deploy/edge-only
cp .env.example .env

# Edit .env:
STATION_ID=factory_01_line_03
CAMERA_URI=rtsp://192.168.1.100:554/stream
# Or for USB camera: /dev/video0
```

### 1.3 Deploy

```bash
docker compose up -d

# Wait for health check
docker compose logs -f station
```

### 1.4 Verify

```bash
# Health check
curl http://localhost:8080/health
# Expected: {"status":"ok","station":{"storage_mode":"local",...}}

# List events
curl http://localhost:8080/api/events?limit=10

# Check evidence stats
curl http://localhost:8080/api/evidence/stats
```

### 1.5 Verify Evidence Capture

Trigger a FAIL/REVIEW detection, then:

```bash
# List evidence files
docker exec intelfactor-station ls -la /data/evidence/$(date +%Y-%m-%d)/

# Get evidence via API
curl http://localhost:8080/api/v1/evidence/manifest?date=$(date +%Y-%m-%d)
```

---

## Phase 2: Hybrid Mode (Cloud Sync)

### 2.1 Get Cloud Credentials

1. Log into https://app.intelfactor.ai
2. Navigate to **Settings → API Keys**
3. Create new key for your station
4. Note: `CLOUD_API_KEY=ifk_xxxxxxxxxxxx`

### 2.2 Configure

```bash
cd deploy/hybrid
cp .env.example .env

# Edit .env:
STATION_ID=factory_01_line_03
CAMERA_URI=rtsp://192.168.1.100:554/stream

# Cloud sync
CLOUD_API_URL=https://api.intelfactor.ai
CLOUD_API_KEY=ifk_your_key_here

# Optional: S3 for evidence images
S3_BUCKET=intelfactor-evidence-prod
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
```

### 2.3 Deploy

```bash
docker compose up -d

# Check both containers
docker compose ps
# Should show: intelfactor-station (healthy), intelfactor-sync-agent (running)
```

### 2.4 Verify Sync

```bash
# Watch sync agent logs
docker logs -f intelfactor-sync-agent

# Expected output:
# 2026-02-14 10:00:00 [sync-cloud] INFO: Cloud sync agent started
# 2026-02-14 10:05:00 [sync-cloud] INFO: Synced 15 events to cloud
# 2026-02-14 10:05:01 [sync-cloud] INFO: Synced 3 triples to cloud
```

### 2.5 Verify in Cloud Dashboard

1. Open https://app.intelfactor.ai
2. Check **Stations** page - your station should show "Online"
3. Check **Events** tab for synced defect events
4. Click an event to view evidence image

---

## API Quick Reference

### Local Station API (http://localhost:8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/events` | GET | List defect events |
| `/api/events/<id>` | GET | Get single event |
| `/api/triples` | GET | List causal triples |
| `/api/feedback` | POST | Submit operator feedback |
| `/api/v1/evidence/<id>` | GET | Evidence metadata |
| `/api/v1/evidence/<id>/image.jpg` | GET | Evidence image |
| `/api/v1/evidence/manifest` | GET | Evidence manifest |
| `/api/recommendations` | GET | Pending RCA recommendations |
| `/api/drift` | GET | Process parameter drift |

### Cloud API (https://api.intelfactor.ai)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events/batch` | POST | Batch upload events |
| `/api/v1/triples/batch` | POST | Batch upload triples |
| `/api/v1/heartbeats` | POST | Station heartbeat |
| `/api/v1/stations/:id/config` | GET | Get station config |

---

## Troubleshooting

### Station won't start

```bash
# Check logs
docker compose logs station

# Verify GPU access
docker exec intelfactor-station nvidia-smi
```

### No events appearing

```bash
# Check if camera is working
docker exec intelfactor-station python3 -c "
import cv2
cap = cv2.VideoCapture('/dev/video0')
print('Camera opened:', cap.isOpened())
ret, frame = cap.read()
print('Got frame:', ret, frame.shape if ret else 'None')
"
```

### Sync agent not connecting

```bash
# Test cloud connectivity
docker exec intelfactor-sync-agent curl -v https://api.intelfactor.ai/health

# Check API key
docker exec intelfactor-sync-agent printenv CLOUD_API_KEY
```

### Reset sync state (re-sync from scratch)

```bash
docker stop intelfactor-sync-agent
docker exec intelfactor-station rm -f /data/*_watermarks.json
docker start intelfactor-sync-agent
```

---

## Network Requirements

### Outbound (Edge → Cloud)

| Destination | Port | Purpose |
|-------------|------|---------|
| api.intelfactor.ai | 443 | API sync |
| s3.amazonaws.com | 443 | Evidence upload |

### Inbound (LAN only)

| Port | Purpose |
|------|---------|
| 8080 | Local dashboard + API |

---

## Files Changed (Patch Summary)

```
deploy/edge-only/Dockerfile      # Fixed CMD, proper pip install
deploy/hybrid/Dockerfile.sync    # Fixed to include full project
packages/inference/cli.py        # Switched to api_v2
```

### Key Fixes Applied

1. **Dockerfile CMD**: Changed from broken `serve --api-only` to working `run_station()`
2. **Dependencies**: Changed from manual pip list to `pip install ".[jetson]"`
3. **API**: Switched from `api.py` to `api_v2.py` (has evidence endpoints)
4. **Sync Dockerfile**: Now copies full project instead of just sync_cloud.py
