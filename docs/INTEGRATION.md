# Integration Guide: app.intelfactor.ai

This guide covers how to build, deploy, and integrate the edge inference engine with the IntelFactor cloud platform.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   EDGE DEVICE                              CLOUD (app.intelfactor.ai)   │
│   ┌─────────────────────┐                  ┌─────────────────────────┐  │
│   │  Station Container  │                  │  Cloud API              │  │
│   │  ┌───────────────┐  │    HTTPS POST    │  api.intelfactor.ai     │  │
│   │  │ Inference     │  │  ──────────────▶ │  ├── /api/v1/events    │  │
│   │  │ RCA Pipeline  │  │                  │  ├── /api/v1/triples   │  │
│   │  │ Local API     │  │                  │  └── /api/v1/heartbeats│  │
│   │  └───────────────┘  │                  └─────────────────────────┘  │
│   │         │           │                              │                │
│   │         ▼           │                              ▼                │
│   │  ┌───────────────┐  │                  ┌─────────────────────────┐  │
│   │  │ SQLite + Evid │  │                  │  Dashboard              │  │
│   │  └───────────────┘  │                  │  app.intelfactor.ai     │  │
│   │         │           │                  │  - Multi-site view      │  │
│   │         ▼           │    S3 UPLOAD     │  - Analytics            │  │
│   │  ┌───────────────┐  │  ──────────────▶ │  - Evidence viewer      │  │
│   │  │ Sync Agent    │  │                  └─────────────────────────┘  │
│   │  └───────────────┘  │                                               │
│   └─────────────────────┘                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Docker** with NVIDIA Container Toolkit (for GPU support)
2. **API credentials** from app.intelfactor.ai
3. **AWS credentials** (optional, for S3 evidence upload)

## Step 1: Build Docker Images

```bash
# Clone the repository
git clone https://github.com/tonesgainz/intelfactor-inference.git
cd intelfactor-inference

# Build all images
make build

# Or build individually:
make build-edge    # Station image
make build-sync    # Sync agent image

# For Jetson devices:
make build-jetson
```

### Verify Build

```bash
docker images | grep intelfactor
# ghcr.io/tonesgainz/intelfactor-inference        latest    ...
# ghcr.io/tonesgainz/intelfactor-inference-sync   latest    ...
```

## Step 2: Get API Credentials

1. Log in to https://app.intelfactor.ai
2. Navigate to **Settings → API Keys**
3. Create a new API key for your station
4. Note the following:
   - `API Key`: `ifk_xxxxxxxxxxxxxxxxxxxx`
   - `API URL`: `https://api.intelfactor.ai`

### Optional: S3 Credentials

For evidence image upload to cloud:

1. Navigate to **Settings → Storage**
2. Get S3 credentials:
   - `S3_BUCKET`: `intelfactor-evidence-prod`
   - `AWS_ACCESS_KEY_ID`: `AKIA...`
   - `AWS_SECRET_ACCESS_KEY`: `...`
   - `AWS_REGION`: `us-west-2`

## Step 3: Configure Environment

```bash
cd deploy/hybrid
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Station identification
STATION_ID=factory_01_line_03
CAMERA_URI=rtsp://192.168.1.100:554/stream

# Model files location
MODELS_DIR=/opt/intelfactor/models

# Evidence storage
EVIDENCE_MAX_GB=50

# ── Cloud Integration ────────────────────────────────────────
CLOUD_API_URL=https://api.intelfactor.ai
CLOUD_API_KEY=ifk_your_api_key_here

# Sync interval (seconds)
SYNC_INTERVAL_SEC=300

# S3 for evidence upload (optional)
S3_BUCKET=intelfactor-evidence-prod
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2
```

## Step 4: Deploy

### Option A: Hybrid Mode (Recommended)

Runs locally with cloud sync:

```bash
cd deploy/hybrid
docker compose up -d

# Check status
docker compose ps
docker compose logs -f station
```

### Option B: Edge-Only Mode

Runs completely offline:

```bash
cd deploy/edge-only
docker compose up -d
```

### Verify Deployment

```bash
# Health check
curl http://localhost:8080/health

# Expected response:
{
  "status": "ok",
  "station": {
    "storage_mode": "local",
    "station_id": "factory_01_line_03"
  }
}
```

## Step 5: Verify Cloud Integration

### Check Sync Status

```bash
# View sync agent logs
docker logs -f intelfactor-sync-agent

# Expected output:
# 2026-02-14 10:00:00 [sync-cloud] INFO: Cloud sync agent started
# 2026-02-14 10:05:00 [sync-cloud] INFO: Synced 15 events to cloud
# 2026-02-14 10:05:01 [sync-cloud] INFO: Synced 3 triples to cloud
```

### Verify in Cloud Dashboard

1. Open https://app.intelfactor.ai
2. Navigate to **Stations**
3. Your station should appear with status "Online"
4. Check **Events** tab for synced defect events
5. Check **Evidence** tab for uploaded images

## API Integration Reference

### Authentication

All requests to the cloud API require authentication:

```bash
curl -X GET https://api.intelfactor.ai/api/v1/stations \
  -H "Authorization: Bearer ifk_your_api_key" \
  -H "X-Station-ID: factory_01_line_03"
```

### Endpoints Used by Sync Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/events/batch` | Batch upload defect events |
| POST | `/api/v1/triples/batch` | Batch upload causal triples |
| POST | `/api/v1/heartbeats` | Station heartbeat |
| GET | `/api/v1/stations/:id/config` | Get station config |

### Event Payload

```json
{
  "events": [
    {
      "event_id": "evt_20260214_100522_a1b2c3",
      "timestamp": "2026-02-14T10:05:22Z",
      "station_id": "factory_01_line_03",
      "defect_type": "scratch_surface",
      "severity": 0.8,
      "confidence": 0.95,
      "verdict": "FAIL",
      "frame_ref": "2026-02-14/evt_20260214_100522_a1b2c3.jpg"
    }
  ]
}
```

### Triple Payload

```json
{
  "triples": [
    {
      "triple_id": "triple_abc123",
      "timestamp": "2026-02-14T10:05:30Z",
      "station_id": "factory_01_line_03",
      "defect_type": "scratch_surface",
      "cause_parameter": "grinding_rpm",
      "cause_value": 3150,
      "cause_target": 3000,
      "cause_drift_pct": 5.0,
      "operator_action": "accepted",
      "status": "verified"
    }
  ]
}
```

## Troubleshooting

### Sync Agent Not Connecting

```bash
# Check network connectivity
docker exec intelfactor-sync-agent curl -v https://api.intelfactor.ai/health

# Verify API key
docker exec intelfactor-sync-agent printenv CLOUD_API_KEY
```

### Events Not Appearing in Cloud

1. Check sync agent logs for errors
2. Verify watermark file:
   ```bash
   docker exec intelfactor-station cat /data/station_01_watermarks.json
   ```
3. Check cloud API response codes in logs

### S3 Upload Failing

```bash
# Test S3 connectivity
docker exec intelfactor-sync-agent python3 -c "
import boto3
s3 = boto3.client('s3')
print(s3.list_buckets())
"
```

### Reset Sync State

To re-sync all data from scratch:

```bash
# Stop sync agent
docker stop intelfactor-sync-agent

# Remove watermarks
docker exec intelfactor-station rm -f /data/*_watermarks.json

# Restart
docker start intelfactor-sync-agent
```

## Network Requirements

### Outbound Connections (Edge → Cloud)

| Destination | Port | Protocol | Purpose |
|-------------|------|----------|---------|
| api.intelfactor.ai | 443 | HTTPS | API sync |
| s3.amazonaws.com | 443 | HTTPS | Evidence upload |

### Inbound Connections (LAN Only)

| Port | Protocol | Purpose |
|------|----------|---------|
| 8080 | HTTP | Local dashboard + API |

### Firewall Rules

```bash
# Allow outbound HTTPS
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

# Allow local dashboard access
iptables -A INPUT -p tcp --dport 8080 -s 192.168.0.0/16 -j ACCEPT
```

## Production Checklist

- [ ] API key stored securely (not in git)
- [ ] S3 credentials rotated regularly
- [ ] Sync interval appropriate for bandwidth (300s default)
- [ ] Evidence retention configured (`EVIDENCE_MAX_GB`)
- [ ] Station ID unique across all sites
- [ ] Health monitoring configured (Prometheus/Grafana)
- [ ] Backup strategy for local SQLite database
- [ ] Log rotation configured

## Support

- **Dashboard**: https://app.intelfactor.ai
- **API Docs**: https://api.intelfactor.ai/docs
- **Support**: support@intelfactor.ai
