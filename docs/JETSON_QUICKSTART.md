# Jetson Quickstart — IntelFactor Station

Deploy IntelFactor edge inspection on **NVIDIA Jetson Orin Nano / Orin Nano Super** running **JetPack 6.x**.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Jetson Orin Nano (8GB) | Orin Nano Super also supported |
| JetPack 6.x (L4T R36+) | Flashed via SDK Manager |
| Docker + NVIDIA Container Toolkit | Ships with JetPack; verify with `docker info` |
| Trained YOLO model | `.onnx` file (export on x86: `yolo export model=best.pt format=onnx`) |
| Network access | For initial setup only — station runs offline after deploy |

## Quick Start (Docker — recommended)

```bash
# 1. Clone repo
git clone <repo-url>
cd intelfactor-inference

# 2. Run bootstrap (checks prereqs, creates directories)
chmod +x scripts/jetson_bootstrap.sh
./scripts/jetson_bootstrap.sh

# 3. Copy your ONNX model and build TRT engine ON the Jetson
sudo cp yolov8n.onnx /opt/intelfactor/models/vision/
./scripts/build_trt_engine.sh /opt/intelfactor/models/vision/yolov8n.onnx fp16
# Takes 5–20 min on Orin Nano. Engine written to /opt/intelfactor/models/vision/

# 4. (Optional) Download Qwen LLM for RCA explanations
./scripts/setup_models.sh --quick
# Or skip: set ENABLE_LOCAL_LLM=false in .env

# 5. Configure .env
cd deploy/edge-only
cp .env.example .env
nano .env
# Set: STATION_ID, CAMERA_PROTOCOL, CAMERA_URI, VISION_ENGINE_PATH

# 6. Start
docker compose up -d

# 7. Verify
curl http://localhost:8080/health
./scripts/validate_station.sh
```

Dashboard: `http://<jetson-ip>:8080`

---

## .env Reference

```env
# Station identity
STATION_ID=wiko-final-inspection-01

# Camera — start with "file" to validate pipeline, then switch to real camera
CAMERA_PROTOCOL=file
CAMERA_URI=/data/sample_images

# Models (host path mounted into container)
MODELS_DIR=/opt/intelfactor/models
VISION_ENGINE_PATH=/models/vision/yolov8n_fp16.engine

# Language model — set false to skip LLM, uses statistical fallback for RCA
ENABLE_LOCAL_LLM=false
LLM_MODEL_PATH=/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf

# Storage
DB_PATH=/data/local.db
EVIDENCE_DIR=/data/evidence
API_PORT=8080
EVIDENCE_MAX_GB=50
```

---

## Camera Protocol Examples

### Start with file mode (no hardware needed)
```env
CAMERA_PROTOCOL=file
CAMERA_URI=/data/sample_images
```

### FLIR Blackfly S (PySpin)
```env
CAMERA_PROTOCOL=pyspin
CAMERA_URI=25423789
```
Find serial via SpinView or the Spinnaker SDK.

### USB Camera
```env
CAMERA_PROTOCOL=usb
CAMERA_URI=/dev/video0
```
Check devices: `ls /dev/video*`

### IP Camera (RTSP)
```env
CAMERA_PROTOCOL=rtsp
CAMERA_URI=rtsp://192.168.1.100:554/stream
```

---

## Capture Frames for Model Training

The station's normal evidence writer only saves **FAIL/REVIEW evidence after inference**. For model training, use the raw frame capture utility before relying on the model:

```bash
# USB camera: save 300 frames, one every 0.5s
python3 scripts/capture_training_frames.py \
  --protocol usb \
  --source /dev/video0 \
  --output-dir /opt/intelfactor/data/training_frames \
  --station-id wiko-final-inspection-01 \
  --label unlabeled \
  --interval-sec 0.5 \
  --max-frames 300

# RTSP camera
python3 scripts/capture_training_frames.py \
  --protocol rtsp \
  --source rtsp://192.168.1.100:554/stream \
  --label unlabeled \
  --max-frames 300

# FLIR Blackfly S via PySpin
python3 scripts/capture_training_frames.py \
  --protocol pyspin \
  --source 25423789 \
  --label unlabeled \
  --max-frames 300
```

Output layout:

```text
/opt/intelfactor/data/training_frames/
  unlabeled/
    20260502T201500Z/
      wiko-final-inspection-01_20260502T201500Z_000001.jpg
      wiko-final-inspection-01_20260502T201500Z_000002.jpg
      manifest.jsonl
```

Use labels such as `pass`, `scratch_surface`, `burr`, or `unlabeled` depending on your collection pass. Annotation tools can ingest the JPEGs; `manifest.jsonl` records source, station, timestamp, frame size, and label.

Recommended factory flow:

1. Start with `--label unlabeled` and collect broad production variation.
2. Add labeled passes for known defects only when an operator/QC lead confirms class names.
3. Keep training frames separate from `/data/evidence`; evidence is for inspection audit trails, training frames are for dataset building.
4. Do not sync training frames off-site unless the factory approves the data policy.

---

## TensorRT Engine Build

**TRT engines are device-specific.** Always build on the target Jetson.

```bash
# FP16 (recommended for Orin Nano — best accuracy/speed tradeoff)
./scripts/build_trt_engine.sh model.onnx fp16

# INT8 (fastest, needs ≥100 calibration images)
./scripts/build_trt_engine.sh model.onnx int8 --calib ./calibration_images/

# Custom output directory
./scripts/build_trt_engine.sh model.onnx fp16 --output-dir /mnt/ssd/models/vision/
```

The build script:
1. Validates inputs and checks for `trtexec`
2. Converts `.pt` → `.onnx` if needed (requires ultralytics + PyTorch)
3. Runs `trtexec` with optimized shapes for YOLO (1×3×640×640)
4. Writes engine + `_manifest.json` with SHA256 checksums
5. Verifies the engine loads on this GPU

**Never copy `.engine` files between different GPU architectures.**

---

## Bare Metal / Systemd (secondary path)

If you prefer running without Docker:

```bash
# Run setup script (creates venv, installs deps, enables systemd service)
sudo ./deploy/station/setup.sh --config configs/station.yaml

# Build TRT engine
./scripts/build_trt_engine.sh /opt/intelfactor/models/vision/yolov8n.onnx fp16

# Download language model
./scripts/setup_models.sh --quick

# Start
sudo systemctl start intelfactor-station
journalctl -u intelfactor-station -f
```

The setup script:
- Creates `/opt/intelfactor/{config,data,models,logs}`
- Creates a Python venv at `/opt/intelfactor/venv` (inherits system CUDA/TRT)
- Installs the package with `[jetson]` extras
- Installs and enables the systemd unit

---

## Validation

After deployment, run:

```bash
# Full validation (checks Docker, GPU, models, data paths, API health)
./scripts/validate_station.sh

# API health
curl http://localhost:8080/health
# → {"status": "ok", "station": {"storage_mode": "local"}}

# Station status
curl http://localhost:8080/api/status

# Create a test event
curl -X POST http://localhost:8080/api/events \
  -H "Content-Type: application/json" \
  -d '{"event_id": "test_001", "station_id": "station_01", "defect_type": "blade_scratch", "confidence": 0.9, "verdict": "FAIL", "severity": 0.7}'

# Retrieve it
curl http://localhost:8080/api/events/test_001
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker compose up` fails with GPU error | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| TRT engine build OOM (exit 137) | Reduce workspace: `--workspace 2048`. Stop other GPU processes first. |
| TRT engine won't load | Engine was built on different GPU arch. Rebuild on this Jetson. |
| Camera not detected | Check `ls /dev/video*` (USB) or SpinView (FLIR). Ensure `/dev` is mounted in compose. |
| Health endpoint unreachable | Check `docker logs intelfactor-station`. Verify port 8080 is not in use. |
| SQLite lock errors | Only one station process should write to the DB. Check for duplicate containers. |
| LLM not loading | Set `ENABLE_LOCAL_LLM=false` in `.env` to use statistical fallback instead. |

---

## Hardware Tier: Orin Nano

| Component | Spec |
|-----------|------|
| Vision model | YOLOv8n TRT FP16 (~15ms latency) |
| Language model | Qwen 2.5-3B INT4 via llama.cpp (optional) |
| RAM budget | ~8GB total, ~6GB usable |
| Storage | SQLite WAL + JPEG evidence ring buffer |
| Evidence quota | Default 50GB FIFO |

---

## Architecture

```
Camera (RTSP/USB/FLIR/file)
 │
 ▼
CameraIngest → StationRuntime.process_frame()
 ├── VisionProvider.detect(frame)  ← TensorRT YOLO
 ├── EvidenceWriter.write(frame)   ← JPEG + JSON sidecar
 └── RCAPipeline.ingest(result)    ← FAIL/REVIEW only
                                      │
                            [every 5 min]
                                      ▼
                            RCAPipeline.run_rca()
                             ├── DefectAccumulator (z-score)
                             ├── ProcessCorrelator (Pearson r)
                             ├── RCAExplainer (LLM or stats)
                             └── ActionRecommender (SOP-mapped)

Flask API ← http://0.0.0.0:8080
 ├── /health, /api/status
 ├── /api/events (CRUD)
 ├── /api/v1/evidence (images)
 ├── /api/triples (RCA feedback)
 └── /api/maintenance/* (sensor health)
```

All data stays on-device. Cloud sync is optional (`CLOUD_API_URL` in `.env`).
