# GAP ANALYSIS: IntelFactor Inference on Jetson Orin Nano Super + FLIR Blackfly S

**Date:** 2026-03-25
**Target Hardware:** Jetson Orin Nano Super (8GB, JetPack 6.x) + FLIR Blackfly S BFS-U3-63S4M (6.3MP Mono8, USB3)
**Goal:** `intelfactor-station run` end-to-end: camera frame in -> TRT inference -> defect detection -> SQLite storage -> operator dashboard at :8080

---

## WORKING (exists and appears functional)

| Component | File | Status |
|-----------|------|--------|
| CLI entry point | `packages/inference/cli.py` | Complete. `intelfactor-station run` parses args, loads config, starts runtime, API, camera |
| Station runtime | `packages/inference/modes/runtime.py` | Complete. Wires vision->accumulator->RCA pipeline. Starts/stops cleanly |
| Hardware auto-detect | `packages/inference/providers/resolver.py` | Complete. Detects Jetson via `/etc/nv_tegra_release`, classifies Orin Nano, selects yolov8n_trt + qwen25_3b_int4 |
| TRT vision provider | `packages/inference/providers/vision_trt.py` | Complete. TRT load, buffer alloc, YOLOv8 output parse, NMS, DefectIQ rules engine |
| llama.cpp language provider | `packages/inference/providers/language_llama.py` | Complete. Qwen chat template, bilingual JSON parse, stub fallback |
| vLLM language provider | `packages/inference/providers/language_vllm.py` | Complete. Server + in-process modes |
| Provider base classes | `packages/inference/providers/base.py` | Complete. Clean ABC for VisionProvider, LanguageProvider |
| Schemas (dataclasses) | `packages/inference/schemas.py` | Complete. All types: DetectionResult, AnomalyAlert, CausalTriple, etc. |
| RCA Pipeline orchestrator | `packages/inference/rca/pipeline.py` | Complete. 4-layer chain: accumulate->correlate->explain->recommend |
| Defect Accumulator (Layer 1) | `packages/inference/rca/accumulator.py` | Complete. SQLite WAL, z-score anomaly detection, 30-day rolling window |
| Process Correlator (Layer 2) | `packages/inference/rca/correlator.py` | Complete. Pearson r, drift detection, parameter readings |
| RCA Explainer (Layer 3) | `packages/inference/rca/explainer.py` | Complete. SLM prompt builder, bilingual output |
| Action Recommender (Layer 4) | `packages/inference/rca/recommender.py` | Complete. SOP-mapped recommendations, triple store |
| Evidence writer | `packages/inference/evidence.py` | Complete. JPEG ring buffer, FIFO quota, manifest.jsonl |
| SQLite storage layer | `packages/inference/storage/sqlite_*.py` | Complete. Events, evidence index, triples -- all WAL mode |
| Storage factory | `packages/inference/storage/factory.py` | Complete. Singleton pattern, STORAGE_MODE env var routing |
| Sensor ingestion service | `packages/ingestion/sensor_service.py` | Complete. MQTT subscriber, z-score scoring, baseline recomputation |
| Maintenance IQ rules engine | `packages/policy/maintenance_iq.py` | Complete. Threshold evaluation, multi-sensor aggregation, SOP mapping |
| Production metrics engine | `packages/visibility/production_metrics.py` | Complete. Throughput, cycle time, utilization, shift summaries |
| Metrics API blueprint | `packages/visibility/metrics_api.py` | Complete. /api/metrics/* endpoints |
| Operator dashboard | `packages/inference/static/index.html` | Complete. Chinese-primary, bilingual toggle, polls 6 API endpoints every 5s |
| Station config | `configs/station.yaml` | Complete. 129 lines, all sections populated |
| Wiko taxonomy | `configs/wiko_taxonomy.yaml` | Complete. 13 defect types, bilingual, SOP-mapped |
| TRT engine build script | `scripts/build_trt_engine.sh` | Complete. .pt->ONNX->trtexec->engine+manifest. OOM detection, verify step |
| System doctor | `scripts/doctor.py` | Complete. GPU, camera, storage, models, disk checks |
| Setup script | `deploy/station/setup.sh` | Complete. Idempotent, creates user, installs service |
| Systemd service | `deploy/systemd/intelfactor-station.service` | Complete. Security hardening, memory limits, restart policy |
| Docker Compose (3 profiles) | `deploy/edge-only/`, `deploy/hybrid/`, `deploy/hub/` | Complete |

---

## STUB/INCOMPLETE (code exists but has issues)

| Component | File | What's Missing |
|-----------|------|----------------|
| **`create_app()` signature mismatch** | `packages/inference/api_v2.py:20` | Defined as `create_app(runtime=None)` but `cli.py:120` calls it with `sensor_service=`, `maintenance_iq=`, `machine_health_config=` kwargs. **This will crash on startup** with `TypeError: create_app() got an unexpected keyword argument 'sensor_service'` |
| **Camera ingest -- OpenCV only** | `packages/inference/ingest.py` | Uses `cv2.VideoCapture()` exclusively. **Zero PySpin/Spinnaker support.** The FLIR Blackfly S BFS-U3-63S4M uses Spinnaker SDK for acquisition. `cv2.VideoCapture("/dev/video0")` will NOT work -- the Blackfly S is not a UVC device; it requires PySpin. |
| **No `camera:` block in station.yaml** | `configs/station.yaml` | The YAML has no `camera:` section. `cli.py:101` checks `raw.get("camera", {}).get("source")` -- without it, camera ingest is **silently skipped** and the station runs API-only. |
| **Mono8 frame handling** | `packages/inference/providers/vision_trt.py:141` | `_preprocess()` does `blob.transpose(2, 0, 1)` assuming 3-channel HWC. The Blackfly S BFS-U3-63S4M outputs **Mono8** (single channel). A grayscale frame is shape `(H, W)` -- the transpose will crash with `AxesError`. |
| **TRT provider stub fallback** | `packages/inference/providers/vision_trt.py:77-80` | If `import tensorrt` fails, silently enters `_stub_mode` returning zero detections. On Jetson with TRT installed, this won't trigger -- but it masks real load failures. |
| **Cloud storage stubs** | `packages/inference/storage/factory.py:36` | `STORAGE_MODE=cloud` raises `NotImplementedError`. Fine for edge-only -- not a blocker. |
| **vLLM provider uses `requests`** | `packages/inference/providers/language_vllm.py:113` | Uses `import requests` for server mode, but code style guide says `httpx over requests`. Minor. |

---

## MISSING (referenced but no code exists)

| Expected Component | Where Referenced | What's Needed |
|-------------------|-----------------|---------------|
| **PySpin camera ingest** | User context: "raw PySpin capture with two branches". No code in repo. | A `CameraIngest` variant (or protocol handler) that uses `PySpin.System`, `cam.GetNodeMap()`, `cam.BeginAcquisition()` to acquire frames from the Blackfly S. Branch A=KVS (working separately), Branch B=local TRT inference (this repo). |
| **`camera:` section in station.yaml** | `cli.py:101` reads `raw.get("camera", {}).get("source")` | Need `camera:` block with `source: "pyspin://0"` or `source: "/dev/video0"`, `protocol: "pyspin"`, `width: 3088`, `height: 2064` (native Blackfly S res), etc. |
| **`CameraProtocol.PYSPIN`** | `packages/inference/ingest.py` -- enum has RTSP/GIGE/USB/FILE | Need `PYSPIN = "pyspin"` enum variant and corresponding `_connect()` branch that uses Spinnaker SDK |
| **`edge.yaml`** | `cli.py:85`, `runtime.py:123` -- loaded at startup | File does not exist in repo. `_load_edge_yaml()` silently returns `{}` if missing, so machine health copilot and process correlator run with empty config. |
| **Maintenance API endpoints** | `test_maintenance_api.py` tests 10 endpoints at `/api/maintenance/*` | **Not wired in `api_v2.py`**. The test file creates its own Flask app with these routes, but the production `create_app()` doesn't register them. The `/api/maintenance/status`, `/api/maintenance/events`, etc. will 404. |
| **Sync heartbeat endpoint** | `CLAUDE.md` mentions `/api/sync/heartbeat` | Not implemented -- known gap per CLAUDE.md |
| **OTA model update** | CLAUDE.md "Known Gaps" | No mechanism -- manual scp only |
| **`bridge_to_station.py`** | Not in repo | No bridge module exists. If KVS branch and TRT branch need a shared capture layer, this would be the integration point. |
| **`paho-mqtt` in dependencies** | `packages/ingestion/sensor_service.py` imports it optionally | Not listed in `pyproject.toml` dependencies (not even optional). Works without broker but won't connect to MQTT if `paho-mqtt` isn't manually installed. |

---

## CRITICAL PATH TO FIRST DETECTION

Steps to go from current state to: **camera frame -> YOLO detection -> SQLite -> visible on :8080**

### 1. Fix `create_app()` signature mismatch

**File:** `packages/inference/api_v2.py:20`

Change `def create_app(runtime: Any = None)` to accept the keyword arguments `cli.py` passes: `sensor_service`, `maintenance_iq`, `machine_health_config`. Or change `cli.py` to only pass `runtime`. **Without this fix, the station will crash on startup.**

### 2. Add `camera:` block to `configs/station.yaml`

**File:** `configs/station.yaml`

Add camera configuration that `cli.py:101` reads:

```yaml
camera:
  source: "pyspin://0"    # or "/dev/video0" for OpenCV fallback
  protocol: pyspin
  fps_target: 5           # 6.3MP frames are big; match inference budget
  width: 3088
  height: 2064
```

### 3. Add PySpin camera protocol to `CameraIngest`

**File:** `packages/inference/ingest.py`

- Add `PYSPIN = "pyspin"` to `CameraProtocol` enum
- In `_connect()`, add a PySpin branch:

```python
if self.config.protocol == CameraProtocol.PYSPIN:
    import PySpin
    self._system = PySpin.System.GetInstance()
    self._cam = self._system.GetCameras()[0]
    self._cam.Init()
    # Configure Mono8, exposure, etc.
    self._cam.BeginAcquisition()
```

- In `_read_frame()`, add PySpin path: `image_result = self._cam.GetNextImage()`, convert to numpy
- In `_release()`, add cleanup: `self._cam.EndAcquisition()`, `self._cam.DeInit()`

### 4. Handle Mono8 -> BGR conversion for YOLO

**File:** `packages/inference/providers/vision_trt.py:134-143`

The Blackfly S outputs Mono8 (1-channel). `_preprocess()` assumes 3-channel BGR. Add:

```python
if len(frame.shape) == 2 or frame.shape[2] == 1:
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
```

before the resize.

### 5. Build TRT engine on the Jetson

**Command:** On the Orin Nano Super, with the ONNX model present:

```bash
./scripts/build_trt_engine.sh yolov8n.onnx fp16
```

This produces `/opt/intelfactor/models/vision/yolov8n_fp16.engine`. The resolver will find it.

### 6. Download Qwen GGUF for RCA

**Command:**

```bash
./scripts/download_qwen.sh
```

Or manually place `Qwen2.5-3B-Instruct-GGUF` file in `/opt/intelfactor/models/`.

### 7. Create `edge.yaml` (or run without it)

**File:** `/opt/intelfactor/config/edge.yaml`

Minimum: `{}` (empty). The runtime gracefully handles missing edge.yaml. For full RCA, add process parameters and machine health config.

### 8. Wire maintenance API endpoints into `api_v2.py`

**File:** `packages/inference/api_v2.py`

The `/api/maintenance/*` endpoints exist only in test fixtures. Register them in `create_app()` using the `sensor_service` and `maintenance_iq` objects. **Not strictly required for first detection**, but the dashboard will be missing machine health data.

### 9. Run the station

```bash
intelfactor-station run --config configs/station.yaml --port 8080
```

### 10. Verify on dashboard

Open `http://<jetson-ip>:8080/` -- should show live detections, events, alerts.

---

## BLOCKERS

| Blocker | Severity | Detail |
|---------|----------|--------|
| **`create_app()` TypeError** | **P0 -- Station won't start** | `cli.py` passes 3 extra kwargs that `api_v2.py` doesn't accept. Immediate crash. |
| **No PySpin support** | **P0 -- No frames** | `CameraIngest` only uses `cv2.VideoCapture`. The FLIR Blackfly S BFS-U3-63S4M is not UVC-compatible; it requires Spinnaker/PySpin SDK. Without PySpin, zero frames reach the pipeline. |
| **No `camera:` in config** | **P0 -- Camera silently skipped** | `cli.py:101` checks `raw.get("camera", {}).get("source")` -- without a camera block, `args.no_camera` is effectively True and no ingest starts. |
| **Mono8 crashes `_preprocess()`** | **P0 -- Inference crash** | `transpose(2, 0, 1)` on a 2D array throws `AxesError`. Every frame from the Blackfly S (Mono8) will crash the vision provider. |
| **PySpin not in `pyproject.toml`** | **P1 -- Install fails** | `spinnaker-python` / `PySpin` must be installed from FLIR's Spinnaker SDK wheel. Not pip-installable. Needs documented install step or a try/except import. |
| **6.3MP resolution at 30 FPS** | **P1 -- Memory/latency** | Default `CameraConfig` is 1920x1080 @ 30fps. The Blackfly S native is 3088x2064. At 30fps that's ~380MB/s of raw pixels. Need to either (a) downsample in PySpin before handing to pipeline, or (b) set `fps_target` to ~5fps for inspection cadence. |
| **`pycuda` assumed in TRT provider** | **P1 -- Import error** | `_trt_detect()` does `import pycuda.driver`. On JetPack 6.x, `pycuda` may not be pre-installed. Need `pip install pycuda` or switch to TRT's built-in CUDA memory management. |
| **`paho-mqtt` not in deps** | **P2 -- Sensor service can't connect to MQTT** | Works in broker-less mode but won't subscribe to sensor topics if broker is available. |

---

## RECOMMENDED FILE CHANGES

| File | Change | Why |
|------|--------|-----|
| `packages/inference/api_v2.py:20` | Change signature to `def create_app(runtime=None, sensor_service=None, maintenance_iq=None, machine_health_config=None)` and wire the maintenance endpoints | **Fixes P0 startup crash** and enables maintenance API |
| `packages/inference/ingest.py` | Add `PYSPIN = "pyspin"` to enum; add PySpin `_connect()`, `_read_frame()`, `_release()` paths | **Enables FLIR Blackfly S capture** |
| `packages/inference/providers/vision_trt.py:134-143` | Add Mono8->BGR conversion before resize: `if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)` | **Fixes crash on single-channel frames** |
| `configs/station.yaml` | Add `camera:` section with `source: "pyspin://0"`, `protocol: pyspin`, `fps_target: 5`, native resolution | **Enables camera ingest** (without it, ingest is silently skipped) |
| `pyproject.toml` | Add `pycuda>=2024.1` to `[jetson]` extras. Add note about Spinnaker SDK manual install | Ensures TRT inference works on Jetson |
| `packages/inference/providers/vision_trt.py` | Consider replacing `pycuda` buffer management with `tensorrt`'s native memory APIs (available in TRT 10.x / JetPack 6.x) | Reduces dependency count, uses JetPack's native stack |
| `deploy/station/setup.sh` | Add Spinnaker SDK install step (download .deb from FLIR, install PySpin wheel) | Real deployments need this |

---

## SUMMARY

The core inference pipeline (TRT -> RCA -> SQLite -> dashboard) is architecturally complete and well-structured. The three **P0 blockers** are:

1. A `create_app()` signature mismatch that crashes on startup
2. Zero PySpin/Spinnaker support for the FLIR Blackfly S
3. No Mono8->BGR conversion for single-channel frames

Fix those three and add a `camera:` block to the config, and you have a working inspection station.
