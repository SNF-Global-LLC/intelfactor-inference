# Roboflow Hosted Integration (Temporary Bridge)

## Overview

This document describes the **temporary** Roboflow hosted inference provider that allows you to get real defect predictions immediately while training your own local TensorRT model.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Local Capture  │────▶│ Roboflow API    │────▶│ Local Evidence  │
│  (Jetson/PC)    │     │ (this provider) │     │ Write + SQLite  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                       │
                              ┌─────────────────┐     │
                              │ Local UI Render │◀────┘
                              │ + Sync          │
                              └─────────────────┘
```

**Key principle**: Only the inference step is hosted. Everything else (capture, storage, UI, sync) remains local.

## Quick Start

### 1. Get Roboflow Credentials

1. Go to https://app.roboflow.com and sign in
2. Navigate to your project (e.g., "cutlery-defects")
3. Get your **API Key** from Settings
4. Note your **workspace**, **project**, and **model version**

### 2. Set Environment Variable

```bash
# macOS/Linux
export ROBOFLOW_API_KEY="your_api_key_here"

# Or add to ~/.bashrc or ~/.zshrc for persistence
```

### 3. Copy Configuration

```bash
# On Jetson/workstation
cp configs/station-roboflow.yaml /opt/intelfactor/config/station.yaml

# Edit to set your workspace/project
nano /opt/intelfactor/config/station.yaml
```

### 4. Start the Station

```bash
# The station will automatically use Roboflow for inference
python -m packages.inference.api_v2
```

## Configuration Reference

### Required Settings

| Config | Description | Example |
|--------|-------------|---------|
| `vision_model` | Must be `roboflow_hosted` | `roboflow_hosted` |
| `roboflow_api_key` | Your API key | `${ROBOFLOW_API_KEY}` |
| `roboflow_workspace` | Workspace name | `kmg-kyoto` |
| `roboflow_project` | Project name | `cutlery-defects` |
| `roboflow_version` | Model version | `3` |

### Optional Settings

| Config | Default | Description |
|--------|---------|-------------|
| `confidence_threshold` | `0.5` | Minimum confidence for detection |
| `fail_threshold` | `0.85` | Auto-FAIL threshold |
| `review_threshold` | `0.50` | Human review threshold |
| `class_name_map` | `{}` | Map Roboflow names to canonical names |

## Taxonomy Alignment

### The Risk

If your Roboflow model uses different class names than your runtime expects, you'll get:
- Wrong SOP mappings
- Incorrect severity calculations
- Broken RCA explanations

### Solution: Class Name Mapping

Add mappings in `station.yaml`:

```yaml
class_name_map:
  # Roboflow class : Canonical class
  "scratch": "blade_scratch"
  "edge_defect": "edge_burr"
  "dent_surface": "surface_dent"
```

### Verifying Alignment

```bash
# Run taxonomy test
python -m pytest tests/test_taxonomy.py -v
```

## Swapping to Local TensorRT

Once your local model is trained:

### 1. Build TensorRT Engine on Jetson

```bash
# Copy ONNX from training machine
scp v1-nano-real.onnx jetson:/opt/intelfactor/models/

# Build engine
/usr/src/tensorrt/bin/trtexec \
  --onnx=/opt/intelfactor/models/v1-nano-real.onnx \
  --saveEngine=/opt/intelfactor/models/v1-nano-real_fp16.engine \
  --fp16 --workspace=4096
```

### 2. Update Configuration

Edit `/opt/intelfactor/config/station.yaml`:

```yaml
# Change this:
vision_model: roboflow_hosted

# To this:
vision_model: yolov8n_trt

# Add model bundle config:
model_bundle:
  path: "/opt/intelfactor/models"
  engine: "v1-nano-real_fp16.engine"
```

### 3. Restart Station

```bash
# Restart the inference service
# The rest of the system (evidence, UI, sync) continues unchanged
```

## API Latency Expectations

| Scenario | Typical Latency |
|----------|-----------------|
| Roboflow hosted (US) | 300-800ms |
| Roboflow hosted (Asia) | 500-1500ms |
| Local TensorRT | 15-30ms |

**Note**: Roboflow latency is acceptable for development but not for production line speeds.

## Troubleshooting

### "Roboflow API request failed"

Check:
1. API key is set: `echo $ROBOFLOW_API_KEY`
2. Workspace/project names are correct
3. Model version exists and is published
4. Network connectivity to `detect.roboflow.com`

### Wrong class names in UI

Add `class_name_map` entries in `station.yaml` to map Roboflow class names to canonical names.

### High latency

Expected for API calls. For production, switch to local TensorRT.

## Cost Considerations

Roboflow hosted inference:
- Free tier: Limited requests
- Paid plans: Per-inference pricing
- For high-volume production, local TensorRT is cheaper

## Security Notes

- Store API key in environment variable, not in config file
- Roboflow API key grants access to your models — treat as sensitive
- API calls are HTTPS encrypted

## Migration Checklist

When ready to switch to local TensorRT:

- [ ] Train YOLOv8n model on Azure/RunPod
- [ ] Export to ONNX
- [ ] Build TensorRT engine on Jetson
- [ ] Update `vision_model` to `yolov8n_trt`
- [ ] Add `model_bundle` config
- [ ] Verify detections match Roboflow output
- [ ] Remove `roboflow_*` config entries
- [ ] Unset `ROBOFLOW_API_KEY` environment variable

## Files Changed

| File | Purpose |
|------|---------|
| `packages/inference/providers/vision_roboflow.py` | New provider implementation |
| `packages/inference/providers/resolver.py` | Added roboflow_hosted resolution |
| `configs/station-roboflow.yaml` | Example configuration |
| `docs/ROBOFLOW_INTEGRATION.md` | This documentation |
