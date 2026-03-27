# Roboflow Hosted Vision Provider — Implementation Summary

## What Was Implemented

### 1. New Provider: `RoboflowHostedVisionProvider`

**File**: `packages/inference/providers/vision_roboflow.py`

A temporary vision inference provider that:
- Sends captured frames to Roboflow API
- Returns normalized `DetectionResult` objects
- Matches the same interface as `TensorRTVisionProvider`
- Supports class name mapping for taxonomy alignment

### 2. Resolver Integration

**File**: `packages/inference/providers/resolver.py`

Updated to:
- Import `RoboflowHostedVisionProvider`
- Add `roboflow_hosted` to `VISION_MODELS` catalog
- Handle `vision_model: roboflow_hosted` config

### 3. Example Configuration

**File**: `configs/station-roboflow.yaml`

Complete station configuration showing:
- Roboflow API credentials setup
- Class name mapping examples
- Canonical defect classes list
- All required thresholds

### 4. Documentation

**File**: `docs/ROBOFLOW_INTEGRATION.md`

Complete integration guide including:
- Quick start instructions
- Taxonomy alignment guidance
- Migration path to local TensorRT
- Troubleshooting

### 5. Test Script

**File**: `scripts/test_roboflow_provider.py`

Stand-alone test script to verify the provider works before full deployment.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  Local Capture  │────▶│ RoboflowHosted      │────▶│ Local Evidence  │
│  (Jetson/PC)    │     │ VisionProvider      │     │ Write + SQLite  │
└─────────────────┘     └─────────────────────┘     └─────────────────┘
                               │                             │
                               │ API Call                    │
                               ▼                             │
                        ┌──────────────┐                     │
                        │ Roboflow API │                     │
                        │ (Hosted)     │                     │
                        └──────────────┘                     │
                                                             │
                              ┌─────────────────┐           │
                              │ Local UI Render │◀──────────┘
                              │ + Sync          │
                              └─────────────────┘
```

**Key**: Only inference is hosted. All other operations remain local.

---

## How to Test

### Option 1: Synthetic Image Test (No file needed)

```bash
# Set your API key
export ROBOFLOW_API_KEY="your_key_here"

# Run test with synthetic image
python scripts/test_roboflow_provider.py \
  --workspace kmg-kyoto \
  --project cutlery-defects \
  --version 3
```

### Option 2: Test with Real Image

```bash
# Set your API key
export ROBOFLOW_API_KEY="your_key_here"

# Test with an actual image
python scripts/test_roboflow_provider.py \
  --image path/to/test/image.jpg \
  --workspace kmg-kyoto \
  --project cutlery-defects \
  --version 3
```

### Option 3: Full Station Test

```bash
# 1. Copy config
cp configs/station-roboflow.yaml /opt/intelfactor/config/station.yaml

# 2. Edit to set your API key and project details
nano /opt/intelfactor/config/station.yaml

# 3. Start station
python -m packages.inference.api_v2

# 4. Open UI and test /inspect endpoint
```

---

## Configuration Required

### Minimum Config (`station.yaml`)

```yaml
vision_model: roboflow_hosted

# Roboflow credentials
roboflow_api_key: ${ROBOFLOW_API_KEY}  # Use env var
roboflow_workspace: your-workspace
roboflow_project: your-project
roboflow_version: 3

# Taxonomy (must match or map to your Roboflow model)
defect_classes:
  - blade_scratch
  - grinding_mark
  # ... etc
```

### Environment Variable

```bash
export ROBOFLOW_API_KEY="rf_xxxxxxxxxxxx"
```

---

## Taxonomy Alignment

### Risk

If Roboflow class names differ from runtime canonical names:
- Wrong SOP mappings
- Incorrect severity calculations
- Broken RCA

### Solution

Use `class_name_map` in config:

```yaml
class_name_map:
  "scratch": "blade_scratch"
  "edge_defect": "edge_burr"
  "dent_surface": "surface_dent"
```

---

## Migration to Local TensorRT

When your local model is ready:

```bash
# 1. Train and export model (on Azure/RunPod)
# 2. Build TensorRT engine on Jetson
# 3. Update config:
```

```yaml
# Change this
vision_model: roboflow_hosted

# To this
vision_model: yolov8n_trt

# Remove Roboflow config
# roboflow_api_key: ...
# roboflow_workspace: ...

# Add local model
model_bundle:
  path: "/opt/intelfactor/models"
  engine: "v1-nano-real_fp16.engine"
```

---

## Files Changed

| File | Changes |
|------|---------|
| `packages/inference/providers/vision_roboflow.py` | **NEW** — Roboflow provider implementation |
| `packages/inference/providers/resolver.py` | Added roboflow_hosted support |
| `configs/station-roboflow.yaml` | **NEW** — Example configuration |
| `docs/ROBOFLOW_INTEGRATION.md` | **NEW** — Integration documentation |
| `scripts/test_roboflow_provider.py` | **NEW** — Test script |

---

## Next Steps

1. **Get Roboflow credentials** from https://app.roboflow.com
2. **Test the provider**: `python scripts/test_roboflow_provider.py`
3. **Deploy to station**: Copy config and start inference service
4. **Verify taxonomy**: Check class names align with SOP mappings
5. **Train local model**: While using Roboflow for real predictions
6. **Swap to TensorRT**: When local model is ready

---

## Status

✅ Provider implemented
✅ Resolver integration complete
✅ Configuration examples ready
✅ Documentation complete
✅ Test script ready

**Ready to test with your Roboflow credentials.**
