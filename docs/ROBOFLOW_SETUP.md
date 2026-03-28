# Roboflow Hosted Inference Setup

## Quick Start

```bash
# Set API key (add to ~/.zshrc)
export ROBOFLOW_API_KEY="your_key_here"

# Test with real image
python scripts/test_roboflow_provider.py \
  --model-id "metal-surface-defects-rmbhy-szb6t/1" \
  --image your_image.jpg
```

## Configuration

Add to `station.yaml`:

```yaml
vision_model: roboflow_hosted
roboflow_api_key: ${ROBOFLOW_API_KEY}
roboflow_model_id: "metal-surface-defects-rmbhy-szb6t/1"
```

## Model IDs

Current working model:
- `metal-surface-defects-rmbhy-szb6t/1` (metal surface defects)

## Migration to Local TensorRT

When local model is ready:

```yaml
# Change this:
vision_model: roboflow_hosted

# To this:
vision_model: yolov8n_trt
model_bundle:
  path: "/opt/intelfactor/models"
  engine: "v1-nano-real_fp16.engine"
```

## Files

- `packages/inference/providers/vision_roboflow.py` - Provider implementation
- `configs/station-roboflow.yaml` - Example configuration
- `scripts/test_roboflow_provider.py` - Test utility
- `docs/ROBOFLOW_INTEGRATION.md` - Full integration guide

## Status

✅ Provider tested and working
✅ API latency: ~3-4 seconds
⚠️  Temporary solution - migrate to TensorRT for production
