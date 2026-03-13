# Model Bundle Integration Summary

**Date**: 2026-03-12  
**Status**: Phase 1 Complete (intelfactor-inference)  

---

## Overview

Integrated retrained knife-inspection model into production stack using standard model bundle contract:

```
model_bundle/
  model.engine       # TensorRT engine
  labels.json        # Class ID → name mapping  
  thresholds.yaml    # Per-class confidence thresholds
  metadata.json      # Model version, info, etc.
```

---

## Files Changed

### 1. New Files

| File | Purpose |
|------|---------|
| `packages/inference/utils/model_bundle.py` | Bundle loader/validator utilities |
| `tests/test_model_bundle.py` | Unit tests for bundle loading |

### 2. Modified Files

| File | Changes |
|------|---------|
| `configs/station.yaml` | Added `model_bundle:` configuration section |
| `packages/inference/providers/resolver.py` | Load bundle in resolver, pass to provider |
| `packages/inference/providers/vision_trt.py` | Apply per-class thresholds, include metadata in output |
| `packages/inference/schemas.py` | Added `threshold_used`, `model_version` to Detection; added `model_name` to DetectionResult |

---

## Configuration (configs/station.yaml)

```yaml
model_bundle:
  path: "models/kyoto-yolo26n-v1"
  engine: "model.engine"
  labels: "labels.json"
  thresholds: "thresholds.yaml"
  metadata: "metadata.json"
  validate_on_load: true
  strict_taxonomy: true
```

---

## Behavior Changes

### 1. Model Loading

**Before**: Legacy path resolution only
```python
engine_path = self._find_engine_path(model_def["name"])
```

**After**: Bundle-aware loading with validation
```python
if bundle_config and bundle_config.get("path"):
    bundle = self._load_model_bundle(bundle_config)
    engine_path = bundle["engine_path"]
    # Validate labels, thresholds, metadata consistency
```

### 2. Class Name Resolution

**Before**: Config-only `defect_classes` list
```python
self.defect_classes = config.get("defect_classes", [])
```

**After**: Bundle labels take priority
```python
self.labels = config.get("labels", {})
self.defect_classes = self._resolve_defect_classes(config)
# Priority: bundle labels > config defect_classes > fallback
```

### 3. Threshold Application

**Before**: Single global threshold + optional YAML path
```python
self.per_class_thresholds = self._load_per_class_thresholds(config)
# Only supported via thresholds_path or inline dict
```

**After**: Bundle thresholds integrated
```python
# Priority: bundle thresholds > inline > YAML file > global fallback
bundle_thresholds = config.get("thresholds")
if bundle_thresholds:
    return bundle_thresholds
```

### 4. Detection Output

**Before**: Basic detection info
```python
Detection(
    defect_type=defect_type,
    confidence=confidence,
    bbox=bbox,
    severity=severity,
)
```

**After**: Enriched with model context
```python
Detection(
    defect_type=defect_type,
    confidence=confidence,
    threshold_used=threshold_used,  # NEW
    model_version=model_version,    # NEW
    bbox=bbox,
    severity=severity,
)
```

### 5. Event Emission

**Before**: Limited model info
```python
DetectionResult(
    ...
    model_version=self.model_spec.model_name,
)
```

**After**: Full bundle metadata
```python
DetectionResult(
    ...
    model_version=self.model_version,  # From bundle metadata
    model_name=self.model_name,        # From bundle metadata
)
```

---

## Validation Behavior

### Fail Loudly Principle

When `validate_on_load: true` (default):
- Missing bundle files → `RuntimeError` at startup
- Inconsistent labels/thresholds → `ModelBundleError`
- Missing expected classes → Error (if `strict_taxonomy: true`)

### Validation Checks

| Check | Behavior |
|-------|----------|
| Engine file exists | Required, raises if missing |
| Labels loadable | Required, validates JSON format |
| Thresholds loadable | Required, validates YAML format |
| Metadata loadable | Required, validates JSON + required fields |
| All labels have thresholds | Required, raises if any class missing threshold |
| Taxonomy match (strict mode) | Optional, validates against canonical 13-class list |

---

## Migration Notes

### For Existing Deployments

**No breaking changes**. Legacy behavior preserved:
- If `model_bundle:` section missing → uses legacy path resolution
- If bundle loading fails and `validate_on_load: false` → falls back to legacy
- Config `defect_classes` still honored if bundle labels unavailable

### Enabling Bundle Mode

1. Create bundle directory:
```bash
mkdir -p models/kyoto-yolo26n-v1
cp best.engine models/kyoto-yolo26n-v1/model.engine
```

2. Create labels.json:
```json
{
  "0": "blade_scratch",
  "1": "grinding_mark",
  ...
}
```

3. Create thresholds.yaml:
```yaml
thresholds:
  blade_scratch: 0.45
  grinding_mark: 0.30
  ...
```

4. Create metadata.json:
```json
{
  "model_name": "kyoto-yolo26n-v1",
  "model_version": "1.0.0",
  "training_date": "2026-03-15",
  "classes": 13
}
```

5. Update station.yaml (see Configuration section)

---

## Testing

### Run Bundle Tests

```bash
pytest tests/test_model_bundle.py -v
```

### Test Bundle Validation

```python
from packages.inference.utils.model_bundle import validate_bundle

bundle = validate_bundle(
    engine_path="models/kyoto-yolo26n-v1/model.engine",
    labels_path="models/kyoto-yolo26n-v1/labels.json",
    thresholds_path="models/kyoto-yolo26n-v1/thresholds.yaml",
    metadata_path="models/kyoto-yolo26n-v1/metadata.json",
    expected_classes=["blade_scratch", "grinding_mark", ...],
)
```

---

## Next Steps (Phase 2-4)

### Phase 2: Backend Event Schema
- Update intelbase inspection event schema
- Persist `defect_class`, `confidence`, `threshold_used`, `model_version`
- Add DynamoDB fields

### Phase 3: Frontend Review UI
- Display evidence crop
- Show defect class, confidence, threshold, model version
- Add review actions (approve, reject as texture/glare/seam, relabel)

### Phase 4: Feedback Export
- Export reviewed events for retraining
- Include image path, predicted class, human decision, failure mode

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Model loading | Legacy path resolution | Bundle-aware with validation |
| Class names | Config-only | Bundle labels + config fallback |
| Thresholds | Global or YAML path | Per-class from bundle |
| Detection output | Basic info | Includes threshold_used, model_version |
| Event emission | model_spec.model_name | Bundle model_version, model_name |
| Validation | None | Strict bundle validation |
| Fallback | None | Preserved legacy behavior |

**Result**: Model bundle integration enables production-grade model versioning, per-class thresholding, and audit trail for knife inspection pipeline.
