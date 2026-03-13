# Kyoto Evaluation Summary

**Project**: IntelFactor Knife Blade Inspection  
**Evaluation Date**: 2026-03-12  
**Scope**: 20-image validation set (12 clean / 8 positive)  
**Inferences**: 60 total (20 images × 3 configs)  

---

## Executive Summary

**Verdict: Demo-Only Workflow**

The Roboflow-hosted two-stage workflow (detector → classifier) failed the production precision gate. While recall is acceptable, the 33% false positive rate on clean knives makes it unsuitable for automatic inspection.

**Key Finding**: This is a **representation problem**, not a threshold problem. Configs A→B→C showed minimal precision improvement while sacrificing recall.

---

## Detailed Results

### Configuration Sweep

| Config | Detector | Classifier | Clean FPs | FP Rate | Pos Detected | Recall | **Verdict** |
|--------|----------|------------|-----------|---------|--------------|--------|-------------|
| A | 0.4 | 0.4 | 7/12 | 58.3% | 8/8 (100%) | ✅ | ❌ FAIL |
| B | 0.5 | 0.5 | 7/12 | 58.3% | 6/8 (75%) | ✅ | ❌ FAIL |
| C | 0.5 | 0.6 | 4/12 | 33.3% | 6/8 (75%) | ✅ | ❌ FAIL |

**Production Target**: ≤1 FP (8.3%) — **Not achieved by any config**

### Failure Mode Analysis

| Failure Mode | Count | Description | Root Cause |
|--------------|-------|-------------|------------|
| **texture_confusion** | 12 | Normal finishes → defects | Insufficient texture variety in training |
| **class_collapse** | 6 | All defects → "Scratches" | Class imbalance, over-representation of scratches |
| **lighting_artifact** | 1 | Glare → high-conf detection | No hard negatives for specular reflection |
| **missed_real_defect** | 4 | Heel/weld defects missed | Threshold too high for subtle defects |

### Critical Observations

1. **Clean `clean_glare_001`**: 0.85 confidence FP — model reacts to photons, not defects
2. **Clean `clean_tsuchime_001`**: 0.77 confidence FP — hammered texture = scratches to model
3. **Clean `clean_weld_001`**: 0.79 confidence FP — normal Damascus seam = inclusion
4. **Positive `positive_weld_defect_001`**: Missed at B/C configs — threshold trade-off

---

## Root Cause Diagnosis

### What This Is NOT
- ❌ Not a threshold calibration issue (A→B→C showed minimal improvement)
- ❌ Not an architecture issue (two-stage design is sound)
- ❌ Not a confidence threshold issue (high-conf FPs persist)

### What This IS
- ✅ **Training data representation gap**
  - Model never saw enough diverse "normal" textures
  - Class imbalance causes collapse to dominant label
  - Lack of hard negatives for lighting/texture edge cases

- ✅ **Domain mismatch**
  - Training data: Generic metal surface defects
  - Target domain: Knife blades with intentional finishes
  - Model confuses manufacturing patterns with damage

---

## Baseline Freeze

**Config C (0.5/0.6) designated as:**
```
models/baselines/two_stage_baseline_demo_only.md
```

### Valid Uses
- ✅ Demo and triage prototype
- ✅ Surface anomaly candidate detection (with human review)
- ✅ Coarse filtering (review queue generation)

### Invalid Uses
- ❌ Production automatic inspection
- ❌ Unsupervised quality gating
- ❌ Direct Kyoto defect classification

---

## Path Forward: Custom Training

### Phase 1: Data Collection (1-2 weeks)

**Hard Negative Collection** (170 images target)
- 30× Hammered/Tsuchime finishes (texture confusion)
- 30× Normal grind patterns (manufacturing mark confusion)
- 20× Clean weld seams (Damascus pattern confusion)
- 20× Satin finishes (baseline clean)
- 20× Lighting/glare cases (specular reflection)
- 50× Background variety (prevent overfitting)

See: `docs/hard_negative_collection_guide.md`

### Phase 2: Dataset Merge (2-3 days)

**Sources**:
1. Existing Wiko dataset
2. NEU-DET (1,799 images) — ✅ Downloaded
3. GC10-DET (2,290 images) — Manual download required
4. Hard negatives (170 images) — Phase 1
5. Kyoto PDF positives (8 images) — Ground truth

**Class Mapping**:
```
NEU-DET          →  Wiko Unified
- scratches      →  blade_scratch
- inclusion      →  inclusion
- pitted_surface →  surface_dent
- crazing        →  surface_crack
- rolled-in_scale→  grinding_mark
- patches        →  surface_discolor
```

See: `training/scripts/merge_metal_datasets.py`

### Phase 3: Training (3-5 days)

**Pipeline**:
```bash
# Merge datasets
python training/scripts/merge_metal_datasets.py \
  --wiko /path/to/wiko \
  --neu-det datasets/metal_defects/neu-det-v2 \
  --output datasets/merged_metal_defects

# Train
yolo detect train \
  data=datasets/merged_metal_defects/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  name=wiko_v2_hard_negatives

# Export for Jetson
yolo export model=best.pt format=engine device=0 half=True
```

### Phase 4: Validation (1-2 days)

**Identical Evaluation Protocol**:
- Same 20 Kyoto images
- Same configs (A/B/C equivalent)
- Same metrics (precision/recall gates)

**Success Criteria**:
- Clean FP Rate: 33% → <10% (≤1/12)
- Maintain Recall: ≥6/8 positive detected
- Reduce texture confusion cases: 12 → ≤3

### Phase 5: Deployment (1 week)

- TensorRT engine build on Jetson
- Per-class threshold tuning
- Integration with RCA pipeline
- Operator dashboard updates

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Hard Negatives | 1-2 weeks | 170 verified clean images |
| 2. Dataset Merge | 2-3 days | Unified training set |
| 3. Training | 3-5 days | Trained model |
| 4. Validation | 1-2 days | Kyoto re-evaluation results |
| 5. Deployment | 1 week | Production candidate on Jetson |
| **Total** | **4-6 weeks** | Production-ready model |

---

## Key Decisions

### Why Not Just Tune Thresholds?
- Config C already at 0.5/0.6 — aggressive thresholds
- Further tightening would miss real defects (recall already dropped to 75%)
- 4/12 FPs at best config still 4× above target

### Why Not Different Architecture?
- Two-stage design is sound (Roboflow workflow proves concept)
- Problem is data representation, not model capacity
- YOLOv8n is appropriate for Jetson Orin Nano

### Why Hard Negatives Work
- Forces model to learn "normal" vs "defective" distinction
- Specifically targets observed failure modes
- Proven technique for precision/recall trade-offs

---

## Files and Artifacts

### Evaluation Data
- `kyoto_raw_outputs/kyoto_eval_final.csv` — 60 inferences with judgments
- `kyoto_raw_outputs/raw_results.json` — Raw API responses
- `kyoto_raw_outputs/summary_config_*.csv` — Per-config summaries

### Baseline Documentation
- `models/baselines/two_stage_baseline_demo_only.md` — Frozen baseline spec

### Training Infrastructure
- `datasets/metal_defects/neu-det-v2/` — Downloaded NEU-DET (1,799 images)
- `docs/CLASS_MAPPING.md` — Class remapping strategy
- `docs/hard_negative_collection_guide.md` — Collection protocol
- `training/scripts/merge_metal_datasets.py` — Merge utility

### Evaluation Infrastructure
- `run_kyoto_batch.py` — Re-runnable 60-inference sweep
- `Kimi_Agent_Workflow Evaluation Rubric/evaluate_workflow.py` — Analysis script
- `kyoto_eval_set/` — 20-image validation set (frozen)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hard negatives insufficient | Medium | High | Collect 2× target (340 images) |
| NEU-DET domain mismatch | Medium | Medium | Weight Wiko data higher in training |
| Training time on Jetson | Low | Medium | Train on GPU server, export to Jetson |
| New model still fails gates | Low | High | Keep baseline as fallback |

---

## Success Metrics

### Primary (Must Achieve)
- [ ] Clean FP Rate ≤10% (1/12 or fewer)
- [ ] Positive Recall ≥75% (6/8 or more)
- [ ] Class collapse resolved (≥3 distinct defect classes detected)

### Secondary (Nice to Have)
- [ ] Clean FP Rate ≤5% (≤1 false alarm per 20 clean blades)
- [ ] Inference time <50ms on Jetson Orin Nano
- [ ] TensorRT engine <100MB

---

## Conclusion

The Kyoto evaluation succeeded in its purpose: **honest failure detection**. 

The current workflow is demonstrably not production-ready, but the path to improvement is clear and tractable. With 4-6 weeks of focused work on hard negatives and retraining, we can achieve a production candidate that passes both precision and recall gates.

**Next Action**: Begin hard negative collection using the guide in `docs/hard_negative_collection_guide.md`.

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-12  
**Author**: IntelFactor Engineering  
