# Hard Negative Collection Guide

**Purpose**: Build a hard-negative training set to fix the 33% false positive rate on clean knives.

**Target**: Reduce clean-blade FP rate from 33% to <10% (≤1 FP per 12 clean images).

---

## Source: Kyoto Evaluation Failures

These specific clean images triggered false detections:

### 🔴 Critical Priority (High-Confidence FPs)

| Image | Finish Type | Failed Configs | Predicted As | Confidence | Collection Action |
|-------|-------------|----------------|--------------|------------|-------------------|
| `clean_glare_001` | Lighting glare | A, B, C | Scratches | 0.85 | **Collect 20+ glare cases** |
| `clean_tsuchime_001` | Hammered/Tsuchime | A, B, C | Scratches | 0.77 | **Collect 30+ hammered finishes** |
| `clean_weld_001` | Normal weld seam | A, B, C | Inclusion | 0.79 | **Collect 20+ clean welds** |
| `clean_grind_001` | Grind pattern | A, B, C | Inclusion | 0.75 | **Collect 30+ grind patterns** |

### 🟡 Medium Priority (Lower-Confidence FPs)

| Image | Finish Type | Failed Configs | Predicted As | Confidence | Collection Action |
|-------|-------------|----------------|--------------|------------|-------------------|
| `clean_grind_002` | Grind pattern | A, B | Scratches | 0.58 | Include in grind set |
| `clean_tsuchime_003` | Hammered | A | Inclusion | 0.59 | Include in hammered set |
| `clean_satin_002` | Satin | A, B | Inclusion | 0.55 | Include in satin set |

### 🟢 Passing (Use as Negative Examples)

| Image | Finish Type | Why It Passed | Value |
|-------|-------------|---------------|-------|
| `clean_satin_001` | Satin | No detection | Positive example of "clean" |
| `clean_satin_003` | Satin | No detection | Positive example of "clean" |
| `clean_tsuchime_002` | Hammered | No detection | Texture variation that works |
| `clean_glare_002` | Lighting | No detection | Glare case that passes |
| `clean_weld_002` | Weld seam | No detection | Weld case that passes |

---

## Collection Targets

### Category 1: Hammered/Tsuchime Finishes (30+ images)
**Why**: All 3 hammered samples triggered FPs. Model cannot distinguish texture from scratches.

**Sources**:
- Shun Premier series knives
- Masakage Koishi/Yuki lines
- Any Damascus with hammered texture
- Kitchen supply websites (Korin, Chef's Knives To Go)

**Collection criteria**:
- [ ] Clean, no visible defects
- [ ] Clear hammered/tsuchime pattern
- [ ] Various lighting angles
- [ ] Different manufacturers
- [ ] Both blade faces and spine views

**Label**: `negative_hammered_###.jpg` + empty `.txt` file

---

### Category 2: Satin Finishes (20+ images)
**Why**: 1/3 satin samples triggered FP. Basic finish should never alarm.

**Sources**:
- Wüsthof Classic
- Zwilling Pro
- Victorinox Fibrox
- Mercer Culinary

**Collection criteria**:
- [ ] Fine satin grind pattern
- [ ] No visible scratches
- [ ] Various grit finishes (120-400 grit equivalent)
- [ ] Different steel types (German, Japanese)

**Label**: `negative_satin_###.jpg` + empty `.txt` file

---

### Category 3: Normal Grind Patterns (30+ images)
**Why**: 2/2 grind pattern samples triggered FPs. Manufacturing marks ≠ defects.

**Sources**:
- Knife sharpening service before/after photos
- Factory edge grind lines
- Spine and choil areas
- Bolster transitions

**Collection criteria**:
- [ ] Visible grind lines from manufacturing
- [ ] Even, intentional pattern
- [ ] Not scratches from damage
- [ ] Various angles (flat, convex, hollow)

**Label**: `negative_grind_###.jpg` + empty `.txt` file

---

### Category 4: Clean Weld Seams (20+ images)
**Why**: 1/2 weld samples triggered FPs. Damascus weld lines are normal.

**Sources**:
- Damascus pattern knives (normal seam)
- San-mai/3-layer construction
- VG-10 core with soft iron cladding

**Collection criteria**:
- [ ] Straight, even weld seam
- [ ] No gaps or cracks
- [ ] Clean pattern transition
- [ ] Both sides of blade

**Label**: `negative_weld_###.jpg` + empty `.txt` file

---

### Category 5: Lighting/Specular Cases (20+ images)
**Why**: Glare at 0.85 confidence is dangerous. Model reacts to photons, not defects.

**Sources**:
- Same knives shot with different lighting
- Direct reflection cases
- Rim lighting on edges
- Overhead vs side lighting

**Collection criteria**:
- [ ] Specular highlights
- [ ] Reflection of light sources
- [ ] Gradients from curved surfaces
- [ ] NOT actual scratches or damage

**Label**: `negative_glare_###.jpg` + empty `.txt` file

---

### Category 6: Diverse Clean Backgrounds (50+ images)
**Why**: Need negative variety to prevent overfitting.

**Collection criteria**:
- [ ] Different cutting board surfaces
- [ ] Various table materials
- [ ] Industrial backgrounds
- [ ] Different color temperatures

**Label**: `negative_bg_###.jpg` + empty `.txt` file

---

## Total Collection Target

| Category | Target | Priority |
|----------|--------|----------|
| Hammered/Tsuchime | 30 | 🔴 Critical |
| Satin Finishes | 20 | 🟡 Medium |
| Grind Patterns | 30 | 🔴 Critical |
| Clean Weld Seams | 20 | 🔴 Critical |
| Lighting/Glare | 20 | 🟡 Medium |
| Background Variety | 50 | 🟢 Low |
| **Total** | **170** | **Minimum viable** |

---

## Labeling Format

All hard negatives get **empty label files**:

```
datasets/hard_negatives/
├── images/
│   ├── negative_hammered_001.jpg
│   ├── negative_hammered_002.jpg
│   └── ...
└── labels/
    ├── negative_hammered_001.txt    # Empty file
    ├── negative_hammered_002.txt    # Empty file
    └── ...
```

Empty `.txt` = no objects = negative example in YOLO training.

---

## Quality Checklist

Before adding to training set:

- [ ] **Human verified clean**: No actual defects visible
- [ ] **Matches failure mode**: Addresses specific FP category
- [ ] **Diverse sources**: Not all from same knife/manufacturer
- [ ] **Good image quality**: >600px, not blurry
- [ ] **Consistent naming**: Follows `negative_{category}_{###}.jpg` pattern
- [ ] **Empty label file**: Present and truly empty

---

## Integration with Training

### Merge Order
1. Existing Wiko training data
2. NEU-DET (with class remapping)
3. **Hard negatives** (this collection)
4. Kyoto PDF positives (defects)

### Class Balance Target
After adding hard negatives:

| Class | Current | Target | Ratio |
|-------|---------|--------|-------|
| blade_scratch | ~40% | ~25% | Reduce dominance |
| inclusion | ~15% | ~10% | Reduce |
| Other defects | ~30% | ~35% | Increase variety |
| **Negative/Background** | **~15%** | **~30%** | **Critical increase** |

---

## Validation Plan

After retraining with hard negatives:

1. **Re-run identical 20-image Kyoto eval**
   - Same images, same configs (A/B/C)
   - Measure FP rate improvement
   - Target: ≤1 FP on 12 clean images

2. **Extended clean test**
   - 50 clean knives (all finish types)
   - Measure precision at 0.5 threshold
   - Target: >90% precision

3. **Defect recall check**
   - Ensure adding negatives doesn't hurt recall
   - Same 8 positive Kyoto images
   - Target: Maintain ≥6/8 detected

---

## Files and Tracking

- Collection spreadsheet: `docs/hard_negative_collection_tracker.csv`
- Image storage: `datasets/hard_negatives/`
- Source attribution: Note manufacturer/source for each image
- Verification log: Who verified each image as "clean"

---

## Expected Impact

Based on literature and similar projects:

| Metric | Before | After Hard Negatives |
|--------|--------|---------------------|
| Clean FP Rate | 33% (4/12) | 5-10% (target: ≤1/12) |
| Precision | 58% | 85%+ |
| Texture Confusion | 12 cases | 2-3 cases |

**Confidence**: High. Hard negatives are a proven technique for exactly this failure mode.
