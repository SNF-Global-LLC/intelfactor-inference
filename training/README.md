# IntelFactor Training Pipeline

YOLO-based metal defect detection model training — from dataset download to
Jetson TensorRT engine.

---

## Folder Structure

```
training/
├── README.md
├── requirements.txt
├── config/
│   ├── train_config.yaml      # Hyperparameters and Ultralytics settings
│   ├── dataset_config.yaml    # Dataset sources (Roboflow, Kaggle, custom)
│   └── augmentations.yaml     # Albumentations pipeline for metal imagery
├── datasets/
│   ├── raw/                   # Downloaded source datasets (gitignored)
│   │   ├── roboflow/          # Auto-populated by download_datasets.py
│   │   ├── kaggle/            # Manual ZIPs placed here
│   │   └── custom/            # Any other YOLO-format datasets
│   └── combined/              # Merged + re-split dataset (gitignored)
│       ├── train/images/
│       ├── train/labels/
│       ├── val/images/
│       ├── val/labels/
│       ├── test/images/
│       ├── test/labels/
│       └── data.yaml          # Written by merge_datasets.py
├── runs/                      # Training output (gitignored)
├── exports/                   # ONNX exports (gitignored)
├── scripts/
│   ├── download_datasets.py
│   ├── merge_datasets.py
│   ├── add_hard_negatives.py
│   ├── analyze_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune_threshold.py
│   └── export_model.py
└── utils/
    ├── paths.py               # Repo-relative path constants
    ├── taxonomy.py            # Defect class definitions (single source of truth)
    ├── yaml_io.py             # YAML helpers
    └── dataset_checks.py     # YOLO label integrity checks
```

---

## Canonical 13-Class Taxonomy

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | scratch | Surface scratches, linear damage |
| 1 | grinding_mark | Aggressive abrasive lines from manufacturing |
| 2 | dent | Pits, dents, surface indentations |
| 3 | weld_defect | Welding seam issues, inclusions |
| 4 | edge_issue | Edge geometry problems |
| 5 | overgrind | Excessive material removal at heel |
| 6 | chip | Edge chipping, micro-chipping |
| 7 | burr | Residual sharpening burrs |
| 8 | crack | Structural cracks in blade |
| 9 | surface_roughness | Unfinished/poor surface finish |
| 10 | contamination | Foreign material on surface |
| 11 | lighting_artifact | False positive from glare/reflection |
| 12 | normal_texture | Valid texture (tsuchime, satin, damascus) |

Source dataset class names are remapped to these via `taxonomy.ALIASES`.
Add new aliases in `utils/taxonomy.py` — do not rename existing classes
without also updating `configs/wiko_taxonomy.yaml` in the main repo.

---

## Local Setup

```bash
# From repo root
cd training

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

No GPU is required for local smoke tests (use `device=cpu`).

---

## Expected Command Sequence

### Phase 4: Dataset Setup

```bash
# Download from Roboflow (set your API key first)
export ROBOFLOW_API_KEY=rf_your_key_here
python scripts/download_datasets.py

# Merge all sources into combined/
python scripts/merge_datasets.py

# Add hard negatives (defect-free images to reduce false positives)
python scripts/add_hard_negatives.py --neg-dir datasets/raw/negatives

# Validate dataset health before training
python scripts/analyze_dataset.py --data datasets/combined/data.yaml
```

### Phase 5: Training

```bash
# Run 1: Sanity check — fast, catch data issues early
# If mAP50 < 0.30 after 30 epochs, fix data before scaling up
python scripts/train.py \
  --name v1-nano-sanity \
  --override model=yolov8n.pt epochs=30 batch=32

# Run 2: Main training candidate
python scripts/train.py \
  --name v1-medium-main \
  --override model=yolov8m.pt

# Evaluate on test split
python scripts/evaluate.py \
  --model runs/v1-medium-main/weights/best.pt \
  --name v1-medium-eval

# Tune per-class confidence thresholds (target 85% recall)
python scripts/tune_threshold.py \
  --model runs/v1-medium-main/weights/best.pt \
  --min-recall 0.85
```

### Phase 6: Export

```bash
# Export ONNX from best checkpoint
python scripts/export_model.py \
  --model runs/v1-medium-main/weights/best.pt \
  --output exports/yolov8m-metal-v1.onnx
```

### Local smoke test (no GPU, no real dataset)

```bash
# Runs 1 epoch on dummy data — just validates the pipeline runs
python scripts/train.py \
  --name smoke-test \
  --override model=yolov8n.pt epochs=1 batch=2 device=cpu imgsz=320
```

---

## Azure VM Usage Notes

See the Azure GPU Training Setup runbook in the project wiki.

Key points:
- Provision the VM **after** the training pipeline is ready (this repo must be committed first)
- Clone this repo on the VM: `git clone https://github.com/<org>/intelfactor-inference.git`
- Run setup: `cd intelfactor-inference/training && pip install -r requirements.txt`
- Use `auto-shutdown` to avoid idle GPU charges
- Deallocate the VM when not training: `az vm deallocate --resource-group intelfactor-ml --name if-train-01`
- Do not commit `datasets/`, `runs/`, or `exports/` — they are gitignored

---

## Jetson ONNX → TensorRT Conversion

TRT engines are device-specific — always build on the **target Jetson**, never cross-compile.

```bash
# 1. SCP the ONNX from Azure VM (or local) to Jetson
scp exports/yolov8m-metal-v1.onnx tony@<JETSON_IP>:/opt/intelfactor/models/

# 2. SSH to Jetson
ssh tony@<JETSON_IP>

# 3. Convert (FP16 recommended for Orin Nano)
/usr/src/tensorrt/bin/trtexec \
  --onnx=/opt/intelfactor/models/yolov8m-metal-v1.onnx \
  --saveEngine=/opt/intelfactor/models/yolov8m-metal-v1_fp16.engine \
  --fp16 --workspace=4096

# 4. Benchmark — SLA is < 25ms per frame
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=/opt/intelfactor/models/yolov8m-metal-v1_fp16.engine --batch=1

# 5. Validate with make doctor
cd ~/intelfactor-inference
make doctor
```

Alternatively use the repo's build script:

```bash
make build-trt MODEL=yolov8m-metal-v1.onnx PRECISION=fp16
```

---

## Phase 7: Upload Artifacts

```bash
aws s3 cp exports/yolov8m-metal-v1.onnx s3://intelfactor-models/metal-defect/v1/
aws s3 cp runs/v1-medium-main/weights/best.pt s3://intelfactor-models/metal-defect/v1/
aws s3 cp runs/v1-medium-main/evaluation/confidence_thresholds.yaml s3://intelfactor-models/metal-defect/v1/
```

---

## Gitignore

The following are excluded from version control (add to `.gitignore` if not present):

```
training/datasets/raw/
training/datasets/combined/
training/runs/
training/exports/
training/logs/
training/.venv/
```
