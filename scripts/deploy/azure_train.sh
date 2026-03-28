#!/bin/bash
# Run training on Azure VM

set -e

cd ~/intelfactor-inference/training

echo "=== Starting YOLOv8n Training ==="
echo "Dataset: $(python3 -c "import yaml; d=yaml.safe_load(open('datasets/combined/data.yaml')); print(d['path'])")"
echo "Classes: 13"
echo "Images: ~9,400 total"
echo ""

# Start training with screen/tmux to persist session
python3 scripts/train.py \
  --name v1-nano-real \
  --override model=yolov8n.pt epochs=100 imgsz=640 device=0 batch=32

echo ""
echo "=== Training Complete ==="
echo "Exporting to ONNX..."

python3 scripts/export_model.py \
  --model runs/v1-nano-real/weights/best.pt \
  --output exports/v1-nano-real.onnx \
  --imgsz 640

echo ""
echo "=== Export Complete ==="
ls -lh exports/v1-nano-real.onnx
echo ""
echo "Copy to Jetson:"
echo "  scp exports/v1-nano-real.onnx tony@<JETSON_IP>:/opt/intelfactor/models/"
