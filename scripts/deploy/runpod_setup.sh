#!/bin/bash
# Run this on RunPod instance
set -e
cd /workspace
tar -xzf combined.tar.gz
pip install -q ultralytics onnx onnxsim
sed -i 's|/home/azureuser|/workspace|g' training/datasets/combined/data.yaml
nvidia-smi
echo "Setup complete. Starting training..."
cd training
tmux new -s train -d "python scripts/train.py --name v1-nano-real --override model=yolov8n.pt epochs=100 imgsz=640 device=0 batch=32"
echo "Training started in tmux session 'train'"
echo "Attach with: tmux attach -t train"
