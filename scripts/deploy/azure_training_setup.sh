#!/bin/bash
# Azure VM Setup Script for IntelFactor Training
# Run this on the Azure VM after creation

set -e

echo "=== IntelFactor Azure Training Setup ==="

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git htop nvtop

# Install CUDA drivers (if not already present)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535
    sudo reboot
fi

echo "GPU Info:"
nvidia-smi

# Setup project directory
mkdir -p ~/intelfactor-inference
cd ~/intelfactor-inference

# Note: You'll need to copy the training data here
# Option 1: git clone (if repo is pushed)
# Option 2: scp -r from local machine
# Option 3: azcopy from blob storage

echo "=== Setup Python Environment ==="
python3 -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics onnx onnxsim pyyaml

echo "=== Verify Installation ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "from ultralytics import YOLO; print('Ultralytics: OK')"

echo ""
echo "=== Setup Complete ==="
echo "Next: Copy training data and run training"
echo ""
echo "Training command:"
echo "  cd ~/intelfactor-inference/training"
echo "  python scripts/train.py --name v1-nano-real --override model=yolov8n.pt epochs=100 imgsz=640 device=0 batch=32"
