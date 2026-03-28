#!/bin/bash
# Complete RunPod deployment script

set -e

echo "=== IntelFactor RunPod Deployment ==="
echo ""

# Check if RUNPOD_IP is provided
if [ -z "$1" ]; then
    echo "Usage: bash deploy_runpod.sh <RUNPOD_IP>"
    echo ""
    echo "Get the IP from RunPod console after deploying PyTorch template"
    exit 1
fi

RUNPOD_IP=$1
echo "Target: root@$RUNPOD_IP"
echo ""

# Step 1: Upload dataset
echo "Step 1/4: Uploading dataset..."
rsync -avz --progress combined.tar.gz root@$RUNPOD_IP:/workspace/ || {
    echo "Failed to upload. Check:"
    echo "  1. RunPod pod is running"
    echo "  2. SSH key is added to RunPod"
    echo "  3. IP address is correct"
    exit 1
}

# Step 2: Upload training scripts
echo ""
echo "Step 2/4: Uploading training scripts..."
rsync -avz --progress training_package/training/ root@$RUNPOD_IP:/workspace/training/

# Step 3: Setup environment and start training
echo ""
echo "Step 3/4: Setting up environment..."
ssh root@$RUNPOD_IP << 'REMOTE'
set -e
cd /workspace

# Extract dataset
echo "Extracting dataset..."
tar -xzf combined.tar.gz

# Fix paths
sed -i 's|/home/azureuser|/workspace|g' training/datasets/combined/data.yaml

# Install dependencies
echo "Installing dependencies..."
pip install -q ultralytics onnx onnxsim

# Verify
echo "Verifying setup..."
nvidia-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "Setup complete!"
REMOTE

# Step 4: Start training
echo ""
echo "Step 4/4: Starting training..."
ssh root@$RUNPOD_IP << 'REMOTE'
cd /workspace/training

# Start training in tmux session
tmux new -d -s train "python scripts/train.py --name v1-nano-real --override model=yolov8n.pt epochs=100 imgsz=640 device=0 batch=32 2>&1 | tee train.log"

echo ""
echo "Training started in tmux session 'train'"
echo ""
echo "To monitor:"
echo "  ssh root@$RUNPOD_IP"
echo "  tmux attach -t train"
echo ""
echo "To check progress:"
echo "  tail -f /workspace/training/runs/v1-nano-real/results.csv"
REMOTE

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor training:"
echo "  ssh root@$RUNPOD_IP 'tmux attach -t train'"
echo ""
echo "After training completes (~2-3 hours), export ONNX:"
echo "  ssh root@$RUNPOD_IP 'cd /workspace/training && python scripts/export_model.py --model runs/v1-nano-real/weights/best.pt --output exports/v1-nano-real.onnx --imgsz 640'"
echo ""
echo "Download ONNX:"
echo "  rsync root@$RUNPOD_IP:/workspace/training/exports/v1-nano-real.onnx ./"
