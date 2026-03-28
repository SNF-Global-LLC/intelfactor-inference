#!/bin/bash
# Deploy training to Azure VM once it's created

if [ -z "$1" ]; then
    # Try to read saved IP
    if [ -f .azure_vm_ip ]; then
        VM_IP=$(cat .azure_vm_ip)
        echo "Using saved IP: $VM_IP"
    else
        echo "Usage: bash deploy_to_azure.sh <VM_IP>"
        exit 1
    fi
else
    VM_IP=$1
fi

set -e

echo "=== Deploying to Azure VM: $VM_IP ==="
echo ""

# Wait for SSH to be ready
echo "Waiting for SSH..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 azureuser@$VM_IP "echo ready" 2>/dev/null; then
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

# Step 1: Upload dataset
echo "Step 1/5: Uploading dataset (471MB)..."
rsync -avz --progress combined.tar.gz azureuser@$VM_IP:~ || {
    echo "Upload failed"
    exit 1
}

# Step 2: Upload training code
echo ""
echo "Step 2/5: Uploading training scripts..."
rsync -avz --progress training/ azureuser@$VM_IP:~/intelfactor-inference/training/

# Step 3: Setup environment
echo ""
echo "Step 3/5: Setting up environment (this takes ~5 min)..."
ssh azureuser@$VM_IP << 'REMOTE'
set -e
cd ~

# Install dependencies
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git htop

# Install PyTorch with CUDA
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ultralytics
pip install -q ultralytics onnx onnxsim

# Extract dataset
mkdir -p intelfactor-inference/training
tar -xzf combined.tar.gz -C intelfactor-inference/training/ 2>/dev/null || true

# Fix data.yaml path
sed -i "s|/home/azureuser|/home/azureuser|g" intelfactor-inference/training/datasets/combined/data.yaml

# Verify
echo ""
echo "Verification:"
nvidia-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "from ultralytics import YOLO; print('Ultralytics: OK')"

echo "Setup complete!"
REMOTE

# Step 4: Start training
echo ""
echo "Step 4/5: Starting training..."
ssh azureuser@$VM_IP << 'REMOTE'
cd ~/intelfactor-inference/training

# Start training in tmux
tmux new -d -s train "python scripts/train.py --name v1-nano-real --override model=yolov8n.pt epochs=100 imgsz=640 device=0 batch=32 2>&1 | tee train.log"

echo "Training started in tmux session 'train'"
echo ""
echo "Monitor: ssh azureuser@$VM_IP 'tmux attach -t train'"
REMOTE

# Step 5: Save connection info
echo ""
echo "Step 5/5: Saving deployment info..."
cat > .training_connection << CONN
VM_IP=$VM_IP
TRAINING_NAME=v1-nano-real
MONITOR_CMD="ssh azureuser@$VM_IP 'tmux attach -t train'"
LOG_CMD="ssh azureuser@$VM_IP 'tail -f ~/intelfactor-inference/training/runs/v1-nano-real/results.csv'"
EXPORT_CMD="ssh azureuser@$VM_IP 'cd ~/intelfactor-inference/training && python scripts/export_model.py --model runs/v1-nano-real/weights/best.pt --output exports/v1-nano-real.onnx --imgsz 640'"
CONN

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo ""
echo "Monitor training:"
echo "  ssh azureuser@$VM_IP 'tmux attach -t train'"
echo ""
echo "Check progress:"
echo "  ssh azureuser@$VM_IP 'tail -f ~/intelfactor-inference/training/runs/v1-nano-real/results.csv'"
echo ""
echo "After training (~2-3 hours), export ONNX:"
echo "  bash export_from_azure.sh"
