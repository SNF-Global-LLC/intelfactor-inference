#!/bin/bash
# Export ONNX from Azure VM and download

if [ -f .azure_vm_ip ]; then
    VM_IP=$(cat .azure_vm_ip)
else
    echo "No saved IP found. Usage: bash export_from_azure.sh <VM_IP>"
    exit 1
fi

echo "Exporting ONNX from Azure VM: $VM_IP"
ssh azureuser@$VM_IP "cd ~/intelfactor-inference/training && python scripts/export_model.py --model runs/v1-nano-real/weights/best.pt --output exports/v1-nano-real.onnx --imgsz 640"

echo "Downloading ONNX..."
rsync -avz azureuser@$VM_IP:~/intelfactor-inference/training/exports/v1-nano-real.onnx ./

echo ""
echo "ONNX downloaded: ./v1-nano-real.onnx"
echo ""
echo "Copy to Jetson:"
echo "  scp v1-nano-real.onnx tony@<JETSON_IP>:/opt/intelfactor/models/"
