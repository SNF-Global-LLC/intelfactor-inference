#!/bin/bash
# Prepare dataset for upload to cloud training environment

set -e

echo "=== Preparing Dataset for Upload ==="

# Create archive of training data
cd training/datasets

# Create tar archive (excludes .DS_Store files)
echo "Creating combined.tar.gz..."
tar -czf ../../combined.tar.gz \
    --exclude=".DS_Store" \
    --exclude="*.cache" \
    combined/

cd ../..

# Show size
ls -lh combined.tar.gz

echo ""
echo "=== Upload Instructions ==="
echo ""
echo "Option 1 - Upload to RunPod/Lambda:"
echo "  rsync -avz --progress combined.tar.gz root@<IP>:/workspace/"
echo "  ssh root@<IP> 'cd /workspace && tar -xzf combined.tar.gz'"
echo ""
echo "Option 2 - Upload to Google Drive for Colab:"
echo "  Upload combined.tar.gz to your Google Drive"
echo "  Mount drive in Colab and extract"
echo ""
echo "Option 3 - Azure (after quota increase):"
echo "  rsync -avz --progress combined.tar.gz azureuser@<IP>:~/"
echo "  ssh azureuser@<IP> 'mkdir -p intelfactor-inference/training && tar -xzf combined.tar.gz -C intelfactor-inference/training/'"
