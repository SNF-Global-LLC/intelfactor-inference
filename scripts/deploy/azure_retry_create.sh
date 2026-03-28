#!/bin/bash
# Retry Azure VM creation after quota approval

VM_NAME="if-train-gpu-01"
RESOURCE_GROUP="intelfactor-ml"
LOCATION="westus2"

echo "=== Creating Azure GPU VM ==="
echo "VM: $VM_NAME"
echo "Size: Standard_NC4as_T4_v3"
echo "Location: $LOCATION"
echo ""

az vm create \
    --resource-group $RESOURCE_GROUP \
    --name $VM_NAME \
    --size Standard_NC4as_T4_v3 \
    --image Ubuntu2204 \
    --admin-username azureuser \
    --ssh-key-values ~/.ssh/id_rsa.pub \
    --public-ip-sku Standard \
    --output table

if [ $? -eq 0 ]; then
    echo ""
    echo "=== VM Created Successfully ==="
    
    # Get IP
    VM_IP=$(az network public-ip list -g $RESOURCE_GROUP \
        --query "[?contains(name, '$VM_NAME')].ipAddress" -o tsv)
    
    echo "VM IP: $VM_IP"
    echo ""
    echo "Next steps:"
    echo "1. Upload dataset:"
    echo "   rsync -avz --progress combined.tar.gz azureuser@$VM_IP:~"
    echo ""
    echo "2. SSH and setup:"
    echo "   ssh azureuser@$VM_IP"
    echo "   tar -xzf combined.tar.gz"
    echo "   cd intelfactor-inference/training"
    echo "   bash azure_training_setup.sh"
    echo "   bash azure_train.sh"
else
    echo "VM creation failed - quota may still be propagating"
    echo "Try again in 15-30 minutes or use RunPod"
fi
