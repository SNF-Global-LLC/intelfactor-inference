#!/bin/bash
# Auto-retry Azure VM deployment until quota is active

VM_NAME="if-train-gpu-01"
RESOURCE_GROUP="intelfactor-ml"
LOCATION="westus2"
MAX_RETRIES=30
RETRY_DELAY=60  # seconds

echo "=== Azure GPU VM Auto-Deploy ==="
echo "Will retry every ${RETRY_DELAY}s until quota propagates"
echo "Max retries: $MAX_RETRIES"
echo ""

for i in $(seq 1 $MAX_RETRIES); do
    echo "[Attempt $i/$MAX_RETRIES] Checking quota and deploying..."
    
    # Check current quota
    QUOTA=$(az vm list-usage -l $LOCATION --query "[?localName=='Standard NCASv3_T4 Family vCPUs'].limit" -o tsv 2>/dev/null)
    echo "  Current quota: $QUOTA"
    
    # Try to create VM
    az vm create \
        --resource-group $RESOURCE_GROUP \
        --name $VM_NAME \
        --size Standard_NC4as_T4_v3 \
        --image Ubuntu2204 \
        --admin-username azureuser \
        --ssh-key-values ~/.ssh/id_rsa.pub \
        --public-ip-sku Standard \
        --output table 2>&1 | grep -E "(ResourceGroup|publicIpAddress|ProvisioningState|VM created successfully)" || true
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "=== SUCCESS! VM Created ==="
        
        # Get IP
        VM_IP=$(az network public-ip list -g $RESOURCE_GROUP \
            --query "[?contains(name, '$VM_NAME')].ipAddress" -o tsv)
        
        echo "VM IP: $VM_IP"
        echo ""
        echo "Next: Deploy training"
        echo "  bash deploy_to_azure.sh $VM_IP"
        
        # Save IP to file
        echo "$VM_IP" > .azure_vm_ip
        exit 0
    fi
    
    echo "  Quota not ready yet. Retrying in ${RETRY_DELAY}s..."
    echo ""
    sleep $RETRY_DELAY
done

echo "Max retries reached. Quota still not active."
echo "Try again later or contact Azure support."
exit 1
