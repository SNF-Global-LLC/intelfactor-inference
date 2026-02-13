#!/bin/bash
# IntelFactor.ai — Build TensorRT Engine from YOLO Model
# Run this ON the target Jetson device (engines are device-specific).
#
# Usage:
#   ./scripts/build_trt_engine.sh yolov8n.pt          # ONNX → TRT on device
#   ./scripts/build_trt_engine.sh yolov8n.onnx         # ONNX → TRT directly
#   ./scripts/build_trt_engine.sh --model yolov8s --fp16  # with options
#
# Output: /opt/intelfactor/models/yolov8n_fp16.engine

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/intelfactor/models}"
PRECISION="${PRECISION:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INPUT_SIZE="${INPUT_SIZE:-640}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"

usage() {
    echo "Usage: $0 [options] <model_file>"
    echo ""
    echo "Options:"
    echo "  --fp16          Use FP16 precision (default)"
    echo "  --int8          Use INT8 precision (requires calibration)"
    echo "  --fp32          Use FP32 precision"
    echo "  --size N        Input size (default: 640)"
    echo "  --batch N       Batch size (default: 1)"
    echo "  --output DIR    Output directory (default: $MODEL_DIR)"
    echo ""
    echo "Examples:"
    echo "  $0 yolov8n.pt"
    echo "  $0 --int8 --size 640 yolov8n.onnx"
    exit 1
}

# Parse arguments
MODEL_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fp16) PRECISION="fp16"; shift ;;
        --int8) PRECISION="int8"; shift ;;
        --fp32) PRECISION="fp32"; shift ;;
        --size) INPUT_SIZE="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --output) MODEL_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) MODEL_FILE="$1"; shift ;;
    esac
done

if [ -z "$MODEL_FILE" ]; then
    echo "Error: No model file specified."
    usage
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi

mkdir -p "$MODEL_DIR"

BASENAME=$(basename "$MODEL_FILE" | sed 's/\.[^.]*$//')
ENGINE_FILE="${MODEL_DIR}/${BASENAME}_${PRECISION}.engine"
ONNX_FILE="${MODEL_DIR}/${BASENAME}.onnx"

echo "========================================"
echo " IntelFactor TRT Engine Builder"
echo "========================================"
echo " Input:     $MODEL_FILE"
echo " Output:    $ENGINE_FILE"
echo " Precision: $PRECISION"
echo " Input:     ${INPUT_SIZE}x${INPUT_SIZE}"
echo " Batch:     $BATCH_SIZE"
echo "========================================"

# Step 1: Convert .pt to ONNX if needed
if [[ "$MODEL_FILE" == *.pt ]]; then
    echo "[1/2] Exporting PyTorch → ONNX..."
    python3 -c "
from ultralytics import YOLO
model = YOLO('$MODEL_FILE')
model.export(format='onnx', imgsz=$INPUT_SIZE, opset=17, simplify=True)
print('ONNX export complete')
" 2>&1
    # ultralytics puts the .onnx next to the .pt
    PT_DIR=$(dirname "$MODEL_FILE")
    EXPORTED="${PT_DIR}/${BASENAME}.onnx"
    if [ -f "$EXPORTED" ]; then
        mv "$EXPORTED" "$ONNX_FILE"
    else
        echo "Error: ONNX export failed."
        exit 1
    fi
    echo "  → $ONNX_FILE"
elif [[ "$MODEL_FILE" == *.onnx ]]; then
    cp "$MODEL_FILE" "$ONNX_FILE"
    echo "[1/2] ONNX input provided, skipping export."
else
    echo "Error: Unsupported format. Use .pt or .onnx"
    exit 1
fi

# Step 2: Build TRT engine
echo "[2/2] Building TensorRT engine..."
TRTEXEC_ARGS=(
    --onnx="$ONNX_FILE"
    --saveEngine="$ENGINE_FILE"
    --workspace="$WORKSPACE_MB"
    --minShapes=images:1x3x${INPUT_SIZE}x${INPUT_SIZE}
    --optShapes=images:${BATCH_SIZE}x3x${INPUT_SIZE}x${INPUT_SIZE}
    --maxShapes=images:${BATCH_SIZE}x3x${INPUT_SIZE}x${INPUT_SIZE}
)

case $PRECISION in
    fp16) TRTEXEC_ARGS+=(--fp16) ;;
    int8) TRTEXEC_ARGS+=(--int8 --fp16) ;;
    fp32) ;; # default
esac

# Use trtexec from TensorRT installation
if command -v trtexec &>/dev/null; then
    trtexec "${TRTEXEC_ARGS[@]}" 2>&1
elif [ -f /usr/src/tensorrt/bin/trtexec ]; then
    /usr/src/tensorrt/bin/trtexec "${TRTEXEC_ARGS[@]}" 2>&1
else
    echo "Error: trtexec not found. Install TensorRT or JetPack."
    exit 1
fi

if [ -f "$ENGINE_FILE" ]; then
    SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
    echo ""
    echo "========================================"
    echo " ✓ Engine built: $ENGINE_FILE ($SIZE)"
    echo "========================================"
else
    echo "Error: Engine build failed."
    exit 1
fi
