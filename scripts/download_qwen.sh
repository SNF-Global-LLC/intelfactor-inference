#!/bin/bash
# IntelFactor.ai — Download Qwen-2.5-3B-Instruct GGUF for Jetson
#
# Downloads the quantized Qwen-2.5-3B model for on-device RCA explanation.
# INT4 quantization fits in 8GB VRAM with room for TensorRT vision.
#
# Usage:
#   ./scripts/download_qwen.sh                    # default Q4_K_M
#   ./scripts/download_qwen.sh --quant Q5_K_M     # higher quality
#   ./scripts/download_qwen.sh --dir /custom/path  # custom model dir

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/intelfactor/models}"
QUANT="Q4_K_M"
HF_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dir DIR       Model directory (default: $MODEL_DIR)"
    echo "  --quant Q       Quantization level (default: Q4_K_M)"
    echo "                  Options: Q4_K_M, Q5_K_M, Q4_K_S, Q8_0"
    echo "  --list          List available quantizations"
    echo ""
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir) MODEL_DIR="$2"; shift 2 ;;
        --quant) QUANT="$2"; shift 2 ;;
        --list)
            echo "Available quantizations for Qwen2.5-3B-Instruct:"
            echo "  Q4_K_S  — smallest, ~1.8GB, fastest inference"
            echo "  Q4_K_M  — balanced, ~2.0GB (recommended for Orin Nano)"
            echo "  Q5_K_M  — higher quality, ~2.3GB"
            echo "  Q8_0    — best quality, ~3.2GB (requires more VRAM)"
            exit 0
            ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$MODEL_DIR"

FILENAME="qwen2.5-3b-instruct-${QUANT,,}.gguf"
FULL_PATH="${MODEL_DIR}/${FILENAME}"

echo "========================================"
echo " IntelFactor Qwen Model Download"
echo "========================================"
echo " Repo:     $HF_REPO"
echo " Quant:    $QUANT"
echo " Output:   $FULL_PATH"
echo "========================================"

if [ -f "$FULL_PATH" ]; then
    SIZE=$(du -h "$FULL_PATH" | cut -f1)
    echo "Model already exists ($SIZE). Skipping download."
    echo "Delete $FULL_PATH to re-download."
    exit 0
fi

# Try huggingface-hub CLI first (preferred)
if command -v huggingface-cli &>/dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "$HF_REPO" "$FILENAME" --local-dir "$MODEL_DIR"
# Try wget with HuggingFace URL
elif command -v wget &>/dev/null; then
    echo "Using wget..."
    URL="https://huggingface.co/${HF_REPO}/resolve/main/${FILENAME}"
    wget -O "$FULL_PATH" "$URL"
# Try curl
elif command -v curl &>/dev/null; then
    echo "Using curl..."
    URL="https://huggingface.co/${HF_REPO}/resolve/main/${FILENAME}"
    curl -L -o "$FULL_PATH" "$URL"
else
    echo "Error: No download tool found (need huggingface-cli, wget, or curl)"
    exit 1
fi

if [ -f "$FULL_PATH" ]; then
    SIZE=$(du -h "$FULL_PATH" | cut -f1)
    echo ""
    echo "========================================"
    echo " ✓ Downloaded: $FILENAME ($SIZE)"
    echo " Path: $FULL_PATH"
    echo "========================================"
else
    echo "Error: Download failed."
    exit 1
fi
