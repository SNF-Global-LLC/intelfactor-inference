#!/bin/bash
# IntelFactor.ai — Model Setup Script
# One-stop script to prepare all models for edge inference.
#
# This script:
#   1. Creates the models directory structure
#   2. Downloads Qwen GGUF for RCA explanation
#   3. Builds or downloads TRT engine for vision
#   4. Verifies all models are ready
#
# Usage:
#   ./scripts/setup_models.sh                     # Interactive setup
#   ./scripts/setup_models.sh --quick             # Minimal setup (download only)
#   ./scripts/setup_models.sh --verify            # Just verify existing models
#   ./scripts/setup_models.sh --from-onnx yolo.onnx  # Build TRT from ONNX

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-/opt/intelfactor/models}"
ENABLE_LOCAL_LLM="${ENABLE_LOCAL_LLM:-true}"

# Model files
VISION_ENGINE=""
LANGUAGE_GGUF=""

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "   IntelFactor Model Setup"
    echo "========================================"
    echo -e "${NC}"
}

detect_device() {
    # Detect Jetson or GPU type
    if [ -f /etc/nv_tegra_release ]; then
        DEVICE_TYPE="jetson"
        if grep -q "R36" /etc/nv_tegra_release 2>/dev/null; then
            JETPACK_VERSION="6.x"
        elif grep -q "R35" /etc/nv_tegra_release 2>/dev/null; then
            JETPACK_VERSION="5.x"
        else
            JETPACK_VERSION="unknown"
        fi

        # Detect Jetson model
        JETSON_MODEL=$(cat /sys/firmware/devicetree/base/model 2>/dev/null | tr -d '\0' || echo "unknown")
        log_info "Detected: $JETSON_MODEL (JetPack $JETPACK_VERSION)"
    elif nvidia-smi &>/dev/null; then
        DEVICE_TYPE="gpu"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
        log_info "Detected: $GPU_NAME"
    else
        DEVICE_TYPE="cpu"
        log_warn "No NVIDIA GPU detected. Vision inference will use stub."
    fi
}

create_directories() {
    log_info "Creating model directories..."
    sudo mkdir -p "$MODEL_DIR"
    sudo chown -R "$(id -u):$(id -g)" "$MODEL_DIR" 2>/dev/null || true

    # Create subdirectories
    mkdir -p "$MODEL_DIR/vision"
    mkdir -p "$MODEL_DIR/language"
    mkdir -p "$MODEL_DIR/configs"

    log_info "Model directory: $MODEL_DIR"
}

download_language_model() {
    # Skip if LLM disabled
    if [ "$ENABLE_LOCAL_LLM" = "false" ]; then
        log_info "Language model: SKIPPED (ENABLE_LOCAL_LLM=false)"
        log_info "RCA explanations will use statistical fallback instead of LLM."
        return 0
    fi

    log_info "Setting up Qwen language model..."

    # Check if already exists
    GGUF_FILES=($(ls "$MODEL_DIR"/language/*.gguf 2>/dev/null || true))
    if [ ${#GGUF_FILES[@]} -gt 0 ]; then
        log_info "Found existing GGUF: ${GGUF_FILES[0]}"
        LANGUAGE_GGUF="${GGUF_FILES[0]}"
        return 0
    fi

    # Download using the existing script
    if [ -f "$SCRIPT_DIR/download_qwen.sh" ]; then
        MODEL_DIR="$MODEL_DIR/language" "$SCRIPT_DIR/download_qwen.sh"
        GGUF_FILES=($(ls "$MODEL_DIR"/language/*.gguf 2>/dev/null || true))
        if [ ${#GGUF_FILES[@]} -gt 0 ]; then
            LANGUAGE_GGUF="${GGUF_FILES[0]}"
        fi
    else
        log_warn "download_qwen.sh not found. Manual download required."
        echo ""
        echo "Download Qwen-2.5-3B-Instruct GGUF manually:"
        echo "  wget -O $MODEL_DIR/language/qwen2.5-3b-instruct-q4_k_m.gguf \\"
        echo "    https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
        echo ""
    fi
}

setup_vision_model() {
    local onnx_file="${1:-}"

    log_info "Setting up vision model..."

    # Check if TRT engine already exists
    ENGINE_FILES=($(ls "$MODEL_DIR"/vision/*.engine 2>/dev/null || true))
    if [ ${#ENGINE_FILES[@]} -gt 0 ]; then
        log_info "Found existing TRT engine: ${ENGINE_FILES[0]}"
        VISION_ENGINE="${ENGINE_FILES[0]}"
        return 0
    fi

    # Check for ONNX to convert
    if [ -n "$onnx_file" ] && [ -f "$onnx_file" ]; then
        log_info "Building TRT engine from: $onnx_file"
        if [ -f "$SCRIPT_DIR/build_trt_engine.sh" ]; then
            MODEL_DIR="$MODEL_DIR/vision" "$SCRIPT_DIR/build_trt_engine.sh" "$onnx_file"
            ENGINE_FILES=($(ls "$MODEL_DIR"/vision/*.engine 2>/dev/null || true))
            if [ ${#ENGINE_FILES[@]} -gt 0 ]; then
                VISION_ENGINE="${ENGINE_FILES[0]}"
            fi
        fi
        return 0
    fi

    # Check for ONNX in common locations
    ONNX_SEARCH=(
        "$MODEL_DIR/vision/*.onnx"
        "$MODEL_DIR/*.onnx"
        "$PROJECT_ROOT/*.onnx"
        "$HOME/yolov8*.onnx"
    )

    for pattern in "${ONNX_SEARCH[@]}"; do
        ONNX_FILES=($(ls $pattern 2>/dev/null || true))
        if [ ${#ONNX_FILES[@]} -gt 0 ]; then
            log_info "Found ONNX: ${ONNX_FILES[0]}"
            setup_vision_model "${ONNX_FILES[0]}"
            return 0
        fi
    done

    # No model found - provide instructions
    log_warn "No vision model found. Manual setup required."
    echo ""
    echo "To set up the YOLOv8 vision model:"
    echo ""
    echo "  Option A: Download pre-trained YOLOv8 and convert to TRT"
    echo "    pip install ultralytics"
    echo "    yolo export model=yolov8n.pt format=onnx imgsz=640"
    echo "    ./scripts/build_trt_engine.sh yolov8n.onnx"
    echo ""
    echo "  Option B: Use your custom trained model"
    echo "    ./scripts/build_trt_engine.sh /path/to/your/model.onnx"
    echo ""
    echo "  Option C: Copy existing TRT engine"
    echo "    cp /path/to/yolov8n_fp16.engine $MODEL_DIR/vision/"
    echo ""
}

create_config() {
    log_info "Creating model configuration..."

    # Detect defect classes from taxonomy
    DEFECT_CLASSES='["scratch_surface", "scratch_edge", "burr", "pit_corrosion", "discoloration", "dent", "crack", "warp", "handle_gap", "handle_crack", "logo_defect", "dimension_out_of_spec", "foreign_material"]'

    CONFIG_FILE="$MODEL_DIR/configs/models.json"
    cat > "$CONFIG_FILE" << EOF
{
  "vision": {
    "engine_path": "${VISION_ENGINE:-$MODEL_DIR/vision/yolov8n_fp16.engine}",
    "input_size": [640, 640],
    "num_classes": 13,
    "defect_classes": $DEFECT_CLASSES,
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45
  },
  "language": {
    "model_path": "${LANGUAGE_GGUF:-$MODEL_DIR/language/qwen2.5-3b-instruct-q4_k_m.gguf}",
    "n_gpu_layers": -1,
    "n_ctx": 4096,
    "temperature": 0.3
  }
}
EOF

    log_info "Config written: $CONFIG_FILE"
}

verify_models() {
    echo ""
    log_info "Verifying model setup..."
    echo ""

    local status=0

    # Vision model
    ENGINE_FILES=($(ls "$MODEL_DIR"/vision/*.engine 2>/dev/null || true))
    if [ ${#ENGINE_FILES[@]} -gt 0 ]; then
        SIZE=$(du -h "${ENGINE_FILES[0]}" | cut -f1)
        echo -e "  ${GREEN}[ok]${NC} Vision engine: ${ENGINE_FILES[0]} ($SIZE)"
    else
        echo -e "  ${RED}[missing]${NC} Vision engine: $MODEL_DIR/vision/*.engine"
        status=1
    fi

    # Language model
    GGUF_FILES=($(ls "$MODEL_DIR"/language/*.gguf 2>/dev/null || true))
    if [ ${#GGUF_FILES[@]} -gt 0 ]; then
        SIZE=$(du -h "${GGUF_FILES[0]}" | cut -f1)
        echo -e "  ${GREEN}[ok]${NC} Language model: ${GGUF_FILES[0]} ($SIZE)"
    elif [ "$ENABLE_LOCAL_LLM" = "false" ]; then
        echo -e "  ${YELLOW}[skip]${NC} Language model: disabled (ENABLE_LOCAL_LLM=false, using statistical fallback)"
    else
        echo -e "  ${RED}[missing]${NC} Language model: $MODEL_DIR/language/*.gguf"
        status=1
    fi

    # Config
    if [ -f "$MODEL_DIR/configs/models.json" ]; then
        echo -e "  ${GREEN}[ok]${NC} Config: $MODEL_DIR/configs/models.json"
    else
        echo -e "  ${YELLOW}[warn]${NC} Config: $MODEL_DIR/configs/models.json (will be generated)"
    fi

    echo ""

    if [ $status -eq 0 ]; then
        log_info "All models ready!"
        echo ""
        echo "Set these environment variables for inference:"
        echo "  export VISION_ENGINE=${ENGINE_FILES[0]}"
        echo "  export LANGUAGE_MODEL=${GGUF_FILES[0]}"
        echo ""
    else
        log_warn "Some models missing. See instructions above."
    fi

    return $status
}

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --quick              Download models only (no TRT build)"
    echo "  --verify             Verify existing models"
    echo "  --from-onnx FILE     Build TRT engine from ONNX file"
    echo "  --dir DIR            Models directory (default: $MODEL_DIR)"
    echo "  --no-llm             Skip language model download (RCA uses statistical fallback)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Environment variables:"
    echo "  ENABLE_LOCAL_LLM     Set to 'false' to skip LLM download (default: true)"
    echo "  MODEL_DIR            Models root directory"
    echo ""
    echo "Examples:"
    echo "  $0                              # Interactive setup"
    echo "  $0 --verify                     # Check model status"
    echo "  $0 --from-onnx yolov8n.onnx     # Build from ONNX"
    echo "  ENABLE_LOCAL_LLM=false $0        # Skip LLM, vision only"
    echo ""
}

# Main
MODE="full"
ONNX_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) MODE="quick"; shift ;;
        --verify) MODE="verify"; shift ;;
        --from-onnx) ONNX_FILE="$2"; shift 2 ;;
        --dir) MODEL_DIR="$2"; shift 2 ;;
        --no-llm) ENABLE_LOCAL_LLM="false"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

print_banner
detect_device

case $MODE in
    verify)
        verify_models
        ;;
    quick)
        create_directories
        download_language_model
        create_config
        verify_models
        ;;
    full)
        create_directories
        download_language_model
        setup_vision_model "$ONNX_FILE"
        create_config
        verify_models
        ;;
esac
