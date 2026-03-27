#!/usr/bin/env bash
# IntelFactor.ai — TensorRT Engine Build Script
#
# Builds a TRT engine from a YOLO .pt or .onnx file on the CURRENT device.
# Engines are architecture-specific: build ON the target hardware.
#
# Usage:
#   ./scripts/build_trt_engine.sh yolov8n.pt fp16
#   ./scripts/build_trt_engine.sh yolov8n.pt int8 --calib ./calibration_images/
#   ./scripts/build_trt_engine.sh yolov8n.onnx fp16
#   make build-trt MODEL=yolov8n.pt PRECISION=fp16
#   make build-trt-int8 MODEL=yolov8n.pt CALIB_DIR=./calibration_images/
#
# Output:
#   /opt/intelfactor/models/vision/<model_name>_<precision>.engine
#   /opt/intelfactor/models/vision/<model_name>_<precision>_manifest.json
#
# Requirements:
#   - trtexec (ships with TensorRT / JetPack)
#   - ultralytics (for .pt → .onnx export)
#   - Python 3.10+

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Parse arguments ───────────────────────────────────────────────────────────
usage() {
    cat <<EOF
${BOLD}IntelFactor TensorRT Engine Builder${NC}

Usage:
  $0 <model_path> [precision] [--calib <dir>] [--output-dir <dir>] [--workspace <mb>]

Arguments:
  model_path    Path to .pt (PyTorch) or .onnx model file
  precision     fp16 (default) or int8
  --calib       Calibration image directory (required for int8, min 100 images)
  --output-dir  Engine output directory (default: /opt/intelfactor/models/vision)
  --workspace   GPU workspace in MB (default: 4096, fits Orin Nano 8GB)
  --skip-verify Skip post-build verification (faster, not recommended for production)
  --help        Show this help

Examples:
  $0 yolov8n.pt fp16
  $0 yolov8n.pt int8 --calib ./calibration_images/
  $0 ./models/yolov8s.onnx fp16 --output-dir /mnt/models/vision/

EOF
    exit 0
}

MODEL_PATH=""
PRECISION="fp16"
CALIB_DIR=""
OUTPUT_DIR="/opt/intelfactor/models/vision"
WORKSPACE_MB=4096
SKIP_VERIFY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)   usage ;;
        --calib)     CALIB_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --workspace)  WORKSPACE_MB="$2"; shift 2 ;;
        --skip-verify) SKIP_VERIFY=true; shift ;;
        -*)          die "Unknown flag: $1  (run $0 --help)" ;;
        *)
            if [[ -z "$MODEL_PATH" ]]; then
                MODEL_PATH="$1"
            elif [[ -z "$PRECISION" ]] || [[ "$PRECISION" == "fp16" ]]; then
                PRECISION="$1"
            fi
            shift
            ;;
    esac
done

[[ -z "$MODEL_PATH" ]] && usage

# ── Validate inputs ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════${NC}"
echo -e "${BOLD}  IntelFactor TensorRT Engine Builder    ${NC}"
echo -e "${BOLD}════════════════════════════════════════${NC}"
echo ""

# Model file must exist
[[ -f "$MODEL_PATH" ]] || die "Model file not found: $MODEL_PATH"

# Normalise precision
PRECISION="${PRECISION,,}"   # to lowercase
[[ "$PRECISION" == "fp16" || "$PRECISION" == "int8" ]] \
    || die "Precision must be fp16 or int8, got: $PRECISION"

# INT8 requires calibration images
if [[ "$PRECISION" == "int8" ]]; then
    [[ -n "$CALIB_DIR" ]] || die "INT8 precision requires --calib <directory> (minimum 100 images)"
    [[ -d "$CALIB_DIR" ]] || die "Calibration directory not found: $CALIB_DIR"
    CALIB_COUNT=$(find "$CALIB_DIR" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    [[ "$CALIB_COUNT" -ge 100 ]] \
        || die "INT8 calibration needs ≥100 images in $CALIB_DIR, found $CALIB_COUNT"
    info "INT8 calibration: $CALIB_COUNT images in $CALIB_DIR"
fi

# ── Dependency checks ─────────────────────────────────────────────────────────
info "Checking dependencies..."

# trtexec
TRTEXEC=$(command -v trtexec 2>/dev/null \
    || ls /usr/src/tensorrt/bin/trtexec 2>/dev/null \
    || ls /usr/local/bin/trtexec 2>/dev/null \
    || echo "")
[[ -n "$TRTEXEC" ]] || die "trtexec not found. Install TensorRT (JetPack or TRT package).
  On Jetson:  sudo apt-get install tensorrt
  Docker:     use nvcr.io/nvidia/l4t-ml:r36.2.0-py3"

ok "trtexec: $TRTEXEC"

# Python 3.10+
PYTHON=$(command -v python3 || command -v python || echo "")
[[ -n "$PYTHON" ]] || die "Python not found"
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python: $PY_VER"

# CUDA available (via nvidia-smi or /dev/nvidia*)
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    ok "GPU: $GPU_INFO"
else
    # Jetson has CUDA but not nvidia-smi in some JetPack versions
    if [[ -d /dev/nvidia0 ]] || [[ -f /etc/nv_tegra_release ]]; then
        ok "GPU: Jetson detected (nvidia-smi unavailable but CUDA present)"
    else
        die "No CUDA GPU detected. trtexec requires a GPU.
  Check: nvidia-smi
  Jetson: verify JetPack installation"
    fi
fi

# Detect device architecture for manifest
DEVICE_ARCH=$(uname -m)
if [[ -f /etc/nv_tegra_release ]]; then
    DEVICE_TYPE="jetson"
    DEVICE_MODEL=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || echo "Jetson Unknown")
elif command -v nvidia-smi &>/dev/null; then
    DEVICE_TYPE="gpu_server"
    DEVICE_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
else
    DEVICE_TYPE="unknown"
    DEVICE_MODEL="unknown"
fi

info "Device: $DEVICE_MODEL ($DEVICE_ARCH)"

# ── Prepare ONNX ──────────────────────────────────────────────────────────────
MODEL_EXT="${MODEL_PATH##*.}"
MODEL_BASENAME=$(basename "$MODEL_PATH" ".$MODEL_EXT")
BUILD_TMP=$(mktemp -d -m 700)
trap 'rm -rf "$BUILD_TMP"' EXIT

if [[ "$MODEL_EXT" == "pt" ]]; then
    info "Exporting .pt → ONNX (ultralytics)..."

    # Check ultralytics is available
    "$PYTHON" -c "import ultralytics" 2>/dev/null \
        || die "ultralytics not installed. Run: pip install ultralytics
  NOTE: ultralytics requires PyTorch. Install on x86 build machine then copy ONNX to Jetson."

    ONNX_PATH="$BUILD_TMP/${MODEL_BASENAME}.onnx"
    "$PYTHON" - <<PYEOF
from ultralytics import YOLO
import shutil, os

model = YOLO("$MODEL_PATH")
export_path = model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    dynamic=False,   # fixed batch for TRT
    simplify=True,
)
shutil.copy(str(export_path), "$ONNX_PATH")
print(f"ONNX exported to: $ONNX_PATH")
PYEOF
    ok "ONNX export complete: $ONNX_PATH"

elif [[ "$MODEL_EXT" == "onnx" ]]; then
    ONNX_PATH="$MODEL_PATH"
    ok "Using provided ONNX: $ONNX_PATH"

else
    die "Unsupported model format: .$MODEL_EXT (expected .pt or .onnx)"
fi

[[ -f "$ONNX_PATH" ]] || die "ONNX file not created: $ONNX_PATH"

# Compute ONNX SHA256
ONNX_SHA256=$(sha256sum "$ONNX_PATH" | cut -d' ' -f1)
info "ONNX SHA256: $ONNX_SHA256"

# ── Build TRT engine ──────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
ENGINE_NAME="${MODEL_BASENAME}_${PRECISION}.engine"
ENGINE_PATH="${OUTPUT_DIR}/${ENGINE_NAME}"

info "Building TensorRT engine: $ENGINE_PATH"
info "  Precision:  $PRECISION"
info "  Workspace:  ${WORKSPACE_MB}MB"
info "  Input:      1x3x640x640 (opt), max 4x3x640x640"

BUILD_START=$(date +%s)

TRTEXEC_ARGS=(
    "--onnx=$ONNX_PATH"
    "--saveEngine=$ENGINE_PATH"
    "--workspace=$WORKSPACE_MB"
    "--minShapes=images:1x3x640x640"
    "--optShapes=images:1x3x640x640"
    "--maxShapes=images:4x3x640x640"
    "--verbose"
)

if [[ "$PRECISION" == "fp16" ]]; then
    TRTEXEC_ARGS+=("--fp16")
else
    TRTEXEC_ARGS+=(
        "--int8"
        "--calib=$CALIB_DIR"
        "--fp16"   # INT8 layers fall back to FP16 when needed
    )
fi

# Run trtexec — capture stderr so OOM and arch errors surface clearly
echo ""
info "Running trtexec (this takes 5–20 minutes on Jetson Orin Nano)..."
echo ""

set +e
"$TRTEXEC" "${TRTEXEC_ARGS[@]}" 2>&1
TRTEXEC_STATUS=$?
set -e

BUILD_END=$(date +%s)
BUILD_SEC=$((BUILD_END - BUILD_START))

if [[ $TRTEXEC_STATUS -ne 0 ]]; then
    echo ""
    if [[ $TRTEXEC_STATUS -eq 137 ]]; then
        die "trtexec killed (likely OOM). Reduce --workspace (try 2048) or use INT8 instead of FP16.
  Orin Nano 8GB headroom: ~4GB for TRT. If other processes are running, try: systemctl stop docker"
    fi
    # Check for architecture mismatch in output
    die "trtexec failed (exit $TRTEXEC_STATUS).
  Common causes:
    - Wrong architecture: engine built on Orin NX won't run on Orin Nano (different SM count)
    - Unsupported ONNX opset: try opset 13 or 17
    - Bad input shape: YOLO expects 3xHxW with H,W multiples of 32
  Run with VERBOSE=1 for full trtexec log."
fi

[[ -f "$ENGINE_PATH" ]] || die "trtexec exited 0 but engine not created at $ENGINE_PATH"

# ── Post-build info ───────────────────────────────────────────────────────────
ENGINE_SIZE=$(du -sh "$ENGINE_PATH" | cut -f1)
ENGINE_SHA256=$(sha256sum "$ENGINE_PATH" | cut -d' ' -f1)

echo ""
ok "Engine built: $ENGINE_PATH"
ok "Engine size:  $ENGINE_SIZE"
ok "Build time:   ${BUILD_SEC}s"
ok "SHA256:       $ENGINE_SHA256"

# ── Write manifest.json ───────────────────────────────────────────────────────
MANIFEST_PATH="${OUTPUT_DIR}/${MODEL_BASENAME}_${PRECISION}_manifest.json"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

"$PYTHON" - <<PYEOF
import json, os

manifest = {
    "model_name": "${MODEL_BASENAME}",
    "precision": "${PRECISION}",
    "device_type": "${DEVICE_TYPE}",
    "device_model": "${DEVICE_MODEL}",
    "device_arch": "${DEVICE_ARCH}",
    "build_date": "${BUILD_DATE}",
    "build_time_sec": ${BUILD_SEC},
    "engine_path": "${ENGINE_PATH}",
    "engine_sha256": "${ENGINE_SHA256}",
    "engine_size_bytes": os.path.getsize("${ENGINE_PATH}"),
    "onnx_sha256": "${ONNX_SHA256}",
    "input_shape": [1, 3, 640, 640],
    "workspace_mb": ${WORKSPACE_MB},
    "trtexec_args": ${TRTEXEC_ARGS[*]@Q},
}

with open("${MANIFEST_PATH}", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest: ${MANIFEST_PATH}")
PYEOF

ok "Manifest:     $MANIFEST_PATH"

# ── Verify engine loads ───────────────────────────────────────────────────────
if [[ "$SKIP_VERIFY" == "true" ]]; then
    warn "Skipping post-build verification (--skip-verify)"
else
    echo ""
    info "Verifying engine loads and runs on this device..."
    VERIFY_SCRIPT="$(dirname "$0")/verify_trt_engine.py"

    if [[ -f "$VERIFY_SCRIPT" ]]; then
        "$PYTHON" "$VERIFY_SCRIPT" "$ENGINE_PATH"
        VERIFY_STATUS=$?
        if [[ $VERIFY_STATUS -ne 0 ]]; then
            warn "Engine built but verification failed — check GPU memory and driver compatibility."
            warn "The engine may still be valid; re-run: python $VERIFY_SCRIPT $ENGINE_PATH"
        else
            ok "Engine verified: inference on zeros tensor succeeded"
        fi
    else
        warn "verify_trt_engine.py not found — skipping verification"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Build Complete${NC}"
echo -e "${BOLD}════════════════════════════════════════${NC}"
echo ""
echo "  Engine:    $ENGINE_PATH"
echo "  Size:      $ENGINE_SIZE"
echo "  Build time: ${BUILD_SEC}s"
echo ""
echo "  Deploy:"
echo "    cp $ENGINE_PATH /opt/intelfactor/models/vision/"
echo "    cp $MANIFEST_PATH /opt/intelfactor/models/vision/"
echo ""
echo "  Verify:"
echo "    python scripts/verify_trt_engine.py $ENGINE_PATH"
echo "    make verify-trt ENGINE=$ENGINE_PATH"
echo ""
echo -e "${YELLOW}  IMPORTANT: This engine was built for ${DEVICE_MODEL}.${NC}"
echo -e "${YELLOW}  It WILL NOT run on a different GPU architecture.${NC}"
echo ""
