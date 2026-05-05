#!/usr/bin/env bash
# IntelFactor station doctor for edge-only and hybrid deployments.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-}"
MODE="${MODE:-hybrid}"

if [[ -z "$ENV_FILE" ]]; then
  if [[ -f "$ROOT_DIR/deploy/$MODE/.env" ]]; then
    ENV_FILE="$ROOT_DIR/deploy/$MODE/.env"
  elif [[ -f "$ROOT_DIR/deploy/hybrid/.env" ]]; then
    ENV_FILE="$ROOT_DIR/deploy/hybrid/.env"
  elif [[ -f "$ROOT_DIR/deploy/edge-only/.env" ]]; then
    ENV_FILE="$ROOT_DIR/deploy/edge-only/.env"
  fi
fi

if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  echo "[INFO] Loaded environment: $ENV_FILE"
else
  echo "[INFO] No .env file found; using defaults and current environment"
fi

STATION_ID="${STATION_ID:-station_01}"
CAMERA_URI="${CAMERA_URI:-/dev/video0}"
MODELS_DIR="${MODELS_DIR:-/opt/intelfactor/models}"
EVIDENCE_DIR="${EVIDENCE_DIR:-/opt/intelfactor/data/evidence}"
DATA_DIR="${DATA_DIR:-/opt/intelfactor/data}"
API_PORT="${API_PORT:-8080}"
CLOUD_API_URL="${CLOUD_API_URL:-}"

HOST_EVIDENCE_DIR="$EVIDENCE_DIR"
if [[ "$HOST_EVIDENCE_DIR" == /data/* ]]; then
  HOST_EVIDENCE_DIR="$DATA_DIR/${HOST_EVIDENCE_DIR#/data/}"
fi

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() {
  echo "[PASS] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  echo "[FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

warn() {
  echo "[WARN] $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

check_cmd() {
  local cmd="$1"
  local label="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "$label available ($(command -v "$cmd"))"
    return 0
  fi
  fail "$label not available"
  return 1
}

echo "IntelFactor station doctor"
echo "Station: $STATION_ID"
echo "Camera:  $CAMERA_URI"
echo

if check_cmd docker "Docker"; then
  if docker info >/dev/null 2>&1; then
    pass "Docker daemon reachable"
  else
    fail "Docker daemon not reachable"
  fi
fi

if docker compose version >/dev/null 2>&1; then
  pass "Docker Compose v2 available"
else
  fail "Docker Compose v2 not available"
fi

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  pass "NVIDIA container runtime registered"
else
  fail "NVIDIA container runtime not registered"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >/dev/null 2>&1 && pass "nvidia-smi responds" || warn "nvidia-smi present but did not respond"
elif command -v tegrastats >/dev/null 2>&1; then
  timeout 3 tegrastats >/tmp/intelfactor-tegrastats.txt 2>/dev/null &
  sleep 1
  pass "tegrastats available"
else
  warn "No nvidia-smi or tegrastats command found"
fi

if [[ -d "$MODELS_DIR" ]]; then
  pass "Models directory exists: $MODELS_DIR"
else
  fail "Models directory missing: $MODELS_DIR"
fi

if [[ -d "$HOST_EVIDENCE_DIR" ]]; then
  if touch "$HOST_EVIDENCE_DIR/.intelfactor-write-test" 2>/dev/null; then
    rm -f "$HOST_EVIDENCE_DIR/.intelfactor-write-test"
    pass "Evidence directory writable: $HOST_EVIDENCE_DIR"
  else
    fail "Evidence directory not writable: $HOST_EVIDENCE_DIR"
  fi
else
  fail "Evidence directory missing: $HOST_EVIDENCE_DIR"
fi

if [[ -d "$DATA_DIR" ]]; then
  FREE_GB="$(df -Pk "$DATA_DIR" | awk 'NR==2 {printf "%.1f", $4 / 1024 / 1024}')"
  if awk "BEGIN {exit !($FREE_GB >= 20)}"; then
    pass "Disk free space: ${FREE_GB}GB"
  else
    fail "Disk free space low: ${FREE_GB}GB"
  fi
else
  fail "Data directory missing: $DATA_DIR"
fi

if [[ "$CAMERA_URI" == /dev/video* ]]; then
  if [[ -e "$CAMERA_URI" ]]; then
    if command -v v4l2-ctl >/dev/null 2>&1; then
      if v4l2-ctl --device="$CAMERA_URI" --all >/tmp/intelfactor-v4l2.txt 2>&1; then
        pass "USB camera visible through v4l2: $CAMERA_URI"
      else
        fail "USB camera exists but v4l2 query failed: $CAMERA_URI"
      fi
    else
      fail "v4l2-ctl not installed for USB camera check"
    fi
  else
    fail "USB camera device missing: $CAMERA_URI"
  fi
elif [[ "$CAMERA_URI" == rtsp://* || "$CAMERA_URI" == rtsps://* ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    if timeout 20 ffmpeg -rtsp_transport tcp -i "$CAMERA_URI" -frames:v 1 -y /tmp/intelfactor-rtsp-test.jpg >/tmp/intelfactor-ffmpeg.txt 2>&1; then
      pass "RTSP frame captured"
    else
      fail "RTSP frame capture failed"
    fi
  else
    fail "ffmpeg not installed for RTSP camera check"
  fi
else
  warn "Camera URI is not USB or RTSP; camera frame check skipped"
fi

LOCAL_HEALTH_URL="http://localhost:${API_PORT}/health"
if command -v curl >/dev/null 2>&1; then
  if curl -fsS "$LOCAL_HEALTH_URL" >/tmp/intelfactor-local-health.json 2>/dev/null; then
    pass "Local health endpoint reachable: $LOCAL_HEALTH_URL"
  else
    fail "Local health endpoint not reachable: $LOCAL_HEALTH_URL"
  fi

  if [[ -n "$CLOUD_API_URL" ]]; then
    CLOUD_HEALTH_URL="${CLOUD_API_URL%/}/health"
    if curl -fsS "$CLOUD_HEALTH_URL" >/tmp/intelfactor-cloud-health.json 2>/dev/null; then
      pass "Cloud health endpoint reachable: $CLOUD_HEALTH_URL"
    else
      warn "Cloud health endpoint not reachable: $CLOUD_HEALTH_URL"
    fi
  fi
else
  fail "curl not installed for health checks"
fi

if [[ "${SYNC_ENABLED:-}" == "true" || -n "$CLOUD_API_URL" ]]; then
  SYNC_STATUS="$(docker inspect -f '{{.State.Status}}' intelfactor-sync-agent 2>/dev/null || true)"
  if [[ "$SYNC_STATUS" == "running" ]]; then
    pass "Hybrid sync agent container running"
  elif [[ -n "$SYNC_STATUS" ]]; then
    fail "Hybrid sync agent container is $SYNC_STATUS"
  else
    warn "Hybrid sync agent container not found"
  fi
fi

echo
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed, $WARN_COUNT warnings"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi

exit 0
