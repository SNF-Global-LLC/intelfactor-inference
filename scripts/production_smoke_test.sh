#!/usr/bin/env bash
# Production smoke test for a running IntelFactor edge station API.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-}"

if [[ -z "$ENV_FILE" ]]; then
  if [[ -f "$ROOT_DIR/deploy/hybrid/.env" ]]; then
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
fi

API_PORT="${API_PORT:-8080}"
BASE_URL="${BASE_URL:-http://localhost:${API_PORT}}"
TODAY="$(date +%F)"
TMP_DIR="${TMPDIR:-/tmp}/intelfactor-smoke"

mkdir -p "$TMP_DIR"

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

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "$1 is required"
    return 1
  fi
  return 0
}

echo "IntelFactor production smoke test"
echo "API: $BASE_URL"
echo

require_cmd curl || exit 1
require_cmd jq || exit 1

HEALTH_JSON="$TMP_DIR/health.json"
EVENTS_JSON="$TMP_DIR/events.json"
MANIFEST_JSON="$TMP_DIR/manifest.json"
IMAGE_OUT="$TMP_DIR/evidence.jpg"

if curl -fsS "$BASE_URL/health" -o "$HEALTH_JSON"; then
  STATUS="$(jq -r '.status // ""' "$HEALTH_JSON")"
  STORAGE_MODE="$(jq -r '.station.storage_mode // ""' "$HEALTH_JSON")"
  if [[ "$STATUS" == "ok" && "$STORAGE_MODE" == "local" ]]; then
    pass "Health endpoint ok with local storage"
  else
    fail "Health endpoint returned unexpected status or storage_mode"
  fi
else
  fail "Health endpoint failed"
fi

if curl -fsS "$BASE_URL/api/events?limit=10" -o "$EVENTS_JSON"; then
  EVENT_COUNT="$(jq -r '.count // (.events | length) // 0' "$EVENTS_JSON")"
  pass "Recent events endpoint reachable (${EVENT_COUNT} events returned)"
else
  fail "Recent events endpoint failed"
fi

if curl -fsS "$BASE_URL/api/v1/evidence/manifest?date=$TODAY" -o "$MANIFEST_JSON"; then
  EVIDENCE_COUNT="$(jq -r '.count // (.entries | length) // 0' "$MANIFEST_JSON")"
  pass "Evidence manifest reachable for $TODAY (${EVIDENCE_COUNT} entries)"
else
  fail "Evidence manifest endpoint failed"
  EVIDENCE_COUNT=0
fi

if [[ "${EVIDENCE_COUNT:-0}" -gt 0 ]]; then
  EVENT_ID="$(jq -r '.entries[] | .event_id // .metadata.event_id // empty' "$MANIFEST_JSON" | head -n 1)"
  if [[ -n "$EVENT_ID" ]]; then
    if curl -fsS "$BASE_URL/api/v1/evidence/$EVENT_ID/image.jpg" -o "$IMAGE_OUT"; then
      if [[ -s "$IMAGE_OUT" ]]; then
        pass "Evidence image retrieved for $EVENT_ID"
      else
        fail "Evidence image for $EVENT_ID was empty"
      fi
    else
      fail "Evidence image endpoint failed for $EVENT_ID"
    fi
  else
    warn "Evidence entries exist but no event_id was present to fetch an image"
  fi
else
  warn "No evidence entries for $TODAY; image retrieval skipped"
fi

echo
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed, $WARN_COUNT warnings"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi

exit 0
