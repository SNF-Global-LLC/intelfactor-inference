#!/bin/bash
# IntelFactor.ai — Soak Test
# Run station for a specified duration, then audit evidence integrity.
#
# Usage:
#   ./scripts/soak_test.sh                     # 8 hour default
#   ./scripts/soak_test.sh --hours 1           # 1 hour quick test
#   ./scripts/soak_test.sh --hours 24          # 24 hour full test
#   ./scripts/soak_test.sh --audit-only        # skip run, audit existing data

set -euo pipefail

HOURS=8
CONFIG="/opt/intelfactor/config/station.yaml"
DATA_DIR="/opt/intelfactor/data"
AUDIT_ONLY=false
SAMPLE_SIZE=50
PORT=8080

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours) HOURS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --audit-only) AUDIT_ONLY=true; shift ;;
        --sample) SAMPLE_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--hours N] [--config PATH] [--audit-only]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

DURATION_SEC=$((HOURS * 3600))
LOG_FILE="/tmp/intelfactor_soak_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo " IntelFactor Soak Test"
echo "========================================"
echo " Duration:    ${HOURS}h (${DURATION_SEC}s)"
echo " Config:      $CONFIG"
echo " Data Dir:    $DATA_DIR"
echo " Log:         $LOG_FILE"
echo " Sample Size: $SAMPLE_SIZE"
echo "========================================"

# ── Run Doctor First ──
if [ "$AUDIT_ONLY" = false ]; then
    echo ""
    echo "[Pre-flight] Running doctor..."
    intelfactor-station --config "$CONFIG" --doctor --no-camera 2>&1 || {
        echo "Doctor failed. Fix issues before soak test."
        exit 1
    }

    # ── Start Station ──
    echo ""
    echo "[Start] Launching station (logging to $LOG_FILE)..."
    intelfactor-station --config "$CONFIG" --port "$PORT" \
        > "$LOG_FILE" 2>&1 &
    STATION_PID=$!
    echo "  Station PID: $STATION_PID"

    # Wait for health
    echo "  Waiting for health endpoint..."
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "  ✓ Station healthy"
            break
        fi
        sleep 1
    done

    # ── Run for Duration ──
    echo ""
    echo "[Running] Soak test for ${HOURS}h..."
    echo "  Start: $(date)"
    echo "  Expected end: $(date -d "+${HOURS} hours" 2>/dev/null || date -v+${HOURS}H 2>/dev/null || echo 'in ${HOURS}h')"

    # Monitor every 5 minutes
    INTERVAL=300
    ELAPSED=0
    while [ $ELAPSED -lt $DURATION_SEC ]; do
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))

        # Check station is still alive
        if ! kill -0 "$STATION_PID" 2>/dev/null; then
            echo "  ✗ Station crashed at ${ELAPSED}s!"
            echo "  Last 20 lines of log:"
            tail -20 "$LOG_FILE"
            exit 1
        fi

        # Log status
        HOURS_DONE=$(echo "scale=1; $ELAPSED / 3600" | bc 2>/dev/null || echo "$((ELAPSED/3600))")
        STATUS=$(curl -sf "http://localhost:${PORT}/api/pipeline/stats" 2>/dev/null || echo "{}")
        echo "  [${HOURS_DONE}h] alive — $STATUS" | head -c 200
        echo ""
    done

    # ── Stop Station ──
    echo ""
    echo "[Stop] Shutting down station..."
    kill "$STATION_PID" 2>/dev/null || true
    wait "$STATION_PID" 2>/dev/null || true
    echo "  ✓ Station stopped"
fi

# ── Audit Evidence ──
echo ""
echo "========================================"
echo " Evidence Traceability Audit"
echo "========================================"

EVIDENCE_DIR="${DATA_DIR}/evidence"
ACC_DB="${DATA_DIR}/accumulator.db"
TRIPLES_DB="${DATA_DIR}/triples.db"

# Count events
if [ -f "$ACC_DB" ]; then
    TOTAL_EVENTS=$(sqlite3 "$ACC_DB" "SELECT COUNT(*) FROM defect_events" 2>/dev/null || echo "0")
    echo " Total events in DB: $TOTAL_EVENTS"
else
    echo " ✗ accumulator.db not found at $ACC_DB"
    TOTAL_EVENTS=0
fi

# Count triples
if [ -f "$TRIPLES_DB" ]; then
    TOTAL_TRIPLES=$(sqlite3 "$TRIPLES_DB" "SELECT COUNT(*) FROM causal_triples" 2>/dev/null || echo "0")
    VERIFIED=$(sqlite3 "$TRIPLES_DB" "SELECT COUNT(*) FROM causal_triples WHERE status='verified'" 2>/dev/null || echo "0")
    echo " Total triples: $TOTAL_TRIPLES (verified: $VERIFIED)"
else
    echo " ✗ triples.db not found at $TRIPLES_DB"
fi

# Count evidence files
if [ -d "$EVIDENCE_DIR" ]; then
    JPEG_COUNT=$(find "$EVIDENCE_DIR" -name "*.jpg" | wc -l)
    JSON_COUNT=$(find "$EVIDENCE_DIR" -name "*.json" | wc -l)
    DISK_USED=$(du -sh "$EVIDENCE_DIR" 2>/dev/null | cut -f1)
    echo " Evidence JPEGs: $JPEG_COUNT"
    echo " Evidence JSON:  $JSON_COUNT"
    echo " Disk used:      $DISK_USED"
else
    echo " ⚠ Evidence dir not found (no FAIL/REVIEW events?)"
    JPEG_COUNT=0
fi

# Sample audit
echo ""
echo " Sampling $SAMPLE_SIZE events..."
PASS=0
FAIL_COUNT=0

if [ -f "$ACC_DB" ] && [ "$TOTAL_EVENTS" -gt 0 ]; then
    SAMPLE_IDS=$(sqlite3 "$ACC_DB" "SELECT event_id FROM defect_events ORDER BY RANDOM() LIMIT $SAMPLE_SIZE" 2>/dev/null)

    for EVENT_ID in $SAMPLE_IDS; do
        # Check metadata exists
        HAS_META=$(sqlite3 "$ACC_DB" "SELECT COUNT(*) FROM defect_events WHERE event_id='$EVENT_ID'" 2>/dev/null || echo "0")

        # Check evidence JPEG exists
        HAS_JPEG=$(find "$EVIDENCE_DIR" -name "${EVENT_ID}.jpg" 2>/dev/null | head -1)

        if [ "$HAS_META" -gt 0 ] && [ -n "$HAS_JPEG" ]; then
            PASS=$((PASS + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "   ✗ $EVENT_ID: meta=$HAS_META jpeg=$([ -n "$HAS_JPEG" ] && echo 'yes' || echo 'NO')"
        fi
    done

    echo ""
    echo " Audit result: $PASS/$SAMPLE_SIZE passed ($FAIL_COUNT missing)"
else
    echo " ⚠ Cannot sample — no events in DB"
fi

# ── Final Report ──
echo ""
echo "========================================"
if [ "$FAIL_COUNT" -eq 0 ] && [ "$TOTAL_EVENTS" -gt 0 ]; then
    echo " ✓ SOAK TEST PASSED"
    echo "   ${TOTAL_EVENTS} events, 0 missing evidence"
else
    echo " ✗ SOAK TEST FAILED"
    [ "$TOTAL_EVENTS" -eq 0 ] && echo "   No events recorded"
    [ "$FAIL_COUNT" -gt 0 ] && echo "   $FAIL_COUNT events missing evidence"
fi
echo "========================================"

# Check log for crashes
if [ -f "$LOG_FILE" ]; then
    CRASH_COUNT=$(grep -c "ERROR\|CRITICAL\|Traceback" "$LOG_FILE" 2>/dev/null || echo "0")
    echo " Log errors: $CRASH_COUNT"
    if [ "$CRASH_COUNT" -gt 0 ]; then
        echo " Last 5 errors:"
        grep "ERROR\|CRITICAL" "$LOG_FILE" | tail -5 | sed 's/^/   /'
    fi
fi

exit $FAIL_COUNT
