#!/bin/bash
# IntelFactor.ai — Station-to-Hub SQLite Sync
# Cron-friendly: rsyncs SQLite files from station to hub data directory.
#
# Usage:
#   ./deploy/station/sync_to_hub.sh                          # defaults
#   ./deploy/station/sync_to_hub.sh --hub 192.168.1.100      # specific hub
#
# Cron (every 5 minutes):
#   */5 * * * * /opt/intelfactor/sync_to_hub.sh >> /opt/intelfactor/logs/sync.log 2>&1

set -euo pipefail

STATION_DATA="/opt/intelfactor/data"
HUB_HOST="${HUB_HOST:-hub.local}"
HUB_DATA="/opt/intelfactor/hub-data/stations"
HUB_USER="${HUB_USER:-intelfactor}"
STATION_ID="${STATION_ID:-$(hostname)}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --hub) HUB_HOST="$2"; shift 2 ;;
        --station-id) STATION_ID="$2"; shift 2 ;;
        --data) STATION_DATA="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--hub HOST] [--station-id ID]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Files to sync
DB_FILES=(
    "accumulator.db"
    "triples.db"
)

echo "[$TIMESTAMP] Syncing $STATION_ID → $HUB_HOST"

for DB in "${DB_FILES[@]}"; do
    SRC="${STATION_DATA}/${DB}"
    if [ ! -f "$SRC" ]; then
        echo "  Skip: $DB (not found)"
        continue
    fi

    # Create a WAL-safe snapshot (SQLite VACUUM INTO or backup)
    SNAPSHOT="/tmp/intelfactor_sync_${DB}"
    sqlite3 "$SRC" ".backup '$SNAPSHOT'" 2>/dev/null || {
        # Fallback: plain copy (safe if WAL checkpoint done)
        cp "$SRC" "$SNAPSHOT"
    }

    # Rsync to hub
    DEST_DIR="${HUB_DATA}/${STATION_ID}"
    rsync -az --timeout=30 "$SNAPSHOT" "${HUB_USER}@${HUB_HOST}:${DEST_DIR}/${DB}" 2>&1 && {
        echo "  ✓ $DB synced"
    } || {
        echo "  ✗ $DB sync failed (hub unreachable?)"
    }

    rm -f "$SNAPSHOT"
done

echo "[$TIMESTAMP] Sync complete"
