"""
IntelFactor Metrics Sync
=========================
Addition to the existing batch_sync.py for syncing production metrics
to DynamoDB. This runs on the same 60s systemd timer.

Integration: Import and call sync_metrics() from your existing batch_sync.py
after the event sync completes.

Deploy to: /opt/intelfactor/scripts/metrics_sync.py
"""

import sqlite3
import boto3
import json
import logging
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger("intelfactor.sync.metrics")

# DynamoDB table for production metrics (create via CloudFormation or console)
METRICS_TABLE = "intelfactor-production-metrics"
SHIFTS_TABLE = "intelfactor-shift-summaries"

DB_PATH = "/opt/intelfactor/data/local.db"
STATION_ID = "SNF-Vision-1"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _decimal(val):
    """Convert float to Decimal for DynamoDB."""
    if val is None:
        return None
    return Decimal(str(round(val, 4)))


def sync_metrics(dynamodb=None):
    """
    Sync unsynced production counts and shift summaries to DynamoDB.
    Call this from your existing batch_sync.py main loop.
    """
    dynamodb = dynamodb or boto3.resource("dynamodb", region_name="us-west-2")
    conn = _connect()

    try:
        # ---- Sync production counts ----
        metrics_table = dynamodb.Table(METRICS_TABLE)
        rows = conn.execute("""
            SELECT id, station_id, hour_bucket, shift_id, object_class,
                   unit_count, first_seen, last_seen
            FROM production_counts
            WHERE synced = 0
            LIMIT 25
        """).fetchall()

        if rows:
            with metrics_table.batch_writer() as batch:
                for row in rows:
                    item = {
                        "pk": f"STATION#{row['station_id']}",
                        "sk": f"HOUR#{row['hour_bucket']}#CLASS#{row['object_class']}",
                        "station_id": row["station_id"],
                        "hour_bucket": row["hour_bucket"],
                        "shift_id": row["shift_id"] or "unknown",
                        "object_class": row["object_class"],
                        "unit_count": row["unit_count"],
                        "first_seen": row["first_seen"],
                        "last_seen": row["last_seen"],
                        "record_type": "production_count",
                        "synced_at": datetime.utcnow().isoformat(),
                    }
                    batch.put_item(Item=item)

            # Mark synced
            ids = [str(r["id"]) for r in rows]
            conn.execute(
                f"UPDATE production_counts SET synced = 1 WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            conn.commit()
            logger.info(f"Synced {len(rows)} production count records")

        # ---- Sync shift summaries ----
        shifts_table = dynamodb.Table(SHIFTS_TABLE)
        shift_rows = conn.execute("""
            SELECT * FROM shift_summaries WHERE synced = 0 LIMIT 10
        """).fetchall()

        if shift_rows:
            with shifts_table.batch_writer() as batch:
                for row in shift_rows:
                    item = {
                        "pk": f"STATION#{row['station_id']}",
                        "sk": f"SHIFT#{row['shift_date']}#{row['shift_id']}",
                        "station_id": row["station_id"],
                        "shift_id": row["shift_id"],
                        "shift_date": row["shift_date"],
                        "shift_start": row["shift_start"],
                        "shift_end": row["shift_end"],
                        "total_units": row["total_units"],
                        "avg_cycle_seconds": _decimal(row["avg_cycle_seconds"]),
                        "min_cycle_seconds": _decimal(row["min_cycle_seconds"]),
                        "max_cycle_seconds": _decimal(row["max_cycle_seconds"]),
                        "cycle_stddev": _decimal(row["cycle_stddev"]),
                        "utilization_pct": _decimal(row["utilization_pct"]),
                        "total_active_seconds": _decimal(row["total_active_seconds"]),
                        "total_idle_seconds": _decimal(row["total_idle_seconds"]),
                        "longest_idle_seconds": _decimal(row["longest_idle_seconds"]),
                        "pass_count": row["pass_count"],
                        "defect_count": row["defect_count"],
                        "review_count": row["review_count"],
                        "narrative": row["narrative"],
                        "record_type": "shift_summary",
                        "synced_at": datetime.utcnow().isoformat(),
                    }
                    batch.put_item(Item=item)

            ids = [str(r["id"]) for r in shift_rows]
            conn.execute(
                f"UPDATE shift_summaries SET synced = 1 WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            )
            conn.commit()
            logger.info(f"Synced {len(shift_rows)} shift summary records")

        # ---- Sync cycle time aggregates (sampled, not individual) ----
        # Individual cycle times stay on-device. We sync hourly aggregates.
        cycle_agg = conn.execute("""
            SELECT
                station_id,
                strftime('%Y-%m-%dT%H:00:00', timestamp) as hour_bucket,
                shift_id,
                COUNT(*) as sample_count,
                AVG(cycle_seconds) as avg_cycle,
                MIN(cycle_seconds) as min_cycle,
                MAX(cycle_seconds) as max_cycle
            FROM cycle_times
            WHERE synced = 0
            GROUP BY station_id, strftime('%Y-%m-%dT%H:00:00', timestamp), shift_id
        """).fetchall()

        if cycle_agg:
            for row in cycle_agg:
                metrics_table.put_item(Item={
                    "pk": f"STATION#{row['station_id']}",
                    "sk": f"CYCLE#{row['hour_bucket']}",
                    "station_id": row["station_id"],
                    "hour_bucket": row["hour_bucket"],
                    "shift_id": row["shift_id"] or "unknown",
                    "sample_count": row["sample_count"],
                    "avg_cycle_seconds": _decimal(row["avg_cycle"]),
                    "min_cycle_seconds": _decimal(row["min_cycle"]),
                    "max_cycle_seconds": _decimal(row["max_cycle"]),
                    "record_type": "cycle_time_aggregate",
                    "synced_at": datetime.utcnow().isoformat(),
                })

            # Mark all unsynced cycle times as synced
            conn.execute("UPDATE cycle_times SET synced = 1 WHERE synced = 0")
            conn.commit()
            logger.info(f"Synced {len(cycle_agg)} cycle time aggregates")

    except Exception as e:
        logger.error(f"Metrics sync error: {e}", exc_info=True)
    finally:
        conn.close()


# ---- DynamoDB Table Creation (run once) ----

def create_tables():
    """
    Create DynamoDB tables for production metrics.
    Run once: python -c "from metrics_sync import create_tables; create_tables()"
    """
    dynamodb = boto3.resource("dynamodb", region_name="us-west-2")

    # Production metrics (counts + cycle time aggregates)
    dynamodb.create_table(
        TableName=METRICS_TABLE,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    # Shift summaries
    dynamodb.create_table(
        TableName=SHIFTS_TABLE,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )

    print(f"Created tables: {METRICS_TABLE}, {SHIFTS_TABLE}")
    print("Tables will be ready in ~30 seconds")


if __name__ == "__main__":
    # Can run standalone for testing
    sync_metrics()
