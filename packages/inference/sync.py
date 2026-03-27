"""
IntelFactor.ai — Station-to-Hub Sync Service
Batch syncs SQLite data from station nodes into hub PostgreSQL.

Design:
- Scans a shared directory for station SQLite files (copied via rsync/NFS/USB).
- Upserts new events and triples into Postgres.
- Tracks sync watermarks to avoid reprocessing.
- Runs as a simple polling loop (no Kafka needed for this).

This is the "simplest first" approach from the architecture doc:
SQLite → Postgres batch sync, not real-time streaming.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SyncService:
    """
    Syncs station SQLite databases to hub PostgreSQL.

    Expected layout:
        /station-data/
            station_1/accumulator.db
            station_1/triples.db
            station_2/accumulator.db
            station_2/triples.db
    """

    def __init__(
        self,
        station_data_dir: str = "/station-data",
        database_url: str = "",
        sync_interval_sec: int = 300,
    ):
        self.station_data_dir = Path(station_data_dir)
        self.database_url = database_url
        self.sync_interval = sync_interval_sec
        self._pg_conn = None
        self._watermarks: dict[str, str] = {}  # station_id → last synced timestamp

    def start(self) -> None:
        """Connect to Postgres and start sync loop."""
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is required. Set it in the environment or pass database_url=."
            )
        self._connect_pg()
        self._load_watermarks()
        logger.info(
            "Sync service started: dir=%s interval=%ds",
            self.station_data_dir, self.sync_interval,
        )

        while True:
            try:
                self._sync_all_stations()
            except Exception as e:
                logger.error("Sync cycle failed: %s", e)

            time.sleep(self.sync_interval)

    def _connect_pg(self) -> None:
        """Connect to hub PostgreSQL."""
        try:
            import psycopg2
            self._pg_conn = psycopg2.connect(self.database_url)
            self._pg_conn.autocommit = False
            logger.info("Connected to hub Postgres")
        except ImportError:
            raise RuntimeError("psycopg2 required (pip install psycopg2-binary)")

    def _load_watermarks(self) -> None:
        """Load last sync timestamps from Postgres."""
        if self._pg_conn is None:
            return

        cur = self._pg_conn.cursor()

        # Get latest synced timestamp per station from events
        cur.execute("""
            SELECT station_id, MAX(timestamp)
            FROM defect_events
            GROUP BY station_id
        """)
        for station_id, ts in cur.fetchall():
            self._watermarks[f"{station_id}_events"] = str(ts) if ts else ""

        # Get latest synced timestamp per station from triples
        cur.execute("""
            SELECT station_id, MAX(timestamp)
            FROM causal_triples
            GROUP BY station_id
        """)
        for station_id, ts in cur.fetchall():
            self._watermarks[f"{station_id}_triples"] = str(ts) if ts else ""

        logger.info("Loaded watermarks for %d stations", len(set(
            k.rsplit("_", 1)[0] for k in self._watermarks
        )))

    def _sync_all_stations(self) -> None:
        """Scan station data directory and sync each station."""
        if not self.station_data_dir.exists():
            logger.warning("Station data dir not found: %s", self.station_data_dir)
            return

        for station_dir in self.station_data_dir.iterdir():
            if not station_dir.is_dir():
                continue

            station_id = station_dir.name

            # Sync accumulator (defect events + anomaly alerts)
            acc_db = station_dir / "accumulator.db"
            if acc_db.exists():
                self._sync_events(station_id, acc_db)

            # Sync triples
            triple_db = station_dir / "triples.db"
            if triple_db.exists():
                self._sync_triples(station_id, triple_db)

    def _sync_events(self, station_id: str, db_path: Path) -> None:
        """Sync defect events from station SQLite to hub Postgres."""
        watermark = self._watermarks.get(f"{station_id}_events", "")

        try:
            sqlite_conn = sqlite3.connect(str(db_path), timeout=5)
            sqlite_conn.execute("PRAGMA journal_mode=WAL")

            # Read new events since watermark
            if watermark:
                rows = sqlite_conn.execute(
                    "SELECT * FROM defect_events WHERE timestamp > ? ORDER BY timestamp",
                    (watermark,),
                ).fetchall()
            else:
                rows = sqlite_conn.execute(
                    "SELECT * FROM defect_events ORDER BY timestamp"
                ).fetchall()

            if not rows:
                sqlite_conn.close()
                return

            # Get column names
            columns = [desc[0] for desc in sqlite_conn.execute(
                "SELECT * FROM defect_events LIMIT 0"
            ).description]

            sqlite_conn.close()

            # Upsert into Postgres
            cur = self._pg_conn.cursor()
            inserted = 0

            for row in rows:
                data = dict(zip(columns, row))
                cur.execute("""
                    INSERT INTO defect_events
                        (event_id, timestamp, station_id, defect_type, severity,
                         confidence, shift, sku, sop_criterion, model_version, frame_ref)
                    VALUES (%(event_id)s, %(timestamp)s, %(station_id)s, %(defect_type)s,
                            %(severity)s, %(confidence)s, %(shift)s, %(sku)s,
                            %(sop_criterion)s, %(model_version)s, %(frame_ref)s)
                    ON CONFLICT (event_id) DO NOTHING
                """, data)
                inserted += cur.rowcount

            self._pg_conn.commit()

            if inserted > 0:
                # Update watermark
                last_ts = dict(zip(columns, rows[-1]))["timestamp"]
                self._watermarks[f"{station_id}_events"] = last_ts
                logger.info("Synced %d events from %s", inserted, station_id)

        except Exception as e:
            logger.error("Event sync failed for %s: %s", station_id, e)
            if self._pg_conn:
                self._pg_conn.rollback()

    def _sync_triples(self, station_id: str, db_path: Path) -> None:
        """Sync causal triples from station SQLite to hub Postgres."""
        watermark = self._watermarks.get(f"{station_id}_triples", "")

        try:
            sqlite_conn = sqlite3.connect(str(db_path), timeout=5)
            sqlite_conn.execute("PRAGMA journal_mode=WAL")

            if watermark:
                rows = sqlite_conn.execute(
                    "SELECT * FROM causal_triples WHERE timestamp > ? ORDER BY timestamp",
                    (watermark,),
                ).fetchall()
            else:
                rows = sqlite_conn.execute(
                    "SELECT * FROM causal_triples ORDER BY timestamp"
                ).fetchall()

            if not rows:
                sqlite_conn.close()
                return

            columns = [desc[0] for desc in sqlite_conn.execute(
                "SELECT * FROM causal_triples LIMIT 0"
            ).description]

            sqlite_conn.close()

            cur = self._pg_conn.cursor()
            upserted = 0

            for row in rows:
                data = dict(zip(columns, row))
                cur.execute("""
                    INSERT INTO causal_triples
                        (triple_id, timestamp, station_id, defect_event_id, defect_type,
                         defect_severity, cause_parameter, cause_value, cause_target,
                         cause_drift_pct, cause_confidence, cause_explanation_zh,
                         cause_explanation_en, recommendation_id, operator_action,
                         operator_id, outcome_measured, status)
                    VALUES (%(triple_id)s, %(timestamp)s, %(station_id)s, %(defect_event_id)s,
                            %(defect_type)s, %(defect_severity)s, %(cause_parameter)s,
                            %(cause_value)s, %(cause_target)s, %(cause_drift_pct)s,
                            %(cause_confidence)s, %(cause_explanation_zh)s,
                            %(cause_explanation_en)s, %(recommendation_id)s,
                            %(operator_action)s, %(operator_id)s,
                            %(outcome_measured)s, %(status)s)
                    ON CONFLICT (triple_id) DO UPDATE SET
                        operator_action = EXCLUDED.operator_action,
                        operator_id = EXCLUDED.operator_id,
                        outcome_measured = EXCLUDED.outcome_measured,
                        status = EXCLUDED.status
                """, data)
                upserted += 1

            self._pg_conn.commit()

            if upserted > 0:
                last_ts = dict(zip(columns, rows[-1]))["timestamp"]
                self._watermarks[f"{station_id}_triples"] = last_ts
                logger.info("Synced %d triples from %s", upserted, station_id)

        except Exception as e:
            logger.error("Triple sync failed for %s: %s", station_id, e)
            if self._pg_conn:
                self._pg_conn.rollback()


def main():
    """Entry point for sync service container."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [sync] %(levelname)s: %(message)s",
    )

    service = SyncService(
        station_data_dir=os.environ.get("STATION_DATA_DIR", "/station-data"),
        database_url=os.environ.get("DATABASE_URL", ""),
        sync_interval_sec=int(os.environ.get("SYNC_INTERVAL_SEC", "300")),
    )
    service.start()


if __name__ == "__main__":
    main()
