-- IntelFactor Production Visibility Schema
-- Migration 001: Production metrics tables
-- Target: /opt/intelfactor/data/local.db (alongside existing events, triples, accumulator)
--
-- These tables sit on top of the existing events table.
-- The metrics engine reads from events and writes aggregated metrics here.
-- Batch sync uploads metrics to DynamoDB alongside inspection events.

-- Production counts: rolled up per station, per hour bucket
CREATE TABLE IF NOT EXISTS production_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    hour_bucket TEXT NOT NULL,          -- ISO 8601 hour: '2026-02-16T14:00:00'
    shift_id TEXT,                       -- 'shift_1', 'shift_2', 'shift_3' or NULL
    object_class TEXT NOT NULL,          -- COCO class: 'knife', 'scissors', etc.
    unit_count INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT,                     -- ISO timestamp of first detection in bucket
    last_seen TEXT,                      -- ISO timestamp of last detection in bucket
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    synced INTEGER DEFAULT 0,
    UNIQUE(station_id, hour_bucket, object_class)
);

-- Cycle times: time between consecutive detections at same station
CREATE TABLE IF NOT EXISTS cycle_times (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    inspection_id TEXT NOT NULL,         -- the detection that ended this cycle
    prev_inspection_id TEXT,             -- the detection that started this cycle
    cycle_seconds REAL NOT NULL,         -- time delta in seconds
    object_class TEXT NOT NULL,
    timestamp TEXT NOT NULL,             -- when this cycle completed
    shift_id TEXT,
    synced INTEGER DEFAULT 0
);

-- Station utilization: tracks active/idle state transitions
CREATE TABLE IF NOT EXISTS station_utilization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    state TEXT NOT NULL,                 -- 'active' or 'idle'
    started_at TEXT NOT NULL,
    ended_at TEXT,                       -- NULL if current state
    duration_seconds REAL,              -- computed when state ends
    shift_id TEXT,
    synced INTEGER DEFAULT 0
);

-- Shift summaries: auto-generated at shift boundaries
CREATE TABLE IF NOT EXISTS shift_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    shift_id TEXT NOT NULL,
    shift_date TEXT NOT NULL,            -- '2026-02-16'
    shift_start TEXT NOT NULL,
    shift_end TEXT NOT NULL,
    total_units INTEGER DEFAULT 0,
    avg_cycle_seconds REAL,
    min_cycle_seconds REAL,
    max_cycle_seconds REAL,
    cycle_stddev REAL,
    utilization_pct REAL,               -- 0.0 to 100.0
    total_active_seconds REAL,
    total_idle_seconds REAL,
    longest_idle_seconds REAL,
    longest_idle_start TEXT,
    defect_count INTEGER DEFAULT 0,     -- FAIL verdicts (future use)
    review_count INTEGER DEFAULT 0,     -- REVIEW verdicts (future use)
    pass_count INTEGER DEFAULT 0,
    narrative TEXT,                      -- AI-generated shift summary
    created_at TEXT DEFAULT (datetime('now')),
    synced INTEGER DEFAULT 0,
    UNIQUE(station_id, shift_id, shift_date)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_counts_station_hour ON production_counts(station_id, hour_bucket);
CREATE INDEX IF NOT EXISTS idx_counts_synced ON production_counts(synced) WHERE synced = 0;
CREATE INDEX IF NOT EXISTS idx_cycle_station_time ON cycle_times(station_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cycle_synced ON cycle_times(synced) WHERE synced = 0;
CREATE INDEX IF NOT EXISTS idx_util_station_state ON station_utilization(station_id, state, started_at);
CREATE INDEX IF NOT EXISTS idx_util_synced ON station_utilization(synced) WHERE synced = 0;
CREATE INDEX IF NOT EXISTS idx_shift_synced ON shift_summaries(synced) WHERE synced = 0;
