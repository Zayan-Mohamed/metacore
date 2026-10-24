"""The live store (ADR 0006, decision 5): what each hourly run fetched, produced and scored.

Public weather and derived values only -- never a CEB ledger row. One row per run, and one row
per stage output keyed to it, so any screen can be rebuilt from the database alone and any
forecast traced back to the exact weather it was computed from.
"""

from __future__ import annotations

import os

import psycopg
from psycopg.types.json import Jsonb

DEFAULT_URL = "postgresql://metacore@127.0.0.1:5433/metacore"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            BIGSERIAL PRIMARY KEY,
    issued_for    TEXT NOT NULL,              -- the local hour the forecast is issued at
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    status        TEXT NOT NULL DEFAULT 'running',   -- running | ok | partial | failed
    stage_ms      JSONB NOT NULL DEFAULT '{}'::jsonb,
    errors        JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS runs_issued_for ON runs (issued_for);

CREATE TABLE IF NOT EXISTS weather_fetches (
    run_id     BIGINT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    fetched_at TEXT NOT NULL,
    model      TEXT NOT NULL,
    meta       JSONB NOT NULL,
    rows       JSONB NOT NULL                 -- per island, NASA POWER layout
);

CREATE TABLE IF NOT EXISTS forecasts (       -- Module 1
    run_id     BIGINT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    issued_for TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    payload    JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS uncertainty (     -- Module 2
    run_id     BIGINT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    payload    JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS grid_checks (     -- Module 4, one row per hour checked
    run_id     BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    hour       TEXT NOT NULL,
    lead_h     INTEGER NOT NULL,
    payload    JSONB NOT NULL,
    PRIMARY KEY (run_id, lead_h)
);

-- Scored after the fact: a forecast for hour H is scored by the run issued at H, whose "now"
-- value is computed from the weather that actually arrived for H.
CREATE TABLE IF NOT EXISTS live_scores (
    forecast_run BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    scored_by    BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    target_hour  TEXT NOT NULL,
    lead_h       INTEGER NOT NULL,
    node         TEXT NOT NULL,
    model        TEXT NOT NULL,
    forecast     DOUBLE PRECISION NOT NULL,
    realised     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (forecast_run, lead_h, node, model)
);
CREATE INDEX IF NOT EXISTS live_scores_target ON live_scores (target_hour);
"""


def url() -> str:
    return os.environ.get("METACORE_DB_URL", DEFAULT_URL)


def connect() -> psycopg.Connection:
    return psycopg.connect(url(), autocommit=True)


def migrate(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


__all__ = ["Jsonb", "connect", "migrate", "url"]
