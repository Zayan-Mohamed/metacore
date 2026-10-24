"""The live store and after-the-fact scoring, against a throwaway database.

Needs a PostgreSQL reachable at METACORE_TEST_DB_URL (default: the local live-mode server's
``metacore_test`` database); skipped when there is none, as in CI.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")

from app import db, pipeline  # noqa: E402

URL = os.environ.get("METACORE_TEST_DB_URL", "postgresql://metacore@127.0.0.1:5433/metacore_test")


@pytest.fixture()
def conn():
    try:
        c = psycopg.connect(URL, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("no test database")
    c.execute("DROP TABLE IF EXISTS live_scores, grid_checks, uncertainty, forecasts, "
              "weather_fetches, runs CASCADE")
    db.migrate(c)
    yield c
    c.close()


def _forecast(conn, issued: str, horizon: list[str], value: float) -> int:
    run = conn.execute("INSERT INTO runs (issued_for, status) VALUES (%s, 'ok') RETURNING id",
                       (issued,)).fetchone()[0]
    payload = {"horizon_hours": horizon, "nodes": [
        {"id": "Nainativu/bus", "current": 0.3,
         "forecast": {m: [value + i / 100 for i in range(len(horizon))] for m in pipeline.MODELS}}]}
    conn.execute("INSERT INTO forecasts (run_id, issued_for, input_sha256, payload) "
                 "VALUES (%s, %s, 'x', %s)", (run, issued, db.Jsonb(payload)))
    return run


def test_a_forecast_is_scored_when_its_hour_arrives(conn) -> None:
    hours = [f"2026-10-25T{h:02d}:00" for h in range(1, 7)]
    early = _forecast(conn, "2026-10-25T00:00", hours, 0.40)
    now = conn.execute("INSERT INTO runs (issued_for) VALUES ('2026-10-25T03:00') RETURNING id"
                       ).fetchone()[0]
    m1 = {"nodes": [{"id": "Nainativu/bus", "current": 0.35}]}
    assert pipeline.score_arrivals(conn, now, "2026-10-25T03:00", m1) == len(pipeline.MODELS)
    rows = conn.execute("SELECT forecast_run, lead_h, model, forecast, realised FROM live_scores "
                        "ORDER BY model").fetchall()
    assert {r[0] for r in rows} == {early}
    assert all(r[1] == 3 for r in rows)                      # issued at 00:00, for 03:00
    assert all(abs(r[3] - 0.42) < 1e-9 and r[4] == 0.35 for r in rows)


def test_scoring_twice_does_not_double_count(conn) -> None:
    hours = [f"2026-10-25T{h:02d}:00" for h in range(1, 7)]
    _forecast(conn, "2026-10-25T00:00", hours, 0.40)
    now = conn.execute("INSERT INTO runs (issued_for) VALUES ('2026-10-25T02:00') RETURNING id"
                       ).fetchone()[0]
    m1 = {"nodes": [{"id": "Nainativu/bus", "current": 0.35}]}
    pipeline.score_arrivals(conn, now, "2026-10-25T02:00", m1)
    pipeline.score_arrivals(conn, now, "2026-10-25T02:00", m1)
    assert conn.execute("SELECT count(*) FROM live_scores").fetchone()[0] == len(pipeline.MODELS)
