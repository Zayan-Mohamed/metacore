"""The live system's read side (ADR 0006): what the hourly pipeline stored, for the control room.

Read-only. The orchestrator (services/orchestrator, ``python -m app.pipeline run``) is the only
writer; this never fetches weather or runs a model, so opening the dashboard costs nothing and
shows exactly what the pipeline computed. 503 when the store is unreachable.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/live", tags=["live"])

DB_URL = os.environ.get("METACORE_DB_URL", "postgresql://metacore@127.0.0.1:5433/metacore")

# Module 3 is not wired into the live chain yet: its actions and gates are built for the retired
# delft-3island network. The screen shows this instead of a decision.
M3_STATUS = {
    "connected": False,
    "reason": ("Module 3's control actions use the retired delft-3island network (N1..N12, tie "
               "lines), so Module 4 rejects them as malformed on the real four-island grid, and "
               "its gate expects SCADA inputs (tripped lines, generation available) that do not "
               "exist. Migration list: docs/contracts/m3-live-migration.md."),
    "owner": "Saabir (Module 3)",
}


def _conn():
    try:
        import psycopg
        return psycopg.connect(DB_URL, autocommit=True, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"live store unreachable: {exc}") from exc


@router.get("/latest")
def latest() -> dict[str, Any]:
    """The newest run with a forecast: weather, M1, M2, M4 for every hour checked, run health."""
    with _conn() as c:
        row = c.execute(
            "SELECT r.id, r.issued_for, r.started_at, r.finished_at, r.status, r.stage_ms, "
            "r.errors, f.payload, w.meta FROM runs r JOIN forecasts f ON f.run_id = r.id "
            "JOIN weather_fetches w ON w.run_id = r.id ORDER BY r.id DESC LIMIT 1").fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="no live run stored yet; start "
                                "`python -m app.pipeline run` in services/orchestrator")
        run_id = row[0]
        m2 = c.execute("SELECT payload FROM uncertainty WHERE run_id = %s", (run_id,)).fetchone()
        grid = c.execute("SELECT payload FROM grid_checks WHERE run_id = %s ORDER BY lead_h",
                         (run_id,)).fetchall()
        last = c.execute("SELECT max(started_at), count(*) FROM runs").fetchone()
        try:
            ledger = c.execute("SELECT file, received, status, result FROM ledger_updates "
                               "ORDER BY received DESC LIMIT 5").fetchall()
        except Exception:  # noqa: BLE001 -- a store from before the ledger feed existed
            ledger = []
    m1 = dict(row[7])
    for n in m1.get("nodes", []):
        n.pop("embedding", None)  # 64 numbers per node; the screen does not draw them
    m1.pop("contract_state_prev", None)
    return {
        "run": {"id": run_id, "issued_for": row[1], "started_at": row[2].isoformat(),
                "finished_at": row[3].isoformat() if row[3] else None, "status": row[4],
                "stage_ms": row[5], "errors": row[6], "runs_stored": last[1]},
        "weather_fetch": row[8],
        "module1": m1,
        "module2": m2[0] if m2 else None,
        "module3": M3_STATUS,
        "module4": [g[0] for g in grid],
        "ledger": {"latest_month": m1.get("ledger_latest_month"), "source": m1.get("ledger_source"),
                   "recent_files": [{"file": f, "received": r, "status": s,
                                     "months_added": res.get("months_added", []),
                                     "failures": res.get("failures", []),
                                     "notes": res.get("notes", [])} for f, r, s, res in ledger]},
    }


@router.get("/history")
def history(hours: int = Query(48, ge=1, le=24 * 60)) -> dict[str, Any]:
    """Per run: issued hour, status, each node's current score, the grid's worst margins."""
    with _conn() as c:
        rows = c.execute(
            "SELECT r.id, r.issued_for, r.status, f.payload->'nodes', "
            "(SELECT payload FROM grid_checks g WHERE g.run_id = r.id AND g.lead_h = 0) "
            "FROM runs r LEFT JOIN forecasts f ON f.run_id = r.id "
            "ORDER BY r.id DESC LIMIT %s", (hours,)).fetchall()
    out = []
    for run_id, issued, status, nodes, grid in reversed(rows):
        out.append({
            "run_id": run_id, "issued_for": issued, "status": status,
            "current": {n["id"]: n["current"] for n in nodes} if nodes else None,
            "v_min_pu": min(b["voltage_pu"] for b in grid["buses"]) if grid else None,
            "max_line_pct": max((ln["loading_pct"] or 0) for ln in grid["lines"]) if grid else None,
            "decision": grid["decision"] if grid else None,
        })
    return {"runs": out}


@router.get("/scores")
def scores(hours: int = Query(168, ge=1, le=24 * 365)) -> dict[str, Any]:
    """After-the-fact scores of forecasts whose target hour arrived in the last ``hours``."""
    with _conn() as c:
        rows = c.execute(
            "SELECT model, lead_h, count(*), sqrt(avg((forecast - realised)^2)), "
            "avg(abs(forecast - realised)) FROM live_scores WHERE scored_by IN "
            "(SELECT id FROM runs ORDER BY id DESC LIMIT %s) GROUP BY model, lead_h "
            "ORDER BY model, lead_h", (hours,)).fetchall()
        span = c.execute("SELECT min(target_hour), max(target_hour), count(DISTINCT target_hour) "
                         "FROM live_scores").fetchone()
    by_model: dict[str, Any] = {}
    for model, lead, n, rmse, mae in rows:
        m = by_model.setdefault(model, {"by_lead": {}, "n": 0, "se": 0.0})
        m["by_lead"][int(lead)] = {"n": n, "rmse": float(rmse), "mae": float(mae)}
        m["n"] += n
        m["se"] += float(rmse) ** 2 * n
    for m in by_model.values():
        m["rmse"] = (m.pop("se") / m["n"]) ** 0.5 if m["n"] else None
    return {"window_runs": hours, "first_scored_hour": span[0], "last_scored_hour": span[1],
            "hours_scored": span[2], "models": by_model}
