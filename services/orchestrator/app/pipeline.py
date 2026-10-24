"""The live Metacore pipeline (ADR 0006): one run per hour, every module in its own process.

    fetch (the only network call) -> M1 forecast + state + snapshots -> M2 uncertainty
        -> [M3: not connected, see docs/contracts/m3-live-migration.md] -> M4 grid check
        -> store everything -> score earlier forecasts whose hour has now arrived

Each stage is a subprocess speaking JSON, the same way the gateway calls the modules, so the
learned models and OpenDSS never share a process (ADR 0003) and a stage that fails is recorded
without taking the others down. A run that loses M2 or M4 is stored as ``partial``.

    python -m app.pipeline once            # run the current hour now
    python -m app.pipeline run             # run now, then at five past every hour, forever
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from app import db

REPO = Path(__file__).resolve().parents[3]
APP = Path(__file__).resolve().parent
PYTHONPATH = os.pathsep.join(str(p) for p in (
    REPO / "services" / "learned" / "module1_state_forecasting" / "src",
    REPO / "packages" / "contracts" / "python",
    REPO / "deterministic-core" / "module4_verification" / "src",
    REPO / "deterministic-core" / "module4_semantic_translation" / "src",
))
TIMEOUT_S = 300
MODELS = ("stgnn", "lstm_1d_residual", "persistence")


class StageFailed(RuntimeError):
    pass


def _run(stage: str, argv: list[str], stdin: dict | None = None) -> tuple[dict, float]:
    env = {**os.environ, "PYTHONPATH": PYTHONPATH + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, *argv], input=json.dumps(stdin) if stdin else None,
                          capture_output=True, text=True, timeout=TIMEOUT_S, cwd=REPO, env=env)
    ms = (time.perf_counter() - t0) * 1000
    if proc.returncode != 0:
        raise StageFailed(f"{stage}: exit {proc.returncode}: {proc.stderr.strip()[-1500:]}")
    try:
        return json.loads(proc.stdout), ms
    except json.JSONDecodeError as exc:
        raise StageFailed(f"{stage}: no JSON on stdout: {proc.stderr.strip()[-800:]}") from exc


def current_hour() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:00")


def run_once(now: str | None = None) -> dict:
    now = now or current_hour()
    conn = db.connect()
    db.migrate(conn)
    run_id = conn.execute("INSERT INTO runs (issued_for) VALUES (%s) RETURNING id",
                          (now,)).fetchone()[0]
    stage_ms: dict[str, float] = {}
    errors: list[str] = []
    status = "ok"
    try:
        weather, stage_ms["fetch"] = _run("fetch", ["-m", "module1.live", "fetch"])
        conn.execute(
            "INSERT INTO weather_fetches (run_id, fetched_at, model, meta, rows) "
            "VALUES (%s, %s, %s, %s, %s)",
            (run_id, weather["meta"]["fetched_at"], weather["meta"]["model"],
             db.Jsonb(weather["meta"]), db.Jsonb(weather["rows"])))

        m1, stage_ms["module1"] = _run("module1", ["-m", "module1.live", "step", "--now", now],
                                       weather)
        conn.execute("INSERT INTO forecasts (run_id, issued_for, input_sha256, payload) "
                     "VALUES (%s, %s, %s, %s)",
                     (run_id, now, m1["input_sha256"], db.Jsonb(m1)))
    except StageFailed as exc:  # no forecast, nothing downstream can run
        errors.append(str(exc))
        _finish(conn, run_id, "failed", stage_ms, errors)
        return {"run_id": run_id, "status": "failed", "errors": errors}

    node_ids = [n["id"] for n in m1["nodes"]]
    try:
        m2, stage_ms["module2"] = _run("module2", [str(APP / "m2_runner.py")], {
            "node_ids": node_ids,
            "node_features": m1["contract_state"]["node_features"],
            "node_features_prev": m1["contract_state_prev"]["node_features"],
            "quality_mask": m1["contract_state"]["quality_mask"]})
        conn.execute("INSERT INTO uncertainty (run_id, payload) VALUES (%s, %s)",
                     (run_id, db.Jsonb(m2)))
    except StageFailed as exc:
        errors.append(str(exc))
        status = "partial"

    try:
        t = 23  # the islands block holds 24 h of history, then the forecast hours
        met = [{k: {isl: d[k][t + lead] for isl, d in m1["islands"].items()}
                for k in ("wind_50m_ms", "precip_mm_hr")}
               for lead in range(len(m1["snapshots"]))]
        m4, stage_ms["module4"] = _run("module4", [str(APP / "m4_runner.py")],
                                       {"snapshots": m1["snapshots"], "meteorology": met})
        for h in m4["hours"]:
            conn.execute("INSERT INTO grid_checks (run_id, hour, lead_h, payload) "
                         "VALUES (%s, %s, %s, %s)",
                         (run_id, h["hour"], h["lead_h"],
                          db.Jsonb({**h, "v_min_pu": m4["v_min_pu"], "v_max_pu": m4["v_max_pu"]})))
    except StageFailed as exc:
        errors.append(str(exc))
        status = "partial"

    t0 = time.perf_counter()
    stage_ms["scored"] = score_arrivals(conn, run_id, now, m1)
    stage_ms["scoring"] = (time.perf_counter() - t0) * 1000
    _finish(conn, run_id, status, stage_ms, errors)
    return {"run_id": run_id, "status": status, "errors": errors, "stage_ms": stage_ms}


def score_arrivals(conn, run_id: int, now: str, m1: dict) -> int:
    """Score every stored forecast that targeted ``now``, against ``now``'s realised score."""
    realised = {n["id"]: n["current"] for n in m1["nodes"]}
    rows = conn.execute(
        "SELECT r.id, f.payload FROM runs r JOIN forecasts f ON f.run_id = r.id "
        "WHERE r.id <> %s AND f.payload->'horizon_hours' ? %s", (run_id, now)).fetchall()
    n = 0
    for forecast_run, payload in rows:
        lead = payload["horizon_hours"].index(now) + 1
        for node in payload["nodes"]:
            for model in MODELS:
                conn.execute(
                    "INSERT INTO live_scores (forecast_run, scored_by, target_hour, lead_h, node, "
                    "model, forecast, realised) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT DO NOTHING",
                    (forecast_run, run_id, now, lead, node["id"], model,
                     node["forecast"][model][lead - 1], realised[node["id"]]))
                n += 1
    return n


def _finish(conn, run_id: int, status: str, stage_ms: dict, errors: list[str]) -> None:
    conn.execute("UPDATE runs SET finished_at = now(), status = %s, stage_ms = %s, errors = %s "
                 "WHERE id = %s",
                 (status, db.Jsonb({k: round(v, 1) for k, v in stage_ms.items()}),
                  db.Jsonb(errors), run_id))


def _sleep_until_next(minute: int = 5) -> None:
    now = datetime.now().astimezone()
    nxt = (now + timedelta(hours=1)).replace(minute=minute, second=0, microsecond=0)
    if now.minute < minute:
        nxt = now.replace(minute=minute, second=0, microsecond=0)
    time.sleep(max(1.0, (nxt - now).total_seconds()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["once", "run"])
    ap.add_argument("--now", default=None, help="hour to run (once only; default: current hour)")
    args = ap.parse_args(argv)
    if args.command == "once":
        print(json.dumps(run_once(args.now)))
        return 0
    while True:
        hour = current_hour()
        conn = db.connect()
        db.migrate(conn)
        done = conn.execute("SELECT 1 FROM runs WHERE issued_for = %s AND status <> 'failed'",
                            (hour,)).fetchone()
        conn.close()
        if not done:
            try:
                print(json.dumps(run_once(hour)), flush=True)
            except Exception as exc:  # noqa: BLE001 -- the loop must outlive one bad hour
                print(json.dumps({"hour": hour, "status": "crashed", "error": repr(exc)}),
                      flush=True)
        _sleep_until_next()


if __name__ == "__main__":
    raise SystemExit(main())
