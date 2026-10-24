"""/api/module1/data-health and /graph: the dashboard's monitoring pages over HTTP."""
from __future__ import annotations

from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_data_health_runs_every_gate_and_never_shows_ledger_figures() -> None:
    r = client.get("/api/module1/data-health")
    assert r.status_code == 200, r.text
    d = r.json()
    ids = {g["id"] for g in d["gates"]}
    assert ids == {"ceb_reconcile", "nasa_power_check", "open_meteo_check", "load_check",
                   "scenario_check"}
    assert d["summary"]["failed"] == 0, [g for g in d["gates"] if g["status"] != "pass"]
    text = r.text.lower()
    for forbidden in ("diesel_l", "cost_rs", "units_kwh", "oil_l", "barrel"):
        assert forbidden not in text, forbidden


def test_graph_has_no_inter_island_edge() -> None:
    r = client.get("/api/module1/graph")
    assert r.status_code == 200, r.text
    d = r.json()
    island = {n["id"]: n["island"] for n in d["nodes"]}
    assert all(island[a] == island[b] for a, b in d["edges"])
    assert len(d["a_hat"]) == len(d["nodes"])
    assert "open_meteo" in d["ablations"] and "nasa_power" in d["ablations"]
