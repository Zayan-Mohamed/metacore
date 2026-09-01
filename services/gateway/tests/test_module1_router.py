"""Exercises /api/module1/assemble end-to-end: real subprocess, real module1.assemble,
real data/ (or .synthetic/) artifacts. This is the wiring test (gateway -> subprocess
-> module1); the assembly logic itself is tested in
services/learned/module1_state_forecasting/tests/test_assemble.py."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def test_module1_assemble_rejects_unknown_island() -> None:
    r = client.post("/api/module1/assemble", json={"island": "atlantis"})
    assert r.status_code == 422


def test_module1_assemble_rejects_unknown_scenario() -> None:
    r = client.post("/api/module1/assemble", json={"scenario": "meteor"})
    assert r.status_code == 422


def test_module1_assemble_returns_contract_shaped_state() -> None:
    r = client.post(
        "/api/module1/assemble",
        json={"island": "eluvaitivu", "scenario": "normal"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["feature_names"]) == 28
    assert len(data["quality_mask"]) == 28
    assert data["node_count"] == len(data["node_features"])
    assert all(len(row) == 28 for row in data["node_features"])
    assert data["embedding_dim"] == 64
    assert data["has_embedding"] is False
    assert data["schema_version"] == "1.0"
    assert data["observed_fraction"] == pytest.approx(12 / 28, abs=1e-4)
    assert data["data_source"] in {"real", "synthetic"}


def test_module1_blackout_drops_observed_fraction() -> None:
    r = client.post(
        "/api/module1/assemble",
        json={"island": "nainativu", "scenario": "blackout"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["degraded"] is True
    assert data["observed_fraction"] == pytest.approx(8 / 28, abs=1e-4)


def test_module1_cyclone_is_flagged_out_of_distribution() -> None:
    r = client.post(
        "/api/module1/assemble",
        json={"island": "delft", "scenario": "cyclone"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["out_of_distribution"] is True
