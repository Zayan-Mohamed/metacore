"""/api/live/*: the control room's read side. Needs the local live store; skipped without one."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)


def _latest():
    r = client.get("/api/live/latest")
    if r.status_code in (503, 404):
        pytest.skip(f"live store unavailable or empty: {r.json().get('detail')}")
    return r


def test_latest_run_carries_every_module_and_says_m3_is_not_connected() -> None:
    r = _latest()
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["module1"]["kind"] == "live" and len(d["module1"]["nodes"]) == 10
    assert all("embedding" not in n for n in d["module1"]["nodes"])
    assert d["module3"]["connected"] is False
    if d["module4"]:
        assert [h["lead_h"] for h in d["module4"]] == list(range(len(d["module4"])))


def test_history_and_scores_answer() -> None:
    _latest()
    assert client.get("/api/live/history?hours=24").status_code == 200
    s = client.get("/api/live/scores?hours=24")
    assert s.status_code == 200 and "models" in s.json()
