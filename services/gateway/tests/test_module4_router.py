"""Exercises /api/module4 endpoints (presets, verify) directly and end-to-end."""

from __future__ import annotations

import pytest
from gateway.routers.module4 import (
    VerifyRequest,
    get_presets,
    verify_action,
)
from verification.types import Decision


@pytest.mark.anyio
async def test_module4_presets_count() -> None:
    data = await get_presets()
    assert len(data) == 6
    assert "nominal_safe" in data
    assert "cyclone_survival" in data
    assert "peak_power_sharing" in data
    assert "unsafe_undervolt" in data
    assert "unsafe_overvolt" in data
    assert "unsafe_overload" in data


@pytest.mark.anyio
async def test_module4_3_approved_actions() -> None:
    presets = await get_presets()
    for key in ("nominal_safe", "cyclone_survival", "peak_power_sharing"):
        req = VerifyRequest(**presets[key]["payload"])
        res = await verify_action(req)
        assert res.decision == Decision.DECISION_APPROVE
        assert len(res.violations) == 0
        assert "verified safe" in res.causal_log.text
        # Verify non-flat live solved bus voltages
        assert all(0.95 <= b.voltage_pu <= 1.05 for b in res.buses)
        assert any(b.island == "Nainativu Island (Grid 1)" for b in res.buses)
        assert any(b.island == "Delft Island (Grid 3)" for b in res.buses)


@pytest.mark.anyio
async def test_module4_3_rejected_actions() -> None:
    presets = await get_presets()
    for key in ("unsafe_undervolt", "unsafe_overvolt", "unsafe_overload"):
        req = VerifyRequest(**presets[key]["payload"])
        res = await verify_action(req)
        assert res.decision == Decision.DECISION_REJECT
        assert len(res.violations) > 0
        assert res.rejection_severity > 0.0
        assert len(res.causal_log.grounded_entities) > 0

    # Explicit check for solved grid panels showing true electrical state
    req_overvolt = VerifyRequest(**presets["unsafe_overvolt"]["payload"])
    res_overvolt = await verify_action(req_overvolt)
    assert any(b.status == "OVERVOLTAGE" and b.voltage_pu > 1.05 for b in res_overvolt.buses)

    req_undervolt = VerifyRequest(**presets["unsafe_undervolt"]["payload"])
    res_undervolt = await verify_action(req_undervolt)
    assert any(
        line.line_name.upper() in ("LINE_2_3", "E_CRIT_1") and not line.is_closed
        for line in res_undervolt.lines
    )
