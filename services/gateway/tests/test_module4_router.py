"""Exercises /api/module4 endpoints (presets, hours, profiles, verify) directly and end-to-end."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from gateway.routers.module4 import (
    HOURS,
    VerifyRequest,
    get_hours,
    get_presets,
    get_profiles,
    verify_action,
)
from verification.types import Decision

APPROVED = ("nominal_safe", "cyclone_survival", "peak_power_sharing")
REJECTED = ("adequacy_breach", "unsafe_undervolt", "unsafe_overvolt", "unsafe_overload")


@pytest.mark.anyio
async def test_module4_presets_hours_profiles() -> None:
    data = await get_presets()
    assert len(data) == 8
    assert set(APPROVED) | set(REJECTED) | {"fragile_export"} == set(data)
    hours = await get_hours()
    assert {h["timestamp"] for h in hours.values()} == {
        "2025-10-05T12:00", "2025-07-20T20:00", "2025-11-28T19:00"}
    profiles = await get_profiles()
    assert {"METACORE-STRICT", "SL-PSR-2016-LV", "EN-50160", "ANSI-C84.1-A"} == set(profiles)


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["plain", "certified"])
async def test_every_preset_gets_its_expected_verdict(mode: str) -> None:
    presets = await get_presets()
    for key, preset in presets.items():
        res = await verify_action(VerifyRequest(**preset["payload"], mode=mode))
        assert res.decision == Decision(preset["expected"][mode]), (key, res.causal_log.text)
        assert res.causal_log.grounded, (key, res.causal_log.ungrounded)


@pytest.mark.anyio
async def test_approved_actions_carry_a_certificate_and_solved_state() -> None:
    presets = await get_presets()
    for key in APPROVED:
        res = await verify_action(VerifyRequest(**presets[key]["payload"]))
        assert res.decision == Decision.DECISION_APPROVE
        assert len(res.violations) == 0
        assert "verified safe" in res.causal_log.text
        assert "0 of 688" in res.causal_log.text
        assert res.certificate["n_scenarios"] == 688 and res.certificate["n_failures"] == 0
        assert res.certificate["upper_bound"] <= 0.01
        assert all(0.95 <= b.voltage_pu <= 1.05 for b in res.buses)
        assert any(b.island == "Nainativu" for b in res.buses)
        assert any(b.island == "Delft-Neduntivu" for b in res.buses)
        assert all(b.island != "Unknown" for b in res.buses)
        assert {a.status for a in res.adequacy} == {"ADEQUATE"}


@pytest.mark.anyio
async def test_the_fragile_export_is_caught_only_by_the_certificate() -> None:
    presets = await get_presets()
    payload = presets["fragile_export"]["payload"]
    res = await verify_action(VerifyRequest(**payload))
    assert res.check_a_decision == Decision.DECISION_APPROVE
    assert res.decision == Decision.DECISION_REJECT
    assert res.certificate["n_failures"] > 0
    assert res.attribution[0]["player"] == "Eluvaitivu"
    assert "points come from Eluvaitivu load uncertainty" in res.causal_log.text
    assert "Eluvaitivu" in res.causal_log.grounded_entities


@pytest.mark.anyio
async def test_rejected_actions_show_the_breach() -> None:
    presets = await get_presets()
    for key in REJECTED:
        res = await verify_action(VerifyRequest(**presets[key]["payload"]))
        assert res.decision == Decision.DECISION_REJECT
        assert len(res.violations) > 0
        assert res.rejection_severity > 0.0
        assert len(res.causal_log.grounded_entities) > 0

    res = await verify_action(VerifyRequest(**presets["unsafe_overvolt"]["payload"]))
    assert any(b.status == "OVERVOLTAGE" and b.voltage_pu > 1.05 for b in res.buses)
    res = await verify_action(VerifyRequest(**presets["unsafe_undervolt"]["payload"]))
    assert any(b.status == "UNDERVOLTAGE" and b.bus_name == "ELU_HYB" for b in res.buses)
    res = await verify_action(VerifyRequest(**presets["unsafe_overload"]["payload"]))
    assert any(line.status == "OVERLOAD" for line in res.lines)
    res = await verify_action(VerifyRequest(**presets["adequacy_breach"]["payload"]))
    nai = next(a for a in res.adequacy if a.island == "Nainativu")
    assert nai.status == "INADEQUATE" and nai.capacity_kva == 380.0


@pytest.mark.anyio
async def test_the_profile_changes_the_limits() -> None:
    res = await verify_action(VerifyRequest(profile="EN-50160"))
    assert res.profile == "EN-50160-v1" and (res.v_min_pu, res.v_max_pu) == (0.90, 1.10)
    with pytest.raises(HTTPException):
        await verify_action(VerifyRequest(profile="NOT-A-STANDARD"))


@pytest.mark.anyio
async def test_storm_hour_runs_conformal_pid() -> None:
    res = await verify_action(VerifyRequest(hour=HOURS["ditwah_evening"]["timestamp"]))
    assert res.certificate["pid"]["enabled"] is True
    off = await verify_action(VerifyRequest(hour=HOURS["ditwah_evening"]["timestamp"],
                                            pid="off"))
    assert off.certificate["pid"] is None
