"""Versioned limit profiles: the same solved state judged against different standards."""

from __future__ import annotations

import json

import pytest
from verification.firewall.limits import (
    PROFILES_DIR,
    PhysicsLimitsChecker,
    SafetyLimitsConfig,
    available_profiles,
)
from verification.firewall.verifier import PhysicsVerifier
from verification.types import Decision, ProposedControlAction, ViolationType

EXPECTED = {
    "METACORE-STRICT": (0.95, 1.05),
    "SL-PSR-2016-LV": (0.94, 1.06),
    "EN-50160": (0.90, 1.10),
    "ANSI-C84.1-A": (0.95, 1.05),
}


class _OneBus:
    """A solved state with a single bus at a chosen voltage and no lines or pin."""

    pin = None

    def __init__(self, v_pu: float) -> None:
        self.v_pu = v_pu

    def get_bus_voltages_pu(self) -> dict[str, float]:
        return {"ELU_F1": self.v_pu}

    def get_line_loadings(self) -> dict:
        return {}


@pytest.mark.parametrize("path", sorted(PROFILES_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_profile_file_parses_with_its_provenance(path) -> None:
    spec = json.loads(path.read_text())
    assert {"profile_id", "version", "v_min_pu", "v_max_pu", "thermal_overload_margin",
            "statistic", "source"} <= spec.keys()
    assert 0.0 < spec["v_min_pu"] < 1.0 < spec["v_max_pu"]
    cfg = SafetyLimitsConfig.from_profile(spec["profile_id"])
    assert cfg.profile == f"{spec['profile_id']}-v{spec['version']}"


def test_the_four_planned_profiles_exist_with_their_bands() -> None:
    profiles = available_profiles()
    assert set(profiles) == set(EXPECTED)
    for pid, (lo, hi) in EXPECTED.items():
        assert (profiles[pid]["v_min_pu"], profiles[pid]["v_max_pu"]) == (lo, hi)


def test_strict_is_the_default() -> None:
    assert PhysicsLimitsChecker().config.profile == "METACORE-STRICT-v1"


def test_a_0_941_pu_bus_fails_strict_and_passes_sl_psr() -> None:
    state = _OneBus(0.941)
    strict = PhysicsLimitsChecker(SafetyLimitsConfig.from_profile("METACORE-STRICT"))
    slpsr = PhysicsLimitsChecker(SafetyLimitsConfig.from_profile("sl-psr-2016-lv"))
    [v] = strict.check_limits(state, converged=True)
    assert v.type == ViolationType.VIOLATION_TYPE_UNDERVOLTAGE and v.limit == 0.95
    assert slpsr.check_limits(state, converged=True) == []


def test_unknown_profile_is_refused() -> None:
    with pytest.raises(KeyError):
        SafetyLimitsConfig.from_profile("NOT-A-STANDARD")


def test_the_verdict_records_which_profile_judged_it() -> None:
    noop = ProposedControlAction(action_id="noop")
    for pid in EXPECTED:
        verdict = PhysicsVerifier(profile=pid).verify(noop)
        assert verdict.decision == Decision.DECISION_APPROVE
        assert verdict.limit_profile.startswith(pid)
