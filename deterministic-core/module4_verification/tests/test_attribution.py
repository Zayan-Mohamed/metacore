"""Shapley risk attribution over the feeder players (C12)."""

from __future__ import annotations

import itertools
import json
from functools import cache
from pathlib import Path

import pytest
from verification.opendss.circuit import load_pin
from verification.risk.attribution import (
    coalition_values,
    island_subsets,
    players_from_pin,
    shapley_values,
)
from verification.risk.calibration import Calibration
from verification.risk.certifier import RiskCertifiedVerifier
from verification.risk.pid import ReplayHistory
from verification.types import DispatchSetpoint, LoadShedCommand, ProposedControlAction

SAMPLES = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
           / "metacore_contracts" / "schema" / "samples")


def test_the_five_players_come_from_the_pin() -> None:
    players = players_from_pin(load_pin())
    assert [p.name for p in players] == [
        "Analaitivu", "Delft-Neduntivu", "Eluvaitivu", "NAI_F1", "NAI_F2"]
    assert players[1].buses == {"DEL_F1", "DEL_F2"}
    assert players[3].buses == {"NAI_F1"}
    pin_buses = {b["node_id"] for b in load_pin()["buses"]}
    assert all(p.buses <= pin_buses for p in players)


def test_only_nainativu_needs_extra_solves() -> None:
    configs = island_subsets(players_from_pin(load_pin()))
    assert configs == [{"Nainativu": frozenset({3})}, {"Nainativu": frozenset({4})}]


def _all_values(n: int, v) -> dict[frozenset[int], float]:
    return {frozenset(c): v(frozenset(c)) for r in range(n + 1)
            for c in itertools.combinations(range(n), r)}


def test_a_single_cause_takes_the_whole_share() -> None:
    """Planted: only Delft (player 1) matters -> Delft gets 100 %, everyone else 0."""
    phi = shapley_values(5, _all_values(5, lambda s: 0.029 if 1 in s else 0.0))
    assert phi[1] == pytest.approx(0.029)
    assert all(abs(x) < 1e-12 for i, x in enumerate(phi) if i != 1)


def test_shared_risk_is_split_and_the_shares_add_up() -> None:
    """The guide's Nainativu example: F1 + F2 together cause more than apart."""
    def v(s: frozenset[int]) -> float:
        nai = {frozenset(): 0.0, frozenset({3}): 0.002, frozenset({4}): 0.001,
               frozenset({3, 4}): 0.006}[s & {3, 4}]
        return nai + (0.023 if 1 in s else 0.0)

    values = _all_values(5, v)
    phi = shapley_values(5, values)
    assert sum(phi) == pytest.approx(values[frozenset(range(5))])  # efficiency
    assert phi[1] == pytest.approx(0.023)
    # the 0.3 points of interaction are split evenly between F1 and F2
    assert phi[3] == pytest.approx(0.002 + 0.0015) and phi[4] == pytest.approx(0.001 + 0.0015)


def test_coalition_values_assemble_island_by_island() -> None:
    players = players_from_pin(load_pin())
    # One scenario: Eluvaitivu fails whenever it moves; Nainativu only if both feeders move.
    table = [{
        "Eluvaitivu": {frozenset(): False, frozenset({2}): True},
        "Nainativu": {frozenset(): False, frozenset({3}): False, frozenset({4}): False,
                      frozenset({3, 4}): True},
        "Analaitivu": {frozenset(): False, frozenset({0}): False},
        "Delft-Neduntivu": {frozenset(): False, frozenset({1}): False},
    }]
    values = coalition_values(players, table)
    assert values[frozenset({2})] == 1.0
    assert values[frozenset({3})] == 0.0 and values[frozenset({3, 4})] == 1.0
    assert values[frozenset()] == 0.0


@cache
def verifier() -> RiskCertifiedVerifier:
    return RiskCertifiedVerifier(history=ReplayHistory.load(Calibration.load()))


def calm() -> dict:
    return json.loads((SAMPLES / "module1_snapshot_v1_calm.json").read_text())


FRAGILE = ProposedControlAction(
    action_id="bess-export-40",
    dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=40.0, q_kvar=20.0)])


def test_a_planted_island_cause_is_recovered_end_to_end() -> None:
    """Only the Eluvaitivu battery is stressed: Eluvaitivu owns the whole risk."""
    verdict = verifier().verify(FRAGILE, snapshot=calm())
    c = verdict.certificate
    total = c.n_failures / c.n_scenarios
    shares = {a.player: a for a in verdict.attribution}
    assert shares["Eluvaitivu"].share == pytest.approx(total, abs=1e-6)
    assert shares["Eluvaitivu"].fraction_of_risk == pytest.approx(1.0, abs=1e-6)
    assert sum(a.share for a in verdict.attribution) == pytest.approx(total, abs=1e-5)
    assert verdict.attribution[0].player == "Eluvaitivu"


def test_same_seed_same_shares() -> None:
    a = verifier().verify(FRAGILE, snapshot=calm())
    b = verifier().verify(FRAGILE, snapshot=calm())
    assert [x.model_dump() for x in a.attribution] == [x.model_dump() for x in b.attribution]


def test_nothing_failed_nothing_to_attribute() -> None:
    safe = ProposedControlAction(
        action_id="shed", load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.3)])
    assert verifier().verify(safe, snapshot=calm()).attribution == []
