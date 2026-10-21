"""The evaluation harness (C16): snapshots, ground truth and reproducibility."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest
from evaluation.benchmark_set import Case, SnapshotFactory, build_cases
from evaluation.run_eval import ground_truth, run
from verification.firewall.verifier import PhysicsVerifier
from verification.types import DispatchSetpoint, LoadShedCommand, ProposedControlAction

SAMPLES = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
           / "metacore_contracts" / "schema" / "samples")


@cache
def factory() -> SnapshotFactory:
    return SnapshotFactory()


@pytest.mark.parametrize("name", ["calm", "nainativu_peak", "ditwah_evening"])
def test_snapshots_match_module1s_fixtures(name: str) -> None:
    fixture = json.loads((SAMPLES / f"module1_snapshot_v1_{name}.json").read_text())
    built = factory().snapshot(fixture["timestamp_lst"])
    assert built["node_states"] == fixture["node_states"]
    assert built["assets_out_of_service"] == fixture["assets_out_of_service"]
    assert built["hybrid_plant_down"] == fixture["hybrid_plant_down"]


def test_ground_truth_replays_the_recorded_next_hours() -> None:
    gate = PhysicsVerifier()
    fragile = Case("2025-10-05T12:00", ProposedControlAction(
        action_id="bess", dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=40.0,
                                                     q_kvar=20.0)]), "bess_export_40",
        "Eluvaitivu")
    safe = Case("2025-10-05T12:00", ProposedControlAction(
        action_id="shed", load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.2)]),
        "shed_nai_f2", "Nainativu")
    assert ground_truth(fragile, factory(), gate) is True
    assert ground_truth(safe, factory(), gate) is False


def test_battery_actions_are_left_out_while_the_hybrid_is_down() -> None:
    cases = build_cases(factory(), "2025-11-28T00:00", "2025-11-28T23:00", stride=24)
    assert {c.family for c in cases} == {"noop", "shed_nai_f2", "standby_nai_dg2",
                                         "hyb_dg_export"}


def test_rerunning_gives_the_same_numbers() -> None:
    first, _ = run(stride=7, quick=True)
    second, _ = run(stride=7, quick=True)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert set(first["E1"]) == {"B0", "B1", "B2", "B3", "B4"}
