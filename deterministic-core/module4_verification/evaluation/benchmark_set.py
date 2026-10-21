"""The benchmark: control actions on recorded hours of the test split, with ground truth.

Test split: Oct to Dec 2025 (Cyclone Ditwah and the Eluvaitivu hybrid decay); nothing from it was
used in calibration. Every ``stride`` hours an hour is replayed (a stride coprime with 24 walks
through every hour of the day), and each action of the family below is proposed on it.

Ground truth needs no model: replay the action on the *recorded* hours t+1 .. t+6. If any of
those real hours breaks a limit (Check A rejects it), the action was truly unsafe.

Snapshots are built the way Module 1 builds them (module1/topology/snapshot.py): island demand
split over the feeder groups by the pin's feeder shares, reactive power at the pin's power
factor, and the hybrid PV and battery out of service in the months the ledger shows them down.
The monthly outage list is taken from Module 1's own snapshot CLI, once per month, so M4 copies
none of Module 1's rules; tests/test_evaluation.py checks the result equals the committed M1
fixtures.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cache

from verification.calibrate.misses import HourlyLoad, read_hourly_load
from verification.opendss.circuit import load_pin
from verification.risk.history import repo_root
from verification.types import DispatchSetpoint, LoadShedCommand, ProposedControlAction

TEST_SPLIT = ("2025-10-01T00:00", "2025-12-31T17:00")  # last start hour with a full window
HORIZON_H = 6
STRIDE_H = 7


@dataclass(frozen=True)
class Case:
    """One action proposed on one recorded hour."""

    timestamp: str
    action: ProposedControlAction
    family: str
    target_island: str  # the island the action acts on (the planted cause for E4)
    extra_outages: tuple[str, ...] = field(default=())


def action_family(hybrid_down: bool) -> list[tuple[str, str, ProposedControlAction]]:
    """(family, target island, action). Battery actions only while the battery is in service."""
    acts = [
        ("noop", "", ProposedControlAction(action_id="noop", rationale="no control action")),
        ("shed_nai_f2", "Nainativu", ProposedControlAction(
            action_id="shed-nai-f2",
            load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.2)])),
        ("standby_nai_dg2", "Nainativu", ProposedControlAction(
            action_id="standby-nai-dg2",
            dispatch=[DispatchSetpoint(node_id="NAI_DG2", p_kw=150.0, q_kvar=50.0)])),
        ("hyb_dg_export", "Eluvaitivu", ProposedControlAction(
            action_id="hyb-dg-export",
            dispatch=[DispatchSetpoint(node_id="ELU_HYB_DG", p_kw=30.0, q_kvar=15.0)])),
    ]
    if not hybrid_down:
        acts += [
            ("bess_export_25", "Eluvaitivu", ProposedControlAction(
                action_id="bess-export-25",
                dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=25.0, q_kvar=10.0)])),
            ("bess_export_40", "Eluvaitivu", ProposedControlAction(
                action_id="bess-export-40",
                dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=40.0, q_kvar=20.0)])),
            ("bess_charge_40", "Eluvaitivu", ProposedControlAction(
                action_id="bess-charge-40",
                dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=-40.0, q_kvar=0.0)])),
        ]
    return acts


class SnapshotFactory:
    """Module-1-shaped GridStateSnapshots for any recorded hour."""

    def __init__(self, load: HourlyLoad | None = None) -> None:
        root = repo_root()
        if root is None:
            raise FileNotFoundError("the evaluation needs a repo checkout (data/processed)")
        self.root = root
        self.load = load or read_hourly_load(root / "data" / "processed" /
                                             "island_load_hourly.csv")
        self.index = {t: i for i, t in enumerate(self.load.timestamps)}
        self.pin = load_pin()
        self.q_ratio = math.tan(math.acos(self.pin["assumptions"]["power_factor"]))
        self.shares = self.pin["assumptions"]["feeder_shares"]

    @cache  # noqa: B019 -- one factory per run; months are few
    def outages_for_month(self, month: str) -> tuple[tuple[str, ...], bool]:
        """(assets out of service, hybrid down) for a month, from Module 1's snapshot CLI."""
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([
            str(self.root / "services" / "learned" / "module1_state_forecasting" / "src"),
            str(self.root / "packages" / "contracts" / "python"),
            os.environ.get("PYTHONPATH", "")])}
        proc = subprocess.run(
            [sys.executable, "-m", "module1.topology", "snapshot", "--timestamp",
             f"{month}-15T12:00"],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=60, check=True)
        snap = json.loads(proc.stdout)
        return tuple(snap["assets_out_of_service"]), bool(snap["hybrid_plant_down"])

    def snapshot(self, timestamp: str, extra_outages: tuple[str, ...] = ()) -> dict:
        row = self.load.kw[self.index[timestamp]]
        island_kw = dict(zip(self.load.islands, row, strict=True))
        outages, hybrid_down = self.outages_for_month(timestamp[:7])
        states = []
        for bus in self.pin["buses"]:
            if bus["role"] == "feeder_group":
                p = island_kw[bus["island"]] * self.shares[bus["node_id"].split("_", 1)[1]]
                states.append({
                    "node_id": bus["node_id"], "voltage_pu": None,
                    "p_kw": round(p, 3), "q_kvar": round(p * self.q_ratio, 3),
                    "soc_fraction": None,
                    "quality": {"voltage_pu": "QUALITY_MISSING",
                                "p_kw": "QUALITY_INTERPOLATED",
                                "q_kvar": "QUALITY_INTERPOLATED",
                                "soc_fraction": "QUALITY_MISSING"}})
            elif bus["role"] == "hybrid_plant":
                states.append({
                    "node_id": bus["node_id"], "voltage_pu": None, "p_kw": None,
                    "q_kvar": None, "soc_fraction": None,
                    "quality": {k: "QUALITY_MISSING"
                                for k in ("voltage_pu", "p_kw", "q_kvar", "soc_fraction")}})
        return {
            "topology_id": self.pin["topology_id"],
            "timestamp_lst": timestamp,
            "node_states": states,
            "assets_out_of_service": sorted(set(outages) | set(extra_outages)),
            "hybrid_plant_down": hybrid_down,
        }

    def later(self, timestamp: str, hours: int) -> str:
        return self.load.timestamps[self.index[timestamp] + hours]


def build_cases(factory: SnapshotFactory, start: str = TEST_SPLIT[0], end: str = TEST_SPLIT[1],
                stride: int = STRIDE_H) -> list[Case]:
    lo, hi = factory.index[start], factory.index[end]
    cases: list[Case] = []
    for t in range(lo, hi + 1, stride):
        ts = factory.load.timestamps[t]
        _, hybrid_down = factory.outages_for_month(ts[:7])
        for family, island, action in action_family(hybrid_down):
            cases.append(Case(ts, action.model_copy(update={"action_id": f"{family}@{ts}"}),
                              family, island))
    return cases
