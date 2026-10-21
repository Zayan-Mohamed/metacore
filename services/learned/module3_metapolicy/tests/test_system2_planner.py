"""Proper System 2: critical-load ranking, shortage-aware shedding, islanded surrogate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from module3.control import (
    critical_load_ranking,
    deliberative_plan,
    reactive_plan,
    shed_priority_order,
)
from module3.env import OBS_DIM, CycloneGatingEnv
from module3.scenario import PROFILES, generate_episode
from module3.twin import Disturbance, LiveM4Twin, SurrogateTwin
from module3.twin.surrogate import connected_groups, group_capacity_kw
from module3.twin.topology import LOAD_KW, NODES, PRIORITY_TIER

ROOT = Path(__file__).resolve().parents[1]
twin = SurrogateTwin()
STORM = Disturbance(grid_lost=True, pv_factor=0.0, diesel_factor=0.5)  # 800 kW vs 1000 kW load
NO_ACTION = {"origin": "SYSTEM1", "breakers": [], "load_shed": [], "dispatch": []}


def _shed(plan):
    return {c["node_id"]: c["shed_fraction"] for c in plan["load_shed"]}


# --------------------------------------------------------------- ranking
def test_ranking_puts_critical_loads_first_and_sheddable_last():
    rank = critical_load_ranking()
    assert [r["tier"] for r in rank] == sorted(r["tier"] for r in rank)  # 1,1,1,2,2,2,3,3,3,3
    assert {r["node_id"] for r in rank[:3]} == {"N1", "N2", "N3"}
    assert [r["rank"] for r in rank] == list(range(1, len(NODES) + 1))


def test_shed_order_is_lowest_tier_first_then_biggest_load():
    order = shed_priority_order(list(NODES))
    assert order[:4] == ["N8", "N9", "N12", "N11"]  # tier 3, biggest first: 120, 100, 90, 80 kW
    assert [PRIORITY_TIER[n] for n in order] == sorted(PRIORITY_TIER[n] for n in order)[::-1]


# --------------------------------------------------------------- islanded surrogate
def test_surplus_supply_keeps_every_island_up():
    out = twin.evaluate(NO_ACTION, Disturbance(grid_lost=True))
    assert out.approved and out.served_fraction == pytest.approx(1.0)


def test_shortfall_collapses_the_island():
    out = twin.evaluate(NO_ACTION, STORM)
    assert not out.approved and out.served_fraction == 0.0
    assert out.detail["mode"] == "islanded"
    assert next(iter(out.detail["shortfall_kw"].values())) == pytest.approx(200.0)


def test_group_with_no_generation_is_dead():
    d = Disturbance(grid_lost=True, tripped_lines=frozenset({"Line_N8_N11"}))
    assert "N11" in twin.evaluate(NO_ACTION, d).violating_buses  # N11 has load, no generator


def test_groups_and_capacity_follow_the_topology():
    groups = connected_groups(frozenset(), grid_lost=True)
    assert len(groups) == 1 and groups[0] == set(NODES)  # the ring joins all three islands
    assert group_capacity_kw(groups[0]) == pytest.approx(1750.0)
    assert group_capacity_kw(groups[0], pv_factor=0.0, diesel_factor=0.5) == pytest.approx(800.0)


# --------------------------------------------------------------- the planner
def test_planner_sheds_only_what_the_shortage_requires_lowest_tier_first():
    plan = deliberative_plan(
        frozenset(), grid_lost=True, pv_factor=0.0, diesel_factor=0.5, reserve_margin=0.0
    )
    shed = _shed(plan)
    shed_kw = sum(LOAD_KW[n] * f for n, f in shed.items())
    assert shed_kw == pytest.approx(200.0, abs=0.1)  # the shortfall, with no reserve
    assert all(PRIORITY_TIER[n] == 3 for n in shed)  # never reaches tier 1 or 2
    out = twin.evaluate(plan, STORM)
    assert out.approved


def test_reserve_margin_sheds_a_little_extra():
    a = deliberative_plan(
        frozenset(), grid_lost=True, pv_factor=0.0, diesel_factor=0.5, reserve_margin=0.0
    )
    b = deliberative_plan(
        frozenset(), grid_lost=True, pv_factor=0.0, diesel_factor=0.5, reserve_margin=0.10
    )
    kw = lambda p: sum(LOAD_KW[n] * f for n, f in _shed(p).items())  # noqa: E731
    assert kw(b) == pytest.approx(kw(a) + 80.0, abs=0.2)  # 10% of 800 kW of supply


def test_critical_loads_go_last_and_only_when_nothing_else_is_left():
    # Almost no supply: PV dead, diesel nearly gone -> only the 200 kW battery remains.
    plan = deliberative_plan(
        frozenset(), grid_lost=True, pv_factor=0.0, diesel_factor=0.0, reserve_margin=0.0
    )
    shed = _shed(plan)
    assert all(shed.get(n) == 1.0 for n in ("N8", "N9", "N11", "N12"))  # tier 3 gone entirely
    assert any(PRIORITY_TIER[n] == 2 for n in shed)  # then tier 2 ...
    served = sum(LOAD_KW[n] * (1 - shed.get(n, 0.0)) for n in NODES)
    assert served == pytest.approx(200.0, abs=0.1)
    assert twin.evaluate(
        plan, Disturbance(grid_lost=True, diesel_factor=0.0, pv_factor=0.0)
    ).approved


def test_no_shedding_when_supply_is_plentiful_or_grid_is_up():
    assert deliberative_plan(frozenset(), grid_lost=True)["load_shed"] == []
    assert deliberative_plan(frozenset())["load_shed"] == []


def test_dispatch_stays_within_unit_ratings_and_skips_pv():
    plan = deliberative_plan(frozenset(), grid_lost=True, pv_factor=0.0, diesel_factor=0.5)
    nodes = {d["node_id"] for d in plan["dispatch"]}
    assert nodes == {"N1", "N4", "N8", "N9"}  # diesel + battery, no PV (N2, N5)
    assert all(0.0 <= d["p_kw"] <= 500.0 for d in plan["dispatch"])


def test_reactive_system1_is_blind_to_the_shortage_and_blacks_out():
    ctx = {"max_node_vulnerability": 0.5, "top_at_risk_nodes": ["N8", "N9"], "severity": "severe"}
    out = twin.evaluate(reactive_plan(ctx, "none"), STORM)
    assert not out.approved  # S1 shed too little: nothing tells it the supply is gone


def test_stale_belief_means_deliberation_cannot_fix_what_it_cannot_see():
    belief_ok = deliberative_plan(frozenset())  # thinks the supply is fine
    assert not twin.evaluate(belief_ok, STORM).approved


# --------------------------------------------------------------- scenario / env / live twin
def test_grid_loss_is_opt_in_and_does_not_change_existing_storms():
    a = generate_episode(PROFILES["ditwah_like"], 80, np.random.default_rng(3))
    assert a.grid_lost is None and not a.truth_grid_lost(10)
    from dataclasses import replace

    b = generate_episode(
        replace(PROFILES["ditwah_like"], grid_loss_prob=1.0), 80, np.random.default_rng(3)
    )
    assert np.array_equal(a.wind, b.wind) and a.tripped == b.tripped  # same storm, plus an event
    assert b.grid_lost is not None and b.grid_lost.any()
    t = int(np.argmax(b.grid_lost))
    assert b.stage[t] in ("severe", "extreme")


def test_env_exposes_grid_loss_to_the_gate_and_scores_it():
    cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg["scenario"]["p_grid_loss"] = 1.0
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(0))
    obs, _ = env.reset(seed=2, options={"cyclone": "ditwah_like"})
    assert obs.shape == (OBS_DIM,) and OBS_DIM == 15
    seen = False
    done = False
    while not done:
        obs, _, _, done, info = env.step(1)
        seen |= bool(info["grid_lost_truth"])
    assert seen


def test_live_m4_refuses_what_it_cannot_represent():
    with pytest.raises(NotImplementedError, match="lost main supply"):
        LiveM4Twin("http://127.0.0.1:1").evaluate(NO_ACTION, STORM)
