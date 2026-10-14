"""SurrogateTwin rules, mirrored from probes of the real M4 (see surrogate.py docstring)."""

from __future__ import annotations

import pytest
from module3.control import deliberative_plan, reactive_plan
from module3.twin import Disturbance, SurrogateTwin, priority_weighted_served
from module3.twin.live_m4 import merge_with_overlay, overlay_commands
from module3.twin.surrogate import energized_buses
from module3.twin.topology import NODES

twin = SurrogateTwin()


def act(shed=(), breakers=()):
    return {
        "origin": "SYSTEM1",
        "breakers": [{"edge_id": e, "closed": False} for e in breakers],
        "load_shed": [{"node_id": n, "shed_fraction": f, "priority_tier": 3} for n, f in shed],
        "dispatch": [],
    }


def test_intact_grid_is_approved_and_fully_served():
    out = twin.evaluate(act(), Disturbance())
    assert out.approved and out.served_fraction == pytest.approx(1.0)


def test_tripped_overhead_feeder_is_a_violation_unless_fully_shed():
    d = Disturbance(tripped_lines=frozenset({"Line_N8_N11"}))
    assert twin.evaluate(act(), d).violating_buses == ["N11"]
    assert twin.evaluate(act([("N11", 0.99)]), d).violating_buses == ["N11"]
    assert twin.evaluate(act([("N11", 1.0)]), d).approved


def test_isolated_generator_bus_cannot_be_fixed_by_shedding():
    d = Disturbance(tripped_lines=frozenset({"Line_N8_N9"}))  # N9 hosts the BESS
    assert "N9" in twin.evaluate(act([("N9", 1.0)]), d).violating_buses


def test_ring_keeps_a_node_energised_when_one_tie_is_lost():
    assert "N2" in energized_buses(frozenset({"Line_N1_N2"}))  # still reachable the long way round
    assert twin.evaluate(act(), Disturbance(tripped_lines=frozenset({"E_crit_1"}))).approved


def test_unknown_node_or_line_is_malformed():
    assert not twin.evaluate(act([("N7", 0.1)]), Disturbance()).approved
    assert not twin.evaluate(act(breakers=["Line_X"]), Disturbance()).approved


def test_priority_weighting_makes_critical_load_cost_more():
    tier1 = priority_weighted_served({"N1": 1.0}, NODES)  # 150 kW, tier 1
    tier3 = priority_weighted_served({"N11": 1.0}, NODES)  # 80 kW, tier 3
    assert tier1 < tier3 < 1.0
    # per kW, a tier-1 kW is worth exactly 3x a tier-3 kW:
    assert (1 - tier1) / 150 == pytest.approx(3 * (1 - tier3) / 80)


def test_planner_sheds_exactly_the_cut_off_buses():
    plan = deliberative_plan(frozenset({"Line_N8_N11", "Line_N8_N12"}))
    assert {c["node_id"] for c in plan["load_shed"]} == {"N11", "N12"}
    assert twin.evaluate(
        plan, Disturbance(tripped_lines=frozenset({"Line_N8_N11", "Line_N8_N12"}))
    ).approved


def test_reactive_path_ignores_topology_and_gets_rejected():
    ctx = {"max_node_vulnerability": 0.9, "top_at_risk_nodes": ["N2", "N5"], "severity": "extreme"}
    d = Disturbance(tripped_lines=frozenset({"Line_N8_N11"}))
    assert not twin.evaluate(reactive_plan(ctx, "none"), d).approved


def test_live_overlay_translation_and_precedence():
    d = Disturbance(tripped_lines=frozenset({"Line_N8_N11"}), pv_factor=0.5, diesel_factor=1.0)
    br, disp = overlay_commands(d)
    assert br == [{"edge_id": "Line_N8_N11", "closed": False}]
    assert {x["node_id"] for x in disp} == {"N2", "N5"} and disp[0]["p_kw"] == pytest.approx(100.0)
    # a policy that tries to close a damaged line is overridden by the overlay
    a = {
        "origin": "SYSTEM2",
        "breakers": [{"edge_id": "Line_N8_N11", "closed": True}],
        "load_shed": [],
        "dispatch": [],
    }
    merged = merge_with_overlay(a, d, "x")
    assert merged["breakers"] == [{"edge_id": "Line_N8_N11", "closed": False}]
