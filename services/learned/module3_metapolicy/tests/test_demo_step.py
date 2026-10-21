"""The dashboard's Train / Generate / Process path, run for real at a tiny size.

These need torch (they train a gate for a few PPO updates). They check the wiring and the one
property the page depends on: what Generate shows is exactly what Process then decides on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from module3.serve.demo import build_config, run_demo
from module3.serve.step import generate, process
from module3.serve.storm import StormInputError

BASE = yaml.safe_load((Path(__file__).resolve().parents[1] / "pipeline.yaml").read_text())
TINY = {"seed": 1, "episode_len": 20, "train_episodes": 10, "eval_episodes": 1}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    state = tmp_path_factory.mktemp("dashboard_state")
    result = run_demo(build_config(BASE, TINY), state)
    return state, result


def test_build_config_maps_the_form_onto_the_pipeline_config():
    cfg = build_config(
        BASE,
        {"seed": 7, "episode_len": 40, "budget_per_episode": 12, "train_episodes": 80,
         "eval_episodes": 5, "deliberation_cost": 0.2, "w_serve": 3.0, "w_viol": 4.0},
    )
    assert cfg["seed"] == 7
    assert cfg["env"] == {"episode_len": 40, "budget_per_episode": 12.0}
    assert cfg["reward"]["deliberation_cost"] == 0.2
    assert cfg["reward"]["w_serve"] == 3.0 and cfg["reward"]["w_viol"] == 4.0
    assert cfg["ppo"]["tasks_per_update"] * cfg["ppo"]["updates"] >= 64
    assert cfg["eval"]["episodes_random"] == 5
    assert BASE["env"]["episode_len"] == 80  # the shared base config is not mutated


def test_run_demo_result_has_the_shape_the_page_reads(trained):
    state, result = trained
    assert set(result["reward"]) == {
        "always_s1", "always_s2", "threshold", "breaker_rule", "oracle", "trained_policy",
    }
    assert {"always_s2", "threshold", "trained_policy"} <= set(result["avg_deliberation_cost"])
    assert set(result["escalation_by_severity"]) == {"normal", "elevated", "severe", "extreme"}
    assert set(result["escalation_by_trigger_reason"]) == {"none", "value", "sensing", "both"}
    assert len(result["decisions"]) == 2 * len(result["decision_context"]) > 0
    json.dumps(result)  # plain JSON, no numpy leaking through
    assert (state / "gate_seed0.pt").is_file()


def test_decision_pairs_share_action_ids(trained):
    _, result = trained
    pcas = [r for r in result["decisions"] if r["message_type"] == "ProposedControlAction"]
    gds = [r for r in result["decisions"] if r["message_type"] == "GatingDecision"]
    assert [r["action_id"] for r in pcas] == [r["action_id"] for r in gds]
    assert all(c["verdict"] in {"APPROVE", "REJECT"} for c in result["decision_context"])


def test_generate_needs_a_trained_gate(tmp_path):
    with pytest.raises(StormInputError, match="no trained gate"):
        generate(tmp_path / "out.json", tmp_path)


def test_process_needs_a_generated_situation(trained, tmp_path):
    state, _ = trained
    (state / "pending.json").unlink(missing_ok=True)
    with pytest.raises(StormInputError, match="Generate first"):
        process(tmp_path / "out.json", state)


def test_process_decides_on_exactly_what_generate_showed(trained, tmp_path):
    state, _ = trained
    gen = generate(tmp_path / "gen.json", state)
    assert len(gen["observation"]) == 15
    assert 0 <= gen["step_index"] < gen["episode_len"] == 20

    first = process(tmp_path / "p1.json", state)
    again = process(tmp_path / "p2.json", state)  # same pending situation, replayed
    assert [f["value"] for f in first["observation"]] == [f["value"] for f in gen["observation"]]
    assert first["chosen"] == again["chosen"]
    assert first["action_probs"] == again["action_probs"]
    assert first["plan"]["origin"] == first["chosen"]
    assert abs(sum(first["action_probs"].values()) - 1.0) < 1e-6
    assert set(first["twin"]) >= {"approved", "violating_buses", "served_fraction", "reward"}
