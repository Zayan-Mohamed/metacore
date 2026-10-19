from pathlib import Path

import pytest
from module3.serve.storm import (
    DEFAULT_CKPT_DIR,
    PARAM_RANGES,
    StormInputError,
    build_params,
    preset_table,
    simulate_storm,
)

needs_ckpt = pytest.mark.skipif(
    not (DEFAULT_CKPT_DIR / "gate_seed0.pt").is_file(),
    reason="trained checkpoints are not in git",
)


def test_unknown_parameter_rejected():
    with pytest.raises(StormInputError):
        build_params({"peak_wind": 100})


@pytest.mark.parametrize("key", sorted(PARAM_RANGES))
def test_out_of_range_rejected(key):
    with pytest.raises(StormInputError):
        build_params({key: PARAM_RANGES[key][1] + 1.0})


def test_presets_cover_every_knob():
    table = preset_table()
    assert set(table["default"]) == set(PARAM_RANGES) | {"ood"}
    assert {"burevi_like", "ditwah_like"} <= set(table["presets"])


def test_missing_checkpoint_is_a_clean_error(tmp_path: Path):
    with pytest.raises(StormInputError, match="no trained gate"):
        simulate_storm({"ckpt_dir": str(tmp_path), "gate_seed": 0})


@needs_ckpt
def test_storm_is_reproducible_and_timeline_is_consistent():
    spec = {"preset": "ditwah_like", "seed": 3, "params": {"grid_loss_prob": 1.0}}
    a, b = simulate_storm(spec), simulate_storm(spec)
    assert a["gate"]["return"] == b["gate"]["return"]
    steps = a["gate"]["steps"]
    assert len(steps) == a["episode_len"]
    assert abs(sum(s["reward"] for s in steps) - a["gate"]["return"]) < 1e-6
    assert all(0.0 <= s["p_s2"] <= 1.0 for s in steps)
    assert set(a["baselines"]) == {"always_s1", "always_s2", "threshold", "breaker_rule", "oracle"}
    assert a["baselines"]["always_s1"]["s2_rate"] == 0.0
    assert a["baselines"]["always_s2"]["s2_rate"] > 0.5


@needs_ckpt
def test_harsher_storm_hurts_the_do_nothing_baseline():
    mild = simulate_storm({"seed": 1, "params": {"peak_wind_kmh": 40.0}})
    harsh = simulate_storm({"seed": 1, "params": {"peak_wind_kmh": 200.0, "trip_scale": 0.4}})
    assert harsh["baselines"]["always_s1"]["return"] < mild["baselines"]["always_s1"]["return"]
