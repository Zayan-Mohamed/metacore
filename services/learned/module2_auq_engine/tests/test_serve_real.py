"""The real head's one-shot serving path, and the debounce it has to answer on.

Regression for a silent under-fire: `serve_real.score_state` makes exactly one scoring call,
but `OnnxAUQ.load` reads `hysteresis` from config.yaml, where it is 2 for the streaming path.
A debounce of 2 can never latch on one call, so the gate returned `trigger: False,
reason: "none"` for a state at u = 1.0 -- the most out-of-distribution reading the metric can
express. `ONESHOT_HYSTERESIS` existed for this and was never applied.

Both axes are covered, because both were dead: the value threshold and the sensing floor are
OR'd but share the one debounce counter. The streaming test is the other half of the fix --
the one-shot override must not leak into `load_engine`, or a caller that scores consecutive
hours through the returned engine would start firing on single-hour spikes.

Exports a tiny model and writes a stand-in card rather than leaning on the committed
edl_real.onnx, for the reason test_onnx.py gives: that artifact is git-ignored and derived, so
a test that needed it could not run on a clean checkout or in CI.
"""

import json

import numpy as np
import pytest
import torch
import yaml
from edl import EDLNet
from metacore_contracts.state_schema import FEATURE_NAMES, calibration_quality

ort = pytest.importorskip("onnxruntime", reason="onnx extra not installed")

import serve_real  # noqa: E402 -- after the onnxruntime guard, since load_engine needs it

D_IN, K = len(FEATURE_NAMES), 3

# Thresholds chosen so the assertions cannot be vacuous. Evidence is softplus, hence strictly
# positive, so value-u = K/(S + K) lies strictly inside (0, 1): everything clears 0.0 and
# nothing clears 1.0. The trigger compares with `>`.
ALWAYS_OVER = 0.0
NEVER_OVER = 1.0


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    """A 28-in / 3-out evidence graph, exported once for this file."""
    pytest.importorskip("onnxscript", reason="onnx extra not installed")
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("serve_real") / "edl_real.onnx"
    torch.onnx.export(
        EDLNet(D_IN, K).eval(), (torch.zeros(1, D_IN),), str(path),
        input_names=["state"], output_names=["evidence"],
        dynamo=True, opset_version=18, external_data=False,
    )
    return path


@pytest.fixture
def stand_in(tiny_onnx, monkeypatch):
    """Point serve_real at the tiny graph and a card carrying a chosen value threshold.

    The of_floor and hysteresis still come from the module's real config.yaml, which is the
    point: the test asserts the one-shot path answers correctly *while* config.yaml says 2.
    """
    def _install(value_threshold: float):
        card = {
            "artifact": tiny_onnx.name,
            "architecture": {"input_dim": D_IN, "hidden": 32, "classes": K,
                             "class_names": ["safe", "stressed", "critical"],
                             "feature_names": list(FEATURE_NAMES)},
            "normalisation": {"mean": [0.0]*D_IN, "std": [1.0]*D_IN},
            "trigger": {"value_threshold": value_threshold,
                        "value_false_alarm_rate": 0.05,
                        "observed_fraction_floor": 0.35},
            "label": {"disclosure": "stand-in card, built by test_serve_real.py"},
        }
        card_path = tiny_onnx.with_suffix(".onnx.json")
        card_path.write_text(json.dumps(card))
        monkeypatch.setattr(serve_real, "MODEL_PATH", tiny_onnx)
        monkeypatch.setattr(serve_real, "CARD_PATH", card_path)
    return _install


def _config_hysteresis() -> int:
    with open(serve_real.HERE / "config.yaml") as fh:
        return int(yaml.safe_load(fh)["trigger"]["hysteresis"])


def _nominal_mask() -> list[str]:
    """The mask every real M1 row carries: 12 observed of 28, so f = 0.4286 > the 0.35 floor."""
    return [calibration_quality(name) for name in FEATURE_NAMES]


def _blind_mask() -> list[str]:
    """One feature observed, f = 1/28 = 0.036 -- an order of magnitude under the floor."""
    mask = ["QUALITY_MISSING"]*D_IN
    mask[0] = "QUALITY_OBSERVED"
    return mask


def test_config_still_asks_for_a_streaming_debounce():
    """Guards the premise. If config.yaml ever drops to 1 the bug below becomes unreachable
    and the rest of this file stops testing anything, so fail loudly rather than pass quietly."""
    assert _config_hysteresis() > 1
    assert serve_real.ONESHOT_HYSTERESIS == 1


def test_value_axis_fires_on_a_single_call(stand_in):
    """The regression. One call, state over the value threshold -> the gate must say so."""
    stand_in(ALWAYS_OVER)
    out = serve_real.score_state([0.1]*D_IN, _nominal_mask())

    assert out["u"] > ALWAYS_OVER, "fixture broken: the state must clear the threshold"
    assert out["input_problems"] == [], "must fire on the trigger, not on a sanitiser escalation"
    assert out["trigger"] is True
    assert out["reason"] == "value"


def test_sensing_axis_fires_on_a_single_call(stand_in):
    """The other axis, which shares the same debounce counter and was equally dead."""
    stand_in(NEVER_OVER)
    out = serve_real.score_state([0.1]*D_IN, _blind_mask())

    assert out["observed_fraction"] == pytest.approx(1/D_IN)
    assert out["input_problems"] == [], "a valid all-missing mask is a quality fact, not damage"
    assert out["trigger"] is True
    assert out["reason"] == "sensing"


def test_an_ordinary_state_still_does_not_fire(stand_in):
    """Negative control: the two tests above must not pass by firing on everything."""
    stand_in(NEVER_OVER)
    out = serve_real.score_state([0.1]*D_IN, _nominal_mask())

    assert out["trigger"] is False
    assert out["reason"] == "none"


def test_load_engine_keeps_the_configured_hysteresis_for_streaming(stand_in):
    """The one-shot override belongs to score_state alone.

    A streaming caller scores the previous hour and then this one precisely so a drop must hold
    for two consecutive hours before it reaches M3. If the override leaked into load_engine,
    such a caller would fire on a single-hour spike and the debounce would be gone entirely.
    """
    stand_in(ALWAYS_OVER)
    auq, _ = serve_real.load_engine()
    assert auq.hysteresis == _config_hysteresis()

    features, mask = np.full(D_IN, 0.1, dtype=np.float32), _nominal_mask()
    auq.reset()
    assert auq.score_state(features, mask).competence_drop is False, "hour 1 must not latch"
    assert auq.score_state(features, mask).competence_drop is True, "hour 2 must latch"
