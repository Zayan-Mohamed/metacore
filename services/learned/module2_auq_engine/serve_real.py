"""Serve the real-distribution EDL head. Deliberately lean, and deliberately separate.

    python serve_real.py --state-json -        # one contract state in, one JSON object out
    python serve_real.py --warm                # prime the ONNX session, print readiness

Why this is its own entry point rather than a branch inside infer_one.py:

* **Cost.** infer_one.py imports torch and trains a baseline fixture on first use -- 13.6 s
  cold, 3.1 s warm. Nothing on the real path needs either: `infer.OnnxAUQ` is NumPy plus
  onnxruntime, and the value threshold is calibrated at *train* time and carried in
  edl_real.onnx.json. So this answers in well under a second, which is what makes the
  head toggle usable live instead of a pause mid-demo.
* **Blast radius.** The synthetic path is the guaranteed PP1 fallback. Keeping the real
  head out of infer_one.py entirely means no edit to it can change what the fallback
  returns -- the strongest version of "do not touch the synthetic path internals".

WHICH HEAD PRODUCED A NUMBER IS PART OF THE ANSWER
--------------------------------------------------
Every response carries `head`, `artifact` and, on this path, `label_disclosure`. The two
heads are trained on different distributions against different targets and their numbers
are not comparable -- a synthetic u of 0.05 and a real u of 0.05 do not mean the same
thing. The field is not decoration; it is what stops the two being mistaken for one
another on a slide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODEL_PATH = HERE / "edl_real.onnx"
CARD_PATH = HERE / "edl_real.onnx.json"

HEAD = "real"
BUILD_COMMAND = "python train_real.py && python export_onnx.py --real"

# One-shot scoring has no history, so the streaming debounce could never fire: with
# config.yaml's hysteresis of 2, a single call leaves the trigger one step short of latching
# and reports "none" even at u = 1.0. Applied in score_state, NOT in load_engine -- callers
# that score consecutive hours through the returned engine must keep the configured 2, which
# is the whole point of the debounce.
ONESHOT_HYSTERESIS = 1
LATENCY_REPEATS = 20


class ArtifactMissing(FileNotFoundError):
    """Raised when edl_real.onnx has not been built on this checkout.

    It is git-ignored and derived, so this is an ordinary state on a fresh clone, not a
    failure. The message carries the command that fixes it.
    """

    def __init__(self, path: Path):
        super().__init__(
            f"{path.name} not found. It is a derived, git-ignored artifact -- build it "
            f"with: {BUILD_COMMAND}"
        )
        self.path = path


def _require_artifacts() -> None:
    for path in (MODEL_PATH, CARD_PATH):
        if not path.exists():
            raise ArtifactMissing(path)


def load_engine():
    """The real head, with its threshold read from the card rather than recalibrated."""
    _require_artifacts()
    from infer import OnnxAUQ

    card = json.loads(CARD_PATH.read_text())
    threshold = card.get("trigger", {}).get("value_threshold")
    if threshold is None:
        raise RuntimeError(
            f"{CARD_PATH.name} carries no trigger.value_threshold; re-export it with "
            f"`python export_onnx.py --real` so serving needs no calibration pass"
        )
    auq = OnnxAUQ.load(model_path=MODEL_PATH, card_path=CARD_PATH,
                       value_threshold=float(threshold))
    return auq, card


def score_state(features, quality_mask=None) -> dict:
    """One real M1 state in, one scored reading out. Never raises on bad input."""
    from sanitize import sanitize_state

    auq, card = load_engine()
    clean = sanitize_state(features, quality_mask)
    f = clean.observed_fraction

    # There is exactly one call below, so the debounce has to answer on it. reset() rebuilds
    # the trigger, so this has to be set before it rather than after.
    auq.hysteresis = ONESHOT_HYSTERESIS
    auq.reset()
    started = time.perf_counter()
    out = auq.score(clean.features, f)
    first_call_ms = (time.perf_counter() - started)*1000.0

    # Best-of after the first call, so the number is the engine's cost and not the
    # one-off session warm-up.
    best = float("inf")
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        auq.evidence(clean.features)
        best = min(best, time.perf_counter() - t0)

    value_u, adjusted_u, _ = auq.uncertainty(auq.evidence(clean.features), f)
    total_evidence = float(auq.evidence(clean.features).sum()) + auq.k

    result = {
        "head": HEAD,
        "artifact": MODEL_PATH.name,
        "u": float(value_u),
        "u_q": float(adjusted_u),
        "evidence": total_evidence,
        "observed_fraction": float(f),
        "trigger": bool(out.competence_drop),
        "reason": out.trigger_reason,
        "state_class": int(out.state_class),
        "class_names": card["architecture"].get("class_names", []),
        "latency_ms": best*1000.0,
        "first_call_ms": first_call_ms,
        # Not available on this path, and deliberately not faked: the softmax and
        # MC-Dropout baselines in infer_one.py are fitted on the SYNTHETIC distribution,
        # so reporting them beside a real-head u would invite exactly the comparison this
        # module exists to prevent. The real baseline comparison is in real_m1_metrics.json.
        "baselines": None,
        "backend": "onnx",
        "value_threshold": float(auq.value_threshold),
        "input_problems": list(clean.problems),
        "label_disclosure": card.get("label", {}).get("disclosure", ""),
    }

    if not clean.ok:
        from trigger import REASON_BOTH, REASON_SENSING, REASON_VALUE

        result["trigger"] = True
        result["u_q"] = 1.0
        result["reason"] = (REASON_BOTH if result["reason"] in (REASON_VALUE, REASON_BOTH)
                            else REASON_SENSING)
    return result


def warm() -> dict:
    """Build the ONNX session once so the first real request is not the one that pays.

    Cheap by construction rather than by caching: there is no fixture to train and no
    calibration pass to run, so this is a session create plus one forward pass.
    """
    started = time.perf_counter()
    auq, card = load_engine()
    zeros = np.zeros(len(auq.feature_names), dtype=np.float32)
    auq.evidence(zeros)
    return {
        "head": HEAD,
        "artifact": MODEL_PATH.name,
        "ready": True,
        "warm_ms": round((time.perf_counter() - started)*1000.0, 3),
        "value_threshold": float(auq.value_threshold),
        "label_disclosure": card.get("label", {}).get("disclosure", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score one state against the real EDL head.")
    ap.add_argument("--state-json", help='{"features": [...28...], "quality_mask": [...]}; '
                                         "use - to read from stdin")
    ap.add_argument("--warm", action="store_true", help="prime the engine and exit")
    args = ap.parse_args()

    try:
        if args.warm:
            print(json.dumps(warm()))
            return 0
        if not args.state_json:
            ap.error("pass --state-json, or --warm")
        raw = sys.stdin.read() if args.state_json == "-" else args.state_json
        payload = json.loads(raw)
        print(json.dumps(score_state(payload.get("features"), payload.get("quality_mask"))))
        return 0
    except ArtifactMissing as exc:
        # Exit 2, matching module1.infer.forecast's convention for "the derived artifact
        # has not been built", so the gateway can tell it apart from a real failure.
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
