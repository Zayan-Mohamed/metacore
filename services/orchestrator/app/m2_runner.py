"""Module 2 on one live hour: every node's 28-feature state through the real EDL head.

Run as its own process (the orchestrator pipes the M1 contract states in on stdin), with
Module 2's directory on the path, exactly as its own ``serve_real.py`` loads the engine. Nothing
of Module 2's logic is reimplemented here: this calls ``OnnxAUQ.score_state`` per node and returns
the M2 -> M3 contract message for each.

The trigger keeps Module 2's configured hysteresis (a drop must hold for two consecutive
hours), so each node is scored as a two-hour stream: the previous hour, then this one.

    python app/m2_runner.py < contract_state.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

M2_DIR = Path(__file__).resolve().parents[2] / "learned" / "module2_auq_engine"


def main() -> int:
    sys.path.insert(0, str(M2_DIR))
    try:
        import serve_real
    except Exception as exc:  # noqa: BLE001
        print(f"Module 2 could not be imported: {exc}", file=sys.stderr)
        return 2
    payload = json.load(sys.stdin)
    try:
        auq, card = serve_real.load_engine()
    except Exception as exc:  # noqa: BLE001 -- e.g. edl_real.onnx not on this machine
        print(f"Module 2 engine unavailable: {exc}", file=sys.stderr)
        return 2
    class_names = card["architecture"].get("class_names", [])
    t0 = time.perf_counter()
    prev = dict(zip(payload["node_ids"], payload["node_features_prev"], strict=True))
    nodes = []
    for node_id, features in zip(payload["node_ids"], payload["node_features"], strict=True):
        auq.reset()
        auq.score_state(prev[node_id], payload["quality_mask"])
        out = auq.score_state(features, payload["quality_mask"])
        nodes.append({"node": node_id, **asdict(out)})
    print(json.dumps({
        "head": "real", "artifact": serve_real.MODEL_PATH.name, "class_names": class_names,
        "value_threshold": float(auq.value_threshold), "hysteresis_h": auq.hysteresis,
        "nodes": nodes,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
