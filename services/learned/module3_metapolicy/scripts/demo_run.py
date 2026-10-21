"""Train + score a small demo gate: `demo_run.py <request.json> <result.json>`.

Thin wrapper around `module3.serve.demo.run_demo`, called by the gateway (`POST /api/module3/run`).
It writes the trained gate to artifacts/dashboard/gate_seed0.pt, which Generate/Process then read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from module3.serve.demo import build_config, run_demo  # noqa: E402
from module3.serve.step import STATE_DIR  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: demo_run.py <request.json> <result.json>", file=sys.stderr)
        return 2
    req = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    result = run_demo(build_config(base, req), STATE_DIR, log=lambda m: print(m, file=sys.stderr))
    Path(argv[2]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
