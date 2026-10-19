"""Run one custom cyclone through the trained gate: `storm_run.py <request.json> <result.json>`.

Thin wrapper around `module3.serve.storm.simulate_storm`, called by the gateway
(`POST /api/module3/storm`). Exit code 3 = the request cannot be served (bad value or no
trained gate), which the gateway turns into HTTP 409 with the message on stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from module3.serve.storm import StormInputError, preset_table, simulate_storm  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "presets":
        Path(argv[2]).write_text(json.dumps(preset_table()), encoding="utf-8")
        return 0
    if len(argv) != 3:
        print("usage: storm_run.py <request.json> <result.json> | presets <result.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        result = simulate_storm(spec)
    except StormInputError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    Path(argv[2]).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
