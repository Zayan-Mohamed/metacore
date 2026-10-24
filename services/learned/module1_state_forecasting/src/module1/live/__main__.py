"""Module 1 live-weather mode from the command line (ADR 0006).

    python -m module1.live fetch                 # the only external call: Open-Meteo -> JSON
    python -m module1.live step [--now HOUR]     # stored weather on stdin -> forecast, M2/M4 inputs

The orchestrator runs ``fetch``, stores the raw weather, and pipes it back into ``step``, so the
forecast never makes a network call and can be re-run from the stored weather.
"""

from __future__ import annotations

import argparse
import json
import sys

from module1.data import open_meteo as om
from module1.live import state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["fetch", "step"])
    ap.add_argument("--now", default=None, help="hour to forecast from (default: current hour)")
    ap.add_argument("--past-days", type=int, default=2)
    ap.add_argument("--forecast-days", type=int, default=2)
    args = ap.parse_args(argv)
    if args.command == "fetch":
        rows, meta = om.fetch_live(args.past_days, args.forecast_days)
        print(json.dumps({"rows": rows, "meta": meta}))
        return 0
    payload = json.load(sys.stdin)
    now = args.now or state.current_hour()
    try:
        out = state.step(payload["rows"], now, payload.get("meta"))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
