"""Dashboard step buttons: `step_run.py generate|process <result.json>`.

Thin wrapper around `module3.serve.step`, called by the gateway (`POST /api/module3/generate`
and `/process`). Exit code 3 = not ready (no trained gate yet, or nothing generated yet), which
the gateway turns into HTTP 409 with the message on stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from module3.serve.step import generate, process  # noqa: E402
from module3.serve.storm import StormInputError  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"generate", "process"}:
        print("usage: step_run.py generate|process <result.json>", file=sys.stderr)
        return 2
    try:
        (generate if argv[1] == "generate" else process)(Path(argv[2]))
    except StormInputError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
