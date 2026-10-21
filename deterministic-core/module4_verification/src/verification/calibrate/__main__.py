"""``python -m verification.calibrate build|check [--hourly-load CSV] [--meteorology DIR]``.

build  rebuild artifacts/calibration_v1.json from the recorded history.
check  rebuild in memory and fail (exit 1) unless it is byte-identical to the committed file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..risk.calibration import DEFAULT_ARTIFACT
from ..risk.history import default_paths
from .build import build_artifact, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m verification.calibrate")
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--hourly-load", type=Path)
    parser.add_argument("--meteorology", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)

    defaults = default_paths() if not (args.hourly_load and args.meteorology) else {}
    text = render(build_artifact(args.hourly_load or defaults["hourly_load"],
                                 args.meteorology or defaults["meteorology"]))
    if args.command == "build":
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text):,} bytes)")
        return 0
    if not args.out.exists() or args.out.read_text() != text:
        print(f"STALE: {args.out} differs from a rebuild of the recorded history", file=sys.stderr)
        return 1
    print(f"OK: {args.out} rebuilds byte-identically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
