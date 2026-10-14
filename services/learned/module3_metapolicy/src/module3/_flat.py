"""Temporary bridge to the older flat scripts (system1.py, m2_stream.py, ...).

The first M3 starter keeps its code as flat files beside `config.yaml`; the new `module3`
package is still being filled in. Until those files are migrated into the package, the
package reaches them by putting their folder on `sys.path`. Delete this file when
`system1.py`, `m2_stream.py` and `priority.py` move under `src/module3/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

FLAT_DIR = Path(__file__).resolve().parents[2]  # .../module3_metapolicy


def ensure_flat_on_path() -> None:
    p = str(FLAT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
