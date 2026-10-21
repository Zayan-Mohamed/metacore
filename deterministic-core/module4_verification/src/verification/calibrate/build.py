"""Build and freeze the calibration artifact (Step 3: "it's frozen").

Everything a verify call needs from history goes into one versioned JSON file:

* the input hashes (load history and meteorology), the splits, horizon, alpha and seed;
* the hazard scaling bounds (calibration split only, so the test months cannot leak);
* every calibration miss row (worst rise / fall per island, in 1e-4 units) and its hazard level;
* the conformal band of every situation group, and the held-out coverage counts.

The file is canonical: sorted keys, fixed rounding, no timestamps of when it was built. Building
twice from the same inputs is byte-identical, and ``artifact_sha256`` seals the body so a
hand-edited or truncated file is refused at load time (risk/calibration.py).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..risk.bands import ALPHA, MIN_GROUP_ROWS, build_band, lookup, row_scores
from ..risk.history import hazard_levels, read_meteorology
from ..risk.situation import HAZARD_CUTS, HAZARD_LEVELS, SEASONS, Situation, hazard_scale_reference
from .groups import group_rows
from .misses import HORIZON_H, persistence_misses, read_hourly_load

SCHEMA = "m4-calibration/1"
CODE_VERSION = "verification-rcv/1.0"
SEED = 20251128
UNIT = 10_000  # miss fractions are stored as integers in 1e-4

SPLITS = {
    "calibration": ("2024-01-01T00:00", "2025-06-30T23:00"),
    "validation": ("2025-07-01T00:00", "2025-09-30T23:00"),
    "test": ("2025-10-01T00:00", "2025-12-31T23:00"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(hourly_load: Path, meteorology: Path, islands: tuple[str, ...]) -> dict:
    met = {f"{i}_hourly.csv": sha256_file(Path(meteorology) / f"{i}_hourly.csv")
           for i in islands}
    combined = hashlib.sha256(
        (sha256_file(hourly_load) + "".join(met[k] for k in sorted(met))).encode()
    ).hexdigest()
    return {"hourly_load": sha256_file(hourly_load), "meteorology": met, "combined": combined}


def canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def seal(body: dict) -> dict:
    """Adds ``artifact_sha256`` over the canonical form of everything else."""
    body = {k: v for k, v in body.items() if k != "artifact_sha256"}
    return {**body, "artifact_sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}


def render(artifact: dict) -> str:
    """The on-disk text: readable header, one miss row per line, stable byte for byte."""
    return json.dumps(artifact, sort_keys=True, indent=1) + "\n"


def build_artifact(hourly_load: Path, meteorology: Path) -> dict:
    load = read_hourly_load(hourly_load)
    wind, rain = read_meteorology(meteorology, load.islands, load.timestamps)
    cal_lo, cal_hi = SPLITS["calibration"]
    end = load.index_of(cal_hi)
    refs = {isl: hazard_scale_reference(wind[: end + 1, j], rain[: end + 1, j])
            for j, isl in enumerate(load.islands)}
    level_of = dict(zip(load.timestamps,
                        hazard_levels(wind, rain, refs, load.islands).tolist(), strict=True))

    cal = persistence_misses(load, cal_lo, cal_hi, HORIZON_H)
    # Quantise first, so the bands are computed from exactly the rows the runtime samples.
    up = np.round(cal.up * UNIT).astype(int)
    down = np.round(cal.down * UNIT).astype(int)
    levels = np.array([level_of[t] for t in cal.timestamps])
    s_up, s_down = row_scores(up / UNIT, down / UNIT)
    groups = {k: build_band(k, s_up[i], s_down[i], ALPHA)
              for k, i in group_rows(cal.timestamps, levels).items()}

    val = persistence_misses(load, *SPLITS["validation"], HORIZON_H)
    v_up, v_down = row_scores(val.up, val.down)
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ts, a, b in zip(val.timestamps, v_up, v_down, strict=True):
        band = lookup(groups, Situation.at(ts, level_of[ts]))
        hits[band.key][0] += int(band.covers(float(a), float(b)))
        hits[band.key][1] += 1

    root = _root_of(Path(hourly_load))
    body = {
        "schema": SCHEMA,
        "code_version": CODE_VERSION,
        "inputs": {
            "hourly_load": _rel(hourly_load, root),
            "meteorology": _rel(meteorology, root),
            "sha256": input_hashes(Path(hourly_load), Path(meteorology), load.islands),
        },
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "horizon_h": HORIZON_H,
        "alpha": ALPHA,
        "min_group_rows": MIN_GROUP_ROWS,
        "seed": SEED,
        "islands": list(load.islands),
        "seasons": {str(m): s for m, s in sorted(SEASONS.items())},
        "hazard": {"levels": list(HAZARD_LEVELS), "cuts": list(HAZARD_CUTS), "refs": refs},
        "groups": {k: b.to_json() for k, b in groups.items()},
        "validation": {
            "split": list(SPLITS["validation"]),
            "covered": int(sum(c for c, _ in hits.values())),
            "rows": int(sum(n for _, n in hits.values())),
            "per_group": {k: {"covered": c, "rows": n} for k, (c, n) in sorted(hits.items())},
        },
        "rows": {
            "start": cal.timestamps[0],
            "count": len(cal),
            "unit": UNIT,
            "format": "hazard_level up[islands] down[islands]",
            "data": [
                " ".join(str(x) for x in (int(lv), *u, *d))
                for lv, u, d in zip(levels, up.tolist(), down.tolist(), strict=True)
            ],
        },
    }
    return seal(body)


def _root_of(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        if (parent / "data" / "processed").is_dir():
            return parent
    return None


def _rel(path: Path, root: Path | None) -> str:
    p = Path(path).resolve()
    # POSIX separators so the sealed artifact is byte-identical on every OS, Windows included.
    return p.relative_to(root).as_posix() if root and p.is_relative_to(root) else p.name
