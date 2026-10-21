"""Load and hash-check the frozen calibration artifact (the runtime half of C7).

A verify call never recomputes a band: it reads them from ``artifacts/calibration_v1.json``.
Two checks guard that file:

* the seal: ``artifact_sha256`` must match the canonical body, so an edited or truncated
  artifact is refused;
* optionally the inputs: when the recorded history is at hand, its hashes must match the ones
  the artifact was built from. A changed input means the bands describe data that no longer
  exists, and the verifier refuses the stale artifact (the certifier turns that into UNVERIFIED).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from ..calibrate.groups import group_rows
from .bands import Band, lookup
from .situation import Situation, hazard_index, hazard_level

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "calibration_v1.json"


class StaleCalibrationError(RuntimeError):
    """The artifact is not sealed correctly or no longer matches its inputs."""


def _canonical_hash(body: dict) -> str:
    body = {k: v for k, v in body.items() if k != "artifact_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class Calibration:
    """The frozen bands, miss rows and hazard scaling, ready for lookup."""

    raw: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_ARTIFACT) -> Calibration:
        raw = json.loads(Path(path).read_text())
        if raw.get("artifact_sha256") != _canonical_hash(raw):
            raise StaleCalibrationError(f"{path}: artifact_sha256 does not match its body")
        return cls(raw)

    # ------------------------------------------------------------------ identity

    @property
    def sha256(self) -> str:
        return self.raw["artifact_sha256"]

    @property
    def version(self) -> str:
        return f"{self.raw['schema']}@{self.sha256[:12]}"

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def alpha(self) -> float:
        return float(self.raw["alpha"])

    @property
    def horizon_h(self) -> int:
        return int(self.raw["horizon_h"])

    @property
    def islands(self) -> tuple[str, ...]:
        return tuple(self.raw["islands"])

    def check_inputs(self, hourly_load: Path, meteorology: Path) -> None:
        """Raises StaleCalibrationError unless the inputs hash to what the artifact recorded."""
        from ..calibrate.build import input_hashes

        now = input_hashes(Path(hourly_load), Path(meteorology), self.islands)
        if now["combined"] != self.raw["inputs"]["sha256"]["combined"]:
            raise StaleCalibrationError(
                "recorded history changed since calibration_v1.json was built; "
                "run `python -m verification.calibrate build`")

    # ------------------------------------------------------------------ bands

    @cached_property
    def bands(self) -> dict[str, Band]:
        return {k: Band.from_json(g, self.alpha) for k, g in self.raw["groups"].items()}

    def band_for(self, situation: Situation) -> Band:
        return lookup(self.bands, situation)

    # ------------------------------------------------------------------ miss rows

    @cached_property
    def _rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.raw["rows"]
        k = len(self.islands)
        data = np.array([[int(x) for x in line.split()] for line in rows["data"]], dtype=int)
        unit = float(rows["unit"])
        return data[:, 0], data[:, 1 : 1 + k] / unit, data[:, 1 + k :] / unit

    @property
    def row_hazard(self) -> np.ndarray:
        return self._rows[0]

    @property
    def up(self) -> np.ndarray:
        return self._rows[1]

    @property
    def down(self) -> np.ndarray:
        return self._rows[2]

    @cached_property
    def row_timestamps(self) -> tuple[str, ...]:
        start = np.datetime64(self.raw["rows"]["start"])
        n = int(self.raw["rows"]["count"])
        return tuple(str(t)[:16] for t in start + np.arange(n) * np.timedelta64(1, "h"))

    @cached_property
    def group_members(self) -> dict[str, np.ndarray]:
        return group_rows(self.row_timestamps, self.row_hazard)

    def rows_in(self, key: str) -> np.ndarray:
        """Indices of the calibration rows in a group, in time order."""
        return self.group_members[key]

    # ------------------------------------------------------------------ hazard

    def hazard_level_at(self, wind: np.ndarray, rain: np.ndarray) -> int:
        """Hazard level from one hour of meteorology, ``wind[i]``/``rain[i]`` per island."""
        refs = self.raw["hazard"]["refs"]
        idx = max(float(hazard_index(np.array([wind[j]]), np.array([rain[j]]), refs[isl])[0])
                  for j, isl in enumerate(self.islands))
        return int(hazard_level(idx))
