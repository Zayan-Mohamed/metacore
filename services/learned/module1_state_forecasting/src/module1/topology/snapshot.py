"""GridStateSnapshot producer: what M1 can honestly hand M4's twin for one hour.

Fills feeder-group demand from the constructed hourly island load (QUALITY_INTERPOLATED), and
leaves bus voltage and battery state of charge QUALITY_MISSING: no telemetry exists for either,
and voltage is what M4's power flow computes. It never invents the missing values.

It also lists hybrid assets the ledger shows were down that month (the scenario library's core
rule: hybrid energy below 0.50 of its median), so M4 does not credit a dead battery bank.
This is a replay of the recorded state, not a forecast input.
"""

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime
from pathlib import Path

from module1.assemble import _load_layout, _resolve_data_root
from module1.data.scenarios import CORE_RATIO, ratios, read_plant_months
from module1.topology.spec import POWER_FACTOR, SCHEMA_VERSION, TOPOLOGY_ID, build_spec

OBSERVED, INTERPOLATED, MISSING = "QUALITY_OBSERVED", "QUALITY_INTERPOLATED", "QUALITY_MISSING"
HYBRID_PLANT = "Eluvaitivu-Hybrid"


def _island_load(load_csv: Path, timestamp: str) -> dict[str, float]:
    with load_csv.open(newline="") as fh:
        rows = {r["island"]: float(r["load_kw"]) for r in csv.DictReader(fh)
                if r["timestamp_lst"] == timestamp}
    if not rows:
        raise ValueError(f"{timestamp!r} is not an hour in {load_csv.name}")
    return rows


def _hybrid_down(tidy_csv: Path, timestamp: str) -> bool:
    months = read_plant_months(tidy_csv).get(HYBRID_PLANT, {})
    if not months:
        return False
    _, ratio = ratios(months)
    key = (int(timestamp[:4]), int(timestamp[5:7]))
    return key in ratio and ratio[key] < CORE_RATIO


def build_snapshot(timestamp: str, repo_root: Path, data_root=None,
                   emitted_at: str | None = None) -> dict:
    spec = build_spec(repo_root)
    root = _resolve_data_root(data_root)
    load_csv, _, _, _ = _load_layout(root)
    tidy_csv = (root / "processed" / "ceb_generation_tidy.csv" if (root / "processed").is_dir()
                else root / "ceb_generation_tidy.csv")
    return snapshot_from(timestamp, _island_load(load_csv, timestamp),
                         _hybrid_down(tidy_csv, timestamp), spec,
                         "real" if root.name == "data" else "synthetic", emitted_at)


def snapshot_from(timestamp: str, island_kw: dict[str, float], hybrid_down: bool, spec: dict,
                  data_source: str, emitted_at: str | None = None,
                  scenario: str | None = None) -> dict:
    """One hour's GridStateSnapshot from island demand. The live worker passes today's load."""
    q_ratio = math.tan(math.acos(POWER_FACTOR))
    shares = spec["assumptions"]["feeder_shares"]

    node_states = []
    for bus in spec["buses"]:
        if bus["role"] != "feeder_group":
            continue
        share = shares[bus["node_id"].split("_", 1)[1]]
        p = island_kw[bus["island"]] * share
        node_states.append({
            "node_id": bus["node_id"],
            "voltage_pu": None,
            "p_kw": round(p, 3),
            "q_kvar": round(p * q_ratio, 3),
            "soc_fraction": None,
            "quality": {"voltage_pu": MISSING, "p_kw": INTERPOLATED,
                        "q_kvar": INTERPOLATED, "soc_fraction": MISSING},
        })
    for bus in spec["buses"]:
        if bus["role"] == "hybrid_plant":
            node_states.append({
                "node_id": bus["node_id"], "voltage_pu": None, "p_kw": None, "q_kvar": None,
                "soc_fraction": None,
                "quality": {"voltage_pu": MISSING, "p_kw": MISSING, "q_kvar": MISSING,
                            "soc_fraction": MISSING},
            })

    out_of_service = [a["element"] for a in spec["assets"] if not a["in_service"]]
    if hybrid_down:
        out_of_service += [a["element"] for a in spec["assets"]
                           if a["bus"].endswith("_HYB") and a["in_service"]
                           and a["asset_type"] in ("PV", "BESS")]
    return {
        "envelope": {
            "schema_version": SCHEMA_VERSION,
            "emitted_at": emitted_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "producer": "module1",
            "scenario": scenario,
            "trace_id": f"m1-snapshot-{timestamp}",
        },
        "topology_id": TOPOLOGY_ID,
        "timestamp_lst": timestamp,
        "update_period_s": spec["snapshot_format"]["update_period_s"],
        "node_states": node_states,
        "assets_out_of_service": sorted(out_of_service),
        "hybrid_plant_down": hybrid_down,
        "data_source": data_source,
    }
