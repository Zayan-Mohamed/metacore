"""The restated grid facts in topology.py must match the real .dss model line for line.

M3 cannot import OpenDSS (ADR 0003), so it copies the numbers. This test is the guard that
turns 'copied' into 'verified': it parses the .dss text and fails if the two ever drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from module3.twin import topology as topo

DSS = (
    Path(__file__).resolve().parents[4]
    / "deterministic-core"
    / "module4_verification"
    / "src"
    / "verification"
    / "opendss"
    / "models"
    / "delft_3island.dss"
)


def _kv(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def _entries(kind: str) -> list[tuple[str, dict[str, str]]]:
    out = []
    for raw in DSS.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*New {kind}\.(\w+)\s+(.*)", raw)
        if m:
            out.append((m.group(1), _kv(m.group(2))))
    return out


def test_dss_file_found():
    assert DSS.is_file(), DSS


def test_loads_match():
    dss = {kv["Bus1"]: float(kv["kW"]) for _, kv in _entries("Load")}
    assert dss == topo.LOAD_KW


def test_generators_match():
    dss = {kv["Bus1"]: (name, float(kv["kW"])) for name, kv in _entries("Generator")}
    mine = {n: (g[0], g[2]) for n, g in topo.GENERATORS.items()}
    assert dss == mine


def test_lines_match():
    dss = {
        name: (kv["Bus1"], kv["Bus2"], float(kv["NormAmps"]), kv["LineCode"])
        for name, kv in _entries("Line")
    }
    mine = {ln.name: (ln.bus1, ln.bus2, ln.norm_amps) for ln in topo.LINES}
    assert {k: v[:3] for k, v in dss.items()} == mine
    overhead = {k for k, v in dss.items() if v[3] == "OverheadFeeder"}
    assert overhead == set(topo.OVERHEAD_LINES)


def test_nodes_are_exactly_the_buses_with_load():
    assert set(topo.NODES) == set(topo.LOAD_KW)
    assert "N7" not in topo.NODES and "N10" not in topo.NODES  # absent from the twin


def test_tiers_follow_the_contract_map():
    assert [topo.PRIORITY_TIER[n] for n in ("N1", "N3", "N4", "N6", "N8", "N12")] == [
        1,
        1,
        2,
        2,
        3,
        3,
    ]
