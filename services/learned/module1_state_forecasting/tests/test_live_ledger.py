"""The live ledger gate: new CEB months are accepted only if they cannot be a mis-read file.

The calibration stage reconciles 2024-2025 against CEB's printed summary. Later months have no
such summary, so this gate must catch a shifted column or a half-filled month on its own.
Records are built from the reconciled table itself, relabelled as 2026.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from module1.data import live_ledger as ll

CAL = Path(__file__).resolve().parents[4] / "data" / "processed" / "ceb_generation_tidy.csv"
needs_cal = pytest.mark.skipif(not CAL.exists(), reason="calibration table absent")


def _ref() -> list[dict]:
    with CAL.open() as fh:
        return list(csv.DictReader(fh))


def _as_2026(month: int) -> list[dict]:
    rows = [dict(r) for r in _ref() if r["year"] == "2025" and int(r["month_num"]) == month]
    for r in rows:
        r["year"] = "2026"
    return rows


@needs_cal
def test_a_plausible_new_month_is_accepted_with_the_rate_caveat() -> None:
    new, failures, notes = ll.gate(_as_2026(3), _ref())
    assert failures == [] and len(new) == 5
    assert any("transport cost not checked" in n for n in notes)


@needs_cal
def test_a_shifted_energy_column_is_rejected() -> None:
    rows = _as_2026(3)
    for r in rows:
        r["units_kwh"] = r["diesel_cost_rs"]  # the next column over: rupees read as kWh
    _, failures, _ = ll.gate(rows, _ref())
    assert failures and any("kWh outside" in f for f in failures)


@needs_cal
def test_a_broken_fuel_layout_is_rejected() -> None:
    rows = _as_2026(3)
    rows[0]["diesel_barrel"] = str(float(rows[0]["diesel_barrel"]) + 5)
    _, failures, _ = ll.gate(rows, _ref())
    assert any("barrels" in f for f in failures)


@needs_cal
def test_a_half_filled_month_is_rejected() -> None:
    rows = _as_2026(3)[:3]
    _, failures, _ = ll.gate(rows, _ref())
    assert any("expected all five" in f for f in failures)


@needs_cal
def test_rewriting_a_reconciled_month_is_rejected() -> None:
    rows = [dict(r) for r in _ref() if r["year"] == "2025" and int(r["month_num"]) == 3]
    rows[0]["units_kwh"] = str(float(rows[0]["units_kwh"]) + 100)
    _, failures, _ = ll.gate(rows, _ref())
    assert any("differs from the reconciled record" in f for f in failures)


@needs_cal
def test_repeating_reconciled_months_adds_nothing() -> None:
    new, failures, _ = ll.gate(_ref(), _ref())
    assert failures == [] and new == []
