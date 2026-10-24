"""New CEB ledger months for the live system: a watched folder, a gate, a live-only store.

CEB keeps the monthly generation workbook by hand. When a newer copy is saved into the inbox
folder (``METACORE_LEDGER_INBOX``, default ``~/.metacore/ledger-inbox``; point it at a synced
OneDrive or Drive folder), the next hourly run:

1. parses it with the same declared layout as the calibration stage (``module1.data.ceb``);
2. gates the new months (below) and rejects the whole file on any failure;
3. appends the accepted months to the **live ledger** (``METACORE_LIVE_LEDGER``, default
   ``~/.metacore/ledger/live_ledger.csv``), which only ``module1.live`` reads.

The 2024-2025 calibration artifacts are never touched: the trained models and Module 4's risk
calibration are fitted on them, and changing them would silently change both. The raw workbook
stays on this machine; the live ledger holds the same columns as the calibration table.

The gate for months without a printed CEB summary to reconcile against:

* **layout pin**: barrels equal litres / 200 on every row (as in ``validate.py``);
* **complete months**: all five generating systems present, energy recorded;
* **plausibility against each system's own record**: monthly energy within 0.3x..2.5x of its
  2024-2025 range, specific fuel consumption within 0.6x..1.6x of its range. A shifted energy
  column breaks these immediately;
* **no rewriting history**: months already reconciled must match exactly.

Transport cost (barrels x rate) is not checked for a year whose barrel rate is not on record:
``validate.py`` forbids inferring a rate, so the gate says it was skipped.

    python -m module1.data.live_ledger ingest [--json]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from module1.data import ceb, validate

REPO_ROOT = Path(__file__).resolve().parents[6]
CALIBRATION = REPO_ROOT / "data" / "processed" / "ceb_generation_tidy.csv"
HOME = Path(os.environ.get("METACORE_HOME", Path.home() / ".metacore"))
INBOX = Path(os.environ.get("METACORE_LEDGER_INBOX", HOME / "ledger-inbox"))
STORE = Path(os.environ.get("METACORE_LIVE_LEDGER", HOME / "ledger" / "live_ledger.csv"))
SEEN = STORE.parent / "seen.json"
SYSTEMS = tuple(ceb.LAYOUT)
ENERGY_BAND = (0.3, 2.5)
SFC_BAND = (0.6, 1.6)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _key(r: dict) -> tuple[str, int, str]:
    return (str(r["year"]), int(r["month_num"]), r["island_system"])


def _num(v) -> float:
    return validate._num(v if isinstance(v, str) else ("" if v is None else str(v)))


def live_rows() -> list[dict]:
    """The ledger the live system uses: the live store if it exists, else the calibration table."""
    return _read(STORE if STORE.exists() else CALIBRATION)


def history_ranges(reference: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for system in SYSTEMS:
        rows = [r for r in reference if r["island_system"] == system and _num(r["units_kwh"]) > 0]
        kwh = [_num(r["units_kwh"]) for r in rows]
        sfc = [_num(r["diesel_l"]) / _num(r["units_kwh"]) for r in rows if _num(r["diesel_l"]) > 0]
        out[system] = {"kwh": (min(kwh), max(kwh)), "sfc": (min(sfc), max(sfc)) if sfc else (0, 0)}
    return out


def gate(records: list[dict], reference: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """(new month rows to accept, failures, notes). Any failure rejects the whole file."""
    known = {_key(r): r for r in reference}
    ranges = history_ranges(reference)
    failures: list[str] = []
    notes: list[str] = []
    new: list[dict] = []
    by_month: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        if r.get("units_kwh") in (None, "") and r.get("diesel_l") in (None, ""):
            continue  # an empty month row: not reported yet
        by_month.setdefault((str(r["year"]), int(r["month_num"])), []).append(r)

    for (year, month), rows in sorted(by_month.items()):
        label = f"{year}-{month:02d}"
        systems = {r["island_system"] for r in rows}
        if systems != set(SYSTEMS):
            failures.append(f"{label}: systems present {sorted(systems)}, expected all five")
            continue
        for r in rows:
            k = _key(r)
            if k in known:
                for field in ceb.MEASURED:
                    if abs(_num(known[k][field]) - _num(r[field])) > validate.TOLERANCE:
                        failures.append(f"{label} {r['island_system']}: {field} differs from the "
                                        f"reconciled record ({known[k][field]} vs {r[field]})")
                continue
            litres, barrels = _num(r["diesel_l"]), _num(r["diesel_barrel"])
            if abs(litres / ceb.LITRES_PER_BARREL - barrels) > validate.TOLERANCE:
                failures.append(f"{label} {r['island_system']}: {litres} L / 200 != {barrels} "
                                "barrels (column layout changed?)")
            kwh = _num(r["units_kwh"])
            lo, hi = ranges[r["island_system"]]["kwh"]
            if not (ENERGY_BAND[0] * lo <= kwh <= ENERGY_BAND[1] * hi):
                failures.append(f"{label} {r['island_system']}: {kwh:,.0f} kWh outside "
                                f"{ENERGY_BAND[0] * lo:,.0f}..{ENERGY_BAND[1] * hi:,.0f} "
                                "(its own 2024-2025 range, widened)")
            if litres > 0 and kwh > 0:
                slo, shi = ranges[r["island_system"]]["sfc"]
                sfc = litres / kwh
                if not (SFC_BAND[0] * slo <= sfc <= SFC_BAND[1] * shi):
                    failures.append(f"{label} {r['island_system']}: {sfc:.3f} L/kWh outside "
                                    f"{SFC_BAND[0] * slo:.3f}..{SFC_BAND[1] * shi:.3f}")
            if (str(r["year"]), r["island_system"]) not in validate.BARREL_RATE_RS:
                notes.append(f"{year} {r['island_system']}: transport cost not checked "
                             "(no barrel rate on record for this year)")
            new.append(r)
    return new, failures, sorted(set(notes))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(inbox: Path = INBOX, store: Path = STORE) -> list[dict]:
    """Process every workbook in the inbox not seen before. One result per file."""
    inbox.mkdir(parents=True, exist_ok=True)
    store.parent.mkdir(parents=True, exist_ok=True)
    seen_path = store.parent / "seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    results = []
    for wb in sorted(inbox.glob("*.xlsx")):
        if wb.name.startswith("~$"):
            continue  # an Excel lock file while the workbook is open
        sha = _sha(wb)
        if sha in seen:
            continue
        result = {"file": wb.name, "sha256": sha,
                  "received": datetime.now().astimezone().isoformat(timespec="seconds")}
        try:
            records = ceb.extract(wb)
        except Exception as exc:  # noqa: BLE001 -- an unreadable file is a rejection, not a crash
            result |= {"status": "rejected", "failures": [f"could not read workbook: {exc}"],
                       "months_added": [], "notes": []}
        else:
            reference = _read(store) if store.exists() else _read(CALIBRATION)
            new, failures, notes = gate(records, reference)
            if failures:
                result |= {"status": "rejected", "failures": failures[:20], "months_added": [],
                           "notes": notes}
            else:
                if not store.exists():
                    shutil.copyfile(CALIBRATION, store)
                with store.open("a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=ceb.FIELDS)
                    for r in new:
                        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in ceb.FIELDS})
                months = sorted({f"{r['year']}-{int(r['month_num']):02d}" for r in new})
                result |= {"status": "accepted" if months else "no new months",
                           "failures": [], "months_added": months, "notes": notes}
        seen[sha] = result
        results.append(result)
    seen_path.write_text(json.dumps(seen, indent=1))
    return results


def status(store: Path = STORE) -> dict:
    """What the live system currently knows: the latest ledger month and the last file seen."""
    rows = live_rows()
    months = sorted({f"{r['year']}-{int(r['month_num']):02d}" for r in rows})
    seen_path = store.parent / "seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    last = max(seen.values(), key=lambda r: r["received"]) if seen else None
    return {"latest_month": months[-1], "months": len(months), "inbox": str(INBOX),
            "source": "live ledger" if store.exists() else "calibration table (2024-2025)",
            "last_file": last}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["ingest", "status"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "ingest":
        out = {"results": ingest(), "status": status()}
    else:
        out = status()
    print(json.dumps(out) if args.json else json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
