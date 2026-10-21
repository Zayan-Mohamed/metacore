"""Run the M4 evaluation and write reports/data-analysis-report/m4/.

    uv run --no-sync python -m evaluation.run_eval [--stride 7] [--quick]

Writes
    eval_results.json   every number in the report; re-running gives identical bytes
    latency.json        wall-clock timings (machine-dependent, kept apart for that reason)
    README.md           the report: E1 to E6 against baselines B0 to B4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from verification.firewall.verifier import PhysicsVerifier
from verification.opendss.circuit import CircuitTwin
from verification.risk.bands import Band
from verification.risk.calibration import Calibration
from verification.risk.certifier import CertifierConfig, RiskCertifiedVerifier
from verification.risk.pid import ConformalPID, ReplayHistory, in_storm
from verification.risk.situation import Situation
from verification.risk.statistics import cp_lower, cp_upper
from verification.types import Decision, ViolationType

from .baselines import baselines
from .benchmark_set import HORIZON_H, TEST_SPLIT, Case, SnapshotFactory, build_cases

OUT = Path(__file__).resolve().parents[3] / "reports" / "data-analysis-report" / "m4"
DITWAH = ("2025-11-25T00:00", "2025-12-01T23:00")
MALFORMED = ViolationType.VIOLATION_TYPE_MALFORMED_ACTION
SCHEMA = "m4-eval/1"


def ground_truth(case: Case, factory: SnapshotFactory, gate: PhysicsVerifier) -> bool | None:
    """True if the action breaks a limit on the recorded hour or any of the next six.

    None when the action is malformed on the hour (not a safety question).
    """
    unsafe = False
    for h in range(0, HORIZON_H + 1):
        ts = factory.later(case.timestamp, h)
        verdict = gate.verify(case.action, factory.snapshot(ts, case.extra_outages))
        if any(v.type == MALFORMED for v in verdict.violations):
            if h == 0:
                return None
            continue
        if verdict.decision != Decision.DECISION_APPROVE:
            unsafe = True
    return unsafe


def _rate(num: int, den: int) -> dict:
    """A rate with its 95 % Clopper-Pearson interval (two one-sided 97.5 % bounds)."""
    if den == 0:
        return {"k": num, "n": 0, "rate": None, "ci95": None}
    return {"k": num, "n": den, "rate": round(num / den, 4),
            "ci95": [round(cp_lower(num, den, 0.975), 4), round(cp_upper(num, den, 0.975), 4)]}


def e1_metrics(rows: list[dict], key: str) -> dict:
    out = {}
    for window, pick in (("all", lambda r: True), ("calm", lambda r: not r["storm"]),
                         ("storm", lambda r: r["storm"]), ("ditwah", lambda r: r["ditwah"])):
        sel = [r for r in rows if pick(r)]
        unsafe = [r for r in sel if r["unsafe"]]
        safe = [r for r in sel if not r["unsafe"]]
        out[window] = {
            "cases": len(sel),
            "unsafe": len(unsafe),
            "false_approval": _rate(sum(r[key] == "APPROVE" for r in unsafe), len(unsafe)),
            "false_rejection": _rate(sum(r[key] != "APPROVE" for r in safe), len(safe)),
            "unverified": sum(r[key] == "UNVERIFIED" for r in sel),
        }
    return out


def e2_coverage(cal: Calibration, hist: ReplayHistory) -> dict:
    """Empirical coverage of the band vs 1 - alpha: static vs conformal PID, per split."""
    h = cal.horizon_h
    res = {}
    for split, (start, end) in {"validation": ("2025-07-01T00:00", "2025-09-30T17:00"),
                                "test": TEST_SPLIT, "ditwah": DITWAH}.items():
        lo, hi = hist.index(start), min(hist.index(end), len(hist.timestamps) - h - 1)
        pid = ConformalPID(cal.alpha)
        used_s: dict[int, Band] = {}
        used_p: dict[int, Band] = {}
        miss = {"static": [], "pid": [], "static_storm": [], "pid_storm": []}
        for t in range(lo - 14 * 24, hi + 1):
            done = t - h
            if done in used_p:
                s = hist.realized_scores(done, h)
                bs, bp = used_s.pop(done), used_p.pop(done)
                pid.observe(bp, *s)
                if done >= lo:
                    storm = in_storm(hist, done)
                    for name, b in (("static", bs), ("pid", bp)):
                        e = 0 if b.covers(*s) else 1
                        miss[name].append(e)
                        if storm:
                            miss[f"{name}_storm"].append(e)
            base = cal.band_for(Situation.at(hist.timestamps[t], int(hist.hazard[t])))
            used_s[t] = base
            used_p[t] = pid.adapted(base, int(hist.hazard[t]))
        res[split] = {
            "target_coverage": round(1 - cal.alpha, 4),
            **{name: {"coverage": _rate(len(v) - sum(v), len(v))} for name, v in miss.items()},
        }
    return res


def run(stride: int, quick: bool) -> tuple[dict, dict]:
    cal = Calibration.load()
    hist = ReplayHistory.load(cal)
    factory = SnapshotFactory()
    circuit = CircuitTwin()
    gate = PhysicsVerifier(circuit)
    cases = build_cases(factory, stride=stride)
    if quick:
        cases = cases[:: max(1, len(cases) // 12)][:12]
    methods = baselines(hist, circuit)
    sensitivity = {scale: baselines(hist, circuit, band_scale=scale, attribution=False)[4]
                   for scale in (1.5, 2.0)}

    rows: list[dict] = []
    attributions: list[dict] = []
    severities: dict[str, list[float]] = defaultdict(list)
    slpsr = PhysicsVerifier(circuit, profile="SL-PSR-2016-LV")
    slpsr_agree = Counter()
    for case in cases:
        truth = ground_truth(case, factory, gate)
        if truth is None:
            continue
        snap = factory.snapshot(case.timestamp, case.extra_outages)
        row = {"timestamp": case.timestamp, "family": case.family, "unsafe": truth,
               "storm": in_storm(hist, hist.index(case.timestamp)),
               "ditwah": DITWAH[0] <= case.timestamp <= DITWAH[1]}
        for m in methods:
            v = m.verify(case.action, snap)
            row[m.key] = v.decision.value.replace("DECISION_", "")
            if m.key == "B4" and v.attribution and v.certificate:
                by_island = v.certificate.failures_by_island
                attributions.append({"case": case.action.action_id,
                                     "target": case.target_island,
                                     "most_failures": max(sorted(by_island),
                                                          key=lambda i: by_island[i]),
                                     "top": v.attribution[0].island,
                                     "grounded": _grounded(v)})
            if m.key in ("B0", "B4") and v.decision == Decision.DECISION_REJECT:
                worst = max((abs(x.margin_fraction) for x in v.violations), default=0.0)
                severities[m.key].append(round(worst, 4))
            if m.key == "B0" and v.decision == Decision.DECISION_REJECT:
                slpsr_agree[slpsr.verify(case.action, snap).decision.value] += 1
        for scale, b in sensitivity.items():
            row[f"B4x{scale}"] = b.verify(case.action, snap).decision.value.replace(
                "DECISION_", "")
        rows.append(row)

    results = {
        "schema": SCHEMA,
        "calibration": cal.version,
        "split": list(TEST_SPLIT),
        "stride_h": stride,
        "quick": quick,
        "cases": len(rows),
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "E1": {m.key: {"name": m.name, **e1_metrics(rows, m.key)} for m in methods},
        "E2": e2_coverage(cal, hist),
        "E4": {
            "rejections_attributed": len(attributions),
            # The island whose feeders fail most is the cause Check B observed directly.
            "top_player_on_failing_island": _rate(
                sum(a["top"] == a["most_failures"] for a in attributions), len(attributions)),
            # The island the action acted on: the planted cause, when the action is the stressor.
            "planted_cause_recovered": _rate(
                sum(a["top"] == a["target"] for a in attributions if a["target"]),
                sum(1 for a in attributions if a["target"])),
            "grounding_pass": _rate(sum(bool(a["grounded"]) for a in attributions),
                                    len(attributions)),
        },
        "E5": {f"x{scale}": e1_metrics(rows, f"B4x{scale}") for scale in (1.0, 1.5, 2.0)
               if scale != 1.0} | {"x1.0": e1_metrics(rows, "B4")},
        "E6": {
            "worst_margin_of_rejections": {
                k: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}
                for k, v in severities.items() if v},
            "b0_rejections_also_rejected_by_sl_psr_2016": dict(slpsr_agree),
        },
    }
    latency = measure_latency(cases[:: max(1, len(cases) // 30)][:30], factory, circuit, hist)
    return results, latency


def _grounded(verdict) -> bool | None:
    try:
        from translation.templates.causal_logger import GroundingValidator, TemplateCausalLogger
        from translation.types import VerificationVerdict as TV
    except ImportError:  # the isolated core lane has no translation package
        return None
    tv = TV.model_validate(verdict.model_dump())
    log = TemplateCausalLogger.generate_log(tv, include_latency=False)
    from translation.abductive.attribution import TopologyEntities

    return GroundingValidator(TopologyEntities.load()).check(log, tv).grounded


def measure_latency(cases: list[Case], factory: SnapshotFactory, circuit: CircuitTwin,
                    hist: ReplayHistory) -> dict:
    """E3: wall-clock per verify call, plain gate vs certificate, with/without early stopping."""
    gate = PhysicsVerifier(circuit)
    with_stop = RiskCertifiedVerifier(circuit, history=hist)
    no_stop = RiskCertifiedVerifier(circuit, history=hist,
                                    config=CertifierConfig(early_stopping=False))
    times: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        snap = factory.snapshot(case.timestamp)
        for name, fn in (("plain_gate", lambda a, s: gate.verify(a, s)),
                         ("rcv_early_stopping", with_stop.verify),
                         ("rcv_full_688", no_stop.verify)):
            t0 = time.perf_counter()
            fn(case.action, snap)
            times[name].append((time.perf_counter() - t0) * 1000.0)

    def summary(v: list[float]) -> dict:
        s = sorted(v)
        return {"n": len(s), "p50_ms": round(statistics.median(s), 1),
                "p95_ms": round(s[int(0.95 * (len(s) - 1))], 1), "max_ms": round(s[-1], 1)}

    return {name: summary(v) for name, v in times.items()}


def render(results: dict, latency: dict) -> str:
    e1 = results["E1"]
    lines = [
        "# M4 evaluation: risk-certified verification",
        "",
        f"Generated by `python -m evaluation.run_eval` (calibration `{results['calibration']}`).",
        f"Test split {results['split'][0]} to {results['split'][1]} (held out from calibration),",
        f"one hour every {results['stride_h']} h, {results['cases']} cases. Ground truth: the",
        "action is unsafe if it breaks a limit on the recorded hour or any of the next six",
        "recorded hours. Rates carry 95 % Clopper-Pearson intervals.",
        "",
        "## E1: false approvals and false rejections",
        "",
        "| Gate | Window | Cases | Unsafe | False approval | False rejection | Unverified |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, m in e1.items():
        for window in ("calm", "storm", "ditwah"):
            w = m[window]
            lines.append(f"| {key} {m['name']} | {window} | {w['cases']} | {w['unsafe']} | "
                         f"{_fmt(w['false_approval'])} | {_fmt(w['false_rejection'])} | "
                         f"{w['unverified']} |")
    lines += ["", "## E2: band coverage (target "
              f"{results['E2']['test']['target_coverage']:.0%})", "",
              "| Split | Static | Conformal PID | Static, storm hours | PID, storm hours |",
              "|---|---|---|---|---|"]
    for split, r in results["E2"].items():
        lines.append(f"| {split} | {_fmt(r['static']['coverage'])} | {_fmt(r['pid']['coverage'])}"
                     f" | {_fmt(r['static_storm']['coverage'])} | "
                     f"{_fmt(r['pid_storm']['coverage'])} |")
    lines += ["", "## E3: latency per verify call (this machine, see latency.json)", "",
              "| Path | p50 | p95 | max |", "|---|---|---|---|"]
    for name, s in latency.items():
        lines.append(f"| {name} | {s['p50_ms']} ms | {s['p95_ms']} ms | {s['max_ms']} ms |")
    e4 = results["E4"]
    lines += ["", "## E4: does the attribution point at the right feeder?", "",
              f"Cases with failing scenarios and an attribution: "
              f"{e4['rejections_attributed']}. The top Shapley player sits on the island with "
              f"the most failing scenarios in {_fmt(e4['top_player_on_failing_island'])}, and on "
              f"the island the action acted on (the planted cause) in "
              f"{_fmt(e4['planted_cause_recovered'])}. The operator sentence passed the "
              f"grounding check in {_fmt(e4['grounding_pass'])}.", "",
              "## E5: sensitivity to the band width (B4)", "",
              "| Band scale | Window | False approval | False rejection |", "|---|---|---|---|"]
    for scale, m in sorted(results["E5"].items()):
        for window in ("calm", "storm"):
            lines.append(f"| {scale} | {window} | {_fmt(m[window]['false_approval'])} | "
                         f"{_fmt(m[window]['false_rejection'])} |")
    e6 = results["E6"]
    lines += ["", "## E6: severity of rejected actions", "",
              "| Gate | Rejections | Worst margin min | median | max |", "|---|---|---|---|---|"]
    for k, s in e6["worst_margin_of_rejections"].items():
        lines.append(f"| {k} | {s['n']} | {s['min']:.1%} | {s['median']:.1%} | {s['max']:.1%} |")
    lines += ["", "B0 rejections re-judged under SL-PSR-2016 LV (+/-6 %): "
              + ", ".join(f"{k.replace('DECISION_', '')} {v}"
                          for k, v in sorted(e6["b0_rejections_also_rejected_by_sl_psr_2016"]
                                             .items())) + ".", "",
              "## Limits of this evaluation", "",
              "- Hourly demand is downscaled from monthly totals, so persistence misses are",
              "  smooth and nearly the same every day; real misses are larger (E5 shows what",
              "  wider bands change).",
              "- The grid model is the shared reference model with stated assumptions (lumped",
              "  feeders, 50/50 split, stiff source).",
              "- The action family is hand-designed; M3's own policy is not in the loop yet.", ""]
    return "\n".join(lines)


def _fmt(r: dict) -> str:
    if r["rate"] is None:
        return "n/a"
    return f"{r['rate']:.1%} ({r['k']}/{r['n']}, {r['ci95'][0]:.1%}-{r['ci95'][1]:.1%})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.run_eval")
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="12 cases, for a smoke test")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    results, latency = run(args.stride, args.quick)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_results.json").write_text(json.dumps(results, indent=1, sort_keys=True)
                                                + "\n")
    (args.out / "latency.json").write_text(json.dumps(latency, indent=1, sort_keys=True) + "\n")
    (args.out / "README.md").write_text(render(results, latency))
    print(f"wrote {args.out} ({results['cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
