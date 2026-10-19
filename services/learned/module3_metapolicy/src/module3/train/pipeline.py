"""End-to-end Meta-RL pipeline: train the gate over several seeds, test it, write a report.

    python -m module3.train.pipeline --config pipeline.yaml --out artifacts/pipeline/run1
    python -m module3.train.pipeline --quick                 # tiny smoke run (seconds)
    python -m module3.train.pipeline --live-m4 http://127.0.0.1:8000   # final test on real M4

What it does, per training seed:
    1. build a fresh gate (GRU by default) and train it with PPO on random cyclones
       (training grid = SurrogateTwin, fast);
    2. test it — and every baseline — on NEW random cyclones and on the held-out
       historical-style profiles (burevi_like, ditwah_like), none of which it trained on;
    3. optionally train a memoryless (MLP) copy as the 'does memory help?' ablation.
Then it aggregates mean +- std across seeds and, if a gateway URL is given, replays a small
set of storms on the REAL M4 firewall (the 'hybrid' final test) to measure the sim-to-real gap.

Outputs under --out: results.json (everything), report.md (the tables), gate_seed<N>.pt.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate
from module3.train.evaluate import BASELINES, evaluate_policy
from module3.train.ppo import PPOConfig, train_gate
from module3.train.warmstart import WarmStartConfig, behaviour_clone, collect_teacher_data
from module3.twin.live_m4 import LiveM4Twin

POLICIES = (*BASELINES, "gate_bc", "gate", "gate_mlp")
METRICS = ("return", "served_fraction", "reject_rate", "s2_rate", "deliberation_cost")


def _train(
    cfg: dict, arch: str, seed: int, log
) -> tuple[RecurrentGate, list[dict], RecurrentGate | None]:
    """Build a gate, optionally warm-start it from a rule, then train it with PPO.

    Returns (final gate, training log, snapshot taken right after the warm start or None).
    """
    pol = cfg["policy"]
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(seed))
    torch.manual_seed(seed)
    model = RecurrentGate(
        hidden=int(pol["hidden"]), arch=arch, feedback=bool(pol.get("feedback", True))
    )
    bc_snapshot = None
    ws = pol.get("warm_start")
    if ws:
        wcfg = WarmStartConfig.from_dict(ws if isinstance(ws, dict) else {})
        x, a = collect_teacher_data(env, wcfg, np.random.default_rng(seed + 12_345))
        stats = behaviour_clone(model, x, a, wcfg, seed)
        agree = stats["agreement_with_teacher"]
        log(f"  warm start: copies '{wcfg.teacher}' on {agree:.0%} of steps")
        bc_snapshot = deepcopy(model)
    hist = train_gate(
        env, model, PPOConfig.from_dict(cfg["ppo"]), int(pol.get("trials_per_task", 1)), seed, log
    )
    return model, hist, bc_snapshot


def _evaluate_all(
    cfg: dict,
    models: dict[str, RecurrentGate],
    seed: int,
    twin=None,
    n_random: int | None = None,
    n_profile: int | None = None,
    storm_offset: int = 0,
) -> dict:
    """Score every policy on one seed's storms: 'random' plus each held-out profile."""
    ev = cfg["eval"]
    env = CycloneGatingEnv(cfg, twin=twin, rng=np.random.default_rng(10_000 + seed))
    thr = float(ev["threshold_u"])
    n_random = int(n_random if n_random is not None else ev["episodes_random"])
    n_profile = int(n_profile if n_profile is not None else ev["episodes_per_profile"])
    out: dict = {}
    suites = {"random": (None, n_random), **{p: (p, n_profile) for p in ev["profiles"]}}
    for suite, (spec, n) in suites.items():
        storm_seeds = [
            storm_offset + 1_000_000 * (seed + 1) + 10_000 * list(suites).index(suite) + i
            for i in range(n)
        ]
        out[suite] = {}
        for pol in BASELINES:
            out[suite][pol] = evaluate_policy(env, pol, storm_seeds, spec, None, thr)
        for name, m in models.items():
            out[suite][name] = evaluate_policy(env, "gate", storm_seeds, spec, m, thr)
    return out


def _agg(per_seed: list[dict]) -> dict:
    """mean/std across seeds for every suite x policy x metric."""
    res: dict = {}
    for suite in per_seed[0]:
        res[suite] = {}
        for pol in per_seed[0][suite]:
            res[suite][pol] = {}
            for m in METRICS:
                v = np.array([s[suite][pol][m] for s in per_seed])
                res[suite][pol][m] = {
                    "mean": float(v.mean()),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                }
            res[suite][pol]["escalation_by_stage"] = {
                st: float(np.mean([s[suite][pol]["escalation_by_stage"][st] for s in per_seed]))
                for st in per_seed[0][suite][pol]["escalation_by_stage"]
            }
            res[suite][pol]["escalation_by_damage"] = {
                k: float(np.mean([x[suite][pol]["escalation_by_damage"][k] for x in per_seed]))
                for k in ("no_trip", "trip")
            }
            res[suite][pol]["monotonic_in_all_seeds"] = all(
                s[suite][pol]["monotonic_nondecreasing"] for s in per_seed
            )
    return res


def _report(agg: dict, n_seeds: int, live: dict | None, arch: str) -> str:
    lines = [f"# Meta-RL pipeline report  ({n_seeds} seeds, gate arch = {arch})", ""]
    lines.append(
        "All numbers are mean +/- std across independent training seeds. "
        "Every policy is scored on identical storms. `return` is per-storm "
        "(80 steps); higher is better.\n"
    )
    for suite, pols in agg.items():
        title = (
            "random cyclones (fresh, unseen)" if suite == "random" else f"held-out profile: {suite}"
        )
        lines += [
            f"## {title}",
            "",
            "| policy | return | served | reject rate | S2 rate | delib. cost "
            "| S2 rate if feeder out | S2 rate if grid intact |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pol in POLICIES:
            if pol not in pols:
                continue
            r = pols[pol]

            def f(m, d=3, r=r):
                return f"{r[m]['mean']:.{d}f} +/- {r[m]['std']:.{d}f}"

            dmg = r["escalation_by_damage"]
            lines.append(
                f"| {pol} | {f('return', 2)} | {f('served_fraction')} | {f('reject_rate')} "
                f"| {f('s2_rate')} | {f('deliberation_cost', 2)} "
                f"| {dmg['trip']:.2f} | {dmg['no_trip']:.2f} |"
            )
        lines.append("")
    if live:
        lines += [
            "## Final test on the REAL M4 firewall (seed-0 gate)",
            "",
            "| suite | policy | return (live) | return (surrogate, same storms) | gap |",
            "|---|---|---|---|---|",
        ]
        for suite, pols in live.items():
            for pol, v in pols.items():
                gap = v["live"] - v["surrogate"]
                lines.append(
                    f"| {suite} | {pol} | {v['live']:.2f} | {v['surrogate']:.2f} | {gap:+.2f} |"
                )
        lines.append("")
    return "\n".join(lines)


def run(cfg: dict, out: Path, live_url: str | None, quick: bool, log=print) -> dict:
    meta = cfg["policy"].get("meta_method", "rl2")
    if meta != "rl2":
        raise NotImplementedError(
            f"policy.meta_method={meta!r} is reserved but not implemented. MAML would need an "
            "inner-loop gradient adaptation step per storm; see docs/module3.md "
            "('Why RL^2 and not MAML') for when it is worth adding."
        )
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(cfg["eval"]["seeds"])
    ablate = bool(cfg["eval"].get("ablate_mlp", False))
    arch = cfg["policy"]["arch"]
    per_seed, histories = [], {}
    models0: dict[str, RecurrentGate] = {}
    t0 = time.time()
    for seed in seeds:
        log(f"\n=== seed {seed}: training {arch} gate ===")
        gate, hist, bc = _train(cfg, arch, seed, log)
        models = {"gate": gate}
        if bc is not None:
            models = {"gate_bc": bc, **models}  # warm-start-only snapshot, for honest comparison
        histories[f"gate_seed{seed}"] = hist
        torch.save(
            {"state_dict": gate.state_dict(), "arch": arch, "hidden": gate.hidden, "config": cfg},
            out / f"gate_seed{seed}.pt",
        )
        if ablate:
            log(f"=== seed {seed}: training memoryless (mlp) ablation ===")
            mlp, mh, _ = _train(cfg, "mlp", seed, log)
            models["gate_mlp"] = mlp
            histories[f"gate_mlp_seed{seed}"] = mh
        per_seed.append(_evaluate_all(cfg, models, seed))
        if seed == seeds[0]:
            models0 = models
        log(f"  seed {seed} done  ({time.time() - t0:.0f}s elapsed)")

    agg = _agg(per_seed)

    live = None
    if live_url:
        log(f"\n=== final test on real M4 at {live_url} (seed {seeds[0]} gate) ===")
        n_live = int(cfg["eval"]["live_episodes"])
        # The real M4 cannot represent a lost main supply (finding F3), so the live test
        # replays only storms WITHOUT grid loss; both twins see the same storms.
        live_cfg = deepcopy(cfg)
        live_cfg.setdefault("scenario", {})["p_grid_loss"] = 0.0
        sur = _evaluate_all(live_cfg, models0, seeds[0], None, n_live, n_live)
        liv = _evaluate_all(
            live_cfg,
            models0,
            seeds[0],
            LiveM4Twin(live_url, tier_weights=cfg.get("tier_weights")),
            n_live,
            n_live,
        )
        live = {
            s: {p: {"live": liv[s][p]["return"], "surrogate": sur[s][p]["return"]} for p in liv[s]}
            for s in liv
        }

    results = {
        "config": cfg,
        "seeds": seeds,
        "aggregate": agg,
        "per_seed": per_seed,
        "live_m4": live,
        "training_history": histories,
        "wall_clock_s": time.time() - t0,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = _report(agg, len(seeds), live, arch)
    (out / "report.md").write_text(report, encoding="utf-8")
    log("\n" + report)
    log(f"\nwrote {out / 'results.json'} and {out / 'report.md'}")
    return results


def _quick(cfg: dict) -> dict:
    c = deepcopy(cfg)
    c["ppo"].update(updates=6, tasks_per_update=8, epochs=2)
    c["eval"].update(seeds=[0, 1], episodes_random=6, episodes_per_profile=4, live_episodes=2)
    return c


def main() -> int:
    here = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(here / "pipeline.yaml"))
    ap.add_argument("--out", default=str(here / "artifacts" / "pipeline" / "latest"))
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--seeds", type=int, nargs="*", help="override the seed list")
    ap.add_argument("--arch", choices=["gru", "mlp"], help="override policy.arch")
    ap.add_argument("--ablate-mlp", action="store_true", help="also train the memoryless ablation")
    ap.add_argument(
        "--p-grid-loss",
        type=float,
        help="share of storms that cut the main supply (islanded operation); default from config",
    )
    ap.add_argument("--live-m4", default=None, help="gateway URL for the final real-M4 test")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.quick:
        cfg = _quick(cfg)
    if a.seeds:
        cfg["eval"]["seeds"] = a.seeds
    if a.p_grid_loss is not None:
        cfg.setdefault("scenario", {})["p_grid_loss"] = a.p_grid_loss
    if a.arch:
        cfg["policy"]["arch"] = a.arch
    if a.ablate_mlp:
        cfg["eval"]["ablate_mlp"] = True
    run(cfg, Path(a.out), a.live_m4 or cfg["eval"].get("live_m4_url"), a.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
