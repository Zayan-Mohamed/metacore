# M4 risk-certified verification: the method

How Module 4 decides whether an action M3 proposes may run. The decision is in
[ADR 0005](../adr/0005-risk-certified-verification.md); this page says what each part does, what
is calibrated from data and what is fixed by design, how to rebuild it, and which test or report
backs each claim. All code is in `deterministic-core/module4_verification/src/verification/`.

## The pipeline

```
OFFLINE, once per data release                     RUNTIME, every verify call
python -m verification.calibrate build             RiskCertifiedVerifier.verify(action, snapshot)
                                                    
island_load_hourly.csv ──> persistence misses       Check A: action on the recorded hour
NASA POWER ──> hazard level                            (OpenDSS + limit profile + adequacy)
  -> situation groups -> conformal bands              REJECT / UNVERIFIED here is final
  -> artifacts/calibration_v1.json (sealed)          Check B: 688 seeded scenarios, next 6 h
                                                       band for this situation (+ PID in storms)
                                                       each scenario solved at peak and trough
                                                     Clopper-Pearson on k of N -> verdict
                                                     Shapley attribution over 5 feeder players
                                                     grounded operator sentence
```

## 1. Check A: the action on the recorded hour

| Part | What it does | Code | Backed by |
|---|---|---|---|
| Shared model | `northern-islands-v1` from PR #13: 4 islanded 400 V diesel microgrids, 13 buses, 9 lines. M4 ships a byte-identical copy of the model and pin | `opendss/circuit.py`, `opendss/models/` | `test_northern_islands_v1.py::test_vendored_model_and_pin_match_the_contracts_copy`, golden Ditwah solution |
| Exact names | Shedding and dispatch match a bus, load or unit by exact name; a bus-level dispatch is split over its in-service units; dispatching an out-of-service unit is malformed | `opendss/circuit.py` | `test_firewall_violations.py`, `test_generation_adequacy.py::test_dispatching_an_out_of_service_unit_is_malformed` |
| Generation adequacy | Island apparent demand (plus charging batteries) must stay within the lead genset's rating plus in-service units' kVA. The stiff source would otherwise hide an overloaded station | `firewall/limits.py` | `test_generation_adequacy.py` (Nainativu peak with standbys out: 0.955 pu looks fine, 460 kVA > 380 kVA) |
| Limit profiles | METACORE-STRICT 0.95-1.05 pu (default), SL-PSR-2016 LV +/-6 %, EN 50160 +/-10 %, ANSI C84.1 Range A +/-5 %; the verdict records the profile | `firewall/profiles/*.json` | `test_profiles.py` |
| Fail-safe | UNVERIFIED for a malformed snapshot, non-convergence, a non-finite number, a solver exception or a blown time budget. One approval path | `firewall/verifier.py` | `test_unverified.py` (fault injection: no NaN anywhere can approve) |

## 2. Calibration: what is learned from data, and how it is frozen

Nothing is fitted by optimisation. The calibration counts and sorts.

| Step | Definition | Code | Backed by |
|---|---|---|---|
| Persistence miss | For each start hour t and island i: `r[i,h] = load[i,t+h] / load[i,t] - 1`, h = 1..6; a row keeps `up = max_h r` and `down = min_h r` for all islands together | `calibrate/misses.py` | `test_calibration_build.py` (hand-made CSV, no data after the window) |
| Splits | Calibration Jan 2024 - Jun 2025 (13,122 rows); validation Jul - Sep 2025; test Oct - Dec 2025 (Ditwah, hybrid decay) | `calibrate/build.py` | `test_calibration_build.py::test_real_calibration_split_row_count` |
| Hazard level | NASA POWER 50 m wind and rain per island, each scaled between its calibration median and 99.9th percentile, combined as `1-(1-hw)(1-hr)`, worst island, cut at 0.4 / 0.7 / 0.9 into NORMAL / ELEVATED / SEVERE / EXTREME | `risk/situation.py`, `risk/history.py` | `test_pid.py::test_ditwah_is_storm_mode` (28 Nov 2025 is SEVERE+) |
| Situation groups | Start hour +/-1 h, season (NE monsoon Oct-Jan, inter-monsoon Feb-Apr, SW monsoon May-Sep), hazard CALM/STORM; a group under 100 rows falls back to hour+season, hour, all | `calibrate/groups.py`, `risk/situation.py` | `test_bands.py::test_lookup_falls_back_when_a_group_is_thin` |
| Conformal band | Per group and per edge, the `ceil((n+1)(1-a))`-th smallest score with a = alpha/2 = 2.5 %, so the two-sided band covers at least 95 % | `risk/bands.py` | `test_bands.py::test_textbook_quantile_on_known_misses`, held-out coverage test |
| Freezing | Input hashes, splits, seed 20251128, hazard bounds, every group's band and every calibration row, sealed by `artifact_sha256` | `calibrate/build.py`, `risk/calibration.py` | `test_calibration_build.py` (byte-identical rebuild; a changed input or a hand edit is refused) |

**Held-out coverage.** On Jul - Sep 2025 the static bands cover 98.4 % of windows against a 95 %
target (artifact `validation` block; [E2](../../reports/data-analysis-report/m4/README.md)): they
over-cover overall, while a few SW-monsoon hours under-cover because the 2025 daily profile
drifted from 2024. Conformal PID steers the long-run miss rate to the target (95.0 % on both
validation and test), which mostly means *narrowing* the over-wide bands; on validation storm
hours it covered 91.5 % against 94.7 % static. With this smooth, downscaled demand PID changed no
verdict in the evaluation (B3 and B4 agree in every E1 row).

## 3. Check B: the certificate

| Part | Definition | Code | Backed by |
|---|---|---|---|
| Scenarios | 688 whole six-hour windows drawn with replacement from the situation's group (block bootstrap, block = horizon), `numpy.random.Generator(PCG64(seed))`, clipped to the band. Each scenario is solved twice: every island at its window peak, and at its trough. Two corner rows (all islands at the band's top, all at its bottom) are reported, not counted | `risk/scenarios.py`, `risk/certifier.py` | `test_scenarios.py` (same seed, same rows; all inside the band; island correlation kept) |
| Peak and trough | Exact when a violation is monotone in each island's demand over the window: feeder drop and current grow with load, a battery exporting into the feeder over-volts as load falls. An assumption of the method | `risk/scenarios.py` docstring | stated |
| Fast path | Snapshot and action are applied once; scenarios only set loads in place (no recompile); adequacy is computed from known demand | `powerflow/solver.py`, `risk/certifier.py` | `test_certifier.py::test_latency_of_688_scenarios` (about 130 ms) |
| Conformal PID | Offset on the band (it can narrow as well as widen): P `+= 0.02 (err - alpha)`, I `= 0.005 x clip(sum of last 72 (err - alpha), +/-4)`, D a fixed scorecaster per hazard level (0 / 2 / 8 / 12 points). Replayed causally over the 14 days before "now". Applied in storm mode (SEVERE or worse now or within 24 h); misses widen scenarios in proportion | `risk/pid.py` | `test_pid.py` (long-run miss rate returns to alpha after a shift; causal; closer to target than static on Oct - Dec 2025) |
| Verdict | One-sided Clopper-Pearson at 99.9 %: APPROVE if upper <= 1 %, REJECT if lower > 1 %, else UNVERIFIED. Batches of 100; early stop only to REJECT, at delta / 7 (Bonferroni); approval judged once at N = 688 | `risk/statistics.py` | `test_statistics.py` (published zero-failure sizes 688 and 59; the guide's four worked rows) |
| Certificate | N, k, bounds, eps, confidence, profile, calibration version and hash, seed, situation, band, PID state, corners, failures per island; no timestamps or latency, so it replays byte for byte | `types.py::RiskCertificate` | `test_determinism_replay.py` |

## 4. Explanation

| Part | Definition | Code | Backed by |
|---|---|---|---|
| Shapley risk attribution | Players: Nainativu F1, Nainativu F2, Delft-Neduntivu, Analaitivu, Eluvaitivu (built from the pin's feeder groups). `v(S)` is the share of the evaluated scenarios that fail when only S moves. Islands are separate grids, so all 32 coalitions come from the Check B solves plus two extra per scenario (Nainativu F1 alone, F2 alone). Exact Shapley values; shares sum to k/N | `risk/attribution.py` | `test_attribution.py` (efficiency; a planted single cause takes 100 %; same seed, same shares) |
| Abductive attribution | A violation is blamed on the action component that touches the same element, else the same feeder chain, else the same island, using the pin | `module4_semantic_translation/.../abductive/attribution.py` | `test_grounding.py` |
| Risk sentence | Built from the certificate and the top Shapley share, e.g. "13.8 % of likely next-6-hour conditions break a limit ...; 13.8 points come from Eluvaitivu load uncertainty" | `.../templates/causal_logger.py` | `test_grounding.py::test_the_risk_sentence` |
| Grounding check | Every claimed entity and every id-shaped token in the text must exist in the pin (or be quoted from a malformed action) | `.../templates/causal_logger.py::GroundingValidator` | `test_grounding.py::test_a_planted_fake_name_is_rejected` |

Note on the players: the load history is island-level and the model splits it 50/50, so Nainativu
F1 and F2 always move by the same multiplier; their split is about each feeder's share of that
move, not about different forecasts.

## 5. Contract and surfaces

- `verification.proto` v1.1, additive only: `DECISION_UNVERIFIED = 3`,
  `VIOLATION_TYPE_GENERATION_INADEQUATE = 6`, the snapshot and profile on the request, the
  certificate, attribution, profile and reasons on the verdict. A v1.0 reader parses a v1.1
  verdict (`packages/contracts/python/tests/test_verification_v1_1.py`); `buf breaking` against
  upstream main passes.
- Gateway: `POST /api/module4/verify` (`mode` certified or plain, `pid`, `profile`, `hour`,
  `outages`), `GET /api/module4/presets|hours|profiles` (`services/gateway/tests/test_module4_router.py`).
- Dashboard `/verification`: the risk certificate card (interval against eps, situation, band,
  PID, corners, replay ids), Shapley bars, grounded explanation, adequacy, voltages and currents
  (`apps/dashboard/e2e/verification.e2e.ts`, Playwright).

## 6. How to rebuild and check

```bash
# from deterministic-core/module4_verification
uv run --isolated python -m verification.calibrate build    # rewrite the artifact
uv run --isolated python -m verification.calibrate check    # byte-identical rebuild, or exit 1
uv run --isolated --with pytest pytest -q                    # core tests (incl. the check above)
uv run --no-sync python -m evaluation.run_eval               # E1-E6 -> reports/data-analysis-report/m4/
```

Rebuild after any change to `data/processed/island_load_hourly.csv` or `data/raw/nasa_power/`.
Until then the gateway's verifier refuses the stale artifact and answers UNVERIFIED.

## 7. What is a design choice, not a measurement

| Choice | Value | Alternative |
|---|---|---|
| Risk target, confidence | eps = 1 %, 99.9 % (N = 688) | 5 % at 95 % (N = 59) |
| Horizon | 6 h, M1's horizon | 1 h |
| Band coverage | 95 % (2.5 % per edge) | 90 % |
| Minimum group size | 100 rows | larger groups, fewer situations |
| PID gains, scorecaster | eta 0.02, k_i 0.005 over 72 h, 0 / 2 / 8 / 12 points | tuned on validation |
| Players | 5 feeders | 4 islands |
| Default profile | METACORE-STRICT | SL-PSR-2016 LV |

## Limitations

1. Demand is constructed from monthly totals, so misses are smooth and repeat almost exactly day
   to day; real misses are larger. E5 reports what wider bands change.
2. The grid model is the team's reference model with the pin's stated assumptions (lumped feeders,
   50/50 split, stiff lead genset, power factor 0.85).
3. M4's guess is persistence; M1 forecasts vulnerability, not demand.
4. The evaluation's action family is hand-designed; M3's policy is not yet in the loop, and M3's
   committed samples still use the retired bus names.
