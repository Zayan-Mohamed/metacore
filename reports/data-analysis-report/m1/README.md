# Module 1 forecaster: results

Held-out test quarter: Oct-Dec 2025 (Cyclone Ditwah, Eluvaitivu battery decay). Metrics files:
`forecast_v0_metrics.json`, `forecast_v0_1_metrics.json`. Regenerate with `python -m module1.train.run`; `--rescore` recomputes the test metrics from the
saved predictions without retraining.

## The claim, stated carefully

**ST-GNN v0.1 is a residual correction on persistence.** Of its gain over a conventional 1-D
LSTM, the residual head (predict the change from the current score) accounts for about 67% under
the equal weighting and 72% under stress-heavy; under hazard-heavy the split is roughly even (45%
residual head, 55% fused inputs and graph). Against a 1-D LSTM given the same residual head, ST-GNN v0.1 still has the lowest mean
error in all three weightings and catches more high-vulnerability onsets early, but it does not
win at every horizon. The graph helps only under the stress-heavy weighting, and weather inputs
do not help at all, which is consistent with NASA POWER giving three of the four islands
identical weather.

## Mean test RMSE, t+1..t+6 (lower is better)

| Model | Equal | Hazard-heavy | Stress-heavy |
|---|---|---|---|
| Persistence | 0.0377 | 0.0422 | 0.0474 |
| 1-D LSTM | 0.0342 | 0.0421 | 0.0359 |
| 1-D LSTM + residual head | 0.0287 | 0.0399 | 0.0238 |
| ST-GNN v0 | 0.0398 | 0.0481 | 0.0292 |
| **ST-GNN v0.1** | 0.0261 | 0.0373 | 0.0190 |
| Graph-free (v0.1) | 0.0262 | 0.0353 | 0.0230 |
| Time-free (v0.1) | 0.0273 | 0.0376 | 0.0208 |
| No-weather (v0.1) | 0.0259 | 0.0351 | 0.0192 |

## Where the gain comes from (error reduction, positive = helps)

| Weighting | Residual head (LSTM -> LSTM+res) | Fused inputs + graph (LSTM+res -> ST-GNN) | Graph | Temporal encoder | Weather |
|---|---|---|---|---|---|
| Equal | +16.0% | +9.2% | +0.5% | +4.6% | -0.8% |
| Hazard-heavy | +5.1% | +6.5% | -5.7% | +0.8% | -6.4% |
| Stress-heavy | +33.7% | +20.1% | +17.3% | +8.7% | +1.2% |

## Onset recall (share of high-vulnerability onsets forecast ahead of time)

| Weighting | Persistence | 1-D LSTM | 1-D LSTM + residual | ST-GNN v0.1 |
|---|---|---|---|---|
| Equal | 21% | 25% | 45% | 53% |
| Hazard-heavy | 14% | 12% | 15% | 28% |
| Stress-heavy | 14% | 22% | 75% | 85% |

## Mean absolute scaled error (MASE), t+1..t+6

MAE divided by the one-step naive forecast's MAE on the training split (Hyndman and Koehler,
2006); below 1 beats that naive forecast. Averaged over six horizons, every model is above 1,
persistence included (about 3.1): a six-hour forecast is harder than a one-hour step, and the test
quarter is rougher than training.

| Model | Equal | Hazard-heavy | Stress-heavy |
|---|---|---|---|
| Persistence | 3.14 | 3.13 | 3.17 |
| 1-D LSTM | 2.86 | 3.07 | 2.31 |
| 1-D LSTM + residual head | 2.16 | 2.99 | 1.39 |
| **ST-GNN v0.1** | 1.94 | 2.45 | 1.14 |
| ST-GNN v0.1, t+1 only | 0.90 | 1.09 | 0.60 |

## Normal hours, storm hours and Cyclone Ditwah (RMSE)

Regimes are keyed on the first forecast hour. Storm: any node's hazard above 0.3 (609 of 2,179
test origins). Ditwah: 25-30 Nov 2025 (144 origins, 139 of them storm hours); its hazard reaches
1.0 against a training 99th percentile of 0.86, so training never saw an event this strong.

| Weighting | Model | Normal | Storm | Ditwah |
|---|---|---|---|---|
| Equal | Persistence | 0.0344 | 0.0505 | 0.0376 |
| | 1-D LSTM + residual | 0.0215 | 0.0448 | 0.0370 |
| | **ST-GNN v0.1** | 0.0197 | 0.0411 | 0.0320 |
| Hazard-heavy | Persistence | 0.0332 | 0.0652 | 0.0454 |
| | 1-D LSTM + residual | 0.0310 | 0.0616 | 0.0504 |
| | **ST-GNN v0.1** | 0.0243 | 0.0626 | 0.0490 |
| Stress-heavy | Persistence | 0.0476 | 0.0551 | 0.0460 |
| | 1-D LSTM + residual | 0.0176 | 0.0369 | 0.0317 |
| | **ST-GNN v0.1** | 0.0145 | 0.0295 | 0.0254 |

## Is the difference real? (Diebold-Mariano, ST-GNN v0.1 vs 1-D LSTM + residual)

Two-sided test on the per-origin squared-error difference, Newey-West variance, Harvey-Leybourne-
Newbold correction. Negative means ST-GNN is better. The conclusions hold with a 24 h or 48 h lag.

| Weighting | All test hours | Normal | Storm | Ditwah |
|---|---|---|---|---|
| Equal | -4.76 (p < 0.001) | -3.03 (p = 0.002) | -4.01 (p < 0.001) | -3.68 (p < 0.001) |
| Hazard-heavy | -2.60 (p = 0.009) | -9.50 (p < 0.001) | +0.44 (p = 0.66) | -0.82 (p = 0.41) |
| Stress-heavy | -6.24 (p < 0.001) | -4.37 (p < 0.001) | -5.35 (p < 0.001) | -3.08 (p = 0.002) |

Under hazard-heavy, ST-GNN's advantage is significant only in normal hours; in storms and in
Ditwah it is no better than the residual LSTM, and in Ditwah it does not beat persistence either
(0.0490 vs 0.0454; DM +1.43, p = 0.15). The graph-free and no-weather ablations do better there
(0.0452, 0.0448).

## Caveats

- **The target is constructed** (hazard x fragility x stress). No dataset observes asset
  vulnerability; absolute errors are against this proxy.
- **The stress term uses the downscaled load**, which is itself built from temperature, so part of
  the stress-heavy gain is the model recovering a constructed diurnal shape. Hazard-heavy is the
  conservative weighting.
- **v0.1's two changes were chosen after reading v0's test results.** Both versions are reported;
  future revisions are diagnosed on validation only.
- **The battery decay is an input, not a forecast.** It is held out of training but reaches the
  model through last month's fragility, so in December the model carries a known step forward.
- **Not every horizon:** against the residual LSTM, ST-GNN v0.1 is +0.2% at t+6 (equal) and
  +1.6% at t+1 (hazard-heavy), and slightly worse in hazard-heavy storm hours (0.0626 vs 0.0616).
- **Ditwah is one event of six days.** Its numbers show behaviour under an unseen cyclone; they
  are not a population estimate. Hourly errors are autocorrelated, which the DM test accounts for
  but a plain RMSE comparison does not.
