# Module 3 on the live system: what has to change (for Saabir)

**Status:** open · **Owner:** Saabir (M3) · **Raised by:** Zayan (M1), 24 Oct 2026

The live pipeline (ADR 0006, `services/orchestrator/app/pipeline.py`) runs M1 → M2 → M4 every
hour on today's weather and shows it in the control room. M3 is shown as **not connected**.
It can't be wired in without changes inside Module 3, and those are yours to make. This is the
list.

## Why it can't run on today's grid as it is

1. **Every name is from the retired `delft-3island` network.** Actions use N1..N12, `Gen_*`, and
   tie lines `Line_1_2`, `Line_2_3`, `E_crit_1` that do not exist on the real islands. M4 runs
   `northern-islands-v1` and rejects these actions as `MALFORMED_ACTION` (pinned by
   `deterministic-core/module4_verification/tests/test_m3_integration.py`).
2. **No entrypoint takes one live situation.** `serve/step.py` and `serve/storm.py` only roll out
   simulated cyclone episodes. Nothing accepts an M2 message plus an M1 state and returns a
   `ProposedControlAction` and a `GatingDecision`.
3. **The gate's observation needs inputs that don't exist live.** Slots 12–14 of the 15-d
   observation (`env/cyclone_env.py`: `tripped_visible`, `gen_available_visible`,
   `grid_lost_visible`) assume SCADA. The islands have none. Slots 7–9 assume the delft
   topology's overhead lines and `time_to_hazard`, which the live M1 does not produce.
4. **The trained gates are delft-trained.** `artifacts/pipeline/final_islanded/gate_seed*.pt`
   learned on the delft surrogate twin and the mock M2 stream `sample_m2_to_m3.jsonl`.

## Where the delft assumptions live

| File | What to change |
|---|---|
| `module3/twin/topology.py` | `SOURCE_BUS`, `NODES`, `LOAD_KW`, `GENERATORS`, `PRIORITY_TIER`, `LINES`, `OVERHEAD_LINES`, `OVERHEAD_END_NODE`, `NODE_EXPOSURE`. Rebuild from the pin `metacore_contracts/schema/module1_topology_v1.json`. |
| `module3/control/priority.py` | `PRIORITY` keyed on N1..N12. Use the feeder groups: `*_F1` (critical, assumed), `*_F2`. |
| `module3/control/reactive.py` | default `top_at_risk_nodes = ["N12","N11","N10"]` (N10 never existed). |
| `module3/control/planner.py` | shed and dispatch targets. On the real islands: shed per feeder group, start standby gensets (`*_DG2`, `*_DG3`), dispatch `ELU_HYB_BESS` / `ELU_HYB_DG`. No breaker can join islands. |
| `twin/surrogate.py`, `twin/scoring.py`, `env/reward.py`, `env/cyclone_env.py`, `scenario/cyclone.py` | all import the delft topology. |
| `twin/live_m4.py` | delft breaker and dispatch overlays; maps `buses[].bus_name` to delft `NODES`, so `energized` is always empty against M4 today. |
| `sample_m3_to_m4.jsonl` | uses N8/N9/N11/N12 and `E_crit_1`. |

## What the live pipeline already gives you each hour

Stored in the live database (`forecasts`, `uncertainty`, `grid_checks`). It's also served by
`GET /api/live/latest`.

- **M1:** `nodes[]` with the current score, t+1..t+6 forecasts from ST-GNN, the residual LSTM
  and persistence, `components_now` (hazard, fragility, stress) and `above_tau_t1`. Per island:
  the last 24 h and next 6 h of wind, rain, temperature and load.
- **M2:** per node, the `M2Output` contract message: `epistemic_uncertainty`,
  `competence_drop`, `trigger_reason`, `state_class`, `class_probabilities`,
  `observed_fraction`. The trigger runs with your config's two-hour hysteresis.
- **M4:** for now and each forecast hour, the verdict on "no action", the risk certificate,
  bus voltages, feeder loadings and station adequacy, on `northern-islands-v1`.

## Smallest change that connects M3

A function `decide(m1: dict, m2: dict, previous: dict | None) -> {plan, decision}` in
`module3/serve/`. It would emit northern-islands-v1 actions and a `GatingDecision`, and carry the
GRU state in `previous`. The orchestrator would run it as `app/m3_runner.py` between M2 and M4,
and M4 would then verify the plan rather than "no action". Where SCADA inputs are missing, the
gate should be told so (quality-masked) rather than given zeros.
