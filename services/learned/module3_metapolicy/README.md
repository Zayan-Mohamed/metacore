# Module 3 — Cost-Aware Metacognitive Gating (Saabir S., J26-DS-317)

A learned **gate** that, at every step of a cyclone, chooses between

- **System 1** — a cheap reactive rule (breakers, priority-tier load shedding), and
- **System 2** — the slow deliberative planner (survival-path dispatch),

under a budget of System 2 calls per storm (default 45 of 80 steps).

The gate is a **PPO-trained recurrent policy (GRU)**. It is not called Meta-RL any more:
the repeated-trials test showed no within-storm adaptation. Full design, numbers and
the reasoning behind each choice: `../../../docs/module3.md`.

Package: `src/module3/` (installed as `module3`). Everything is driven by `pipeline.yaml`.

## Layout

| Folder | What lives there |
|---|---|
| `scenario/` | wind-driven cyclone generator (`cyclone.py`), M2 stream replay + jitter (`m2_stream.py`) |
| `env/` | `CycloneGatingEnv` — Gymnasium env, **15-d observation**, `Discrete(2)` action; `reward.py` |
| `control/` | System 1 (`reactive.py`), System 2 (`planner.py`), node tiers (`priority.py`) |
| `twin/` | grid twin that judges a plan: `SurrogateTwin` (fast, training) and `LiveM4Twin` (real M4 over HTTP, final test) |
| `policy/` | `RecurrentGate` (GRU) |
| `train/` | `ppo.py`, `warmstart.py`, `evaluate.py` (baselines), `pipeline.py` (seeds → train → test → report) |
| `serve/` | `demo.py` / `step.py` (dashboard `/gating`), `storm.py` (dashboard `/cyclone`), `guard.py` |
| `scripts/` | `demo_run.py`, `step_run.py`, `storm_run.py` (called by the gateway) and the experiment scripts |

Observation: slots 0–11 are M2 / M1 / budget (`u`, `trigger_reason`, state class, severity,
time-to-hazard, budget left, observed fraction, …); 12–14 are SCADA (`tripped_visible`,
`gen_available_visible`, `grid_lost_visible`).

Reward per step: `w_serve·served − w_viol·violating_fraction − deliberation_cost(S2) − w_latency·lat/budget`.
The latency values (2 / 18) are cost weights, not measurements.

## Run

From the repo root:

```bash
# full pipeline: seeds -> train -> test on random + held-out storms -> results.json + report.md
uv run python -m module3.train.pipeline

# tests (needs torch)
uv run pytest services/learned/module3_metapolicy/tests -q
```

Baselines in every report: `always_s1`, `always_s2`, `threshold` (u > cutoff), `breaker_rule`,
`oracle`, and the trained `gate`.

## Run — from the dashboard

The gateway (`services/gateway/gateway/routers/module3.py`) shells out to the scripts above:

- `POST /api/module3/run` → `scripts/demo_run.py` (quick train + evaluate, ~15–150 s)
- `POST /api/module3/generate`, `/process` → `scripts/step_run.py` (one decision at a time)
- storm routes → `scripts/storm_run.py` (the `/cyclone` page)

```bash
uv sync --all-packages
uv run uvicorn gateway.main:app --app-dir services/gateway --reload --port 8000
cd apps/dashboard && pnpm install && pnpm dev     # /gating and /cyclone
```

The gateway process needs `torch`, `gymnasium`, `numpy`, `pyyaml` importable — true for
`uv run` above, not for the slim gateway docker image.

The verdict shown on `/gating` comes from the **surrogate twin**, not the real M4.

## Cause-aware gating (`m2-out/0.3`)

| `trigger_reason` | Meaning | Gate response |
|---|---|---|
| `value` | unseen conditions (cyclone) | escalate to System 2 |
| `sensing` | missing / blackout data | stay on System 1 — don't deliberate on absent data |
| `both` | cyclone + comms loss | escalate |
| `none` | nominal | System 1 unless severity demands otherwise |

## M2 and M4 contracts

- Input: the published mock `sample_m2_to_m3.jsonl` (`m2-out/0.3`). Keep it synced with
  `../module2_auq_engine/sample_m2_to_m3.jsonl`; a sibling copy is used as a fallback.
- Output to M4: `M3_TO_M4_CONTRACT.md` and `sample_m3_to_m4.jsonl`.
- Budget exhaustion is not an agent action: at 0 budget the env forces System 1.

## Not built

MAML (documented only); ONNX export + Go serving (`export/` is a stub).
