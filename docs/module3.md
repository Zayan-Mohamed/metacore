# Module 3 — Cost-Aware Metacognitive Gating & Meta-Policy

**Owner:** Saabir S. · **Code:** `services/learned/module3_metapolicy/` · **Contract:** `packages/contracts/proto/module3.proto`

This is the single reference document for Module 3. If something about M3 changes
(formulation, reward, observation, contract, status), change it here. The code and this file
are the sources of truth; `docs/module3-explained.html` is a generated viva explainer and
`services/learned/module3_metapolicy/README.md` is the run guide.

> Last verified against code: 2026-10-15, branch `m3/gate-policy`. Part I (§1–§13) describes the
> Week-7 toy; **Part II (§14–§22) describes the Meta-RL pipeline** (PPO + GRU / RL²).
> Numbers marked *(README, not re-run)* were copied from the module README and have not been
> re-measured for this document.

---

## 1. Purpose

M3 is the metacognitive gate. Every control step it decides whether the microgrid is handled by

- **System 1** — a cheap, fast, rule-based reactive path, or
- **System 2** — a deliberative survival-optimisation path that costs budget and time.

The gate must pay for deliberation only when it buys survival: escalate more as hazard severity
rises, and stay on System 1 when thinking harder cannot help (missing data, calm conditions).
Whatever path is chosen, its proposed action goes to the M4 physics firewall; nothing reaches
actuation without an APPROVE.

```
M1 (context) ─┐
              ├─► M3 gate ─► System 1 / System 2 ─► ProposedControlAction ─► M4 firewall ─► actuation
M2 (u, cause) ─┘                  └─► GatingDecision (audit/dashboard)
```

M3 is learned (Meta-RL). M4 is deterministic and has no ML dependencies; they meet only through
the protobuf contract.

---

## 2. Formal problem definition

### 2.1 Choice of formalism: MDP (SMDP as documented extension)

The gate is formulated as a **finite-horizon, fixed-step MDP** with a hard per-episode
deliberation budget carried in the state.

Why an MDP and not an SMDP:

- A control step is on the scale of minutes (`time_to_hazard` is expressed in minutes and
  clipped at ±30). The stand-in latencies are 2 ms for System 1 and 18 ms for System 2
  (`run_demo.py`), which is negligible against the step. The hazard does not evolve
  meaningfully while System 2 "thinks", so action duration need not be modelled.
- Deliberation cost is a scalar reward term plus a budget, which is the natural MDP form
  (deliberation-cost framing, Harb 2018).
- `budget_remaining` is part of the observation, so the process stays Markov without a
  constrained-MDP formulation.

**When this assumption breaks → SMDP.** If the real System 2 runs an optimiser or OpenDSS in
the loop and takes seconds to minutes, System 2 becomes a temporally extended action whose
duration τ matters ("Finding the Time to Think"). The model then becomes an SMDP: System 2
carries a duration, discounting is γ^τ, and the reward integrates damage accrued during τ.
The latency-and-thermal profiling task (§10) provides the S2 duration against the control
period; switch formalism if it is a meaningful fraction of it.

### 2.2 Definition

| Element | Definition |
|---|---|
| Horizon | Fixed length `env.episode_len` = 80 steps; episode ends by truncation, never termination |
| Discount | Undiscounted (γ = 1) |
| State / observation | 12-d vector, §3 |
| Action | `Discrete(2)`: 0 = System 1, 1 = System 2 (request) |
| Effective action | Forced to System 1 whenever `budget_remaining <= 0` (§4) |
| Reward | survival benefit − deliberation cost − M4 reject penalty, §5 |
| Constraint | `budget_per_episode` = 45 System 2 calls per episode (of 80 steps) |
| Transitions | Scenario-driven, see §2.3 |

### 2.3 Properties worth stating in the paper

- **Dynamics are exogenous.** The M1 context and M2 observation at step t+1 do not depend on the
  action at step t. The only state component the action changes is `budget_remaining`. Formally
  this is a budget-constrained contextual decision problem with sequential coupling through
  the budget; it is still correctly an MDP, but the claim should not be oversold as long-horizon
  control of the grid.
- **Partial observability is collapsed.** `u` and `observed_fraction` are noisy views of the
  true grid state. M2's output is treated as the observed state (standard simplification;
  strictly a POMDP).
- **Reward depends on mocks.** The M4 reject term comes from `mock_verify`, not real OpenDSS
  (§7). Results are about the gate under a stand-in verifier.

---

## 3. Observation (12-d)

`Box(-1, 1, (12,), float32)`, built in `gating_env._build_obs`. Names and meanings are also
exposed to the dashboard through `OBS_FIELDS` / `describe_observation`.

| # | Name | Source | Meaning |
|---|---|---|---|
| 0 | `epistemic_uncertainty` | M2 | u: 0 = sure … 1 = no evidence (quality-adjusted) |
| 1 | `reason_value` | M2 | 1 if `trigger_reason` ∈ {value, both} |
| 2 | `reason_sensing` | M2 | 1 if `trigger_reason` ∈ {sensing, both} |
| 3 | `state_class` | M2 | safety class scaled to 0..1 (`/(k-1)`, k = 3) |
| 4–6 | `class_prob_0..2` | M2 | class probabilities (padded/truncated to k) |
| 7 | `max_node_vulnerability` | M1 | worst node exposure, 0..1 |
| 8 | `severity` | M1 | hazard stage scaled 0 / 0.33 / 0.66 / 1.0 |
| 9 | `time_to_hazard` | M1 | minutes to onset, clipped ±30, divided by 30 |
| 10 | `budget_remaining` | env | S2 calls left as a fraction of the budget |
| 11 | `observed_fraction` | M2 | share of M1 features actually measured |

Inputs M3 consumes from other modules:

- **M2 → M3** (`m2-out/0.3`, see `services/learned/module2_auq_engine/M2_TO_M3_CONTRACT.md`):
  `epistemic_uncertainty`, `competence_drop`, `trigger_reason` (`none|value|sensing|both`),
  `observed_fraction`, `state_class`, `class_probabilities`. Older `0.1`/`0.2` streams still
  load; missing `observed_fraction` defaults to 1.0 and a missing `trigger_reason` is derived
  from `competence_drop` and the `OF_FLOOR` (0.35).
- **M1 → M3** (`SystemContext`, currently **mocked**): `max_node_vulnerability`,
  `mean_node_vulnerability`, `top_at_risk_nodes`, `time_to_hazard_onset_min`, `severity`.

---

## 4. Actions and the budget rule

- Action 0 → `system1_action(raw)`, cost 0.
- Action 1 → `system2_action(raw, rng)`, cost `reward.deliberation_cost`, budget decremented by 1.
- If `budget_remaining <= 0` the env forces System 1 and sets
  `budget_exhausted_fallback = true` when System 2 had been requested. **Budget exhaustion is
  not an agent action** — the policy cannot choose to exhaust or bypass it.

### 4.1 Cause-aware gating (M2 `trigger_reason`)

| `trigger_reason` | Meaning | Intended gate behaviour |
|---|---|---|
| `value` | unseen conditions (cyclone) | escalate to System 2 |
| `sensing` | missing / blackout data | **conservative System 1** — deliberating on absent data buys nothing |
| `both` | cyclone + comms loss | escalate (worst case) |
| `none` | nominal | System 1 unless severity demands otherwise |

Treating a value drop and a sensing drop identically wastes budget on blackouts. The
`u > threshold` baseline is reported so this claim can be scored, not asserted.

---

## 5. Reward

Per step: `reward = benefit − cost − reject`.

- `cost` = `deliberation_cost` if the effective action is System 2, else 0.
- `reject` = `reject_penalty` if `mock_verify` returns REJECT, else 0.
- `benefit` is the first matching case below (`gating_env.step`):

| # | Condition (effective action, cause, severity) | Benefit |
|---|---|---|
| 1 | S2 and `trigger_reason == sensing` | `sensing_escalation_penalty` (−1.2) |
| 2 | S2, calm (`severity_norm < calm_severity_cutoff`), `u < trigger_threshold`, not value-axis | `needless_escalation_penalty` (−0.8) |
| 3 | S2, calm, not value-axis (u ≥ threshold) | `needless_partial_scale` × needless penalty (−0.4) |
| 4 | S2 and value-axis (`value`/`both`) | `benefit_scale × max(severity_norm, value_severity_floor) × max(u, value_uncertainty_floor)` |
| 5 | S2 otherwise | `benefit_scale × severity_norm × u` |
| 6 | S1, value-axis, `severity_norm > missed_severity_cutoff` | −`missed_escalation_penalty` (−2.5) |
| 7 | anything else | 0 |

`severity_norm` = stage index / 3. Current values (`config.yaml`):

| Parameter | Value |
|---|---|
| `deliberation_cost` | 0.10 |
| `benefit_scale` | 4.0 |
| `needless_escalation_penalty` | −0.8 |
| `sensing_escalation_penalty` | −1.2 |
| `missed_escalation_penalty` | 2.5 (applied as −2.5, value/both only) |
| `reject_penalty` | 1.5 |
| `trigger_threshold` | 0.40 (also the threshold baseline) |
| `calm_severity_cutoff` | 0.25 |
| `needless_partial_scale` | 0.5 |
| `value_severity_floor` / `value_uncertainty_floor` | 0.35 / 0.5 |
| `missed_severity_cutoff` | 0.5 |

**Not yet in the reward:** a latency/thermal penalty. The planned design is survival benefit
against a latency/thermal penalty; only the deliberation cost and budget stand in for it today.

### Planned: M4 → M3 feedback

A REJECT from M4 should eventually raise deliberation-cost pressure (or shape the policy), not
merely discard the action. Today the reject term is a flat penalty from the mock verifier.

---

## 6. Scenario and data

### 6.1 M1 context (mock) — `synthetic_context.py`

`sample_scenario` builds one episode that walks `normal → elevated → severe → extreme` in
equal quarters (stage boundaries overridable). Per step:

| Stage | `max_node_vulnerability` range | Named at-risk nodes |
|---|---|---|
| normal | 0.00–0.20 | 1 |
| elevated | 0.20–0.45 | 3 |
| severe | 0.45–0.75 | 5 |
| extreme | 0.75–1.00 | 8 |

Nodes come from `N1..N12`. Hazard onset is at `hazard_onset_step_frac` (0.55) of the episode;
`time_to_hazard_onset_min` starts at `lead_time_at_start_min` (30) and counts down through
zero, going negative once the hazard is underway.

### 6.2 M2 stream — `m2_stream.py`

`replay_stream` bootstrap-resamples M2's published mock (`sample_m2_to_m3.jsonl`) with small
jitter (u ±0.03, class probabilities ±0.02, `observed_fraction` ±0.02). It is **not** new
synthetic M2 data. The draw is biased by severity so axes appear realistically:

| Severity | Draw |
|---|---|
| normal | mostly `none`; 12% `sensing` |
| elevated | 15% `sensing`, 20% `value`, rest `none` |
| severe / extreme | 15% `both`, 10% `sensing`, 75% `value` |

A local vendored copy of the M2 sample is preferred over `../module2_auq_engine/…` when it
exists. **Keep the vendored copy in sync with M2's** — a stale copy silently trains against
numbers M2 no longer produces. `OF_FLOOR` mirrors M2's
`config.yaml: trigger.observed_fraction_floor`; update it if M2 recalibrates.

---

## 7. Control paths and the mock verifier

Priority tiers (`priority.py`): N1–N3 = tier 1 (critical), N4–N6 = tier 2, N7–N12 = tier 3.

**System 1 (`system1.py`)** — cheap reactive heuristic. Sheds load proportional to
vulnerability, lowest tiers first, never tier 1; leaves dispatch untouched; no breaker
commands. Under a sensing drop it sheds less (scale 0.45 vs 0.8, max 0.25 per node) because the
grid cannot be seen — *conservative fallback*.

**System 2 (`system2.py`)** — deliberative stand-in. Sheds tier 3 then tier 2, never tier 1,
with a lower total shed (scale 0.55); adds a dispatch bump on generation node `N4`; keeps
breaker `E_crit_1` closed. It is a stand-in, not a real optimiser; the real
load-shed-prioritisation / critical-load-ranking policy is open work (§10).

**Mock M4 (`verifier.py::mock_verify`)** — not a control path. APPROVE/REJECT with a
SHA-256-stable pseudo-random draw. Reject probability starts at 0.02, +0.55 if a tier-1 node is
shed, +0.35 for S1 under extreme (+0.15 under severe); multiplied by 0.4 for System 2
("deliberation reduces rejection risk"). This builds in the assumption that S2 is safer, so
reported gains are conditional on it until real M4 verdicts replace it.

---

## 8. Training and evaluation

### 8.1 Policy — `policy.py`

MLP `12 → 32 → 32 → 2` with Tanh. Update is vanilla REINFORCE with **per-trajectory return
standardisation** (z-scored undiscounted return is the advantage; no moving-average baseline),
gradient clipping at 1.0.

### 8.2 Pipeline — `run_demo.py`

1. Seed Python, NumPy and Torch from `config.yaml: seed`; deterministic algorithms on.
2. Initial bias toward System 1 (`bias[0]=+0.4`, `bias[1]=−0.4`) so early training does not burn
   the budget on calm steps.
3. **Behaviour-cloning warm-start** (40 episodes) onto a cause-aware heuristic: sensing-only → S1;
   value/both → S2 when severity ≥ 0.66, or ≥ 0.33 with u ≥ 0.85; otherwise S2 only at extreme with
   u ≥ 0.5.
4. **REINFORCE** fine-tune: `train.episodes` = 150, `lr` = 0.004, update every
   `batch_episodes` = 5.
5. Evaluate over `eval.n_episodes` = 12 against baselines.
6. Emit one `ProposedControlAction` + `GatingDecision` pair per severity stage.

### 8.3 Baselines and metrics — `m3_evaluate.py`

No number is reported without the baseline it beats (repo convention).

| Item | What it is |
|---|---|
| always-S1 / always-S2 | trivial bounds |
| `u > 0.40` threshold | the "why not just a threshold?" baseline |
| trained policy | the contribution |
| total reward | mean over eval episodes |
| avg deliberation cost | per-step mean cost |
| escalation by severity | must be **non-decreasing** normal → extreme (sensing-only steps excluded) |
| escalation by `trigger_reason` | value/both high, sensing and none low |

Recorded result *(README, not re-run)*: total reward was −107.75 for always-S1, −99.33 for
always-S2 and 117.98 for the trained policy (the README does not record the threshold
baseline's number); escalation
by severity 0.000 / 0.261 / 1.000 / 1.000 (monotonic: true); escalation by `trigger_reason`
none/value/sensing/both = 0.000 / 1.000 / 0.000 / 1.000. After any config change, re-run
`python run_demo.py` and judge the *pattern*: policy beats all three baselines, escalation is
non-decreasing, sensing is pinned to ~0.

---

## 9. Interfaces

### 9.1 M3 → M4 contract (`m3-out/0.1`, JSON transport; proto is the source of truth)

Two message types per control step, alternating in `sample_m3_to_m4.jsonl`, correlated by
`action_id`.

**`ProposedControlAction`** — `action_id`, `origin` (`SYSTEM1`/`SYSTEM2`), `breakers`
`[{edge_id, closed}]`, `load_shed` `[{node_id, shed_fraction 0..1, priority_tier}]` (tier 1 = most
critical), `dispatch` `[{node_id, p_kw, q_kvar}]`, `rationale`, `schema_version`,
`message_type`.

**`GatingDecision`** — `action_id`, `chosen` (effective path after budget fallback),
`epistemic_at_decision`, `expected_survival_benefit`, `deliberation_cost`, `latency_ms`,
`budget_exhausted_fallback`, `schema_version`, `message_type` (the sample also carries
`timestamp`).

Rules: publish schema before implementation; minor-version bumps only; ping the M4 owner
before changing field names. The contract freezes at PP1. Full field table:
`services/learned/module3_metapolicy/M3_TO_M4_CONTRACT.md` (v0.1 draft). M4 already ingests the
sample stream in `deterministic-core/module4_verification/tests/test_m3_integration.py`.

`action_id` and timestamps in the sample are derived from seed + emit index, so reruns are
reproducible; the committed `sample_m3_to_m4.jsonl` is the M4 contract fixture.

### 9.2 Gateway and dashboard

| Route | What it does |
|---|---|
| `POST /api/module3/run` | Runs `run_demo.py <config> <output>` as a subprocess with overridden knobs (seed, episode length, budget, train/eval episodes, reward weights); returns reward, escalation, decisions and `decision_context`. Never overwrites the committed fixture. |
| `POST /api/module3/generate` | `interactive.py generate`: draws one 12-d situation from a trained config |
| `POST /api/module3/process` | `interactive.py process`: runs the saved policy (`artifacts/dashboard/policy.pt`) on it and returns the plan |

Dashboard page: `/gating` — run-config form, reward and escalation charts, expandable decision
log (M2 input, mock M4 verdict, full proposed plan), and the Generate / Process step-through.
`decision_context` is dashboard-only; it never travels on the M3→M4 wire. This path needs
`torch`, `gymnasium`, `numpy`, `pyyaml` importable in the gateway process (true for
`uv run` from the repo root; **not** for the gateway's own slim docker image — wiring the
compose path is future work).

---

## 10. Status against plan

Derived from the ClickUp list *Metacore M3 – Cost-Aware Metacognitive Gating* and the code.

**Phase A — Foundations & Design (TAF) — COMPLETE.** Literature locked in; contracts published;
formal MDP definition is §2 of this document (the Meta-RL formulation is Part II, §14–§15).

**Phase B — Prototype / Week-7 demo — COMPLETE.** Scaffold + Gym env, reward-shaping draft,
System 1, System 2 stub, threshold gate and toy Meta-RL all exist. The Tier-1 file-based dry run
(M1→M2→M3→M4) worked, and the Charter / Proposal and the toy-Meta-RL demo were submitted. The
panel's only feedback was to make the data-validation approach explicit; it was already in the
submission, and it will be stated clearly again at PP1.

**Phase C — Core Gating Engine (PP1, ~50%) — IN PROGRESS.** PP1 deliverable: a working gating
engine plus System 2 policy with metrics and a live demo. Task order is set in ClickUp; the
Meta-RL pipeline is first.

* **System 2 (proper) — IN PROGRESS (started 2026-10-17), option C chosen.** Built: critical-load
  ranking and a shortage-aware planner, plus an islanded (main-supply-lost) model in the
  surrogate (§23). Waiting on: M4 to represent a lost main supply (change request, §24) so the
  planner can be tested against the real firewall.
* **Training stability — FIXED (2026-10-17, §25).** Standardising the critic's target made all 5
  seeds beat the rule in both worlds. Open caveat: the GRU's memory is not shown to help (§25.6).
* **Latency / thermal / gate tuning — IN PROGRESS (2026-10-19, §27).** Done: measured latency
  (our side <1 ms p99; real M4 p50 30 ms, p99 48 ms, max 65 ms — over the 50 ms budget), a tested
  `LatencyGuard` (deadline + circuit breaker, not yet wired in), and gate tuning (no gain: the
  greedy gate is already stable). Not done: the thermal half (undefined, §27.4); the reward's
  invented 2/18 ms latency constants need a decision (§27.5).
* **GRU memory test — RUN (2026-10-19, §26).** With the damage hidden the GRU helps a little
  (blind: +0.3 to +0.5) and a lot when severity is also hidden (the memoryless gate collapses to
  always-S1).
* **PP1 pack — WRITTEN (2026-10-19, §30).** Checklist, headline metrics, a new real-M4 check of all
  five final gates (surrogate agrees to within 0.5 return; gate next to the oracle), a verified demo
  runbook, what not to claim, and who has to act before PP1.
* **Cyclone lab dashboard page — BUILT (2026-10-19, §29).** `/cyclone` in the dashboard: set the
  storm's parameters (or pick a preset / random storm), play it through a trained gate and the
  baselines, read the timeline. Not yet the live-M4 or the deliberation-cost view.
* **Repeated-trials test — RUN (2026-10-19, §28). Negative.** Trained and tested on 4 plays of the
  same storm, the GRU does not improve from play 1 to play 4 (fresh damage +0.4 ± 6.1; identical
  replay −1.5 ± 1.4), and the memoryless gate scores higher in this setup. No evidence of
  within-storm meta-learning.
* **Meta-RL pipeline — BUILT, READY FOR DEV REVIEW (2026-10-17).** Pushed on `m3/gate-policy`; no
  PR opened; ClickUp has no "PR review by dev" status yet. PPO + GRU (RL²) gate, simulated
  cyclones with four disturbances, physical priority-weighted reward, surrogate-trained /
  live-M4-tested. Design: Part II (§14–§20). Results: §22. Open caveats: §21 (F1–F8).
  Not yet done under this task: the dashboard page to set cyclone parameters and test the gate
  (planned). The proper System 2 is its own task — see §23.

**Open work (Phases C–F), in rough dependency order**

1. Replace `synthetic_context` with the live M1 `SystemContext` (agree with the M1 owner).
2. Replace `mock_verify` with real M4 verdicts; implement the M4 → M3 feedback loop.
3. Real System 2: load-shed prioritisation and critical-load ranking, replacing the stand-in.
4. ~~Full Meta-RL pipeline with reward shaping on simulated cyclone episodes tied to the OpenDSS
   twin~~ — **built, awaiting review** (Part II). Still open: swap the mock M2 stream for M2's
   real uncertainty API; replace the placeholder Burevi/Ditwah profiles with real data.
5. Latency/thermal profiling and enforcement; robust gate tuning (hysteresis, calibrated
   escalation threshold) — also decides MDP vs SMDP (§2.1).
6. Ablation for the paper: cause-aware gate vs a `competence_drop`-only gate; Meta-RL vs
   fixed-threshold vs always-deliberate.
7. Benchmarks against historical Sri Lankan cyclone profiles (Burevi, Ditwah): survivability
   gained vs deliberation cost.
8. Integration and contract-conformance tests with M2 and M4; end-to-end M2 → M3 → M4 wiring;
   dashboard gating-decision *timeline* (the current decision log is not yet a timeline).
9. ONNX export and the Go `gating_decision_svc` hot path (the Go service is currently a stub).
10. Paper, thesis chapter, MLOps report, PP2 demo.

---

## 11. Mocked vs real

| Piece | Now | Later |
|---|---|---|
| M1 `SystemContext` | mock (`synthetic_context.py`) | live M1 adapter |
| M2 uncertainty | real published mock stream (`m2-out/0.3`), resampled | M2 real API |
| M4 verdicts | `mock_verify` (Week-7 demo) · `SurrogateTwin` for Meta-RL training · **real M4 over HTTP for the final test** (`LiveM4Twin`) | real OpenDSS verdicts over gRPC |
| Cyclone / damage | mock 4-stage ramp (Week-7) · simulated cyclones with 4 disturbances (Meta-RL pipeline); Burevi/Ditwah profiles are placeholders | fitted to real best-track data |
| System 1 / System 2 | heuristic stand-ins | S2: real optimiser |
| Transport | JSON, field names from `module3.proto` | gRPC |
| Serving | Python / gateway subprocess | ONNX + Go `gating_decision_svc` |

---

## 12. File map

All paths under `services/learned/module3_metapolicy/` unless noted. **This table covers the
Week-7 flat files. The Meta-RL pipeline's files (`src/module3/**`, `pipeline.yaml`, `scripts/`)
are mapped function-by-function in §19.**

| File | Role |
|---|---|
| `gating_env.py` | Gymnasium env, 12-d observation, reward, budget rule, `OBS_FIELDS` |
| `synthetic_context.py` | mock M1 context + severity schedule |
| `m2_stream.py` | M2 JSONL loader and reason-aware replay |
| `system1.py`, `system2.py`, `priority.py` | control-path stand-ins and shared tier map |
| `verifier.py` | mock M4 |
| `policy.py` | MLP + REINFORCE |
| `m3_evaluate.py` | baselines and escalation metrics |
| `run_demo.py` | BC warm-start → REINFORCE → eval → contract sample |
| `interactive.py` | dashboard Generate / Process step-through |
| `config.yaml` | seed, episode length, budget, reward weights, training/eval sizes |
| `sample_m2_to_m3.jsonl` | vendored M2 stream (keep synced) |
| `sample_m3_to_m4.jsonl`, `M3_TO_M4_CONTRACT.md` | M4 contract fixture and field table |
| `tests/` | `test_gating_env`, `test_policy`, `test_verifier`, `test_interactive` |
| `src/module3/` | empty scaffold packages (`env`, `policy`, `train`, `export`); code is still in flat files |
| `services/gateway/gateway/routers/module3.py` | `/api/module3/*` |
| `apps/dashboard/src/routes/gating/` | `/gating` page |
| `packages/contracts/proto/module3.proto` | wire contract |

## 13. Running it

```bash
# from services/learned/module3_metapolicy/
pip install -r requirements.txt
python run_demo.py                 # trains, prints the table, rewrites sample_m3_to_m4.jsonl

# dashboard path, from the repo root
uv sync --all-packages
uv run uvicorn gateway.main:app --app-dir services/gateway --reload --port 8000
cd apps/dashboard && pnpm install && pnpm dev     # open /gating
```

Reproducibility: everything is seeded from `config.yaml: seed`; `mock_verify` uses SHA-256
rather than salted `hash()`. Per-checkpoint model cards are a repo convention and are not yet
written for M3.

---

# Part II — Meta-RL pipeline (PPO + GRU, RL²-style)

Added 2026-10-15 for the ClickUp task *"Build full Meta-RL training pipeline with reward shaping
against simulated cyclone episodes linked to OpenDSS twin"*. Parts of §1–§13 describe the Week-7
toy (REINFORCE + hand-written reward table); that code is **unchanged** and still powers the
dashboard demo. Everything below is the new pipeline in `src/module3/`.
Reference papers: [`module3-research.md`](module3-research.md).

## 14. What the pipeline does, in one picture

```
  random CYCLONE  (scenario/cyclone.py)             M2 uncertainty stream (published mock)
  wind -> vulnerability, 4 disturbances                          |
        |                                                        |
        +--------------> OBSERVATION (15-d) <--------------------+
                               |
                               v
                    RecurrentGate  (GRU memory, PPO-trained)
                               |  picks  System 1  or  System 2
              +----------------+-----------------+
              v                                  v
   System 1: reactive_plan             System 2: deliberative_plan
   (vulnerability-based shed)          (reads breaker status, sheds cut-off buses)
              +----------------+-----------------+
                               v
                  GRID TWIN scores the action under the storm's damage
                  (SurrogateTwin to train  |  LiveM4Twin to test)
                               v
        PHYSICAL REWARD = priority-weighted load served
                          - voltage violations - deliberation cost - latency cost
```

**Hybrid plan (agreed).** Train on the fast `SurrogateTwin`; test the trained policy against the
real M4 firewall (`LiveM4Twin`). We never train against live M4: it costs ~13 ms per call (it
recompiles the circuit every time), so one training run would need ~25 million calls.

## 15. Why PPO + GRU (RL²) — and why not MAML

### The three terms, plainly
* **PPO** — the algorithm that trains the gate. It improves the policy in small, *clipped* steps
  so one noisy batch of storms cannot undo earlier learning.
* **GRU** — a small recurrent layer with a memory. The gate sees the last step's action and reward
  as well as the current observation, so it can notice patterns like "uncertainty has been rising
  for five steps".
* **RL²** — the idea that makes it *Meta*-RL: train that recurrent policy across many different
  cyclones. The memory then learns to work out *which kind of storm this is* and to behave
  accordingly — adaptation by a forward pass, not by retraining.

### Why RL² and not MAML  *(a design decision to defend — read carefully)*
MAML (Finn et al. 2017) is the other well-known meta-learning method. It trains a starting set of
weights so that **a few gradient-descent steps on data from a new task** give a good policy. We
did not use it because it does not fit what the gate has to do:

1. **There is no spare data to adapt on.** MAML adapts by *collecting trajectories from the new
   task, then taking gradient steps, then acting*. A "task" for us is a single cyclone and one
   80-step episode that is lived once. There is no earlier run of that storm to learn from. RL²
   adapts *during* the episode, from what it has seen so far.
2. **Adaptation would sit in the real-time path.** The trained gate is exported to ONNX and served
   from Go (`gating_decision_svc`) inside a ~50 ms end-to-end reactive budget that M4 also needs.
   MAML needs gradient computation and an optimiser at deployment; RL² needs one GRU forward pass
   per step with a small hidden vector carried between steps.
3. **It is harder to train.** MAML for RL differentiates through policy-gradient estimates
   (second-order terms), which are noisy and unstable; our rewards are already sparse, driven by
   rare feeder trips. PPO + GRU is a standard, stable recipe.
4. **The thing to adapt to is a latent state, not a new skill.** What differs between storms is
   hidden (how exposed is the grid *right now*; is M2's `u` telling the truth?). A memory that
   infers a hidden state is the natural tool; fast re-training is not what is needed.
5. **Evidence.** Ni et al. (ICML 2022) report that recurrent model-free RL is a strong baseline
   against specialised meta-RL/POMDP methods. A simple baseline that holds up is the right default
   for a safety-critical component.

### MAML stays as an option
`policy.meta_method` in `pipeline.yaml` is `rl2` (used). The value `maml` is **reserved**: the
pipeline raises `NotImplementedError` with a pointer here. MAML becomes the right tool if:

* we want **site-specific adaptation offline** — fine-tune a trained gate on logged storms from a
  *new* microgrid, where there is data and time for gradient steps (not in the hot loop); or
* the **topology changes** between deployments and few-shot adaptation to a new grid is needed.

To add it: implement `train/maml.py` (inner loop = a few PPO/REINFORCE steps on one task's
trajectories; outer loop = update the initial weights), reuse `CycloneGatingEnv` and
`RecurrentGate(arch="mlp")` unchanged, and add the comparison to the ablation table.

### The ablation that keeps us honest
`--ablate-mlp` trains the *same* network with the GRU swapped for a memoryless layer. If the GRU
gate does not beat the MLP gate, the memory is not earning its keep and the paper must say so.

## 16. The simulated cyclone — the four disturbances

Code: `src/module3/scenario/cyclone.py`. One storm is generated up front from a seed (a cyclone
does not react to what the controller does), so every policy can be scored on the *identical*
storm. Wind follows a bell-shaped pulse; the severity stage comes from wind speed
(≥ 40 elevated, ≥ 80 severe, ≥ 120 extreme, km/h).

| # | Disturbance | How it is modelled | Parameter |
|---|---|---|---|
| 1 | **PV derate** | PV output falls linearly from 100% at 50 km/h to 0% at 100 km/h | cut-in / cut-out wind |
| 2 | **Overhead-feeder trips** | each step an overhead feeder (`Line_N8_N11`, `Line_N8_N12`) trips with probability `trip_scale × v³ × boost`, v = vulnerability of its end node; a trip is **permanent** | `trip_scale`, `ood_trip_boost` |
| 3 | **Diesel availability** | once the storm is severe, each step has a chance a diesel unit is lost for the rest of the episode (availability drops to 50%) | `diesel_loss_prob` |
| 4 | **Communications cut** | outages start more often in high wind and last 3–8 steps; M2 then reports `sensing` (or `both`) and what the controller *knows* freezes at its last good value | `comms_rate` |

`ood=True` storms are "unseen conditions": M2 reports value-axis uncertainty (`value`) and trips
are more likely than System 1's vulnerability rule anticipates. This is what makes `u` informative.

**Task distribution (training).** Peak wind 45–190 km/h, peak position 0.35–0.70 of the episode,
width 0.07–0.20, 30% OOD. **Held-out profiles (testing):** `burevi_like`, `ditwah_like`.

> ⚠ These two are **placeholder parameter sets** with a plausible shape, not fitted to real track
> data. No survivability claim may be made about the real cyclones until they are replaced with
> best-track / wind data. Tracked in `module3-research.md` → "Still to read / cite properly".

## 17. The grid twin: surrogate for training, real M4 for testing

| | `SurrogateTwin` | `LiveM4Twin` |
|---|---|---|
| File | `twin/surrogate.py` | `twin/live_m4.py` |
| Speed | microseconds | ~13 ms per call (HTTP + circuit recompile) |
| Used for | PPO training, bulk evaluation | final test of trained policies |
| Imports OpenDSS? | no | no — talks HTTP to `POST /api/module4/verify` |

M3 may not import OpenDSS (ADR 0003), so `twin/topology.py` restates the grid facts and
`tests/test_twin_topology.py` parses `delft_3island.dss` and fails if they ever drift.

**How the storm is applied to the real M4 without changing it.** The disturbance rides along as
extra commands in the verify request: a tripped line becomes `{edge_id, closed: false}`; PV/diesel
loss becomes a lowered dispatch setpoint. (A native `scenario` field would be cleaner — see §21.)

**Validation.** `scripts/validate_surrogate.py` replays random scenarios on both twins:
300 cases → **97% agreement** on APPROVE/REJECT and on the exact set of violating buses; mean
absolute error in served-load fraction 0.004. Every disagreement was *surrogate rejects, live M4
approves* — see finding F2 in §21.

## 18. The physical reward — what we prioritise and how

```
reward = w_serve * served_fraction
       - w_viol  * (violating buses / 10)
       - deliberation_cost            (only if System 2 was used)
       - w_latency * path_latency_ms / 50 ms
```
Defaults (`pipeline.yaml`): `w_serve=1.0, w_viol=2.0, deliberation_cost=0.05, w_latency=0.05`;
latency 2 ms (S1) / 18 ms (S2) are stand-ins until the profiling task measures them.

**What "priority-weighted load served" prioritises.** Each node has a priority tier, and a kW
served at a higher tier counts more:

| Tier | Nodes | Weight | Meaning |
|---|---|---|---|
| 1 | N1, N2, N3 | 3 | critical — never shed by either control path |
| 2 | N4, N5, N6 | 2 | important |
| 3 | N8, N9, N11, N12 | 1 | sheddable first |

`served_fraction = Σ weight × load_kW × (1 − shed) × [bus energised]  /  Σ weight × load_kW`.
So losing 100 kW of tier-1 load costs three times losing 100 kW of tier-3 load, and a bus that a
tripped feeder cuts off counts as fully unserved. Code: `twin/scoring.py::priority_weighted_served`
(weights editable in `pipeline.yaml: tier_weights`), combined in `env/reward.py::step_reward`.

Versus the Week-7 table (`gating_env.py`: 7 hand-written cases such as "escalate on a calm step:
−0.8"): the new reward scores **outcomes on the grid**, so no rule has to be guessed. The old table
stays as the dashboard demo and as an ablation baseline.

## 19. Code map — where everything is, and what goes in and out

All paths under `services/learned/module3_metapolicy/` (`src/module3/` unless noted).

### 19.1 System 1 — the reactive path
| Function | File | Input → Output |
|---|---|---|
| `system1_action(obs_raw)` | `system1.py` (flat, Week-7) | `obs_raw` dict with `max_node_vulnerability`, `trigger_reason`, `top_at_risk_nodes` → `{origin:"SYSTEM1", breakers:[], load_shed:[{node_id, shed_fraction, priority_tier}], dispatch:[], rationale}` |
| `reactive_plan(context, trigger_reason)` | `control/reactive.py` | thin wrapper: merges the two inputs and calls `system1_action` (one shared System 1) |

How it decides: sheds `vulnerability × 0.8` of load (× 0.45 if telemetry is lost, max 0.25 per
node), lowest-priority tiers first, never tier 1. Uses `priority.py::PRIORITY`. **It never looks at
which lines are tripped** — that blindness is what System 2 fixes.

### 19.2 System 2 — the deliberative path
| Function | File | Input → Output |
|---|---|---|
| `deliberative_plan(known_tripped, context, *, grid_lost, pv_factor, diesel_factor, tier_weights, reserve_margin)` | `control/planner.py` | the believed grid state (stale if comms are down) → `{origin:"SYSTEM2", load_shed:[…], dispatch:[…], breakers:[], rationale}`. Full algorithm in §23 |
| `critical_load_ranking()`, `shed_priority_order(nodes)` | same | the ranked list of loads / the order they are given up |
| `connected_groups`, `group_capacity_kw`, `unit_capacity_kw` | `twin/surrogate.py` | groups of buses joined by working lines; the generation a group has |
| `energized_buses(tripped)` | `twin/surrogate.py` | tripped line names → set of buses still connected to the source (graph walk) |
| `system2_action(obs_raw, rng)` | `system2.py` (flat, Week-7) | the old stand-in used by the dashboard: protects tier 1, closes `E_crit_1`, bumps `N4` dispatch |

System 2 thinks about the grid (topology, supply, shortage); System 1 only thinks about exposure.
The proper System 2 task is in progress — see §23.

### 19.3 The gate (the learned part)
| Item | File | What it does |
|---|---|---|
| `RecurrentGate` | `policy/recurrent_gate.py` | GRU (or MLP) actor-critic. Input = 15-d obs + prev action + prev reward + trial-done flag (19-d). Output = 2 logits (S1/S2), a value, the memory. Masks S2 when the budget is 0. |
| `build_inputs(obs, prev_action, prev_reward, trial_done)` | same | assembles the 18-d RL² input |
| `collect_tasks(env, model, n_tasks, trials, rng)` | `train/ppo.py` | runs the gate through storms, records what PPO needs |
| `compute_gae(rewards, values, γ, λ)` | same | advantages and return targets |
| `ppo_update(model, opt, batch, cfg, rng)` | same | clipped PPO epochs over whole-storm minibatches |
| `train_gate(env, model, cfg, trials, seed)` | same | the training loop |

### 19.4 Environment, scenario, twin, reward
| Item | File | Input → Output |
|---|---|---|
| `CycloneGatingEnv` (`reset`, `step`) | `env/cyclone_env.py` | `step(0/1)` → obs (15), reward, truncated, `info` (served, violations, reward parts, …) |
| `describe_observation(obs)` | same | names/sources for each of the 15 slots |
| `step_reward(outcome, used_s2, cfg)` | `env/reward.py` | twin outcome → `(reward, parts)` |
| `generate_episode(params, n, rng)` | `scenario/cyclone.py` | storm parameters → `CycloneEpisode` (`.disturbance(t)`, `.known(t)`, `.context(t)`) |
| `sample_params`, `resolve_params`, `PROFILES` | same | random storm / named profile |
| `SurrogateTwin.evaluate(action, disturbance)` | `twin/surrogate.py` | → `TwinOutcome(approved, violating_buses, served_fraction, …)` |
| `LiveM4Twin.evaluate`, `merge_with_overlay`, `overlay_commands` | `twin/live_m4.py` | same interface, over HTTP |
| `priority_weighted_served`, `shed_fractions` | `twin/scoring.py` | shed plan + energised buses → 0..1 |
| grid constants | `twin/topology.py` | nodes, loads, generators, lines, tiers, exposure |

### 19.5 Evaluation and the runner
| Item | File | What it does |
|---|---|---|
| `run_episode`, `evaluate_policy`, `aggregate` | `train/evaluate.py` | play identical storms with `always_s1`, `always_s2`, `threshold`, `breaker_rule`, `oracle`, `gate` |
| `run`, `main` | `train/pipeline.py` | seeds → train → test on random + held-out → optional live-M4 test → `results.json` + `report.md` |
| `validate_surrogate.py` | `scripts/` | surrogate-vs-live agreement check |
| `pipeline.yaml` | module root | every knob |
| `_flat.py` | `src/module3/` | temporary bridge to the flat files (`system1.py`, `m2_stream.py`) |

**The 15-d observation** (slots 0–11 identical to the toy env; 12–14 are new): `0 u · 1 reason_value ·
2 reason_sensing · 3 state_class · 4–6 class_prob · 7 max_node_vulnerability · 8 severity ·
9 time_to_hazard · 10 budget_remaining · 11 observed_fraction · 12 tripped_visible (share of
overhead feeders known tripped) · 13 gen_available_visible · 14 grid_lost_visible`. Slots 12–14 come from breaker/SCADA
status and freeze at their last value when telemetry is lost.

### 19.6 The baselines (what the gate must beat)
`always_s1`, `always_s2` · `threshold` (S2 iff `u > 0.40`) · `breaker_rule` (S2 iff a tripped
feeder is visible and storm ≥ elevated) · `oracle` (S2 iff a feeder is *actually* tripped — uses
ground truth the gate cannot see, so it is a ceiling, not a competitor).

## 20. How to run

```bash
# from services/learned/module3_metapolicy/  (workspace venv: .venv at the repo root)
python -m module3.train.pipeline --quick                         # ~20 s smoke run
python -m module3.train.pipeline --ablate-mlp                    # 5 seeds + memoryless ablation
python -m module3.train.pipeline --ablate-mlp --live-m4 http://127.0.0.1:8000   # + real-M4 test
python scripts/validate_surrogate.py --n 300                     # surrogate vs real M4
# live tests need the gateway:  uv run uvicorn gateway.main:app --app-dir services/gateway --port 8000
pytest tests -q
```
Outputs land in `artifacts/pipeline/<run>/` (git-ignored): `results.json`, `report.md`, one
`gate_seed<N>.pt` per seed. **Planned, not built:** a dashboard page to set cyclone parameters,
run the gate and watch decisions/violations step by step.

## 21. Findings and open questions for the team (from building and probing the pipeline)

Everything here was observed by running the real M4 locally on 2026-10-15; reproduction steps
are given so the owner can confirm. **F2 and F3 are for Hariswara (M4).**

| # | Finding | Evidence | Effect on M3 | Ask |
|---|---|---|---|---|
| **F1** | **Buses N7 and N10 do not exist in the twin.** The Week-7 flat code (`synthetic_context.NODE_POOL`, `priority.PRIORITY`, `mock_verify`) still uses N1..N12. | `delft_3island.dss` has no N7/N10; M4 returns `MALFORMED_ACTION` for any action naming them (probe: shed N7 → REJECT). | The Meta-RL pipeline uses the real 10 nodes only. The dashboard demo would be rejected by live M4 whenever S1/S2 picked N7/N10. | M3 (me): fix the flat code before wiring the demo to live M4. |
| **F2** | **M4's load matching is a substring match.** `set_load_shed("N1", 1.0)` also zeroes `Load_N11` and `Load_N12` (`node_id.lower() in load_name.lower()`). | Direct call to `CircuitTwin.set_load_shed("N1", 1.0)` changed N1 (150→0), **N11 (80→0), N12 (90→0)**. `N2` alone is correct. | None for training (the pipeline never sheds N1). It would silently over-shed in production. | Hariswara: match on the bus name exactly (`circuit.py::set_load_shed`, also `set_generator_dispatch`). |
| **F2b** | **M4 sometimes approves a state with a dead bus that still carries load.** | Trip `Line_N8_N12` alone → REJECT (N12 at 0 pu). Trip `Line_N8_N12` + `Line_N4_N6` and shed N6 100% → **APPROVE**, although N12 (90 kW) is de-energised. 3% of 300 random cases disagreed with the surrogate this way. | The surrogate is *stricter* than M4 here (it rejects). Not used to justify any M3 result. | Hariswara: check how bus voltages are read when several buses are islanded (`get_bus_voltages_pu`). Possibly related to F2 (not confirmed). |
| **F3** | **The stiff slack source (100 MVA short-circuit) makes generation disturbances invisible to the verdict.** PV derate and diesel loss never produce a violation; only topology (feeder/tie trips) does. | Probe: PV → 0 kW, all diesel → 0 kW both APPROVE. Tripping `Line_Source_N1` collapses *every* bus to 0 pu because no local source is grid-forming. | Disturbances #1 and #3 reach the gate through its observation (`gen_available_visible`) but do not change the reward. Honest consequence: the pipeline is currently driven by **feeder trips and comms loss**. | Hariswara: add a "grid-lost" scenario with grid-forming local diesel/BESS so islanded operation is representable; then PV/diesel losses and System 2 *dispatch* start to matter. |
| **F4** | **Priority tiers disagree.** The `.dss` names loads `_Pri` for N1, N3, N4 and `_NonPri` for N2, N5; M3's tier map says N1–N3 critical, N4–N6 important. | `delft_3island.dss` load names vs `priority.py`. | The reward weights follow M3's map. | Agree one critical-load list with Hariswara/the supervisor; it directly sets what "priority-weighted" means. |
| **F5** | **`verify()` costs ~13 ms, but only ~0.2 ms is the power-flow solve.** The rest is recompiling the circuit on every call (`reset_to_base`). | Timed 20 calls; M4's own `solve_latency_ms` reports only the solve. | The latency terms in the reward and the "latency under load" task must not quote 0.2 ms as the M4 cost. | Hariswara: snapshot/restore instead of recompiling; report end-to-end latency. |
| **F6** | **Severity-monotone escalation is the wrong yardstick once storms cause damage.** A tripped feeder stays out after the wind drops, so escalating late in an episode is *correct*. | Gate escalates at stage `normal` when a feeder is still out. | The report shows escalation **conditional on damage** (S2 rate if a feeder is out vs if the grid is intact) instead of only by severity. | none |
| **F7** | The historical profiles are placeholders. | `scenario/cyclone.py::PROFILES`. | Held-out results are about *a held-out storm shape*, not Burevi/Ditwah. | Get best-track/wind data. |
| **F8** | The surrogate has no thermal or over-voltage model. | `twin/surrogate.py` docstring. | Fine while no policy proposes extreme dispatch; must be revisited when System 2 starts dispatching. | Revisit with the proper System 2 task. |

**Requested M4 interface change (nice-to-have).** A native `scenario` field on `/verify`
(tripped lines, PV/diesel factors) so M3 stops encoding storm damage as fake breaker/dispatch
commands. Needs Hariswara's agreement before any change.

## 22. Results — first full run (2026-10-15)  *(superseded by §25.6 — produced before the stability fix)*

Run: `python -m module3.train.pipeline --ablate-mlp --live-m4 http://127.0.0.1:8000`; 5 seeds,
120 PPO updates × 16 storms each, 39 min wall clock. Per-storm return over 80 steps, higher is
better, mean ± std across the 5 training seeds. Every policy is scored on identical storms.
Full tables: `artifacts/pipeline/run_5seed/report.md` (git-ignored; regenerate with the command).

| policy | fresh random storms | held-out `ditwah_like` | held-out `burevi_like` |
|---|---|---|---|
| always S1 | 68.98 ± 1.23 | 64.12 ± 0.82 | 73.49 ± 0.90 |
| always S2 | 67.74 ± 1.04 | 62.17 ± 0.81 | 71.58 ± 0.75 |
| threshold (`u > 0.40`) | 70.25 ± 1.21 | 66.53 ± 0.61 | 73.54 ± 0.85 |
| breaker rule | **72.26 ± 0.80** | 67.59 ± 0.62 | **75.03 ± 0.64** |
| oracle (cheats, ceiling) | 75.03 ± 0.46 | 73.89 ± 0.23 | 76.80 ± 0.43 |
| **gate (GRU, PPO)** | 70.93 ± 3.58 | **68.32 ± 4.71** | 73.95 ± 2.03 |
| gate, memoryless (MLP) | 70.52 ± 2.99 | 66.93 ± 3.91 | 74.03 ± 1.56 |

Per-seed return on fresh storms (the spread is the story):

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| gate (GRU) | 74.56 | 74.07 | 67.70 | 66.76 | 71.55 |
| gate (MLP) | 73.40 | 69.79 | 67.70 | 67.77 | 73.92 |
| breaker rule | 72.29 | 72.82 | 71.31 | 71.65 | 73.24 |

### What this does and does not show
* **The pipeline works end to end** — training, held-out testing, ablation, and a real-M4 test.
* **The trained gate beats the naive baselines on average** (always-S1/S2, and the `u`-threshold
  on held-out `ditwah_like`), and in 2 of 5 seeds (0, 1) it beats *every* non-cheating baseline and
  gets close to the oracle (74.6 vs 75.0).
* **It is NOT yet reliably better than a simple breaker-status rule.** Averaged over seeds the gate
  (70.93) trails the rule (72.26), because **PPO training is unstable across seeds**: two seeds
  collapsed into a fixed habit — seed 2 to *never* escalate (S2 rate 0.00) and seed 3 to *always*
  escalate (S2 rate 0.56, the budget cap; entropy 0.04). *(Correction, 2026-10-17: an earlier
  version of this note said both seeds stayed on System 1; the training logs show seed 3 did the
  opposite.)* Any paper claim must be made on the distribution over seeds, not on seed 0.
* **GRU vs MLP is not distinguishable** at this sample size (70.9 vs 70.5; ranges overlap). Do not
  claim memory helps until a longer or tuned run separates them. `ditwah_like` leans GRU
  (68.3 vs 66.9) but the std is ~4.
* **Live M4 agrees with the surrogate on every policy** (return gap 0.00 in all 21 cells, seed-0
  gate, 6 storms per suite; ~10k real verify calls). This confirms the surrogate for the actions
  these policies produce; it does not contradict the 97% random-case agreement in §17, because
  the two control paths never emit the odd multi-trip corner cases where they differ (F2b).
  Caveat: seed 0 is the *best* seed, and 6 storms is small.
* **Escalation tracks damage, not just severity** (see F6): the oracle escalates 91–97% of
  steps when a feeder is out and 0% otherwise; the gate does 37–43% when out and ~19% when the
  grid is intact — it still wastes budget on intact-grid steps and misses many real outages.

### Why it is not yet better, and what to try (in order)
1. **Stabilise training**: more updates and larger batches; lower learning rate; behaviour-clone
   the gate onto the breaker rule first, then fine-tune with PPO (the same recipe the Week-7
   demo used); normalise rewards.
2. **Sharper reward signal**: the cost of one dead bus (0.2 + lost load) vs one S2 call (0.07) is
   modest, so "do nothing" is a deep local optimum. Try a larger `w_viol` and re-check.
3. **Longer horizon / `trials_per_task` > 1** to give the GRU something to adapt over.
4. **Re-run with ≥ 10 seeds** once stable, and report confidence intervals.

---

# Part III — System 2 (proper) and the lost-supply scenario

Added 2026-10-17 for the ClickUp task *"Develop System 2 deliberative survival policy properly
(load-shed prioritisation, critical-load ranking) replacing Week-7 heuristic"*. Decision taken
with Saabir: **option C** — build the shortage model on the M3 side now (B), and ask M4 for a real
lost-supply scenario (A) so it can be verified later.

## 23. System 2 — critical-load ranking and shortage-aware shedding

### 23.1 The idea in plain words
Picture a cyclone cutting the main power connection. The three islands must run on their own
diesel, battery and solar. If the islands *need* more power than their generators can *make*, the
grid blacks out — unless someone switches off enough load first. System 2's job is to decide **who
gets switched off, in what order, and how much**, so the lights stay on where it matters most and
the blackout is avoided.

System 1 cannot do this: it only looks at "how exposed is each node?" and sheds in proportion.
It does not know the supply is gone or how much generation is left. That blindness is exactly what
deliberation pays for.

### 23.2 The ranking (who is given up first)
Code: `control/planner.py::shed_priority_order`, `critical_load_ranking`.

Loads are given up in this order: **lowest priority tier first; inside a tier, the largest load
first** (so fewer buses are touched); **tier 1 only as a last resort**, when every other load in
the group is already off. With the default weights (3/2/1):

| Give-up order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Node | N8 | N9 | N12 | N11 | N4 | N5 | N6 | N1 | N2 | N3 |
| Tier | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 1 | 1 | 1 |
| Load (kW) | 120 | 100 | 90 | 80 | 100 | 80 | 60 | 150 | 120 | 100 |

> ⚠ **Inside a tier the order is only a size tie-break, not a judgement of criticality.** For
> example N3 comes out "most critical" of the three tier-1 nodes purely because it is the smallest.
> A genuine critical-load ranking (hospital, water pumping, …) needs operator input — finding F4.
> Until then, the tiers carry the meaning and the in-tier order should not be quoted.

### 23.3 The algorithm (six steps)
1. **Groups.** Split the buses into groups still joined by working lines. With the main supply
   lost, each group must stand on its own generators.
2. **Supply.** Capacity of a group = PV × `pv_factor` + diesel × `diesel_factor` + battery.
   (While the main supply is up, a group joined to it has unlimited supply; a group cut off from
   it is dead and is shed entirely — the Part II behaviour.)
3. **Shortfall.** `excess = demand − capacity × (1 − reserve_margin)`. The 5% reserve margin is
   deliberate headroom so one further small loss does not cause a blackout.
4. **Rank** the loads as in §23.2.
5. **Shed.** Walk down the ranking, shedding each load fully until the excess is covered; the last
   one may be partly shed. Shed fractions are rounded **up**, never down.
6. **Dispatch.** Ask the dispatchable units (diesel, battery) to cover what PV does not.

**Worked example.** Main supply lost, PV dead (0%), diesel at half. Demand 1000 kW; capacity
= 600 (diesel) + 200 (battery) = 800 kW. With the 5% margin the target is 760 kW, so 240 kW must
go: N8 (120) + N9 (100) fully, then 20 kW of N12 (22.2%). All of tier 1 and 2 are untouched and
the twin approves the plan (served 88% of the priority-weighted load). Doing nothing (System 1's
effective behaviour) leaves a 200 kW shortfall and the whole system blacks out.

### 23.4 How a shortage is judged (and why it is an assumption)
`twin/surrogate.py::_evaluate_islanded`: a group survives only if its generation covers its
remaining demand; otherwise **every bus in the group reads as a violation and nothing is served**
(underfrequency collapse). The new disturbance `grid_lost` (opt-in: `scenario.p_grid_loss`, or
`--p-grid-loss`) starts during a severe step and lasts 12–30 steps. The gate sees it through new
observation slot 14, `grid_lost_visible` (stale if telemetry is down).

> ⚠ The real M4 cannot represent this (finding F3), so the islanded model is **not validated by
> `validate_surrogate.py`** and `LiveM4Twin` refuses to evaluate it (raises `NotImplementedError`).
> Live-M4 tests therefore run only on storms **without** grid loss. Closing that gap is the M4
> change request in §24.

### 23.5 What this System 2 does not do (yet)
* **Greedy, not optimal.** It is a rule-based ranking, not an optimiser; a knapsack/MILP over
  shed fractions could serve more priority-weighted load for the same shortfall.
* No battery state-of-charge, ramp limits, frequency dynamics or time horizon — the shortage is
  judged instant by instant.
* The reserve margin is a fixed 5%, not learned or tuned.
* No tie-line reconfiguration (breakers are never re-closed; storm-damaged lines stay out).
* Dispatch commands are within unit ratings but are **not** checked by any firewall.
* Its input is the *believed* state: under a data blackout it plans on stale information and can
  be wrong (this is deliberate and tested).

### 23.6 Results in the shortage world  *(superseded by §25.6 — produced before the stability fix)*
Run (2026-10-17): `python -m module3.train.pipeline --p-grid-loss 0.6`, 5 seeds, 60% of storms cut
the main supply, 18.5 min. Surrogate twin only (the real M4 cannot represent a lost supply, §23.4),
so **none of this is verified by M4**. Return per storm, mean ± std across seeds, identical storms
for every policy:

| policy | fresh random storms | held-out `ditwah_like` | held-out `burevi_like` |
|---|---|---|---|
| always S1 | 61.19 ± 0.86 | 54.05 ± 3.89 | 64.24 ± 4.53 |
| always S2 | 62.94 ± 0.85 | 54.33 ± 1.54 | 66.84 ± 1.34 |
| threshold (`u > 0.40`) | 65.28 ± 1.40 | 63.72 ± 0.84 | 66.11 ± 3.50 |
| outage rule (S2 if an outage is visible) | **71.02 ± 1.21** | **65.14 ± 0.96** | **73.00 ± 0.33** |
| oracle (cheats, ceiling) | 73.49 ± 0.85 | 71.17 ± 0.58 | 74.11 ± 0.56 |
| **gate (GRU, PPO)** | 69.31 ± 5.48 | 64.86 ± 9.49 | 70.51 ± 3.56 |

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| gate, fresh storms | **73.82** | 64.37 | **73.33** | 62.39 | **72.65** |
| outage rule | 71.41 | 72.49 | 70.71 | 69.18 | 71.33 |
| gate, `ditwah_like` | **71.10** | 53.32 | **71.86** | 55.69 | **72.31** |
| outage rule, `ditwah_like` | 64.50 | 64.01 | 65.31 | 65.32 | 66.54 |

**Reading.**
* **Shortage makes deliberation matter more.** Always-S1 falls from 68.98 (Part II world) to 61.19
  and the oracle-vs-S1 gap widens from about 6 to 12 points, as expected when a blackout is
  possible. The planner is doing the work: with the right trigger it recovers most of that gap.
* **3 of 5 seeds (0, 2, 4) beat the outage rule on fresh storms and on `ditwah_like`** (by 1–2
  points on fresh storms and 6–7 on `ditwah_like`). **Seeds 1 and 3 failed to learn** (62–64, close
  to always-S1). That is better than Part II (2 of 5) but the instability is not fixed, so the
  seed-mean (69.31) still trails the rule (71.02). Quote the distribution over seeds, not the best.
* **The gate over-escalates:** it uses System 2 on 47% of steps (rule 18%, oracle 34%) and 38% of
  steps where the grid is intact (rule 0%). It buys safety with budget; a better-tuned reward or
  training schedule should cut this.
* **Not shown:** whether the GRU beats a memoryless net (no ablation was run here), and any
  real-M4 result (impossible until §24 is done).

## 24. DRAFT change request to M4 — a "main supply lost" scenario  *(not sent)*

**To:** Hariswara (M4) · **From:** Saabir (M3) · **Status:** draft, for Saabir to review and send.

**Why.** M3's System 2 must decide which loads to shed when local generation cannot cover demand.
The current twin cannot tell a good shed plan from a bad one: the source has a 100 MVA
short-circuit rating, so any shortfall is covered and M4 approves every plan. Tripping
`Line_Source_N1` does not help either — every bus then reads 0 pu because nothing in the circuit is
grid-forming. (Probes and reproduction: `docs/module3.md` §21, F3.)

**Ask (minimal).**
1. A way to run the model **without the main source** while the local units form the grid:
   either a grid-forming mode for the diesel/battery units (for example a `Vsource`-equivalent or
   isochronous generator on one unit per island) or a second `.dss` model for the islanded case.
2. Real **generation limits** in that mode: PV output scaled by a factor, diesel available only up
   to a factor of rating, battery at rating. An island whose demand exceeds its generation should
   fail verification (non-convergence or under-voltage) instead of being absorbed.
3. A **scenario field** on `POST /api/module4/verify` — `{tripped_lines: [...], grid_lost: bool,
   pv_factor, diesel_factor}` — so M3 stops encoding storm damage as fake breaker/dispatch commands.

**Acceptance tests (M3 will run these).**
* Grid lost, PV 0, diesel 0.5, **no shedding** → REJECT (shortfall ≈ 200 kW).
* Same, shedding N8 and N9 fully and ~22% of N12 → APPROVE.
* Grid lost, plentiful supply, no shedding → APPROVE.
* An island with no generator → REJECT unless fully shed and floating.

**How M3 will use it.** `LiveM4Twin` stops raising on `grid_lost`; the final live test then
covers the shortage storms, and the surrogate's islanded model is validated against it with
`scripts/validate_surrogate.py`.

**Related small fixes found on the way (separate, lower priority).** F2 — `set_load_shed("N1")`
also sheds N11 and N12 (substring match); F2b — some multi-trip states are approved although a bus
with load is dead; F5 — `verify()` recompiles the circuit each call (~13 ms; the solve is ~0.2 ms).

---

# Part IV — Training stability

## 25. Why some seeds failed, and the fix (2026-10-17)

### 25.1 What was going wrong
Across seeds the same code gave very different gates (e.g. 62 to 74 return per storm). The
training logs showed the failing seeds did **not** just learn slowly — they **collapsed into a
fixed habit**: the policy's randomness (its *entropy*) fell to ~0 and it settled on one behaviour
for every step — *always* System 2 (S2 rate pinned at the 0.56 budget cap) or *never* System 2
(S2 rate 0.00). After that PPO had nothing left to learn from, because a policy that never varies
produces no useful comparison between choices.

> Correction: an earlier version of §22 said the failed seeds of the first run "never left System
> 1". Seed 2 did, but seed 3 collapsed the other way (always System 2). The training logs, not the
> evaluation scores, are what show this.

### 25.2 The cause we found
The **critic** (the network part that predicts "how much reward is still to come") has to output
numbers around 50–70, because an 80-step storm adds up to that much reward. Its error is therefore
huge compared with the policy's learning signal, and both share the same network and the same
gradient clip (0.5). The critic's loss swamped the policy's, so the policy barely moved while the
critic caught up and then drifted into a fixed habit.

### 25.3 The fix: standardise the critic's target
`ppo.value_norm: true` (code: `train/ppo.py::ppo_update`, `policy/recurrent_gate.py::to_return_units`).
The critic is trained to predict `(return − mean) / std` of the returns seen in the first batch, so
its loss has unit scale and no longer drowns the policy. Rollouts convert its output back to real
reward units before computing advantages. The mean/std are stored with the weights.

### 25.4 How we chose it (a controlled sweep)
`scripts/stability_sweep.py` trains each recipe on 5 seeds and scores it on **validation storms that
are disjoint from the final-report storms**, so picking a recipe does not leak into the reported
numbers. Shortage world (`p_grid_loss = 0.6`), return per storm:

| recipe | mean ± std | worst seed | seeds beating the visible-outage rule | `ditwah_like` | train time / job |
|---|---|---|---|---|---|
| A — previous setup | 69.20 ± 6.47 | 60.7 | 3 / 5 | 65.7 | 431 s |
| **B — standardised critic** | **73.48 ± 1.34** | **71.4** | **5 / 5** | **72.3** | 436 s |
| C — B + lr 3e-4, batch 32, entropy 0.02→0.002, KL stop, critic warm-up | 73.40 ± 1.30 | 71.3 | 5 / 5 | 71.8 | 993 s |
| D — C + warm start (copy the rule first) | 73.45 ± 1.30 | 71.4 | 5 / 5 | 71.8 | 1008 s |
| *warm start alone (no PPO)* | *65.31 ± 3.33* | *61.0* | – | *59.9* | – |

Same storms for reference: always-S1 61.15 · `u`-threshold 66.00 · visible-outage rule 71.32 ·
oracle 73.68.

* **B alone fixes it.** Seed-to-seed spread falls ~5× and every seed beats the rule.
* **C and D add nothing** over B and cost 2.3× the compute, so **B is the default**. The other
  switches stay in the code (all off by default) in case a harder setting needs them.
* **The warm start is not a crutch.** Copying the rule alone scores 65.3 — far below the trained
  gate — and warm-starting does not change the final score, so what the gate learned came from PPO.
* The oracle is not a strict ceiling: it only escalates when it sees damage, so a gate that
  anticipates can edge past it (73.48 vs 73.68 here, 72.3 vs 72.1 on `ditwah_like`).

### 25.5 Caveats
* 5 seeds and 40 validation storms per seed: a clear effect (std 6.5 → 1.3, worst 60.7 → 71.4),
  but not a basis for fine rankings between B, C and D.
* The sweep ran on the surrogate twin only; nothing here is verified by the real M4.
* Hyper-parameters other than the ones in the table were not explored.

### 25.6 Final results with the fixed recipe
Recipe B (`value_norm: true`) as the default, 5 seeds, **fresh test storms** (not the validation
storms used to choose the recipe). Return per storm, mean ± std across seeds, identical storms for
every policy. Surrogate twin only — **not verified by the real M4**.

**Shortage world** (`--p-grid-loss 0.6`, with the memoryless ablation; 35 min):

| policy | fresh storms | held-out `ditwah_like` | held-out `burevi_like` |
|---|---|---|---|
| always S1 | 61.19 ± 0.86 | 54.05 ± 3.89 | 64.24 ± 4.53 |
| `u`-threshold | 65.28 ± 1.40 | 63.72 ± 0.84 | 66.11 ± 3.50 |
| visible-outage rule | 71.02 ± 1.21 | 65.14 ± 0.96 | 73.00 ± 0.33 |
| oracle (cheats) | 73.49 ± 0.85 | 71.17 ± 0.58 | 74.11 ± 0.56 |
| **gate (GRU)** | **73.32 ± 0.92** | **71.33 ± 0.75** | **73.53 ± 0.69** |
| gate, memoryless (MLP) | 73.28 ± 1.27 | 71.32 ± 0.83 | 73.91 ± 0.84 |

**Original world** (`--p-grid-loss 0`; 21 min):

| policy | fresh storms | held-out `ditwah_like` | held-out `burevi_like` |
|---|---|---|---|
| always S1 | 68.98 ± 1.23 | 64.12 ± 0.82 | 73.49 ± 0.90 |
| visible-outage rule | 72.26 ± 0.80 | 67.59 ± 0.62 | 75.03 ± 0.64 |
| oracle (cheats) | 75.03 ± 0.46 | 73.89 ± 0.23 | 76.80 ± 0.43 |
| **gate (GRU)** | **74.57 ± 0.40** | **73.54 ± 0.45** | **75.90 ± 0.34** |

Per-seed gate vs the rule on fresh storms (the stability check):

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| shortage: gate | 73.8 | 74.6 | 73.1 | 72.1 | 73.0 |
| shortage: rule | 71.4 | 72.5 | 70.7 | 69.2 | 71.3 |
| original: gate | 74.9 | 74.5 | 73.9 | 74.6 | 74.9 |
| original: rule | 72.3 | 72.8 | 71.3 | 71.6 | 73.2 |

**Reading.**
* **Stability is fixed.** All 5 seeds beat the rule on fresh storms in both worlds (before: 3 of 5
  and 2 of 5). Seed-to-seed std fell from about 5.5 to about 0.9 (shortage) and 0.4 (original).
* **The gate reaches the oracle on fresh storms** (73.32 vs 73.49; 74.57 vs 75.03) and edges past
  it on `ditwah_like` (the oracle is not a strict ceiling, see §25.4). It beats the rule by about
  6 points on held-out `ditwah_like` in both worlds.
* **It wastes far less System 2:** on steps where the grid is intact it escalates 9% of the time
  (before the fix: 38%); overall 38% vs the oracle's 34%.
* **Weak spot:** on `burevi_like` the gain over the rule is small (73.53 vs 73.00 in the shortage
  world — within noise; 75.90 vs 75.03 in the original world), and it escalates more than needed
  there (26% of intact-grid steps in the shortage world).
* **The GRU's memory is not shown to help.** The memoryless gate scores the same (73.28 vs 73.32;
  71.32 vs 71.33 on `ditwah_like`; slightly *better* on `burevi_like`). The gate is given the
  outage status directly (observation slots 12–14), so the task is nearly memoryless. Do **not**
  claim learned meta-adaptation from these results. A task where the storm's severity or type is
  hidden and must be inferred from history, or `policy.trials_per_task > 1`, is needed to test the
  RL² claim.
* **Still unverified:** nothing here has been run against the real M4 for the shortage world (it
  cannot represent it), and the Burevi/Ditwah profiles are placeholders.

*These numbers supersede §22 and §23.6, which were produced before the fix.*

---

# Part V — Does the GRU's memory help?

## 26. Memory test: hiding the damage from the gate (2026-10-19)

### 26.1 Why this test, and a flaw it exposed
In §25.6 the memoryless gate (MLP) matched the GRU. That is expected when the gate is *told* what
broke (observation slots 12–14), because there is nothing left to remember. Memory can only matter
when the gate must **infer** the storm's state from what it has already seen.

Flaw found while designing it: the "memoryless" MLP was not memoryless. Both architectures also
receive the **previous action and previous reward** (the RL² input), and the previous reward leaks
the storm's damage. So the MLP had one step of memory. Fix: `policy.feedback: false` zeroes the
previous action/reward (`policy/recurrent_gate.py::RecurrentGate.forward`). The `_nofb` arms below
are the strictly memoryless controls.

### 26.2 Setup (code: `scripts/memory_test.py`, `env.hide_obs`, `policy.feedback`)
Shortage world (`p_grid_loss = 0.6`), 5 seeds per arm, scored on validation storms (offset
500_000_000, disjoint from the final-report storms). `env.hide_obs` zeroes the named observation
slots (default: none, so earlier results reproduce).

| world | hidden slots | what the gate cannot see |
|---|---|---|
| visible | – | nothing (the §25 setup) |
| blind | 12, 13, 14 | which feeders tripped, generation left, grid lost |
| very blind | 8, 9, 12, 13, 14 | also severity and time to hazard |

### 26.3 Results (return per storm, mean ± std over 5 seeds)
| arm | mean | std | worst | S2 rate | S2 when grid intact |
|---|---|---|---|---|---|
| visible GRU | 73.48 | 1.34 | 71.40 | 0.36 | 0.07 |
| visible MLP | 73.42 | 1.15 | 71.74 | 0.36 | 0.08 |
| blind GRU | 72.71 | 1.00 | 71.04 | 0.36 | 0.11 |
| blind MLP | 72.22 | 0.95 | 71.00 | 0.45 | 0.25 |
| blind GRU, no feedback | 72.78 | 1.06 | 71.08 | 0.37 | 0.12 |
| blind MLP, no feedback | 72.43 | 1.30 | 70.49 | 0.49 | 0.29 |
| very blind GRU | 72.15 | 1.61 | 69.65 | 0.26 | 0.04 |
| very blind MLP | 61.15 | 2.52 | 57.03 | 0.00 | 0.00 |
| very blind MLP, no feedback | 61.15 | 2.52 | 57.03 | 0.00 | 0.00 |

Reference on the same storms: always-S1 61.15 · `u`-threshold 66.00 · visible-outage rule 71.32 ·
oracle 73.68. **The outage rule is only valid in the visible world:** it reads slots 8/12/14, so
with those hidden it degrades to always-S1 (61.15). Compare the blind gates to 71.32.

Paired by seed (same storms), memory arm minus control:

| comparison | mean diff | seeds GRU wins | per-seed |
|---|---|---|---|
| visible | +0.06 | 2/5 | −0.2 −0.3 +0.1 +1.0 −0.3 |
| blind (both get feedback) | +0.48 | 5/5 | +0.2 0.0 +0.7 +0.1 +1.3 |
| blind: GRU vs strictly memoryless | +0.28 | 4/5 | +0.3 +0.6 −0.5 +0.1 +1.0 |
| blind, no feedback either side | +0.35 | 4/5 | +0.3 +0.6 −0.4 +0.6 +0.6 |
| very blind (both get feedback) | +10.99 | 5/5 | +9.7 +12.6 +8.6 +11.8 +12.3 |
| very blind: GRU vs strictly memoryless | +10.99 | 5/5 | same |

### 26.4 Reading
* **Visible: memory is irrelevant**, as expected.
* **Blind: a small, consistent edge** (+0.3 to +0.5, under half a seed-to-seed std). The GRU also
  wastes fewer System 2 calls on an intact grid (11% vs 25–29%). Real but modest; not a basis for
  a headline claim.
* **Very blind: a large gap, but mostly a training failure of the memoryless gate.** The MLP scored
  exactly 61.15 on all 5 seeds, identical to always-S1, with S2 rate 0.00: it learned never to
  escalate. The GRU learned a working policy (72.15). That shows the GRU's hidden state carries
  useful information, but the +11 is not "11 points of inference".
* **Open:** the GRU may simply be keeping a *clock* (counting steps) rather than inferring damage.
  Not yet checked.
* **Not shown:** that the gate adapts to an unseen storm *within* an episode. That is the actual
  RL² claim and needs repeated trials of the same storm (`policy.trials_per_task > 1`) — next test.

### 26.5 Caveats
5 seeds; surrogate twin only; validation storms; hyper-parameters unchanged from §25.

---

# Part VI — Latency, deadline enforcement and gate tuning

## 27. Latency profiling, the deadline guard, and gate tuning (2026-10-19)

ClickUp task: *Latency and thermal-budget profiling and enforcement; robust gate tuning
(hysteresis, calibrated escalation threshold)*. Three parts: **measure** (§27.1), **enforce**
(§27.2), **tune** (§27.3). The *thermal* half is not done — see §27.4.

### 27.1 Measured latency (code: `scripts/profile_latency.py`)
One control step = gate forward → System 1 or System 2 plan → verification. Each stage was timed
with `perf_counter` over simulated storms (single CPU thread, GRU hidden 64). Milliseconds.

| stage | p50 | p95 | p99 | max |
|---|---|---|---|---|
| gate forward (GRU, batch 1) | 0.29 | 0.51 | 0.69 | 33.4 (first-call outlier) |
| System 1 plan | 0.012 | 0.019 | 0.029 | 0.10 |
| System 2 plan (greedy planner) | 0.021 | 0.068 | 0.101 | 0.43 |
| surrogate twin evaluate | 0.024 | 0.044 | 0.064 | 0.13 |
| **real M4 verify, end to end (gateway + HTTP + recompile)** | **30.2** | **46.1** | **48.2** | **65.4** |

System 2 planning by situation (p50 / p99 ms): intact 0.021 / 0.039 · feeder out 0.022 / 0.056 ·
grid lost 0.075 / 0.136.

**What this says**
* **Our side is negligible.** Gate + System 1/2 plan together are under 1 ms at p99. The decision
  is dominated by M4: median 30 ms, p99 48 ms, and the slowest call (65 ms) **exceeds the 50 ms
  real-time budget**. (Earlier, §21 F5 measured ~13 ms; this run is slower because every call
  carries the storm overlay commands and the circuit is recompiled each time. Machine load was
  not controlled, so treat the absolute figure as indicative.)
* **System 1 vs System 2 is not a latency trade-off today.** The greedy System 2 planner costs
  ~0.01 ms more than System 1. Both paths then pay the *same* M4 verify cost.
* **The reward's latency constants are invented and wrong.** `reward.latency_s1_ms = 2` and
  `latency_s2_ms = 18` are stand-ins; measured planning times are ~0.01 and ~0.02 ms. Today that
  term (`w_latency · Δms / 50 = 0.016` per System 2 step) is really just a hidden extra
  deliberation cost. It has NOT been changed, because changing it would invalidate the
  results in §25–26; it is relabelled instead (§27.5).
* **MDP vs SMDP (§2.1) is settled for now:** System 2 takes ≈0.02 ms against a control period
  of tens of ms, so the fixed-step MDP stands. This must be re-measured if System 2 becomes an
  optimiser or runs OpenDSS in the loop.

### 27.2 Enforcement: `LatencyGuard` (code: `src/module3/serve/guard.py`)
A late plan is a wrong plan in real time. The guard wraps the System 2 call:
1. **Deadline** — if System 2's plan takes longer than `s2_deadline_ms` (default 5 ms), its
   answer is discarded and the System 1 plan is used for that step.
2. **Circuit breaker** — after `trip_after` (3) overruns in a row, System 2 is switched off for
   `cooldown_steps` (10) steps and then retried. This protects the *following* steps; a Python
   call cannot be interrupted half-way, so the guard bounds the damage rather than pre-empting it.
3. **Time budget** — System 2 is refused up front if the time already spent this step leaves less
   than `s2_deadline_ms` before `step_deadline_ms` (50 ms).

It uses an injectable clock, so the 7 tests in `tests/test_latency_guard.py` are exact (deadline,
breaker open/close, streak reset, time-budget refusal, summary counts).
**Limit:** the guard protects against a slow *System 2*. It cannot speed up M4, which is where
the real tail (§27.1) is — that is an M4 fix (F5).
**Not yet wired** into the serving path (the Go `gating_decision_svc` is still a stub); it is a
tested component, not an integrated one.

### 27.3 Gate tuning: calibrated threshold and hysteresis (code: `scripts/tune_gate.py`, `train/evaluate.py::GateTuning`)
No retraining: the saved §25.6 gates (shortage world, 5 seeds) output P(System 2). Per seed, a
grid of rules was searched on **validation** storms, then the winner scored on the final-report
storms. `flips` = System 1 ↔ System 2 switches per storm.

| rule | fresh storms: return | flips / storm | S2 rate | S2 on intact grid |
|---|---|---|---|---|
| greedy (P > 0.5) | 73.32 ± 0.92 | 1.1 ± 0.2 | 0.38 | 0.09 |
| calibrated threshold | 73.37 ± 0.91 | 1.1 ± 0.2 | 0.38 | 0.08 |
| hysteresis (on/off thresholds + hold) | 73.36 ± 0.90 | 1.0 ± 0.2 | 0.37 | 0.07 |
| *held-out `ditwah_like`: greedy / calibrated / hysteresis* | 71.33 / 71.33 / 71.30 | 1.0 / 1.0 / 1.0 | | |

**Reading: tuning changes nothing.** The differences (+0.05) are far inside the seed-to-seed
spread (0.9). The gate already switches about once per storm (on at the outage, rarely off), so
there is no chatter for hysteresis to remove, and P = 0.5 is already about the best threshold.
The chosen thresholds vary by seed (0.4–0.7) without a pattern, which is what you see when the
return surface is flat. **Decision: keep plain greedy as the default**; `GateTuning` stays as an
opt-in option, to be revisited if a harder setting (noisier telemetry, a real System 2) makes the
gate chatter. The greedy row reproduces the §25.6 numbers exactly, a consistency check on the
evaluation.

### 27.4 What is NOT done: "thermal"
The task says *thermal-budget*, and the repo does not define it. Two readings:
* **Line thermal loading** (current ≤ ampacity; M4 reports `THERMAL_OVERLOAD`). The surrogate has
  no thermal model (F8), so System 2's dispatch is never checked against line limits during
  training. Doing this properly needs a flow estimate per line, or the real M4 in the loop.
* **Compute heat** on the edge device (duty-cycle limit on System 2). Not measurable on this
  machine.
Nothing was built for either; the choice is the owner's (§27.5).

### 27.5 Decisions
1. **Reward latency constants — DECIDED (2026-10-19): relabel, keep the values.** `latency_s1_ms: 2`
   and `latency_s2_ms: 18` stay, so §25–26 remain valid, but they are now documented (in
   `pipeline.yaml` and `env/reward.py`) as a second path-dependent *deliberation cost* (S2 pays
   0.018 per step, S1 0.002), **not** measured milliseconds. No code behaviour changed. The honest
   caveat: the gate is therefore penalised for System 2 twice (0.05 + 0.018 per step), and any
   statement that the gate "trades latency for safety" is not supported by this reward.
2. **Thermal — OPEN:** which reading of §27.4, and is a surrogate approximation enough for PP1?
3. **M4 tail latency — DECIDED: raise with the M4 owner.** Message below (§27.6). Not yet sent.

### 27.6 DRAFT note to M4 — end-to-end `verify()` tail latency  *(not sent)*
**To:** Hariswara (M4) · **From:** Saabir (M3) · **Status:** draft, for Saabir to send alongside §24.

**What M3 measured** (`scripts/profile_latency.py --live-m4`, real calls through the gateway
`POST /api/module4/verify`, single machine, load not controlled; storm-overlay commands included,
no `grid_lost` cases):

| p50 | p95 | p99 | max |
|---|---|---|---|
| 30.2 ms | 46.1 ms | 48.2 ms | 65.4 ms |

**Why it matters.** The M4 real-time budget is 50 ms. The p99 is at the limit and the slowest calls
are over it. M3's own work (gate + System 1/2 plan) is under 1 ms at p99, so M4 is effectively the
entire decision latency.

**Likely cause** (same as F5): every call recompiles the circuit (`reset_to_base`); the power-flow
solve itself is ~0.2 ms. The `solve_latency_ms` M4 reports therefore understates the cost the
caller sees.

**Ask.** (1) Compile once and snapshot/restore (or re-apply only the changed elements) per call.
(2) Report end-to-end handler latency, not only the solve time, in the response.
(3) If possible, a target for p99 through the gateway so M3 can set its guard deadline
(`LatencyGuard.step_deadline_ms`, currently 50) from a real number.

**Caveat.** The earlier §21 F5 figure was ~13 ms; the difference is not yet explained (storm
overlay commands, load on the machine). A re-measurement on a quiet machine would settle it.

---

# Part VII — Repeated trials

## 28. Repeated-trials test: does the GRU improve the more it sees of a storm? (2026-10-19)

ClickUp task: Meta-RL gate (z8py7aaz6y). Code: `scripts/trials_test.py`,
`train/evaluate.py::run_trials`, `policy.replay` in `train/ppo.py::collect_tasks`.
Raw results: `artifacts/trials_test/latest.json` (git-ignored).

**Question.** §26 showed the GRU's hidden state carries information *within* a storm. The RL² claim
is stronger: given several trials of the *same* task and memory kept across them, the policy gets
better by trial K. A memoryless gate cannot improve across trials, so it is the control.

**Setup.** Blind world (SCADA slots 12–14 hidden, 60% of storms lose the main supply). Each storm
is played 4 times in a row; the GRU keeps hidden state, previous action and previous reward across
trials (the first step of a later trial sees `trial_done = 1`). Both gates are *trained* the same
way (4 trials per task). Control: MLP with no previous-action/reward feedback (strictly
memoryless). Two kinds of repetition: `fresh` (same storm settings, damage redrawn each trial) and
`exact` (the identical storm replayed). 5 seeds × 2 architectures × 2 modes, 40 evaluation storms
per run on the validation storm range, greedy actions.

**Result** (mean return per trial index; always-S1 and the oracle are scored on the same trials):

| mode | arm | trial 1 | trial 2 | trial 3 | trial 4 | last − first |
|---|---|---|---|---|---|---|
| fresh | GRU | 68.25 | 68.75 | 68.59 | 68.62 | +0.37 ± 6.14 |
| fresh | MLP, no feedback | 71.07 | 71.15 | 71.36 | 71.25 | +0.18 ± 0.54 |
| fresh | always-S1 / oracle | 61.08 | 59.59 | 59.86 | 60.96 | oracle ≈ 73.8 |
| exact | GRU | 69.45 | 69.81 | 70.36 | 67.96 | −1.49 ± 1.43 |
| exact | MLP, no feedback | 70.90 | 70.90 | 70.90 | 70.90 | 0 (by construction) |
| exact | always-S1 / oracle | 62.44 | 62.44 | 62.44 | 62.44 | oracle 73.99 |

GRU improvement minus MLP improvement: +0.19 (fresh; GRU larger on 3/5 seeds), −1.49 (exact; 0/5).

**Reading: no evidence of within-storm meta-learning. Negative result.**
* Fresh damage: the average gain is +0.4 with a seed-to-seed spread of 6.1, i.e. noise. Per seed the
  GRU improved on 3 seeds (e.g. 65.0 → 71.9) and got *worse* on 2 (66.9 → 61.0; 72.1 → 66.0).
* Identical replay is the easiest case for learning a storm (the same damage every time), and the
  GRU did not exploit it: it rose on trials 1–3 on average, then fell at trial 4 (trial 4 was the
  lowest of the four on 4 of 5 seeds). One seed was flat and one sat at 64 from the start.
* **The GRU's behaviour drifts with trial number.** On several seeds its System 2 rate climbs
  steadily across trials (e.g. 0.16 → 0.30 → 0.54 → 0.56) while the storm is identical. A policy
  whose actions change with no new information about the storm is behaving like a *trial counter*
  (a clock), not a learner. That was the unchecked suspicion listed after §26; this is the first
  evidence for it, but it is not a direct test.
* The memoryless gate is simply **better and steadier here** (71.07 vs 68.25 on trial 1, fresh).
  Training the GRU over 320-step sequences is harder and less stable than the MLP over 80.

**What this does and does not say.** It does not show the GRU is useless: §26 stands (its memory
helps when information is hidden *within* a storm). It shows that, in this environment, a storm
repeated 4 times gives the GRU nothing it learns to use. Likely reasons, not yet tested: with
`fresh` damage there is little shared across trials beyond the storm settings, which the gate can
mostly infer from what it sees; with `exact` there is something to memorise, but PPO with 120
updates and a 4× longer horizon may simply not find it.

**Consequences for the claim.** Do not describe the M3 gate as "meta-learning within a storm" in
the PP1 deliverable. Defensible wording: *a recurrent gate that uses history to infer hidden storm
state (§26)*; RL²-style adaptation across repeated trials was tested and **not** observed (§28).

**Possible follow-ups (not done):** a direct clock test (feed the trained GRU a constant storm
state and read how its output depends on step index); more updates or a curriculum from 1 to 4
trials; a task latent that is hidden, persistent and decision-relevant (e.g. which feeder is
fragile), which is what RL² needs to have something to infer.

---

# Part VIII — Dashboard

## 29. Cyclone lab: set the storm's parameters and test the gate (2026-10-19)

ClickUp: dashboard gating-decision timeline (z8py7aaz79) — partly covered, see "Not covered".
Page: `apps/dashboard/src/routes/cyclone/` (route `/cyclone`; linked from `/gating`).

**What it does.** Sets one storm, plays it through a trained gate and the five reference policies,
and shows the result. The storm does not react to the controller, so every policy faces the
identical storm and differences in return come only from the decisions.

* **Parameters (7 sliders + 1 switch):** peak wind, peak timing, storm duration, line fragility
  (`trip_scale`), comms-loss rate, diesel-loss probability, main-supply-cut probability, and
  "unseen conditions" (`ood`). Presets: default, `burevi_like`, `ditwah_like` (placeholder
  parameter sets, as in §10), plus a **Random storm** drawn from the training distribution. Plus
  the storm seed and which of the five trained gates (training seeds 0–4) to use.
* **Timeline** (shared x-axis): wind with the elevated/severe/extreme lines; damage (feeders
  actually out, main supply lost, telemetry down); the gate's P(System 2) with the 0.5 line;
  per-step reward. Steps where the gate chose System 2 are shaded across all lanes. Hovering a
  step shows its wind, damage, M2 `u` and reason, P(S2), decision, reward, load served and the
  twin's verdict.
* **Comparison table:** return, System 2 share, load served and twin-reject rate for the gate,
  always-S1, always-S2, `u > threshold`, the outage rule and the oracle, on that storm.

**Plumbing.** `module3.serve.storm.simulate_storm` (validated inputs: an unknown knob or an
out-of-range value is rejected, never ignored) ← `scripts/storm_run.py` ← gateway
`POST /api/module3/storm` and `GET /api/module3/storm/presets` (same subprocess pattern as the
older `/run`) ← the page. Tests: `tests/test_storm.py` (12) and three in
`services/gateway/tests/test_module3_router.py`.

**Checked** by running the page in a browser against a live gateway: presets load, a Ditwah-like
storm runs, the timeline renders and the hover detail works.
Example (storm seed 1, `ditwah_like`, gate seed 0): gate 75.30 vs outage rule 72.18, always-S1
69.91, always-S2 67.98, oracle 75.15, with one S1→S2 switch just as the first feeders trip.
**One storm is one sample**; the averaged evidence is §25.6, and the page says so.

**Limits and honesty notes.**
* The grid is the **surrogate twin**, not the live M4 firewall (the live twin refuses `grid_lost`,
  §24 is unsent).
* The oracle is a reference that reads the true damage, **not an upper bound**: the gate beats it
  on the example storm. The page labels it that way.
* **The trained checkpoints are not in git** (`artifacts/` is ignored, ~110 KB each). On a fresh
  clone the page answers "no trained gate … run the pipeline first" (HTTP 409). Decision for the
  team: commit the five `final_islanded` checkpoints, or publish them with the results.
* The presets request takes a few seconds because it starts a Python process; the knobs appear
  after it returns.
* Like the older `/run` endpoint it needs the gateway to run in the shared workspace venv (torch),
  not the gateway's slim Docker image.

**Not covered (still open on the timeline task):** the *deliberation cost incurred* is only
visible through the reward lane (no separate cost trace); no live-M4 mode; no dashboard control
for the gate's tuning options (§27.3) or the latency guard.

---

# Part IX — PP1 deliverable

## 30. PP1 pack: what M3 delivers, the numbers, the demo, and what we do not claim (2026-10-19)

ClickUp: *PP1 deliverable: working gating engine + System 2 policy at ~50% with metrics and live
demo* (z8py7aaz73). The roadmap's PP1 line is "four core engines with measured metrics;
forward-direction contracts frozen and live". This section is the M3 evidence for that line, written
so that every number points at the section that produced it.

### 30.1 PP1 checklist: criterion → evidence → status

| PP1 needs | Evidence | Status |
|---|---|---|
| A working gating engine | PPO + GRU gate trained on simulated cyclones with four disturbances and a priority-weighted physical reward (Part II, §14–§20); 5 seeds, stable after the critic fix (§25) | **Done** |
| A System 2 policy | Critical-load ranking + shortage-aware shedding planner, with an islanded model in the surrogate (§23) | **Done in the surrogate.** Not testable on the real M4 until M4 can represent a lost main supply (§24, **unsent**) |
| Measured metrics | Return, load served, reject rate, System 2 share, S2 use by damage state, vs five baselines, 5 seeds, fresh and held-out storms (§25.6, table below) | **Done** |
| Measured on the real firewall | All five final gates scored on the real M4 and on the surrogate, same storms (§30.2 below) | **Done** for storms without grid loss |
| Latency measured | Our side < 1 ms p99; real M4 p50 30 ms, p99 48 ms, max 65 ms vs 50 ms budget (§27.1) | **Done**; M4's tail is over budget (note to M4 drafted, §27.6, unsent) |
| Live demo | Dashboard **Cyclone lab** (§29) + runbook in §30.3 | **Done**, runs offline on the surrogate |
| Forward contracts frozen and live | M3 → M4 contract `m3-out/0.1` is defined and used by the older demo (§9); the Meta-RL gate talks to M4 over the HTTP verify endpoint through the gateway | **Partial.** The Go `gating_decision_svc` is still a stub; M2 → M3 still uses a mock stream (ClickUp z8py7aaz6z, blocked on M2) |

### 30.2 Headline metrics

Return per 80-step storm, higher is better; mean ± std over 5 independent training seeds;
identical storms for every policy; fresh storms not used to pick the recipe. **Surrogate twin.**

| world | policy | fresh storms | held-out `ditwah_like` | held-out `burevi_like` |
|---|---|---|---|---|
| shortage (60% lose main supply) | always System 1 | 61.19 ± 0.86 | 54.05 ± 3.89 | 64.24 ± 4.53 |
| | `u` threshold | 65.28 ± 1.40 | 63.72 ± 0.84 | 66.11 ± 3.50 |
| | visible-outage rule | 71.02 ± 1.21 | 65.14 ± 0.96 | 73.00 ± 0.33 |
| | **gate** | **73.32 ± 0.92** | **71.33 ± 0.75** | **73.53 ± 0.69** |
| | oracle (reads the true damage) | 73.49 ± 0.85 | 71.17 ± 0.58 | 74.11 ± 0.56 |
| original (no main-supply loss) | visible-outage rule | 72.26 ± 0.80 | 67.59 ± 0.62 | 75.03 ± 0.64 |
| | **gate** | **74.57 ± 0.40** | **73.54 ± 0.45** | **75.90 ± 0.34** |

The gate beats the best rule on fresh storms in both worlds on **all five seeds**, by about 2 points
on fresh storms and about 6 on held-out `ditwah_like`. It escalates to System 2 on 9% of
intact-grid steps (38% before the fix). Full tables: §25.6.

**Real-M4 check (new, `scripts/live_check.py`, `artifacts/live_check/latest.json`).** The five final
gates and the baselines were scored on the real M4 firewall and on the surrogate, same storms (8
random + 4 per held-out profile per seed; storms **without** grid loss, which the real M4 cannot
represent; about 39,800 real verify calls):

| suite | policy | real M4 | surrogate | gap | largest per-seed gap | real-M4 reject rate |
|---|---|---|---|---|---|---|
| random | always System 1 | 68.84 | 68.84 | +0.000 | 0.000 | 0.368 |
| random | visible-outage rule | 72.10 | 71.91 | +0.189 | 0.491 | 0.196 |
| random | **gate** | **74.61** | 74.42 | +0.189 | 0.491 | 0.055 |
| random | oracle | 74.85 | 74.66 | +0.189 | 0.491 | 0.044 |
| `burevi_like` | rule / **gate** / oracle | 75.26 / **75.87** / 76.96 | 75.20 / 75.81 / 76.90 | +0.060 | 0.301 | gate 0.017 |
| `ditwah_like` | rule / **gate** / oracle | 67.83 / **73.76** / 73.92 | identical | 0.000 | 0.000 | gate 0.036 |

(Other baselines: `results` file.) **Reading:** the surrogate we train on agrees with the real
firewall to within 0.5 return on any seed (about 0.2 on average), the policy ranking is the same on
both, and on the real M4 the gate again sits next to the oracle and about 2.5 (random) / 6
(`ditwah_like`) above the rule. The real M4 scores slightly *higher* than the surrogate where they
differ, so the surrogate is the more pessimistic of the two. Caveats: storms with grid loss are not
covered (the real M4 cannot represent them); each cell is 8 or 4 storms per seed.

**Latency (§27.1):** gate + System 1/2 planning < 1 ms at p99; real M4 verify p50 30 ms / p99 48 ms /
max 65 ms against the 50 ms budget. **Tests:** 89 in the M3 package (stability, memory knobs, guard,
storm service, repeated trials) plus the gateway wiring tests.

### 30.3 Live demo runbook (all steps verified on this machine)

Needs the trained gates in `artifacts/pipeline/final_islanded/` (**not in git**; see §30.5).

```bash
# terminal 1 — gateway, from services/gateway, workspace venv (needs torch)
../../.venv/Scripts/python.exe -m uvicorn gateway.main:app --port 8000
# terminal 2 — dashboard, from apps/dashboard
node_modules/.bin/vite --port 5173        # open http://localhost:5173/cyclone
```

A five-minute story that matches the evidence:
1. **The problem (30 s).** A cyclone damages the grid. Deliberating (System 2) costs compute and
   budget; reacting (System 1) is cheap but can be wrong. The gate decides which, step by step.
2. **Mild storm (60 s).** Peak wind 40 km/h, seed 1. The gate stays on System 1 throughout (return 79.64);
   *always System 2* scores 76.79: it spends budget for nothing.
3. **`ditwah_like` preset (90 s).** Hover the step where the first feeders trip: the gate's P(S2)
   jumps from ~0 to ~1 within a few steps of the damage, and System 2 stays on. Show the table:
   gate 75.3 vs outage rule 72.2 vs always-S1 69.9 (storm seed 1, gate 0).
4. **Main supply cut (60 s).** Start from `ditwah_like`, raise "Main supply cut" to 1.0 and run (seed 1): gate
   74.33, oracle 74.28, outage rule 71.12, always-S1 39.47. The planner sheds the lowest-priority
   loads first (§23). Say plainly this is the surrogate twin.
5. **Random storm (30 s).** Click it a few times. Say: one storm is one sample, the averages are
   the table in §30.2.
6. **Close (30 s).** Real-M4 agreement and latency numbers from §30.2.

### 30.4 What we claim, and what we do not

**Claim (supported):**
* A learned gate that escalates to System 2 almost exactly when real damage exists, matching an
  oracle that reads the true damage and beating the best visible-outage rule on fresh and held-out
  storms, on all five seeds (§25.6, §30.2).
* The surrogate we train on agrees with the real M4 firewall (§30.2, §17).
* Our decision path costs under 1 ms; M4 dominates latency (§27.1).

**Do not claim:**
* **Meta-learning / memory.** The memoryless gate scores the same (§25.6). The GRU's memory helps a
  little when damage is hidden within a storm (§26), and it did **not** improve across repeated
  plays of the same storm (§28). Say "a recurrent gate that uses history to infer hidden state".
* **Survivability on Burevi or Ditwah.** Both profiles are placeholder parameter sets (§10, §21).
* **Real-M4 verification of the shortage scenario.** M4 cannot represent a lost main supply yet (§24).
* **That latency is a trade-off the gate manages.** System 2 plans in ~0.02 ms; the reward's
  "latency" term is a relabelled deliberation cost, not measured time (§27.5).
* **A thermal budget.** Not defined, not built (§27.4).

### 30.5 Open items going into PP1 (who has to act)

| Item | Owner | Why it matters for PP1 |
|---|---|---|
| Send the "main supply lost" change request (§24) | Saabir → Hariswara | Without it System 2's shortage logic stays surrogate-only |
| Send the M4 tail-latency note (§27.6) | Saabir → Hariswara | M4 p99 / max exceed the 50 ms budget |
| Define "thermal" (§27.4) | Saabir | Half of the latency/thermal task is unbuilt |
| Commit or publish the five gate checkpoints (~110 KB each) | Saabir | The demo and the Cyclone lab page need them; a fresh clone has none |
| Add a "PR review by dev" ClickUp status; open the PR into `main` (do not merge) | Saabir | The Meta-RL work is on `m3/gate-policy` only |
| Swap the mock M2 stream for the real one | M2 owner (z8py7aaz6z) | PP1 asks for live forward contracts |
| Replace the placeholder Burevi/Ditwah profiles with real data | Saabir | Needed before any survivability claim (PP2) |
| Wire the latency guard into serving; the Go `gating_decision_svc` is a stub | Saabir | Guard is tested but not integrated |

### 30.6 Questions a panel is likely to ask, and the honest answer

* **"Is the gate just a rule?"** A rule on visible outages gets 71.0 and the gate 73.3 (shortage
  world, fresh storms); on held-out `ditwah_like` the gap is 6 points. It also wastes little
  System 2: it escalates on 9% of intact-grid steps (38% before the fix).
* **"Why is it as good as the oracle?"** The gate is *given* outage status in the observation
  (slots 12–14), so this is not a hard inference problem; the oracle is a rule, not an optimum, and
  the gate can edge past it (§25.4). Hiding those slots costs a little (§26).
* **"Did you train on the same storms you report?"** No: validation storms chose the recipe,
  reported storms are fresh (§25.6), and two storm profiles are held out.
* **"Why only one seed in the live test before?"** It now covers all five seeds (§30.2).
* **"Is it meta-RL?"** It is trained with PPO and a recurrent policy in the RL² style; we tested for
  adaptation across repeated storms and did not find it (§28). MAML is documented and not used (§15).
* **"Why do the reward's latency numbers say 2 ms and 18 ms?"** They are a deliberation-cost
  weighting, not measurements (§27.5).
* **"Does it work on the real firewall?"** Yes for storms without a lost main supply (§30.2); the
  shortage scenario waits on M4 (§24).
