# Module 3 — Cost-Aware Metacognitive Gating & Meta-Policy

**Owner:** Saabir S. · **Code:** `services/learned/module3_metapolicy/` · **Contract:** `packages/contracts/proto/module3.proto`

This is the single reference document for Module 3. If something about M3 changes
(formulation, reward, observation, contract, status), change it here. The code and this file
are the sources of truth; `docs/module3-explained.html` is a generated viva explainer and
`services/learned/module3_metapolicy/README.md` is the run guide.

> Last verified against code: 2026-10-15, branch `m3/gate-policy` at `0541f39`.
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

**Phase A — Foundations & Design (TAF)** — literature done; contracts published; formal MDP
definition is §2 of this document.

**Phase B — Prototype / Week-7 demo** — scaffold + Gym env, reward-shaping draft, System 1,
System 2 stub, threshold gate and toy Meta-RL all exist. Tier-1 rehearsal and the Week-7 demo and
write-up are tracked outside the repo.

**Open work (Phases C–F), in rough dependency order**

1. Replace `synthetic_context` with the live M1 `SystemContext` (agree with the M1 owner).
2. Replace `mock_verify` with real M4 verdicts; implement the M4 → M3 feedback loop.
3. Real System 2: load-shed prioritisation and critical-load ranking, replacing the stand-in.
4. Full Meta-RL pipeline with reward shaping on simulated cyclone episodes tied to the OpenDSS
   twin; swap the mock M2 stream for M2's real uncertainty API.
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
| M4 verdicts | `mock_verify` | real OpenDSS verdicts over gRPC |
| System 1 / System 2 | heuristic stand-ins | S2: real optimiser |
| Transport | JSON, field names from `module3.proto` | gRPC |
| Serving | Python / gateway subprocess | ONNX + Go `gating_decision_svc` |

---

## 12. File map

All paths under `services/learned/module3_metapolicy/` unless noted.

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
