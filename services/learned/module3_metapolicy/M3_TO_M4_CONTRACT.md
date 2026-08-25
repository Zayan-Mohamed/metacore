# M3 → M4 Output Contract (draft v0.1) — for Hariswara (Module 4)

Per the master plan register, your verifier builds against this **mock stream**
until my real gRPC output is live. Two message types are emitted per control step
(alternating lines in `sample_m3_to_m4.jsonl`): a `ProposedControlAction` and its
accompanying `GatingDecision`. Field names match `packages/contracts/proto/module3.proto`
(JSON transport, same precedent as M2's starter).

## ProposedControlAction

| Field | Type | Meaning | Range |
|---|---|---|---|
| `action_id` | string (uuid) | correlates action ↔ decision ↔ verdict | — |
| `origin` | string | which path produced it | `SYSTEM1` / `SYSTEM2` |
| `breakers` | list[{edge_id, closed}] | breaker commands | — |
| `load_shed` | list[{node_id, shed_fraction, priority_tier}] | shed 0..1; tier 1 = most critical | — |
| `dispatch` | list[{node_id, p_kw, q_kvar}] | setpoints (often empty on SYSTEM1) | — |
| `rationale` | string | short dashboard summary | — |
| `schema_version` | string | contract version | `m3-out/0.1` |
| `message_type` | string | discriminator for the JSONL mix | `ProposedControlAction` |

## GatingDecision

| Field | Type | Meaning | Range |
|---|---|---|---|
| `action_id` | string | same as the paired action | — |
| `chosen` | string | effective path after budget fallback | `SYSTEM1` / `SYSTEM2` |
| `epistemic_at_decision` | float | M2's `u` at the gate | 0..1 |
| `expected_survival_benefit` | float | reward-model benefit term | — |
| `deliberation_cost` | float | cost charged this step (0 if S1) | ≥ 0 |
| `latency_ms` | float | stand-in path latency | — |
| `budget_exhausted_fallback` | bool | true when S2 was requested but budget forced S1 | — |
| `schema_version` | string | `m3-out/0.1` | — |
| `message_type` | string | `GatingDecision` | — |

Example (`GatingDecision` under a cyclone-like step):
```json
{"action_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","chosen":"SYSTEM2",
 "epistemic_at_decision":1.0,"expected_survival_benefit":1.2,"deliberation_cost":0.15,
 "latency_ms":18.0,"budget_exhausted_fallback":false,"schema_version":"m3-out/0.1",
 "message_type":"GatingDecision"}
```

Rule of thumb: **a REJECT here should eventually feed back as deliberation-cost
pressure, not just a discarded action.** Generate a live mock stream with
`python run_demo.py`. Contract to be frozen at PP1; ping me before you hard-code
field names.
