# M2 → M3 Output Contract (draft v0.1) — for Saabir (Module 3)

Per the master plan register and M3 FR1, your gate builds against this **mock stream**
until my real output is live. One message is emitted per state / control step.

| Field | Type | Meaning | Range |
|---|---|---|---|
| `timestamp` | float (epoch s) | when the state was scored | — |
| `epistemic_uncertainty` | float | **u = K/S** — the gating signal | 0.0 (sure) … 1.0 (no evidence) |
| `aleatoric_proxy` | float | predictive entropy (irreducible noise) | ≥ 0 |
| `competence_drop` | bool | trigger to escalate to System 2 | true / false |
| `state_class` | int | argmax safety class | 0 safe … K-1 critical |
| `class_probabilities` | float[K] | Dirichlet-mean p over safety classes | sums to 1 |
| `schema_version` | string | contract version | `m2-out/0.1` |

Example (a cyclone state):
```json
{"timestamp":1786565266.25,"epistemic_uncertainty":1.0,"aleatoric_proxy":0.0,
 "competence_drop":true,"state_class":0,"class_probabilities":[0.333,0.333,0.333],
 "schema_version":"m2-out/0.1"}
```
Gate rule of thumb: **low u → stay System 1; `competence_drop == true` → escalate to System 2.**
Generate a live mock stream with `contract.build_output(...)` (see `run_demo.py`).
Contract to be frozen at PP1 (master plan §5); ping me before you hard-code field names.
