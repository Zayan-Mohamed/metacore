# deterministic-core — Module 4

**AUDIT: zero ML dependencies · sole caller of OpenDSS**

A top-level sibling of `services/`, not a child of it, so the learned/deterministic boundary is
visible in the first line of `ls`. See [ADR 0003](../docs/adr/0003-deterministic-core-isolation.md).

| Package | Role |
|---|---|
| `module4_verification` | The risk-certified safety shield. Check A loads the recorded hour into the `northern-islands-v1` OpenDSS twin, applies the proposed action and checks it against a versioned limit profile and generation adequacy. Check B runs it on 688 seeded scenarios of the next six hours from frozen conformal bands and certifies the risk with an exact Clopper-Pearson bound. Returns `APPROVE`, `REJECT` or `UNVERIFIED` with a certificate and Shapley risk attribution. Deterministic and replayable. See [ADR 0005](../docs/adr/0005-risk-certified-verification.md) and [the method](../docs/m4/rcv-method.md). |
| `module4_semantic_translation` | Turns that structured evidence into a human-readable causal log. Sits **downstream of the decision** and can never change it. |

## Invariants

1. Neither `pyproject.toml` declares an ML dependency. `infra/ci/check_core_purity.sh` fails the
   build if one appears, and runs as a pre-commit hook.
2. No component outside this directory imports OpenDSS.
3. Failure is safe: a malformed action is `REJECT`; a malformed snapshot, solver failure,
   non-finite number, stale calibration or a risk bound that straddles the target is
   `UNVERIFIED`. Neither is ever `APPROVE`.
5. Data-calibrated, not learned: the bands are counted from recorded history offline and frozen
   in `artifacts/calibration_v1.json`; `python -m verification.calibrate check` asserts a
   byte-identical rebuild.
4. OpenDSS versions are pinned and physics tests run on Linux in CI, so a verdict is never
   operating-system dependent.

## Scope

Single-step, model-based safety shielding with a statistical look-ahead: each proposed action is
checked on the current recorded hour and on calibrated scenarios of the next six hours.
Multi-step / model-predictive shielding, the nearest safe action, anytime-valid tests and
rare-event sampling are future work, not delivered. Live data is out of scope (ADR 0004).
