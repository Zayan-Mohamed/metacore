# ADR 0005 — M4 certifies the risk of an action over the next hours, not just its safety now

**Status:** proposed · **Deciders:** Hariswara (M4), Zayan (M1 contract and topology), Saabir (M3
fixtures and the proto change)

## Context

Until now M4 judged an M3 action on one grid state: apply it, solve the power flow, check the
limits, answer APPROVE or REJECT. The action then stays in force for hours while demand moves.
An action can pass on the snapshot and break a limit an hour later, and M4 could not see that or
say how close to the edge an approval was. M4 is the last line of defence against the learned
modules' mistakes, so it should not take its input state as exact either.

Two facts about the shared model (PR #13, `northern-islands-v1`) shape the answer:

- the four islands are separate 400 V diesel microgrids, each with a stiff lead genset, so the
  power flow alone never fails to supply an island however overloaded its station is;
- the only recorded demand is hourly per island, constructed from monthly ledger totals.

## Decision

M4 becomes a **risk-certified safety shield** (runtime assurance, Simplex pattern):

1. **Check A** stays the gate on the recorded hour, on the shared model, with two additions: a
   generation-adequacy limit per island (demand against in-service genset kVA), and versioned
   limit profiles (METACORE-STRICT default; SL-PSR-2016 LV, EN 50160, ANSI C84.1 Range A).
2. A third verdict, **UNVERIFIED**: anything M4 cannot establish (solver error, non-convergence,
   a malformed snapshot, a non-finite number, a blown time budget, a stale calibration, or a risk
   bound that straddles the target) is not an approval. Approval has one code path.
3. **Check B** runs the action on 688 seeded scenarios of the next six hours. Scenarios are whole
   past six-hour windows of persistence misses from the same situation (hour of day, season,
   hazard), clipped to a split-conformal band. The band is frozen offline in
   `artifacts/calibration_v1.json`; in storms conformal PID adapts it online from the misses it
   observes.
4. The count of failing scenarios becomes the verdict through an exact one-sided Clopper-Pearson
   bound: APPROVE only if the risk is at most 1 % at 99.9 % confidence (k = 0 of 688), REJECT if
   it is above 1 % with the same confidence, UNVERIFIED otherwise.
5. **Shapley risk attribution** over five feeder players (Nainativu F1, Nainativu F2, Delft,
   Analaitivu, Eluvaitivu) says whose uncertainty owns the risk; the operator sentence names only
   entities in the topology pin and is checked for it.
6. `verification.proto` v1.1 carries all of this additively.

## Rationale

- **Data-calibrated, not learned.** The bands come from counting and sorting past misses, frozen
  with their input hashes. At runtime there is a table lookup, seeded sampling, power flows and
  an exact binomial formula. No ML import enters `deterministic-core` (ADR 0003 holds) and every
  verdict replays byte for byte from its certificate.
- **Independent of the module it guards.** M4 measures how wrong "same as now" has been from the
  recorded history itself instead of trusting M1's confidence in its own forecast.
- **Fail-safe by construction.** UNVERIFIED gives M3 a clear "not approved, try a safer action"
  instead of forcing M4 to guess when the evidence is unclear.
- **Exact, small statistics.** One-sided Clopper-Pearson with N = 688 needs no distributional
  assumption and no scipy. Early stopping only ever ends a run in REJECT (Bonferroni over the
  seven looks), so looking early cannot inflate a false approval.

## Consequences

- A verify call costs about 130 ms instead of about 8 ms (688 scenarios, two solves each, loads
  set in place). M3's decision period is an hour in replay, so this fits; the evaluation reports
  p50 and p95.
- The certificate needs a recorded hour (a snapshot or a timestamp). Without one the certifier
  answers UNVERIFIED; the plain gate is still available as `PhysicsVerifier`.
- M3's committed sample stream still names the retired `delft_3island` buses, so on the shared
  model its actions are REJECTed as malformed until M3 adopts the new names.
  `delft_3island.dss` stays at its old path, marked retired, only as M3's frozen training
  topology for its drift test; M4 never compiles it.
- Rebuilding the calibration is one command and must be byte-identical:
  `python -m verification.calibrate check` runs in the core test lane.
- Stated limitations: demand is downscaled from monthly totals (misses are smoother than real
  ones; the evaluation's band-width sensitivity shows what wider bands change), the model rests
  on the topology pin's assumptions, and the risk target and confidence are design choices.

See [docs/m4/rcv-method.md](../m4/rcv-method.md) for the method in full.
