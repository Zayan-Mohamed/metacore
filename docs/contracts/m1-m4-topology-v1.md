# M1 -> M4 topology v1 (draft): review note for Module 4

> **Update (M4 risk-certified verification, ADR 0005):** M4 now runs on `northern-islands-v1`
> by default (items 1, 3 and 4 below are done on M4's side: new bus names, a generation-adequacy
> limit, the default switched and `delft_3island.dss` retired). M3's names are still M3's change.

**Status:** DRAFT for joint sign-off (Zayan, M1 / Hariswara, M4).

## Why

The master plan committed the M1 -> M4 topology and snapshot format to v0 by S1 W5 and frozen by
W8, as "one canonical topology file, jointly owned". Neither was produced. M4 has been running on
`delft_3island.dss`, which contradicts the CEB field record the proposal is built on:

| `delft_3island.dss` | Field record (CEB Jaffna ledger, tender, 17 Aug 2026 interview) |
|---|---|
| Subsea tie-lines join the islands (`Line_1_2`, `Line_2_3`, `E_crit_1`) | Every island is fully islanded; no inter-island cable exists |
| A 100 MVA grid source feeds Nainativu | No grid source; each island's diesel station is its only source |
| 11 kV everywhere | 400 V LT feeders from each power station; no substation on any island |
| PV on Nainativu (200 kW) and Analaitivu (150 kW) | Both diesel-only today (hybrid rollout planned, not built) |
| Delft and Eluvaitivu merged into one cluster with a 200 kW battery | Separate islands ~25 km apart; the only battery is Eluvaitivu's (~134 kWh, 60 kW hybrid) |
| Genset ratings (e.g. Nainativu 500 kW) | CEB 2025 fleet: Nainativu 880 kVA, Delft 830, Analaitivu 450, Eluvaitivu 100 + 30 |

## What this change adds

| Artifact | Owner | What it is |
|---|---|---|
| `services/learned/module1_state_forecasting/src/module1/topology/spec.py` | M1 | The single parameter table. Generates both artifacts below. |
| `packages/contracts/python/metacore_contracts/topology/northern_islands_v1.dss` | joint | Generated OpenDSS model: 4 islands, 13 buses, 9 lines, 10 generators (9 dispatchable; the wind turbines are out of service), 8 loads |
| `.../schema/module1_topology_v1.json` | joint | The pin: buses, lines, assets, loads, M1 node map, assumptions, snapshot format; every value tagged `measured` / `assumed` / `derived` |
| `.../schema/samples/module1_snapshot_v1_*.json` | M1 | Fixture snapshots: calm hour, Nainativu record peak, Ditwah evening (hybrid down) |
| `module1.proto` | M1 | Additive v1.0: `GridTopology.assets`, `NodeState.quality`, `GridStateSnapshot.assets_out_of_service` (buf breaking check passes) |
| `CircuitTwin.apply_snapshot`, `PhysicsVerifier.verify(action, snapshot=None)` | **M4 (yours)** | Opt-in. Loads a snapshot after the reset; a snapshot for another topology or naming an unknown bus is REJECTed as malformed |
| `GET /api/module1/topology`, `POST /api/module1/snapshot` | M1 | The pin and one hour's snapshot over HTTP |

Regenerate and check: `python -m module1.topology build` / `build --check` (CI fails on a hand edit).

## The model in one paragraph

Each island's lead diesel genset is a stiff source at its power-station bus (the AVR holds 1.0
p.u.), with the island's other gensets as dispatchable `Generator`s at 0 kW. Two feeder groups
per island (F1 assumed to carry the critical consumers, F2 the rest) hang off the station through
lumped 400 V lines whose ampacity is the number of parallel 100 A drop-out-fuse feeders needed for
125% of the constructed peak. Eluvaitivu's hybrid plant (PV, wind out of service, battery, 30 kW
backup genset) sits on its own bus tied to F1. On M4's own solver every fixture converges and
stays inside 0.95-1.05 p.u.; the tightest is the Nainativu record peak at 0.955 p.u. with the
F1 feeder at 348 of 500 A.

## What is measured and what is assumed

Measured: island count and islanding, 400 V distribution, genset fleet and ratings, ~100 A
drop-out-fuse feeders, the hybrid plant's components and the battery expiry, the hourly island
demand (constructed from the monthly ledger, so `QUALITY_INTERPOLATED`).

Assumed (all in `spec.py`, replace when the distribution control centre's single-line diagrams
arrive): feeder grouping and the 50/50 demand split, which feeder carries the critical load,
feeder count rule and lumped length (0.4 km), cable impedance (3x95 mm2 ABC), the hybrid plant's
connection point, power factor 0.85, and a stiff source.

## What M4 and M3 would change to adopt it (not done here)

1. **Bus names.** M3 (`priority.py`, `system1.py`, `system2.py`, `sample_m3_to_m4.jsonl`), M4's
   attribution rules (`attribution.py`, which pattern-matches tie-line names), the gateway's
   module4 presets and several M4 tests use `N1`..`N12`. The pin's `node_map` translates M1's
   forecaster nodes to the new buses; M3's priority tiers would key on `*_F1` (critical) / `*_F2`.
2. **Tie-line actions disappear.** Breaker commands on `E_crit_1` etc. are now MALFORMED, which is
   correct. M3's System 2 plans would shed per feeder group or dispatch standby gensets / the BESS.
3. **Genset capacity is not checked yet.** A stiff source never fails to supply, so overloading a
   station would pass. Proposed: a source-overload limit comparing each island's source power plus
   dispatched generation against its installed kVA (in the pin). This is the M4 limit that makes
   "genset outage at peak" a rejectable scenario.
4. **Switch the default** to `northern-islands-v1` once 1-3 land, and retire `delft_3island.dss`.

## Open questions

- Should the hybrid battery be grid-forming when the diesel station is off (it is the reference
  inside the hybrid plant per the interview)? v1 keeps one source per island for solvability.
- Feeder layout, hospital and RO-plant feeders, and line data: requested from CEB's distribution
  control centre; until then the assumptions above stand and are stated, not hidden.

## Sign-off

- [ ] Hariswara (M4): model, snapshot hook, migration plan
- [ ] Zayan (M1): pin, generator, snapshot producer
- [ ] Saabir (M3): impact on action library acknowledged
