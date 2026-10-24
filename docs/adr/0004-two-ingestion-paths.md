# ADR 0004 — Ingestion is two separate paths: offline calibration and simulated runtime

**Status:** accepted · **Deciders:** whole team (owner: Zayan, M1)

**Amended by:** [ADR 0006](0006-live-weather-mode.md), which adds a `live-weather` mode for
Module 1 (live weather, modelled electrical state) run in Python. Everything below still holds
for electrical data.

## Context

`packages/contracts/proto/module1.proto` describes a real-time control system. `StreamSnapshot`
returns a stream, `GridStateSnapshot` carries `update_period_s`, `Quality` has a `QUALITY_STALE`
member defined as "last known value, past its freshness budget", and `Health` reports
`last_step_latency_ms`. Those fields only mean something at second-to-minute cadence.

The data the state entity can actually supply is monthly. `data/external/ceb_jaffna` is a 120-row
ledger — two years, twelve months, five generating systems. It is not a sampling limitation that
better collection would fix. The site interview establishes that no higher-rate record exists or
will: data entry is manual, reports are printed and filed, there is no historian and no SCADA, and
the only protection relays in the entire fleet sit inside one vendor's hybrid cabinet on one island.
Between a monthly total and a contract written for second-scale dispatch there are five to six
orders of magnitude, and no ingestion work closes that gap because there is nothing on the other
side of it to ingest.

`services/realtime/ingestion_svc` is currently one undifferentiated box holding both jobs. That
ambiguity blocks the work behind it: whether a given adapter is Go or Python, whether it publishes
to NATS or writes a DVC-tracked artifact, and whether the bus belongs in the picture at all, are all
answered differently depending on which job is meant.

## Decision

Ingestion splits into two paths that share no code and no transport.

**Calibration path** — offline, batch, Python, owned by M1, in
`services/learned/module1_state_forecasting/src/module1/data/`. It reads `data/external/**` and
`data/raw/**`, and emits one versioned **parameter set** into `data/processed/`. Every stage is a
DVC stage. It never imports a bus client and is never on a request path.

**Runtime path** — streaming, Go, `services/realtime/ingestion_svc`. It consumes a frozen parameter
set and the scenario library, and publishes the proto messages. It never reads `data/external/**`
and never reads a calibration intermediate; the parameter set is the only interface between the two.

Three further constraints follow, and are part of this decision:

1. **The runtime path has exactly two source modes: `simulated` and `replay`. There is no `live`
   mode.** No field telemetry exists to connect to. Any future SCADA integration is a new adapter
   behind the same interface, not an assumed default that is merely unimplemented.
2. **Calibration artifacts are hourly.** Below that nothing in the evidence chain constrains the
   value — a monthly energy total and a weather reanalysis do not determine a five-minute set point.
   The simulator may step faster; it interpolates, and says so.
3. **Anything not measured carries a `QualityMask` that is not `QUALITY_OBSERVED`.** Downscaled
   load is `QUALITY_INTERPOLATED`. Renewable share inferred from a diesel-SFC counterfactual is
   `QUALITY_INTERPOLATED`. This is not a style preference: M2's entire contribution is detecting
   when the state it was handed is untrustworthy, and a synthetic value labelled as observed defeats
   the experiment rather than decorating it.

The parameter set ships with a synthetic fallback covering every field. `data/README.md` promises
that nothing in `external/` becomes a build requirement; the fallback is what makes that true in
code rather than in prose, and CI builds against it.

Implemented in [`module1/data/synthetic.py`](../../services/learned/module1_state_forecasting/src/module1/data/synthetic.py),
run by `task data:synthetic` and by the `data` CI lane. It generates a monthly ledger and two
years of hourly meteorology for the four sites, and every downstream stage and gate runs against
it unmodified — no gate is relaxed to admit it. The annual totals are the transcribed
`Data_CEB_Jaffna.pdf` figures already held in `validate.py` as the reconciliation reference, so
the reconciliation gate is genuinely exercised; nothing below the year is real, and no weather
value is real. Outputs are labelled `synthetic: true` and carry a `PROVENANCE.json` saying not to
publish results computed from them.

## Rationale

The two jobs differ on every axis that architecture responds to — cadence (months vs seconds),
language, transport, failure mode, and whether a stale answer is wrong or merely old. Fusing them
produces a component that is correct for neither, and in practice means a batch spreadsheet parse
acquires a message bus it has no use for.

Naming the absence of a live mode is the load-bearing half. The failure this project is written
about is a hybrid plant that degraded to 3% of its output over three months while nobody noticed,
because no one was watching a number a model could have watched. A pipeline that leaves a `live`
branch stubbed invites exactly the reading we are arguing against: that the telemetry is there and
merely unwired. It is not there. The contribution is what an agent can do without it.

Routing calibration through M1 matches ownership already recorded in `docs/roadmap.md`, where the
scenario library and the replay harness are M1-owned shared assets, and it fills
`module1/data/`, which exists for this and is empty.

## Consequences

- **There is no live demo, and the papers must say so.** Every evaluation is replay or simulation
  against the ID/OOD scenario library. Any figure implying a stream from a real island is wrong.
- **`ingestion_svc` is now a misleading name** — it ingests from a simulator, not from the field.
  Kept for now because renaming touches compose profiles, the Dockerfile and `go.work`; revisit
  before Phase E rather than during scaffolding.
- **Calibration correctness cannot be validated against held-out telemetry**, because there is
  none. It is validated instead by reconciliation invariants — the 2024 and 2025 annual totals in
  the spreadsheet already reconcile exactly against the independently-produced PDF summary tables,
  and that check becomes a pipeline stage rather than a one-off.
- DVC covers the calibration path only. The runtime path is reproducible through the scenario
  library and its `library_version`, which is already carried in `ScenarioRef`.
- Two CI lanes, and a check that `ingestion_svc` does not import a calibration package — the same
  shape of guard as `infra/ci/check_core_purity.sh`, for the same reason.
