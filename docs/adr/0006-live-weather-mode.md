# ADR 0006 — A live-weather mode for Module 1: live hazard, modelled electrical state

**Status:** proposed · **Deciders:** Zayan (M1, owner); supervisor and CEB Jaffna for the hosting
and data-handling clauses (7 and 8). **Amends:** ADR 0004 (source modes, runtime language) and
proposal NFR7 (where processing runs).

## Context

The PP1 panel asked Module 1 to show the ST-GNN producing a forecast from current data, and to
show that the forecast can be checked. The proposal already commits to part of this: FR1 ingests
NASA POWER "in both historical batch and near-real-time modes", SO4 serves the forecaster on a
streaming path, and FR11 sends the per-node traces to the operator dashboard.

Three facts constrain how:

- **There is still no live electrical data.** ADR 0004's finding stands: no SCADA, no historian,
  a monthly manual ledger, and the last ledger month we hold is December 2025.
- **NASA POWER cannot feed a forecast for today.** Measured on 24 Oct 2026 for Nainativu:
  hourly wind, temperature and rain end at 21 Oct 23:00 (about 2.5 days behind) and hourly
  irradiance ends at 30 Jul 2026. POWER attributes the solar gap to a CERES processing delay.
- **Open-Meteo serves the current hour and the next days**, hourly, for the same variables, and
  its Historical Forecast API holds the archived forecasts back to January 2024 (checked for
  1-2 Jan 2024), so the model can be trained on the same product it will see live.

ADR 0004 says there is no `live` mode and that the runtime path is Go. NFR7 says processing runs
on the secure institutional server with no external calls at inference. A private SLIIT server is
not available. Each of these has to change explicitly, not drift.

## Decision

1. **A third source mode, `live-weather`, for Module 1 only.** Weather is live; the electrical
   state stays modelled. ADR 0004's constraint holds for everything electrical: no live mode is
   claimed for load, generation or asset state, and **none is needed**: the electrical inputs
   come from the calibrated 2024-2025 dataset the model was trained and tested on. Live
   operation only has to fill the hours after the last ledger month, as follows.
   - Load for months with no ledger is the same calendar month of the last ledger year, scaled
     by the same diurnal model, and flagged `QUALITY_INTERPOLATED`.
   - Fragility is carried forward from the last ledger month and flagged `QUALITY_STALE` once
     past its freshness budget (one month). The flag already exists in `common.proto`.
   - The five electrical features stay `QUALITY_MISSING`, as offline.

2. **Open-Meteo is the live weather source, and the model is retrained on it.** Training uses the
   Historical Forecast API for 2024-2025, the same product as live, on the same event-based split.
   Mapping to the 28-feature schema: 80 m wind stands in for 50 m, km/h becomes m/s, hPa becomes
   kPa, and shortwave radiation (W/m², mean of the preceding hour) becomes Wh/m². Clear-sky GHI
   is computed from solar geometry: Open-Meteo lists the variable but returns no values for it
   (checked live and historical, four models). Each fetch records the grid cell Open-Meteo actually answered from, as the NASA
   path does, so the islands' spatial resolution stays visible. The NASA-trained v0.1
   results remain the paper results; the gap between the two sources is reported as a finding,
   not tuned away.

3. **Fetching is split from inference.** One fetcher is the only component that makes external
   calls. It stores each raw response with its source, fetch time and grid cell. The forecaster
   reads only from the database. NFR7's "no external calls at inference" therefore still holds.

4. **The runtime is Python inside Module 1 (`module1.live`), not the Go `ingestion_svc`.** It
   reuses the feature engineering and the model code instead of re-implementing them. At an
   hourly cadence there is nothing for NATS to do, so the bus is not used. `ingestion_svc` keeps
   ADR 0004's simulated and replay modes.

5. **Storage is PostgreSQL.** Four tables: weather fetches, assembled states, forecasts (with
   the model version and a hash of the inputs) and live scores. It holds public weather and
   derived values only, never ledger rows.

6. **Validation is prospective (shadow mode).** Every hour the worker issues a t+1 to t+6
   forecast and stores it. Once the observed weather for those hours has been fetched, the
   realised target is computed and the forecast is scored, alongside persistence and the
   residual 1-D LSTM on the same hours. Three operational checks run with it: data freshness,
   inference latency against NFR1 (150 ms), and determinism (the same input hash and model
   version must give the same output).

7. **Data minimisation, then encryption.** The server receives a parameter bundle, not the
   ledger: each island's monthly energy, installed genset capacity, the Eluvaitivu hybrid's
   monthly decay ratio, the fitted normalisation statistics and the checkpoint. No fuel, oil, cost,
   transport or other per-plant rows leave the author's machine. The bundle is encrypted with
   `age`; the key is held outside the repository and the decrypted bundle lives only in memory.
   This protects against a leaked repository, disk snapshot or backup. It does not protect
   against the hosting provider or anyone with root on the machine, and is not described as
   doing so.

8. **Hosting is a provider VM behind Cloudflare Zero Trust.** A Cloudflare Tunnel connects out
   from the VM, so no inbound port is open, and Cloudflare Access admits only listed accounts
   (team, supervisor, panel). Cloudflare terminates TLS at its edge, so dashboard traffic is
   visible to Cloudflare; this goes in the ethics report. Nothing CEB-derived is deployed until
   the supervisor and CEB confirm in writing that hosting the derived bundle is within the
   academic-use terms.

## Rationale

The panel's question is whether the model runs on today's data and whether that can be checked.
Weather is the only input that is genuinely current, and it drives the hazard term, which is the
part of the target that changes hour to hour. Making weather live while keeping every other
input honestly flagged answers the question without contradicting ADR 0004's central claim.

Training on Open-Meteo's archived forecasts rather than feeding Open-Meteo into a NASA-trained
model removes a silent input shift between training and serving. Keeping both models lets the
shift be measured.

Prospective scoring is the strongest validation available without ground truth: the forecast is
fixed and stored before the outcome exists, so it cannot be fitted to it.

Sending a minimal bundle instead of the ledger is a stronger protection than encrypting the
ledger, because data that is not on the server cannot leak from it. Encryption then covers what
must be there.

## Consequences

- **The claim is "live hazard ingestion and live vulnerability forecasting over a modelled
  electrical state"**, never live grid monitoring. Survival prioritisation over the live plant
  remains future work, as the proposal states.
- **Live scores test the weather-driven part of the target.** Load is extrapolated from 2025
  and fragility goes stale, so the stress and fragility terms are not being validated live. The
  stale-fragility flag grows with every month without a new ledger. The monthly 2026 ledger
  pages from CEB (a dataset update, not a live feed) would replace the copied load with real
  monthly energy, refresh fragility, and give a further unseen period for testing.
- **Two trained models exist**: NASA-trained v0.1 (paper results) and the Open-Meteo-trained
  model (live). Each carries its weather source in its model card.
- **NFR7 changes** from "institutional server" to "provider VM, data-minimised and encrypted,
  behind Zero Trust", pending the written consent in clause 8. The ethics report records it.
- **Open-Meteo's free tier is for non-commercial use**, which covers this research project. A
  commercial deployment (the proposal's licence model) would need a paid plan or another
  provider behind the same fetcher interface.
- **ADR 0004 is amended, not replaced**: its two ingestion paths, its rule that nothing
  unmeasured is labelled observed, and its "no live electrical mode" all stand.
