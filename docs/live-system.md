# The live system (ADR 0006)

Every hour the pipeline fetches today's weather for the four islands and runs it through the
modules. It stores everything, and the control room at `/` shows it.

```
Open-Meteo (only network call)
  → Module 1: today's inputs, per-asset vulnerability t+1..t+6 (ST-GNN, residual LSTM,
              persistence), the 28-feature state, grid snapshots for now..now+6
  → Module 2: uncertainty per asset (real EDL head, two-hour hysteresis)
  → Module 3: not connected (docs/contracts/m3-live-migration.md)
  → Module 4: can the grid carry each hour with no action? (risk-certified, northern-islands-v1)
  → PostgreSQL: runs, weather_fetches, forecasts, uncertainty, grid_checks, live_scores
  → scoring: each stored forecast is compared with the score its hour actually reached
```

**What is live and what is modelled.** Weather is today's: a numerical weather model, so it is
marked interpolated. Load is the latest ledger year's same month, shaped by today's temperature.
Fragility is carried forward from the last ledger month (stale). The islands have no electrical
telemetry, and nothing claims they do.

## Run it on this laptop

```bash
# 1. the database (once per boot); data lives outside iCloud in ~/.metacore/pgdata
/opt/homebrew/opt/postgresql@18/bin/pg_ctl -D ~/.metacore/pgdata \
  -o "-p 5433 -c listen_addresses=127.0.0.1" -l ~/.metacore/postgres.log start

# 2. the hourly pipeline (runs now, then at five past every hour)
cd services/orchestrator && nohup ../../.venv/bin/python -m app.pipeline run >> ~/.metacore/pipeline.log 2>&1 &

# 3. the gateway and the dashboard
uv run uvicorn gateway.main:app --app-dir services/gateway --port 8010
cd apps/dashboard && GATEWAY_URL=http://127.0.0.1:8010 pnpm dev     # → http://localhost:5173
```

`python -m app.pipeline once` runs a single hour. The laptop must stay awake: an hour missed
while it sleeps is simply not forecast. When it wakes, the next run catches up.

## On the home server (Raspberry Pi)

`docker-compose.yml` has the `db` service under profile `live`. Set `METACORE_DB_PASSWORD`. The
pipeline and gateway read `METACORE_DB_URL`. Module 2's `edl_real.onnx` must be on the machine:
it is git-ignored, so ask Duwaragie to publish it. Hosting the CEB-derived bundle needs the signed
approval first (ADR 0006, clauses 7 and 8).
