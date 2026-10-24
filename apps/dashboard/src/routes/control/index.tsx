/**
 * Control room — the live system on one screen (ADR 0006).
 *
 * Everything shown is what the hourly pipeline (services/orchestrator, app.pipeline) computed from
 * today's weather and stored; the page only reads it (gateway /api/live/*) and refreshes every
 * minute. Layout follows Apple's Human Interface Guidelines: a large title, the four islands
 * first, then what the system did, what needs attention and why, and the detail behind it.
 *
 * Each island is its own grid -- no cable joins them -- so every grid figure is per island.
 * Electrical inputs are modelled and labelled as such; nothing implies live telemetry.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import Info from "../../components/Info";
import LineChart from "../../components/LineChart";
import {
  fetchLive,
  fetchLiveHistory,
  fetchLiveScores,
  type LiveGridHour,
  type LiveHistory,
  type LiveLatest,
  type LiveNode,
  type LiveScores,
} from "../../lib";
import "../../styles/page.css";
import "./control.css";

const REFRESH_MS = 60_000;
const STALE_MIN = 75;
const NOMINAL_V = 400; // the islands' low-voltage network, line to line

const ISLAND_NAME: Record<string, string> = {
  Nainativu: "Nainativu", "Delft-Neduntivu": "Delft", Analaitivu: "Analaitivu", Eluvaitivu: "Eluvaitivu",
};
const ISLAND_ORDER = ["Nainativu", "Delft-Neduntivu", "Analaitivu", "Eluvaitivu"];
const PREFIX: Record<string, string> = {
  Nainativu: "NAI", "Delft-Neduntivu": "DEL", Analaitivu: "ANA", Eluvaitivu: "ELU",
};
const ASSET: Record<string, { name: string; help: string }> = {
  bus: { name: "Busbar & feeders", help: "The island's main connection point at the power station, where the generators join and the 400 V lines leave for the village. Stands for the network and its customers. Overhead lines make it the most exposed to wind." },
  diesel: { name: "Diesel power station", help: "The island's diesel generators: its main source of electricity. Housed indoors, so least exposed to weather." },
  pv: { name: "Solar array", help: "Solar panels of Eluvaitivu's hybrid plant. Their fragility follows the plant's output in the CEB ledger." },
  bess: { name: "Battery storage", help: "Battery bank of Eluvaitivu's hybrid plant. It failed in late 2025 (plant at about 3% of normal output); CEB has tendered a replacement." },
  wind: { name: "Wind turbines", help: "Eluvaitivu's turbines, out of service." },
};
const MODEL_LABEL = { stgnn: "ST-GNN (ours)", lstm_1d_residual: "LSTM baseline", persistence: "No change" } as const;
const MODEL_HELP: Record<string, string> = {
  stgnn: "Module 1's spatiotemporal graph neural network: reads 24 h of every asset's inputs and lets connected assets share information.",
  lstm_1d_residual: "The fair baseline: a recurrent network that sees only each asset's own score history.",
  persistence: "The simplest forecast: the next six hours look exactly like now.",
};
const MODEL_COLOR: Record<string, string> = {
  stgnn: "var(--series-stgnn)", lstm_1d_residual: "var(--series-lstm)", persistence: "var(--series-pers)",
};
const HAZARD = ["Normal", "Elevated", "Severe", "Extreme"];

type Level = "ok" | "watch" | "alert";
const LEVEL: Record<Level, { label: string; symbol: string }> = {
  ok: { label: "Normal", symbol: "✓" },
  watch: { label: "Watch", symbol: "!" },
  alert: { label: "Alert", symbol: "✕" },
};

const asset = (n: { asset: string }) => ASSET[n.asset]?.name ?? n.asset;
const volts = (pu: number) => `${Math.round(pu * NOMINAL_V)} V`;
const hhmm = (s: string) => s.slice(11, 16);
const minutesSince = (iso: string) => Math.round((Date.now() - new Date(iso).getTime()) / 60000);

function heat(v: number, tau: number) {
  const pct = Math.max(0, Math.min(100, ((v - 0.15) / 0.45) * 100));
  return {
    background: `color-mix(in oklch, var(--blue) ${pct.toFixed(0)}%, var(--surface-2))`,
    color: pct > 58 ? "var(--bg)" : "var(--text)",
    boxShadow: v > tau ? "inset 0 0 0 2px var(--amber)" : undefined,
  };
}

/** One island at one checked hour: the worst of voltage, feeder loading and station capacity. */
function islandGrid(g: LiveGridHour, isl: string) {
  const p = PREFIX[isl];
  const buses = g.buses.filter((b) => b.bus.toUpperCase().startsWith(p));
  const lines = g.lines.filter((l) => l.line.toUpperCase().startsWith(p));
  const vmin = Math.min(...buses.map((b) => b.voltage_pu));
  const busiest = lines.reduce((m, l) => ((l.loading_pct ?? 0) > (m?.loading_pct ?? -1) ? l : m), lines[0]);
  const ad = g.adequacy.find((a) => a.island === isl);
  const cap = ad ? ad.demand_kva / ad.capacity_kva : 0;
  const riskFails = (g.certificate as { failures_by_island?: Record<string, number> } | null)
    ?.failures_by_island?.[isl] ?? 0;
  const breach = vmin < g.v_min_pu || buses.some((b) => b.voltage_pu > g.v_max_pu)
    || (busiest?.loading_pct ?? 0) >= 100 || cap > 1;
  const level: Level = breach ? "alert" : riskFails > 0 || vmin < g.v_min_pu + 0.01 || cap > 0.9 ? "watch" : "ok";
  return { vmin, busiest, ad, cap, riskFails, level };
}

/** The dominant term of a node's score now, in words. */
function driver(n: LiveNode) {
  const c = n.components_now;
  const top = (Object.entries(c) as [string, number][]).sort((a, b) => b[1] - a[1])[0][0];
  return { hazard: "weather hazard", fragility: "asset fragility", stress: "electrical load" }[top] ?? top;
}

type Alert = { level: Level; island?: string; title: string; meaning: string; next: string };

function buildAlerts(d: LiveLatest): Alert[] {
  const out: Alert[] = [];
  const m1 = d.module1;
  const age = minutesSince(d.run.started_at);
  if (age > STALE_MIN) out.push({ level: "alert", title: `No update for ${age} minutes`,
    meaning: "The hourly pipeline has not produced a new run. Everything on this screen is from the last run.",
    next: "Check that the laptop is awake and the pipeline process is running (docs/live-system.md)." });
  if (d.run.status !== "ok") out.push({ level: "alert", title: `Last run ${d.run.status}`,
    meaning: d.run.errors.join(" · ") || "A stage failed.", next: "The other stages' results are still shown." });
  for (const g of d.module4) {
    for (const isl of ISLAND_ORDER) {
      const s = islandGrid(g, isl);
      if (s.level === "alert") out.push({ level: "alert", island: isl,
        title: `${ISLAND_NAME[isl]} grid outside its limits at ${hhmm(g.hour)}`,
        meaning: `With the forecast load and no control action, lowest voltage ${volts(s.vmin)}, busiest feeder ${s.busiest?.loading_pct?.toFixed(0)}%, station ${Math.round(s.cap * 100)}% loaded.`,
        next: "Module 3 would propose an action (shed load, start a standby genset); it is not connected yet." });
    }
  }
  for (const n of m1.nodes) {
    const h = n.forecast.stgnn.findIndex((v) => v > m1.tau);
    if (h < 0) continue;
    const isHybrid = n.asset === "pv" || n.asset === "bess";
    out.push({ level: "watch", island: n.island,
      title: `${ISLAND_NAME[n.island]} ${asset(n).toLowerCase()}: high vulnerability from ${hhmm(m1.horizon_hours[h])}`,
      meaning: `Forecast score ${n.forecast.stgnn[h].toFixed(2)} is above the alert level ${m1.tau.toFixed(2)} (exceeded in only 5% of training hours). Driven mostly by ${driver(n)}${isHybrid && m1.fragility_carried_forward ? ": the latest ledger month (Dec 2025) shows the hybrid plant at about 3% of normal output" : ""}.`,
      next: isHybrid && m1.fragility_carried_forward
        ? "Clears when a newer CEB ledger page shows the plant running again. If the battery has been replaced, ask CEB for the 2026 pages."
        : "Watch the next runs; Module 4 shows whether the grid itself stays within limits." });
  }
  for (const n of d.module2?.nodes ?? []) {
    if (!n.competence_drop) continue;
    const [isl, a] = n.node.split("/");
    out.push({ level: "watch", island: isl,
      title: `Module 2 is not confident about ${ISLAND_NAME[isl] ?? isl} ${asset({ asset: a }).toLowerCase()}`,
      meaning: `Its uncertainty (${n.epistemic_uncertainty.toFixed(2)}) stayed above ${d.module2!.value_threshold.toFixed(2)} for ${d.module2!.hysteresis_h} hours in a row: this state does not look like the states Module 2 was trained on (reason: ${n.trigger_reason}).`,
      next: "In the full system this sends the decision to Module 3's slow, careful path. Until Module 3 is connected it is a flag for a human." });
  }
  const latest = d.ledger?.latest_month ?? m1.ledger_latest_month ?? "2025-12";
  for (const f of d.ledger?.recent_files ?? []) {
    if (minutesSince(f.received) > 24 * 60) continue;
    if (f.status === "rejected") out.push({ level: "alert", title: `Ledger file rejected: ${f.file}`,
      meaning: `It failed the checks, so none of it was used: ${f.failures.slice(0, 3).join("; ")}${f.failures.length > 3 ? ` (+${f.failures.length - 3} more)` : ""}.`,
      next: "Fix or re-export the workbook and save it into the inbox again; the next hourly run re-checks it." });
    else if (f.status === "accepted") out.push({ level: "ok", title: `Ledger updated: ${f.months_added.join(", ")}`,
      meaning: `${f.file} passed the checks and its new months are in use from this run.${f.notes.length ? ` Note: ${f.notes[0]}.` : ""}`,
      next: "Load and fragility for those months now come from CEB's figures instead of last year's." });
  }
  if (m1.fragility_carried_forward) out.push({ level: "ok", title: `Ledger data ends in ${latest}`,
    meaning: `Load for months after ${latest} uses the same month of the latest year CEB recorded, shaped by today's temperature. Asset fragility uses ${latest}, the last month available.`,
    next: "Save a newer CEB workbook into the ledger inbox (~/.metacore/ledger-inbox); the next hourly run checks it and uses it automatically." });
  if (!d.module3.connected) out.push({ level: "ok", title: "Module 3 (decision) not connected",
    meaning: "The grid check below assumes no control action is taken.", next: `Owner: ${d.module3.owner}.` });
  return out;
}

export default function ControlRoute() {
  const [d, setD] = useState<LiveLatest | null>(null);
  const [hist, setHist] = useState<LiveHistory | null>(null);
  const [scores, setScores] = useState<LiveScores | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<keyof typeof MODEL_LABEL>("stgnn");
  const [cell, setCell] = useState<{ isl: string; lead: number } | null>(null);

  const load = useCallback(() => {
    fetchLive().then((x) => { setD(x); setError(null); })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    fetchLiveHistory(48).then(setHist).catch(() => undefined);
    fetchLiveScores(168).then(setScores).catch(() => undefined);
  }, []);
  useEffect(() => {
    load();
    const id = window.setInterval(load, REFRESH_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const alerts = useMemo(() => (d ? buildAlerts(d) : []), [d]);

  if (error && !d) {
    return (
      <div className="cr"><div className="cr__inner">
        <h1 className="cr__large">Northern Islands</h1>
        <section className="cr__card cr__card--error">
          <h2>Live data unavailable</h2>
          <p className="cr__secondary">{error}</p>
          <p className="cr__secondary">Start the database and the pipeline: see docs/live-system.md.</p>
        </section>
      </div></div>
    );
  }
  if (!d) return <div className="cr"><div className="cr__inner"><h1 className="cr__large">Northern Islands</h1><p className="cr__secondary">Connecting…</p></div></div>;

  const m1 = d.module1;
  const age = minutesSince(d.run.started_at);
  const next = new Date(d.run.started_at);
  next.setHours(next.getHours() + 1, 5, 0, 0);
  const now0 = d.module4[0];

  const islandSummary = ISLAND_ORDER.map((isl) => {
    const w = m1.islands[isl];
    const nodes = m1.nodes.filter((n) => n.island === isl);
    const worst = nodes.map((n) => ({ n, v: Math.max(...n.forecast.stgnn) }))
      .sort((a, b) => b.v - a.v)[0];
    const grids = d.module4.map((g) => islandGrid(g, isl));
    const gridLevel: Level = grids.some((g) => g.level === "alert") ? "alert"
      : grids.some((g) => g.level === "watch") ? "watch" : "ok";
    const m2drop = (d.module2?.nodes ?? []).some((n) => n.node.startsWith(isl) && n.competence_drop);
    const level: Level = gridLevel === "alert" ? "alert"
      : gridLevel === "watch" || worst.v > m1.tau || m2drop ? "watch" : "ok";
    return { isl, w, worst, now: now0 ? islandGrid(now0, isl) : null, level, m2drop };
  });

  const timeline = [
    { t: d.weather_fetch.fetched_at, ms: d.run.stage_ms.fetch, text: `Fetched today's weather for the four islands from Open-Meteo (${d.weather_fetch.model}): the last 2 days and the next 2.` },
    { ms: d.run.stage_ms.module1, text: `Module 1 built each island's inputs and forecast ${m1.nodes.length} assets for ${hhmm(m1.horizon_hours[0])}–${hhmm(m1.horizon_hours[m1.horizon_hours.length - 1])}. ${m1.above_tau_t1.length} above the alert level in the first hour.` },
    { ms: d.run.stage_ms.module2, text: d.module2 ? `Module 2 checked how far to trust each asset's state: ${d.module2.nodes.filter((n) => n.competence_drop).length} not trusted.` : "Module 2 did not run." },
    { ms: undefined, text: "Module 3 skipped: not connected to the live grid yet.", muted: true },
    { ms: d.run.stage_ms.module4, text: d.module4.length ? `Module 4 solved all four island grids for ${d.module4.length} hours with no control action: ${d.module4.every((g) => g.decision === "DECISION_APPROVE") ? "every hour within limits" : "see the grid check"} (risk tested on 688 futures per hour).` : "Module 4 did not run." },
    { ms: d.run.stage_ms.scoring, text: `Scored ${d.run.stage_ms.scored ?? 0} earlier forecasts whose hour has now arrived.` },
  ];

  return (
    <div className="cr">
      <div className="cr__inner">
        {/* ------------------------------------------------------------ title */}
        <header className="cr__header">
          <div>
            <h1 className="cr__large">Northern Islands</h1>
            <p className="cr__secondary">
              <span className="cr__live">● Live</span> · Forecast issued {hhmm(m1.issued_for)} ·
              updated {age} min ago · next update about {next.toTimeString().slice(0, 5)}
            </p>
          </div>
          <nav className="cr__header-links">
            <span className={`cr__pill cr__pill--${d.run.status === "ok" && age <= STALE_MIN ? "ok" : "alert"}`}>
              {d.run.status === "ok" && age <= STALE_MIN ? "✓ System running" : "✕ Needs attention"}
            </span>
            <Link to="/modules">How it works</Link>
          </nav>
        </header>

        {/* ------------------------------------------------------------ islands */}
        <section className="cr__section">
          <h2 className="cr__title2">Right now<Info text="Each island is a separate grid fed by its own diesel power station; no cable joins them. Every figure below is for that island alone." /></h2>
          <div className="cr__islands">
            {islandSummary.map((s) => (
              <article key={s.isl} className={`cr__card cr__island cr__island--${s.level}`}>
                <div className="cr__island-head">
                  <h3>{ISLAND_NAME[s.isl]}</h3>
                  <span className={`cr__pill cr__pill--${s.level}`}>{LEVEL[s.level].symbol} {LEVEL[s.level].label}</span>
                </div>
                <dl className="cr__facts">
                  <div><dt>Weather</dt><dd>Wind {s.w.wind_50m_ms[23].toFixed(1)} m/s · rain {s.w.precip_mm_hr[23].toFixed(1)} mm/h</dd></div>
                  {s.now ? (
                    <div><dt>Grid now</dt><dd>
                      Lowest voltage {volts(s.now.vmin)}
                      <Info text={`${s.now.vmin.toFixed(3)} per-unit of the 400 V nominal. Limits 380–420 V (0.95–1.05 per-unit).`} />
                      {" "}· station {Math.round(s.now.cap * 100)}% loaded
                      <Info text={`Forecast demand ${s.now.ad?.demand_kva.toFixed(0)} kVA against ${s.now.ad?.capacity_kva.toFixed(0)} kVA of generators in service.`} />
                    </dd></div>
                  ) : null}
                  <div><dt>Most vulnerable</dt><dd>
                    {asset(s.worst.n)} {s.worst.v.toFixed(2)}{s.worst.v > m1.tau ? " · above alert level" : ""}
                  </dd></div>
                </dl>
              </article>
            ))}
          </div>
        </section>

        {/* ------------------------------------------------------------ what it did + alerts */}
        <div className="cr__two">
          <section className="cr__card">
            <h2 className="cr__title3">What the system did this hour</h2>
            <ol className="cr__timeline">
              {timeline.map((s, i) => (
                <li key={i} className={s.muted ? "is-muted" : undefined}>
                  <span className="cr__dot" aria-hidden="true" />
                  <span>{s.text}</span>
                  {s.ms !== undefined ? <span className="cr__ms">{s.ms >= 1000 ? `${(s.ms / 1000).toFixed(1)} s` : `${Math.round(s.ms)} ms`}</span> : null}
                </li>
              ))}
            </ol>
            <p className="cr__footnote">Run {d.run.id} of {d.run.runs_stored} · inputs {m1.input_sha256.slice(0, 10)} · nothing here was computed in your browser.</p>
          </section>

          <section className="cr__card">
            <h2 className="cr__title3">Needs attention <span className="cr__count">{alerts.filter((a) => a.level !== "ok").length}</span></h2>
            <ul className="cr__alerts">
              {alerts.map((a, i) => (
                <li key={i} className={`cr__alert cr__alert--${a.level}`}>
                  <div className="cr__alert-title">
                    <span className={`cr__symbol cr__symbol--${a.level}`}>{a.level === "ok" ? "i" : LEVEL[a.level].symbol}</span>
                    {a.title}
                  </div>
                  <p><strong>What it means.</strong> {a.meaning}</p>
                  <p><strong>What happens next.</strong> {a.next}</p>
                </li>
              ))}
            </ul>
          </section>
        </div>

        {/* ------------------------------------------------------------ grid check (M4) */}
        <section className="cr__card">
          <h2 className="cr__title3">Grid safety check — each island, next 6 hours <span className="cr__module">Module 4</span>
            <Info text="For every hour, Module 4 solves each island's electrical network with the forecast load and no control action, and checks voltages (380–420 V), feeder ratings and generator capacity. It then tests 688 plausible futures to certify the risk of a breach is at most 1%." />
          </h2>
          <p className="cr__secondary">Load is modelled from the CEB ledger, not metered. Select a cell for detail.</p>
          <div className="cr__matrix" role="table" aria-label="Grid check per island and hour">
            <div className="cr__mrow cr__mrow--head" role="row">
              <span>Island</span>
              {d.module4.map((g) => <span key={g.lead_h}>{g.lead_h === 0 ? "Now" : hhmm(g.hour)}</span>)}
            </div>
            {ISLAND_ORDER.map((isl) => (
              <div key={isl} className="cr__mrow" role="row">
                <span className="cr__mlabel">{ISLAND_NAME[isl]}</span>
                {d.module4.map((g) => {
                  const s = islandGrid(g, isl);
                  const on = cell?.isl === isl && cell.lead === g.lead_h;
                  return (
                    <button key={g.lead_h} type="button" className={`cr__mcell cr__mcell--${s.level}${on ? " is-on" : ""}`}
                      onClick={() => setCell(on ? null : { isl, lead: g.lead_h })}
                      aria-label={`${ISLAND_NAME[isl]} ${hhmm(g.hour)}: ${LEVEL[s.level].label}, lowest voltage ${volts(s.vmin)}`}>
                      <span>{LEVEL[s.level].symbol}</span>
                      <span className="cr__mval">{volts(s.vmin)}</span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
          {cell ? (() => {
            const g = d.module4[cell.lead];
            const s = islandGrid(g, cell.isl);
            const p = PREFIX[cell.isl];
            return (
              <div className="cr__detail">
                <h3>{ISLAND_NAME[cell.isl]} at {hhmm(g.hour)} — {LEVEL[s.level].label}</h3>
                <dl className="cr__facts cr__facts--row">
                  <div><dt>Lowest voltage</dt><dd>{volts(s.vmin)} ({s.vmin.toFixed(3)} pu)</dd></div>
                  <div><dt>Busiest feeder</dt><dd>{s.busiest ? `${s.busiest.loading_pct?.toFixed(0)}% of ${s.busiest.rating_amps.toFixed(0)} A` : "—"}</dd></div>
                  <div><dt>Station</dt><dd>{s.ad ? `${s.ad.demand_kva.toFixed(0)} of ${s.ad.capacity_kva.toFixed(0)} kVA` : "—"}</dd></div>
                  <div><dt>Weather hazard</dt><dd>{HAZARD[g.hazard_level] ?? "Unknown"} (whole system)</dd></div>
                  <div><dt>Risky futures</dt><dd>{s.riskFails} of {g.certificate?.n_scenarios ?? "—"}</dd></div>
                </dl>
                <p className="cr__footnote">
                  Out of service on this island: {g.assets_out_of_service.filter((a) => a.startsWith(p)).join(", ") || "none"}.
                  {" "}<Link to="/grid">Network diagram →</Link>
                </p>
              </div>
            );
          })() : null}
        </section>

        {/* ------------------------------------------------------------ vulnerability (M1) */}
        <section className="cr__card">
          <h2 className="cr__title3">Vulnerability forecast — every asset, next 6 hours <span className="cr__module">Module 1</span>
            <Info text="A score from 0 to 1 combining weather hazard, the asset's fragility and electrical load. It is a constructed indicator, not a measurement. The amber ring marks the alert level, exceeded in only 5% of training hours." />
          </h2>
          <div className="cr__segmented" role="radiogroup" aria-label="Forecast model">
            {(Object.keys(MODEL_LABEL) as (keyof typeof MODEL_LABEL)[]).map((m) => (
              <button key={m} type="button" role="radio" aria-checked={m === model}
                className={m === model ? "is-on" : undefined} onClick={() => setModel(m)} title={MODEL_HELP[m]}>
                {MODEL_LABEL[m]}
              </button>
            ))}
          </div>
          <div className="cr__heat" role="table" aria-label="Vulnerability per asset">
            <div className="cr__hrow cr__hrow--head" role="row">
              <span>Island</span><span>Asset</span><span>Now</span>
              {m1.horizon_hours.map((h) => <span key={h}>{hhmm(h)}</span>)}
            </div>
            {ISLAND_ORDER.flatMap((isl) => m1.nodes.filter((n) => n.island === isl)).map((n, i, arr) => (
              <div key={n.id} className={`cr__hrow${i > 0 && arr[i - 1].island !== n.island ? " is-first" : ""}`} role="row">
                <span className="cr__hisland">{i === 0 || arr[i - 1].island !== n.island ? ISLAND_NAME[n.island] : ""}</span>
                <span className="cr__hasset">{asset(n)}<Info text={ASSET[n.asset]?.help ?? n.asset} /></span>
                <span className="cr__cell" style={heat(n.current, m1.tau)}>{n.current.toFixed(2)}</span>
                {n.forecast[model].map((v, h) => (
                  <span key={h} className="cr__cell" style={heat(v, m1.tau)} title={`${asset(n)}, ${ISLAND_NAME[n.island]}, ${hhmm(m1.horizon_hours[h])}: ${v.toFixed(3)}`}>{v.toFixed(2)}</span>
                ))}
              </div>
            ))}
          </div>
          <p className="cr__footnote">Alert level {m1.tau.toFixed(2)} · {MODEL_HELP[model]}</p>
        </section>

        {/* ------------------------------------------------------------ M2 + M3 */}
        <div className="cr__two">
          <section className="cr__card">
            <h2 className="cr__title3">Can we trust the picture? <span className="cr__module">Module 2</span>
              <Info text="Module 2 estimates how unfamiliar each asset's state is compared with what it learned from (0 = familiar, 1 = completely unfamiliar). If it stays above the threshold for two hours, it flags a competence drop." />
            </h2>
            {d.module2 ? (
              <div className="cr__bars">
                {ISLAND_ORDER.flatMap((isl) => d.module2!.nodes.filter((n) => n.node.startsWith(isl))).map((n) => {
                  const [isl, a] = n.node.split("/");
                  const high = n.epistemic_uncertainty > d.module2!.value_threshold;
                  return (
                    <div key={n.node} className="cr__brow">
                      <span>{ISLAND_NAME[isl]} · {asset({ asset: a })}</span>
                      <span className="cr__track" title={`Uncertainty ${n.epistemic_uncertainty.toFixed(3)}`}>
                        <span style={{ width: `${Math.min(100, n.epistemic_uncertainty * 100)}%`, background: high ? "var(--amber)" : "var(--blue)" }} />
                        <span className="cr__thr" style={{ left: `${d.module2!.value_threshold * 100}%` }} />
                      </span>
                      <span className="cr__bval">{n.competence_drop ? "! not trusted" : high ? "rising" : "trusted"}</span>
                    </div>
                  );
                })}
                <p className="cr__footnote">Line on each bar: the threshold ({d.module2.value_threshold.toFixed(2)}). {Math.round((d.module2.nodes[0]?.observed_fraction ?? 0) * 100)}% of inputs are directly observed; the rest are modelled.</p>
              </div>
            ) : <p className="cr__secondary">Module 2 did not run this hour.</p>}
          </section>

          <section className="cr__card">
            <h2 className="cr__title3">What should we do? <span className="cr__module">Module 3</span></h2>
            <span className="cr__pill cr__pill--watch">! Not connected</span>
            <p className="cr__secondary">{d.module3.reason}</p>
            <p className="cr__footnote">Owner: {d.module3.owner}. Until it is connected, the grid check assumes no action is taken.</p>
          </section>
        </div>

        {/* ------------------------------------------------------------ weather */}
        <section className="cr__card">
          <h2 className="cr__title3">Weather — past day and next 6 hours<Info text="Open-Meteo, the ICON weather model. Its grid gives Eluvaitivu and Analaitivu the same values, so three lines are drawn for four islands." /></h2>
          <LineChart
            ariaLabel="Wind at 50 m by island, past 24 h and next 6 h"
            xLabels={m1.islands.Nainativu.hours}
            nowIndex={23}
            height={170}
            format={(v) => v.toFixed(0)}
            series={["Nainativu", "Delft-Neduntivu", "Eluvaitivu"].map((isl, i) => ({
              id: isl, label: `${ISLAND_NAME[isl]}${isl === "Eluvaitivu" ? " & Analaitivu" : ""} — wind (m/s)`,
              color: ["var(--series-stgnn)", "var(--series-lstm)", "var(--series-pers)"][i],
              values: m1.islands[isl].wind_50m_ms,
            }))}
          />
        </section>

        {/* ------------------------------------------------------------ skill + history */}
        <div className="cr__two">
          <section className="cr__card">
            <h2 className="cr__title3">How good are the forecasts?<Info text="Every forecast is stored when it is made and compared with the score its hour actually reached, so it cannot be adjusted after the fact. Lower error is better." /></h2>
            {scores && Object.keys(scores.models).length ? (
              <>
                <p className="cr__secondary">{scores.hours_scored} hour{scores.hours_scored === 1 ? "" : "s"} scored so far. Early numbers are noisy; they settle over days.</p>
                <table className="cr__table">
                  <thead><tr><th>Model</th><th>Compared</th><th>Error</th><th>1 h ahead</th><th>6 h ahead</th></tr></thead>
                  <tbody>
                    {(["stgnn", "lstm_1d_residual", "persistence"] as const).filter((m) => scores.models[m]).map((m) => {
                      const s = scores.models[m];
                      return (
                        <tr key={m}>
                          <td><span className="cr__swatch" style={{ background: MODEL_COLOR[m] }} />{MODEL_LABEL[m]}</td>
                          <td>{s.n}</td><td>{s.rmse?.toFixed(3) ?? "—"}</td>
                          <td>{s.by_lead[1]?.rmse.toFixed(3) ?? "—"}</td><td>{s.by_lead[6]?.rmse.toFixed(3) ?? "pending"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            ) : <p className="cr__secondary">The first forecasts are scored one hour after the first run.</p>}
          </section>

          <section className="cr__card">
            <h2 className="cr__title3">Recent hours</h2>
            <table className="cr__table">
              <thead><tr><th>Issued</th><th>Run</th><th>Lowest voltage (any island)</th><th>Grid</th></tr></thead>
              <tbody>
                {(hist?.runs ?? []).slice(-8).reverse().map((r) => (
                  <tr key={r.run_id}>
                    <td>{r.issued_for.slice(5).replace("T", " ")}</td>
                    <td>{r.status === "ok" ? "✓ complete" : r.status}</td>
                    <td>{r.v_min_pu !== null ? volts(r.v_min_pu) : "—"}</td>
                    <td>{r.decision === "DECISION_APPROVE" ? "✓ within limits" : r.decision ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>

        {/* ------------------------------------------------------------ honesty */}
        <section className="cr__card cr__card--quiet">
          <h2 className="cr__title3">What is live and what is modelled</h2>
          <dl className="cr__facts cr__facts--row">
            <div><dt>Weather</dt><dd>Today's, from a weather model</dd></div>
            <div><dt>Load</dt><dd>{m1.ledger_year_used} ledger, same month, shaped by today's temperature</dd></div>
            <div><dt>CEB ledger</dt><dd>Up to {d.ledger?.latest_month ?? m1.ledger_latest_month ?? "2025-12"} · {d.ledger?.source ?? m1.ledger_source ?? "calibration"}<Info text="New monthly workbooks saved into ~/.metacore/ledger-inbox are checked (layout, complete months, plausibility against each plant's own record, no rewriting of reconciled months) and used from the next hourly run. A file that fails is rejected and shown as an alert." /></dd></div>
            <div><dt>Asset fragility</dt><dd>{m1.fragility_carried_forward ? `Carried from ${d.ledger?.latest_month ?? m1.ledger_latest_month ?? "2025-12"} (stale)` : "From the ledger"}</dd></div>
            <div><dt>Electrical readings</dt><dd>None exist on the islands</dd></div>
          </dl>
          <p className="cr__footnote"><Link to="/data-health">Data checks →</Link> · <Link to="/model-graph">How the model works →</Link></p>
        </section>
      </div>
    </div>
  );
}
