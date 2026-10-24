/**
 * Control room — the live system on one screen (ADR 0006).
 *
 * Everything here is what the hourly pipeline (services/orchestrator, app.pipeline) computed from
 * today's weather and stored; the page only reads it (gateway /api/live/*) and refreshes every
 * minute. Nothing is replayed and nothing is computed in the browser except the alert list.
 *
 * Panels: run health · weather now and next 6 h · per-asset vulnerability (M1) · uncertainty
 * (M2) · decision (M3, not connected) · grid check per hour (M4) · alerts · live forecast skill
 * · input quality. Electrical inputs are modelled and labelled as such; no panel implies live
 * telemetry from the islands.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import LineChart from "../../components/LineChart";
import {
  fetchLive,
  fetchLiveHistory,
  fetchLiveScores,
  type LiveGridHour,
  type LiveLatest,
  type LiveScores,
  type LiveHistory,
} from "../../lib";
import "../../styles/page.css";
import "./control.css";

const REFRESH_MS = 60_000;
const STALE_MIN = 75;
const ISL_SHORT: Record<string, string> = { "Delft-Neduntivu": "Delft" };
const short = (s: string) => ISL_SHORT[s] ?? s;
const ISLAND_PREFIX: Record<string, string> = {
  Nainativu: "NAI", "Delft-Neduntivu": "DEL", Analaitivu: "ANA", Eluvaitivu: "ELU",
};
const MODEL_LABEL: Record<string, string> = {
  stgnn: "ST-GNN (ours)", lstm_1d_residual: "1-D LSTM + residual", persistence: "Persistence",
};
const MODEL_COLOR: Record<string, string> = {
  stgnn: "var(--series-stgnn)", lstm_1d_residual: "var(--series-lstm)", persistence: "var(--series-pers)",
};
const DECISION: Record<string, { label: string; cls: string }> = {
  DECISION_APPROVE: { label: "✓ SAFE", cls: "pass" },
  DECISION_REJECT: { label: "✕ LIMIT BREACH", cls: "fail" },
  DECISION_UNVERIFIED: { label: "? UNVERIFIED", cls: "warn" },
};
const HAZARD = ["NORMAL", "ELEVATED", "SEVERE", "EXTREME"];
const Q_SHORT: Record<string, string> = {
  QUALITY_OBSERVED: "observed", QUALITY_INTERPOLATED: "interpolated", QUALITY_STALE: "stale",
  QUALITY_MISSING: "missing",
};

function heat(v: number, tau: number) {
  const pct = Math.max(0, Math.min(100, ((v - 0.15) / 0.45) * 100));
  return {
    background: `color-mix(in oklch, var(--blue) ${pct.toFixed(0)}%, var(--surface-2))`,
    color: pct > 58 ? "var(--bg)" : "var(--text)",
    boxShadow: v > tau ? "inset 0 0 0 2px var(--amber)" : undefined,
  };
}

function minutesSince(iso: string) {
  return Math.round((Date.now() - new Date(iso).getTime()) / 60000);
}

type Alert = { level: "critical" | "warning" | "info"; text: string };

function buildAlerts(d: LiveLatest): Alert[] {
  const out: Alert[] = [];
  const age = minutesSince(d.run.started_at);
  if (age > STALE_MIN) out.push({ level: "critical", text: `No new run for ${age} min: the pipeline may have stopped.` });
  if (d.run.status !== "ok") out.push({ level: "critical", text: `Last run ${d.run.status}: ${d.run.errors.join(" · ") || "see run log"}` });
  const m1 = d.module1;
  for (const n of m1.nodes) {
    const h = n.forecast.stgnn.findIndex((v) => v > m1.tau);
    if (h >= 0) out.push({ level: "warning", text: `${short(n.island)} ${n.asset}: vulnerability above τ from ${m1.horizon_hours[h].slice(11)} (t+${h + 1}, ${n.forecast.stgnn[h].toFixed(3)})` });
  }
  for (const g of d.module4) {
    if (g.decision !== "DECISION_APPROVE") out.push({ level: g.decision === "DECISION_REJECT" ? "critical" : "warning", text: `Grid at ${g.hour.slice(11)}: ${DECISION[g.decision]?.label ?? g.decision}${g.violations.length ? ` (${g.violations.length} violations)` : ""}${g.unverified_reasons.length ? ` — ${g.unverified_reasons[0]}` : ""}` });
  }
  for (const n of d.module2?.nodes ?? []) {
    if (n.competence_drop) out.push({ level: "warning", text: `Module 2: competence drop at ${n.node} (${n.trigger_reason})` });
  }
  if (m1.fragility_carried_forward) out.push({ level: "info", text: `Fragility carried forward from the last ledger month (${m1.ledger_year_used}-12): Eluvaitivu hybrid plant still treated as down until CEB sends newer ledger pages.` });
  if (!d.module3.connected) out.push({ level: "info", text: "Module 3 (decision) is not connected to the live grid yet." });
  return out;
}

export default function ControlRoute() {
  const [d, setD] = useState<LiveLatest | null>(null);
  const [hist, setHist] = useState<LiveHistory | null>(null);
  const [scores, setScores] = useState<LiveScores | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lead, setLead] = useState(0);
  const [model, setModel] = useState<"stgnn" | "lstm_1d_residual" | "persistence">("stgnn");
  const [clock, setClock] = useState(new Date());

  const load = useCallback(() => {
    fetchLive().then((x) => { setD(x); setError(null); })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    fetchLiveHistory(48).then(setHist).catch(() => undefined);
    fetchLiveScores(168).then(setScores).catch(() => undefined);
  }, []);
  useEffect(() => {
    load();
    const a = window.setInterval(load, REFRESH_MS);
    const b = window.setInterval(() => setClock(new Date()), 1000);
    return () => { window.clearInterval(a); window.clearInterval(b); };
  }, [load]);

  const alerts = useMemo(() => (d ? buildAlerts(d) : []), [d]);

  if (error && !d) {
    return (
      <div className="pg cr"><div className="cr__inner">
        <section className="pg__card pg__card--error">
          <div className="pg__card-title">Live store unavailable</div>
          <pre className="pg__pre mono">{error}</pre>
          <div className="pg__note">Start the database and the pipeline: see services/orchestrator (python -m app.pipeline run).</div>
        </section>
      </div></div>
    );
  }
  if (!d) return <div className="pg cr"><div className="cr__inner"><p className="pg__subtitle">Connecting to the live store…</p></div></div>;

  const m1 = d.module1;
  const age = minutesSince(d.run.started_at);
  const grid: LiveGridHour | undefined = d.module4[lead];
  const nextRun = new Date(new Date(d.run.started_at).getTime());
  nextRun.setHours(nextRun.getHours() + 1, 5, 0, 0);
  const islands = Object.keys(m1.islands);

  return (
    <div className="pg cr">
      <div className="cr__inner">
        {/* ---------------------------------------------------------- status bar */}
        <header className="cr__bar">
          <div className="cr__title">
            <strong>MetaCore</strong> · Northern islands control room
            <span className="pg__badge pg__badge--good">LIVE</span>
          </div>
          <div className="cr__bar-items mono">
            <span>{clock.toLocaleString("en-GB", { timeZone: "Asia/Colombo", hour12: false })} SLST</span>
            <span>forecast issued {m1.issued_for.replace("T", " ")}</span>
            <span className={age > STALE_MIN ? "cr__bad" : undefined}>updated {age} min ago</span>
            <span>next run ~{nextRun.toTimeString().slice(0, 5)}</span>
            <span className={`pg__status pg__status--${d.run.status === "ok" ? "pass" : "fail"}`}>
              {d.run.status === "ok" ? "✓" : "!"} run {d.run.status}
            </span>
            <Link to="/modules" className="cr__link">modules →</Link>
          </div>
        </header>

        <div className="cr__grid">
          {/* ------------------------------------------------------ alerts */}
          <section className="pg__card cr__alerts">
            <div className="pg__card-title">Alerts <span className="pg__badge">{alerts.length}</span></div>
            <ul className="cr__alert-list">
              {alerts.map((a, i) => (
                <li key={i} className={`cr__alert cr__alert--${a.level}`}>
                  <span className="cr__alert-tag mono">{a.level === "critical" ? "✕ CRIT" : a.level === "warning" ? "! WARN" : "i INFO"}</span>
                  {a.text}
                </li>
              ))}
              {alerts.length === 0 ? <li className="pg__card-sub">Nothing to report.</li> : null}
            </ul>
          </section>

          {/* ------------------------------------------------------ weather */}
          <section className="pg__card cr__weather">
            <div className="pg__card-title">Weather — now and next 6 h</div>
            <div className="pg__card-sub">Open-Meteo ICON, fetched {d.weather_fetch.fetched_at.replace("T", " ").slice(0, 16)}. Drives the hazard term.</div>
            <div className="pg__table-wrap">
              <table className="pg__table">
                <thead><tr><th>Island</th><th className="num">Wind 50 m now</th><th className="num">max next 6 h</th><th className="num">Rain now</th><th className="num">next 6 h total</th><th className="num">Temp</th></tr></thead>
                <tbody>
                  {islands.map((isl) => {
                    const w = m1.islands[isl];
                    const nxt = (a: number[]) => a.slice(24, 30);
                    return (
                      <tr key={isl}>
                        <td>{short(isl)}</td>
                        <td className="num">{w.wind_50m_ms[23].toFixed(1)} m/s</td>
                        <td className="num">{Math.max(...nxt(w.wind_50m_ms)).toFixed(1)} m/s</td>
                        <td className="num">{w.precip_mm_hr[23].toFixed(1)} mm/h</td>
                        <td className="num">{nxt(w.precip_mm_hr).reduce((s, v) => s + v, 0).toFixed(1)} mm</td>
                        <td className="num">{w.temp_2m_c[23].toFixed(1)} °C</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <LineChart
              ariaLabel="Wind at 50 m, past 24 h and next 6 h, by island"
              xLabels={m1.islands[islands[0]].hours}
              nowIndex={23}
              height={150}
              format={(v) => v.toFixed(0)}
              series={islands.filter((isl) => isl !== "Analaitivu").map((isl, i) => ({
                id: isl, label: `${short(isl)} wind (m/s)`,
                color: ["var(--series-stgnn)", "var(--series-lstm)", "var(--series-pers)"][i],
                values: m1.islands[isl].wind_50m_ms,
              }))}
            />
            <div className="pg__note">Analaitivu not drawn: it shares Eluvaitivu's weather grid cell.</div>
          </section>

          {/* ------------------------------------------------------ vulnerability */}
          <section className="pg__card cr__vuln">
            <div className="pg__card-title">Asset vulnerability — Module 1</div>
            <div className="pg__card-sub">Score now and forecast t+1 to t+6. Amber ring: above τ = {m1.tau.toFixed(3)}. A constructed proxy (hazard, fragility, stress), not a measurement.</div>
            <div className="pg__btns">
              {(["stgnn", "lstm_1d_residual", "persistence"] as const).map((m) => (
                <button key={m} type="button" className={`pg__btn pg__btn--sm${m === model ? "" : " pg__btn--ghost"}`}
                  onClick={() => setModel(m)}>{MODEL_LABEL[m]}</button>
              ))}
            </div>
            <div className="cr__heat" role="table" aria-label="Per-asset vulnerability forecast">
              <div className="cr__heat-row cr__heat-row--head" role="row">
                <span>Asset</span><span>now</span>
                {m1.horizon_hours.map((h) => <span key={h}>{h.slice(11, 13)}h</span>)}
              </div>
              {m1.nodes.map((n) => (
                <div key={n.id} className="cr__heat-row" role="row">
                  <span className="cr__heat-label">{short(n.island)} · {n.asset}</span>
                  <span className="cr__cell mono" style={heat(n.current, m1.tau)}>{n.current.toFixed(2)}</span>
                  {n.forecast[model].map((v, h) => (
                    <span key={h} className="cr__cell mono" style={heat(v, m1.tau)} title={`${n.id} ${m1.horizon_hours[h]}: ${v.toFixed(3)}`}>{v.toFixed(2)}</span>
                  ))}
                </div>
              ))}
            </div>
            <div className="pg__note mono">input {m1.input_sha256.slice(0, 12)} · forward pass {m1.latency_ms} ms · {m1.models.stgnn}</div>
          </section>

          {/* ------------------------------------------------------ grid (M4) */}
          <section className="pg__card cr__gridpanel">
            <div className="pg__card-title">Can the grid carry it? — Module 4</div>
            <div className="pg__card-sub">Each hour's forecast load solved on the four-island network with no control action, certified over 688 scenarios of the next 6 h. Load is modelled from the ledger, not metered.</div>
            <div className="cr__hours">
              {d.module4.map((g) => (
                <button key={g.lead_h} type="button" className={`cr__hour${g.lead_h === lead ? " is-on" : ""}`} onClick={() => setLead(g.lead_h)}>
                  <span className="mono">{g.hour.slice(11, 16)}</span>
                  <span className={`pg__status pg__status--${DECISION[g.decision]?.cls ?? "warn"}`}>{DECISION[g.decision]?.label ?? g.decision}</span>
                  <span className="pg__card-sub">{HAZARD[g.hazard_level] ?? "?"}{g.certificate ? ` · risk ≤ ${(g.certificate.upper_bound * 100).toFixed(1)}%` : ""}</span>
                </button>
              ))}
            </div>
            {grid ? (
              <div className="pg__table-wrap">
                <table className="pg__table">
                  <thead><tr><th>Island</th><th className="num">Lowest voltage</th><th className="num">Busiest feeder</th><th className="num">Demand / station capacity</th></tr></thead>
                  <tbody>
                    {islands.map((isl) => {
                      const p = ISLAND_PREFIX[isl];
                      const bs = grid.buses.filter((b) => b.bus.toUpperCase().startsWith(p));
                      const ls = grid.lines.filter((l) => l.line.toUpperCase().startsWith(p));
                      const vmin = Math.min(...bs.map((b) => b.voltage_pu));
                      const lmax = ls.reduce((m, l) => ((l.loading_pct ?? 0) > (m?.loading_pct ?? -1) ? l : m), ls[0]);
                      const a = grid.adequacy.find((x) => x.island === isl);
                      return (
                        <tr key={isl}>
                          <td>{short(isl)}</td>
                          <td className={`num${vmin < grid.v_min_pu ? " cr__bad" : ""}`}>{vmin.toFixed(3)} pu</td>
                          <td className={`num${(lmax?.loading_pct ?? 0) >= 100 ? " cr__bad" : ""}`}>{lmax ? `${lmax.loading_pct?.toFixed(0)}% (${lmax.line})` : "—"}</td>
                          <td className={`num${a && a.demand_kva > a.capacity_kva ? " cr__bad" : ""}`}>{a ? `${a.demand_kva.toFixed(0)} / ${a.capacity_kva.toFixed(0)} kVA (${Math.round((100 * a.demand_kva) / a.capacity_kva)}%)` : "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : <div className="pg__card-sub">No Module 4 check stored for this run.</div>}
            {grid?.assets_out_of_service.length ? (
              <div className="pg__note">Out of service: {grid.assets_out_of_service.join(", ")}.</div>
            ) : null}
            <div className="pg__note">Limits {grid?.v_min_pu}–{grid?.v_max_pu} pu. <Link to="/grid">Network detail →</Link></div>
          </section>

          {/* ------------------------------------------------------ uncertainty (M2) */}
          <section className="pg__card cr__unc">
            <div className="pg__card-title">How far to trust the state — Module 2</div>
            {d.module2 ? (
              <>
                <div className="pg__card-sub">Epistemic uncertainty per asset (real EDL head). Competence drop fires above {d.module2.value_threshold.toFixed(2)} for {d.module2.hysteresis_h} consecutive hours.</div>
                <div className="cr__bars">
                  {d.module2.nodes.map((n) => (
                    <div key={n.node} className="cr__bar-row">
                      <span className="cr__heat-label">{short(n.node.split("/")[0])} · {n.node.split("/")[1]}</span>
                      <span className="cr__bar-track" title={`u = ${n.epistemic_uncertainty.toFixed(3)}`}>
                        <span style={{ width: `${Math.min(100, n.epistemic_uncertainty * 100)}%`,
                          background: n.epistemic_uncertainty > d.module2!.value_threshold ? "var(--amber)" : "var(--blue)" }} />
                        <span className="cr__bar-thr" style={{ left: `${d.module2!.value_threshold * 100}%` }} />
                      </span>
                      <span className="mono cr__bar-val">{n.epistemic_uncertainty.toFixed(2)}</span>
                      <span className="pg__card-sub">{d.module2!.class_names[n.state_class] ?? n.state_class}{n.competence_drop ? " · DROP" : ""}</span>
                    </div>
                  ))}
                </div>
                <div className="pg__note">Observed share of inputs {d.module2.nodes[0]?.observed_fraction.toFixed(2)} — weather, load and fragility are modelled, so Module 2 sees a partly-sensed state.</div>
              </>
            ) : <div className="pg__card-sub">Module 2 did not run this hour: {d.run.errors.join(" · ")}</div>}
          </section>

          {/* ------------------------------------------------------ decision (M3) */}
          <section className="pg__card cr__m3">
            <div className="pg__card-title">Decision — Module 3 <span className="pg__status pg__status--warn">! NOT CONNECTED</span></div>
            <div className="pg__card-sub">{d.module3.reason}</div>
            <div className="pg__note">Owner: {d.module3.owner}. Until it is connected the grid panel shows the grid with no control action.</div>
          </section>

          {/* ------------------------------------------------------ skill */}
          <section className="pg__card cr__skill">
            <div className="pg__card-title">Live forecast skill — scored after the fact</div>
            {scores && Object.keys(scores.models).length ? (
              <>
                <div className="pg__card-sub">{scores.hours_scored} hours scored ({scores.first_scored_hour?.replace("T", " ")} → {scores.last_scored_hour?.replace("T", " ")}). Each forecast is stored when issued and compared with the score its hour actually reached.</div>
                <div className="pg__table-wrap">
                  <table className="pg__table">
                    <thead><tr><th>Model</th><th className="num">Pairs</th><th className="num">RMSE</th><th className="num">RMSE t+1</th><th className="num">RMSE t+6</th></tr></thead>
                    <tbody>
                      {Object.entries(scores.models).map(([mname, s]) => (
                        <tr key={mname}>
                          <td><span className="pg__legend-item"><span className="pg__swatch" style={{ background: MODEL_COLOR[mname] }} />{MODEL_LABEL[mname] ?? mname}</span></td>
                          <td className="num">{s.n}</td>
                          <td className="num">{s.rmse?.toFixed(4) ?? "—"}</td>
                          <td className="num">{s.by_lead[1]?.rmse.toFixed(4) ?? "—"}</td>
                          <td className="num">{s.by_lead[6]?.rmse.toFixed(4) ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div className="pg__card-sub">No forecast has reached its hour yet. The first scores appear one hour after the first run, and t+6 scores after six.</div>}
          </section>

          {/* ------------------------------------------------------ history */}
          <section className="pg__card cr__hist">
            <div className="pg__card-title">Last {hist?.runs.length ?? 0} runs</div>
            {hist && hist.runs.length > 1 ? (
              <LineChart
                ariaLabel="Lowest bus voltage per run"
                xLabels={hist.runs.map((r) => r.issued_for)}
                height={150}
                format={(v) => v.toFixed(3)}
                threshold={{ value: grid?.v_min_pu ?? 0.95, label: "voltage limit" }}
                series={[{ id: "vmin", label: "Lowest bus voltage (pu)", color: "var(--series-stgnn)",
                  values: hist.runs.map((r) => r.v_min_pu) }]}
              />
            ) : <div className="pg__card-sub">History builds up hourly; trends appear after the second run.</div>}
            <div className="pg__table-wrap">
              <table className="pg__table">
                <thead><tr><th>Issued</th><th>Status</th><th className="num">Lowest voltage</th><th className="num">Busiest feeder</th><th>Grid</th></tr></thead>
                <tbody>
                  {(hist?.runs ?? []).slice(-6).reverse().map((r) => (
                    <tr key={r.run_id}>
                      <td className="mono">{r.issued_for.replace("T", " ")}</td>
                      <td>{r.status}</td>
                      <td className="num">{r.v_min_pu?.toFixed(3) ?? "—"}</td>
                      <td className="num">{r.max_line_pct !== null ? `${r.max_line_pct.toFixed(0)}%` : "—"}</td>
                      <td>{r.decision ? DECISION[r.decision]?.label ?? r.decision : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* ------------------------------------------------------ data quality */}
          <section className="pg__card cr__quality">
            <div className="pg__card-title">What is live and what is modelled</div>
            <div className="pg__table-wrap">
              <table className="pg__table">
                <tbody>
                  {Object.entries(m1.quality).map(([k, q]) => (
                    <tr key={k}><td>{k.replace(/_/g, " ")}</td><td><span className={`pg__q pg__q--${Q_SHORT[q]}`}>{Q_SHORT[q]}</span></td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pg__note">
              Weather is today's (a numerical weather model, so interpolated). Load is the
              {` ${m1.ledger_year_used} `}ledger's same month shaped by today's temperature. No live
              electrical telemetry exists on the islands. Stage times: {Object.entries(d.run.stage_ms).filter(([k]) => k !== "scored").map(([k, v]) => `${k} ${Math.round(v)} ms`).join(" · ")}.
              {" "}<Link to="/data-health">Data health →</Link>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
