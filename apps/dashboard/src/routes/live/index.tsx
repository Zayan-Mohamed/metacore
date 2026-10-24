/**
 * Module 1 — Live page, in replay mode (ADR 0006).
 *
 * Until the hourly worker runs, this steps through the held-out test quarter (Oct-Dec 2025) with
 * the Open-Meteo-trained ST-GNN's stored forecasts, exactly as the live page will step through
 * real hours. At each replay hour the page only uses what would be known then:
 *   - forecasts issued up to now;
 *   - realised scores up to now (a forecast for t+3 is scored three hours after it was issued).
 * Rolling scores are computed here from those known pairs, for ST-GNN, the fair baseline
 * (1-D LSTM + residual head) and persistence, the same comparison the live worker will store.
 *
 * Data: gateway GET /api/module1/replay  <- module1.infer.replay (committed JSON).
 */
import { useEffect, useMemo, useState } from "react";

import LineChart from "../../components/LineChart";
import { fetchModule1Replay, type Module1Replay, type Quality } from "../../lib";
import "../../styles/page.css";
import "./live.css";

type ModelId = "stgnn" | "lstm_1d_residual" | "persistence";

const MODELS: { id: ModelId; label: string; color: string; dashed?: boolean }[] = [
  { id: "stgnn", label: "ST-GNN (ours)", color: "var(--series-stgnn)" },
  { id: "lstm_1d_residual", label: "1-D LSTM + residual", color: "var(--series-lstm)" },
  { id: "persistence", label: "Persistence", color: "var(--series-pers)", dashed: true },
];

const WINDOWS = [
  { id: 24, label: "Last 24 h" },
  { id: 168, label: "Last 7 days" },
  { id: 100000, label: "Since replay start" },
];

const JUMPS = [
  { label: "Quarter start", at: "2025-10-01T23:00" },
  { label: "Heavy-rain evening", at: "2025-10-21T15:00" },
  { label: "Ditwah approach", at: "2025-11-27T06:00" },
  { label: "Ditwah peak", at: "2025-11-28T22:00" },
  { label: "Hybrid decay (Dec)", at: "2025-12-05T18:00" },
];

const ISLAND_SHORT: Record<string, string> = { "Delft-Neduntivu": "Delft" };
const short = (s: string) => ISLAND_SHORT[s] ?? s;
const QUALITY_CLASS: Record<Quality, string> = {
  QUALITY_OBSERVED: "observed", QUALITY_INTERPOLATED: "interpolated",
  QUALITY_STALE: "stale", QUALITY_MISSING: "missing",
};
const QUALITY_GROUPS: Record<string, string> = {
  weather: "Weather (Open-Meteo ICON)", load: "Load (constructed from monthly ledger)",
  fragility: "Fragility (last ledger month)", electrical: "Electrical readings",
  time_and_topology: "Time and topology",
};

/** Everything the page needs at one replay hour, from the packed integers. */
function useReplayMath(r: Module1Replay | null, cur: number, windowH: number) {
  return useMemo(() => {
    if (!r) return null;
    const s = 1 / r.scale;
    const H = r.horizon_h;
    const o0 = r.first_origin;
    const nowHour = o0 + cur;
    const truthAt = (hour: number, n: number) => r.truth[hour][n] * s;
    const fc = (m: ModelId, o: number, n: number, h: number) =>
      m === "persistence" ? truthAt(o0 + o, n) : r.forecast[m][o][n][h] * s;

    // Rolling scores over forecasts whose outcome is already known (target hour <= now).
    const from = Math.max(0, cur - windowH);
    const score: Record<ModelId, { se: number; k: number; caught: number }> = {
      stgnn: { se: 0, k: 0, caught: 0 }, lstm_1d_residual: { se: 0, k: 0, caught: 0 },
      persistence: { se: 0, k: 0, caught: 0 },
    };
    for (let o = from; o <= cur; o++) {
      for (let h = 0; h < H; h++) {
        if (o + h + 1 > cur) continue;
        for (let n = 0; n < r.nodes.length; n++) {
          const y = truthAt(o0 + o + h + 1, n);
          for (const m of MODELS) {
            const e = fc(m.id, o, n, h) - y;
            score[m.id].se += e * e;
            score[m.id].k += 1;
          }
        }
      }
    }
    // Onsets in the window, up to now: was each one forecast above tau from an earlier origin?
    let onsets = 0;
    for (let hour = Math.max(o0 + from + 1, o0 + H); hour <= nowHour; hour++) {
      for (let n = 0; n < r.nodes.length; n++) {
        if (!(truthAt(hour, n) > r.tau && truthAt(hour - 1, n) <= r.tau)) continue;
        onsets += 1;
        for (const m of MODELS) {
          for (let h = 1; h <= H; h++) {
            const o = hour - h - o0;
            if (o >= 0 && fc(m.id, o, n, h - 1) > r.tau) {
              score[m.id].caught += 1;
              break;
            }
          }
        }
      }
    }
    const rolling = MODELS.map((m) => ({
      ...m,
      rmse: score[m.id].k ? Math.sqrt(score[m.id].se / score[m.id].k) : null,
      pairs: score[m.id].k,
      caught: score[m.id].caught,
    }));
    return { s, H, o0, nowHour, truthAt, fc, rolling, onsets };
  }, [r, cur, windowH]);
}

export default function LiveRoute() {
  const [r, setR] = useState<Module1Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(400);
  const [windowH, setWindowH] = useState(168);
  const [model, setModel] = useState<ModelId>("stgnn");
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    fetchModule1Replay().then((d) => {
      setR(d);
      const start = d.hours.indexOf("2025-11-27T06:00") - d.first_origin;
      setCur(Math.max(0, start));
      const nai = d.nodes.findIndex((n) => n.id === "Nainativu/bus");
      setSelected(Math.max(0, nai));
    }).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const origins = r ? r.forecast.stgnn.length : 0;
  useEffect(() => {
    if (!playing || !r) return;
    const id = window.setInterval(() => {
      setCur((c) => {
        if (c + 1 >= origins) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, speedMs);
    return () => window.clearInterval(id);
  }, [playing, speedMs, r, origins]);

  const m = useReplayMath(r, cur, windowH);

  if (error) {
    return (
      <div className="pg"><div className="pg__inner">
        <section className="pg__card pg__card--error">
          <div className="pg__card-title">Replay unavailable</div>
          <pre className="pg__pre mono">{error}</pre>
        </section>
      </div></div>
    );
  }
  if (!r || !m) {
    return <div className="pg"><div className="pg__inner"><p className="pg__subtitle">Loading the replay…</p></div></div>;
  }

  const now = r.hours[m.nowHour];
  const inDitwah = Object.entries(r.events).find(([, [a, b]]) => now >= a && now < b);
  const node = r.nodes[selected];

  // Selected node: the last 48 known hours, then the 6 forecast hours.
  const back = 48;
  const lo = Math.max(0, m.nowHour - back);
  const xs = r.hours.slice(lo, m.nowHour + m.H + 1);
  const nowIdx = m.nowHour - lo;
  const actual = xs.map((_, i) => (lo + i <= m.nowHour ? m.truthAt(lo + i, selected) : null));
  const forecastSeries = MODELS.map((mm) => ({
    id: mm.id, label: mm.label, color: mm.color, dashed: mm.dashed,
    values: xs.map((_, i) => {
      const h = lo + i - m.nowHour; // 0 = now, 1..H = forecast hours
      if (h < 0 || h > m.H) return null;
      return h === 0 ? m.truthAt(m.nowHour, selected) : m.fc(mm.id, cur, selected, h - 1);
    }),
  }));

  const t1 = r.nodes.map((nd, n) => ({ nd, n, v: m.fc(model, cur, n, 0) }));
  const above = t1.filter((x) => x.v > r.tau);
  const islands = Object.keys(r.weather);
  const best = m.rolling.filter((x) => x.rmse !== null).sort((a, b) => (a.rmse! - b.rmse!))[0]?.id;

  return (
    <div className="pg">
      <div className="pg__inner">
        <header className="pg__header">
          <h1>
            Module 1 — Live
            <span className="pg__badge pg__badge--warn" title={r.note}>REPLAY · not live</span>
            <span className="pg__badge">{r.model}</span>
            <span className="pg__badge">weather: {r.weather_source}</span>
            {inDitwah ? <span className="pg__badge pg__badge--bad">Cyclone {inDitwah[0]}</span> : null}
          </h1>
          <p className="pg__subtitle">
            The held-out test quarter (October to December 2025), stepped hour by hour as the live
            worker will step through real hours. At each hour the page shows only what would be
            known then: the forecast just issued, and scores for earlier forecasts whose outcome has
            now happened. The score is a constructed vulnerability proxy, not an observation.
          </p>
        </header>

        <section className="pg__card">
          <div className="pg__card-title">Replay clock</div>
          <div className="live__clock mono">{now.replace("T", " ")}</div>
          <input className="pg__range" type="range" min={0} max={origins - 1} value={cur}
            aria-label="Replay hour" onChange={(e) => { setPlaying(false); setCur(Number(e.target.value)); }} />
          <div className="pg__btns">
            <button type="button" className="pg__btn pg__btn--sm" onClick={() => setPlaying((p) => !p)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" className="pg__btn pg__btn--sm pg__btn--ghost" disabled={cur === 0}
              onClick={() => setCur((c) => Math.max(0, c - 1))}>− 1 h</button>
            <button type="button" className="pg__btn pg__btn--sm pg__btn--ghost" disabled={cur >= origins - 1}
              onClick={() => setCur((c) => Math.min(origins - 1, c + 1))}>+ 1 h</button>
            <button type="button" className="pg__btn pg__btn--sm pg__btn--ghost" disabled={cur >= origins - 1}
              onClick={() => setCur((c) => Math.min(origins - 1, c + 24))}>+ 24 h</button>
            <label className="live__speed">
              <span>Speed</span>
              <select className="pg__input" value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))}>
                <option value={800}>1 h / 0.8 s</option>
                <option value={400}>1 h / 0.4 s</option>
                <option value={120}>1 h / 0.12 s</option>
              </select>
            </label>
          </div>
          <div className="pg__btns">
            {JUMPS.map((j) => {
              const idx = r.hours.indexOf(j.at) - r.first_origin;
              return (
                <button key={j.at} type="button" className="pg__btn pg__btn--sm pg__btn--ghost" disabled={idx < 0}
                  onClick={() => { setPlaying(false); setCur(idx); }}>{j.label}</button>
              );
            })}
          </div>
        </section>

        <section className="pg__kpis">
          <div className="pg__kpi">
            <div className="pg__kpi-label">Nodes above τ at t+1</div>
            <div className="pg__kpi-value">{above.length} / {r.nodes.length}</div>
            <div className="pg__kpi-note">τ = {r.tau.toFixed(3)} (95th percentile of training scores)</div>
          </div>
          <div className="pg__kpi">
            <div className="pg__kpi-label">Weather data age</div>
            <div className="pg__kpi-value">0 h</div>
            <div className="pg__kpi-note">Replay: the hour's weather is always present. Live: hours since the last Open-Meteo fetch.</div>
          </div>
          <div className="pg__kpi">
            <div className="pg__kpi-label">Forecasts scored ({WINDOWS.find((w) => w.id === windowH)?.label.toLowerCase()})</div>
            <div className="pg__kpi-value">{m.rolling[0].pairs.toLocaleString()}</div>
            <div className="pg__kpi-note">node × horizon pairs whose outcome has happened</div>
          </div>
          <div className="pg__kpi">
            <div className="pg__kpi-label">Onsets in window</div>
            <div className="pg__kpi-value">{m.onsets}</div>
            <div className="pg__kpi-note">upward crossings of τ, any node</div>
          </div>
        </section>

        <section className="pg__card">
          <div className="pg__card-title">Rolling scores — scored after the fact</div>
          <div className="pg__card-sub">
            Each forecast is stored when issued and scored once its hour arrives, so it cannot be
            fitted to the outcome. Same three models the live worker will store.
          </div>
          <div className="pg__btns">
            {WINDOWS.map((w) => (
              <button key={w.id} type="button"
                className={`pg__btn pg__btn--sm${w.id === windowH ? "" : " pg__btn--ghost"}`}
                onClick={() => setWindowH(w.id)}>{w.label}</button>
            ))}
          </div>
          <div className="pg__table-wrap">
            <table className="pg__table">
              <thead>
                <tr><th>Model</th><th className="num">RMSE</th><th className="num">Onsets warned</th><th className="num">Test-quarter RMSE</th><th className="num">Test-quarter onset recall</th></tr>
              </thead>
              <tbody>
                {m.rolling.map((x) => (
                  <tr key={x.id} className={x.id === best ? "is-best" : undefined}>
                    <td><span className="pg__legend-item"><span className="pg__swatch" style={{ background: x.color }} />{x.label}</span></td>
                    <td className="num">{x.rmse === null ? "—" : x.rmse.toFixed(4)}</td>
                    <td className="num">{m.onsets ? `${x.caught} / ${m.onsets} (${Math.round((100 * x.caught) / m.onsets)}%)` : "—"}</td>
                    <td className="num">{r.test_summary[x.id].rmse_mean.toFixed(4)}</td>
                    <td className="num">{Math.round(r.test_summary[x.id].onset_recall * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pg__note">
            Bold row: lowest RMSE in this window. Test-quarter columns are the full-quarter results
            from the training run, for reference.
          </div>
        </section>

        <section className="pg__card">
          <div className="pg__card-title">Every node: the forecast just issued</div>
          <div className="pg__card-sub">t+1 to t+6 hours. Darker is more vulnerable; an amber ring marks a score above τ. Select a row to chart it.</div>
          <div className="pg__btns">
            {MODELS.map((mm) => (
              <button key={mm.id} type="button"
                className={`pg__btn pg__btn--sm${mm.id === model ? "" : " pg__btn--ghost"}`}
                onClick={() => setModel(mm.id)}>{mm.label}</button>
            ))}
          </div>
          <div className="live__grid" role="table" aria-label="Per-node forecast">
            <div className="live__grid-row live__grid-row--head" role="row">
              <span role="columnheader">Node</span>
              <span role="columnheader" className="num">now</span>
              {Array.from({ length: m.H }, (_, h) => <span key={h} role="columnheader" className="num">t+{h + 1}</span>)}
            </div>
            {r.nodes.map((nd, n) => (
              <button type="button" key={nd.id} role="row"
                className={`live__grid-row${n === selected ? " is-selected" : ""}`} onClick={() => setSelected(n)}>
                <span role="cell" className="live__node">{short(nd.island)} · {nd.asset}</span>
                <span role="cell" className="live__cell mono" style={heat(m.truthAt(m.nowHour, n), r.tau)}>{m.truthAt(m.nowHour, n).toFixed(3)}</span>
                {Array.from({ length: m.H }, (_, h) => {
                  const v = m.fc(model, cur, n, h);
                  return <span key={h} role="cell" className="live__cell mono" style={heat(v, r.tau)} title={`${nd.id} t+${h + 1}: ${v.toFixed(3)}`}>{v.toFixed(3)}</span>;
                })}
              </button>
            ))}
          </div>
        </section>

        <section className="pg__card">
          <div className="pg__card-title">{short(node.island)} · {node.asset}: 48 hours known, 6 hours forecast</div>
          <div className="pg__card-sub">Left of “now”: the realised score. Right: what each model forecast at this hour.</div>
          <LineChart
            ariaLabel={`Vulnerability of ${node.id}: realised and forecast`}
            xLabels={xs}
            nowIndex={nowIdx}
            threshold={{ value: r.tau, label: "τ (event threshold)" }}
            series={[
              { id: "actual", label: "Realised score", color: "var(--text)", width: 2.5, values: actual },
              ...forecastSeries,
            ]}
          />
        </section>

        <div className="pg__cols">
          <section className="pg__card">
            <div className="pg__card-title">Island weather this hour</div>
            <div className="pg__card-sub">The hazard term is driven by 50 m wind and rainfall. Open-Meteo ICON, three grid cells for four islands.</div>
            <div className="pg__table-wrap">
              <table className="pg__table">
                <thead><tr><th>Island</th><th className="num">Wind 50 m (m/s)</th><th className="num">Rain (mm/h)</th><th className="num">Sun (Wh/m²)</th></tr></thead>
                <tbody>
                  {islands.map((isl) => {
                    const w = r.weather[isl];
                    const f = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "—" : v.toFixed(d));
                    return (
                      <tr key={isl}>
                        <td>{short(isl)}</td>
                        <td className="num">{f(w.wind_50m_ms[m.nowHour])}</td>
                        <td className="num">{f(w.precip_mm_hr[m.nowHour])}</td>
                        <td className="num">{f(w.ghi_wh_m2[m.nowHour], 0)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className="pg__card">
            <div className="pg__card-title">Input quality in live-weather mode</div>
            <div className="pg__card-sub">How every input is labelled for Module 2. Nothing filled in is ever marked observed.</div>
            <div className="pg__table-wrap">
              <table className="pg__table">
                <tbody>
                  {Object.entries(r.quality).map(([k, q]) => (
                    <tr key={k}>
                      <td>{QUALITY_GROUPS[k] ?? k}</td>
                      <td><span className={`pg__q pg__q--${QUALITY_CLASS[q]}`}>{QUALITY_CLASS[q]}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

// Colour domain for the heat cells: scores sit in roughly 0.15 .. 0.7, so a 0 .. 1 ramp would
// paint every cell the same. Same ramp as routes/vulnerability.
function heat(v: number, tau: number) {
  const pct = Math.max(0, Math.min(100, ((v - 0.2) / 0.45) * 100));
  return {
    background: `color-mix(in oklch, var(--blue) ${pct.toFixed(0)}%, var(--surface-2))`,
    color: pct > 58 ? "var(--bg)" : "var(--text)",
    boxShadow: v > tau ? "inset 0 0 0 2px var(--amber)" : undefined,
  };
}
