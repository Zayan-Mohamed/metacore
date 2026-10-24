/**
 * Module 1 — Data health.
 *
 * Runs every calibration gate now (the same functions the DVC stages run, via gateway GET
 * /api/module1/data-health -> module1.data.health) and shows what each source is, how fresh it
 * is, what is measured and what is constructed, and how the two weather sources compare.
 *
 * The CEB ledger is academic-use data: this page shows counts and pass/fail only, never an
 * energy, fuel or cost figure.
 */
import { useCallback, useEffect, useState } from "react";

import { fetchModule1DataHealth, type Module1DataHealth } from "../../lib";
import "../../styles/page.css";

const STATUS_LABEL = { pass: "✓ PASS", fail: "✕ FAIL", error: "! ERROR" } as const;
const QUALITY_ORDER = ["QUALITY_OBSERVED", "QUALITY_INTERPOLATED", "QUALITY_STALE", "QUALITY_MISSING"];
const QUALITY_SHORT: Record<string, string> = {
  QUALITY_OBSERVED: "observed", QUALITY_INTERPOLATED: "interpolated",
  QUALITY_STALE: "stale", QUALITY_MISSING: "missing",
};
const VAR_LABEL: Record<string, string> = {
  ghi_wh_m2: "Irradiance", wind_50m_ms: "Wind 50 m", temp_2m_c: "Temperature", precip_mm_hr: "Rainfall",
};
const ALL_VAR_LABEL: Record<string, string> = {
  ...VAR_LABEL, ghi_clearsky_wh_m2: "Clear-sky irradiance", wind_10m_ms: "Wind 10 m",
  humidity_2m_pct: "Humidity", pressure_kpa: "Pressure",
};

function bytes(n?: number) {
  if (n === undefined) return "—";
  return n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${(n / 1e3).toFixed(0)} KB`;
}

export default function DataHealthRoute() {
  const [d, setD] = useState<Module1DataHealth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const run = useCallback(() => {
    setPending(true);
    fetchModule1DataHealth()
      .then((x) => { setD(x); setError(null); })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setPending(false));
  }, []);
  useEffect(run, [run]);

  const allPass = d && d.summary.failed === 0;

  return (
    <div className="pg">
      <div className="pg__inner">
        <header className="pg__header">
          <h1>
            Module 1 — Data health
            {d ? (
              <span className={`pg__badge ${allPass ? "pg__badge--good" : "pg__badge--bad"}`}>
                {d.summary.passed} / {d.summary.gates} gates pass
              </span>
            ) : null}
            {pending ? <span className="pg__badge">running checks…</span> : null}
          </h1>
          <p className="pg__subtitle">
            Every calibration gate, run now on the committed data. A gate stops the pipeline rather
            than warning, because no held-out telemetry exists to catch a wrong value later. Below
            the gates: what each source is, what is measured and what is constructed.
          </p>
          <div className="pg__btns">
            <button type="button" className="pg__btn pg__btn--sm" onClick={run} disabled={pending}>Run checks again</button>
            {d ? <span className="pg__card-sub mono">checked {d.generated_at.replace("T", " ")} · {Math.round(d.generated_ms)} ms</span> : null}
          </div>
        </header>

        {error ? (
          <section className="pg__card pg__card--error">
            <div className="pg__card-title">Checks could not run</div>
            <pre className="pg__pre mono">{error}</pre>
          </section>
        ) : null}

        {d ? (
          <>
            <section className="pg__card">
              <div className="pg__card-title">1. Pipeline gates</div>
              <div className="pg__table-wrap">
                <table className="pg__table">
                  <thead><tr><th>Status</th><th>Gate</th><th>What it checks</th><th>Inputs</th><th className="num">Time</th></tr></thead>
                  <tbody>
                    {d.gates.map((g) => (
                      <tr key={g.id}>
                        <td><span className={`pg__status pg__status--${g.status}`}>{STATUS_LABEL[g.status]}</span></td>
                        <td>
                          {g.title}
                          <div className="pg__card-sub mono">{g.id}</div>
                          {g.failures.length ? <pre className="pg__pre mono">{g.failures.join("\n")}</pre> : null}
                          {g.error ? <pre className="pg__pre mono">{g.error}</pre> : null}
                        </td>
                        <td className="pg__card-sub">{g.checks}</td>
                        <td>
                          {g.inputs.map((f) => (
                            <div key={f.path} className="pg__card-sub mono">
                              {f.path} · {f.present ? `${bytes(f.bytes)}${f.files && f.files > 1 ? ` in ${f.files} files` : ""} · ${f.modified?.slice(0, 10) ?? ""}` : "absent"}
                            </div>
                          ))}
                        </td>
                        <td className="num">{g.duration_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <div className="pg__cols">
              <section className="pg__card">
                <div className="pg__card-title">2. CEB generation ledger</div>
                <div className="pg__card-sub">{d.ledger.governance}</div>
                <div className="pg__table-wrap">
                  <table className="pg__table">
                    <tbody>
                      <tr><td>Rows</td><td className="num">{d.ledger.rows}</td></tr>
                      <tr><td>Generating systems</td><td className="num">{d.ledger.systems.length}</td></tr>
                      <tr><td>Years</td><td className="num">{d.ledger.years.join(", ")}</td></tr>
                      <tr><td>Months per system</td><td className="num">{d.ledger.months_per_system}</td></tr>
                      <tr><td>Raw workbook on this machine</td><td className="num">{d.ledger.raw_workbook_on_this_machine ? "yes" : "no (synthetic fallback available)"}</td></tr>
                    </tbody>
                  </table>
                </div>
                <div className="pg__note">{d.ledger.systems.join(" · ")}</div>
              </section>

              <section className="pg__card">
                <div className="pg__card-title">3. Constructed hourly load</div>
                <div className="pg__card-sub">
                  Monthly ledger energy spread over the hours with an assumed daily shape. Monthly
                  totals are exact; the shape is not measured, so every row is <span className="pg__q pg__q--interpolated">interpolated</span>.
                </div>
                <div className="pg__table-wrap">
                  <table className="pg__table">
                    <thead><tr><th>Island</th><th className="num">Peak kW</th><th className="num">Mean kW</th><th className="num">Load factor</th></tr></thead>
                    <tbody>
                      {Object.entries(d.load.islands).map(([isl, v]) => (
                        <tr key={isl}><td>{isl}</td><td className="num">{v.peak_kw.toFixed(0)}</td><td className="num">{v.mean_kw.toFixed(0)}</td><td className="num">{v.load_factor.toFixed(2)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>

            <section className="pg__card">
              <div className="pg__card-title">4. Weather sources</div>
              <div className="pg__card-sub">
                NASA POWER trains the paper model; Open-Meteo trains the live model (ADR 0006).
                “Distinct series” counts how many different series four islands actually receive.
              </div>
              <div className="pg__table-wrap">
                <table className="pg__table">
                  <thead><tr><th>Source</th><th>Model</th><th>Range</th><th className="num">Hours per island</th><th className="num">Missing values</th><th>Distinct series (of 4 islands)</th></tr></thead>
                  <tbody>
                    {(["nasa_power", "open_meteo"] as const).map((k) => {
                      const w = d.weather[k];
                      const missing = Object.values(w.missing_values).reduce((a, b) => a + b, 0);
                      return (
                        <tr key={k}>
                          <td>{w.source}</td>
                          <td className="mono">{w.model ?? "MERRA-2 / CERES"}</td>
                          <td className="mono">{w.range[0]} – {w.range[1]}</td>
                          <td className="num">{w.hours_per_site.toLocaleString()}</td>
                          <td className="num">{missing.toLocaleString()}</td>
                          <td className="mono">
                            {Object.entries(w.distinct_series).map(([c, n]) => `${ALL_VAR_LABEL[c] ?? c} ${n}`).join(" · ")}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="pg__card">
              <div className="pg__card-title">5. NASA POWER against Open-Meteo, 2024–2025</div>
              <div className="pg__card-sub">{d.source_comparison.note}</div>
              <div className="pg__table-wrap">
                <table className="pg__table">
                  <thead>
                    <tr><th>Island</th><th>Variable</th><th className="num">Correlation, same hour</th><th className="num">Best correlation (lag)</th><th className="num">Mean Open-Meteo</th><th className="num">Mean NASA</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(d.source_comparison.islands).flatMap(([isl, per]) =>
                      Object.keys(VAR_LABEL).map((c, i) => {
                        const v = per[c];
                        return (
                          <tr key={`${isl}-${c}`}>
                            <td>{i === 0 ? isl : ""}</td>
                            <td>{VAR_LABEL[c]}</td>
                            <td className="num">{v.corr_same_hour.toFixed(3)}</td>
                            <td className="num">{v.corr_best.toFixed(3)} ({v.best_lag_h > 0 ? "+" : ""}{v.best_lag_h} h)</td>
                            <td className="num">{v.mean_open_meteo.toFixed(2)}</td>
                            <td className="num">{v.mean_nasa_power.toFixed(2)}</td>
                          </tr>
                        );
                      }))}
                  </tbody>
                </table>
              </div>
              <div className="pg__table-wrap">
                <table className="pg__table">
                  <thead><tr><th>Annual rainfall (mm)</th>{Object.keys(d.source_comparison.islands).map((i) => <th key={i} className="num">{i}</th>)}</tr></thead>
                  <tbody>
                    {(["open_meteo", "nasa_power"] as const).map((src) => (
                      <tr key={src}>
                        <td>{src === "open_meteo" ? "Open-Meteo ICON, 2024 / 2025" : "NASA POWER (corrected), 2024 / 2025"}</td>
                        {Object.values(d.source_comparison.islands).map((per, i) => (
                          <td key={i} className="num">{per.annual_rain_mm[src].map((x) => x.toLocaleString()).join(" / ")}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="pg__note">
                ICON rains about twice the regional climate (roughly 1,200–1,300 mm a year). Reported, not
                corrected: each model's hazard scaling is fitted on its own source's training split.
              </div>
            </section>

            <div className="pg__cols">
              <section className="pg__card">
                <div className="pg__card-title">6. State features by quality</div>
                <div className="pg__card-sub">The {d.features.total} features Module 2 receives, by how each is obtained offline.</div>
                <div className="pg__table-wrap">
                  <table className="pg__table">
                    <thead><tr><th>Group</th>{QUALITY_ORDER.map((q) => <th key={q} className="num">{QUALITY_SHORT[q]}</th>)}</tr></thead>
                    <tbody>
                      {Object.entries(d.features.by_group).map(([g, counts]) => (
                        <tr key={g}><td>{g}</td>{QUALITY_ORDER.map((q) => <td key={q} className="num">{counts[q] ?? "·"}</td>)}</tr>
                      ))}
                      <tr className="is-best"><td>Total</td>{QUALITY_ORDER.map((q) => <td key={q} className="num">{d.features.by_quality[q] ?? 0}</td>)}</tr>
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="pg__card">
                <div className="pg__card-title">7. Scenario library</div>
                <div className="pg__card-sub">Periods found by one fixed rule applied to all five plants, from the per-plant ledger.</div>
                <div className="pg__table-wrap">
                  <table className="pg__table">
                    <tbody>
                      {Object.entries(d.scenarios.counts).map(([k, v]) => (
                        <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="num">{v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {d.scenarios.out_of_distribution.map((s) => (
                  <div key={s.scenario_id} className="pg__note">
                    <span className="pg__badge pg__badge--bad">out of distribution</span>{" "}
                    <span className="mono">{s.scenario_id}</span>: {s.plant}, {s.start_month} to {s.end_month}.
                  </div>
                ))}
              </section>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
