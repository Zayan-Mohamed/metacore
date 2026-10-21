/**
 * Module 4 — Risk-Certified Verification & Grounded Causal Translation.
 *
 * Check A runs the proposed action on a recorded hour through OpenDSS and the limit profile.
 * Check B runs it on 688 seeded scenarios of the next six hours, drawn from conformal bands
 * calibrated on 2024-2025 (adapted by conformal PID in storms), and turns k of N failures into an
 * exact Clopper-Pearson verdict: APPROVE, REJECT or UNVERIFIED. Shapley attribution says which
 * feeder's uncertainty owns the risk; every sentence is grounded in the topology pin.
 *
 * Zero-ML isolated core (ADR 0003).
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  type Module4Decision,
  type Module4Hour,
  type Module4Mode,
  type Module4PidMode,
  type Module4Preset,
  type Module4Profile,
  type Module4VerifyResult,
  type RiskCertificate,
  fetchModule4Hours,
  fetchModule4Presets,
  fetchModule4Profiles,
  runModule4Verify,
} from "../../lib";
import "./verification.css";

const SCHEMA_VERSION = "1.1";
const DEFAULT_PRESET = "fragile_export";

const DECISION_LABEL: Record<Module4Decision, string> = {
  DECISION_APPROVE: "APPROVE",
  DECISION_REJECT: "REJECT",
  DECISION_UNVERIFIED: "UNVERIFIED",
};

const DECISION_NOTE: Record<Module4Decision, string> = {
  DECISION_APPROVE: "Safe to actuate",
  DECISION_REJECT: "Blocked by physics",
  DECISION_UNVERIFIED: "Not established: treat as not approved",
};

function decisionClass(d: Module4Decision): string {
  if (d === "DECISION_APPROVE") return "approve";
  if (d === "DECISION_REJECT") return "reject";
  return "unverified";
}

function formatViolationType(type: string): string {
  return type.replace(/^VIOLATION_TYPE_/, "").replace(/_/g, " ");
}

function pct(x: number, digits = 2): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** The Clopper-Pearson interval drawn against the risk target. */
function RiskInterval({ c }: { c: RiskCertificate }) {
  const max = Math.max(c.epsilon * 3, Math.min(1, c.upper_bound * 1.15), 0.02);
  const x = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  const rate = c.n_scenarios ? c.n_failures / c.n_scenarios : 0;
  return (
    <div className="verification__interval" data-testid="risk-interval">
      <div className="verification__interval-track">
        <div
          className={`verification__interval-range verification__interval-range--${decisionClass(c.decision)}`}
          style={{ left: x(c.lower_bound), width: `calc(${x(c.upper_bound)} - ${x(c.lower_bound)})` }}
        />
        <div className="verification__interval-point" style={{ left: x(rate) }} title={`k/N = ${pct(rate)}`} />
        <div className="verification__interval-eps" style={{ left: x(c.epsilon) }}>
          <span>ε {pct(c.epsilon, 0)}</span>
        </div>
      </div>
      <div className="verification__interval-scale mono">
        <span>0%</span>
        <span>{pct(max, 1)}</span>
      </div>
    </div>
  );
}

export default function VerificationRoute() {
  const [presets, setPresets] = useState<Record<string, Module4Preset>>({});
  const [hours, setHours] = useState<Record<string, Module4Hour>>({});
  const [profiles, setProfiles] = useState<Record<string, Module4Profile>>({});
  const [selectedKey, setSelectedKey] = useState<string>(DEFAULT_PRESET);
  const [hour, setHour] = useState<string>("");
  const [profile, setProfile] = useState<string>("METACORE-STRICT");
  const [mode, setMode] = useState<Module4Mode>("certified");
  const [pid, setPid] = useState<Module4PidMode>("auto");
  const [result, setResult] = useState<Module4VerifyResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchModule4Presets(), fetchModule4Hours(), fetchModule4Profiles()])
      .then(([p, h, pr]) => {
        setPresets(p);
        setHours(h);
        setProfiles(pr);
        const key = p[DEFAULT_PRESET] ? DEFAULT_PRESET : Object.keys(p)[0];
        if (key) {
          setSelectedKey(key);
          setHour(p[key].payload.hour);
        }
      })
      .catch((err) => {
        setError(
          err instanceof Error
            ? `Gateway connection error: ${err.message}`
            : "Failed to load Module 4 presets from the gateway",
        );
      });
  }, []);

  const currentPreset = presets[selectedKey];

  const run = useCallback(async () => {
    if (!currentPreset || !hour) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runModule4Verify({ ...currentPreset.payload, hour, profile, mode, pid });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setLoading(false);
    }
  }, [currentPreset, hour, profile, mode, pid]);

  // Re-run whenever the action or any setting changes.
  useEffect(() => {
    void run();
  }, [run]);

  const handleSelectPreset = (key: string) => {
    setSelectedKey(key);
    if (presets[key]) setHour(presets[key].payload.hour);
  };

  const hourKey = useMemo(
    () => Object.entries(hours).find(([, h]) => h.timestamp === hour)?.[0] ?? "custom",
    [hours, hour],
  );

  const c = result?.certificate ?? null;
  const expected = currentPreset?.expected[mode];
  const maxShare = Math.max(1e-9, ...(result?.attribution ?? []).map((a) => a.share));

  return (
    <div className="verification">
      <div className="verification__inner">
        <header className="verification__header">
          <div className="verification__title-row">
            <div className="verification__mark">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="m9 12 2 2 4-4" />
              </svg>
            </div>
            <div>
              <h1>
                Module 4 — Risk-Certified Verification &amp; Grounded Causal Translation
                <span className="verification__badge">m4-verify/{SCHEMA_VERSION}</span>
                <span className="verification__badge verification__badge--core" title="Zero-ML isolated physics core. Sole distribution solver caller.">
                  OpenDSS · northern-islands-v1 · Zero-ML Core
                </span>
                {loading && (
                  <span className="verification__badge" style={{ color: "var(--blue)", borderColor: "var(--blue)" }}>
                    {mode === "certified" ? "certifying 688 scenarios…" : "solving AC power-flow…"}
                  </span>
                )}
              </h1>
              <p className="verification__subtitle">
                Check A runs the action on a recorded hour. Check B runs it on 688 seeded scenarios of the next six hours
                from calibrated conformal bands, and certifies the risk of breaking a limit with an exact Clopper–Pearson
                bound: APPROVE only when the risk is at most 1% at 99.9% confidence.
              </p>
            </div>
          </div>
        </header>

        {/* 1. Action, hour and gate settings */}
        <section className="verification__card">
          <div className="verification__card-title">Proposed Control Action from Module 3</div>
          <div className="verification__card-sub">
            Each scenario is an action on a recorded hour. The dot shows what the selected gate is expected to decide.
          </div>

          <div className="verification__btns" role="group" aria-label="Action presets">
            {Object.entries(presets).map(([key, p]) => {
              const isSelected = key === selectedKey;
              const exp = p.expected[mode];
              return (
                <button
                  key={key}
                  type="button"
                  disabled={loading}
                  aria-pressed={isSelected}
                  data-testid={`preset-${key}`}
                  className={`verification__btn ${isSelected ? "verification__btn--selected" : "verification__btn--ghost"}`}
                  onClick={() => handleSelectPreset(key)}
                >
                  <span className={`verification__dot verification__dot--${decisionClass(exp)}`} />
                  <span>{p.title}</span>
                </button>
              );
            })}
          </div>

          <div className="verification__controls">
            <label className="verification__control">
              <span>Recorded hour</span>
              <select
                data-testid="hour-select"
                value={hourKey}
                disabled={loading}
                onChange={(e) => {
                  const h = hours[e.target.value];
                  if (h) setHour(h.timestamp);
                }}
              >
                {Object.entries(hours).map(([key, h]) => (
                  <option key={key} value={key}>{h.title}</option>
                ))}
                <option value="custom" disabled>Custom hour</option>
              </select>
            </label>
            <label className="verification__control">
              <span>Any hour, 2024–2025</span>
              <input
                type="datetime-local"
                data-testid="hour-input"
                min="2024-01-01T00:00"
                max="2025-12-31T17:00"
                step={3600}
                value={hour}
                disabled={loading}
                onChange={(e) => e.target.value && setHour(`${e.target.value.slice(0, 13)}:00`)}
              />
            </label>
            <label className="verification__control">
              <span>Limit profile</span>
              <select data-testid="profile-select" value={profile} disabled={loading} onChange={(e) => setProfile(e.target.value)}>
                {Object.values(profiles).map((p) => (
                  <option key={p.profile_id} value={p.profile_id}>
                    {p.profile_id} ({p.v_min_pu.toFixed(2)}–{p.v_max_pu.toFixed(2)} pu)
                  </option>
                ))}
              </select>
            </label>
            <div className="verification__control">
              <span>Gate</span>
              <div className="verification__segmented" role="group" aria-label="Gate mode">
                <button
                  type="button"
                  data-testid="mode-certified"
                  aria-pressed={mode === "certified"}
                  disabled={loading}
                  onClick={() => setMode("certified")}
                >
                  Risk-certified
                </button>
                <button
                  type="button"
                  data-testid="mode-plain"
                  aria-pressed={mode === "plain"}
                  disabled={loading}
                  onClick={() => setMode("plain")}
                >
                  Plain gate
                </button>
              </div>
            </div>
            <label className="verification__control">
              <span>Conformal PID</span>
              <select data-testid="pid-select" value={pid} disabled={loading || mode === "plain"} onChange={(e) => setPid(e.target.value as Module4PidMode)}>
                <option value="auto">Storm mode only</option>
                <option value="on">Always on</option>
                <option value="off">Off (static bands)</option>
              </select>
            </label>
          </div>

          {currentPreset && (
            <div className="verification__action-preview">
              <div className="verification__action-meta">
                <span className="mono">action_id: {currentPreset.payload.action_id}</span>
                <span className="mono">origin: {currentPreset.payload.origin}</span>
                <span>rationale: &ldquo;{currentPreset.payload.rationale}&rdquo;</span>
              </div>
              <p className="verification__action-desc">{currentPreset.description}</p>
              <div className="verification__action-grid">
                <div className="verification__action-item">
                  <strong>Breakers ({currentPreset.payload.breakers.length})</strong>
                  <div className="mono">
                    {currentPreset.payload.breakers.length === 0
                      ? "None (no switching)"
                      : currentPreset.payload.breakers.map((b) => `${b.edge_id}: ${b.closed ? "CLOSE" : "OPEN"}`).join(", ")}
                  </div>
                </div>
                <div className="verification__action-item">
                  <strong>Load shed ({currentPreset.payload.load_shed.length})</strong>
                  <div className="mono">
                    {currentPreset.payload.load_shed.length === 0
                      ? "None (0% shed)"
                      : currentPreset.payload.load_shed.map((ls) => `${ls.node_id}: ${(ls.shed_fraction * 100).toFixed(1)}% (tier ${ls.priority_tier})`).join(", ")}
                  </div>
                </div>
                <div className="verification__action-item">
                  <strong>Dispatch ({currentPreset.payload.dispatch.length})</strong>
                  <div className="mono">
                    {currentPreset.payload.dispatch.length === 0
                      ? "None (baseline)"
                      : currentPreset.payload.dispatch.map((d) => `${d.node_id}: ${d.p_kw.toFixed(1)} kW / ${d.q_kvar.toFixed(1)} kvar`).join(", ")}
                  </div>
                </div>
                <div className="verification__action-item">
                  <strong>Out of service this hour</strong>
                  <div className="mono">
                    {currentPreset.payload.outages.length === 0 ? "As recorded" : `+ ${currentPreset.payload.outages.join(", ")}`}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {error && (
          <div className="verification__error" role="alert">
            {error}
          </div>
        )}

        {result && (
          <>
            {/* 2. KPIs */}
            <div className="verification__kpi-row" data-testid="kpis">
              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Verdict ({result.mode === "certified" ? "certified" : "plain gate"})</div>
                <div className={`verification__kpi-value mono verification__kpi-value--${decisionClass(result.decision)}`} data-testid="verdict">
                  {DECISION_LABEL[result.decision]}
                </div>
                <div className="verification__kpi-note">
                  {DECISION_NOTE[result.decision]}
                  {expected && expected !== result.decision && (
                    <span style={{ color: "var(--amber)" }}> · expected {DECISION_LABEL[expected]}</span>
                  )}
                </div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Check A (snapshot)</div>
                <div className={`verification__kpi-value mono verification__kpi-value--${decisionClass(result.check_a_decision)}`} data-testid="check-a">
                  {DECISION_LABEL[result.check_a_decision]}
                </div>
                <div className="verification__kpi-note mono">{result.hour.replace("T", " ")}</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Risk upper bound</div>
                <div className="verification__kpi-value mono" style={{ color: c ? (c.upper_bound <= c.epsilon ? "var(--green)" : "var(--red)") : "var(--text-dim)" }} data-testid="upper-bound">
                  {c ? pct(c.upper_bound) : "—"}
                </div>
                <div className="verification__kpi-note">{c ? `target ε ≤ ${pct(c.epsilon, 0)} at ${pct(c.confidence, 1)}` : "plain gate: no certificate"}</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Failing scenarios</div>
                <div className="verification__kpi-value mono" data-testid="k-of-n">
                  {c ? `${c.n_failures} / ${c.n_scenarios}` : "—"}
                </div>
                <div className="verification__kpi-note">{c ? (c.stopped_early ? "stopped early" : `of ${c.n_max} planned`) : "next 6 hours not checked"}</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Solve time</div>
                <div className="verification__kpi-value mono" style={{ color: "var(--blue)" }}>
                  {result.solve_latency_ms.toFixed(1)} ms
                </div>
                <div className="verification__kpi-note">{c ? "all power flows of the call" : "one AC power flow"}</div>
              </div>
            </div>

            {/* 3. Risk certificate */}
            {c ? (
              <section className="verification__card verification__risk-card" data-testid="risk-card">
                <div className="verification__card-title">Risk Certificate (Check B)</div>
                <div className="verification__card-sub">
                  k of N likely next-{c.horizon_h}-hour conditions broke a limit. The bar is the exact Clopper–Pearson range for the
                  true rate at {pct(c.confidence, 1)} confidence: green means entirely below ε, red entirely above, amber straddling.
                </div>
                <RiskInterval c={c} />
                <div className="verification__risk-grid">
                  <div>
                    <strong>Range for the true rate</strong>
                    <span className="mono">{pct(c.lower_bound)} – {pct(c.upper_bound)}</span>
                  </div>
                  <div>
                    <strong>Situation</strong>
                    <span className="mono">{String(c.situation.hour).padStart(2, "0")}:00 · {c.situation.season} · {c.situation.hazard}</span>
                  </div>
                  <div>
                    <strong>Band (group {c.band.key})</strong>
                    <span className="mono">+{pct(c.band.q_up ?? 0, 1)} / −{pct(c.band.q_down ?? 0, 1)} · n={c.band.n}</span>
                  </div>
                  <div>
                    <strong>Conformal PID</strong>
                    <span className="mono" data-testid="pid-state">
                      {c.pid === null
                        ? "off"
                        : c.pid.enabled
                          ? `on${c.pid.storm ? " (storm)" : ""} · +${pct(c.pid.extra ?? 0, 1)} (P ${pct(c.pid.p_offset ?? 0, 1)}, I ${pct(c.pid.i_term ?? 0, 1)}, D ${pct(c.pid.d_term ?? 0, 1)})`
                          : "standing by (calm)"}
                    </span>
                  </div>
                  <div>
                    <strong>Band corners</strong>
                    <span className="mono">
                      top {c.corner_top_safe ? "safe" : "fails"} · bottom {c.corner_bottom_safe ? "safe" : "fails"}
                    </span>
                  </div>
                  <div>
                    <strong>Replay</strong>
                    <span className="mono" title={c.calibration_sha256}>
                      {c.calibration_version} · seed {c.seed} · {c.limit_profile}
                    </span>
                  </div>
                </div>
                <p className="verification__caveat" data-testid="risk-caveat">
                  This risk rests on hourly demand downscaled from CEB monthly totals, not metered load: its misses are smoother
                  than real ones, so the bands may understate sudden swings.
                </p>
                {c.notes.length > 0 && (
                  <ul className="verification__notes">
                    {c.notes.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                )}
              </section>
            ) : (
              <section className="verification__card verification__risk-card verification__risk-card--muted" data-testid="risk-card">
                <div className="verification__card-title">Risk Certificate (Check B)</div>
                <div className="verification__card-sub" style={{ marginBottom: 0 }}>
                  {result.mode === "plain"
                    ? "Plain gate (baseline B0): only this hour was checked. Switch to Risk-certified to check the next six hours."
                    : result.decision === "DECISION_UNVERIFIED"
                      ? `Not issued: ${result.unverified_reasons.join("; ")}`
                      : "Not needed: the action already breaks a limit on this hour (Check A)."}
                </div>
              </section>
            )}

            {/* 4. Risk attribution */}
            {result.attribution.length > 0 && (
              <section className="verification__card" data-testid="attribution">
                <div className="verification__card-title">Risk Attribution (Shapley shares by feeder)</div>
                <div className="verification__card-sub">
                  Which feeder's demand uncertainty owns the risk. Shares add up exactly to the failure rate; “alone” is the rate when
                  only that feeder moves.
                </div>
                <div className="verification__attr-list">
                  {result.attribution.map((a) => (
                    <div key={a.player} className="verification__attr-row">
                      <div className="verification__attr-name">
                        <strong className="mono">{a.player}</strong>
                        <span>{a.buses.join(" + ")}</span>
                      </div>
                      <div className="verification__attr-track">
                        <div className="verification__attr-fill" style={{ width: `${(a.share / maxShare) * 100}%` }} />
                      </div>
                      <div className="verification__attr-value mono">
                        {pct(a.share)} <span>({pct(a.fraction_of_risk, 0)} of risk · alone {pct(a.alone)})</span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* 5. Operator explanation */}
            <section className={`verification__card verification__causal-card verification__causal-card--${decisionClass(result.decision)}`}>
              <div className="verification__card-title">Operator Explanation (grounded causal log)</div>
              <div className="verification__card-sub">
                Built only from the structured evidence; every network name is checked against the northern-islands-v1 topology pin.
              </div>
              <div className="verification__causal-text" data-testid="causal-log">{result.causal_log.text}</div>
              <div className="verification__chip-row">
                <span className="verification__chip-label">Grounded entities:</span>
                {result.causal_log.grounded_entities.length === 0 ? (
                  <span className="mono" style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>None (grid normal)</span>
                ) : (
                  result.causal_log.grounded_entities.map((ent) => (
                    <span key={ent} className={`verification__chip ${result.decision !== "DECISION_APPROVE" ? "verification__chip--reject" : ""}`}>
                      {ent}
                    </span>
                  ))
                )}
                <span
                  className={`verification__grounding ${result.causal_log.grounded ? "" : "verification__grounding--bad"}`}
                  data-testid="grounding"
                >
                  {result.causal_log.grounded ? "✓ grounded in the topology" : `✗ ungrounded: ${result.causal_log.ungrounded.join(", ")}`}
                </span>
              </div>
            </section>

            {/* 6. Violations */}
            {result.violations.length > 0 && (
              <section className="verification__card" style={{ borderColor: "var(--red)" }}>
                <div className="verification__card-title" style={{ color: "var(--red)" }}>
                  Detected Violations &amp; Abductive Attribution
                </div>
                <div className="verification__card-sub">
                  {c && c.n_failures > 0
                    ? "Worst case per element across the failing scenarios, and how many scenarios it failed in."
                    : "Limits exceeded on the recorded hour, and the action component that most plausibly caused each."}
                </div>
                <div className="verification__table-wrap">
                  <table className="verification__violations-table">
                    <thead>
                      <tr>
                        <th>Violation</th>
                        <th>Element</th>
                        <th>Measured</th>
                        <th>Limit</th>
                        <th>Margin</th>
                        <th>Attributed to</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.violations.map((v, i) => (
                        <tr key={i}>
                          <td className="mono" style={{ fontWeight: 600, color: "var(--red)" }}>{formatViolationType(v.type)}</td>
                          <td className="mono" style={{ fontWeight: 600 }}>{v.element_id}</td>
                          <td className="mono">{v.measured.toFixed(4)}</td>
                          <td className="mono">{v.limit.toFixed(4)}</td>
                          <td className="mono" style={{ color: "var(--red)", fontWeight: 600 }}>
                            {v.margin_fraction >= 0 ? "+" : ""}{(v.margin_fraction * 100).toFixed(2)}%
                          </td>
                          <td className="mono" style={{ color: "var(--amber)", fontWeight: 600 }}>{v.attributed_component || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {/* 7. Generation adequacy */}
            <section className="verification__card">
              <div className="verification__card-title">Generation Adequacy per Island</div>
              <div className="verification__card-sub">
                Island demand against the in-service generating capacity (the lead genset is a stiff source in the model, so this is
                checked separately).
              </div>
              <div className="verification__adequacy" data-testid="adequacy">
                {result.adequacy.map((a) => {
                  const fill = Math.min(100, (a.demand_kva / Math.max(1, a.capacity_kva)) * 100);
                  return (
                    <div key={a.island} className={`verification__adequacy-item ${a.status === "INADEQUATE" ? "verification__adequacy-item--bad" : ""}`}>
                      <div className="verification__bus-header">
                        <strong>{a.island}</strong>
                        <span className="mono">{a.demand_kva.toFixed(0)} / {a.capacity_kva.toFixed(0)} kVA</span>
                      </div>
                      <div className="verification__bus-bar-track">
                        <div className={`verification__bus-bar-fill ${a.status === "INADEQUATE" ? "verification__bus-bar-fill--viol" : ""}`} style={{ width: `${fill}%` }} />
                      </div>
                      <div className="verification__bus-footer">
                        <span className="mono">{a.station}</span>
                        <span className="mono">{a.status}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* 8. Bus voltages */}
            <section className="verification__card">
              <div className="verification__card-title">Bus Voltages on the Recorded Hour (OpenDSS AC Power Flow)</div>
              <div className="verification__card-sub">
                Per-unit voltage across the four island microgrids with the action applied. Red outside {result.v_min_pu.toFixed(2)}–
                {result.v_max_pu.toFixed(2)} pu ({result.profile}).
              </div>
              <div className="verification__bus-grid">
                {result.buses.map((b) => {
                  const isViol = b.status !== "SAFE";
                  const fillPct = Math.min(100, Math.max(0, ((b.voltage_pu - 0.85) / 0.3) * 100));
                  return (
                    <div
                      key={b.bus_name}
                      className={`verification__bus-card ${isViol ? "verification__bus-card--undervolt" : ""}`}
                    >
                      <div className="verification__bus-header">
                        <strong className="mono">{b.bus_name}</strong>
                        <span className="mono" style={{ fontWeight: 700, color: isViol ? "var(--red)" : "var(--green)" }}>
                          {b.voltage_pu.toFixed(4)} pu
                        </span>
                      </div>
                      <div className="verification__bus-island">{b.island}</div>
                      <div className="verification__bus-bar-track">
                        <div className={`verification__bus-bar-fill ${isViol ? "verification__bus-bar-fill--viol" : ""}`} style={{ width: `${fillPct}%` }} />
                      </div>
                      <div className="verification__bus-footer">
                        <span>Min {result.v_min_pu.toFixed(2)}</span>
                        <span className="mono">{b.status}</span>
                        <span>Max {result.v_max_pu.toFixed(2)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* 9. Lines */}
            <section className="verification__card">
              <div className="verification__card-title">Feeder Currents &amp; Ratings</div>
              <div className="verification__card-sub">Feeder currents and thermal loading on the recorded hour, from the AC solver.</div>
              <div className="verification__table-wrap">
                <table className="verification__lines-table">
                  <thead>
                    <tr>
                      <th>Line</th>
                      <th>Current (A)</th>
                      <th>Rating (A)</th>
                      <th>Margin</th>
                      <th>Breaker</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.lines.map((l) => (
                      <tr key={l.line_name}>
                        <td className="mono" style={{ fontWeight: 600 }}>{l.line_name}</td>
                        <td className="mono">{l.current_amps.toFixed(2)}</td>
                        <td className="mono">{l.norm_amps.toFixed(0)}</td>
                        <td className="mono" style={{ color: l.margin_fraction > 0 ? "var(--red)" : "var(--text-muted)" }}>
                          {l.margin_fraction >= 0 ? "+" : ""}{(l.margin_fraction * 100).toFixed(1)}%
                        </td>
                        <td className="mono">{l.is_closed ? "CLOSED" : "OPEN"}</td>
                        <td>
                          <span className={`verification__status-pill verification__status-pill--${l.status.toLowerCase()}`}>{l.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="verification__footnote mono">
                snapshot: {result.snapshot_source}
                {result.unverified_reasons.length > 0 && ` · unverified: ${result.unverified_reasons.join("; ")}`}
              </div>
            </section>
          </>
        )}
        {!result && !error && (
          <div className="verification__card verification__card--loading">Loading the gateway…</div>
        )}
      </div>
    </div>
  );
}
