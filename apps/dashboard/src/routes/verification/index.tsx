/**
 * Module 4 — Deterministic Physics Verification & Grounded Causal Translation.
 *
 * Evaluates proposed control actions against OpenDSS AC power-flow equations,
 * enforces statutory grid boundaries (0.95 <= V_pu <= 1.05, line thermal ampacities),
 * and generates human-readable abductive causal diagnostic logs for operators.
 *
 * Zero-ML isolated core (ADR 0003).
 */

import { useEffect, useState } from "react";
import {
  type Module4Preset,
  type Module4VerifyResult,
  fetchModule4Presets,
  runModule4Verify,
} from "../../lib";
import "./verification.css";

const SCHEMA_VERSION = "1.0";

function formatViolationType(type: string): string {
  return type.replace(/^VIOLATION_TYPE_/, "");
}

export default function VerificationRoute() {
  const [presets, setPresets] = useState<Record<string, Module4Preset>>({});
  const [selectedKey, setSelectedKey] = useState<string>("nominal_safe");
  const [result, setResult] = useState<Module4VerifyResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load presets on mount with proper error handling
  useEffect(() => {
    fetchModule4Presets()
      .then((data) => {
        setPresets(data);
        const initialKey = data["nominal_safe"] ? "nominal_safe" : Object.keys(data)[0];
        if (initialKey && data[initialKey]) {
          setSelectedKey(initialKey);
          handleRun(data[initialKey].payload);
        }
      })
      .catch((err) => {
        setError(
          err instanceof Error
            ? `Gateway connection error: ${err.message}`
            : "Failed to fetch presets from Gateway API",
        );
      });
  }, []);

  const handleSelectPreset = (key: string) => {
    setSelectedKey(key);
    if (presets[key]) {
      handleRun(presets[key].payload);
    }
  };

  const handleRun = async (payload: Module4Preset["payload"]) => {
    setLoading(true);
    setError(null);
    try {
      const res = await runModule4Verify(payload);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification execution failed");
    } finally {
      setLoading(false);
    }
  };

  const currentPreset = presets[selectedKey];

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
                Module 4 — Deterministic Physics Verification &amp; Grounded Causal Translation
                <span className="verification__badge">m4-verify/{SCHEMA_VERSION}</span>
                <span className="verification__badge verification__badge--core" title="Zero-ML isolated physics core. Sole distribution solver caller.">
                  OpenDSS Direct · Zero-ML Core
                </span>
                {loading && (
                  <span className="verification__badge" style={{ color: "var(--blue)", borderColor: "var(--blue)" }}>
                    solving AC power-flow…
                  </span>
                )}
              </h1>
              <p className="verification__subtitle">
                Sub-millisecond AC power-flow verification against statutory grid bounds (0.95 &le; V &le; 1.05 pu),
                line thermal ampacities, and abductive causal translation for operator diagnostics.
              </p>
            </div>
          </div>
        </header>

        {/* 1. Action Presets & Trigger */}
        <section className="verification__card">
          <div className="verification__card-title">1. Proposed Control Action from Module 3</div>
          <div className="verification__card-sub">
            Select a representative control action to test the physics firewall (3 Approved safe actions vs 3 Rejected unsafe actions).
          </div>

          <div className="verification__row-label">Curated Test Scenarios (3 Approved / 3 Rejected)</div>
          <div className="verification__btns">
            {Object.entries(presets).map(([key, p]) => {
              const isSelected = key === selectedKey;
              const isApproved = p.title.toLowerCase().includes("approved") || p.title.toLowerCase().includes("safe");
              return (
                <button
                  key={key}
                  type="button"
                  disabled={loading}
                  className={`verification__btn ${isSelected ? "" : "verification__btn--ghost"}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    borderColor: isSelected ? (isApproved ? "var(--green)" : "var(--red)") : undefined,
                  }}
                  onClick={() => handleSelectPreset(key)}
                >
                  <span
                    style={{
                      display: "inline-block",
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: isApproved ? "var(--green)" : "var(--red)",
                    }}
                  />
                  <span>{p.title}</span>
                </button>
              );
            })}
          </div>

          {currentPreset && (
            <div className="verification__action-preview">
              <div className="verification__action-meta">
                <span className="mono">action_id: {currentPreset.payload.action_id}</span>
                <span className="mono">origin: {currentPreset.payload.origin}</span>
                <span>rationale: &ldquo;{currentPreset.payload.rationale}&rdquo;</span>
              </div>
              <div className="verification__action-grid">
                <div className="verification__action-item">
                  <strong>Breakers ({currentPreset.payload.breakers.length}):</strong>
                  <div className="mono">
                    {currentPreset.payload.breakers.length === 0
                      ? "None (No switching)"
                      : currentPreset.payload.breakers.map((b) => `${b.edge_id}: ${b.closed ? "CLOSE" : "TRIP"}`).join(", ")}
                  </div>
                </div>
                <div className="verification__action-item">
                  <strong>Load Shed ({currentPreset.payload.load_shed.length}):</strong>
                  <div className="mono">
                    {currentPreset.payload.load_shed.length === 0
                      ? "None (0% shed)"
                      : currentPreset.payload.load_shed.map((ls) => `${ls.node_id}: ${(ls.shed_fraction * 100).toFixed(1)}% (Tier ${ls.priority_tier})`).join(", ")}
                  </div>
                </div>
                <div className="verification__action-item">
                  <strong>Dispatch ({currentPreset.payload.dispatch.length}):</strong>
                  <div className="mono">
                    {currentPreset.payload.dispatch.length === 0
                      ? "None (Baseline)"
                      : currentPreset.payload.dispatch.map((d) => `${d.node_id}: ${d.p_kw.toFixed(1)} kW / ${d.q_kvar.toFixed(1)} kVAR`).join(", ")}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {error && <div style={{ color: "var(--red)", background: "var(--red-soft)", padding: "1rem", borderRadius: "6px" }}>{error}</div>}

        {/* 2. Verification KPIs */}
        {result && (
          <>
            <div className="verification__kpi-row">
              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Firewall Decision</div>
                <div className={`verification__kpi-value mono ${result.decision === "DECISION_APPROVE" ? "verification__kpi-value--approve" : "verification__kpi-value--reject"}`}>
                  {result.decision === "DECISION_APPROVE" ? "APPROVE" : "REJECT"}
                </div>
                <div className="verification__kpi-note">{result.decision === "DECISION_APPROVE" ? "Safe to actuate" : "Blocked by physics"}</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Solve Latency</div>
                <div className="verification__kpi-value mono" style={{ color: "var(--blue)" }}>
                  {result.solve_latency_ms.toFixed(2)} ms
                </div>
                <div className="verification__kpi-note">budget: &lt; 50.0 ms</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Violations Count</div>
                <div className="verification__kpi-value mono" style={{ color: result.violations.length > 0 ? "var(--red)" : "var(--green)" }}>
                  {result.violations.length}
                </div>
                <div className="verification__kpi-note">statutory breaches</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Rejection Severity (S)</div>
                <div className="verification__kpi-value mono" style={{ color: result.rejection_severity > 0 ? "var(--amber)" : "var(--text-muted)" }}>
                  {result.rejection_severity.toFixed(4)}
                </div>
                <div className="verification__kpi-note">normalized for M2 AUQ</div>
              </div>

              <div className="verification__card verification__kpi">
                <div className="verification__kpi-label">Voltage Limits</div>
                <div className="verification__kpi-value mono">
                  0.95 – 1.05
                </div>
                <div className="verification__kpi-note">per-unit statutory bound</div>
              </div>
            </div>

            {/* 3. Operator Causal Explanation Log */}
            <section className={`verification__card verification__causal-card ${result.decision === "DECISION_REJECT" ? "verification__causal-card--reject" : ""}`}>
              <div className="verification__card-title">Operator Diagnostic Explanation (Abductive Causal Log)</div>
              <div className="verification__card-sub">
                Deterministic natural language synthesis grounded strictly in observed physics violations (Grounding Invariant: entities &sube; violations).
              </div>

              <div className="verification__causal-text">
                {result.causal_log.text}
              </div>

              <div className="verification__chip-row">
                <span className="verification__chip-label">Grounded Entities:</span>
                {result.causal_log.grounded_entities.length === 0 ? (
                  <span className="mono" style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>None (Full Grid Normal)</span>
                ) : (
                  result.causal_log.grounded_entities.map((ent) => (
                    <span key={ent} className={`verification__chip ${result.decision === "DECISION_REJECT" ? "verification__chip--reject" : ""}`}>
                      {ent}
                    </span>
                  ))
                )}
                <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "var(--text-dim)" }} className="mono">
                  generator: {result.causal_log.generator}
                </span>
              </div>
            </section>

            {/* 4. Abductive Attribution Breakdown Table (When Violations Occur) */}
            {result.violations.length > 0 && (
              <section className="verification__card" style={{ borderColor: "var(--red)" }}>
                <div className="verification__card-title" style={{ color: "var(--red)" }}>
                  2. Detected Violations &amp; Abductive Root-Cause Attribution
                </div>
                <div className="verification__card-sub">
                  Physical limits exceeded and the inferred causative component from the proposed control action.
                </div>

                <table className="verification__violations-table">
                  <thead>
                    <tr>
                      <th>Violation Type</th>
                      <th>Element ID</th>
                      <th>Measured Value</th>
                      <th>Statutory Limit</th>
                      <th>Margin Fraction</th>
                      <th>Attributed Component</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.violations.map((v, i) => (
                      <tr key={i}>
                        <td className="mono" style={{ fontWeight: 600, color: "var(--red)" }}>
                          {formatViolationType(v.type)}
                        </td>
                        <td className="mono" style={{ fontWeight: 600 }}>{v.element_id}</td>
                        <td className="mono">{v.measured.toFixed(4)}</td>
                        <td className="mono">{v.limit.toFixed(4)}</td>
                        <td className="mono" style={{ color: "var(--red)", fontWeight: 600 }}>
                          {v.margin_fraction >= 0 ? "+" : ""}{(v.margin_fraction * 100).toFixed(2)}%
                        </td>
                        <td className="mono" style={{ color: "var(--amber)", fontWeight: 600 }}>
                          {v.attributed_component || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}

            {/* 5. Nodal Voltage Profile Grid */}
            <section className="verification__card">
              <div className="verification__card-title">
                {result.violations.length > 0 ? "3." : "2."} Bus Voltage Profiles across 3 Islands (OpenDSS AC Power-Flow)
              </div>
              <div className="verification__card-sub">
                Per-unit voltage levels across Nainativu, Analaitivu, and Delft island microgrids. Highlighted red if &lt; 0.95 pu or &gt; 1.05 pu.
              </div>

              <div className="verification__bus-grid">
                {result.buses.map((b) => {
                  const isViol = b.status !== "SAFE";
                  const fillPct = Math.min(100, Math.max(0, (b.voltage_pu / 1.15) * 100));
                  return (
                    <div
                      key={b.bus_name}
                      className={`verification__bus-card ${b.status === "UNDERVOLTAGE" ? "verification__bus-card--undervolt" : b.status === "OVERVOLTAGE" ? "verification__bus-card--overvolt" : ""}`}
                    >
                      <div className="verification__bus-header">
                        <strong className="mono">{b.bus_name}</strong>
                        <span className="mono" style={{ fontWeight: 700, color: isViol ? "var(--red)" : "var(--green)" }}>
                          {b.voltage_pu.toFixed(4)} pu
                        </span>
                      </div>
                      <div className="verification__bus-island">{b.island}</div>
                      <div className="verification__bus-bar-track">
                        <div
                          className={`verification__bus-bar-fill ${isViol ? "verification__bus-bar-fill--viol" : ""}`}
                          style={{ width: `${fillPct}%` }}
                        />
                      </div>
                      <div className="verification__bus-footer">
                        <span>Min: 0.95 pu</span>
                        <span className="mono">{b.status}</span>
                        <span>Max: 1.05 pu</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* 6. Branch Power Flow & Line Loadings */}
            <section className="verification__card">
              <div className="verification__card-title">
                {result.violations.length > 0 ? "4." : "3."} Line Ampacity &amp; Switch Statuses
              </div>
              <div className="verification__card-sub">
                Distribution branch current flows and thermal loading margins computed by the snapshot AC solver.
              </div>

              <table className="verification__lines-table">
                <thead>
                  <tr>
                    <th>Line / Tie-Switch</th>
                    <th>Current (A)</th>
                    <th>Norm Rating (A)</th>
                    <th>Margin Fraction</th>
                    <th>Breaker State</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.lines.map((l) => (
                    <tr key={l.line_name}>
                      <td className="mono" style={{ fontWeight: 600 }}>{l.line_name}</td>
                      <td className="mono">{l.current_amps.toFixed(2)} A</td>
                      <td className="mono">{l.norm_amps.toFixed(2)} A</td>
                      <td className="mono" style={{ color: l.margin_fraction > 0 ? "var(--red)" : "var(--text-muted)" }}>
                        {l.margin_fraction >= 0 ? "+" : ""}{(l.margin_fraction * 100).toFixed(1)}%
                      </td>
                      <td className="mono">{l.is_closed ? "CLOSED" : "OPEN / TRIPPED"}</td>
                      <td>
                        <span className={`verification__status-pill verification__status-pill--${l.status.toLowerCase()}`}>
                          {l.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
