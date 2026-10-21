// Module 3 — System 1 / System 2 decisions and the deliberation cost incurred.
// A run configuration form drives a real train+eval pass of the gating policy
// (services/learned/module3_metapolicy/scripts/demo_run.py, via gateway's /api/module3/run)
// and renders the result: reward vs. baselines, escalation by severity / trigger_reason,
// and the paired ProposedControlAction/GatingDecision log.
import { useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router-dom";

import {
  MODULE3_DEFAULT_CONFIG,
  Module3RunError,
  generateModule3Situation,
  pairModule3Decisions,
  processModule3Gate,
  runModule3,
} from "../../lib";
import type {
  Module3DecisionRow,
  Module3GenerateResult,
  Module3ProcessResult,
  Module3RunConfig,
  Module3RunResult,
  ObservationField,
} from "../../lib";
import "./gating.css";

const SEVERITY_ORDER = ["normal", "elevated", "severe", "extreme"];
const REASON_ORDER = ["none", "value", "sensing", "both"];
const REASON_COLOR: Record<string, string> = {
  none: "var(--text-muted)",
  value: "var(--amber)",
  sensing: "var(--violet)",
  both: "var(--amber)",
};

type Status = "idle" | "running" | "done" | "error";

interface FieldSpec {
  key: keyof Module3RunConfig;
  label: string;
  min: number;
  max: number;
  step: number;
  hint: string;
}

const FIELDS: FieldSpec[] = [
  { key: "seed", label: "Seed", min: 0, max: 10_000, step: 1, hint: "RNG seed" },
  { key: "episodeLen", label: "Episode length", min: 20, max: 200, step: 1, hint: "steps/episode" },
  { key: "budgetPerEpisode", label: "Budget / episode", min: 1, max: 200, step: 1, hint: "S2 calls allowed" },
  { key: "trainEpisodes", label: "Train episodes", min: 10, max: 300, step: 1, hint: "storms for PPO to train on" },
  { key: "evalEpisodes", label: "Eval episodes", min: 1, max: 50, step: 1, hint: "averaged for metrics" },
  { key: "deliberationCost", label: "Deliberation cost", min: 0, max: 5, step: 0.01, hint: "reward penalty for S2" },
  { key: "wServe", label: "Load-served weight", min: 0, max: 10, step: 0.1, hint: "reward for load kept on" },
  { key: "wViol", label: "Violation weight", min: 0, max: 20, step: 0.1, hint: "penalty for flagged buses" },
];

function formatObsValue(field: ObservationField): string {
  if (field.name === "reason_value" || field.name === "reason_sensing") {
    return field.value >= 0.5 ? "1" : "0";
  }
  return field.value.toFixed(4);
}

function ObservationGrid({
  observation,
  raw,
  stepIndex,
  episodeLen,
}: {
  observation: ObservationField[];
  raw: Module3GenerateResult["raw"];
  stepIndex: number;
  episodeLen: number;
}) {
  return (
    <div className="gating__card" style={{ marginTop: 14 }}>
      <div className="gating__card-title">2. Generated input — what the gate sees</div>
      <div className="gating__card-sub">
        step {stepIndex + 1} of {episodeLen} in this simulated cyclone · u = {raw.epistemic_uncertainty.toFixed(4)} ·{" "}
        {raw.severity} / {raw.trigger_reason} · M2 numbers replay Duwaragie’s published mock; storm and grid damage are simulated
      </div>
      <div className="gating__obs-grid">
        {observation.map((field) => (
          <div className="gating__obs-cell" key={field.index}>
            <div className="gating__obs-idx mono">[{field.index}]</div>
            <div className="gating__obs-name mono">{field.name}</div>
            <div className="gating__obs-val mono">{formatObsValue(field)}</div>
            <div className="gating__obs-src">
              <span className={`gating__src-pill gating__src-pill--${field.source}`}>{field.source}</span>
              {field.meaning}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PlanCard({ processed }: { processed: Module3ProcessResult }) {
  const { plan } = processed;
  const isS1 = plan.origin === "SYSTEM1";
  return (
    <div className="gating__card" style={{ marginTop: 14 }}>
      <div className="gating__card-title">3. Action plan — what we hand to M4</div>
      <div className="gating__card-sub">
        gate picked {processed.chosen}
        {processed.requested !== processed.chosen ? ` (asked ${processed.requested}, budget forced S1)` : ""}
        {" · "}
        P(S1)={processed.action_probs.SYSTEM1.toFixed(2)} / P(S2)={processed.action_probs.SYSTEM2.toFixed(2)}
        {" · "}
        u at decision = {processed.epistemic_at_decision.toFixed(4)}
      </div>
      <div className="gating__plan">
        <div className="gating__plan-head">
          <span
            className="mono gating__pill"
            style={{
              background: isS1 ? "var(--blue-soft)" : "var(--amber-soft)",
              color: isS1 ? "var(--blue)" : "var(--amber)",
            }}
          >
            {plan.origin}
          </span>
          <span className="gating__plan-rationale">{plan.rationale}</span>
        </div>
        <div className="gating__plan-row">
          <span className="label">breakers</span>
          {plan.breakers.length === 0 ? (
            <span className="gating__detail-empty">none</span>
          ) : (
            <span>
              {plan.breakers.map((b) => `${b.edge_id} ${b.closed ? "CLOSED" : "OPEN"}`).join(", ")}
            </span>
          )}
        </div>
        <div className="gating__plan-row">
          <span className="label">load_shed</span>
          {plan.load_shed.length === 0 ? (
            <span className="gating__detail-empty">none</span>
          ) : (
            <span>
              {plan.load_shed
                .map((ls) => `${ls.node_id} ${(ls.shed_fraction * 100).toFixed(1)}% (tier ${ls.priority_tier})`)
                .join(" · ")}
            </span>
          )}
        </div>
        <div className="gating__plan-row">
          <span className="label">dispatch</span>
          {plan.dispatch.length === 0 ? (
            <span className="gating__detail-empty">none</span>
          ) : (
            <span>
              {plan.dispatch.map((d) => `${d.node_id} ${d.p_kw.toFixed(1)} kW, ${d.q_kvar.toFixed(1)} kvar`).join(" · ")}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function CheckIcon({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

// The M2 input state that produced this decision, and the full proposed action plan
// (breakers/load_shed/dispatch) System1/System2 returned — neither travels over the
// M3->M4 wire contract itself, both come from serve/demo.py's dashboard-only `context`.
function DecisionDetail({ row }: { row: Module3DecisionRow }) {
  const ctx = row.context;
  const { breakers, load_shed, dispatch } = row.action;

  return (
    <div className="gating__detail">
      <div>
        <div className="gating__detail-section-title">M2 input @ this step</div>
        {ctx ? (
          <dl className="gating__detail-kv">
            <dt>severity</dt>
            <dd>{ctx.severity}</dd>
            <dt>trigger_reason</dt>
            <dd>{ctx.trigger_reason}</dd>
            <dt>epistemic_uncertainty (u)</dt>
            <dd>{ctx.epistemic_uncertainty.toFixed(4)}</dd>
            <dt>observed_fraction</dt>
            <dd>{ctx.observed_fraction.toFixed(4)}</dd>
            <dt>competence_drop</dt>
            <dd>{String(ctx.competence_drop)}</dd>
            <dt>step reward</dt>
            <dd>{ctx.reward.toFixed(3)}</dd>
          </dl>
        ) : (
          <div className="gating__detail-empty">no input context recorded for this decision</div>
        )}

        <div className="gating__detail-section-title" style={{ marginTop: 16 }}>
          Grid twin verdict (surrogate, not the real M4)
        </div>
        {ctx ? (
          <>
            <span
              className="mono gating__verdict-pill"
              style={{
                background: ctx.verdict === "APPROVE" ? "var(--green-soft)" : "var(--red-soft)",
                color: ctx.verdict === "APPROVE" ? "var(--green)" : "var(--red)",
              }}
            >
              {ctx.verdict}
            </span>
            <div className="gating__detail-list" style={{ marginTop: 8 }}>
              <div className="gating__detail-list-item">
                load served: <b>{(ctx.served_fraction * 100).toFixed(0)}%</b>
              </div>
              {ctx.violating_buses.length > 0 && (
                <div className="gating__detail-list-item">
                  flagged buses: <b>{ctx.violating_buses.join(", ")}</b>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="gating__detail-empty">no verdict recorded</div>
        )}
      </div>

      <div>
        <div className="gating__detail-section-title">Proposed action plan ({row.action.origin})</div>

        <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>breakers</div>
        {breakers.length > 0 ? (
          <div className="gating__detail-list">
            {breakers.map((b, i) => (
              <div className="gating__detail-list-item" key={i}>
                <b>{b.edge_id}</b> — {b.closed ? "closed" : "open"}
              </div>
            ))}
          </div>
        ) : (
          <div className="gating__detail-empty" style={{ marginBottom: 12 }}>
            none
          </div>
        )}

        <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>load_shed</div>
        {load_shed.length > 0 ? (
          <div className="gating__detail-list">
            {load_shed.map((ls, i) => (
              <div className="gating__detail-list-item" key={i}>
                <b>{ls.node_id}</b> — shed {(ls.shed_fraction * 100).toFixed(1)}% (priority tier{" "}
                {ls.priority_tier})
              </div>
            ))}
          </div>
        ) : (
          <div className="gating__detail-empty" style={{ marginBottom: 12 }}>
            none
          </div>
        )}

        <div style={{ fontSize: 11, color: "var(--text-dim)", marginBottom: 4 }}>dispatch</div>
        {dispatch.length > 0 ? (
          <div className="gating__detail-list">
            {dispatch.map((d, i) => (
              <div className="gating__detail-list-item" key={i}>
                <b>{d.node_id}</b> — {d.p_kw.toFixed(1)} kW, {d.q_kvar.toFixed(1)} kVAR
              </div>
            ))}
          </div>
        ) : (
          <div className="gating__detail-empty">none</div>
        )}
      </div>
    </div>
  );
}

export default function GatingRoute() {
  const [config, setConfig] = useState<Module3RunConfig>(MODULE3_DEFAULT_CONFIG);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Module3RunResult | null>(null);
  const [situation, setSituation] = useState<Module3GenerateResult | null>(null);
  const [processed, setProcessed] = useState<Module3ProcessResult | null>(null);
  const [busy, setBusy] = useState<"idle" | "train" | "generate" | "process">("idle");
  const [filter, setFilter] = useState<"ALL" | "SYSTEM1" | "SYSTEM2">("ALL");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  function updateField(key: keyof Module3RunConfig, value: string) {
    const n = Number(value);
    setConfig((c) => ({ ...c, [key]: Number.isFinite(n) ? n : c[key] }));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy("train");
    setStatus("running");
    setError(null);
    setSituation(null);
    setProcessed(null);
    try {
      const r = await runModule3(config);
      setResult(r);
      setStatus("done");
    } catch (err) {
      const message = err instanceof Module3RunError ? err.message : "Run failed unexpectedly";
      setError(message);
      setStatus("error");
    } finally {
      setBusy("idle");
    }
  }

  async function handleGenerate() {
    setBusy("generate");
    setError(null);
    setProcessed(null);
    try {
      setSituation(await generateModule3Situation());
    } catch (err) {
      const message = err instanceof Module3RunError ? err.message : "Generate failed";
      setError(message);
    } finally {
      setBusy("idle");
    }
  }

  async function handleProcess() {
    setBusy("process");
    setError(null);
    try {
      setProcessed(await processModule3Gate());
    } catch (err) {
      const message = err instanceof Module3RunError ? err.message : "Process failed";
      setError(message);
    } finally {
      setBusy("idle");
    }
  }

  const rows = result ? pairModule3Decisions(result.decisions, result.decision_context) : [];
  const filteredRows = rows.filter((r) => filter === "ALL" || r.action.origin === filter);

  const rewardBars = result
    ? [
        { label: "Trained policy", value: result.reward.trained_policy, color: "var(--green)" },
        { label: "Breaker rule", value: result.reward.breaker_rule, color: "var(--text-muted)" },
        { label: "u-threshold baseline", value: result.reward.threshold, color: "var(--text-muted)" },
        { label: "Always-S2 baseline", value: result.reward.always_s2, color: "var(--text-muted)" },
        { label: "Always-S1 baseline", value: result.reward.always_s1, color: "var(--text-muted)" },
        { label: "Oracle (reads true damage)", value: result.reward.oracle, color: "var(--text-muted)" },
      ]
    : [];
  const rewardScale = result
    ? Math.max(1, ...Object.values(result.reward).map((v) => Math.abs(v)))
    : 1;

  const baselineCandidates = result
    ? [
        { label: "always-S1", value: result.reward.always_s1 },
        { label: "always-S2", value: result.reward.always_s2 },
        { label: "u-threshold", value: result.reward.threshold },
        { label: "breaker-rule", value: result.reward.breaker_rule },
      ]
    : [];
  const bestOfBaselines = baselineCandidates.length
    ? baselineCandidates.reduce((best, row) => (row.value > best.value ? row : best))
    : { label: "baseline", value: 0 };
  const bestBaseline = bestOfBaselines.value;
  const bestBaselineLabel = bestOfBaselines.label;
  const deltaVsBaseline = result ? result.reward.trained_policy - bestBaseline : 0;

  return (
    <div className="gating">
      <div className="gating__inner">
        <header className="gating__header">
          <div className="gating__title-row">
            <div className="gating__mark">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 12h4l3 8 4-16 3 8h4" />
              </svg>
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <h1>Module 3 — Gating &amp; Meta-Policy</h1>
                <span className="mono gating__badge">m3-out/0.1</span>
              </div>
              <p className="gating__subtitle">
                Three steps: <b>Train</b> the chooser, <b>Generate</b> one situation (the 15 numbers
                the gate sees, including M2’s u), then <b>Process</b> to get the action plan.{" "}
                <Link to="/cyclone">Set your own cyclone and test the PPO gate →</Link>
              </p>
            </div>
          </div>
        </header>

        <form className="gating__card gating__run-card" onSubmit={handleSubmit}>
          <div className="gating__card-title">1. Train the gate</div>
          <div className="gating__card-sub">
            Practice storms only — these knobs are not M2’s u. Train, then Generate a situation, then Process it.
          </div>
          <div className="gating__form-grid">
            {FIELDS.map((f) => (
              <div className="gating__field" key={f.key}>
                <label htmlFor={f.key}>{f.label}</label>
                <input
                  id={f.key}
                  type="number"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={config[f.key]}
                  onChange={(e) => updateField(f.key, e.target.value)}
                  disabled={busy !== "idle"}
                />
                <span className="gating__field-hint">{f.hint}</span>
              </div>
            ))}
          </div>
          <div className="gating__form-footer">
            <div className="gating__steps">
              <button className="gating__run-btn" type="submit" disabled={busy !== "idle"}>
                {busy === "train" ? "Training…" : "1. Train"}
              </button>
              <button
                className="gating__run-btn gating__run-btn--ghost"
                type="button"
                disabled={busy !== "idle" || !result}
                onClick={handleGenerate}
              >
                {busy === "generate" ? "Generating…" : "2. Generate situation"}
              </button>
              <button
                className="gating__run-btn gating__run-btn--ghost"
                type="button"
                disabled={busy !== "idle" || !situation}
                onClick={handleProcess}
              >
                {busy === "process" ? "Processing…" : "3. Process gate"}
              </button>
            </div>
            {busy === "train" && (
              <div className="gating__run-status">
                <span className="gating__spinner" />
                training the chooser on fake storms + Duwaragie’s file…
              </div>
            )}
          </div>
          {error && <div className="gating__error">{error}</div>}
        </form>

        {situation && (
          <ObservationGrid
            observation={situation.observation}
            raw={situation.raw}
            stepIndex={situation.step_index}
            episodeLen={situation.episode_len}
          />
        )}
        {processed && <PlanCard processed={processed} />}

        {!result && status !== "error" && (
          <div className="gating__empty">
            {status === "running"
              ? "Training the chooser — Generate and Process unlock after this finishes."
              : "Train the gate first. Then Generate a situation (you will see the 15 inputs, including u). Then Process to get the plan."}
          </div>
        )}

        {result && (
          <>
            <div className="gating__kpi-row">
              <div className="gating__card">
                <div className="gating__kpi-label">Trained policy reward</div>
                <div className="gating__kpi-value mono" style={{ color: "var(--green)" }}>
                  {result.reward.trained_policy >= 0 ? "+" : ""}
                  {result.reward.trained_policy.toFixed(2)}
                </div>
                <div className="gating__kpi-note">total episode reward</div>
              </div>
              <div className="gating__card">
                <div className="gating__kpi-label">Vs. best baseline</div>
                <div className="gating__kpi-value mono">
                  {deltaVsBaseline >= 0 ? "+" : ""}
                  {deltaVsBaseline.toFixed(1)}
                </div>
                <div className="gating__kpi-note">
                  beats {bestBaselineLabel} ({bestBaseline.toFixed(2)})
                </div>
              </div>
              <div className="gating__card">
                <div className="gating__kpi-label">Avg. deliberation cost</div>
                <div className="gating__kpi-value mono">
                  {result.avg_deliberation_cost.trained_policy.toFixed(3)}
                </div>
                <div className="gating__kpi-note">
                  vs {result.avg_deliberation_cost.threshold.toFixed(3)} threshold /{" "}
                  {result.avg_deliberation_cost.always_s2.toFixed(3)} always-S2
                </div>
              </div>
              <div className="gating__card">
                <div className="gating__kpi-label">Monotonicity check</div>
                <div className="gating__pass">
                  <span
                    className="gating__pass-icon"
                    style={{
                      background: result.monotonic_nondecreasing ? "var(--green-soft)" : "var(--red-soft)",
                      color: result.monotonic_nondecreasing ? "var(--green)" : "var(--red)",
                    }}
                  >
                    {result.monotonic_nondecreasing ? (
                      <CheckIcon />
                    ) : (
                      <span className="mono" style={{ fontSize: 12, fontWeight: 700 }}>
                        !
                      </span>
                    )}
                  </span>
                  <span className="mono" style={{ fontSize: 18, fontWeight: 600, color: result.monotonic_nondecreasing ? "var(--green)" : "var(--red)" }}>
                    {result.monotonic_nondecreasing ? "PASS" : "FAIL"}
                  </span>
                </div>
                <div className="gating__kpi-note">escalation non-decreasing across severity</div>
              </div>
            </div>

            <div className="gating__card" style={{ marginTop: 14 }}>
              <div className="gating__card-title">Reward vs. baselines</div>
              <div className="gating__card-sub">
                total episode reward — trained policy vs. always-S1 / always-S2 / u&gt;threshold / breaker rule, with the oracle as a ceiling
              </div>
              <div className="gating__bars">
                {rewardBars.map((bar) => {
                  const magPct = Math.min(50, (Math.abs(bar.value) / rewardScale) * 50);
                  const leftPct = bar.value < 0 ? 50 - magPct : 50;
                  return (
                    <div className="gating__bar-row" key={bar.label}>
                      <div className="gating__bar-label">{bar.label}</div>
                      <div className="gating__bar-track">
                        <div className="gating__bar-zero" />
                        <div
                          className="gating__bar-fill"
                          style={{ left: `${leftPct}%`, width: `${magPct}%`, background: bar.color }}
                        />
                      </div>
                      <div className="gating__bar-value mono" style={{ color: bar.color }}>
                        {bar.value >= 0 ? "+" : ""}
                        {bar.value.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="gating__charts-row">
              <div className="gating__card">
                <div className="gating__card-title">Escalation rate by hazard severity</div>
                <div className="gating__card-sub">
                  excludes sensing-triggered steps · non-decreasing = core thesis
                </div>
                <div className="gating__chart">
                  {SEVERITY_ORDER.map((label) => {
                    const value = result.escalation_by_severity[label] ?? 0;
                    return (
                      <div className="gating__chart-col" key={label}>
                        <div className="gating__chart-val mono">{value.toFixed(3)}</div>
                        <div className="gating__chart-track">
                          <div
                            className="gating__chart-fill"
                            style={{ height: `${Math.max(3, value * 100)}%`, background: "var(--amber)" }}
                          />
                        </div>
                        <div className="gating__chart-label">{label}</div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="gating__card">
                <div className="gating__card-title">Escalation rate by trigger_reason</div>
                <div className="gating__card-sub">cause-aware split — sensing loss stays on System 1</div>
                <div className="gating__chart">
                  {REASON_ORDER.map((reason) => {
                    const value = result.escalation_by_trigger_reason[reason] ?? 0;
                    return (
                      <div className="gating__chart-col" key={reason}>
                        <div className="gating__chart-val mono">{value.toFixed(3)}</div>
                        <div className="gating__chart-track">
                          <div
                            className="gating__chart-fill"
                            style={{ height: `${Math.max(3, value * 100)}%`, background: REASON_COLOR[reason] }}
                          />
                        </div>
                        <div className="gating__chart-label">{reason}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="gating__card gating__log-card">
              <div className="gating__log-header">
                <div>
                  <div className="gating__card-title">Decision log</div>
                  <div className="gating__card-sub">
                    this run · showing {filteredRows.length} of {rows.length}
                  </div>
                </div>
                <div className="gating__chips">
                  {(["ALL", "SYSTEM1", "SYSTEM2"] as const).map((f) => (
                    <button
                      key={f}
                      type="button"
                      className="gating__chip"
                      onClick={() => setFilter(f)}
                      style={{
                        background: filter === f ? (f === "SYSTEM1" ? "var(--blue-soft)" : f === "SYSTEM2" ? "var(--amber-soft)" : "var(--text)") : "transparent",
                        color: filter === f ? (f === "SYSTEM1" ? "var(--blue)" : f === "SYSTEM2" ? "var(--amber)" : "var(--bg)") : "var(--text-muted)",
                      }}
                    >
                      {f === "ALL" ? "All" : f === "SYSTEM1" ? "System 1" : "System 2"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="gating__log-table">
                <div className="gating__log-row gating__log-head">
                  <span>Action</span>
                  <span>Path</span>
                  <span>Trigger</span>
                  <span>Epistemic @ decision</span>
                  <span>Cost</span>
                  <span>Latency</span>
                  <span>Rationale</span>
                </div>
                {filteredRows.map((row) => {
                  const isS1 = row.action.origin === "SYSTEM1";
                  const bg = isS1 ? "var(--blue-soft)" : "var(--amber-soft)";
                  const fg = isS1 ? "var(--blue)" : "var(--amber)";
                  const isSensingFallback = row.action.rationale.includes("sensing-fallback");
                  const isExpanded = expandedId === row.actionId;
                  const reason = row.context?.trigger_reason ?? "none";
                  return (
                    <div key={row.actionId}>
                      <div
                        className="gating__log-row"
                        style={{ cursor: "pointer" }}
                        onClick={() => setExpandedId(isExpanded ? null : row.actionId)}
                      >
                        <div className="mono" style={{ fontSize: 12, color: "var(--text-dim)", display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ display: "inline-block", transform: isExpanded ? "rotate(90deg)" : "none", transition: "transform 0.1s" }}>
                            ▸
                          </span>
                          {row.actionId.slice(0, 8)}
                        </div>
                        <div>
                          <span className="mono gating__pill" style={{ background: bg, color: fg }}>
                            {isS1 ? "S1" : "S2"}
                          </span>
                        </div>
                        <div>
                          {row.context ? (
                            <span
                              className="mono gating__pill"
                              style={{ background: "var(--surface-2)", color: REASON_COLOR[reason] }}
                              title={`severity: ${row.context.severity}`}
                            >
                              {row.context.severity} / {reason}
                            </span>
                          ) : (
                            <span style={{ fontSize: 11, color: "var(--text-dim)" }}>—</span>
                          )}
                        </div>
                        <div className="gating__gauge">
                          <div className="gating__gauge-track">
                            <div
                              className="gating__gauge-fill"
                              style={{ width: `${Math.round(row.decision.epistemic_at_decision * 100)}%`, background: fg }}
                            />
                          </div>
                          <div className="gating__gauge-val mono">{row.decision.epistemic_at_decision.toFixed(3)}</div>
                        </div>
                        <div className="mono" style={{ fontSize: 12, color: "var(--text-muted)" }}>
                          {row.decision.deliberation_cost.toFixed(3)}
                        </div>
                        <div className="mono" style={{ fontSize: 12, color: "var(--text-muted)" }}>
                          {row.decision.latency_ms.toFixed(1)} ms
                        </div>
                        <div className="gating__rationale">
                          <span>{row.action.rationale}</span>
                          {isSensingFallback && <span className="gating__tag">sensing-fallback</span>}
                        </div>
                      </div>
                      {isExpanded && <DecisionDetail row={row} />}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="gating__legend">
              <div className="gating__legend-items">
                <div className="gating__legend-item">
                  <span className="gating__legend-dot" style={{ background: "var(--blue)" }} />
                  System 1 — cheap, reactive
                </div>
                <div className="gating__legend-item">
                  <span className="gating__legend-dot" style={{ background: "var(--amber)" }} />
                  System 2 — deliberative, spends budget
                </div>
                <div className="gating__legend-item">
                  <span className="gating__legend-dot" style={{ background: "var(--violet)" }} />
                  sensing-fallback — conservative S1 on missing data
                </div>
              </div>
              <div className="mono" style={{ fontSize: 11, color: "var(--text-dim)" }}>
                services/learned/module3_metapolicy
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
