// Module 3 — Cyclone lab. Set the storm's parameters, play that one storm through a trained gate and
// the baselines, and read the decisions off a shared timeline: wind, damage, the gate's P(System 2),
// the System 1/2 choice and the reward. The storm does not react to the controller, so every policy
// is scored on the identical storm (services/learned/module3_metapolicy/src/module3/serve/storm.py).
import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router-dom";

import { STORM_KNOBS, countSwitches, fetchStormPresets, runStorm, Module3RunError } from "../../lib";
import type {
  StormKnob,
  StormObsField,
  StormParams,
  StormPresets,
  StormResult,
  StormStep,
} from "../../lib";
import "../gating/gating.css";
import "./cyclone.css";

interface KnobInfo {
  label: string;
  unit: string;
  step: number;
  hint: string;
}

const KNOBS: Record<StormKnob, KnobInfo> = {
  peak_wind_kmh: { label: "Peak wind", unit: "km/h", step: 5, hint: "strongest sustained wind. 40 / 80 / 120 km/h mark elevated / severe / extreme" },
  peak_frac: { label: "Peak timing", unit: "of episode", step: 0.01, hint: "where the strongest wind falls (0 = start, 1 = end)" },
  width_frac: { label: "Storm duration", unit: "of episode", step: 0.01, hint: "how long the storm lasts; bigger = a slower, wider pulse" },
  trip_scale: { label: "Line fragility", unit: "per step", step: 0.01, hint: "chance a feeder trips each step at full exposure" },
  comms_rate: { label: "Comms loss rate", unit: "per step", step: 0.01, hint: "chance telemetry drops at peak wind (the gate then sees stale data)" },
  diesel_loss_prob: { label: "Diesel loss", unit: "per severe step", step: 0.01, hint: "chance a diesel unit is lost, permanently" },
  grid_loss_prob: { label: "Main supply cut", unit: "probability", step: 0.05, hint: "chance the storm cuts the main supply (islanded operation)" },
};

const POLICIES: Array<{ key: string; label: string; note: string }> = [
  { key: "gate", label: "Trained gate", note: "the PPO + GRU gate" },
  { key: "always_s1", label: "Always System 1", note: "never deliberates" },
  { key: "always_s2", label: "Always System 2", note: "always deliberates" },
  { key: "threshold", label: "u > threshold", note: "escalate when M2's uncertainty is high" },
  { key: "breaker_rule", label: "Outage rule", note: "escalate when an outage is visible" },
  { key: "oracle", label: "Oracle", note: "reads the true damage; a reference, not deployable and not an upper bound" },
];

const STAGE_LINES: Array<[number, string]> = [
  [40, "elevated"],
  [80, "severe"],
  [120, "extreme"],
];

const W = 1000;
const PAD_L = 64;
const PAD_R = 16;

function fmt(x: number, d = 2): string {
  return (x >= 0 ? "+" : "") + x.toFixed(d);
}

function Timeline({
  steps,
  hovered,
  onHover,
}: {
  steps: StormStep[];
  hovered: number | null;
  onHover: (t: number | null) => void;
}) {
  const n = steps.length;
  const plotW = W - PAD_L - PAD_R;
  const x = (t: number) => PAD_L + (t / Math.max(n - 1, 1)) * plotW;
  const colW = plotW / n;

  const maxWind = Math.max(130, ...steps.map((s) => s.wind_kmh)) * 1.05;
  const maxTrip = Math.max(2, ...steps.map((s) => s.tripped));
  const rewards = steps.map((s) => s.reward);
  const rMin = Math.min(...rewards, 0);
  const rMax = Math.max(...rewards, 0.01);

  // lane geometry (y offsets inside the svg viewBox)
  const lanes = {
    wind: { top: 12, h: 110 },
    damage: { top: 142, h: 70 },
    gate: { top: 232, h: 100 },
    reward: { top: 352, h: 80 },
  };
  const H = lanes.reward.top + lanes.reward.h + 26;

  const yWind = (v: number) => lanes.wind.top + lanes.wind.h * (1 - v / maxWind);
  const yGate = (p: number) => lanes.gate.top + lanes.gate.h * (1 - p);
  const yRew = (r: number) => lanes.reward.top + lanes.reward.h * (1 - (r - rMin) / (rMax - rMin));

  const windPath = steps.map((s, i) => `${i ? "L" : "M"}${x(s.t).toFixed(1)},${yWind(s.wind_kmh).toFixed(1)}`).join(" ");
  const pPath = steps
    .map((s, i) => `${i ? "L" : "M"}${x(s.t).toFixed(1)},${yGate(s.p_s2 ?? 0).toFixed(1)}`)
    .join(" ");
  const rewPath = steps.map((s, i) => `${i ? "L" : "M"}${x(s.t).toFixed(1)},${yRew(s.reward).toFixed(1)}`).join(" ");

  return (
    <svg
      className="cyclone__timeline"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Storm timeline: wind, damage, gate probability and decision, reward"
      onMouseLeave={() => onHover(null)}
    >
      {/* lane labels */}
      <text x={6} y={lanes.wind.top + 12} className="cyclone__lane-label">wind</text>
      <text x={6} y={lanes.wind.top + 26} className="cyclone__lane-unit">km/h</text>
      <text x={6} y={lanes.damage.top + 12} className="cyclone__lane-label">damage</text>
      <text x={6} y={lanes.gate.top + 12} className="cyclone__lane-label">gate</text>
      <text x={6} y={lanes.gate.top + 26} className="cyclone__lane-unit">P(S2)</text>
      <text x={6} y={lanes.reward.top + 12} className="cyclone__lane-label">reward</text>

      {/* System 2 steps, shaded across every lane */}
      {steps.map((s) =>
        s.action === 1 ? (
          <rect
            key={`s2-${s.t}`}
            x={x(s.t) - colW / 2}
            y={lanes.wind.top}
            width={colW}
            height={lanes.reward.top + lanes.reward.h - lanes.wind.top}
            className="cyclone__s2-band"
          />
        ) : null,
      )}

      {/* wind lane */}
      {STAGE_LINES.map(([v, name]) => (
        <g key={name}>
          <line x1={PAD_L} x2={W - PAD_R} y1={yWind(v)} y2={yWind(v)} className="cyclone__grid" />
          <text x={W - PAD_R - 2} y={yWind(v) - 3} textAnchor="end" className="cyclone__lane-unit">{name}</text>
        </g>
      ))}
      <path d={windPath} className="cyclone__wind" />

      {/* damage lane: grid-lost shading, feeders tripped as bars, comms-down ticks */}
      <line x1={PAD_L} x2={W - PAD_R} y1={lanes.damage.top + lanes.damage.h} y2={lanes.damage.top + lanes.damage.h} className="cyclone__grid" />
      {steps.map((s) => (
        <g key={`d-${s.t}`}>
          {s.grid_lost && (
            <rect x={x(s.t) - colW / 2} y={lanes.damage.top} width={colW} height={lanes.damage.h} className="cyclone__gridlost" />
          )}
          {s.tripped > 0 && (
            <rect
              x={x(s.t) - colW * 0.35}
              width={colW * 0.7}
              y={lanes.damage.top + lanes.damage.h * (1 - s.tripped / maxTrip)}
              height={lanes.damage.h * (s.tripped / maxTrip)}
              className="cyclone__tripped"
            />
          )}
          {!s.comms_up && (
            <rect x={x(s.t) - colW / 2} y={lanes.damage.top + lanes.damage.h + 2} width={colW} height={4} className="cyclone__comms" />
          )}
        </g>
      ))}

      {/* gate lane: probability and the 0.5 line */}
      <line x1={PAD_L} x2={W - PAD_R} y1={yGate(0.5)} y2={yGate(0.5)} className="cyclone__grid cyclone__grid--dash" />
      <text x={W - PAD_R - 2} y={yGate(0.5) - 3} textAnchor="end" className="cyclone__lane-unit">0.5</text>
      <path d={pPath} className="cyclone__prob" />

      {/* reward lane */}
      <line x1={PAD_L} x2={W - PAD_R} y1={yRew(0)} y2={yRew(0)} className="cyclone__grid" />
      <path d={rewPath} className="cyclone__reward" />

      {/* x axis */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const t = Math.round(f * (n - 1));
        return (
          <text key={f} x={x(t)} y={H - 6} textAnchor="middle" className="cyclone__lane-unit">
            step {t}
          </text>
        );
      })}

      {/* hover columns */}
      {steps.map((s) => (
        <rect
          key={`h-${s.t}`}
          x={x(s.t) - colW / 2}
          y={lanes.wind.top}
          width={colW}
          height={lanes.reward.top + lanes.reward.h - lanes.wind.top}
          className={`cyclone__hover${hovered === s.t ? " cyclone__hover--on" : ""}`}
          onMouseEnter={() => onHover(s.t)}
        />
      ))}
    </svg>
  );
}

// The action the chosen control path proposed this step: who is cut, by how much, and what the
// generators are asked to supply. This is what M3 hands to M4 (M3_TO_M4_CONTRACT.md).
function PlanPanel({ step }: { step: StormStep }) {
  const { plan } = step;
  const nothing = !plan.load_shed.length && !plan.dispatch.length && !plan.breakers.length;
  return (
    <div className="cyclone__plan">
      <div className="cyclone__plan-title">
        Action proposed by System {step.action + 1} <span className="mono">({plan.origin})</span>
      </div>
      {nothing && <div className="cyclone__plan-empty">no load shed, no dispatch, no breaker commands</div>}
      {plan.load_shed.length > 0 && (
        <table className="cyclone__plan-table">
          <thead>
            <tr><th>load shed at</th><th>cut</th><th>tier</th></tr>
          </thead>
          <tbody>
            {plan.load_shed.map((c) => (
              <tr key={c.node_id}>
                <td className="mono">{c.node_id}</td>
                <td className="mono">{(c.shed_fraction * 100).toFixed(1)}%</td>
                <td className="mono">{c.priority_tier}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {plan.dispatch.length > 0 && (
        <table className="cyclone__plan-table">
          <thead>
            <tr><th>dispatch at</th><th>set-point</th></tr>
          </thead>
          <tbody>
            {plan.dispatch.map((c) => (
              <tr key={c.node_id}>
                <td className="mono">{c.node_id}</td>
                <td className="mono">{c.p_kw.toFixed(1)} kW</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {plan.breakers.length > 0 && (
        <div className="mono">
          breakers: {plan.breakers.map((b) => `${b.edge_id} ${b.closed ? "close" : "open"}`).join(", ")}
        </div>
      )}
      <div className="cyclone__plan-why">{plan.rationale}</div>
    </div>
  );
}

// The 15 numbers the gate was given this step, with where each one comes from.
function ObsPanel({ step, fields }: { step: StormStep; fields: StormObsField[] }) {
  return (
    <div className="cyclone__plan">
      <div className="cyclone__plan-title">What the gate saw (its 15 inputs)</div>
      <table className="cyclone__plan-table">
        <thead>
          <tr><th>#</th><th>input</th><th>from</th><th>value</th><th>meaning</th></tr>
        </thead>
        <tbody>
          {fields.map((f) => (
            <tr key={f.index}>
              <td className="mono">{f.index}</td>
              <td className="mono">{f.name}</td>
              <td>{f.source}</td>
              <td className="mono">{(step.obs[f.index] ?? 0).toFixed(3)}</td>
              <td>{f.meaning}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StepDetail({ step, fields }: { step: StormStep | null; fields: StormObsField[] }) {
  if (!step) {
    return <div className="cyclone__detail cyclone__detail--empty">Hover the timeline to read one step.</div>;
  }
  return (
    <>
    <dl className="cyclone__detail">
      <div><dt>step</dt><dd className="mono">{step.t}</dd></div>
      <div><dt>wind</dt><dd className="mono">{step.wind_kmh.toFixed(0)} km/h ({step.stage})</dd></div>
      <div><dt>feeders out</dt><dd className="mono">{step.tripped}{step.grid_lost ? " + main supply lost" : ""}</dd></div>
      <div><dt>telemetry</dt><dd className="mono">{step.comms_up ? "up" : "DOWN"}</dd></div>
      <div><dt>M2 u / reason</dt><dd className="mono">{step.u.toFixed(3)} / {step.trigger}</dd></div>
      <div><dt>P(System 2)</dt><dd className="mono">{step.p_s2 === null ? "-" : step.p_s2.toFixed(2)}</dd></div>
      <div>
        <dt>decision</dt>
        <dd className="mono">
          System {step.action + 1}
          {step.budget_fallback ? " (budget forced S1)" : ""}
        </dd>
      </div>
      <div><dt>reward</dt><dd className="mono">{fmt(step.reward, 3)}</dd></div>
      <div><dt>load served</dt><dd className="mono">{(step.served * 100).toFixed(0)}%</dd></div>
      <div>
        <dt>grid twin (surrogate)</dt>
        <dd className="mono">{step.approved ? "approved" : "rejected"}{step.violations ? ` (${step.violations} buses)` : ""}</dd>
      </div>
    </dl>
    <div className="cyclone__panels">
      <PlanPanel step={step} />
      <ObsPanel step={step} fields={fields} />
    </div>
    </>
  );
}

export default function CyclonePage() {
  const [presets, setPresets] = useState<StormPresets | null>(null);
  const [params, setParams] = useState<StormParams | null>(null);
  const [preset, setPreset] = useState<string>("custom");
  const [seed, setSeed] = useState(1);
  const [gateSeed, setGateSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StormResult | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  useEffect(() => {
    fetchStormPresets()
      .then((p) => {
        setPresets(p);
        setParams(p.default);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "could not load storm presets"));
  }, []);

  function choosePreset(name: string) {
    setPreset(name);
    if (!presets) return;
    if (name === "custom") setParams(presets.default);
    else if (presets.presets[name]) setParams(presets.presets[name]);
  }

  function setKnob(key: StormKnob, value: number) {
    setParams((p) => (p ? { ...p, [key]: value } : p));
    setPreset("custom");
  }

  async function submit(e: FormEvent, random = false) {
    e.preventDefault();
    if (!params) return;
    setBusy(true);
    setError(null);
    try {
      const r = await runStorm(random ? { random: true, seed, gate_seed: gateSeed } : { params, seed, gate_seed: gateSeed });
      setResult(r);
      setHovered(null);
      if (random) {
        setParams(r.params);
        setPreset("custom");
      }
    } catch (err) {
      setError(err instanceof Module3RunError ? err.message : "Storm run failed unexpectedly");
    } finally {
      setBusy(false);
    }
  }

  const rows = useMemo(() => {
    if (!result) return [];
    const all = [{ key: "gate", ...result.gate }, ...Object.entries(result.baselines).map(([key, v]) => ({ key, ...v }))];
    return POLICIES.map((p) => ({ ...p, ...all.find((a) => a.key === p.key)! }));
  }, [result]);

  const retMin = Math.min(0, ...rows.map((r) => r.return));
  const retMax = Math.max(0.01, ...rows.map((r) => r.return));
  const realBaselines = rows.filter((r) => !["gate", "oracle"].includes(r.key));
  const bestBaseline = realBaselines.length ? realBaselines.reduce((a, b) => (b.return > a.return ? b : a)) : null;
  const oracle = rows.find((r) => r.key === "oracle");
  const steps = result?.gate.steps ?? [];

  return (
    <div className="gating cyclone">
      <div className="gating__inner">
        <header className="gating__header">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h1>Cyclone lab</h1>
              <span className="mono gating__badge">module 3</span>
            </div>
            <p className="gating__subtitle">
              Set the storm, then watch the trained gate handle it. Every policy below faces the <b>identical</b> storm, so
              differences in reward come only from the decisions. <Link to="/gating">Train and step through the older demo gate</Link>
            </p>
          </div>
        </header>

        <form className="gating__card gating__run-card" onSubmit={(e) => submit(e)}>
          <div className="gating__card-title">1. The storm</div>
          <div className="gating__card-sub">
            Start from a preset or set the knobs yourself. The two named presets are placeholder parameter sets with the right shape,
            not real Burevi/Ditwah data.
          </div>

          <div className="cyclone__presets">
            {["custom", ...(presets ? Object.keys(presets.presets) : [])].map((name) => (
              <button
                key={name}
                type="button"
                className={`cyclone__chip${preset === name ? " cyclone__chip--on" : ""}`}
                onClick={() => choosePreset(name)}
                disabled={busy || !presets}
              >
                {name === "custom" ? "default storm" : name.replace("_", " ")}
              </button>
            ))}
          </div>

          {params && presets && (
            <div className="cyclone__knobs">
              {STORM_KNOBS.map((k) => {
                const [lo, hi] = presets.ranges[k];
                const info = KNOBS[k];
                return (
                  <div className="cyclone__knob" key={k}>
                    <div className="cyclone__knob-head">
                      <label htmlFor={k}>{info.label}</label>
                      <span className="mono">
                        {params[k].toFixed(info.step < 0.1 ? 2 : 0)} <small>{info.unit}</small>
                      </span>
                    </div>
                    <input
                      id={k}
                      type="range"
                      min={lo}
                      max={hi}
                      step={info.step}
                      value={params[k]}
                      disabled={busy}
                      onChange={(e) => setKnob(k, Number(e.target.value))}
                    />
                    <span className="gating__field-hint">{info.hint}</span>
                  </div>
                );
              })}
              <div className="cyclone__knob">
                <div className="cyclone__knob-head">
                  <label htmlFor="ood">Unseen conditions</label>
                  <input
                    id="ood"
                    type="checkbox"
                    checked={params.ood}
                    disabled={busy}
                    onChange={(e) => {
                      setParams({ ...params, ood: e.target.checked });
                      setPreset("custom");
                    }}
                  />
                </div>
                <span className="gating__field-hint">M2 reports value-axis uncertainty and the grid is hit harder than System 1 expects</span>
              </div>
            </div>
          )}

          <div className="gating__form-footer">
            <div className="gating__steps">
              <div className="gating__field" style={{ width: 110 }}>
                <label htmlFor="seed">Storm seed</label>
                <input id="seed" type="number" min={0} max={1_000_000} value={seed} disabled={busy} onChange={(e) => setSeed(Number(e.target.value))} />
                <span className="gating__field-hint">random damage and timing</span>
              </div>
              <div className="gating__field" style={{ width: 110 }}>
                <label htmlFor="gate">Trained gate</label>
                <input id="gate" type="number" min={0} max={4} value={gateSeed} disabled={busy} onChange={(e) => setGateSeed(Math.min(4, Math.max(0, Number(e.target.value))))} />
                <span className="gating__field-hint">training seed 0-4</span>
              </div>
              <button className="gating__run-btn" type="submit" disabled={busy || !params}>
                {busy ? "Running…" : "Run this storm"}
              </button>
              <button className="gating__run-btn gating__run-btn--ghost" type="button" disabled={busy || !params} onClick={(e) => submit(e, true)}>
                Random storm
              </button>
            </div>
            {busy && (
              <div className="gating__run-status">
                <span className="gating__spinner" />
                playing the storm through the gate and five baselines…
              </div>
            )}
          </div>
          {error && <div className="gating__error">{error}</div>}
        </form>

        {!result && !error && <div className="gating__empty">Run a storm to see how the gate handles it.</div>}

        {result && (
          <>
            <div className="gating__kpi-row">
              <div className="gating__card">
                <div className="gating__kpi-label">Gate return</div>
                <div className="gating__kpi-value mono" style={{ color: "var(--green)" }}>{result.gate.return.toFixed(2)}</div>
                <div className="gating__kpi-note">sum of step rewards, {result.episode_len} steps</div>
              </div>
              <div className="gating__card">
                <div className="gating__kpi-label">Vs. best real baseline</div>
                <div className="gating__kpi-value mono">{bestBaseline ? fmt(result.gate.return - bestBaseline.return, 2) : "-"}</div>
                <div className="gating__kpi-note">{bestBaseline ? `${bestBaseline.label} ${bestBaseline.return.toFixed(2)}` : ""}</div>
              </div>
              <div className="gating__card">
                <div className="gating__kpi-label">Gap to oracle</div>
                <div className="gating__kpi-value mono">{oracle ? fmt(result.gate.return - oracle.return, 2) : "-"}</div>
                <div className="gating__kpi-note">oracle {oracle?.return.toFixed(2)}</div>
              </div>
              <div className="gating__card">
                <div className="gating__kpi-label">System 2 used</div>
                <div className="gating__kpi-value mono">{(result.gate.s2_rate * 100).toFixed(0)}%</div>
                <div className="gating__kpi-note">{countSwitches(steps)} switches between S1 and S2</div>
              </div>
            </div>

            <div className="gating__card" style={{ marginTop: 14 }}>
              <div className="gating__card-title">2. Timeline of this storm</div>
              <div className="gating__card-sub">
                Shaded columns are steps where the gate chose System 2. Damage lane: red bars = feeders actually out, purple wash =
                main supply lost, blue strip = telemetry down.
              </div>
              <Timeline steps={steps} hovered={hovered} onHover={setHovered} />
              <StepDetail step={hovered === null ? null : (steps[hovered] ?? null)} fields={result.obs_fields} />
            </div>

            <div className="gating__card" style={{ marginTop: 14 }}>
              <div className="gating__card-title">3. Same storm, every policy</div>
              <div className="gating__card-sub">return = sum of step rewards. Higher is better.</div>
              <div className="cyclone__table">
                <div className="cyclone__row cyclone__row--head">
                  <span>Policy</span><span>Return</span><span /><span>System 2</span><span>Load served</span><span>Twin rejects</span>
                </div>
                {rows.map((r) => {
                  const span = retMax - retMin || 1;
                  const left = ((Math.min(0, r.return) - retMin) / span) * 100;
                  const width = (Math.abs(r.return) / span) * 100;
                  return (
                    <div className={`cyclone__row${r.key === "gate" ? " cyclone__row--gate" : ""}`} key={r.key} title={r.note}>
                      <span>{r.label}</span>
                      <span className="mono">{r.return.toFixed(2)}</span>
                      <span className="cyclone__bar"><i style={{ left: `${left}%`, width: `${Math.max(width, 1)}%` }} /></span>
                      <span className="mono">{(r.s2_rate * 100).toFixed(0)}%</span>
                      <span className="mono">{(r.served_fraction * 100).toFixed(0)}%</span>
                      <span className="mono">{(r.reject_rate * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
              <p className="cyclone__note">
                One storm is one sample: a single seed can favour any policy. The averaged results over 40 storms and 5 training seeds
                are in docs/module3.md section 25.6. The grid here is the surrogate twin, not the live M4 firewall.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
