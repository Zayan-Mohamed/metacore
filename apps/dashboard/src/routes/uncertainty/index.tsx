/**
 * Module 2 — Agentic Epistemic Uncertainty Engine.
 *
 * Shows the quantity M3's gate actually consumes: the quality-aware epistemic scalar u'.
 * The point of the page is that u alone is not enough — an unseen situation and a blind
 * sensor are different failures, and only u' separates them (see trigger_reason).
 *
 * Styling follows routes/gating: tokens come from src/styles/tokens.css (:root), every
 * number renders in IBM Plex Mono, charts are div track/fill rather than a chart library.
 */
import { useEffect, useState } from "react";
import { runModule2 } from "../../lib";
import "./uncertainty.css";

/** Dirichlet classes — matches M2's K=3 safety classes plus the null hypothesis. */
const K = 4;
/** Evidence the EDL head emits on a fully in-distribution, fully observed reading. */
const E0 = 60;

const VALUE_TRIGGER_U = 0.5;
const SENSING_TRIGGER_F = 0.45;
/** baselines.MC_PASSES — the forward passes MC-Dropout pays per score. */
const MC_PASSES = 20;

type TriggerReason = "none" | "value" | "sensing" | "both";

interface UncertaintyState {
  /** Distribution shift, 0 (seen before) .. 1 (never seen). */
  novelty: number;
  /** Share of M1 features actually measured, 0.1 .. 1. */
  observedFraction: number;
}

interface UncertaintyResult {
  u: number;
  uq: number;
  totalEvidence: number;
  observedFraction: number;
  triggered: boolean;
  reason: TriggerReason;
  softmaxU: number;
  mcU: number;
  edlU: number;
  latencyLabel: string;
  /** Where the numbers came from — additive, so the render path is unchanged. */
  source?: "live" | "fallback";
  /** "onnx" or "torch" when live; absent on the fallback. */
  backend?: string;
}

/**
 * Offline stand-in for the engine, used only when the gateway is unreachable.
 *
 * These are hand-fitted curves, NOT the trained model — the shape is right and the
 * numbers are not. The live path (computeUncertainty) is the real EDL head; this exists
 * so the page still renders something honest-looking when the API is down.
 */
function simulateUncertainty(state: UncertaintyState): UncertaintyResult {
  const n = state.novelty;
  const f = state.observedFraction;

  // Evidence collapses as the situation leaves the training distribution.
  const sigma = E0 * Math.pow(1 - n, 1.6);

  // u = K / S, the Dirichlet vacuity. S is the total evidence mass.
  const u = K / (K + sigma);
  // Quality-aware: unobserved features cannot contribute evidence, so scale sigma by f.
  const uq = K / (K + sigma * f);

  const valueAxis = u > VALUE_TRIGGER_U;
  const sensingAxis = f < SENSING_TRIGGER_F;
  const reason: TriggerReason =
    valueAxis && sensingAxis ? "both" : valueAxis ? "value" : sensingAxis ? "sensing" : "none";

  return {
    u,
    uq,
    totalEvidence: K + sigma,
    observedFraction: f,
    triggered: valueAxis || sensingAxis,
    reason,
    // Baselines: softmax stays confident off-distribution; MC-Dropout reacts but is noisy.
    softmaxU: 0.06 + 0.1 * n,
    mcU: 0.08 + 0.55 * n + (sensingAxis ? 0.1 : 0),
    edlU: uq,
    latencyLabel: "0.03 ms",
    source: "fallback",
  };
}

/**
 * Scores one situation against the REAL trained EDL engine.
 *
 * POSTs to /api/module2/run, which shells out to
 * services/learned/module2_auq_engine/infer_one.py — the exported edl.onnx graph when
 * onnxruntime is installed, the torch head otherwise. Falls back to simulateUncertainty
 * if the gateway is unreachable, so the page always renders.
 */
async function computeUncertainty(state: UncertaintyState): Promise<UncertaintyResult> {
  try {
    const live = await runModule2(state.novelty, state.observedFraction);
    return {
      u: live.u,
      uq: live.u_q,
      totalEvidence: live.evidence,
      observedFraction: live.observed_fraction,
      triggered: live.trigger,
      reason: live.reason,
      softmaxU: live.baselines.softmax,
      mcU: live.baselines.mc_dropout,
      edlU: live.baselines.edl,
      latencyLabel: `${live.latency_ms.toFixed(3)} ms`,
      source: "live",
      backend: live.backend,
    };
  } catch {
    return simulateUncertainty(state);
  }
}

interface Scenario {
  id: string;
  label: string;
  novelty: number;
  observedFraction: number;
}

const SCENARIOS: Scenario[] = [
  { id: "normal", label: "Normal", novelty: 0.05, observedFraction: 1.0 },
  { id: "cyclone", label: "Cyclone (OOD)", novelty: 0.85, observedFraction: 1.0 },
  { id: "sensor", label: "Sensor loss", novelty: 0.1, observedFraction: 0.3 },
  { id: "both", label: "Both", novelty: 0.85, observedFraction: 0.3 },
];

const REASON_COPY: Record<TriggerReason, string> = {
  none: "none — evidence is sufficient and the sensors are reporting",
  value: "value — the situation is outside the training distribution",
  sensing: "sensing — too few features observed to trust the reading",
  both: "both — unseen situation observed through a degraded sensor set",
};

function bandColour(v: number): string {
  if (v < 0.4) return "var(--green)";
  if (v <= 0.66) return "var(--amber)";
  return "var(--red)";
}

function RingGauge({ value }: { value: number }) {
  const size = 132;
  const stroke = 12;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, value));
  const colour = bandColour(clamped);

  return (
    <div className="uncertainty__gauge">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`u prime ${clamped.toFixed(3)}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface-2)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={colour}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <div className="uncertainty__gauge-centre">
        <div className="uncertainty__gauge-value mono" style={{ color: colour }}>
          {clamped.toFixed(3)}
        </div>
        <div className="uncertainty__gauge-caption">u&prime;</div>
      </div>
    </div>
  );
}

const PIPELINE = [
  { label: "M1 state", note: "64-d + quality mask" },
  { label: "EDL head", note: "Dirichlet evidence", learned: true },
  { label: "u = K / S", note: "epistemic vacuity" },
  { label: "quality-aware", note: "scale by observed f" },
  { label: "trigger", note: "value / sensing" },
  { label: "M3", note: "gate input" },
];

function PipelineStrip({ reason }: { reason: TriggerReason }) {
  return (
    <div className="uncertainty__pipeline">
      {PIPELINE.map((step, i) => (
        <div className="uncertainty__pipe-group" key={step.label}>
          <div
            className={
              step.learned ? "uncertainty__chip uncertainty__chip--learned" : "uncertainty__chip"
            }
          >
            <div className="uncertainty__chip-label">{step.label}</div>
            <div className="uncertainty__chip-note">{step.note}</div>
            {step.learned ? <span className="uncertainty__chip-tag mono">LEARNED</span> : null}
            {step.label === "trigger" ? (
              <span
                className="uncertainty__chip-tag mono"
                style={{
                  color: reason === "none" ? "var(--green)" : "var(--red)",
                  background: reason === "none" ? "var(--green-soft)" : "var(--red-soft)",
                }}
              >
                {reason.toUpperCase()}
              </span>
            ) : null}
          </div>
          {i < PIPELINE.length - 1 ? <span className="uncertainty__pipe-arrow">→</span> : null}
        </div>
      ))}
    </div>
  );
}

interface BaselineRow {
  label: string;
  value: number;
  colour: string;
}

function BaselineBars({ rows }: { rows: BaselineRow[] }) {
  return (
    <div className="uncertainty__bars">
      {rows.map((row) => (
        <div className="uncertainty__bar-row" key={row.label}>
          <div className="uncertainty__bar-label">{row.label}</div>
          <div className="uncertainty__bar-track">
            <div
              className="uncertainty__bar-fill"
              style={{ width: `${Math.max(0, Math.min(1, row.value)) * 100}%`, background: row.colour }}
            />
          </div>
          <div className="uncertainty__bar-value mono">{row.value.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

export default function UncertaintyRoute() {
  const [novelty, setNovelty] = useState(SCENARIOS[0].novelty);
  const [observedFraction, setObservedFraction] = useState(SCENARIOS[0].observedFraction);

  // Render the offline curves immediately, then replace them with the engine's answer.
  const [result, setResult] = useState<UncertaintyResult>(() =>
    simulateUncertainty({ novelty: SCENARIOS[0].novelty, observedFraction: SCENARIOS[0].observedFraction }),
  );
  const [pending, setPending] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setPending(true);
    computeUncertainty({ novelty, observedFraction })
      .then((next) => {
        if (!cancelled) setResult(next);
      })
      .finally(() => {
        if (!cancelled) setPending(false);
      });
    return () => {
      cancelled = true;
    };
  }, [novelty, observedFraction]);

  const activeScenario = SCENARIOS.find(
    (s) => Math.abs(s.novelty - novelty) < 1e-9 && Math.abs(s.observedFraction - observedFraction) < 1e-9,
  );

  function applyScenario(s: Scenario) {
    setNovelty(s.novelty);
    setObservedFraction(s.observedFraction);
  }

  const kpis: Array<{ label: string; value: string; note: string }> = [
    { label: "Epistemic u", value: result.u.toFixed(3), note: "K / S, quality-blind" },
    { label: "Quality-aware u'", value: result.uq.toFixed(3), note: "what M3 gates on" },
    { label: "Total evidence", value: result.totalEvidence.toFixed(1), note: "S = K + Σe" },
    { label: "Observed fraction", value: result.observedFraction.toFixed(2), note: "share measured" },
    { label: "Latency", value: result.latencyLabel, note: "per inference" },
  ];

  return (
    <div className="uncertainty">
      <div className="uncertainty__inner">
        <header className="uncertainty__header">
          <div className="uncertainty__title-row">
            <div className="uncertainty__mark">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" />
                <circle cx="12" cy="12" r="3.2" />
              </svg>
            </div>
            <div>
              <h1>
                Module 2 — Agentic Epistemic Uncertainty Engine
                <span className="uncertainty__badge">m2-out/0.3</span>
                <span
                  className="uncertainty__badge"
                  style={
                    result.source === "live"
                      ? { color: "var(--green)", borderColor: "var(--green)" }
                      : { color: "var(--amber)", borderColor: "var(--amber)" }
                  }
                  title={
                    result.source === "live"
                      ? "Numbers come from the trained EDL engine via /api/module2/run"
                      : "Gateway unreachable — showing offline stand-in curves"
                  }
                >
                  {pending
                    ? "scoring…"
                    : result.source === "live"
                      ? `live · ${result.backend}`
                      : "offline fallback"}
                </span>
              </h1>
              <p className="uncertainty__subtitle">
                Evidential deep learning turns a prediction into evidence, so the agent can say
                &ldquo;I do not know&rdquo; — and say <em>why</em>.
              </p>
            </div>
          </div>
        </header>

        <section className="uncertainty__card uncertainty__controls">
          <div className="uncertainty__card-title">1. Choose a situation</div>
          <div className="uncertainty__card-sub">
            Presets move both sliders; the sliders are the two axes M2 reports separately.
          </div>

          <div className="uncertainty__scenarios">
            {SCENARIOS.map((s) => (
              <button
                key={s.id}
                type="button"
                className={
                  activeScenario?.id === s.id
                    ? "uncertainty__btn"
                    : "uncertainty__btn uncertainty__btn--ghost"
                }
                onClick={() => applyScenario(s)}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="uncertainty__sliders">
            <label className="uncertainty__field">
              <span className="uncertainty__field-label">
                Situation novelty (distribution shift)
                <span className="mono uncertainty__field-num">{(novelty * 100).toFixed(0)}%</span>
              </span>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={Math.round(novelty * 100)}
                onChange={(e) => setNovelty(Number(e.target.value) / 100)}
              />
              <span className="uncertainty__field-hint">0% = seen in training · 100% = never seen</span>
            </label>

            <label className="uncertainty__field">
              <span className="uncertainty__field-label">
                Sensing quality (observed fraction)
                <span className="mono uncertainty__field-num">
                  {(observedFraction * 100).toFixed(0)}%
                </span>
              </span>
              <input
                type="range"
                min={10}
                max={100}
                step={1}
                value={Math.round(observedFraction * 100)}
                onChange={(e) => setObservedFraction(Number(e.target.value) / 100)}
              />
              <span className="uncertainty__field-hint">
                below {Math.round(SENSING_TRIGGER_F * 100)}% trips the sensing axis
              </span>
            </label>
          </div>
        </section>

        <div className="uncertainty__kpi-row">
          {kpis.map((k) => (
            <div className="uncertainty__card uncertainty__kpi" key={k.label}>
              <div className="uncertainty__kpi-label">{k.label}</div>
              <div className="uncertainty__kpi-value mono">{k.value}</div>
              <div className="uncertainty__kpi-note">{k.note}</div>
            </div>
          ))}
        </div>

        <section className="uncertainty__card uncertainty__verdict">
          <RingGauge value={result.uq} />
          <div className="uncertainty__verdict-body">
            <div
              className="uncertainty__pill mono"
              style={{
                color: result.triggered ? "var(--red)" : "var(--green)",
                background: result.triggered ? "var(--red-soft)" : "var(--green-soft)",
              }}
            >
              {result.triggered ? "COMPETENCE DROP" : "CLEAR"}
            </div>
            <div className="uncertainty__reason">
              <span className="uncertainty__reason-key">trigger_reason</span>
              <span className="mono uncertainty__reason-val">{result.reason}</span>
            </div>
            <p className="uncertainty__reason-copy">{REASON_COPY[result.reason]}</p>
            <p className="uncertainty__reason-copy uncertainty__reason-copy--dim">
              M3 escalates to System 2 on the value axis. A sensing drop holds System 1 —
              deliberating on missing data buys nothing.
            </p>
          </div>
        </section>

        <section className="uncertainty__card">
          <div className="uncertainty__card-title">Pipeline</div>
          <div className="uncertainty__card-sub">
            One learned step, bracketed by deterministic ones.
          </div>
          <PipelineStrip reason={result.reason} />
        </section>

        <section className="uncertainty__card">
          <div className="uncertainty__card-title">Baselines at this operating point</div>
          <div className="uncertainty__card-sub">
            Same situation, three uncertainty estimators.
          </div>
          <BaselineBars
            rows={[
              { label: "Softmax", value: result.softmaxU, colour: "var(--text-dim)" },
              { label: "MC-Dropout", value: result.mcU, colour: "var(--amber)" },
              { label: "EDL — ours", value: result.edlU, colour: "var(--blue)" },
            ]}
          />
          <p className="uncertainty__caption">
            Softmax collapses toward 0 as the situation leaves the training distribution: its score
            is a normalised likelihood, not evidence, so it cannot report absence. MC-Dropout is
            worse than uninformative here — its predictive entropy is <em>higher</em> on the normal
            case (where the safety classes genuinely overlap) than on the cyclone, because entropy
            measures class ambiguity, not missing evidence. Only the evidential head rises with
            novelty, and it does so in one forward pass rather than {" "}
            {MC_PASSES}.
          </p>
          <p className="uncertainty__caption">
            Scales differ by construction — softmax is 1 − max p (≤ 2⁄3 for three classes),
            MC-Dropout is predictive entropy (≤ ln 3 ≈ 1.10), and u&prime; is a vacuity in [0, 1].
            Read the bars as a ranking at this operating point, not as one common axis.
          </p>
        </section>
      </div>
    </div>
  );
}
