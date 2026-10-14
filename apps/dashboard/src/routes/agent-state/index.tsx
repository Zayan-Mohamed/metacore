/**
 * Module 1 — Multi-Modal Spatiotemporal State Representation.
 *
 * The state every other view aligns to: M1 fuses six input modalities into one
 * 28-feature node vector plus a 64-d embedding, and stamps each feature with the
 * QualityMask value the offline artifacts can actually support (ADR 0004).
 *
 * The feature vector shown here is assembled by the gateway from the offline
 * calibration artifacts (services/learned/module1_state_forecasting, via
 * gateway routers/module1.py → module1.assemble). The learned 64-d node_embedding
 * is a zero placeholder — the ST-GNN encoder that would fill it does not exist yet.
 * When the gateway is unreachable the page falls back to built-in representative
 * values, and the badge says so.
 *
 * The contract shape (feature order, groups, calibration quality, schema version)
 * is taken verbatim from
 * packages/contracts/python/metacore_contracts/schema/module1_state_v1.json.
 *
 * Styling follows routes/gating and routes/uncertainty: tokens from
 * src/styles/tokens.css (:root), every number in IBM Plex Mono, div track/fill bars.
 */
import { useEffect, useState } from "react";
import { assembleModule1, type Module1AssembleResult, type QualityFlag } from "../../lib";
import "./agent-state.css";

const EMBEDDING_DIM = 64;
const SCHEMA_VERSION = "1.0";

type Quality = "observed" | "interpolated" | "missing";
type GroupId = "electrical" | "resource" | "meteorology" | "demand" | "temporal" | "topology";
type ScenarioId = "normal" | "cyclone" | "blackout";
type IslandId = "eluvaitivu" | "analaitivu" | "nainativu" | "delft";

interface Feature {
  name: string;
  group: GroupId;
  unit: string;
  /** QualityMask value in the offline calibration artifacts — a floor, not a promise. */
  calib: Quality;
  /** Representative normalised value on a nominal step, before scenario shift. */
  base: number;
}

interface GroupMeta {
  id: GroupId;
  label: string;
  source: string;
}

const GROUPS: GroupMeta[] = [
  { id: "electrical", label: "Electrical", source: "simulator / replay" },
  { id: "resource", label: "Resource", source: "NASA POWER CERES + MERRA-2" },
  { id: "meteorology", label: "Meteorology", source: "NASA POWER MERRA-2" },
  { id: "demand", label: "Demand", source: "CEB ledger → downscaled hourly" },
  { id: "temporal", label: "Temporal", source: "timestamp_lst" },
  { id: "topology", label: "Static topology", source: "GridTopology.Node" },
];

// Order, groups and calibration quality are the v1 pin. `base` values are the
// offline fallback only — the live path replaces them with assembled numbers.
const FEATURES: Feature[] = [
  { name: "p_kw_norm", group: "electrical", unit: "p.u. rating", calib: "missing", base: 0.62 },
  { name: "q_kvar_norm", group: "electrical", unit: "p.u. rating", calib: "missing", base: 0.18 },
  { name: "voltage_pu", group: "electrical", unit: "p.u.", calib: "missing", base: 1.0 },
  { name: "soc_fraction", group: "electrical", unit: "0..1", calib: "missing", base: 0.54 },
  { name: "asset_online", group: "electrical", unit: "0/1", calib: "missing", base: 1.0 },

  { name: "ghi_wh_m2_norm", group: "resource", unit: "p.u. clear-sky", calib: "interpolated", base: 0.47 },
  { name: "clearsky_index", group: "resource", unit: "0..1", calib: "interpolated", base: 0.78 },
  { name: "wind_10m_ms_norm", group: "resource", unit: "p.u. site max", calib: "interpolated", base: 0.31 },
  { name: "wind_50m_ms_norm", group: "resource", unit: "p.u. site max", calib: "interpolated", base: 0.36 },
  { name: "pv_available_kw_norm", group: "resource", unit: "p.u. panel", calib: "interpolated", base: 0.44 },

  { name: "temp_2m_c_norm", group: "meteorology", unit: "z-scored", calib: "interpolated", base: 0.21 },
  { name: "humidity_2m_pct_norm", group: "meteorology", unit: "0..1", calib: "interpolated", base: 0.73 },
  { name: "precip_mm_hr_norm", group: "meteorology", unit: "p.u. site max", calib: "interpolated", base: 0.04 },
  { name: "pressure_kpa_norm", group: "meteorology", unit: "z-scored", calib: "interpolated", base: 0.05 },

  { name: "load_kw_norm", group: "demand", unit: "p.u. peak", calib: "interpolated", base: 0.66 },
  { name: "load_ramp_kw_per_h_norm", group: "demand", unit: "p.u./h", calib: "interpolated", base: 0.12 },

  { name: "hour_sin", group: "temporal", unit: "-1..1", calib: "observed", base: 0.5 },
  { name: "hour_cos", group: "temporal", unit: "-1..1", calib: "observed", base: -0.87 },
  { name: "doy_sin", group: "temporal", unit: "-1..1", calib: "observed", base: -0.31 },
  { name: "doy_cos", group: "temporal", unit: "-1..1", calib: "observed", base: 0.95 },

  { name: "nominal_kv_norm", group: "topology", unit: "p.u. base", calib: "observed", base: 0.4 },
  { name: "critical_load", group: "topology", unit: "0/1", calib: "observed", base: 1.0 },
  { name: "is_bus", group: "topology", unit: "0/1", calib: "observed", base: 1.0 },
  { name: "is_pv", group: "topology", unit: "0/1", calib: "observed", base: 0.0 },
  { name: "is_wind", group: "topology", unit: "0/1", calib: "observed", base: 0.0 },
  { name: "is_bess", group: "topology", unit: "0/1", calib: "observed", base: 0.0 },
  { name: "is_diesel", group: "topology", unit: "0/1", calib: "observed", base: 0.0 },
  { name: "is_load", group: "topology", unit: "0/1", calib: "observed", base: 0.0 },
];

interface Island {
  id: IslandId;
  label: string;
  /** NASA POWER site coordinates (nasa_power.ISLAND_SITES). */
  coords: string;
  /** Small per-island nudge on load and resource, so the fallback vector is not identical. */
  loadBias: number;
  windBias: number;
}

const ISLANDS: Island[] = [
  { id: "eluvaitivu", label: "Eluvaitivu", coords: "9.760 N, 79.770 E", loadBias: 0.0, windBias: 0.0 },
  { id: "analaitivu", label: "Analaitivu", coords: "9.720 N, 79.790 E", loadBias: -0.06, windBias: 0.03 },
  { id: "nainativu", label: "Nainativu", coords: "9.615 N, 79.775 E", loadBias: 0.09, windBias: -0.02 },
  { id: "delft", label: "Delft-Neduntivu", coords: "9.520 N, 79.690 E", loadBias: 0.14, windBias: 0.07 },
];

interface Scenario {
  id: ScenarioId;
  label: string;
  scenarioId: string;
  outOfDistribution: boolean;
  degraded: boolean;
  blurb: string;
}

const SCENARIOS: Scenario[] = [
  {
    id: "normal",
    label: "Normal",
    scenarioId: "mock-island-normal",
    outOfDistribution: false,
    degraded: false,
    blurb: "Nominal quality, in-distribution. Every feature carries the QualityMask value the offline pin assigns it.",
  },
  {
    id: "cyclone",
    label: "Cyclone (OOD)",
    scenarioId: "mock-island-cyclone",
    outOfDistribution: true,
    degraded: false,
    blurb: "The lowest-pressure hour in the island's NASA POWER record — a genuine surface-pressure tail, well outside the training range. The mask is unchanged; the values are what shift.",
  },
  {
    id: "blackout",
    label: "Comms blackout",
    scenarioId: "mock-island-blackout",
    outOfDistribution: false,
    degraded: true,
    blurb: "In-distribution values, but the clock source is gone: the four temporal features drop to QUALITY_MISSING and observed_fraction falls with them.",
  },
];

const QUALITY_COLOUR: Record<Quality, string> = {
  observed: "var(--green)",
  interpolated: "var(--amber)",
  missing: "var(--red)",
};

const QUALITY_PROTO: Record<Quality, string> = {
  observed: "QUALITY_OBSERVED",
  interpolated: "QUALITY_INTERPOLATED",
  missing: "QUALITY_MISSING",
};

const QUALITY_FROM_FLAG: Record<QualityFlag, Quality> = {
  QUALITY_OBSERVED: "observed",
  QUALITY_INTERPOLATED: "interpolated",
  QUALITY_MISSING: "missing",
};

/** Runtime quality for this feature under the chosen scenario — offline fallback. */
function runtimeQuality(f: Feature, scenario: Scenario): Quality {
  if (scenario.degraded && f.group === "temporal") return "missing";
  return f.calib;
}

/** Representative normalised value — offline fallback, NOT the real producer. */
function featureValue(f: Feature, island: Island, scenario: Scenario): number {
  let v = f.base;
  if (f.group === "demand") v += island.loadBias;
  if (f.name.startsWith("wind_")) v += island.windBias;

  if (scenario.id === "cyclone") {
    if (f.name === "pressure_kpa_norm") v = -2.1;
    else if (f.name.startsWith("wind_")) v = 0.94;
    else if (f.name === "precip_mm_hr_norm") v = 0.81;
    else if (f.name === "ghi_wh_m2_norm" || f.name === "pv_available_kw_norm") v = 0.06;
    else if (f.name === "clearsky_index") v = 0.12;
    else if (f.name === "humidity_2m_pct_norm") v = 0.97;
    else if (f.name === "load_kw_norm") v += 0.18;
  }

  if (scenario.degraded && f.group === "temporal") return 0;
  return Number(v.toFixed(2));
}

interface GroupRow {
  meta: GroupMeta;
  count: number;
  quality: Quality;
  present: boolean;
}

/** Per-group quality = its best feature: observed > interpolated > missing. */
function groupsFromQuality(quality: Quality[]): GroupRow[] {
  return GROUPS.map((meta) => {
    const qualities = FEATURES.map((f, i) => (f.group === meta.id ? quality[i] : null)).filter(
      (q): q is Quality => q !== null,
    );
    const q: Quality = qualities.includes("observed")
      ? "observed"
      : qualities.includes("interpolated")
        ? "interpolated"
        : "missing";
    return { meta, count: qualities.length, quality: q, present: q !== "missing" };
  });
}

type ViewSource = "live-real" | "live-synthetic" | "fallback";

interface StateView {
  values: number[];
  quality: Quality[];
  observed: number;
  observedFraction: number;
  modalitiesPresent: number;
  perGroup: GroupRow[];
  scenarioId: string;
  timestamp: string | null;
  outOfDistribution: boolean;
  degraded: boolean;
  nodeCount: number;
  nodeNames: string[];
  hasEmbedding: boolean;
  source: ViewSource;
}

function fallbackView(island: Island, scenario: Scenario): StateView {
  const quality = FEATURES.map((f) => runtimeQuality(f, scenario));
  const values = FEATURES.map((f) => featureValue(f, island, scenario));
  const perGroup = groupsFromQuality(quality);
  const observed = quality.filter((q) => q === "observed").length;
  return {
    values,
    quality,
    observed,
    observedFraction: observed / FEATURES.length,
    modalitiesPresent: perGroup.filter((g) => g.present).length,
    perGroup,
    scenarioId: scenario.scenarioId,
    timestamp: null,
    outOfDistribution: scenario.outOfDistribution,
    degraded: scenario.degraded,
    nodeCount: 1,
    nodeNames: ["island"],
    hasEmbedding: false,
    source: "fallback",
  };
}

function liveView(p: Module1AssembleResult): StateView {
  // Align the payload to the page's FEATURES order by name, so a schema reorder
  // can never silently misalign the grid.
  const byName = new Map(p.feature_names.map((n, i) => [n, i] as const));
  const values: number[] = [];
  const quality: Quality[] = [];
  for (const f of FEATURES) {
    const i = byName.get(f.name);
    values.push(i === undefined ? 0 : p.node_features[0][i]);
    quality.push(i === undefined ? "missing" : QUALITY_FROM_FLAG[p.quality_mask[i]]);
  }
  const perGroup = groupsFromQuality(quality);
  return {
    values,
    quality,
    observed: quality.filter((q) => q === "observed").length,
    observedFraction: p.observed_fraction,
    modalitiesPresent: perGroup.filter((g) => g.present).length,
    perGroup,
    scenarioId: p.scenario_id,
    timestamp: p.timestamp_lst,
    outOfDistribution: p.out_of_distribution,
    degraded: p.degraded,
    nodeCount: p.node_count,
    nodeNames: p.node_names,
    hasEmbedding: p.has_embedding,
    source: p.data_source === "real" ? "live-real" : "live-synthetic",
  };
}

const PIPELINE = [
  { label: "raw sources", note: "CEB · NASA POWER · topology" },
  { label: "reconcile", note: "300 invariants / 10 island-yr", gate: true },
  { label: "calibration set", note: "versioned params" },
  { label: "assemble", note: "28-d vector + 64-d embed" },
  { label: "QualityMask", note: "per-feature Quality" },
  { label: "M2", note: "evidential head" },
];

function GroupPill({ group }: { group: GroupId }) {
  return <span className={`agent-state__gpill agent-state__gpill--${group}`}>{group}</span>;
}

export default function AgentStateRoute() {
  const [islandId, setIslandId] = useState<IslandId>("eluvaitivu");
  const [scenarioId, setScenarioId] = useState<ScenarioId>("normal");

  const island = ISLANDS.find((i) => i.id === islandId)!;
  const scenario = SCENARIOS.find((s) => s.id === scenarioId)!;

  const [view, setView] = useState<StateView>(() => fallbackView(island, scenario));
  const [pending, setPending] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setPending(true);
    assembleModule1(islandId, scenarioId)
      .then((p) => {
        if (!cancelled) setView(liveView(p));
      })
      .catch(() => {
        if (!cancelled) {
          setView(
            fallbackView(
              ISLANDS.find((i) => i.id === islandId)!,
              SCENARIOS.find((s) => s.id === scenarioId)!,
            ),
          );
        }
      })
      .finally(() => {
        if (!cancelled) setPending(false);
      });
    return () => {
      cancelled = true;
    };
  }, [islandId, scenarioId]);

  const kpis = [
    { label: "Observed fraction", value: view.observedFraction.toFixed(3), note: "share QUALITY_OBSERVED" },
    { label: "Features observed", value: `${view.observed} / ${FEATURES.length}`, note: "QUALITY_OBSERVED groups" },
    { label: "Modalities present", value: `${view.modalitiesPresent} / ${GROUPS.length}`, note: "≥ 1 non-missing feature" },
    {
      label: "Embedding dim",
      value: String(EMBEDDING_DIM),
      note: view.hasEmbedding ? "learned per-node width" : "reserved — encoder not built",
    },
    { label: "Schema", value: `v${SCHEMA_VERSION}`, note: "module1_state_v1" },
  ];

  const isLive = view.source !== "fallback";
  const sourceRoot = view.source === "live-real" ? "data/" : ".synthetic/";

  return (
    <div className="agent-state">
      <div className="agent-state__inner">
        <header className="agent-state__header">
          <div className="agent-state__title-row">
            <div className="agent-state__mark">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2 2 7l10 5 10-5-10-5Z" />
                <path d="m2 17 10 5 10-5" />
                <path d="m2 12 10 5 10-5" />
              </svg>
            </div>
            <div>
              <h1>
                Module 1 — Multi-Modal Spatiotemporal State Representation
                <span className="agent-state__badge">m1-state/{SCHEMA_VERSION}</span>
                {isLive ? (
                  <span
                    className="agent-state__badge agent-state__badge--live"
                    title="node_features assembled by module1.assemble from the offline calibration artifacts. node_embedding is still a zero placeholder."
                  >
                    live · {sourceRoot}
                  </span>
                ) : (
                  <span
                    className="agent-state__badge agent-state__badge--proto"
                    title="Gateway unreachable — showing the page's built-in representative values. Run `uv run uvicorn gateway.main:app` from the repo root for a state assembled from data/."
                  >
                    offline · representative
                  </span>
                )}
                {pending ? <span className="agent-state__badge">assembling…</span> : null}
              </h1>
              <p className="agent-state__subtitle">
                Six input modalities fused into one node vector, each feature stamped with the
                QualityMask value the offline artifacts can support — the honest state of the
                evidence, and the reason M2&rsquo;s epistemic head has something to do.
              </p>
            </div>
          </div>
        </header>

        <section className="agent-state__card">
          <div className="agent-state__card-title">1. Assemble a state</div>
          <div className="agent-state__card-sub">
            Island sets the NASA POWER site and load series; scenario sets the QualityMask and picks
            the hour.
          </div>

          <div className="agent-state__row-label">Island</div>
          <div className="agent-state__btns">
            {ISLANDS.map((i) => (
              <button
                key={i.id}
                type="button"
                className={i.id === islandId ? "agent-state__btn" : "agent-state__btn agent-state__btn--ghost"}
                onClick={() => setIslandId(i.id)}
              >
                {i.label}
              </button>
            ))}
          </div>

          <div className="agent-state__row-label">Scenario</div>
          <div className="agent-state__btns">
            {SCENARIOS.map((s) => (
              <button
                key={s.id}
                type="button"
                className={s.id === scenarioId ? "agent-state__btn" : "agent-state__btn agent-state__btn--ghost"}
                onClick={() => setScenarioId(s.id)}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="agent-state__scenario-line">
            <span className="mono">
              site {island.coords} · scenario_id {view.scenarioId} · out_of_distribution{" "}
              {String(view.outOfDistribution)} · degraded {String(view.degraded)}
              {view.timestamp ? ` · t ${view.timestamp}` : ""}
            </span>
            <p className="agent-state__scenario-blurb">{scenario.blurb}</p>
          </div>
        </section>

        <div className="agent-state__kpi-row">
          {kpis.map((k) => (
            <div className="agent-state__card agent-state__kpi" key={k.label}>
              <div className="agent-state__kpi-label">{k.label}</div>
              <div className="agent-state__kpi-value mono">{k.value}</div>
              <div className="agent-state__kpi-note">{k.note}</div>
            </div>
          ))}
        </div>

        <section className="agent-state__card">
          <div className="agent-state__card-title">State assembly</div>
          <div className="agent-state__card-sub">
            The offline calibration path (ADR 0004). The reconciliation gate stands in for held-out
            telemetry — CEB&rsquo;s islands have no SCADA and no historian.
          </div>
          <div className="agent-state__pipeline">
            {PIPELINE.map((step, i) => (
              <div className="agent-state__pipe-group" key={step.label}>
                <div className={step.gate ? "agent-state__chip agent-state__chip--gate" : "agent-state__chip"}>
                  <div className="agent-state__chip-label">{step.label}</div>
                  <div className="agent-state__chip-note">{step.note}</div>
                  {step.gate ? <span className="agent-state__chip-tag mono">GATE</span> : null}
                </div>
                {i < PIPELINE.length - 1 ? <span className="agent-state__pipe-arrow">→</span> : null}
              </div>
            ))}
          </div>
        </section>

        <section className="agent-state__card">
          <div className="agent-state__card-title">Input modalities</div>
          <div className="agent-state__card-sub">
            Six groups, 28 features. Only temporal and static topology are observed at node level —
            everything else is interpolated or unavailable.
          </div>
          <div className="agent-state__mods">
            {view.perGroup.map(({ meta, count, quality, present }) => (
              <div className="agent-state__mod" key={meta.id}>
                <div className="agent-state__mod-head">
                  <span className="agent-state__mod-name">{meta.label}</span>
                  <span
                    className="agent-state__qpill mono"
                    style={{ color: QUALITY_COLOUR[quality], background: "var(--surface-2)" }}
                  >
                    {present ? QUALITY_PROTO[quality] : "QUALITY_MISSING"}
                  </span>
                </div>
                <div className="agent-state__mod-source">{meta.source}</div>
                <div className="agent-state__mod-count mono">
                  {count} feature{count === 1 ? "" : "s"}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="agent-state__card">
          <div className="agent-state__card-title">QualityMask.per_feature</div>
          <div className="agent-state__card-sub">
            One cell per feature, in schema order. Blank source cells become QUALITY_MISSING, never
            0.0 — &ldquo;no lube oil was used&rdquo; and &ldquo;we do not know&rdquo; are different.
          </div>
          <div className="agent-state__mask">
            {FEATURES.map((f, i) => {
              const q = view.quality[i];
              return (
                <span
                  key={f.name}
                  className="agent-state__mask-cell"
                  style={{ background: QUALITY_COLOUR[q] }}
                  title={`${f.name} — ${QUALITY_PROTO[q]}`}
                />
              );
            })}
          </div>
          <div className="agent-state__legend">
            {(["observed", "interpolated", "missing"] as Quality[]).map((q) => (
              <span className="agent-state__legend-item" key={q}>
                <span className="agent-state__legend-dot" style={{ background: QUALITY_COLOUR[q] }} />
                {QUALITY_PROTO[q]}
              </span>
            ))}
          </div>
        </section>

        <section className="agent-state__card">
          <div className="agent-state__card-title">State vector — {island.label}</div>
          <div className="agent-state__card-sub">
            node_features[0]
            {view.nodeCount > 1 ? ` of ${view.nodeCount} — ${view.nodeNames.join(", ")}` : ""}, 28
            columns.{" "}
            {isLive
              ? `Assembled from ${sourceRoot} at ${view.timestamp}.`
              : "Representative of a nominal step — gateway offline."}
          </div>
          <div className="agent-state__vec">
            {FEATURES.map((f, idx) => {
              const q = view.quality[idx];
              return (
                <div className="agent-state__cell" key={f.name}>
                  <div className="agent-state__cell-top">
                    <span className="agent-state__cell-idx mono">{idx}</span>
                    <span className="agent-state__cell-dot" style={{ background: QUALITY_COLOUR[q] }} />
                  </div>
                  <div className="agent-state__cell-name">{f.name}</div>
                  <div className="agent-state__cell-val mono">{view.values[idx].toFixed(2)}</div>
                  <div className="agent-state__cell-foot">
                    <GroupPill group={f.group} />
                    <span className="agent-state__cell-unit">{f.unit}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="agent-state__card agent-state__notes">
          <div className="agent-state__card-title">Notes</div>
          <ul>
            <li>
              <b>Two ingestion paths (ADR 0004).</b> This is the offline calibration half — batch
              only, never on a request path. The streaming half lives in
              services/realtime/ingestion_svc and shares no code; the versioned parameter set is the
              only interface between them.
            </li>
            <li>
              <b>Assembled, not learned.</b> node_features come from module1.assemble over the
              calibration artifacts (real <span className="mono">data/</span> or the{" "}
              <span className="mono">.synthetic/</span> stand-in). The electrical group is
              QUALITY_MISSING because the offline path has no power-flow simulator — the runtime
              simulator fills it per step. node_embedding is a zero placeholder; the ST-GNN encoder
              that fills it does not exist yet.
            </li>
            <li>
              <b>Spatial degeneracy.</b> One irradiance series and two wind series cover all four
              islands (~55–111 km NASA POWER cells), so the resource modality carries diurnal and
              cyclone signal but almost no inter-island signal.
            </li>
            <li>
              <b>Nodes.</b> Each island is a bus plus the generation assets the CEB ledger records —
              only Eluvaitivu runs a hybrid (PV + storage) plant. A full single-line diagram arrives
              with the ST-GNN; the grid above shows node 0, the bus.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}
