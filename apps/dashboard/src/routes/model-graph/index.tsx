/**
 * Module 1 — Model graph: what the ST-GNN's graph is, what it does, and whether it helps.
 *
 * Structure: gateway GET /api/module1/graph (module1.graph): nodes, edges, the propagation matrix
 * A_hat the graph convolution multiplies by, and the graph ablations for both weather sources.
 * Embeddings: gateway POST /api/module1/forecast at a chosen hour. Similarity is computed on the
 * embeddings centred across nodes: after the ReLU every embedding is positive, so raw cosine is
 * 0.85-1.0 for every pair and says nothing. Centred, it shows which nodes the model treats alike.
 */
import { useEffect, useMemo, useState } from "react";

import {
  fetchModule1Graph,
  forecastModule1,
  type Module1ForecastResult,
  type Module1Graph,
  type Module1Weighting,
} from "../../lib";
import "../../styles/page.css";
import "./model-graph.css";

const ISLAND_SHORT: Record<string, string> = { "Delft-Neduntivu": "Delft" };
const short = (s: string) => ISLAND_SHORT[s] ?? s;
const WEIGHTING_LABEL: Record<Module1Weighting, string> = {
  equal: "Equal", hazard_heavy: "Hazard-heavy", stress_heavy: "Stress-heavy",
};
const SOURCE_LABEL: Record<string, string> = { nasa_power: "NASA POWER (paper model)", open_meteo: "Open-Meteo ICON (live model)" };
const HOURS = [
  { label: "Ditwah peak", at: "2025-11-28T12:00" },
  { label: "Calm day", at: "2025-10-05T12:00" },
  { label: "Battery decay", at: "2025-12-10T19:00" },
];

function seq(v: number, lo: number, hi: number) {
  const pct = Math.max(0, Math.min(100, ((v - lo) / (hi - lo || 1)) * 100));
  return {
    background: `color-mix(in oklch, var(--blue) ${pct.toFixed(0)}%, var(--surface-2))`,
    color: pct > 58 ? "var(--bg)" : "var(--text-muted)",
  };
}

/** Diverging: orange (opposite) - neutral grey (unrelated) - blue (alike). */
function div(v: number) {
  const t = Math.max(-1, Math.min(1, v));
  const hue = t >= 0 ? "var(--series-stgnn)" : "var(--series-lstm)";
  const pct = Math.round(Math.abs(t) * 100);
  return {
    background: `color-mix(in oklch, ${hue} ${pct}%, var(--surface-2))`,
    color: pct > 60 ? "var(--bg)" : "var(--text-muted)",
  };
}

function centred(vectors: number[][]) {
  const k = vectors[0]?.length ?? 0;
  const mu = Array.from({ length: k }, (_, i) => vectors.reduce((s, v) => s + v[i], 0) / vectors.length);
  return vectors.map((v) => v.map((x, i) => x - mu[i]));
}

function cosine(a: number[], b: number[]) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0;
}

/** Four stars side by side: each island's bus with its assets around it. */
function GraphDiagram({ g, hover, setHover }: { g: Module1Graph; hover: string | null; setHover: (s: string | null) => void }) {
  const islands = [...new Set(g.nodes.map((n) => n.island))];
  const W = 760, H = 250, colW = W / islands.length;
  const pos = new Map<string, { x: number; y: number }>();
  islands.forEach((isl, k) => {
    const cx = colW * k + colW / 2, cy = 92;
    const members = g.nodes.filter((n) => n.island === isl);
    const bus = members.find((n) => n.asset === "bus")!;
    pos.set(bus.id, { x: cx, y: cy });
    const leaves = members.filter((n) => n !== bus);
    leaves.forEach((n, i) => {
      const ang = Math.PI / 2 + (leaves.length === 1 ? 0 : ((i - (leaves.length - 1) / 2) * Math.PI) / 3);
      pos.set(n.id, { x: cx + Math.cos(ang) * 72, y: cy + Math.sin(ang) * 72 });
    });
  });
  const idx = new Map(g.nodes.map((n, i) => [n.id, i]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The forecaster's graph: four island stars, no edges between islands">
      {islands.map((isl, k) => (
        <text key={isl} x={colW * k + colW / 2} y={20} textAnchor="middle" fontSize={12} fill="var(--text-muted)">{short(isl)}</text>
      ))}
      {islands.slice(1).map((_, k) => (
        <line key={k} x1={colW * (k + 1)} x2={colW * (k + 1)} y1={30} y2={H - 10} stroke="var(--border)" strokeDasharray="3 5" />
      ))}
      {g.edges.map(([a, b]) => {
        const p = pos.get(a)!, q = pos.get(b)!;
        const w = g.a_hat[idx.get(a)!][idx.get(b)!];
        const on = hover === a || hover === b;
        return (
          <g key={`${a}-${b}`}>
            <line x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke={on ? "var(--blue)" : "var(--text-dim)"} strokeWidth={1 + w * 5} />
            <text x={(p.x + q.x) / 2} y={(p.y + q.y) / 2 + 3.5} textAnchor="middle" fontSize={10} fill="var(--text-muted)"
              fontFamily="IBM Plex Mono, monospace" stroke="var(--surface)" strokeWidth={4} paintOrder="stroke">{w.toFixed(2)}</text>
          </g>
        );
      })}
      {g.nodes.map((n) => {
        const p = pos.get(n.id)!;
        const on = hover === n.id;
        return (
          <g key={n.id} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)} style={{ cursor: "default" }}>
            <circle cx={p.x} cy={p.y} r={20} fill="var(--surface-2)"
              stroke={on ? "var(--blue)" : "var(--text-muted)"} strokeWidth={on ? 2.5 : 1.5} />
            <text x={p.x} y={p.y + 3.5} textAnchor="middle" fontSize={9} fill="var(--text)">{n.asset}</text>
          </g>
        );
      })}
    </svg>
  );
}

export default function ModelGraphRoute() {
  const [g, setG] = useState<Module1Graph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [at, setAt] = useState(HOURS[0].at);
  const [fc, setFc] = useState<Module1ForecastResult | null>(null);
  const [fcError, setFcError] = useState<string | null>(null);

  useEffect(() => {
    fetchModule1Graph().then(setG).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(() => {
    let cancelled = false;
    forecastModule1(at, "equal")
      .then((r) => { if (!cancelled) { setFc(r); setFcError(null); } })
      .catch((e: unknown) => !cancelled && setFcError(e instanceof Error ? e.message : String(e)));
    return () => { cancelled = true; };
  }, [at]);

  const sim = useMemo(() => {
    if (!fc || !g) return null;
    const byId = new Map(fc.nodes.map((n) => [n.id, n.embedding]));
    const c = centred(g.nodes.map((n) => byId.get(n.id) ?? []));
    return c.map((a) => c.map((b) => cosine(a, b)));
  }, [fc, g]);

  if (error) {
    return (
      <div className="pg"><div className="pg__inner">
        <section className="pg__card pg__card--error"><div className="pg__card-title">Graph unavailable</div><pre className="pg__pre mono">{error}</pre></section>
      </div></div>
    );
  }
  if (!g) return <div className="pg"><div className="pg__inner"><p className="pg__subtitle">Loading the graph…</p></div></div>;

  const hn = g.nodes.find((n) => n.id === hover);

  return (
    <div className="pg">
      <div className="pg__inner">
        <header className="pg__header">
          <h1>
            Module 1 — Model graph
            <span className="pg__badge">{g.nodes.length} nodes · {g.edges.length} edges</span>
            <span className="pg__badge">1 graph convolution layer (v0.1)</span>
          </h1>
          <p className="pg__subtitle">
            The ST-GNN reads each node's last 24 hours with a recurrent encoder, then lets each node
            mix in its neighbours' state through one graph convolution, then predicts the change in
            vulnerability for the next six hours. This page shows the graph it mixes over, the exact
            mixing weights, what the mixing does to the 64-number node embeddings, and whether the
            graph improved the forecast.
          </p>
        </header>

        <section className="pg__card">
          <div className="pg__card-title">1. The graph</div>
          <div className="pg__card-sub">{g.rule} Edge labels and thickness are the mixing weight between the two nodes. Hover a node.</div>
          <div className="mg__diagram"><GraphDiagram g={g} hover={hover} setHover={setHover} /></div>
          <div className="pg__note">
            {hn ? (
              <>
                <span className="mono">{hn.id}</span>: keeps {hn.self_weight.toFixed(2)} of its own vector and takes{" "}
                {hn.neighbour_weight.toFixed(2)} in total from {hn.degree} neighbour{hn.degree === 1 ? "" : "s"}
                {hn.neighbours.length ? ` (${hn.neighbours.map((x) => x.split("/")[1]).join(", ")})` : ""}. Hazard exposure {hn.exposure}.
              </>
            ) : "Hover a node to read how much it keeps of itself and how much it takes from its neighbours."}
          </div>
        </section>

        <div className="pg__cols">
          <section className="pg__card">
            <div className="pg__card-title">2. The layer</div>
            <div className="pg__card-sub">{g.layer}</div>
            <div className="pg__card-title" style={{ marginTop: 16 }}>Propagation matrix Â</div>
            <div className="pg__card-sub">Row i: how node i mixes every node's vector. Zero between islands.</div>
            <div className="mg__matrix" role="table" aria-label="Propagation matrix">
              <div className="mg__mrow" role="row">
                <span />
                {g.nodes.map((n) => <span key={n.id} className="mg__mhead" title={n.id}>{short(n.island).slice(0, 3)} {n.asset.slice(0, 3)}</span>)}
              </div>
              {g.a_hat.map((row, i) => (
                <div key={i} className="mg__mrow" role="row">
                  <span className="mg__mlabel" title={g.nodes[i].id}>{short(g.nodes[i].island).slice(0, 3)} {g.nodes[i].asset}</span>
                  {row.map((v, j) => (
                    <span key={j} className="mg__mcell mono" style={seq(v, 0, 0.5)}
                      title={`${g.nodes[i].id} ← ${g.nodes[j].id}: ${v.toFixed(3)}`}>{v ? v.toFixed(2) : ""}</span>
                  ))}
                </div>
              ))}
            </div>
          </section>

          <section className="pg__card">
            <div className="pg__card-title">3. Node embeddings after the graph layer</div>
            <div className="pg__card-sub">
              How alike the model treats each pair of nodes: cosine similarity of the 64-number
              embeddings sent to Module 2, centred across nodes, at one hour (NASA-trained v0.1, the
              checkpoint the forecast endpoint serves). Blue = alike, orange = opposite, grey = unrelated.
            </div>
            <div className="pg__btns">
              {HOURS.map((h) => (
                <button key={h.at} type="button" className={`pg__btn pg__btn--sm${h.at === at ? "" : " pg__btn--ghost"}`}
                  onClick={() => setAt(h.at)}>{h.label}</button>
              ))}
            </div>
            {fcError ? <pre className="pg__pre mono">{fcError}</pre> : null}
            {sim ? (
              <div className="mg__matrix" role="table" aria-label="Embedding similarity">
                <div className="mg__mrow" role="row">
                  <span />
                  {g.nodes.map((n) => <span key={n.id} className="mg__mhead" title={n.id}>{short(n.island).slice(0, 3)} {n.asset.slice(0, 3)}</span>)}
                </div>
                {sim.map((row, i) => (
                  <div key={i} className="mg__mrow" role="row">
                    <span className="mg__mlabel">{short(g.nodes[i].island).slice(0, 3)} {g.nodes[i].asset}</span>
                    {row.map((v, j) => (
                      <span key={j} className="mg__mcell mono" style={div(v)}
                        title={`${g.nodes[i].id} · ${g.nodes[j].id}: ${v.toFixed(3)}`}>{v.toFixed(2)}</span>
                    ))}
                  </div>
                ))}
              </div>
            ) : null}
            <div className="pg__note">
              The embeddings are organised by asset type, not by island: buses resemble buses and diesel
              sets resemble diesel sets, while a bus and its own diesel set point in opposite directions.
              Analaitivu, Nainativu and Delft come out almost identical (similarity 0.98 or more at all
              three hours above), because NASA POWER gives those islands the same or near-same weather and
              each is the same bus–diesel pair; only their load differs. That is the same reason the graph
              adds little (section 4).
            </div>
          </section>
        </div>

        <section className="pg__card">
          <div className="pg__card-title">4. Does the graph help? Ablations on the test quarter</div>
          <div className="pg__card-sub">
            Error reduction from adding each component (positive = it helps). Graph: ST-GNN against the
            same model without the graph layer. Each source's model is scored on its own target.
          </div>
          <div className="pg__table-wrap">
            <table className="pg__table">
              <thead>
                <tr><th>Weather source</th><th>Weighting</th><th className="num">ST-GNN RMSE</th><th className="num">Graph</th><th className="num">Temporal encoder</th><th className="num">Weather inputs</th></tr>
              </thead>
              <tbody>
                {Object.entries(g.ablations).flatMap(([src, per]) =>
                  (Object.keys(per) as Module1Weighting[]).map((w, i) => {
                    const a = per[w];
                    const pct = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
                    return (
                      <tr key={`${src}-${w}`}>
                        <td>{i === 0 ? SOURCE_LABEL[src] ?? src : ""}</td>
                        <td>{WEIGHTING_LABEL[w]}</td>
                        <td className="num">{a.rmse.stgnn.toFixed(4)}</td>
                        <td className="num">{pct(a.graph_gain_pct)}</td>
                        <td className="num">{pct(a.temporal_gain_pct)}</td>
                        <td className="num">{pct(a.weather_gain_pct)}</td>
                      </tr>
                    );
                  }))}
              </tbody>
            </table>
          </div>
          <div className="pg__note">
            The graph helps only under the stress-heavy weighting with NASA weather (+17.3%). Each island
            is a star of one to three assets, so one round of message passing has little to carry; a
            larger feeder model is where the graph claim can be tested properly.
          </div>
        </section>
      </div>
    </div>
  );
}
