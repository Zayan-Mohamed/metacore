/**
 * Module 1 — Grid: the four-island network Module 4 verifies against, and one hour's state.
 *
 * Topology: gateway GET /api/module1/topology (the committed northern-islands-v1 pin).
 * State:    gateway POST /api/module1/snapshot (module1.topology snapshot for that hour).
 *
 * Every island is drawn separately because every island is electrically separate: no cable joins
 * them. Feeder loading is apparent power over the lumped feeder's ampacity at 400 V; voltage and
 * battery state of charge are not measured, and the page says so instead of drawing a number.
 */
import { useEffect, useMemo, useState } from "react";

import {
  fetchModule1Topology,
  snapshotModule1,
  type Module1Snapshot,
  type Module1Topology,
  type TopologyAsset,
} from "../../lib";
import "../../styles/page.css";
import "./grid.css";

const PRESETS = [
  { label: "Calm midday", at: "2025-10-05T12:00" },
  { label: "Nainativu record peak", at: "2025-07-20T20:00" },
  { label: "Ditwah evening (hybrid down)", at: "2025-11-28T19:00" },
  { label: "Battery decay month", at: "2025-12-10T19:00" },
];
const ISLAND_ORDER = ["Nainativu", "Delft-Neduntivu", "Analaitivu", "Eluvaitivu"];

function loadingColor(pct: number) {
  if (pct >= 100) return "var(--red)";
  if (pct >= 80) return "var(--amber)";
  return "var(--green)";
}

function assetLabel(a: TopologyAsset) {
  return `${a.element.split("_").slice(1).join(" ")} · ${a.asset_type.toLowerCase()} ${a.rating_kva} kVA`;
}

export default function GridRoute() {
  const [topo, setTopo] = useState<Module1Topology | null>(null);
  const [snap, setSnap] = useState<Module1Snapshot | null>(null);
  const [at, setAt] = useState(PRESETS[2].at);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    fetchModule1Topology().then(setTopo).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(() => {
    let cancelled = false;
    setPending(true);
    snapshotModule1(at)
      .then((s) => { if (!cancelled) { setSnap(s); setError(null); } })
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setPending(false));
    return () => { cancelled = true; };
  }, [at]);

  const state = useMemo(() => new Map((snap?.node_states ?? []).map((n) => [n.node_id, n])), [snap]);
  const out = new Set(snap?.assets_out_of_service ?? []);

  return (
    <div className="pg">
      <div className="pg__inner">
        <header className="pg__header">
          <h1>
            Module 1 — Grid
            {topo ? <span className="pg__badge">{topo.topology_id}</span> : null}
            {snap?.hybrid_plant_down ? <span className="pg__badge pg__badge--bad">Eluvaitivu hybrid plant down</span> : null}
            {pending ? <span className="pg__badge">loading snapshot…</span> : null}
          </h1>
          <p className="pg__subtitle">
            The network Module 4 verifies every action against, generated from one parameter table in
            Module 1: four separate 400 V islands, each fed only by its own diesel power station, with
            the CEB 2025 generator fleet. Below each island: one hour's feeder demand from the
            constructed hourly load. Feeder grouping, cable lengths and impedances are assumptions
            until CEB's single-line diagrams arrive.
          </p>
        </header>

        <section className="pg__card">
          <div className="pg__card-title">Hour</div>
          <div className="pg__btns">
            {PRESETS.map((p) => (
              <button key={p.at} type="button" className={`pg__btn pg__btn--sm${p.at === at ? "" : " pg__btn--ghost"}`}
                onClick={() => setAt(p.at)}>{p.label}</button>
            ))}
            <input type="datetime-local" className="pg__input mono" min="2024-01-01T00:00" max="2025-12-31T23:00"
              step={3600} value={at} aria-label="Snapshot hour"
              onChange={(e) => e.target.value && setAt(e.target.value.slice(0, 13) + ":00")} />
          </div>
        </section>

        {error ? (
          <section className="pg__card pg__card--error">
            <div className="pg__card-title">Grid unavailable</div>
            <pre className="pg__pre mono">{error}</pre>
          </section>
        ) : null}

        {topo ? (
          <div className="grid__islands">
            {ISLAND_ORDER.map((isl) => {
              const buses = topo.buses.filter((b) => b.island === isl);
              const lines = topo.lines.filter((l) => buses.some((b) => b.node_id === l.from_node));
              const assets = topo.assets.filter((a) => a.island === isl);
              const ps = buses.find((b) => b.role === "power_station") ?? buses[0];
              const diesel = assets.filter((a) => a.asset_type === "DIESEL");
              const installed = diesel.filter((a) => !out.has(a.element)).reduce((s, a) => s + a.rating_kva, 0);
              const demandKva = buses.reduce((s, b) => {
                const n = state.get(b.node_id);
                return n?.p_kw != null ? s + Math.hypot(n.p_kw, n.q_kvar ?? 0) : s;
              }, 0);
              return (
                <section key={isl} className="pg__card grid__island">
                  <div className="pg__card-title">{isl}</div>
                  <div className="pg__card-sub">
                    Demand {demandKva.toFixed(0)} kVA of {installed} kVA installed diesel
                    ({installed ? Math.round((100 * demandKva) / installed) : 0}%)
                  </div>

                  <div className="grid__bus grid__bus--ps">
                    <span className="mono">{ps.node_id}</span> power station
                    <ul className="grid__assets">
                      {assets.filter((a) => a.bus === ps.node_id).map((a) => (
                        <li key={a.element} className={out.has(a.element) || !a.in_service ? "is-out" : undefined}>
                          {assetLabel(a)}{a.grid_forming ? " · grid-forming" : ""}
                          {out.has(a.element) || !a.in_service ? " · out of service" : ""}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {lines.map((l) => {
                    const to = buses.find((b) => b.node_id === l.to_node);
                    const n = state.get(l.to_node);
                    const kva = n?.p_kw != null ? Math.hypot(n.p_kw, n.q_kvar ?? 0) : null;
                    const amps = kva !== null ? (kva * 1000) / (Math.sqrt(3) * topo.base_kv * 1000) : null;
                    const pct = amps !== null ? (100 * amps) / l.ampacity_a : null;
                    const toAssets = assets.filter((a) => a.bus === l.to_node);
                    return (
                      <div key={l.edge_id} className="grid__feeder">
                        <div className="grid__line" aria-hidden="true" />
                        <div className="grid__bus">
                          <div className="grid__bus-head">
                            <span className="mono">{l.to_node}</span>
                            {to?.critical_load ? <span className="pg__badge">critical (assumed)</span> : null}
                          </div>
                          <div className="pg__card-sub mono">
                            {l.length_km} km · {l.ampacity_a} A{l.circuits ? ` (${l.circuits} × 100 A feeders)` : ""} · {l.provenance}
                          </div>
                          {pct !== null ? (
                            <>
                              <div className="grid__bar" role="img" aria-label={`Feeder loading ${pct.toFixed(0)} percent`}>
                                <span style={{ width: `${Math.min(100, pct)}%`, background: loadingColor(pct) }} />
                              </div>
                              <div className="pg__card-sub mono">
                                {n!.p_kw!.toFixed(1)} kW · {amps!.toFixed(0)} A · {pct.toFixed(0)}% of rating{" "}
                                <span className="pg__q pg__q--interpolated">interpolated</span>
                              </div>
                            </>
                          ) : (
                            <div className="pg__card-sub">no demand at this bus</div>
                          )}
                          <div className="pg__card-sub">
                            voltage <span className="pg__q pg__q--missing">missing</span> — Module 4 computes it from the power flow
                          </div>
                          {toAssets.length ? (
                            <ul className="grid__assets">
                              {toAssets.map((a) => (
                                <li key={a.element} className={out.has(a.element) || !a.in_service ? "is-out" : undefined}>
                                  {assetLabel(a)}{out.has(a.element) || !a.in_service ? " · out of service" : ""}
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </div>
                      </div>
                    );
                  })}
                </section>
              );
            })}
          </div>
        ) : null}

        {topo ? (
          <section className="pg__card">
            <div className="pg__card-title">Assumptions in this model</div>
            <div className="pg__card-sub">Stated in the pin, replaced when CEB's single-line diagrams arrive.</div>
            <div className="pg__table-wrap">
              <table className="pg__table">
                <tbody>
                  {Object.entries(topo.assumptions as Record<string, unknown>).map(([k, v]) => (
                    <tr key={k}><td>{k.replace(/_/g, " ")}</td><td className="num">{typeof v === "object" ? JSON.stringify(v) : String(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
