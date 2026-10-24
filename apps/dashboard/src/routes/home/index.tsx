/**
 * Landing page: one card per module, in pipeline order, each linking to that module's pages.
 * The demo starts here instead of on a module page, so nobody has to type a route.
 */
import { Link } from "react-router-dom";

import "./home.css";

type Page = { to: string; label: string; blurb: string };
type ModuleCard = { tag: string; title: string; owner: string; summary: string; pages: Page[] };

const MODULES: ModuleCard[] = [
  {
    tag: "M1",
    title: "State & Vulnerability Forecasting",
    owner: "Zayan",
    summary: "Fuses weather, load and asset data into one state per node and forecasts each node's vulnerability six hours ahead.",
    pages: [
      { to: "/vulnerability", label: "Vulnerability forecast", blurb: "ST-GNN v0.1 on the held-out quarter" },
      { to: "/agent-state", label: "State representation", blurb: "28 features and their quality mask" },
    ],
  },
  {
    tag: "M2",
    title: "Epistemic Uncertainty Engine",
    owner: "Duwaragie",
    summary: "Says how far the model can trust a state, and whether doubt comes from unseen conditions or missing data.",
    pages: [{ to: "/uncertainty", label: "Uncertainty engine", blurb: "Evidential head, two trigger axes" }],
  },
  {
    tag: "M3",
    title: "Cost-Aware Metacognitive Gating",
    owner: "Saabir",
    summary: "Decides, step by step through a cyclone, when to react cheaply and when to deliberate.",
    pages: [
      { to: "/cyclone", label: "Cyclone lab", blurb: "Set a storm and watch the trained gate" },
      { to: "/gating", label: "Gating walkthrough", blurb: "Train, generate a situation, process it" },
    ],
  },
  {
    tag: "M4",
    title: "Risk-Certified Verification",
    owner: "Hariswara",
    summary: "Checks every proposed action on the OpenDSS twin and certifies its risk over the next six hours.",
    pages: [{ to: "/verification", label: "Verification & causal log", blurb: "APPROVE, REJECT or UNVERIFIED, with reasons" }],
  },
];

export default function HomeRoute() {
  return (
    <div className="home">
      <div className="home__inner">
        <header className="home__header">
          <h1>MetaCore</h1>
          <p>
            Metacognitive control for islanded microgrids in Sri Lanka's northern islands. Each module
            hands its output to the next: state → uncertainty → gate → verification.
          </p>
        </header>

        <ol className="home__grid">
          {MODULES.map((m, i) => (
            <li key={m.tag} className="home__card">
              <div className="home__card-head">
                <span className="home__tag mono">{m.tag}</span>
                <span className="home__owner">{m.owner}</span>
              </div>
              <h2>{m.title}</h2>
              <p className="home__summary">{m.summary}</p>
              <div className="home__pages">
                {m.pages.map((p) => (
                  <Link key={p.to} to={p.to} className="home__page">
                    <span className="home__page-label">{p.label} →</span>
                    <span className="home__page-blurb">{p.blurb}</span>
                  </Link>
                ))}
              </div>
              {i < MODULES.length - 1 && <span className="home__arrow" aria-hidden="true">→</span>}
            </li>
          ))}
        </ol>

        <footer className="home__footer mono">J26-DS-317 · SLIIT</footer>
      </div>
    </div>
  );
}
