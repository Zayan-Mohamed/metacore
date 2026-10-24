/**
 * Slim bar above every module page: back to the landing page, or straight to another module.
 * Lives in the layout, not in the pages, so no module's page code changes.
 */
import { Link, NavLink, Outlet } from "react-router-dom";

import "./navbar.css";

const LINKS = [
  { to: "/live", tag: "M1", label: "Live" },
  { to: "/vulnerability", tag: "M1", label: "Forecast" },
  { to: "/agent-state", tag: "M1", label: "State" },
  { to: "/model-graph", tag: "M1", label: "Graph" },
  { to: "/grid", tag: "M1", label: "Grid" },
  { to: "/data-health", tag: "M1", label: "Data" },
  { to: "/uncertainty", tag: "M2", label: "Uncertainty" },
  { to: "/cyclone", tag: "M3", label: "Cyclone lab" },
  { to: "/gating", tag: "M3", label: "Gating" },
  { to: "/verification", tag: "M4", label: "Verification" },
];

export function NavLayout() {
  return (
    <>
      <nav className="navbar" aria-label="Modules">
        <Link to="/" className="navbar__home">
          ← MetaCore
        </Link>
        <div className="navbar__links">
          {LINKS.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) => `navbar__link${isActive ? " navbar__link--active" : ""}`}
            >
              <span className="navbar__tag">{l.tag}</span>
              {l.label}
            </NavLink>
          ))}
        </div>
      </nav>
      <Outlet />
    </>
  );
}
