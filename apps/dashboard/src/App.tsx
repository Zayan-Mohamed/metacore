/**
 * The four module views share one time axis so a single replayed episode reads across all of them:
 * vulnerability map (M1) -> uncertainty (M2) -> gating timeline (M3) -> verification log (M4).
 * Keeping the axis shared is what makes the dashboard an explanation rather than four widgets.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import { NavLayout } from "./components/NavBar";
import AgentStateRoute from "./routes/agent-state";
import CycloneRoute from "./routes/cyclone";
import DataHealthRoute from "./routes/data-health";
import GatingRoute from "./routes/gating";
import GridRoute from "./routes/grid";
import HomeRoute from "./routes/home";
import LiveRoute from "./routes/live";
import ModelGraphRoute from "./routes/model-graph";
import UncertaintyRoute from "./routes/uncertainty";
import VerificationRoute from "./routes/verification";
import VulnerabilityRoute from "./routes/vulnerability";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomeRoute />} />
      <Route element={<NavLayout />}>
        <Route path="/live" element={<LiveRoute />} />
        <Route path="/data-health" element={<DataHealthRoute />} />
        <Route path="/grid" element={<GridRoute />} />
        <Route path="/model-graph" element={<ModelGraphRoute />} />
        <Route path="/agent-state" element={<AgentStateRoute />} />
        <Route path="/gating" element={<GatingRoute />} />
        <Route path="/cyclone" element={<CycloneRoute />} />
        <Route path="/uncertainty" element={<UncertaintyRoute />} />
        <Route path="/verification" element={<VerificationRoute />} />
        <Route path="/vulnerability" element={<VulnerabilityRoute />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
