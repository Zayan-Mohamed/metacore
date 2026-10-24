/**
 * "/" is the control room: the live system on one screen (ADR 0006). The module pages below it
 * explain each component; "/modules" is their index.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import { NavLayout } from "./components/NavBar";
import AgentStateRoute from "./routes/agent-state";
import CycloneRoute from "./routes/cyclone";
import DataHealthRoute from "./routes/data-health";
import GatingRoute from "./routes/gating";
import GridRoute from "./routes/grid";
import ControlRoute from "./routes/control";
import HomeRoute from "./routes/home";
import ModelGraphRoute from "./routes/model-graph";
import UncertaintyRoute from "./routes/uncertainty";
import VerificationRoute from "./routes/verification";
import VulnerabilityRoute from "./routes/vulnerability";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ControlRoute />} />
      <Route path="/modules" element={<HomeRoute />} />
      <Route element={<NavLayout />}>
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
