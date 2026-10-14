/**
 * The four module views share one time axis so a single replayed episode reads across all of them:
 * vulnerability map (M1) -> uncertainty (M2) -> gating timeline (M3) -> verification log (M4).
 * Keeping the axis shared is what makes the dashboard an explanation rather than four widgets.
 */
import { Navigate, Route, Routes } from "react-router-dom";

import AgentStateRoute from "./routes/agent-state";
import GatingRoute from "./routes/gating";
import UncertaintyRoute from "./routes/uncertainty";
import VerificationRoute from "./routes/verification";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/gating" replace />} />
      <Route path="/agent-state" element={<AgentStateRoute />} />
      <Route path="/gating" element={<GatingRoute />} />
      <Route path="/uncertainty" element={<UncertaintyRoute />} />
      <Route path="/verification" element={<VerificationRoute />} />
    </Routes>
  );
}
