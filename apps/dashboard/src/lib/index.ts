// API client and websocket stream against services/gateway.
// Requests go through vite's `/api` proxy (vite.config.ts) -> local uvicorn or compose `gateway`.

export interface Module3RunConfig {
  seed: number;
  episodeLen: number;
  budgetPerEpisode: number;
  trainEpisodes: number;
  evalEpisodes: number;
  deliberationCost: number;
  benefitScale: number;
  sensingEscalationPenalty: number;
}

export const MODULE3_DEFAULT_CONFIG: Module3RunConfig = {
  seed: 0,
  episodeLen: 80,
  budgetPerEpisode: 45,
  trainEpisodes: 150,
  evalEpisodes: 12,
  deliberationCost: 0.1,
  benefitScale: 4.0,
  sensingEscalationPenalty: -1.2,
};

// action_id ties a ProposedControlAction to its GatingDecision — see M3_TO_M4_CONTRACT.md.
export interface ProposedControlAction {
  action_id: string;
  origin: "SYSTEM1" | "SYSTEM2";
  breakers: Array<{ edge_id: string; closed: boolean }>;
  load_shed: Array<{ node_id: string; shed_fraction: number; priority_tier: number }>;
  dispatch: Array<{ node_id: string; p_kw: number; q_kvar: number }>;
  rationale: string;
  schema_version: string;
  message_type: "ProposedControlAction";
}

export interface GatingDecision {
  action_id: string;
  chosen: "SYSTEM1" | "SYSTEM2";
  epistemic_at_decision: number;
  expected_survival_benefit: number;
  deliberation_cost: number;
  latency_ms: number;
  budget_exhausted_fallback: boolean;
  schema_version: string;
  message_type: "GatingDecision";
  timestamp: number;
}

export type Module3Record = ProposedControlAction | GatingDecision;

// Not part of the M3->M4 wire contract (M3_TO_M4_CONTRACT.md) -- this is the M2 input
// state and M4-mock verdict that produced each decision, added purely for the dashboard.
// See run_demo.py's `context` list.
export interface Module3DecisionContext {
  action_id: string;
  severity: string;
  trigger_reason: "none" | "value" | "sensing" | "both";
  epistemic_uncertainty: number;
  observed_fraction: number;
  competence_drop: boolean;
  reward: number;
  verdict: "APPROVE" | "REJECT";
  violations: Array<{
    type: string;
    element_id: string;
    measured: number;
    limit: number;
    margin_fraction: number;
    attributed_component: string;
  }>;
}

export interface Module3RunResult {
  reward: {
    always_s1: number;
    always_s2: number;
    threshold: number;
    trained_policy: number;
  };
  avg_deliberation_cost: { always_s2: number; threshold: number; trained_policy: number };
  escalation_by_severity: Record<string, number>;
  monotonic_nondecreasing: boolean;
  escalation_by_trigger_reason: Record<string, number>;
  decisions: Module3Record[];
  decision_context: Module3DecisionContext[];
}

export class Module3RunError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "Module3RunError";
  }
}

function toRequestBody(cfg: Module3RunConfig) {
  return {
    seed: cfg.seed,
    episode_len: cfg.episodeLen,
    budget_per_episode: cfg.budgetPerEpisode,
    train_episodes: cfg.trainEpisodes,
    eval_episodes: cfg.evalEpisodes,
    deliberation_cost: cfg.deliberationCost,
    benefit_scale: cfg.benefitScale,
    sensing_escalation_penalty: cfg.sensingEscalationPenalty,
  };
}

export async function runModule3(cfg: Module3RunConfig): Promise<Module3RunResult> {
  return postModule3("/api/module3/run", toRequestBody(cfg));
}

export interface ObservationField {
  index: number;
  name: string;
  source: "M1" | "M2" | "env" | string;
  meaning: string;
  value: number;
}

export interface Module3GenerateResult {
  step_index: number;
  episode_len: number;
  observation: ObservationField[];
  raw: {
    epistemic_uncertainty: number;
    trigger_reason: "none" | "value" | "sensing" | "both";
    observed_fraction: number;
    competence_drop: boolean;
    severity: string;
    max_node_vulnerability: number;
    top_at_risk_nodes?: string[];
    [key: string]: unknown;
  };
  seed_used: number;
}

export interface Module3ProcessResult {
  chosen: "SYSTEM1" | "SYSTEM2";
  requested: "SYSTEM1" | "SYSTEM2";
  budget_exhausted_fallback: boolean;
  action_probs: { SYSTEM1: number; SYSTEM2: number };
  epistemic_at_decision: number;
  deliberation_cost: number;
  plan: ProposedControlAction;
  observation: ObservationField[];
  raw: Module3GenerateResult["raw"];
}

export async function generateModule3Situation(): Promise<Module3GenerateResult> {
  return postModule3("/api/module3/generate");
}

export async function processModule3Gate(): Promise<Module3ProcessResult> {
  return postModule3("/api/module3/process");
}

// Shared POST + FastAPI error unwrapping. `detail` is a string for our own HTTPExceptions
// and an array of {msg} for pydantic validation errors.
async function postJson<T>(
  url: string,
  body: unknown,
  makeError: (detail: string, status: number) => Error,
): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null);
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : Array.isArray(payload?.detail)
          ? payload.detail.map((d: { msg?: string }) => d.msg).join("; ")
          : `request failed with status ${res.status}`;
    throw makeError(detail, res.status);
  }
  return (await res.json()) as T;
}

async function postModule3<T>(url: string, body?: unknown): Promise<T> {
  return postJson<T>(url, body, (detail, status) => new Module3RunError(detail, status));
}

// ---------------------------------------------------------------- Module 2
// Scores one situation against the trained EDL engine via
// services/learned/module2_auq_engine/infer_one.py (gateway routers/module2.py).

export class Module2RunError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "Module2RunError";
  }
}

export interface Module2RunResult {
  /** Plain epistemic vacuity u = K/S, quality-blind. */
  u: number;
  /** Quality-aware u' — what M3 actually gates on. */
  u_q: number;
  /** Total evidence mass S = K + Σe. */
  evidence: number;
  observed_fraction: number;
  trigger: boolean;
  reason: "none" | "value" | "sensing" | "both";
  latency_ms: number;
  baselines: { softmax: number; mc_dropout: number; edl: number };
  /** "onnx" when the exported graph ran, "torch" when it fell back to the head. */
  backend: string;
  /** Calibrated value-axis threshold the trigger compared u against. */
  value_threshold: number;
}

export async function runModule2(
  novelty: number,
  observedFraction: number,
): Promise<Module2RunResult> {
  return postJson<Module2RunResult>(
    "/api/module2/run",
    { novelty, observed_fraction: observedFraction },
    (detail, status) => new Module2RunError(detail, status),
  );
}

// Pairs each ProposedControlAction with its GatingDecision (and M2 input / M4 verdict
// context, when present) by action_id, in emission order.
export interface Module3DecisionRow {
  actionId: string;
  action: ProposedControlAction;
  decision: GatingDecision;
  context?: Module3DecisionContext;
}

export function pairModule3Decisions(
  records: Module3Record[],
  context: Module3DecisionContext[] = [],
): Module3DecisionRow[] {
  const byId = new Map<string, Partial<Module3DecisionRow>>();
  for (const rec of records) {
    const entry = byId.get(rec.action_id) ?? { actionId: rec.action_id };
    if (rec.message_type === "ProposedControlAction") entry.action = rec;
    else entry.decision = rec;
    byId.set(rec.action_id, entry);
  }
  const contextById = new Map(context.map((c) => [c.action_id, c]));
  const rows: Module3DecisionRow[] = [];
  for (const entry of byId.values()) {
    if (entry.action && entry.decision) {
      rows.push({
        actionId: entry.actionId!,
        action: entry.action,
        decision: entry.decision,
        context: contextById.get(entry.actionId!),
      });
    }
  }
  return rows;
}

// Module 4: Deterministic Physics Verification & Grounded Causal Translation

export interface ViolationDTO {
  type:
    | "VIOLATION_TYPE_UNSPECIFIED"
    | "VIOLATION_TYPE_UNDERVOLTAGE"
    | "VIOLATION_TYPE_OVERVOLTAGE"
    | "VIOLATION_TYPE_THERMAL_OVERLOAD"
    | "VIOLATION_TYPE_NON_CONVERGENCE"
    | "VIOLATION_TYPE_MALFORMED_ACTION";
  element_id: string;
  limit: number;
  measured: number;
  margin_fraction: number;
  attributed_component: string;
}

export interface CausalLogDTO {
  action_id: string;
  text: string;
  grounded_entities: string[];
  generator: string;
}

export interface GridBusState {
  bus_name: string;
  island: string;
  voltage_pu: number;
  status: "SAFE" | "UNDERVOLTAGE" | "OVERVOLTAGE";
}

export interface GridLineState {
  line_name: string;
  current_amps: number;
  norm_amps: number;
  margin_fraction: number;
  is_closed: boolean;
  status: "NORMAL" | "OVERLOAD" | "TRIPPED";
}

export interface Module4VerifyResult {
  action_id: string;
  decision: "DECISION_APPROVE" | "DECISION_REJECT";
  solve_latency_ms: number;
  violations: ViolationDTO[];
  rejection_severity: number;
  causal_log: CausalLogDTO;
  buses: GridBusState[];
  lines: GridLineState[];
}

export interface Module4Preset {
  title: string;
  description: string;
  payload: {
    action_id: string;
    origin: "SYSTEM1" | "SYSTEM2";
    rationale: string;
    breakers: Array<{ edge_id: string; closed: boolean }>;
    load_shed: Array<{ node_id: string; shed_fraction: number; priority_tier: number }>;
    dispatch: Array<{ node_id: string; p_kw: number; q_kvar: number }>;
  };
}

export class Module4RunError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "Module4RunError";
  }
}

export async function fetchModule4Presets(): Promise<Record<string, Module4Preset>> {
  const res = await fetch("/api/module4/presets");
  if (!res.ok) {
    throw new Module4RunError(`Failed to fetch presets (HTTP ${res.status})`, res.status);
  }
  return (await res.json()) as Record<string, Module4Preset>;
}

export async function runModule4Verify(
  payload: Module4Preset["payload"],
): Promise<Module4VerifyResult> {
  return postJson<Module4VerifyResult>(
    "/api/module4/verify",
    payload,
    (detail, status) => new Module4RunError(detail, status),
  );
}

