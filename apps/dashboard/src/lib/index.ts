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

export type Module2Head = "synthetic" | "real";

export interface Module2RunResult {
  /**
   * Which trained head produced these numbers. The two are fitted on different
   * distributions against different targets, so their values are NOT comparable — a
   * synthetic u of 0.05 and a real u of 0.05 do not mean the same thing. The UI must
   * always show this.
   */
  head: Module2Head;
  /** "edl.onnx" or "edl_real.onnx". */
  artifact: string;
  /**
   * Set only by the real head: its 3-class label is discretised from M1's constructed
   * vulnerability proxy, not an observed safety class. Show it wherever its numbers show.
   */
  label_disclosure?: string | null;
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
  /**
   * null on the real head, and deliberately not faked: the baselines are fitted on the
   * SYNTHETIC distribution, so showing them beside a real-head u would invite exactly
   * the comparison the head field exists to prevent.
   */
  baselines: { softmax: number; mc_dropout: number; edl: number } | null;
  /** "onnx" when the exported graph ran, "torch" when it fell back to the head. */
  backend: string;
  /** Calibrated value-axis threshold the trigger compared u against. */
  value_threshold: number;
  /**
   * Empty when the input was clean. Non-empty means the engine repaired unusable input
   * (truncated vector, NaN, bad mask) and forced the trigger, rather than raising —
   * u_q is pinned to 1 in that case. See module2/sanitize.py.
   */
  input_problems?: string[];
}

/**
 * Synthetic path — the novelty/quality sliders. `observedFraction` is taken as given
 * because a manufactured state has no QualityMask behind it.
 */
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

/**
 * Contract path — one real M1 state vector in.
 *
 * `features` is the pinned 28-feature `node_features` row and `qualityMask` its matching
 * per-feature flags; both come straight off `assembleModule1State`. The sensing fraction
 * is DERIVED from the mask server-side, so it is not a parameter here — passing one would
 * let the caller claim a data quality the mask does not support.
 */
export async function runModule2State(
  features: number[],
  qualityMask: QualityFlag[],
  head: Module2Head = "synthetic",
): Promise<Module2RunResult> {
  return postJson<Module2RunResult>(
    "/api/module2/run",
    { features, quality_mask: qualityMask, head },
    (detail, status) => new Module2RunError(detail, status),
  );
}

export interface Module2WarmResult {
  head: Module2Head;
  artifact: string;
  ready: boolean;
  warm_ms: number;
  value_threshold: number;
  label_disclosure: string;
}

/**
 * Prime the real head so the first toggle in a live demo is not the call that pays.
 * ~190 ms: its threshold is calibrated at train time, so there is no fixture to build.
 */
export async function warmModule2Real(): Promise<Module2WarmResult> {
  return postJson<Module2WarmResult>(
    "/api/module2/warm",
    {},
    (detail, status) => new Module2RunError(detail, status),
  );
}

// ---------------------------------------------------------------- Module 1
// Assembles one state vector from the offline calibration artifacts via
// services/learned/module1_state_forecasting/src/module1/assemble.py
// (gateway routers/module1.py). Standard-library + NumPy, no ML stack.

export class Module1RunError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "Module1RunError";
  }
}

export type Module1Scenario = "normal" | "cyclone" | "blackout";
export type QualityFlag = "QUALITY_OBSERVED" | "QUALITY_INTERPOLATED" | "QUALITY_MISSING";

export interface Module1AssembleResult {
  island: string;
  island_name: string;
  scenario: Module1Scenario;
  /** ScenarioRef id — a scenario_library window id, or a derived nominal/pmin tag. */
  scenario_id: string;
  timestamp_lst: string;
  out_of_distribution: boolean;
  degraded: boolean;
  schema_version: string;
  embedding_dim: number;
  /** Always false today — the learned ST-GNN encoder does not exist yet. */
  has_embedding: boolean;
  node_count: number;
  node_names: string[];
  feature_names: string[];
  /** [node_count x 28], row-major. The dashboard renders node_features[0] (the bus). */
  node_features: number[][];
  /** One Quality per feature, parallel to feature_names. */
  quality_mask: QualityFlag[];
  observed_fraction: number;
  /** "real" when data/ is present, "synthetic" when it fell back to .synthetic/. */
  data_source: "real" | "synthetic";
  generated_ms: number;
}

export async function assembleModule1(
  island: string,
  scenario: Module1Scenario,
  timestamp?: string,
): Promise<Module1AssembleResult> {
  return postJson<Module1AssembleResult>(
    "/api/module1/assemble",
    { island, scenario, timestamp: timestamp ?? null },
    (detail, status) => new Module1RunError(detail, status),
  );
}

export type Module1Weighting = "equal" | "hazard_heavy" | "stress_heavy";

export interface Module1ForecastNode {
  id: string;
  island: string;
  asset: string;
  /** Constructed vulnerability score at the chosen hour. */
  current: number;
  /** The 24 hours up to and including the chosen hour. */
  history: number[];
  /** ST-GNN v0.1, t+1 .. t+6. */
  forecast: number[];
  persistence: number[];
  /** The score those hours actually reached; null past the end of the record. */
  target: number[] | null;
  components: { hazard: number; fragility: number; stress: number };
  /** The 64-d node embedding the M1 -> M2 contract reserves. */
  embedding: number[];
}

export interface Module1ForecastResult {
  timestamp_lst: string;
  /** "test" is held out; "train" is a replay of data the model was fitted on. */
  split: "train" | "val" | "test" | "none";
  weighting: Module1Weighting;
  weights: { hazard: number; fragility: number; stress: number };
  variant: string;
  horizon_h: number;
  history_h: number;
  /** High-vulnerability threshold: 95th percentile of training scores. */
  tau: number;
  embedding_dim: number;
  has_embedding: boolean;
  ranking_t1: string[];
  above_tau_t1: string[];
  latency_ms: number;
  data_source: "real" | "synthetic";
  /** Quality of the vulnerability score itself (a constructed proxy). */
  quality: string;
  /** M1 -> M2 contract fields, identical to /assemble's: node_features[i] belongs to nodes[i]. */
  schema_version: string;
  feature_names: string[];
  node_features: number[][];
  quality_mask: QualityFlag[];
  observed_fraction: number;
  nodes: Module1ForecastNode[];
  test_summary: Record<"stgnn" | "persistence" | "lstm_1d", { rmse_mean: number; onset_recall: number }> | null;
  generated_ms: number;
}

// Runs the trained ST-GNN v0.1 via module1.infer.forecast (gateway routers/module1.py).
// 503 means no checkpoint has been trained on this machine; the detail names the command.
export async function forecastModule1(
  timestamp: string,
  weighting: Module1Weighting,
): Promise<Module1ForecastResult> {
  return postJson<Module1ForecastResult>(
    "/api/module1/forecast",
    { timestamp, weighting },
    (detail, status) => new Module1RunError(detail, status),
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

// Module 4: Risk-Certified Verification & Grounded Causal Translation
// Gateway routers/module4.py. A verdict is Check A (the action on the recorded hour) and, in
// "certified" mode, Check B (688 seeded scenarios of the next six hours) with its certificate.

export type Module4Decision = "DECISION_APPROVE" | "DECISION_REJECT" | "DECISION_UNVERIFIED";

export interface ViolationDTO {
  type:
    | "VIOLATION_TYPE_UNSPECIFIED"
    | "VIOLATION_TYPE_UNDERVOLTAGE"
    | "VIOLATION_TYPE_OVERVOLTAGE"
    | "VIOLATION_TYPE_THERMAL_OVERLOAD"
    | "VIOLATION_TYPE_NON_CONVERGENCE"
    | "VIOLATION_TYPE_MALFORMED_ACTION"
    | "VIOLATION_TYPE_GENERATION_INADEQUATE";
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
  grounded: boolean;
  ungrounded: string[];
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

export interface IslandAdequacy {
  island: string;
  station: string;
  demand_kva: number;
  capacity_kva: number;
  status: "ADEQUATE" | "INADEQUATE";
}

export interface RiskCertificate {
  decision: Module4Decision;
  n_scenarios: number;
  n_max: number;
  n_failures: number;
  lower_bound: number;
  upper_bound: number;
  epsilon: number;
  confidence: number;
  stopped_early: boolean;
  horizon_h: number;
  limit_profile: string;
  calibration_version: string;
  calibration_sha256: string;
  seed: number;
  timestamp: string;
  situation: { hour?: number; season?: string; hazard?: string };
  band: { key?: string; n?: number; q_up?: number; q_down?: number };
  pid: null | ({ enabled: boolean; storm: boolean } & Partial<{
    p_offset: number;
    i_term: number;
    d_term: number;
    extra: number;
    updates: number;
    recent_misses: number;
    recent_windows: number;
  }>);
  corner_top_safe: boolean | null;
  corner_bottom_safe: boolean | null;
  failures_by_island: Record<string, number>;
  failing_elements: Record<string, number>;
  notes: string[];
}

export interface RiskAttribution {
  player: string;
  island: string;
  buses: string[];
  share: number;
  fraction_of_risk: number;
  alone: number;
}

export interface Module4VerifyResult {
  action_id: string;
  decision: Module4Decision;
  check_a_decision: Module4Decision;
  mode: Module4Mode;
  hour: string;
  profile: string;
  v_min_pu: number;
  v_max_pu: number;
  snapshot_source: string;
  solve_latency_ms: number;
  violations: ViolationDTO[];
  unverified_reasons: string[];
  rejection_severity: number;
  certificate: RiskCertificate | null;
  attribution: RiskAttribution[];
  causal_log: CausalLogDTO;
  buses: GridBusState[];
  lines: GridLineState[];
  adequacy: IslandAdequacy[];
}

export type Module4Mode = "certified" | "plain";
export type Module4PidMode = "auto" | "on" | "off";

export interface Module4ActionPayload {
  action_id: string;
  origin: "SYSTEM1" | "SYSTEM2";
  rationale: string;
  breakers: Array<{ edge_id: string; closed: boolean }>;
  load_shed: Array<{ node_id: string; shed_fraction: number; priority_tier: number }>;
  dispatch: Array<{ node_id: string; p_kw: number; q_kvar: number }>;
  hour: string;
  outages: string[];
}

export interface Module4Preset {
  title: string;
  description: string;
  expected: Record<Module4Mode, Module4Decision>;
  payload: Module4ActionPayload;
}

export interface Module4Hour {
  timestamp: string;
  title: string;
  description: string;
}

export interface Module4Profile {
  profile_id: string;
  version: number;
  title: string;
  v_min_pu: number;
  v_max_pu: number;
  statistic: string;
  source: string;
  note: string;
}

export interface Module4VerifyRequest extends Module4ActionPayload {
  profile: string;
  mode: Module4Mode;
  pid: Module4PidMode;
}

export class Module4RunError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "Module4RunError";
  }
}

async function getModule4<T>(path: string): Promise<T> {
  const res = await fetch(`/api/module4/${path}`);
  if (!res.ok) {
    throw new Module4RunError(`Failed to fetch ${path} (HTTP ${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

export const fetchModule4Presets = () => getModule4<Record<string, Module4Preset>>("presets");
export const fetchModule4Hours = () => getModule4<Record<string, Module4Hour>>("hours");
export const fetchModule4Profiles = () => getModule4<Record<string, Module4Profile>>("profiles");

export async function runModule4Verify(
  request: Module4VerifyRequest,
): Promise<Module4VerifyResult> {
  return postJson<Module4VerifyResult>(
    "/api/module4/verify",
    request,
    (detail, status) => new Module4RunError(detail, status),
  );
}
