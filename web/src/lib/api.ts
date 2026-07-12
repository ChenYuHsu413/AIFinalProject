/**
 * Typed client for the FastAPI backend (see docs/WEB_REVAMP_PLAN.md).
 *
 * The base URL is read from NEXT_PUBLIC_API_BASE_URL so the same build works in
 * dev (local uvicorn) and in production (GCP VM + nginx, where nginx proxies
 * /api to uvicorn). Falls back to "/api" when unset.
 */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new ApiError(res.status, `GET ${path} → ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    throw new ApiError(res.status, `POST ${path} → ${res.status}`);
  }
  return res.json() as Promise<T>;
}

/** Shape of GET /health. */
export interface Health {
  status: "ok" | "model_missing";
  model_loaded: boolean;
  message: string | null;
}

/** One demo sample row from GET /servo/samples (features + ground-truth ylabel/DV). */
export type ServoSample = Record<string, number | string>;

export interface ServoTopFeature {
  feature: string;
  z: number;
  hint: string;
}

/** Shape of POST /servo/predict. */
export interface ServoPrediction {
  predicted_health_state: string;
  health_state_zh: string;
  health_state_proba: Record<string, number>;
  model_confidence: number;
  degradation_score: number;
  health_score: number;
  risk_level: "Low" | "Medium" | "High";
  consistency_warning: string | null;
  top_features: ServoTopFeature[];
  maintenance_advice: string[];
  placeholder: boolean;
}

/** GET /servo/simulate/options. */
export interface ServoSimulateOptions {
  classifiers: string[];
  regressors: string[];
  algo_labels: Record<string, string>;
}

/** GET /servo/feature_sets. */
export type ServoFeatureSets = Record<
  string,
  { label: string; desc: string; columns: string[] }
>;

/** POST /servo/simulate. ``task`` echoes the backend ("classification"/"regression");
 *  discriminate clf vs reg by the presence of confusion_matrix / r2, not this string. */
export interface ServoSimResult {
  task: string;
  algo: string;
  feature_set: string;
  n_samples: number;
  n_features: number;
  train_time_s: number;
  explanation: string[];
  accuracy?: number;
  macro_f1?: number;
  labels?: string[];
  confusion_matrix?: number[][];
  mae?: number;
  rmse?: number;
  r2?: number;
}

/** GET /servo/reference_metrics. */
export interface ServoReferenceMetrics {
  clf: { macro_f1?: number; model?: string };
  reg: { r2?: number; mae?: number; model?: string };
  dl: {
    note?: string;
    mlp_classification_macro_f1?: number;
    mlp_regression?: { r2?: number; mae?: number };
    reconstruction_error_by_class?: Record<string, number>;
  };
}

/** GET /servo/cnn_results (Phase B 1D-CNN on raw-waveform envelopes; {} when not built). */
export interface ServoCnnResults {
  method?: string;
  framework?: string;
  note?: string;
  window?: {
    len: number;
    channels: string[];
    n_train: number;
    n_test: number;
    subset: boolean;
  };
  architecture?: { cnn: string; autoencoder: string };
  classifier?: {
    accuracy: number;
    macro_f1: number;
    confusion_matrix: number[][];
    labels: string[];
  };
  autoencoder?: { reconstruction_error_by_class: Record<string, number> };
}

/** GET /servo/augment_results (train-only augmentation experiment; {} when not run). */
export interface ServoAugmentResults {
  experiment?: string;
  eval?: string;
  seeds?: number[];
  conditions?: Record<string, string>;
  results?: Record<
    string,
    {
      mean: number;
      std: number;
      delta_vs_baseline?: number;
      robust_gain?: boolean | null;
      per_seed?: number[];
    }
  >;
  verdict?: string;
  note?: string;
}

/** One row of GET /servo/glossary. */
export interface GlossaryEntry {
  name: string;
  zh: string;
  desc: string;
  meaning: string;
  anomaly: string;
}

/** One doc from GET /knowledge/documents. */
export interface KnowledgeDoc {
  source: string;
  title: string;
  preview: string;
  chars: string;
}

/** One hit from GET /knowledge/search. */
export interface KnowledgeHit {
  text: string;
  score: string | number;
  source: string;
  title: string;
  topic: string;
}

/** GET /servo/assistant/providers. */
export interface AssistantProviders {
  providers: string[];
}

/** POST /servo/assistant/report | /qa → {text, source}. */
export interface AssistantResponse {
  text: string;
  source: string;
}

/** Shape of GET /servo/model_info. */
export interface ServoModelInfo {
  feature_set: string | null;
  feature_columns: string[];
  labels: string[] | null;
  clf_model: string | null;
  reg_model: string | null;
  clf_macro_f1: number | null;
  reg_r2: number | null;
  placeholder: boolean | null;
}

// --- Servo real-time monitor (S5 — self-contained backend aggregation) ------
// The backend computes windowing / inference / smoothing / alert hysteresis /
// drift; the UI only renders these enriched per-window events (SSE).

/** Alert-engine state carried on each window (backend-computed hysteresis). */
export interface ServoMonitorAlertState {
  active: boolean;
  high_streak: number;
  low_streak: number;
  active_alert_id: string | null;
}

/** Per-window drift snapshot (reconstruction error vs training P95). */
export interface ServoMonitorDriftStatus {
  available: boolean;
  instant_recon_error?: number;
  rolling_recon_error?: number;
  threshold_p95?: number;
  triggered?: boolean;
}

/** One alert / consistency / drift / work-order event (feed + polling endpoint). */
export interface ServoMonitorEvent {
  id: string;
  type: string;
  ts?: string;
  stream_t?: number | null;
  trigger_rule?: string;
  clear_rule?: string;
  message?: string;
  alert_id?: string;
  reason?: string;
  rolling_recon_error?: number;
  instant_recon_error?: number;
  baseline_p95?: number;
  snapshot?: Record<string, unknown>;
  model_version?: string;
}

/** GET /servo/monitor/stream → one enriched window (SSE `data:` frame). */
export interface ServoMonitorFrame {
  window_index: number;
  window_ts: number;
  predicted_health_state: string;
  smoothed_state: string;
  recent_states: string[];
  true_label: string | null;
  health_state_proba: Record<string, number> | null;
  degradation_score: number;
  model_confidence: number;
  risk_level: string;
  consistency_warning: string | null;
  window_rows: number | null;
  alert_state: ServoMonitorAlertState;
  drift_status: ServoMonitorDriftStatus;
  model_version: string;
  replay_segment: { key: string; label: string; injected: boolean };
  events: ServoMonitorEvent[];
}

/** A background-generated LLM work-order draft, linked to its alert. */
export interface ServoMonitorWorkOrder {
  id: string;
  alert_id: string;
  type: "work_order_draft";
  ts?: string;
  llm_source?: string;
  text: string;
}

/** GET /servo/monitor/events → paginated feed + linked work orders. */
export interface ServoMonitorEventsResponse {
  events: ServoMonitorEvent[];
  work_orders: ServoMonitorWorkOrder[];
  total: number;
  limit: number;
  offset: number;
}

// --- Servo MLOps status panel (S5 P2 — read-only) ---------------------------

/** One registered model version (metrics + CRC32 + active marker). */
export interface MlopsVersion {
  version: string;
  active: boolean;
  created: string | null;
  macro_f1: number | null;
  dv_r2: number | null;
  dv_mae: number | null;
  feature_set: string | null;
  placeholder: boolean | null;
  note: string | null;
  eval_mode: string | null;
  clf_model: string | null;
  reg_model: string | null;
  crc32: Record<string, string> | null;
  has_gate_report: boolean;
}

/** One validation-gate check line. */
export interface MlopsGateCheck {
  name: string;
  status: "PASS" | "FAIL" | "SKIP" | "BLOCKED" | string;
  detail: string;
  numbers: Record<string, unknown>;
}

/** The latest gate_report.json (validation gate result). */
export interface MlopsGateReport {
  candidate_dir: string;
  active_version: string | null;
  passed: boolean;
  created: string;
  tolerances: Record<string, number>;
  checks: MlopsGateCheck[];
  _source?: string;
}

/** One drift → retrain → promote causal-chain event. */
export interface MlopsEvent {
  id: string;
  type: string;
  ts?: string;
  trigger?: string;
  mode?: string;
  reason?: string;
  gate_passed?: boolean;
  old_active?: string | null;
  new_version?: string | null;
  rolling_recon_error?: number;
  baseline_p95?: number;
  error?: string;
  stream_t?: number | null;
}

/** GET /servo/mlops → the full read-only panel payload. */
export interface MlopsStatus {
  registry: {
    active_version: string | null;
    updated: string | null;
    versions: MlopsVersion[];
    outputs_consistent: boolean | null;
  };
  gate_report: MlopsGateReport | null;
  timeline: MlopsEvent[];
}

// --- Live Monitor v3 (synthetic demo track) ---------------------------------

/** One replay frame from a monitor scenario pack (down-sampled cadence). */
export interface MonitorFrame {
  t: number;
  stage_int: number;
  health: number;
  rul: number;
  gt_prob: number;
  warning: number;
  alarm: number;
  trip: number;
  pred_prob: number;
  pred_cat: string;
  // one 0..1 severity per subsystem key (temperature/current/…)
  [subsystem: string]: number | string;
}

/** GET /monitor/scenarios → one scenario summary row. */
export interface MonitorScenarioSummary {
  scenario_id: number;
  scenario_name: string;
  fault_category: string;
  n_frames: number;
  lead_to_alarm_s: number | null;
  held_out: boolean;
}

/** GET /monitor/scenarios. */
export interface MonitorScenarios {
  available: boolean;
  reason?: string;
  scenarios: MonitorScenarioSummary[];
  subsystems: string[];
  out_hz: number;
  eval: {
    early_warning_f1_heldout?: number;
    median_lead_to_alarm_s?: number;
    horizon_s?: number;
    dataset?: string;
    note?: string;
  };
}

/** GET /monitor/scenario/{id} → full replay pack. */
export interface MonitorPack {
  meta: {
    scenario_id: number;
    scenario_name: string;
    fault_category: string;
    alarm_code: string;
    root_cause: string;
    maintenance_action: string;
    out_hz: number;
    n_frames: number;
    gt_warning_t: number | null;
    gt_alarm_t: number | null;
    model_alert_t: number | null;
    lead_time_s: number | null;
    lead_to_alarm_s: number | null;
    predicted_category: string;
  };
  subsystems: string[];
  frames: MonitorFrame[];
}
