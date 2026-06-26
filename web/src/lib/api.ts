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
