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
