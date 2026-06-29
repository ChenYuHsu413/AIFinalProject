/**
 * Tiny SSR-safe localStorage cache. Used to seed the fleet / ops hooks from the
 * last model-computed payload so a refresh shows the previous real values
 * immediately instead of flashing the mock placeholder.
 *
 * IMPORTANT: only call these inside effects / event handlers, never in a render
 * or a useState initializer — reading localStorage during render would diverge
 * from the server HTML and cause a hydration mismatch.
 */
export function readCache<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function writeCache(key: string, value: unknown): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota / disabled storage — caching is best-effort */
  }
}
