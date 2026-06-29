"use client";

import { useEffect, useState } from "react";

import { apiGet } from "@/lib/api";
import { readCache, writeCache } from "@/lib/cache";
import { FLEET as MOCK_FLEET, type Equipment } from "@/lib/mock";

/** `loading` = first visit, no data yet (show skeletons); `cache` = seeded from
 *  the previous model payload; `model` = fresh from the backend; `mock` =
 *  offline fallback. */
export type FleetSource = "loading" | "cache" | "model" | "mock";

const CACHE_KEY = "cc:fleet";

/**
 * Fleet data from the backend `/servo/fleet` (synthetic equipment identities
 * whose health is computed by the *real* reference model). To avoid the value
 * flash on open/refresh (mock → model), the hook starts in a `loading` state
 * (so the UI can skeleton instead of painting mock numbers), then — after mount
 * — seeds from the localStorage cache for an instant real value and refreshes
 * from the network. Mock is only shown as an offline fallback. The cache is read
 * in the effect (never the initializer) so SSR and first client paint agree.
 */
export function useFleet(): { fleet: Equipment[]; source: FleetSource } {
  const [fleet, setFleet] = useState<Equipment[]>(MOCK_FLEET);
  const [source, setSource] = useState<FleetSource>("loading");

  useEffect(() => {
    let alive = true;

    // C: instant real value from the last successful fetch. Seeding from the
    // cache here (post-mount, not in the initializer) is intentional double
    // render: skeleton → cached value.
    const cached = readCache<Equipment[]>(CACHE_KEY);
    if (cached?.length) {
      /* eslint-disable react-hooks/set-state-in-effect */
      setFleet(cached);
      setSource("cache");
      /* eslint-enable react-hooks/set-state-in-effect */
    }

    // Refresh from the network.
    apiGet<Equipment[]>("/servo/fleet")
      .then((f) => {
        if (!alive) return;
        if (Array.isArray(f) && f.length) {
          setFleet(f);
          setSource("model");
          writeCache(CACHE_KEY, f);
        } else {
          // Empty payload and no cache → fall back to the mock fleet.
          setSource((s) => (s === "loading" ? "mock" : s));
        }
      })
      .catch(() => {
        // Offline: keep the cached value if we have one, else mock.
        if (alive) setSource((s) => (s === "loading" ? "mock" : s));
      });

    return () => {
      alive = false;
    };
  }, []);

  return { fleet, source };
}
