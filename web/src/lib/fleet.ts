"use client";

import { useEffect, useState } from "react";

import { apiGet } from "@/lib/api";
import { FLEET as MOCK_FLEET, type Equipment } from "@/lib/mock";

export type FleetSource = "mock" | "model";

/**
 * Fleet data, sourced from the backend `/servo/fleet` (synthetic equipment
 * identities whose health is computed by the *real* reference model over
 * representative demo runs). Starts from the local mock so the UI is never empty
 * or in a loading-flash state, then upgrades to the model-computed fleet once it
 * arrives. Falls back to mock if the backend is unreachable.
 */
export function useFleet(): { fleet: Equipment[]; source: FleetSource } {
  const [fleet, setFleet] = useState<Equipment[]>(MOCK_FLEET);
  const [source, setSource] = useState<FleetSource>("mock");

  useEffect(() => {
    let alive = true;
    apiGet<Equipment[]>("/servo/fleet")
      .then((f) => {
        if (alive && Array.isArray(f) && f.length) {
          setFleet(f);
          setSource("model");
        }
      })
      .catch(() => {
        /* keep mock fallback */
      });
    return () => {
      alive = false;
    };
  }, []);

  return { fleet, source };
}
