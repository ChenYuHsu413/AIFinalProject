"use client";

import { useEffect, useState } from "react";

import { apiGet } from "@/lib/api";
import { readCache, writeCache } from "@/lib/cache";
import {
  ALERTS as MOCK_ALERTS,
  WORK_ORDERS as MOCK_WORK_ORDERS,
  type FleetAlert,
  type WorkOrder,
} from "@/lib/mock";

export type OpsSource = "loading" | "cache" | "model" | "mock";

const ALERTS_KEY = "cc:alerts";
const WO_KEY = "cc:workorders";

/**
 * Alerts + work orders, sourced from the backend (`/servo/alerts`,
 * `/servo/work_orders`), derived from the real model-driven fleet. Same
 * loading/cache strategy as {@link useFleet}: starts in `loading`, seeds from
 * the localStorage cache after mount for an instant real value, refreshes from
 * the network, and falls back to mock only when offline.
 */
export function useFleetOps(): {
  alerts: FleetAlert[];
  workOrders: WorkOrder[];
  source: OpsSource;
} {
  const [alerts, setAlerts] = useState<FleetAlert[]>(MOCK_ALERTS);
  const [workOrders, setWorkOrders] = useState<WorkOrder[]>(MOCK_WORK_ORDERS);
  const [source, setSource] = useState<OpsSource>("loading");

  useEffect(() => {
    let alive = true;

    // Seed from cache post-mount (intentional skeleton → cached double render).
    const cachedA = readCache<FleetAlert[]>(ALERTS_KEY);
    const cachedW = readCache<WorkOrder[]>(WO_KEY);
    if (cachedA?.length || cachedW?.length) {
      /* eslint-disable react-hooks/set-state-in-effect */
      if (cachedA?.length) setAlerts(cachedA);
      if (cachedW?.length) setWorkOrders(cachedW);
      setSource("cache");
      /* eslint-enable react-hooks/set-state-in-effect */
    }

    Promise.all([
      apiGet<FleetAlert[]>("/servo/alerts"),
      apiGet<WorkOrder[]>("/servo/work_orders"),
    ])
      .then(([a, w]) => {
        if (!alive) return;
        if (Array.isArray(a)) {
          setAlerts(a);
          writeCache(ALERTS_KEY, a);
        }
        if (Array.isArray(w)) {
          setWorkOrders(w);
          writeCache(WO_KEY, w);
        }
        setSource("model");
      })
      .catch(() => {
        if (alive) setSource((s) => (s === "loading" ? "mock" : s));
      });

    return () => {
      alive = false;
    };
  }, []);

  return { alerts, workOrders, source };
}
