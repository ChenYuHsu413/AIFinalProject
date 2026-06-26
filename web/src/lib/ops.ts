"use client";

import { useEffect, useState } from "react";

import { apiGet } from "@/lib/api";
import {
  ALERTS as MOCK_ALERTS,
  WORK_ORDERS as MOCK_WORK_ORDERS,
  type FleetAlert,
  type WorkOrder,
} from "@/lib/mock";

export type OpsSource = "mock" | "model";

/**
 * Alerts + work orders, sourced from the backend (`/servo/alerts`,
 * `/servo/work_orders`) — both are *derived from the real model-driven fleet*
 * (risk / state / top feature come from the model). Starts from the local mock
 * so the UI is never empty, then upgrades once the API responds; falls back to
 * mock if the backend is unreachable.
 */
export function useFleetOps(): {
  alerts: FleetAlert[];
  workOrders: WorkOrder[];
  source: OpsSource;
} {
  const [alerts, setAlerts] = useState<FleetAlert[]>(MOCK_ALERTS);
  const [workOrders, setWorkOrders] = useState<WorkOrder[]>(MOCK_WORK_ORDERS);
  const [source, setSource] = useState<OpsSource>("mock");

  useEffect(() => {
    let alive = true;
    Promise.all([
      apiGet<FleetAlert[]>("/servo/alerts"),
      apiGet<WorkOrder[]>("/servo/work_orders"),
    ])
      .then(([a, w]) => {
        if (!alive) return;
        if (Array.isArray(a)) setAlerts(a);
        if (Array.isArray(w)) setWorkOrders(w);
        setSource("model");
      })
      .catch(() => {
        /* keep mock fallback */
      });
    return () => {
      alive = false;
    };
  }, []);

  return { alerts, workOrders, source };
}
