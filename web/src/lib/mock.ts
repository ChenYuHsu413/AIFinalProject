/**
 * Mock fleet / telemetry / alert data for the Command Center UI.
 *
 * The backend currently exposes a *single* servo model + demo samples; it has
 * no multi-equipment, telemetry-stream, alert or work-order endpoints. Until
 * the Servo Dataset module lands those, the fleet view is driven from here.
 *
 * Everything is deterministic (no Date.now / Math.random) so the demo is stable
 * across reloads. Shapes are intentionally close to what real endpoints would
 * return, so swapping this file for a typed API client later is mechanical.
 */

import { scoreToState, type RiskLevel } from "@/lib/servo";

export type EquipmentStatus = "running" | "warning" | "maintenance" | "offline";
export type HealthState = "LN" | "LO" | "MED" | "HI";

export interface Equipment {
  id: string;
  name: string;
  location: string;
  status: EquipmentStatus;
  healthScore: number; // 0..100
  state: HealthState;
  risk: RiskLevel;
  degradation: number; // DV, 0..1
  confidence: number; // 0..1
  topFeature: { feature: string; z: number; hint: string };
  uptimeHours: number;
  lastUpdated: string; // relative label (mock)
}

/** A single telemetry channel for the trend charts. */
export interface TelemetryPoint {
  t: string; // HH:mm label
  torque: number;
  rotor_speed: number;
  position_error: number;
  current: number;
}

export interface FleetAlert {
  id: string;
  time: string;
  equipment: string;
  type: string;
  severity: "info" | "warning" | "critical";
  predictedState: HealthState;
  suggestedAction: string;
  status: "open" | "ack" | "in_progress" | "resolved";
  workOrderId: string | null;
}

export interface WorkOrder {
  id: string;
  equipment: string;
  title: string;
  priority: "low" | "medium" | "high";
  status: "draft" | "scheduled" | "in_progress" | "done";
  assignee: string;
  due: string;
}

/** Deterministic, smooth-ish telemetry series (sine + per-seed offset). */
function makeTelemetry(seed: number, health: number): TelemetryPoint[] {
  const pts: TelemetryPoint[] = [];
  const drift = (100 - health) / 100; // sicker units wander further
  for (let i = 0; i < 24; i++) {
    const hh = String(i).padStart(2, "0");
    const wob = Math.sin((i + seed) / 2.4) + Math.cos((i + seed * 2) / 3.1);
    pts.push({
      t: `${hh}:00`,
      torque: +(42 + wob * 2.5 + drift * 9).toFixed(2),
      rotor_speed: +(1500 - wob * 18 - drift * 60).toFixed(0),
      position_error: +(0.04 + Math.abs(wob) * 0.012 + drift * 0.09).toFixed(4),
      current: +(8.4 + wob * 0.35 + drift * 1.6).toFixed(2),
    });
  }
  return pts;
}

export const FLEET: Equipment[] = [
  {
    id: "servo-a01",
    name: "Servo-A01",
    location: "產線 1 · X 軸",
    status: "running",
    healthScore: 91,
    ...scoreToState(91),
    degradation: 0.09,
    confidence: 0.94,
    topFeature: { feature: "rotor_speed", z: 0.6, hint: "轉速平穩，無明顯異常" },
    uptimeHours: 1840,
    lastUpdated: "12 秒前",
  },
  {
    id: "servo-a02",
    name: "Servo-A02",
    location: "產線 1 · Y 軸",
    status: "warning",
    healthScore: 64,
    ...scoreToState(64),
    degradation: 0.36,
    confidence: 0.81,
    topFeature: { feature: "position_error", z: 2.4, hint: "位置誤差升高，疑似跟隨誤差" },
    uptimeHours: 2210,
    lastUpdated: "8 秒前",
  },
  {
    id: "servo-a03",
    name: "Servo-A03",
    location: "產線 2 · Z 軸",
    status: "warning",
    healthScore: 47,
    ...scoreToState(47),
    degradation: 0.53,
    confidence: 0.77,
    topFeature: { feature: "torque", z: 3.1, hint: "扭矩偏高，疑似負載過載或潤滑不良" },
    uptimeHours: 3050,
    lastUpdated: "5 秒前",
  },
  {
    id: "servo-testbench",
    name: "Servo-TestBench",
    location: "實驗台 · 加速壽命",
    status: "maintenance",
    healthScore: 29,
    ...scoreToState(29),
    degradation: 0.72,
    confidence: 0.69,
    topFeature: { feature: "i_3p_a", z: 4.2, hint: "A 相電流異常，疑似繞組或軸承劣化" },
    uptimeHours: 5120,
    lastUpdated: "3 秒前",
  },
];

export const TELEMETRY: Record<string, TelemetryPoint[]> = Object.fromEntries(
  FLEET.map((e, i) => [e.id, makeTelemetry(i * 3, e.healthScore)]),
);

export const ALERTS: FleetAlert[] = [
  {
    id: "ALM-2041",
    time: "10:42",
    equipment: "Servo-TestBench",
    type: "電流異常 (A 相)",
    severity: "critical",
    predictedState: "HI",
    suggestedAction: "停機檢查繞組與軸承，安排更換",
    status: "in_progress",
    workOrderId: "WO-3307",
  },
  {
    id: "ALM-2038",
    time: "10:31",
    equipment: "Servo-A03",
    type: "扭矩過載",
    severity: "warning",
    predictedState: "MED",
    suggestedAction: "檢查負載與滾珠導螺桿潤滑",
    status: "ack",
    workOrderId: "WO-3305",
  },
  {
    id: "ALM-2035",
    time: "10:18",
    equipment: "Servo-A02",
    type: "跟隨誤差升高",
    severity: "warning",
    predictedState: "MED",
    suggestedAction: "校正位置迴路增益，檢查機械背隙",
    status: "open",
    workOrderId: null,
  },
  {
    id: "ALM-2030",
    time: "09:54",
    equipment: "Servo-A01",
    type: "溫升提醒",
    severity: "info",
    predictedState: "LO",
    suggestedAction: "持續觀察，確認散熱與環境溫度",
    status: "resolved",
    workOrderId: "WO-3301",
  },
];

export const WORK_ORDERS: WorkOrder[] = [
  {
    id: "WO-3307",
    equipment: "Servo-TestBench",
    title: "更換軸承 + 繞組絕緣檢測",
    priority: "high",
    status: "in_progress",
    assignee: "維修班 A",
    due: "今日 16:00",
  },
  {
    id: "WO-3305",
    equipment: "Servo-A03",
    title: "潤滑保養 + 負載複核",
    priority: "medium",
    status: "scheduled",
    assignee: "維修班 B",
    due: "明日 10:00",
  },
  {
    id: "WO-3301",
    equipment: "Servo-A01",
    title: "散熱風扇清潔",
    priority: "low",
    status: "done",
    assignee: "維修班 B",
    due: "已完成",
  },
];

/** 14-point fleet average-health history for the hero area chart. */
export const FLEET_HEALTH_HISTORY: { t: string; avg: number; worst: number }[] =
  Array.from({ length: 14 }, (_, i) => {
    const day = i + 1;
    const wob = Math.sin(i / 1.7) * 4 + Math.cos(i / 2.9) * 3;
    const avg = Math.round(72 - i * 0.9 + wob);
    return {
      t: `D${String(day).padStart(2, "0")}`,
      avg,
      worst: Math.max(12, Math.round(avg - 30 - (i % 3) * 2)),
    };
  });

/** Aggregate KPIs for the Overview header, with mock trend deltas. */
export function fleetSummary() {
  const total = FLEET.length;
  const avgHealth =
    FLEET.reduce((s, e) => s + e.healthScore, 0) / Math.max(1, total);
  const highRisk = FLEET.filter((e) => e.risk === "High").length;
  const activeAlerts = ALERTS.filter((a) => a.status !== "resolved").length;
  return {
    total,
    avgHealth: Math.round(avgHealth),
    highRisk,
    activeAlerts,
    // Trend vs previous shift (mock, deterministic).
    trends: {
      total: { value: "+1", up: true },
      avgHealth: { value: "-4.2%", up: false },
      highRisk: { value: "+1", up: false },
      activeAlerts: { value: "+2", up: false },
    },
  };
}
