/**
 * Dashboard adapter — turns the raw fleet / alert / work-order data into the
 * *operator-facing* view model the Command Center homepage renders.
 *
 * All derivation lives here (riskLevel, recommendedAction, slaText, owner,
 * impactScope, rulEstimate, topSignals, plant status) so the JSX components stay
 * dumb. Everything is deterministic — no Date.now / Math.random — so the demo is
 * stable across reloads, matching the mock data contract in lib/mock.ts.
 *
 * Backend reality: the model gives health/state/risk/topFeature per unit; fields
 * like RUL, SLA, owner and impact are *frontend fallbacks* derived from the tier
 * (or read from a matching alert / work order when one exists). They are clearly
 * labelled as 推估 in the UI.
 */

import type { Equipment, FleetAlert, WorkOrder } from "@/lib/mock";

/* ── 4-tier operational status ─────────────────────────────────────────────
   綠 normal · 黃 observe · 橘 warning · 紅 critical — distinct from the model's
   LN/LO/MED/HI health state so the production map reads at a glance. */
export type NodeTier = "normal" | "observe" | "warning" | "critical";

export function tierOf(score: number): NodeTier {
  if (score >= 80) return "normal";
  if (score >= 60) return "observe";
  if (score >= 40) return "warning";
  return "critical";
}

/** Literal Tailwind classes per tier (kept literal so the JIT keeps them). */
export const TIER_META: Record<
  NodeTier,
  {
    zh: string;
    en: string;
    dot: string;
    text: string;
    chip: string;
    bar: string;
    ring: string;
    soft: string;
  }
> = {
  normal: {
    zh: "正常",
    en: "Normal",
    dot: "bg-emerald-400",
    text: "text-emerald-600 dark:text-emerald-300",
    chip: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 ring-1 ring-inset ring-emerald-500/30",
    bar: "bg-emerald-500",
    ring: "ring-emerald-500/30",
    soft: "bg-emerald-500/10",
  },
  observe: {
    zh: "觀察",
    en: "Observe",
    dot: "bg-yellow-400",
    text: "text-yellow-700 dark:text-yellow-300",
    chip: "bg-yellow-500/15 text-yellow-700 dark:text-yellow-300 ring-1 ring-inset ring-yellow-500/30",
    bar: "bg-yellow-500",
    ring: "ring-yellow-500/30",
    soft: "bg-yellow-500/10",
  },
  warning: {
    zh: "警戒",
    en: "Warning",
    dot: "bg-orange-400",
    text: "text-orange-600 dark:text-orange-300",
    chip: "bg-orange-500/15 text-orange-700 dark:text-orange-300 ring-1 ring-inset ring-orange-500/30",
    bar: "bg-orange-500",
    ring: "ring-orange-500/30",
    soft: "bg-orange-500/10",
  },
  critical: {
    zh: "高風險",
    en: "Critical",
    dot: "bg-red-400",
    text: "text-red-600 dark:text-red-300",
    chip: "bg-red-500/15 text-red-700 dark:text-red-300 ring-1 ring-inset ring-red-500/30",
    bar: "bg-red-500",
    ring: "ring-red-500/30",
    soft: "bg-red-500/10",
  },
};

/* ── feature → maintenance language ───────────────────────────────────────── */
const FEATURE_LABEL: Record<string, string> = {
  torque: "扭矩波動",
  rotor_speed: "轉速穩定性",
  position_error: "位置誤差 / 跟隨誤差",
  current: "電流",
  i_3p_a: "A 相電流",
  i_3p_b: "B 相電流",
  i_3p_c: "C 相電流",
  temperature: "溫度",
  vibration: "振動能量",
};

export function featureLabel(feature: string): string {
  return FEATURE_LABEL[feature] ?? feature;
}

export interface Signal {
  label: string;
  hint: string;
}

/**
 * Up to three operator-readable signals. The first is the *real* model top
 * feature; the rest are tier-derived secondary hints (deterministic) so the card
 * always shows a Top-3 even though the model surfaces a single top feature.
 */
function topSignals(unit: Equipment, tier: NodeTier): Signal[] {
  const primary: Signal = {
    label: featureLabel(unit.topFeature.feature),
    hint: `${unit.topFeature.hint}（z=${unit.topFeature.z}）`,
  };
  const secondary: Record<NodeTier, Signal[]> = {
    critical: [
      { label: "振動能量", hint: "高頻段能量上升，與軸承劣化一致" },
      { label: "溫度", hint: "溫升偏高，注意散熱與潤滑" },
    ],
    warning: [
      { label: "速度穩定性", hint: "轉速波動略增" },
      { label: "溫度", hint: "溫度略升，建議追蹤" },
    ],
    observe: [
      { label: "負載", hint: "負載略高於常態，持續觀察" },
    ],
    normal: [],
  };
  return [primary, ...secondary[tier]].slice(0, 3);
}

/* ── tier-based operational fallbacks ─────────────────────────────────────── */
const ACTION_FALLBACK: Record<NodeTier, string> = {
  critical: "建議停機檢查軸承、負載與三相電流",
  warning: "今日內檢查潤滑與負載，安排巡檢",
  observe: "持續觀察健康趨勢，下一班次複查",
  normal: "正常運轉，無需動作",
};

const SLA_FALLBACK: Record<NodeTier, string> = {
  critical: "立即處理",
  warning: "今日內",
  observe: "本班次內",
  normal: "—",
};

const IMPACT_FALLBACK: Record<NodeTier, string> = {
  critical: "停線 / 設備停機風險",
  warning: "產線精度可能下降",
  observe: "影響有限，預防性追蹤",
  normal: "無立即影響",
};

const OWNER_FALLBACK: Record<NodeTier, string> = {
  critical: "維修班 A",
  warning: "維修班 B",
  observe: "巡檢組",
  normal: "—",
};

/** Deterministic RUL fallback from health + degradation, formatted with a unit. */
function rulEstimate(unit: Equipment): string {
  const hours = Math.max(2, Math.round((1 - unit.degradation) * unit.healthScore * 2.4));
  if (hours >= 48) return `約 ${Math.round(hours / 24)} 天`;
  return `約 ${hours} 小時`;
}

/* ── the per-motor operator view model ────────────────────────────────────── */
export interface MotorView {
  id: string;
  name: string;
  location: string;
  healthScore: number;
  state: Equipment["state"];
  risk: Equipment["risk"];
  status: Equipment["status"];
  confidence: number; // 0..1
  lastUpdated: string;
  degradation: number;
  tier: NodeTier;
  recommendedAction: string;
  slaText: string;
  owner: string;
  impactScope: string;
  rulEstimate: string;
  signals: Signal[];
  alert: FleetAlert | null;
  workOrder: WorkOrder | null;
  /** Lower = act sooner; used for ranking Action Required. */
  actionPriority: number;
}

export function toMotorView(
  unit: Equipment,
  alerts: FleetAlert[],
  workOrders: WorkOrder[],
): MotorView {
  const tier = tierOf(unit.healthScore);
  const alert =
    alerts.find((a) => a.equipment === unit.name && a.status !== "resolved") ??
    alerts.find((a) => a.equipment === unit.name) ??
    null;
  const workOrder =
    workOrders.find((w) => w.equipment === unit.name && w.status !== "done") ??
    workOrders.find((w) => w.equipment === unit.name) ??
    null;

  return {
    id: unit.id,
    name: unit.name,
    location: unit.location,
    healthScore: unit.healthScore,
    state: unit.state,
    risk: unit.risk,
    status: unit.status,
    confidence: unit.confidence,
    lastUpdated: unit.lastUpdated,
    degradation: unit.degradation,
    tier,
    // Prefer real alert/work-order text; fall back to tier defaults.
    recommendedAction: alert?.suggestedAction ?? ACTION_FALLBACK[tier],
    slaText: SLA_FALLBACK[tier],
    owner: workOrder?.assignee ?? OWNER_FALLBACK[tier],
    impactScope: IMPACT_FALLBACK[tier],
    rulEstimate: rulEstimate(unit),
    signals: topSignals(unit, tier),
    alert,
    workOrder,
    actionPriority: unit.healthScore,
  };
}

export function toMotorViews(
  fleet: Equipment[],
  alerts: FleetAlert[],
  workOrders: WorkOrder[],
): MotorView[] {
  return fleet.map((u) => toMotorView(u, alerts, workOrders));
}

/* ── plant-level status ───────────────────────────────────────────────────── */
export type PlantLevel = "normal" | "warning" | "critical";

export const PLANT_META: Record<
  PlantLevel,
  { zh: string; chip: string; dot: string; glow: string }
> = {
  normal: {
    zh: "正常",
    chip: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-200 ring-1 ring-inset ring-emerald-500/40",
    dot: "bg-emerald-400",
    glow: "from-emerald-500/10",
  },
  warning: {
    zh: "警戒中",
    chip: "bg-amber-500/15 text-amber-700 dark:text-amber-200 ring-1 ring-inset ring-amber-500/40",
    dot: "bg-amber-400",
    glow: "from-amber-500/10",
  },
  critical: {
    zh: "高風險",
    chip: "bg-red-500/15 text-red-700 dark:text-red-200 ring-1 ring-inset ring-red-500/40",
    dot: "bg-red-400",
    glow: "from-red-500/10",
  },
};

export interface PlantSummary {
  level: PlantLevel;
  avgHealth: number;
  highRiskCount: number; // critical tier
  todayAlerts: number; // active (non-resolved) alerts
  openWorkOrders: number; // not done
  lastUpdated: string;
}

export function plantSummary(
  views: MotorView[],
  alerts: FleetAlert[],
  workOrders: WorkOrder[],
): PlantSummary {
  const total = Math.max(1, views.length);
  const avgHealth = Math.round(
    views.reduce((s, v) => s + v.healthScore, 0) / total,
  );
  const highRiskCount = views.filter((v) => v.tier === "critical").length;
  const warnCount = views.filter((v) => v.tier === "warning").length;
  const level: PlantLevel =
    highRiskCount > 0 ? "critical" : warnCount > 0 ? "warning" : "normal";
  const todayAlerts = alerts.filter((a) => a.status !== "resolved").length;
  const openWorkOrders = workOrders.filter((w) => w.status !== "done").length;
  // Freshest unit label (mock labels are "N 秒前"); fall back to a stable string.
  const lastUpdated = views[0]?.lastUpdated ?? "剛剛";

  return { level, avgHealth, highRiskCount, todayAlerts, openWorkOrders, lastUpdated };
}

/** Build a single-paragraph maintenance brief from the worst units. */
export function maintenanceBrief(views: MotorView[]): string {
  const critical = views.filter((v) => v.tier === "critical");
  const warning = views.filter((v) => v.tier === "warning");
  const parts: string[] = [];

  for (const v of critical) {
    parts.push(
      `${v.name} 顯示高風險退化，主要異常集中在${v.signals[0]?.label ?? "關鍵特徵"}。建議優先${v.recommendedAction}。`,
    );
  }
  for (const v of warning) {
    parts.push(
      `${v.name} 目前處於警戒狀態，建議${v.recommendedAction}。`,
    );
  }
  if (parts.length === 0) {
    return "目前全廠設備健康狀態良好，無高風險或警戒設備。建議維持例行巡檢與資料監控。";
  }
  parts.push("其餘設備可持續觀察。");
  return parts.join("");
}
