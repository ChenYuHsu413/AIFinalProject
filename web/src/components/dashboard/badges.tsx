import { cn } from "@/lib/utils";
import {
  HEALTH_COLOR,
  HEALTH_EN,
  HEALTH_ZH,
  RISK_COLOR,
  RISK_ZH,
  type RiskLevel,
} from "@/lib/servo";
import { TIER_META, type NodeTier } from "@/lib/dashboard";
import type { EquipmentStatus } from "@/lib/mock";

/** Risk level pill (Low / Medium / High). */
export function RiskBadge({
  level,
  className,
}: {
  level: RiskLevel;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold",
        RISK_COLOR[level],
        className,
      )}
    >
      風險 {RISK_ZH[level]} · {level}
    </span>
  );
}

/** Health-state pill (LN / LO / MED / HI). */
export function HealthBadge({
  state,
  className,
}: {
  state: keyof typeof HEALTH_COLOR;
  className?: string;
}) {
  const c = HEALTH_COLOR[state] ?? HEALTH_COLOR.MED;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold",
        c.chip,
        className,
      )}
    >
      {HEALTH_ZH[state]}
      <span className="opacity-60">({HEALTH_EN[state] ?? state})</span>
    </span>
  );
}

const STATUS_META: Record<
  EquipmentStatus,
  { label: string; dot: string; text: string }
> = {
  running: { label: "運轉中", dot: "bg-emerald-400", text: "text-emerald-600 dark:text-emerald-300" },
  warning: { label: "警示", dot: "bg-amber-400", text: "text-amber-600 dark:text-amber-300" },
  maintenance: { label: "維護中", dot: "bg-sky-400", text: "text-sky-600 dark:text-sky-300" },
  offline: { label: "離線", dot: "bg-slate-500", text: "text-slate-600 dark:text-slate-400" },
};

/**
 * 4-tier operational status light (綠 normal · 黃 observe · 橘 warning · 紅
 * critical). Pulses for warning/critical so the eye is drawn to what needs
 * attention. Used by the production line map and motor cards.
 */
export function StatusIndicator({
  tier,
  withLabel = false,
  size = "md",
  className,
}: {
  tier: NodeTier;
  withLabel?: boolean;
  size?: "sm" | "md";
  className?: string;
}) {
  const m = TIER_META[tier];
  const pulse = tier === "warning" || tier === "critical";
  const dim = size === "sm" ? "h-2 w-2" : "h-2.5 w-2.5";
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span className={cn("relative flex", dim)}>
        {pulse && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full rounded-full opacity-60 command-pulse",
              m.dot,
            )}
          />
        )}
        <span className={cn("relative inline-flex rounded-full", dim, m.dot)} />
      </span>
      {withLabel && (
        <span className={cn("text-xs font-medium", m.text)}>{m.zh}</span>
      )}
    </span>
  );
}

/** 4-tier risk pill (matches StatusIndicator tiers). */
export function TierBadge({
  tier,
  className,
}: {
  tier: NodeTier;
  className?: string;
}) {
  const m = TIER_META[tier];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold",
        m.chip,
        className,
      )}
    >
      {m.zh} · {m.en}
    </span>
  );
}

/** Animated status dot + label for an equipment card. */
export function StatusDot({
  status,
  withLabel = true,
}: {
  status: EquipmentStatus;
  withLabel?: boolean;
}) {
  const m = STATUS_META[status];
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        {(status === "running" || status === "warning") && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full rounded-full opacity-60 command-pulse",
              m.dot,
            )}
          />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", m.dot)} />
      </span>
      {withLabel && (
        <span className={cn("text-xs font-medium", m.text)}>{m.label}</span>
      )}
    </span>
  );
}
