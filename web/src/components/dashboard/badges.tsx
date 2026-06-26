import { cn } from "@/lib/utils";
import {
  HEALTH_COLOR,
  HEALTH_EN,
  HEALTH_ZH,
  RISK_COLOR,
  RISK_ZH,
  type RiskLevel,
} from "@/lib/servo";
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
  running: { label: "運轉中", dot: "bg-emerald-400", text: "text-emerald-300" },
  warning: { label: "警示", dot: "bg-amber-400", text: "text-amber-300" },
  maintenance: { label: "維護中", dot: "bg-sky-400", text: "text-sky-300" },
  offline: { label: "離線", dot: "bg-slate-500", text: "text-slate-400" },
};

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
