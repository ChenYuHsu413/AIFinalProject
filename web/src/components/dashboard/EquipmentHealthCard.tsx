import { Activity, ChevronRight } from "lucide-react";
import Link from "next/link";

import { HEALTH_COLOR } from "@/lib/servo";
import type { Equipment } from "@/lib/mock";
import { cn } from "@/lib/utils";

import { HealthBadge, RiskBadge, StatusDot } from "./badges";

/** Compact equipment health card for the fleet grid. */
export function EquipmentHealthCard({ unit }: { unit: Equipment }) {
  const c = HEALTH_COLOR[unit.state];
  return (
    <Link
      href={`/equipment/${unit.id}`}
      className="group relative block overflow-hidden rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md"
    >
      {/* left status rail */}
      <span className={cn("absolute inset-y-0 left-0 w-1", c.bar)} />

      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Activity className={cn("h-4 w-4", c.text)} />
            <span className="font-semibold tracking-tight">{unit.name}</span>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">{unit.location}</p>
        </div>
        <StatusDot status={unit.status} />
      </div>

      <div className="mt-3 flex items-end justify-between">
        <div>
          <span className={cn("text-3xl font-bold tabular-nums", c.text)}>
            {unit.healthScore}
          </span>
          <span className="ml-1 text-xs text-muted-foreground">/100</span>
        </div>
        <div className="flex flex-col items-end gap-1">
          <HealthBadge state={unit.state} />
          <RiskBadge level={unit.risk} />
        </div>
      </div>

      {/* health bar */}
      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full", c.bar)}
          style={{ width: `${unit.healthScore}%` }}
        />
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        <div className="flex justify-between">
          <dt className="text-muted-foreground">退化 DV</dt>
          <dd className="font-medium tabular-nums">{unit.degradation.toFixed(2)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-muted-foreground">信心</dt>
          <dd className="font-medium tabular-nums">
            {(unit.confidence * 100).toFixed(0)}%
          </dd>
        </div>
      </dl>

      <div className="mt-2.5 rounded-lg border border-border/60 bg-muted/30 px-2.5 py-1.5">
        <p className="text-[11px] text-muted-foreground">
          主要異常特徵{" "}
          <span className="font-mono font-medium text-foreground">
            {unit.topFeature.feature}
          </span>{" "}
          (z={unit.topFeature.z})
        </p>
        <p className="mt-0.5 text-[11px] text-muted-foreground/80">
          {unit.topFeature.hint}
        </p>
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>更新於 {unit.lastUpdated}</span>
        <span className="inline-flex items-center gap-0.5 text-primary opacity-0 transition-opacity group-hover:opacity-100">
          檢視 <ChevronRight className="h-3 w-3" />
        </span>
      </div>
    </Link>
  );
}
