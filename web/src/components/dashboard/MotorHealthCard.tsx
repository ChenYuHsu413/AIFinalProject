import Link from "next/link";
import { ChevronRight, Clock, Cpu, Gauge, TimerReset, Wrench } from "lucide-react";

import { TIER_META, type MotorView } from "@/lib/dashboard";
import { cn } from "@/lib/utils";
import { StatusIndicator } from "./badges";
import { HealthScoreGauge } from "./HealthScoreGauge";

/**
 * Operator-oriented motor card: health + RUL + the top-3 signals translated into
 * maintenance language, the recommended action, SLA and model confidence — not
 * just a number. Click-through to the equipment detail page.
 */
export function MotorHealthCard({ view }: { view: MotorView }) {
  const m = TIER_META[view.tier];
  return (
    <Link
      href={`/equipment/${view.id}`}
      className="group relative flex flex-col overflow-hidden rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md"
    >
      <span className={cn("absolute inset-y-0 left-0 w-1", m.bar)} />

      {/* header */}
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Cpu className={cn("h-4 w-4", m.text)} />
            <span className="truncate font-semibold tracking-tight">{view.name}</span>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">{view.location}</p>
        </div>
        <StatusIndicator tier={view.tier} withLabel size="sm" />
      </div>

      {/* radial gauge ④ + key metrics */}
      <div className="mt-2 flex items-center gap-3">
        <HealthScoreGauge score={view.healthScore} size={116} className="shrink-0" />
        <dl className="min-w-0 flex-1 space-y-2">
          <div>
            <dt className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <TimerReset className="h-3 w-3" />
              剩餘壽命（推估）
            </dt>
            <dd className={cn("text-sm font-bold tabular-nums", m.text)}>
              {view.rulEstimate}
            </dd>
          </div>
          <div>
            <dt className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <Gauge className="h-3 w-3" />
              模型信心
            </dt>
            <dd className="text-sm font-medium tabular-nums">
              {Math.round(view.confidence * 100)}%
            </dd>
          </div>
          <div>
            <dt className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <Clock className="h-3 w-3" />
              處理時限
            </dt>
            <dd className="text-sm font-medium">{view.slaText}</dd>
          </div>
        </dl>
      </div>

      {/* top signals */}
      <div className="mt-3">
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          主要異常特徵
        </p>
        <ul className="mt-1 space-y-1">
          {view.signals.map((s, i) => (
            <li key={i} className="flex items-start gap-1.5 text-xs">
              <span className={cn("mt-1.5 h-1 w-1 shrink-0 rounded-full", m.bar)} />
              <span>
                <span className="font-medium text-foreground">{s.label}</span>
                <span className="text-muted-foreground"> — {s.hint}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* recommended action */}
      <div className="mt-3 rounded-lg border border-border/60 bg-muted/30 px-2.5 py-2">
        <p className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
          <Wrench className="h-3 w-3" />
          建議動作
        </p>
        <p className="mt-0.5 text-xs text-foreground">{view.recommendedAction}</p>
      </div>

      {/* footer */}
      <div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>更新於 {view.lastUpdated}</span>
        <span className="inline-flex items-center gap-0.5 text-primary opacity-0 transition-opacity group-hover:opacity-100">
          詳情 <ChevronRight className="h-3 w-3" />
        </span>
      </div>
    </Link>
  );
}

/** Grid wrapper so the homepage just hands over the views. */
export function MotorHealthGrid({ views }: { views: MotorView[] }) {
  return (
    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {views.map((v) => (
        <MotorHealthCard key={v.id} view={v} />
      ))}
    </section>
  );
}
