"use client";

import Link from "next/link";
import {
  AlertTriangle,
  ArrowRight,
  Bot,
  ClipboardList,
  Clock,
  Gauge,
  MapPin,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

import { TIER_META, type MotorView } from "@/lib/dashboard";
import { cn } from "@/lib/utils";

/**
 * Action Required — the loudest block on the page. Surfaces the units that need
 * a human now (critical + warning, worst first) and tells the operator the next
 * step: open the unit, raise/track a work order, or ask the assistant.
 */
export function ActionRequiredPanel({ views }: { views: MotorView[] }) {
  const queue = views
    .filter((v) => v.tier === "critical" || v.tier === "warning")
    .sort((a, b) => a.actionPriority - b.actionPriority);

  if (queue.length === 0) {
    return (
      <section className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-5">
        <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-300">
          <ShieldAlert className="h-5 w-5" />
          <h2 className="text-sm font-semibold uppercase tracking-wide">
            目前無需立即處理
          </h2>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          全廠設備皆在安全範圍內，維持例行巡檢即可。
        </p>
      </section>
    );
  }

  const [lead, ...rest] = queue;

  return (
    <section>
      <div className="mb-3 flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-red-500/15 text-red-600 dark:text-red-300 ring-1 ring-inset ring-red-500/30">
          <AlertTriangle className="h-3.5 w-3.5" />
        </span>
        <h2 className="text-sm font-semibold uppercase tracking-wide">
          需要立即處理 · Action Required
        </h2>
        <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-xs font-semibold text-red-700 dark:text-red-300 ring-1 ring-inset ring-red-500/30">
          {queue.length} 台
        </span>
      </div>

      <LeadCard view={lead} />

      {rest.length > 0 && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {rest.map((v) => (
            <SecondaryRow key={v.id} view={v} />
          ))}
        </div>
      )}
    </section>
  );
}

function LeadCard({ view }: { view: MotorView }) {
  const m = TIER_META[view.tier];
  const isCritical = view.tier === "critical";
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border p-5 shadow-md",
        isCritical
          ? "border-red-500/40 bg-gradient-to-br from-red-500/10 to-card"
          : "border-orange-500/40 bg-gradient-to-br from-orange-500/10 to-card",
      )}
    >
      {/* pulsing accent rail for critical */}
      <span
        className={cn(
          "absolute inset-y-0 left-0 w-1.5",
          m.bar,
          isCritical && "command-pulse",
        )}
      />

      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        {/* left: identity + diagnosis */}
        <div className="min-w-0 flex-1 pl-2">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-lg font-bold tracking-tight">{view.name}</h3>
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold",
                m.chip,
              )}
            >
              <ShieldAlert className="h-3 w-3" />
              {m.zh} · {m.en}
            </span>
          </div>
          <p className="mt-1 inline-flex items-center gap-1 text-xs text-muted-foreground">
            <MapPin className="h-3 w-3" />
            {view.location}
          </p>

          {/* anomaly summary */}
          <p className="mt-3 text-sm leading-relaxed">
            <span className="font-medium text-foreground">異常摘要：</span>
            <span className="text-muted-foreground">
              {view.signals.map((s) => s.label).join("、")}
              {isCritical ? "，疑似軸承退化" : "，建議追蹤"}
            </span>
          </p>

          {/* recommended action */}
          <div
            className={cn(
              "mt-3 rounded-lg border px-3 py-2",
              isCritical
                ? "border-red-500/30 bg-red-500/5"
                : "border-orange-500/30 bg-orange-500/5",
            )}
          >
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              建議處置
            </p>
            <p className="mt-0.5 text-sm font-medium">{view.recommendedAction}</p>
          </div>
        </div>

        {/* right: metrics */}
        <div className="grid shrink-0 grid-cols-2 gap-3 lg:w-64">
          <Metric
            icon={Gauge}
            label="健康分數"
            value={`${view.healthScore}`}
            unit="/100"
            className={m.text}
          />
          <Metric
            icon={Sparkles}
            label="模型信心"
            value={`${Math.round(view.confidence * 100)}`}
            unit="%"
          />
          <Metric icon={Clock} label="處理時限" value={view.slaText} />
          <Metric
            icon={ClipboardList}
            label="工單"
            value={view.workOrder ? view.workOrder.id : "尚未建立"}
            sub={view.workOrder ? woStatusZh(view.workOrder.status) : undefined}
          />
        </div>
      </div>

      {/* CTAs */}
      <div className="mt-4 flex flex-wrap gap-2 pl-2">
        <Link
          href={`/equipment/${view.id}`}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors",
            isCritical ? "bg-red-600 hover:bg-red-500" : "bg-orange-600 hover:bg-orange-500",
          )}
        >
          查看設備詳情 <ArrowRight className="h-3.5 w-3.5" />
        </Link>
        <Link
          href="/alerts"
          className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 text-sm font-medium transition-colors hover:bg-muted"
        >
          <ClipboardList className="h-3.5 w-3.5" />
          {view.workOrder ? "追蹤工單" : "開啟工單"}
        </Link>
        <Link
          href="/servo/assistant"
          className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 text-sm font-medium transition-colors hover:bg-muted"
        >
          <Bot className="h-3.5 w-3.5" />
          詢問維護助理
        </Link>
      </div>
    </div>
  );
}

function SecondaryRow({ view }: { view: MotorView }) {
  const m = TIER_META[view.tier];
  return (
    <Link
      href={`/equipment/${view.id}`}
      className="group flex items-center gap-3 rounded-xl border border-border/70 bg-card/60 p-3 transition-colors hover:border-orange-500/40 hover:bg-muted/40"
    >
      <span className={cn("flex h-9 w-9 items-center justify-center rounded-lg", m.chip)}>
        <AlertTriangle className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-2 text-sm font-semibold">
          {view.name}
          <span className={cn("text-xs font-medium", m.text)}>· {m.zh}</span>
        </p>
        <p className="truncate text-xs text-muted-foreground">
          {view.recommendedAction}
        </p>
      </div>
      <div className="text-right">
        <p className={cn("text-sm font-bold tabular-nums", m.text)}>
          {view.healthScore}
        </p>
        <p className="text-[11px] text-muted-foreground">{view.slaText}</p>
      </div>
    </Link>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  unit,
  sub,
  className,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  className?: string;
}) {
  return (
    <div className="rounded-lg border border-border/60 bg-card/50 px-3 py-2">
      <p className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
        <Icon className="h-3 w-3" />
        {label}
      </p>
      <p className={cn("mt-0.5 text-sm font-bold tabular-nums", className)}>
        {value}
        {unit && (
          <span className="ml-0.5 text-xs font-normal text-muted-foreground">
            {unit}
          </span>
        )}
      </p>
      {sub && <p className="text-[11px] text-muted-foreground">{sub}</p>}
    </div>
  );
}

function woStatusZh(status: string): string {
  return (
    {
      draft: "草稿",
      scheduled: "已排程",
      in_progress: "處理中",
      done: "已完成",
    }[status] ?? status
  );
}
