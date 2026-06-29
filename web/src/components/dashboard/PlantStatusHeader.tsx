"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ClipboardList,
  Factory,
  Gauge,
  RefreshCw,
  Wifi,
  WifiOff,
} from "lucide-react";

import { apiGet, type Health } from "@/lib/api";
import { PLANT_META, type PlantSummary } from "@/lib/dashboard";
import { cn } from "@/lib/utils";
import { CountUp } from "./count-up";

/**
 * Plant Status Header — the first thing a line supervisor sees: is the plant OK,
 * how healthy is the fleet on average, how many units are critical, and is the
 * backend live. Reads the real /health endpoint for the API badge.
 */
export function PlantStatusHeader({
  summary,
  loading = false,
}: {
  summary: PlantSummary;
  /** First visit with no cached data — skeleton the KPIs instead of painting
   *  the mock placeholder (avoids the 58 → 73 value flash). */
  loading?: boolean;
}) {
  const m = PLANT_META[summary.level];
  const [api, setApi] = useState<"connecting" | "online" | "offline">(
    "connecting",
  );

  useEffect(() => {
    let alive = true;
    apiGet<Health>("/health")
      .then(() => alive && setApi("online"))
      .catch(() => alive && setApi("offline"));
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section
      className={cn(
        "relative overflow-hidden rounded-2xl border border-border/70 bg-gradient-to-br to-card p-5 shadow-sm",
        loading ? "" : m.glow,
      )}
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        {/* identity + plant state */}
        <div className="flex items-center gap-4">
          <span
            className={cn(
              "flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ring-1 ring-inset",
              loading ? "bg-muted text-muted-foreground ring-border" : m.chip,
            )}
          >
            <Factory className="h-6 w-6" />
          </span>
          <div>
            <div className="flex items-center gap-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              <Activity className="h-3.5 w-3.5" />
              全廠即時狀態 · Plant Status
            </div>
            {loading ? (
              <div className="mt-1 flex items-center gap-2.5">
                <Skeleton className="h-7 w-24" />
                <Skeleton className="h-5 w-32 rounded-full" />
              </div>
            ) : (
              <div className="mt-0.5 flex items-center gap-2.5">
                <span className="relative flex h-2.5 w-2.5">
                  <span
                    className={cn(
                      "absolute inline-flex h-full w-full rounded-full opacity-70 command-pulse",
                      m.dot,
                    )}
                  />
                  <span
                    className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", m.dot)}
                  />
                </span>
                <h1 className="text-2xl font-bold tracking-tight">{m.zh}</h1>
                <span
                  className={cn(
                    "rounded-full px-2.5 py-0.5 text-xs font-semibold",
                    m.chip,
                  )}
                >
                  {summary.highRiskCount > 0
                    ? `${summary.highRiskCount} 台設備需立即處理`
                    : summary.level === "warning"
                      ? "有設備需要關注"
                      : "全線運轉正常"}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* KPI strip */}
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:flex lg:items-center lg:gap-6">
          <Kpi
            icon={Gauge}
            label="平均健康度"
            value={`${summary.avgHealth}`}
            unit="/100"
            tone={summary.avgHealth >= 70 ? "ok" : summary.avgHealth >= 50 ? "warn" : "bad"}
            loading={loading}
          />
          <Kpi
            icon={AlertTriangle}
            label="高風險設備"
            value={`${summary.highRiskCount}`}
            unit="台"
            tone={summary.highRiskCount > 0 ? "bad" : "ok"}
            loading={loading}
          />
          <Kpi
            icon={Activity}
            label="今日告警"
            value={`${summary.todayAlerts}`}
            unit="筆"
            tone={summary.todayAlerts > 0 ? "warn" : "ok"}
            loading={loading}
          />
          <Kpi
            icon={ClipboardList}
            label="未處理工單"
            value={`${summary.openWorkOrders}`}
            unit="筆"
            tone={summary.openWorkOrders > 0 ? "warn" : "ok"}
            loading={loading}
          />
        </div>
      </div>

      {/* meta footer */}
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border/50 pt-3 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <RefreshCw className="h-3 w-3" />
          最後更新 {summary.lastUpdated}
        </span>
        <ApiBadge state={api} />
        <span className="ml-auto inline-flex items-center gap-1 text-muted-foreground/70">
          示意機群 · 健康分數為參考模型輸出
        </span>
      </div>
    </section>
  );
}

function Skeleton({ className }: { className?: string }) {
  return (
    <span
      className={cn("inline-block animate-pulse rounded bg-muted/70", className)}
      aria-hidden
    />
  );
}

function Kpi({
  icon: Icon,
  label,
  value,
  unit,
  tone,
  loading = false,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  unit: string;
  tone: "ok" | "warn" | "bad";
  loading?: boolean;
}) {
  const color = {
    ok: "text-emerald-600 dark:text-emerald-300",
    warn: "text-amber-600 dark:text-amber-300",
    bad: "text-red-600 dark:text-red-300",
  }[tone];
  return (
    <div className="flex items-center gap-2.5">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/50 text-muted-foreground">
        <Icon className="h-4 w-4" />
      </span>
      <div className="leading-tight">
        <p className="text-[11px] text-muted-foreground">{label}</p>
        {loading ? (
          <Skeleton className="mt-1 h-5 w-12" />
        ) : (
          <p className={cn("text-lg font-bold tabular-nums", color)}>
            <CountUp value={Number(value)} />
            <span className="ml-0.5 text-xs font-normal text-muted-foreground">
              {unit}
            </span>
          </p>
        )}
      </div>
    </div>
  );
}

function ApiBadge({ state }: { state: "connecting" | "online" | "offline" }) {
  if (state === "online") {
    return (
      <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-300">
        <Wifi className="h-3 w-3" /> API Connected
      </span>
    );
  }
  if (state === "offline") {
    return (
      <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-300">
        <WifiOff className="h-3 w-3" /> API 離線 · 顯示 fallback 資料
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-muted-foreground">
      <Wifi className="h-3 w-3" /> 連線中…
    </span>
  );
}
