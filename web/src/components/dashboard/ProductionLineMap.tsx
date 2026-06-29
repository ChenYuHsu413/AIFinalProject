"use client";

import Link from "next/link";
import { Cpu, Factory } from "lucide-react";

import { TIER_META, type MotorView } from "@/lib/dashboard";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { StatusIndicator } from "./badges";

/**
 * Production Line Map — a lightweight spatial view: units grouped by the line
 * they sit on (parsed from `location`, e.g. "產線 1 · X 軸" → "產線 1"). Each
 * node is a status light + health + a hover tooltip with the recommended action;
 * clicking opens the equipment detail page. No 3D/heavy SVG — just CSS.
 */
export function ProductionLineMap({ views }: { views: MotorView[] }) {
  const lines = groupByLine(views);

  return (
    <Card>
      <CardHeader>
        <CardDescription className="flex items-center gap-1.5">
          <Factory className="h-3.5 w-3.5" />
          產線設備拓撲
        </CardDescription>
        <CardTitle className="text-lg">現場設備分佈 · Line Map</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {lines.map(([line, units]) => (
          <div key={line} className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <div className="flex w-28 shrink-0 items-center gap-2 text-xs font-medium text-muted-foreground">
              <span className="h-px flex-1 bg-border sm:hidden" />
              <span className="whitespace-nowrap">{line}</span>
              <span className="hidden h-8 w-px bg-border sm:block" />
            </div>
            <div className="grid flex-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {units.map((u) => (
                <Node key={u.id} view={u} />
              ))}
            </div>
          </div>
        ))}

        {/* legend */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border/50 pt-3 text-[11px] text-muted-foreground">
          <span>狀態：</span>
          {(["normal", "observe", "warning", "critical"] as const).map((t) => (
            <span key={t} className="inline-flex items-center gap-1.5">
              <span className={cn("h-2 w-2 rounded-full", TIER_META[t].dot)} />
              {TIER_META[t].zh}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function Node({ view }: { view: MotorView }) {
  const m = TIER_META[view.tier];
  return (
    <Link
      href={`/equipment/${view.id}`}
      className={cn(
        "group/node relative flex items-center gap-2.5 rounded-lg border bg-card/60 px-3 py-2 transition-all hover:-translate-y-0.5 hover:shadow-md",
        "border-border/70 hover:border-primary/40",
      )}
    >
      <span className={cn("flex h-8 w-8 items-center justify-center rounded-lg", m.soft, m.text)}>
        <Cpu className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-1.5 text-sm font-semibold leading-tight">
          <span className="truncate">{view.name}</span>
          <StatusIndicator tier={view.tier} size="sm" />
        </p>
        <p className="text-[11px] text-muted-foreground">
          {m.zh} · {view.healthScore}/100
        </p>
      </div>

      {/* hover tooltip — pure CSS, no extra dep */}
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-2 w-56 -translate-x-1/2 rounded-lg border border-border bg-popover p-3 text-xs opacity-0 shadow-xl transition-opacity group-hover/node:opacity-100"
      >
        <span className="flex items-center justify-between">
          <span className="font-semibold text-popover-foreground">{view.name}</span>
          <span className={cn("font-semibold", m.text)}>{m.zh}</span>
        </span>
        <span className="mt-1 block text-muted-foreground">
          健康 {view.healthScore}/100 · 信心 {Math.round(view.confidence * 100)}%
        </span>
        <span className="mt-1.5 block text-muted-foreground">
          <span className="text-popover-foreground">建議：</span>
          {view.recommendedAction}
        </span>
        <span className="mt-1 block text-[10px] text-muted-foreground/70">
          更新於 {view.lastUpdated}
        </span>
      </span>
    </Link>
  );
}

/** Parse the line prefix from a location string; fall back to "其他". */
function groupByLine(views: MotorView[]): [string, MotorView[]][] {
  const map = new Map<string, MotorView[]>();
  for (const v of views) {
    const line = v.location.split("·")[0].trim() || "其他";
    const arr = map.get(line) ?? [];
    arr.push(v);
    map.set(line, arr);
  }
  return [...map.entries()];
}
