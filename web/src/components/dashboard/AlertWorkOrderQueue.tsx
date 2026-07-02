"use client";

import Link from "next/link";
import { AlertTriangle, ArrowRight, Info, Siren, User } from "lucide-react";

import type { FleetAlert, WorkOrder } from "@/lib/mock";
import type { MotorView } from "@/lib/dashboard";
import { cn } from "@/lib/utils";

/**
 * Alert & Work Order Queue — the alerts table reframed as a dispatchable queue:
 * severity · equipment · issue · impact · recommended action · owner · SLA ·
 * status. Owner / impact / SLA come from the motor view adapter when the alert
 * itself doesn't carry them.
 */
export function AlertWorkOrderQueue({
  alerts,
  workOrders,
  views,
}: {
  alerts: FleetAlert[];
  workOrders: WorkOrder[];
  views: MotorView[];
}) {
  const byName = new Map(views.map((v) => [v.name, v]));

  return (
    <div className="overflow-x-auto rounded-xl border border-border/70 bg-card/70 shadow-sm">
      <table className="w-full min-w-[920px] text-sm">
        <thead>
          <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <th className="px-4 py-3 font-medium">嚴重度</th>
            <th className="px-4 py-3 font-medium">設備</th>
            <th className="px-4 py-3 font-medium">異常類型</th>
            <th className="px-4 py-3 font-medium">影響範圍</th>
            <th className="px-4 py-3 font-medium">建議處置</th>
            <th className="px-4 py-3 font-medium">負責人</th>
            <th className="px-4 py-3 font-medium">SLA</th>
            <th className="px-4 py-3 font-medium">狀態</th>
            <th className="px-4 py-3 font-medium"></th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => {
            const sev = SEVERITY[a.severity];
            const SevIcon = sev.icon;
            const view = byName.get(a.equipment);
            // Only an ACTIVE work order overrides the alert's own status — a
            // finished old order must not paint a fresh alert as resolved.
            const wo = workOrders.find(
              (w) => w.equipment === a.equipment && w.status !== "done",
            );
            const status = wo ? WO_STATUS[wo.status] : ALERT_STATUS[a.status];
            return (
              <tr
                key={a.id}
                className="border-b border-border/40 transition-colors last:border-0 hover:bg-muted/30"
              >
                <td className="px-4 py-3">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset",
                      sev.cls,
                    )}
                  >
                    <SevIcon className="h-3 w-3" />
                    {sev.label}
                  </span>
                </td>
                <td className="whitespace-nowrap px-4 py-3 font-medium">
                  {view ? (
                    <Link
                      href={`/equipment/${view.id}`}
                      className="transition-colors hover:text-primary hover:underline"
                    >
                      {a.equipment}
                    </Link>
                  ) : (
                    a.equipment
                  )}
                </td>
                <td className="px-4 py-3">{a.type}</td>
                <td className="px-4 py-3 text-xs text-muted-foreground">
                  {view?.impactScope ?? "—"}
                </td>
                <td className="px-4 py-3 text-xs text-muted-foreground">
                  {a.suggestedAction}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-xs text-muted-foreground">
                  <span className="inline-flex items-center gap-1">
                    <User className="h-3 w-3" />
                    {wo?.assignee ?? view?.owner ?? "待指派"}
                  </span>
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-xs font-medium">
                  {view?.slaText ?? "—"}
                </td>
                <td className={cn("whitespace-nowrap px-4 py-3 text-xs font-medium", status.cls)}>
                  {status.label}
                </td>
                <td className="px-4 py-3">
                  {view ? (
                    <Link
                      href={`/equipment/${view.id}`}
                      className="inline-flex items-center gap-0.5 text-xs font-medium text-primary hover:underline"
                    >
                      處理 <ArrowRight className="h-3 w-3" />
                    </Link>
                  ) : (
                    <span className="inline-flex items-center gap-0.5 text-xs font-medium text-muted-foreground">
                      處理 <ArrowRight className="h-3 w-3" />
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const SEVERITY: Record<
  FleetAlert["severity"],
  { label: string; cls: string; icon: typeof Info }
> = {
  info: { label: "資訊", cls: "text-sky-700 dark:text-sky-300 bg-sky-500/15 ring-sky-500/30", icon: Info },
  warning: { label: "警示", cls: "text-amber-700 dark:text-amber-300 bg-amber-500/15 ring-amber-500/30", icon: AlertTriangle },
  critical: { label: "嚴重", cls: "text-red-700 dark:text-red-300 bg-red-500/15 ring-red-500/30", icon: Siren },
};

const ALERT_STATUS: Record<FleetAlert["status"], { label: string; cls: string }> = {
  open: { label: "New", cls: "text-red-600 dark:text-red-300" },
  ack: { label: "Acknowledged", cls: "text-amber-600 dark:text-amber-300" },
  in_progress: { label: "In Progress", cls: "text-sky-600 dark:text-sky-300" },
  resolved: { label: "Resolved", cls: "text-emerald-600 dark:text-emerald-300" },
};

const WO_STATUS: Record<WorkOrder["status"], { label: string; cls: string }> = {
  draft: { label: "New", cls: "text-red-600 dark:text-red-300" },
  scheduled: { label: "Acknowledged", cls: "text-amber-600 dark:text-amber-300" },
  in_progress: { label: "In Progress", cls: "text-sky-600 dark:text-sky-300" },
  done: { label: "Resolved", cls: "text-emerald-600 dark:text-emerald-300" },
};
