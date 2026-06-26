import Link from "next/link";
import { AlertTriangle, Info, Siren } from "lucide-react";

import { FLEET, type FleetAlert } from "@/lib/mock";
import { cn } from "@/lib/utils";

import { HealthBadge } from "./badges";

const SEVERITY: Record<
  FleetAlert["severity"],
  { label: string; cls: string; icon: typeof Info }
> = {
  info: { label: "資訊", cls: "text-sky-300 bg-sky-500/15 ring-sky-500/30", icon: Info },
  warning: { label: "警示", cls: "text-amber-300 bg-amber-500/15 ring-amber-500/30", icon: AlertTriangle },
  critical: { label: "嚴重", cls: "text-red-300 bg-red-500/15 ring-red-500/30", icon: Siren },
};

const STATUS_LABEL: Record<FleetAlert["status"], { label: string; cls: string }> = {
  open: { label: "未處理", cls: "text-red-300" },
  ack: { label: "已確認", cls: "text-amber-300" },
  in_progress: { label: "處理中", cls: "text-sky-300" },
  resolved: { label: "已解決", cls: "text-emerald-300" },
};

export function AlertTable({ alerts }: { alerts: FleetAlert[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border/70 bg-card/70 shadow-sm backdrop-blur-sm">
      <table className="w-full min-w-[760px] text-sm">
        <thead>
          <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <th className="px-4 py-3 font-medium">時間</th>
            <th className="px-4 py-3 font-medium">設備</th>
            <th className="px-4 py-3 font-medium">告警類型</th>
            <th className="px-4 py-3 font-medium">嚴重度</th>
            <th className="px-4 py-3 font-medium">預測狀態</th>
            <th className="px-4 py-3 font-medium">建議處置</th>
            <th className="px-4 py-3 font-medium">狀態</th>
            <th className="px-4 py-3 font-medium">工單</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => {
            const sev = SEVERITY[a.severity];
            const st = STATUS_LABEL[a.status];
            const SevIcon = sev.icon;
            return (
              <tr
                key={a.id}
                className="border-b border-border/40 transition-colors last:border-0 hover:bg-muted/30"
              >
                <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-muted-foreground">
                  {a.time}
                </td>
                <td className="whitespace-nowrap px-4 py-3 font-medium">
                  {(() => {
                    const unit = FLEET.find((u) => u.name === a.equipment);
                    return unit ? (
                      <Link
                        href={`/equipment/${unit.id}`}
                        className="text-foreground transition-colors hover:text-primary hover:underline"
                      >
                        {a.equipment}
                      </Link>
                    ) : (
                      a.equipment
                    );
                  })()}
                </td>
                <td className="px-4 py-3">{a.type}</td>
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
                <td className="px-4 py-3">
                  <HealthBadge state={a.predictedState} />
                </td>
                <td className="px-4 py-3 text-xs text-muted-foreground">
                  {a.suggestedAction}
                </td>
                <td className={cn("whitespace-nowrap px-4 py-3 text-xs font-medium", st.cls)}>
                  {st.label}
                </td>
                <td className="whitespace-nowrap px-4 py-3 font-mono text-xs">
                  {a.workOrderId ?? <span className="text-muted-foreground">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
