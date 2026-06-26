import { ClipboardList, Clock, User } from "lucide-react";

import type { WorkOrder } from "@/lib/mock";
import { cn } from "@/lib/utils";

const PRIORITY: Record<WorkOrder["priority"], { label: string; cls: string }> = {
  high: { label: "高", cls: "text-red-300 bg-red-500/15 ring-red-500/30" },
  medium: { label: "中", cls: "text-amber-300 bg-amber-500/15 ring-amber-500/30" },
  low: { label: "低", cls: "text-emerald-300 bg-emerald-500/15 ring-emerald-500/30" },
};

const STATUS: Record<WorkOrder["status"], { label: string; cls: string }> = {
  draft: { label: "草稿", cls: "text-slate-300" },
  scheduled: { label: "已排程", cls: "text-sky-300" },
  in_progress: { label: "處理中", cls: "text-amber-300" },
  done: { label: "已完成", cls: "text-emerald-300" },
};

export function WorkOrderCard({ order }: { order: WorkOrder }) {
  const p = PRIORITY[order.priority];
  const s = STATUS[order.status];
  return (
    <div className="rounded-xl border border-border/70 bg-card/70 p-4 shadow-sm backdrop-blur-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/15 text-primary ring-1 ring-inset ring-primary/30">
            <ClipboardList className="h-4 w-4" />
          </span>
          <div>
            <p className="font-mono text-xs text-muted-foreground">{order.id}</p>
            <p className="text-sm font-semibold leading-tight">{order.equipment}</p>
          </div>
        </div>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset",
            p.cls,
          )}
        >
          優先 {p.label}
        </span>
      </div>

      <p className="mt-3 text-sm">{order.title}</p>

      <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <User className="h-3 w-3" />
          {order.assignee}
        </span>
        <span className="inline-flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {order.due}
        </span>
        <span className={cn("font-medium", s.cls)}>{s.label}</span>
      </div>
    </div>
  );
}
