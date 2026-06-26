"use client";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { AlertTable } from "@/components/dashboard/AlertTable";
import { WorkOrderCard } from "@/components/dashboard/WorkOrderCard";
import { PageTitle, Note } from "@/components/ui-kit";
import { useFleetOps } from "@/lib/ops";

export default function AlertsPage() {
  const { alerts, workOrders, source } = useFleetOps();
  const open = alerts.filter((a) => a.status === "open").length;
  const critical = alerts.filter((a) => a.severity === "critical").length;
  const wipOrders = workOrders.filter((w) => w.status === "in_progress").length;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <PageTitle
        title="告警 / 工單中心"
        desc="設備告警事件與維修工單追蹤"
      />

      {source === "model" ? (
        <Note tone="info" className="mb-6">
          告警與工單由<b>真實模型驅動的機群</b>衍生（風險 / 狀態 / 異常特徵來自模型，
          後端 <code className="font-mono">/servo/alerts</code>、
          <code className="font-mono">/servo/work_orders</code>）。工單排程與
          ID 屬示意性運維包裝。
        </Note>
      ) : (
        <Note tone="warn" className="mb-6">
          後端未連線，目前顯示 mock fallback。連線後將改由
          <code className="font-mono">/servo/alerts</code> 即時衍生。
        </Note>
      )}

      <section className="mb-8 grid grid-cols-1 gap-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card *:data-[slot=card]:shadow-xs sm:grid-cols-3">
        <MetricCard
          label="未處理告警"
          value={open}
          unit="筆"
          valueClassName={open > 0 ? "text-red-400" : "text-emerald-400"}
          footerMuted="需立即指派工單"
        />
        <MetricCard
          label="嚴重告警"
          value={critical}
          unit="筆"
          valueClassName={critical > 0 ? "text-amber-400" : "text-emerald-400"}
          footerMuted="severity = critical"
        />
        <MetricCard
          label="處理中工單"
          value={wipOrders}
          unit="張"
          footerMuted="維修班執行中"
        />
      </section>

      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide">告警事件</h2>
      <section className="mb-8">
        <AlertTable alerts={alerts} />
      </section>

      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide">維修工單</h2>
      {workOrders.length > 0 ? (
        <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {workOrders.map((w) => (
            <WorkOrderCard key={w.id} order={w} />
          ))}
        </section>
      ) : (
        <p className="text-sm text-muted-foreground">目前無維修工單。</p>
      )}
    </div>
  );
}
